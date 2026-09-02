#!/usr/bin/env python3
"""The ZML generator's test harness — run from the repository root.

Builds `@tensorspine//:tspl` in a ZML checkout, then checks the generator against the
language's own tools: for every corpus document, what `tspl` reads out of the derived
document must equal what `tools/derive.py` put in it.

    generators/zml/tests/run_zml.py [--zml DIR] [--model-artifacts DIR] [--model NAME] [--keep]

Then, for whichever checkpoints it is given, the numbers against the reference
generator's committed fixtures: llama3-8b, colbert-v2, and the two hybrids.

Two directories, both shell variables, neither with a default inside the tree:
$ZML_HOME is the ZML checkout that is the build root, and $TENSORSPINE_MODEL_ARTIFACTS is the
one runtime directory — `derived/` for the documents, `weights/<artifact>/` for the
checkpoints they locate tensors in, `dumps/` for what a run leaves behind. Prints `skip`
and exits 0 for whatever is absent, so the suite runs anywhere and says which checks it
did not make.
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))          # generators/zml/tests
GENERATOR = os.path.dirname(HERE)                          # generators/zml
ROOT = os.path.dirname(os.path.dirname(GENERATOR))
MODELS = os.path.join(ROOT, 'data', 'models')
TARGET = '@tensorspine//:tspl'

COUNTS = re.compile(
    r'(\d+) occurrences, (\d+) values, (\d+) parameter tensors, '
    r'(\d+) states, (\d+) edges, (\d+) ordered')


def fixture_metadata(path):
    """A fixture's metadata (docs/TENSORSPINE-FIXTURE.md), from its header alone: the prompt,
    the truncation and the tolerance are the fixture's, never written down here."""
    import json as json_mod
    import struct
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json_mod.loads(f.read(n))
    meta = {k: json_mod.loads(v) for k, v in (header.get('__metadata__') or {}).items()}
    if meta.get('schema') != 'tensorspine-fixture/1' or meta.get('kind') != 'integration':
        raise ValueError(f'{path}: not an integration fixture on tensorspine-fixture/1')
    return meta


def expected(path):
    """What the language put in the document, counted here."""
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    return (len(d['d1']['nodes']), len(d['d2']['values']), len(d['d3']['tensors']),
            len(d['d4']['states']), len(d['d1']['edges']), len(d['d1']['topological_order']))


def derive(out_dir, only=None):
    """The derived documents, through the language's own tool — never re-implemented here."""
    args = [os.path.join(ROOT, 'tools', 'tensorspine'), '--derive', '-o', out_dir]
    args.append(os.path.join(MODELS, f'{only}.json') if only else MODELS)
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL)
    return sorted(f for f in os.listdir(out_dir) if f.endswith('.derived.json'))


FIXTURE = 'generators/reference/fixtures/llama3-8b.3layers.hf.safetensors'
COLBERT_FIXTURE = 'generators/reference/fixtures/colbert-v2.12layers.hf.safetensors'

# The artifact each document's D3 locations name, under `$TENSORSPINE_MODEL_ARTIFACTS/weights`.
# GLOSSARY calls one of these directories "the artifact the document wraps"; the variable
# is the plural of that, and holds the derived documents beside them.
WEIGHTS = {
    'llama3-8b': 'Meta-Llama-3-8B',
    'colbert-v2': 'colbertv2.0',
    'qwen3.5-4b-text': 'Qwen3.5-4B',
    'qwen3.8-27b-text': 'Qwen3.8-27B',
}

# The hybrids, and the fixture each is checked against. Both are three gated-delta layers
# then one attention layer, at different quantities: 32 value heads against 48, a
# convolution 8192 channels wide against 10240, a 2560 residual stream against 5120. One
# document could have been fitted; two at different sizes cannot be.
HYBRIDS = [
    ('qwen3.5-4b-text', 'generators/reference/fixtures/qwen3.5-4b-text.4layers.hf.safetensors'),
    ('qwen3.8-27b-text', 'generators/reference/fixtures/qwen3.8-27b-text.4layers.hf.safetensors'),
]

# Relative to each value's own scale: an f32 rounding budget, recorded from what the
# runs actually produce, not chosen in advance. Layers 1 and 2 carry a 225.7 activation,
# so their absolute error is larger while the relative error does not move.
TOLERANCE = 5e-06


def evaluate(binary, derived, checkpoint, scratch, dumps):
    """The numbers, against the reference generator's fixture — the oracle.

    The fixture is `transformers` hooked at D6's layer cuts and at every state, on the
    same six token identifiers, dumped by the reference. Two generators agreeing there
    is the evidence; the embedding and the first norm are checked against arithmetic
    written down here instead, being short enough that the check inherits nobody's
    reading of them."""
    try:
        import numpy as np
        import torch
        from safetensors.numpy import load_file
        from safetensors.torch import safe_open
    except ImportError as e:
        print(f'skip: evaluation needs numpy, torch and safetensors ({e})')
        return 0, 0

    index = os.path.join(checkpoint, 'model.safetensors.index.json')
    if not os.path.isfile(index):
        print(f'skip: no checkpoint at {checkpoint}')
        return 0, 0
    fixture = os.path.join(ROOT, FIXTURE)
    if not os.path.isfile(fixture):
        print(f'skip: no fixture at {FIXTURE}')
        return 0, 0

    weight_map = json.load(open(index, encoding='utf-8'))['weight_map']

    def weight(name):
        with safe_open(os.path.join(checkpoint, weight_map[name]), framework='pt') as f:
            return f.get_tensor(name).to(torch.float32).numpy()

    with open(derived, encoding='utf-8') as f:
        doc = json.load(f)
    ids = fixture_metadata(fixture)['ids']              # the fixture's own prompt
    eps = doc['d1']['nodes']['decoder/attn_n[layer=0]']['arguments']['eps']
    rows = weight('model.embed_tokens.weight')[ids]
    scale = weight('model.layers.0.input_layernorm.weight')

    fx = load_file(fixture)
    os.makedirs(dumps, exist_ok=True)

    # value -> (what it must equal, the tolerance relative to that value's own scale)
    wanted = [('embed.output', rows, 0.0),
              ('decoder/attn_n[layer=0].output',
               rows / np.sqrt((rows ** 2).mean(-1, keepdims=True) + eps) * scale, TOLERANCE)]
    for layer in range(3):
        wanted.append((f'decoder/ffn_r[layer={layer}].output',
                       fx[f'value/decoder/ffn_r[layer={layer}].output'], TOLERANCE))

    checked, failed = 0, 0
    for value, want, tol in wanted:
        path = os.path.join(scratch, 'value.bin')
        run = subprocess.run([binary, f'--derived={derived}', f'--checkpoint={checkpoint}',
                              f'--until={value}', f'--ids={",".join(map(str, ids))}', f'--out={path}', f'--dump={dumps}'],
                             capture_output=True)
        checked += 1
        if run.returncode != 0:
            print(f'FAIL {value}: {run.stderr.decode(errors="replace")[-600:]}')
            failed += 1
            continue
        got = np.fromfile(path, dtype=np.float32).reshape(want.shape)
        err = float(np.abs(got - want).max()) / max(float(np.abs(want).max()), 1e-30)
        ok = err <= tol
        failed += 0 if ok else 1
        print(f'{"OK  " if ok else "FAIL"} {value}: {err:.2e} of scale (tolerance {tol:.0e})')

    # the states the last run left behind: every component of every KV the fixture holds
    for layer in range(3):
        for component in ('k', 'v'):
            path = os.path.join(dumps, f'decoder.attn.kv[layer={layer}].{component}.bin')
            if not os.path.isfile(path):
                continue
            want = fx[f'state/decoder.attn.kv[layer={layer}]/{component}']
            got = np.fromfile(path, dtype=np.float32).reshape(want.shape)
            err = float(np.abs(got - want).max()) / float(np.abs(want).max())
            ok = err <= TOLERANCE
            checked += 1
            failed += 0 if ok else 1
            print(f'{"OK  " if ok else "FAIL"} state kv[layer={layer}].{component}: '
                  f'{err:.2e} of scale (tolerance {TOLERANCE:.0e})')
    return checked, failed


MANIFEST = os.path.join(GENERATOR, 'capabilities.json')
UNIT_FIXTURES = os.path.join(ROOT, 'generators', 'reference', 'fixtures', 'contracts')


def _raw(path, compute, shape):
    """Bytes tspl wrote in the compute dtype, as an f32 array on `shape`; an append state's
    buffer is longer than the positions the fixture recorded, so the leading rows are taken."""
    import numpy as np
    if compute == 'bf16':
        got = (np.fromfile(path, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    else:
        got = np.fromfile(path, dtype=np.float32)
    if got.size == int(np.prod(shape)):
        return got.reshape(shape)
    return got.reshape([-1] + list(shape[1:]))[:shape[0]]


def unit_fixtures(binary, scratch):
    """This generator as a conformer (Specification §4.2): every unit fixture the reference
    witness recorded (docs/TENSORSPINE-FIXTURE.md) whose contract and arguments this
    generator's manifest admits, run by tspl from the fixture's own document with the
    fixture as its checkpoint, at f32 and at bf16, and compared — every output and every
    state after every invocation — within the tolerance the fixture states for that dtype.
    A fixture the manifest refuses is skipped and says why: the manifest is the authority on
    what is claimed, and the fixtures are the check on what is claimed."""
    try:
        import numpy as np
        from safetensors.numpy import load_file
    except ImportError as e:
        print(f'skip: unit fixtures need numpy and safetensors ({e})')
        return 0, 0
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    import artifact
    import capabilities as capabilities_mod
    with open(MANIFEST, encoding='utf-8') as f:
        manifest = json.load(f)
    fixtures = sorted(glob.glob(os.path.join(UNIT_FIXTURES, '*', '*.safetensors')))
    if not fixtures:
        print(f'skip: no unit fixtures under {os.path.relpath(UNIT_FIXTURES, ROOT)}')
        return 0, 0
    checked = failed = 0
    for fixture in fixtures:
        meta = artifact.read_metadata(fixture)
        cid = f"{meta['contract']['name']}@{meta['contract']['version']}"
        entry = manifest['contracts'].get(cid)
        if entry is None:
            print(f"skip {meta['id']}: no entry for {cid} in the manifest")
            continue
        reasons = capabilities_mod.supports(entry, meta['arguments'])
        if reasons:
            print(f"skip {meta['id']}: the manifest does not admit {reasons[0]}")
            continue
        work = tempfile.mkdtemp(prefix='tspl-unit-', dir=scratch)
        doc = dict(meta['document'], catalog=[{'base': os.path.join(ROOT, 'data', 'catalog') + os.sep}])
        model_path = os.path.join(work, doc['model'] + '.json')
        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(doc, f)
        run = subprocess.run([os.path.join(ROOT, 'tools', 'tensorspine'), '--derive', model_path, '-o', work],
                             capture_output=True, text=True)
        derived = os.path.join(work, doc['model'] + '.derived.json')
        if run.returncode != 0 or not os.path.isfile(derived):
            print(f"FAIL {meta['id']}: the fixture's document does not derive: {run.stdout[-300:]}")
            checked += 1
            failed += 1
            continue
        fx = load_file(fixture)
        counts = []
        for k, delivered in enumerate(meta['invocations']):
            n = {v for v in delivered.values()}
            assert len(n) == 1, f"{meta['id']}: an invocation delivers one element count to every input"
            counts.append(str(n.pop()))
            for name in delivered:
                t = fx[f'in/{k}/{name}']
                (t.astype(np.int32) if t.dtype.kind in 'iu' else t.astype(np.float32)).tofile(os.path.join(work, f'in.{k}.{name}.bin'))
        output = next(iter(doc['interfaces']['outputs']))
        for compute, tol in sorted(meta['tolerance'].items()):
            if compute not in manifest['compute_dtypes']:
                continue
            out_dir = os.path.join(work, compute)
            os.makedirs(out_dir, exist_ok=True)
            for f in os.listdir(work):
                if f.startswith('in.'):
                    shutil.copy(os.path.join(work, f), out_dir)
            run = subprocess.run([binary, f'--derived={derived}', f'--checkpoint={fixture}', f'--unit={out_dir}',
                                  f'--invocations={",".join(counts)}', f'--compute={compute}'], capture_output=True)
            checked += 1
            if run.returncode != 0:
                print(f"FAIL {meta['id']} at {compute}: {run.stderr.decode(errors='replace')[-500:]}")
                failed += 1
                continue
            worst, bad = 0.0, []
            for k in range(len(counts)):
                want = fx[f'out/{k}/{output}']
                pairs = [(f'out/{k}/{output}', os.path.join(out_dir, f'out.{k}.bin'), want)]
                for key in fx:
                    if key.startswith(f'state/{k}/'):
                        _, _, identity, component = key.split('/')
                        pairs.append((key, os.path.join(out_dir, f'state.{k}.{identity}.{component}.bin'), fx[key]))
                for key, path, want in pairs:
                    if not os.path.isfile(path):
                        bad.append(f'{key}: nothing written')
                        continue
                    got = _raw(path, compute, want.shape)
                    d = np.abs(got - want)
                    worst = max(worst, float(d.max()) if d.size else 0.0)
                    if bool((d > tol['atol'] + tol['rtol'] * np.abs(want)).any()):
                        bad.append(f'{key}: max |d| {float(d.max()):.2e}')
            failed += 1 if bad else 0
            print(f"{'FAIL' if bad else 'OK  '} {meta['id']} at {compute}: {len(counts)} invocation(s), max |d| {worst:.2e} "
                  f"(atol {tol['atol']:g} rtol {tol['rtol']:g})" + (f"  {bad[:2]}" if bad else ''))
    return checked, failed


def manifest(binary):
    """The manifest, regenerated from the primitives' own tables and diffed against the
    committed one — `generators/CAPABILITIES.md`'s rule, that a manifest comes from the
    code and never from a hand-maintained list, only holds if something checks. The
    version and the date are read back out of the committed file and passed in, so the
    diff is about the tables and not about the calendar."""
    if not os.path.isfile(MANIFEST):
        print(f'FAIL manifest: {MANIFEST} is not committed')
        return 1, 1
    with open(MANIFEST, encoding='utf-8') as f:
        committed = f.read()
    known = json.loads(committed)['generator']

    out = os.path.join(tempfile.mkdtemp(prefix='tspl-manifest-'), 'capabilities.json')
    try:
        run = subprocess.run([binary, f'--capabilities={out}',
                              f'--version={known["version"]}', f'--generated={known["generated"]}'],
                             capture_output=True)
        if run.returncode != 0:
            print(f'FAIL manifest: {run.stderr.decode(errors="replace")[-400:]}')
            return 1, 1
        with open(out, encoding='utf-8') as f:
            regenerated = f.read()
    finally:
        shutil.rmtree(os.path.dirname(out), ignore_errors=True)

    if regenerated != committed:
        print('FAIL manifest: regenerating it from the primitives gives something else; '
              f'run `tspl --capabilities={MANIFEST} --version=… --generated=…`')
        return 1, 1
    print(f'OK   manifest: {len(json.loads(committed)["contracts"])} contracts, regenerated identically')

    # And the language's own reader agrees it can run what it claims.
    reader = subprocess.run([os.path.join(ROOT, 'tools', 'tensorspine'), '--capabilities',
                             MANIFEST, os.path.join(MODELS, 'llama3-8b.json')], capture_output=True)
    text = (reader.stdout + reader.stderr).decode(errors='replace')
    if 'can run' not in text:
        print(f'FAIL manifest: the reader does not agree it can run llama3-8b\n{text.strip()[-400:]}')
        return 2, 1
    print('OK   manifest: tensorspine --capabilities agrees it can run llama3-8b')
    return 2, 0


def colbert(binary, derived_dir, checkpoint, scratch):
    """The other generator's fixture for a document with no generative output and no
    state at all — the shape llama3-8b cannot exercise. Its identifiers and its cuts are
    the reference's; agreeing with them is two generators agreeing, not one agreeing with
    itself."""
    try:
        import numpy as np
        from safetensors.numpy import load_file
    except ImportError:
        return 0, 0
    fixture = os.path.join(ROOT, COLBERT_FIXTURE)
    if not os.path.isfile(fixture) or not os.path.isdir(checkpoint):
        print(f'skip: colbert needs {COLBERT_FIXTURE} and a checkpoint at {checkpoint}')
        return 0, 0

    derived = os.path.join(derived_dir, 'colbert-v2.derived.json')
    if not os.path.isfile(derived):
        return 0, 0
    fx = load_file(fixture)
    ids = ','.join(map(str, fixture_metadata(fixture)['ids']))      # the fixture's own prompt

    checked = failed = 0
    for value, key in [(f'enc/ffn_n[layer={i}].output', f'value/enc/ffn_n[layer={i}].output') for i in (0, 11)] \
            + [('pooler.output', 'value/pooler.output')]:
        path = os.path.join(scratch, 'colbert.bin')
        run = subprocess.run([binary, f'--derived={derived}', f'--checkpoint={checkpoint}',
                              f'--until={value}', f'--ids={ids}', f'--out={path}'], capture_output=True)
        checked += 1
        if run.returncode != 0:
            print(f'FAIL colbert {value}: {run.stderr.decode(errors="replace")[-400:]}')
            failed += 1
            continue
        want = fx[key]
        got = np.fromfile(path, dtype=np.float32).reshape(want.shape)
        err = float(np.abs(got - want).max()) / float(np.abs(want).max())
        ok = err <= TOLERANCE
        failed += 0 if ok else 1
        print(f'{"OK  " if ok else "FAIL"} colbert {value}: {err:.2e} of scale (tolerance {TOLERANCE:.0e})')
    return checked, failed


def weights(artifacts, model):
    """Where this document's weights are, by the layout — `weights/<artifact>` under the
    one directory. `None` when they are not there, and the checks that need them say they
    did not run rather than guess at another location."""
    if not artifacts:
        return None
    path = os.path.join(artifacts, 'weights', WEIGHTS[model])
    return path if os.path.isdir(path) else None


def qwen(binary, derived_dir, checkpoint, model, fixture_path, scratch, dumps):
    """A hybrid document: three gated-delta layers and one attention layer.

    Its four kinds of state are the two laws llama3-8b cannot exercise — the recurrent
    matrix is `fixed`, read and written whole, and the convolution history is a `window`
    consumed as a ring — beside the KV cache, which is the one it can. So this is where
    the state machinery is actually decided, and where a layout choice would show: a
    window held as a rotating ring rather than a chronological slide would give the same
    outputs and a different dumped buffer, and the fixture holds the buffer.

    The attention layer carries four branches at once besides — partial rope, mrope,
    an rms qk_norm and the per-head output gate — and none of them is separable from the
    others in a checkpoint, which is why they arrive together."""
    try:
        import numpy as np
        from safetensors.numpy import load_file
    except ImportError:
        return 0, 0
    fixture = os.path.join(ROOT, fixture_path)
    if not os.path.isfile(fixture) or not os.path.isdir(checkpoint):
        print(f'skip: {model} needs {fixture_path} and a checkpoint at {checkpoint}')
        return 0, 0

    derived = os.path.join(derived_dir, f'{model}.derived.json')
    if not os.path.isfile(derived):
        return 0, 0
    fx = load_file(fixture)
    ids = ','.join(map(str, fixture_metadata(fixture)['ids']))      # the fixture's own prompt
    os.makedirs(dumps, exist_ok=True)

    checked = failed = 0
    for layer in range(4):
        value = f'decoder/mlp_r[layer={layer}].output'
        path = os.path.join(scratch, 'hybrid.bin')
        run = subprocess.run([binary, f'--derived={derived}', f'--checkpoint={checkpoint}',
                              f'--until={value}', f'--ids={ids}', f'--out={path}', f'--dump={dumps}'],
                             capture_output=True)
        checked += 1
        if run.returncode != 0:
            print(f'FAIL {model} {value}: {run.stderr.decode(errors="replace")[-400:]}')
            failed += 1
            continue
        want = fx[f'value/{value}']
        got = np.fromfile(path, dtype=np.float32).reshape(want.shape)
        err = float(np.abs(got - want).max()) / float(np.abs(want).max())
        ok = err <= TOLERANCE
        failed += 0 if ok else 1
        print(f'{"OK  " if ok else "FAIL"} {model} {value}: {err:.2e} of scale (tolerance {TOLERANCE:.0e})')

    # every state the deepest run left behind, named by its D4 identity whatever layout
    # held it: three convolution histories, three recurrent matrices, one KV cache
    states = [(f'decoder.gdn.conv[layer={i}]', 'w') for i in range(3)] \
        + [(f'decoder.gdn.recurrent[layer={i}]', 's') for i in range(3)] \
        + [('decoder.attn.kv[layer=3]', c) for c in ('k', 'v')]
    for identity, component in states:
        path = os.path.join(dumps, f'{identity}.{component}.bin')
        key = f'state/{identity}/{component}'
        if not os.path.isfile(path) or key not in fx:
            print(f'FAIL {model} state {identity}.{component}: nothing dumped')
            checked += 1
            failed += 1
            continue
        want = fx[key]
        got = np.fromfile(path, dtype=np.float32).reshape(want.shape)
        err = float(np.abs(got - want).max()) / float(np.abs(want).max())
        ok = err <= TOLERANCE
        checked += 1
        failed += 0 if ok else 1
        print(f'{"OK  " if ok else "FAIL"} {model} state {identity}.{component}: '
              f'{err:.2e} of scale (tolerance {TOLERANCE:.0e})')
    return checked, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zml', default=os.environ.get('ZML_HOME'),
                    help='the ZML checkout that is the build root ($ZML_HOME)')
    ap.add_argument('--model-artifacts', default=os.environ.get('TENSORSPINE_MODEL_ARTIFACTS'),
                    help='the one runtime directory ($TENSORSPINE_MODEL_ARTIFACTS): `derived/` holds the '
                         'documents, `weights/<artifact>/` the checkpoints they locate tensors in, '
                         '`dumps/` what a run leaves behind. Without it the documents go to a '
                         'temporary directory and every numerical check is skipped. Weights that '
                         'live elsewhere are reached by making `weights` a symlink')
    ap.add_argument('--model', help='one document by name, instead of the whole corpus')
    ap.add_argument('--compilation-mode', default='dbg', choices=('opt', 'dbg', 'fastbuild'),
                    help="Bazel's -c: `dbg` by default, which is what every measurement here used "
                         'and what runs the full model, and which keeps leak checking. `opt` runs '
                         'faster and rebuilds XLA from scratch to get there')
    ap.add_argument('--keep', action='store_true', help='keep the derived documents')
    a = ap.parse_args()

    if not a.zml:
        print('skip: no ZML checkout given (--zml or $ZML_HOME)')
        return 0

    bazel = os.path.join(a.zml, 'bazel.sh')
    if not os.path.isfile(bazel):
        print(f'skip: no ZML checkout at {a.zml} (--zml or $ZML_HOME)')
        return 0
    if not os.path.isfile(os.path.join(a.zml, 'MODULE.bazel')):
        print(f'skip: {a.zml} has no MODULE.bazel')
        return 0
    # The generator is injected into the ZML build rather than declared in its
    # MODULE.bazel: nothing in a repository we do not own has to be edited, and the path
    # is computed here instead of written down anywhere.
    inject = f'--inject_repository=tensorspine={GENERATOR}'

    build = subprocess.run([bazel, 'build', '-c', a.compilation_mode, inject, TARGET], cwd=a.zml,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if build.returncode != 0:
        sys.stderr.write(build.stderr.decode(errors='replace')[-4000:])
        print('FAIL: build')
        return 1
    binary = os.path.join(a.zml, 'bazel-out', f'k8-{a.compilation_mode}', 'bin',
                          'external', '+local_repository+tensorspine', 'tspl')
    if not os.path.isfile(binary):
        print(f'FAIL: built, but no binary at {binary}')
        return 1

    # One directory, three roles. Unset, the documents go somewhere temporary and every
    # check that needs weights says it did not run.
    if a.model_artifacts:
        out_dir = os.path.join(a.model_artifacts, 'derived')
        dumps_root = os.path.join(a.model_artifacts, 'dumps')
    else:
        out_dir = tempfile.mkdtemp(prefix='tspl-derived-')
        dumps_root = out_dir
    os.makedirs(out_dir, exist_ok=True)
    scratch = tempfile.mkdtemp(prefix='tspl-work-')
    try:
        names = derive(out_dir, a.model)
        failed = 0
        for name in names:
            path = os.path.join(out_dir, name)
            model = name[:-len('.derived.json')]
            run = subprocess.run([binary, f'--derived={path}'], capture_output=True)
            out = (run.stdout + run.stderr).decode(errors='replace')
            m = COUNTS.search(out)
            if run.returncode != 0 or not m:
                print(f'FAIL {model}: {out.strip().splitlines()[-1] if out.strip() else "no output"}')
                failed += 1
                continue
            got, want = tuple(int(x) for x in m.groups()), expected(path)
            if got != want:
                print(f'FAIL {model}: tspl {got} != derive {want}')
                failed += 1
            else:
                print(f'OK   {model}: {got[0]} occurrences, {got[2]} tensors, {got[3]} states')
        checked = len(names)
        if not a.model_artifacts:
            print('\nskip: no artifacts directory (--artifacts or $TENSORSPINE_MODEL_ARTIFACTS); '
                  'the numerical checks did not run')

        llama = weights(a.model_artifacts, 'llama3-8b')
        derived_llama = os.path.join(out_dir, 'llama3-8b.derived.json')
        if llama and os.path.isfile(derived_llama):
            print()
            more, bad = evaluate(binary, derived_llama, llama, scratch,
                                 os.path.join(dumps_root, 'llama3-8b'))
            checked += more
            failed += bad

        if weights(a.model_artifacts, 'colbert-v2'):
            print()
            more, bad = colbert(binary, out_dir, weights(a.model_artifacts, 'colbert-v2'), scratch)
            checked += more
            failed += bad

        for model, fixture_path in HYBRIDS:
            checkpoint = weights(a.model_artifacts, model)
            if not checkpoint:
                continue
            print()
            more, bad = qwen(binary, out_dir, checkpoint, model, fixture_path, scratch,
                             os.path.join(dumps_root, model))
            checked += more
            failed += bad

        print()
        more, bad = manifest(binary)
        checked += more
        failed += bad

        print()
        more, bad = unit_fixtures(binary, scratch)
        checked += more
        failed += bad
        print(f'\n{checked - failed} passed, {failed} failed')
        return 1 if failed else 0
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        if a.keep or a.model_artifacts:
            print(f'derived documents kept in {out_dir}')
        else:
            shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
