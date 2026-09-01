#!/usr/bin/env python3
"""The ZML generator's test harness — run from the repository root.

Builds `@tensorspine//:tspl` in a ZML checkout, then checks the generator against the
language's own tools: for every corpus document, what `tspl` reads out of the derived
document must equal what `tools/derive.py` put in it.

    generators/zml/tests/run_zml.py [--zml DIR] [--runtime-dir DIR] [--checkpoint DIR]
                                    [--model NAME] [--keep]

Runtime inputs — the ZML checkout, a working directory, a checkpoint — are named by
$ZML_HOME, $TENSORSPINE_RUNTIME_DIR and $TENSORSPINE_CHECKPOINT, or by the matching flag.
None has a default inside the tree: prints `skip` and exits 0 for whatever is absent, so
the suite runs anywhere and says which checks it did not make.
"""
import argparse
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

# Relative to each value's own scale: an f32 rounding budget, recorded from what the
# runs actually produce, not chosen in advance. Layers 1 and 2 carry a 225.7 activation,
# so their absolute error is larger while the relative error does not move.
TOLERANCE = 5e-06


def evaluate(binary, derived, checkpoint, out_dir):
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
    ids = [128000, 791, 6864, 315, 9822, 374]          # the fixture's own prompt
    eps = doc['d1']['nodes']['decoder/attn_n[layer=0]']['arguments']['eps']
    rows = weight('model.embed_tokens.weight')[ids]
    scale = weight('model.layers.0.input_layernorm.weight')

    fx = load_file(fixture)
    dumps = os.path.join(out_dir, 'dump')
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
        path = os.path.join(out_dir, 'value.bin')
        run = subprocess.run([binary, f'--derived={derived}', f'--checkpoint={checkpoint}',
                              f'--until={value}', f'--out={path}', f'--dump={dumps}'],
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zml', default=os.environ.get('ZML_HOME'),
                    help='the ZML checkout that is the build root ($ZML_HOME)')
    ap.add_argument('--runtime-dir', default=os.environ.get('TENSORSPINE_RUNTIME_DIR'),
                    help='where derived documents go ($TENSORSPINE_RUNTIME_DIR; a temporary directory by default)')
    ap.add_argument('--checkpoint', default=os.environ.get('TENSORSPINE_CHECKPOINT'),
                    help='the safetensors repository D3 locates weights in ($TENSORSPINE_CHECKPOINT); '
                         'without it the numerical checks are skipped')
    ap.add_argument('--model', help='one document by name, instead of the whole corpus')
    ap.add_argument('--compilation-mode', default='opt', choices=('opt', 'dbg', 'fastbuild'),
                    help="Bazel's -c: `opt` by default. A `dbg` build gives `init.gpa` a leak-checking "
                         'allocator that does not return freed pages, and the loader then holds several '
                         'times the weights (measured: 12.93 GiB for 3.18 GiB)')
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

    if a.runtime_dir:
        out_dir = os.path.join(a.runtime_dir, 'derived')
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = tempfile.mkdtemp(prefix='tspl-derived-')
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
        if a.checkpoint:
            llama = os.path.join(out_dir, 'llama3-8b.derived.json')
            if os.path.isfile(llama):
                print()
                more, bad = evaluate(binary, llama, a.checkpoint, out_dir)
                checked += more
                failed += bad
        else:
            print('\nskip: no checkpoint (--checkpoint or $TENSORSPINE_CHECKPOINT); '
                  'the numerical checks did not run')
        print(f'\n{checked - failed} passed, {failed} failed')
        return 1 if failed else 0
    finally:
        if a.keep or a.runtime_dir:
            print(f'derived documents kept in {out_dir}')
        else:
            shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
