#!/usr/bin/env python3
"""The ONNX generator's harness — run from the repository root.

  1. Every corpus document derives (through the language's own tool) and the generator reads
     back the counts the language put in it.
  2. This generator as a conformer (Specification §4.2): every unit fixture the reference witness
     recorded whose contract, arguments and states the manifest admits is emitted from the
     fixture's own document with the fixture as its checkpoint, run through onnxruntime invocation
     by invocation, and compared — every output and every state — within the fixture's tolerance
     for f32. A fixture the manifest refuses is skipped and says why.
  3. Every committed integration fixture whose document the manifest can run and whose checkpoint
     is on disk: the document truncated as the fixture says, emitted with the values crossing
     every layer cut as outputs, the prefill compared value by value, state by state and on the
     logits within the fixture's tolerance, then the greedy tokens; `skip` when the checkpoint is
     absent.
  4. The manifest regenerates from the emitters' tables identically, and the language's reader
     agrees it can run llama3-8b.

    generators/onnx/tests/run_onnx.py [--model-artifacts DIR]
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GENERATOR = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(GENERATOR))
MODELS = os.path.join(ROOT, 'data', 'models')
TOOL = os.path.join(ROOT, 'tools', 'tensorspine')
MANIFEST = os.path.join(GENERATOR, 'capabilities.json')
REFERENCE_FIXTURES = os.path.join(ROOT, 'generators', 'reference', 'fixtures')
UNIT_FIXTURES = os.path.join(REFERENCE_FIXTURES, 'contracts')
CHECKPOINTS = {'llama3-8b': 'Meta-Llama-3-8B', 'shieldstral-3b': 'Shieldstral-1.0-3B', 'colbert-v2': 'colbertv2.0',
               'qwen3.5-4b-text': 'Qwen3.5-4B', 'qwen3.8-27b-text': 'Qwen3.8-27B', 'qwen3.5-35b-a3b': 'Qwen3.5-35B-A3B',
               'whisper-large-v3': 'whisper-large-v3', 'voxtral-realtime': 'Voxtral-Mini-4B-Realtime-2602'}

sys.path.insert(0, GENERATOR)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import artifact                        # noqa: E402
import capabilities as capabilities_mod   # noqa: E402
import graph as graph_mod              # noqa: E402
import loader                          # noqa: E402
import registry                        # noqa: E402
from emit import Emitter, Refusal      # noqa: E402
from session import Session, greedy    # noqa: E402


def check(label, ok, detail=''):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f"\n         {detail}" if detail and not ok else ''))
    return ok


def derive(model_path, out_dir):
    run = subprocess.run([TOOL, '--derive', model_path, '-o', out_dir], capture_output=True, text=True)
    name = os.path.basename(model_path)[:-5]
    path = os.path.join(out_dir, name + '.derived.json')
    if run.returncode != 0 or not os.path.isfile(path):
        raise RuntimeError(f"{name} does not derive: {run.stdout[-300:]}")
    return path


def within(got, want, atol, rtol):
    d = np.abs(got.astype(np.float32) - want.astype(np.float32))
    return bool((d <= atol + rtol * np.abs(want)).all()), float(d.max()) if d.size else 0.0


# --- 1: the corpus reads back ---------------------------------------------------------------

def corpus_counts(scratch):
    ok = True
    for path in sorted(glob.glob(os.path.join(MODELS, '*.json'))):
        name = os.path.basename(path)[:-5]
        try:
            g = graph_mod.load(derive(path, scratch))
        except (RuntimeError, ValueError) as e:
            ok &= check(f"{name}: derives and reads back", False, str(e)[:200])
            continue
        doc = g.doc
        want = (len(doc['d1']['nodes']), len(doc['d2']['values']), len(doc['d3']['tensors']), len(doc['d4']['states']),
                len(doc['d1']['edges']), len(doc['d1']['topological_order']))
        ok &= check(f"{name}: {want[0]} occurrences, {want[1]} values, {want[2]} tensors, {want[3]} states read back", g.counts() == want)
    return ok


# --- 2: the unit fixtures -----------------------------------------------------------------------

def unit_fixtures(scratch, manifest):
    ok = True
    prims = registry.load_primitives()
    for fixture in sorted(glob.glob(os.path.join(UNIT_FIXTURES, '*', '*.safetensors'))):
        meta = artifact.read_metadata(fixture)
        cid = f"{meta['contract']['name']}@{meta['contract']['version']}"
        entry = manifest['contracts'].get(cid)
        if entry is None:
            print(f"  skip {meta['id']}: no entry for {cid} in the manifest")
            continue
        reasons = capabilities_mod.supports(entry, meta['arguments'])
        if reasons:
            print(f"  skip {meta['id']}: the manifest does not admit {reasons[0]}")
            continue
        work = tempfile.mkdtemp(prefix='unit-', dir=scratch)
        doc = dict(meta['document'], catalog=[{'base': os.path.join(ROOT, 'data', 'catalog') + os.sep}])
        model_path = os.path.join(work, doc['model'] + '.json')
        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(doc, f)
        g = graph_mod.load(derive(model_path, work))
        laws = {s['law'] for s in g.states.values()}
        if laws - set(manifest['state_laws']):
            print(f"  skip {meta['id']}: state law {sorted(laws - set(manifest['state_laws']))} not implemented")
            continue
        from safetensors.numpy import load_file
        fx = load_file(fixture)
        emitter = Emitter(g, loader.Source(g, fixture), prims)
        session = Session(emitter, g)
        atol, rtol = meta['tolerance']['f32']['atol'], meta['tolerance']['f32']['rtol']
        worst, bad = 0.0, []
        try:
            for k, delivered in enumerate(meta['invocations']):
                ins = {}
                for name in delivered:
                    t = fx[f"in/{k}/{name}"]
                    ins[name] = t.astype(np.int64) if t.dtype.kind in 'iu' else t.astype(np.float32)
                outs = session.run(ins)
                for oname, o in g.interfaces['outputs'].items():
                    got, want = outs[f"{o['node']}.{o['port']}"], fx[f"out/{k}/{oname}"]
                    fine, d = within(got, want, atol, rtol) if got.shape == want.shape else (False, float('inf'))
                    worst = max(worst, d)
                    if not fine:
                        bad.append(f"out/{k}/{oname}: max |d| {d:.2e} (shapes {got.shape} vs {want.shape})")
                for key in fx:
                    if key.startswith(f"state/{k}/"):
                        _s, _k, ident, comp = key.split('/')
                        got, want = session.states[f"{ident}/{comp}"], fx[key]
                        fine, d = within(got, want, atol, rtol) if got.shape == want.shape else (False, float('inf'))
                        worst = max(worst, d)
                        if not fine:
                            bad.append(f"{key}: max |d| {d:.2e} (shapes {got.shape} vs {want.shape})")
        except Exception as e:  # noqa: BLE001  a refusal, or the runtime rejecting the graph: reported, not fatal
            bad.append(f"{type(e).__name__}: {str(e)[:300]}")
        ok &= check(f"{meta['id']} at f32: {len(meta['invocations'])} invocation(s), max |d| {worst:.2e} (atol {atol:g} rtol {rtol:g})", not bad, str(bad[:2]))
    return ok


# --- 3: the integration fixtures --------------------------------------------------------------

def truncated(model_path, composition, stop, out_dir):
    """The document with the composition's index range shortened, with every literal quantity equal
    to the old extent that an index expression addressing the composition cites, and every literal
    equal to the old last index in the bindings addressing it — the reference harness's rule."""
    with open(model_path, encoding='utf-8') as f:
        model = json.load(f)
    comp = model['compositions'][composition]
    index = next(iter(comp['indices']))
    old = comp['indices'][index]['stop']['literal']
    comp['indices'][index]['stop'] = {'literal': stop}

    def quantities(node, out):
        if isinstance(node, dict):
            if isinstance(node.get('quantity'), str):
                out.add(node['quantity'])
            for v in node.values():
                quantities(v, out)
        elif isinstance(node, list):
            for v in node:
                quantities(v, out)
        return out
    cited = set()

    def cite(node):
        if isinstance(node, dict):
            if node.get('kind') == 'generated' and node.get('composition') == composition:
                quantities(node.get('indices', {}), cited)
            elif 'site' in node and 'indices' in node:
                quantities(node['indices'], cited)
            for v in node.values():
                cite(v)
        elif isinstance(node, list):
            for v in node:
                cite(v)
    cite(model.get('bindings', {}))
    cite(comp.get('bindings', {}))
    for q in cited:
        entry = model['quantities'].get(q)
        if entry and entry['source'].get('kind') == 'literal' and entry['source'].get('value') == old:
            entry['source']['value'] = stop

    def walk(node):
        if isinstance(node, dict):
            if set(node) == {'literal'} and node['literal'] == old - 1:
                node['literal'] = stop - 1
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(model.get('bindings', {}))
    walk(comp.get('bindings', {}))
    model['catalog'] = [{'base': os.path.join(ROOT, 'data', 'catalog') + os.sep}]
    model['model'] = f"{model['model']}-{stop}layers"
    path = os.path.join(out_dir, f"{os.path.basename(model_path)[:-5]}.{stop}layers.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(model, f, indent=1)
    return path


def integration_fixtures(scratch, manifest, artifacts):
    ok = True
    prims = registry.load_primitives()
    for fixture in sorted(glob.glob(os.path.join(REFERENCE_FIXTURES, '*.hf.safetensors'))):
        meta = artifact.read_metadata(fixture)
        document = meta['document']
        tag = os.path.basename(fixture).replace('.hf.safetensors', '').rsplit('.', 1)[-1]
        checkpoint = os.path.join(artifacts, 'weights', CHECKPOINTS.get(document, document)) if artifacts else None
        if not checkpoint or not os.path.isdir(checkpoint):
            print(f"  skip {document} ({tag}): checkpoint not on disk")
            continue
        work = tempfile.mkdtemp(prefix='fixture-', dir=scratch)
        path = truncated(os.path.join(MODELS, document + '.json'), meta['truncation']['composition'], meta['truncation']['layers'], work)
        g = graph_mod.load(derive(path, work))
        if [k for k in artifact.read_header(fixture) if k.startswith('in/')]:
            print(f"  skip {document} ({tag}): the fixture delivers non-token inputs, not emitted yet")
            continue
        em = Emitter(g, None, prims)
        refused = em.refusals(em.evaluable({g.token_input}))
        if refused:
            print(f"  skip {document} ({tag}): {refused[0]}")
            continue
        errors, _a, stats = loader.verify(g, checkpoint)
        ok &= check(f"{document} ({tag}): the {meta['truncation']['layers']}-layer document verifies against the checkpoint", not errors, errors[:1])
        from safetensors.numpy import load_file
        theirs = load_file(fixture)
        atol, rtol = meta['tolerance']['f32']['atol'], meta['tolerance']['f32']['rtol']
        session = Session(Emitter(g, loader.Source(g, checkpoint), prims), g, dump=True)
        ids = meta['ids']
        encoder = g.generative is None
        out = session.run({g.token_input: np.asarray(ids, dtype=np.int64)}) if encoder else session.prefill(ids)
        ours = {f"value/{k}": v for k, v in out.items()}
        if not encoder:
            for key, buf in session.states_after_prefill.items():
                ours[f"state/{key}"] = buf
            o = g.interfaces['outputs'][g.generative[0]]
            logits = out[f"{o['node']}.{o['port']}"]
            ours['logits/last'] = logits[-1]
            ours['logits/argmax'] = logits.argmax(-1)
        worst, bad, compared = 0.0, [], 0
        one_sided = sorted(set(theirs) ^ set(ours))     # a layer output the cuts do not carry, as the reference's own comparison
        for key, want in theirs.items():
            if key not in ours:
                continue
            got = ours[key]
            compared += 1
            if key.endswith('argmax'):
                if not np.array_equal(got, want):
                    bad.append(f"{key}: {int((got != want).sum())} differ")
                continue
            fine, d = within(got, want, atol, rtol) if got.shape == want.shape else (False, float('inf'))
            worst = max(worst, d)
            if not fine:
                bad.append(f"{key}: max |d| {d:.2e}" + ('' if got.shape == want.shape else f" shapes {got.shape} vs {want.shape}"))
        ok &= check(f"{document} ({tag}): {compared} values, states and logits within atol {atol:g} rtol {rtol:g} of transformers (max |d| {worst:.2e}); "
                    f"{len(one_sided)} key(s) on one side only", not bad, str(bad[:3]))
        if not encoder:
            nxt = greedy(out, g)
            tokens = [nxt]
            for _ in range(len(meta['tokens']) - 1):
                nxt = greedy(session.decode(nxt), g)
                tokens.append(nxt)
            ok &= check(f"{document} ({tag}): greedy tokens {tokens} equal transformers' {meta['tokens']}", tokens == meta['tokens'])
    return ok


# --- 4: the manifest ---------------------------------------------------------------------------

def manifest_check(scratch):
    with open(MANIFEST, encoding='utf-8') as f:
        committed = json.load(f)
    sys.path.insert(0, GENERATOR)
    import tsonnx
    fresh = tsonnx.manifest()
    for m in (fresh, committed):
        m['generator'] = {k: v for k, v in m['generator'].items() if k not in ('version', 'generated')}
    ok = check("the committed manifest is what the emitters' tables generate", fresh == committed)
    run = subprocess.run([TOOL, '--capabilities', MANIFEST, os.path.join(MODELS, 'llama3-8b.json')], capture_output=True, text=True)
    ok &= check("tensorspine --capabilities agrees the generator can run llama3-8b", 'can run' in run.stdout, run.stdout[-300:])
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model-artifacts', default=os.environ.get('TENSORSPINE_MODEL_ARTIFACTS'),
                    help='the runtime directory: weights/<artifact>/ for the checkpoints ($TENSORSPINE_MODEL_ARTIFACTS)')
    a = ap.parse_args(argv)
    scratch = tempfile.mkdtemp(prefix='tensorspine-onnx-')
    with open(MANIFEST, encoding='utf-8') as f:
        manifest = json.load(f)
    ok = corpus_counts(scratch)
    ok &= unit_fixtures(scratch, manifest)
    ok &= integration_fixtures(scratch, manifest, a.model_artifacts)
    ok &= manifest_check(scratch)
    shutil.rmtree(scratch, ignore_errors=True)
    print("onnx: all good" if ok else "onnx: FAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
