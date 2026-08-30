#!/usr/bin/env python3
"""V17 against a checkpoint (§3.4, I9): the located tensors of a document against safetensors
headers — existence, shape with unit axes dropped, dtype — and the four location forms.

  1. Synthetic headers built from llama3-8b's own D3: clean; one name absent; one shape wrong;
     one dtype wrong; unit axes dropped; an unnamed physical tensor is advice, not an error.
  2. The forms on a synthetic D3: stack, concat (parts must sum), slice (must fit).
  3. The real Llama 3 8B checkpoint when it is on disk: zero errors, nothing unnamed; else skip.

    python3 tests/run_artifact.py
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import artifact                        # noqa: E402
import catalog as catalog_mod          # noqa: E402
import derive                          # noqa: E402

LLAMA = os.path.join(ROOT, 'data', 'models', 'llama3-8b.json')
SHIELD_DOC = os.path.join(ROOT, 'data', 'models', 'shieldstral-3b.json')
SHIELDSTRAL = os.path.expanduser('~/work/perso/huggingface/Shieldstral-1.0-3B')
CHECKPOINT = os.path.expanduser('~/work/perso/huggingface/Meta-Llama-3-8B')


def check(label, ok, detail=''):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f"\n         {detail}" if detail and not ok else ''))
    return ok


def headers_of(d3):
    out = {}
    for t in d3['tensors']:
        out[t['location']['tensor']] = {'dtype': t['dtype'], 'shape': [a['extent'] for a in t['shape']], 'file': 'x'}
    return out


def main():
    ok = True
    with open(LLAMA, encoding='utf-8') as f:
        model = json.load(f)
    cat = catalog_mod.load_for(LLAMA, model)
    d3 = derive.products(LLAMA, cat)['d3']
    ok &= check("llama3-8b: every D3 tensor carries an evaluated location",
                all('location' in t for t in d3['tensors']) and len(d3['tensors']) == 291)
    h = headers_of(d3)
    e, a, s = artifact.check(d3, h)
    ok &= check("clean headers: no error, no advice", not e and not a and s['located'] == 291)
    h2 = dict(h); del h2['model.layers.3.self_attn.q_proj.weight']
    e, a, s = artifact.check(d3, h2)
    ok &= check("an absent tensor is an error naming it", len(e) == 1 and 'absent' in e[0] and 'q_proj' in e[0], e[:1])
    h2 = copy.deepcopy(h); h2['model.norm.weight']['shape'] = [4095]
    e, a, s = artifact.check(d3, h2)
    ok &= check("a wrong shape is an error", len(e) == 1 and 'has shape' in e[0], e[:1])
    h2 = copy.deepcopy(h); h2['lm_head.weight']['dtype'] = 'f32'
    e, a, s = artifact.check(d3, h2)
    ok &= check("a wrong dtype is an error", len(e) == 1 and 'is f32' in e[0], e[:1])
    h2 = copy.deepcopy(h); h2['model.norm.weight']['shape'] = [1, 4096, 1]
    e, a, s = artifact.check(d3, h2)
    ok &= check("unit axes the logical shape lacks are dropped", not e)
    h2 = dict(h); h2['model.rotary_emb.inv_freq'] = {'dtype': 'f32', 'shape': [64], 'file': 'x'}
    e, a, s = artifact.check(d3, h2)
    ok &= check("an unnamed physical tensor is advice", not e and len(a) == 1 and s['unnamed'] == 1)

    def d3_of(location, shape, dtype='bf16'):
        return {'tensors': [{'identity': 't', 'dtype': dtype, 'location': location,
                             'shape': [{'axis': 'a', 'extent': n} for n in shape]}]}
    stack = {'stack': {'axis': 'e', 'dim': 0, 'parts': [{'tensor': f'w.{i}'} for i in range(3)]}}
    hh = {f'w.{i}': {'dtype': 'bf16', 'shape': [4], 'file': 'x'} for i in range(3)}
    e, a, s = artifact.check(d3_of(stack, [3, 4]), hh)
    ok &= check("stack: three [4] make [3, 4]", not e, e[:1])
    e, a, s = artifact.check(d3_of(stack, [3, 4]), {k: v for k, v in hh.items() if k != 'w.1'})
    ok &= check("stack: a missing part is an error", len(e) == 1 and 'absent' in e[0])
    concat = {'concat': {'axis': 'r', 'dim': 0, 'parts': [{'tensor': 'g'}, {'tensor': 'u'}]}}
    hh = {'g': {'dtype': 'bf16', 'shape': [2, 4], 'file': 'x'}, 'u': {'dtype': 'bf16', 'shape': [4, 4], 'file': 'x'}}
    e, a, s = artifact.check(d3_of(concat, [6, 4]), hh)
    ok &= check("concat: [2,4] and [4,4] make [6, 4]", not e, e[:1])
    e, a, s = artifact.check(d3_of(concat, [5, 4]), hh)
    ok &= check("concat: parts that do not sum are an error", len(e) == 1 and 'sum to' in e[0], e[:1])
    sl = {'slice': {'tensor': 'big', 'axis': 'r', 'dim': 0, 'offset': 3, 'extent': 4}}
    hh = {'big': {'dtype': 'bf16', 'shape': [10, 4], 'file': 'x'}}
    e, a, s = artifact.check(d3_of(sl, [4, 4]), hh)
    ok &= check("slice: [3, 7) of [10, 4] fits [4, 4]", not e, e[:1])
    sl['slice']['offset'] = 8
    e, a, s = artifact.check(d3_of(sl, [4, 4]), hh)
    ok &= check("slice: [8, 12) of [10, 4] does not fit", len(e) == 1 and 'does not fit' in e[0], e[:1])
    if os.path.isdir(CHECKPOINT):
        real = artifact.read_headers(CHECKPOINT)
        e, a, s = artifact.check(d3, real)
        ok &= check(f"Meta-Llama-3-8B on disk: {s['located']} located, {s['physical']} physical, "
                    f"{s['unnamed']} unnamed, {len(e)} errors", not e and s['unnamed'] == 0, e[:2])
    else:
        print("  skip Meta-Llama-3-8B (not on disk)")
    if os.path.isdir(SHIELDSTRAL):
        with open(SHIELD_DOC, encoding='utf-8') as f:
            sh = json.load(f)
        sh_d3 = derive.products(SHIELD_DOC, catalog_mod.load_for(SHIELD_DOC, sh))['d3']
        e, a, s = artifact.check(sh_d3, artifact.read_headers(SHIELDSTRAL))
        ok &= check(f"Shieldstral-1.0-3B on disk, one file without an index: {s['located']} located, {s['physical']} physical, "
                    f"{s['unnamed']} unnamed, {len(e)} errors", not e and s['unnamed'] == 0 and s['located'] == 458, e[:2])
    else:
        print("  skip Shieldstral-1.0-3B (not on disk)")
    print("artifact: all good" if ok else "artifact: FAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
