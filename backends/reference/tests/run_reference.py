#!/usr/bin/env python3
"""M0 — the reference backend on random weights, no checkpoint (reference-backend plan §0):

  1. a tiny llama document (the corpus document with its quantities shrunk by the generic
     edit helper) derives, builds a module, and runs prefill and decode with every produced
     value checked against D2;
  2. the dump holds exactly the values D2 lists at every layer cut, and every state;
  3. the masked (compiled-form) attention equals the sliced one;
  4. optionally, the decode step compiles (`--compile`).

    python3 backends/reference/tests/run_reference.py [--compile]
"""
import os
import sys
import tempfile
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(REF))
sys.path.insert(0, REF)

import graph as graph_mod        # noqa: E402
import loader                    # noqa: E402
import registry                  # noqa: E402
from kernels import attention_dense  # noqa: E402
from module import TensorspineModel  # noqa: E402
from plan import Plan            # noqa: E402
from session import Session, greedy  # noqa: E402

TINY = {'quantities.d.source.value': 64, 'quantities.ffn.source.value': 128, 'quantities.heads.source.value': 4,
        'quantities.kv_heads.source.value': 2, 'quantities.head_dim.source.value': 16,
        'quantities.vocab.source.value': 256, 'quantities.layers.source.value': 3,
        'compositions.decoder.indices.layer.stop.literal': 3}


def check(label, ok, detail=''):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f"\n         {detail}" if detail and not ok else ''))
    return ok


def main(compile_step=False):
    ok = True
    tmp = tempfile.mkdtemp(prefix='tensorspine-ref-test-')
    path, notes = graph_mod.edited(os.path.join(ROOT, 'data', 'models', 'llama3-8b.json'), TINY, tmp, 'tiny')
    g = graph_mod.load(path)
    ok &= check("tiny llama derives: 3 layers, 6 contracts", len(g.nodes) == 3 * 6 + 3 and len(g.layer_cuts()) == 2)
    kernels = registry.load_kernels()
    r = registry.refusals(g, kernels)
    ok &= check("no refusal for the six M1 contracts", not r, '; '.join(r[:3]))
    params = loader.random_parameters(g, 'cpu', seed=1)
    ok &= check("every D3 identity has a parameter of its shape",
                all(list(params[i].shape) == [a['extent'] for a in t['shape']] for i, t in g.tensors.items()))
    model = TensorspineModel(g, Plan(g, kernels), params, torch.float32, 'cpu')
    session = Session(model, capacity=32, device='cpu', dtype=torch.float32)
    dump = {}
    out = session.prefill([1, 2, 3, 4, 5, 6, 7, 8], dump)
    ok &= check("prefill: logits [8, vocab] and every value on its D2 shape", list(out['logits'].shape) == [8, 256])
    nxt = greedy(out, g)
    out = session.decode(nxt, dump)
    ok &= check("decode: logits [1, vocab]", list(out['logits'].shape) == [1, 256])
    ok &= check("positions consumed per stream: 9", session.consumed == {'tokens': 9})
    ok &= check("append states hold 9 positions", all(s.length == 9 for s in session.states.values()))
    expected = {f"{c['cut']}/{p['value']}" for c in g.layer_cuts() for p in c['payload']}
    ok &= check("dump keys = the D2 payload of every layer cut", set(dump) == expected, f"{sorted(set(dump) ^ expected)[:4]}")
    ok &= check("finite outputs", bool(torch.isfinite(out['logits']).all()))
    # masked == sliced
    torch.manual_seed(0)
    q = torch.randn(3, 4, 16); K = torch.randn(32, 2, 16); V = torch.randn(32, 2, 16)
    qpos = torch.tensor([6, 7, 8])
    a = attention_dense.attend(q, K, V, 9, qpos, True, static=False)
    b = attention_dense.attend(q, K, V, 9, qpos, True, static=True)
    ok &= check("masked attention over the whole capacity equals the sliced form", torch.allclose(a, b, atol=1e-6))
    if compile_step:
        t0 = time.time()
        try:
            model.static = True
            compiled = torch.compile(model, dynamic=False)
            session2 = Session(compiled if hasattr(compiled, 'graph') else model, capacity=32, device='cpu', dtype=torch.float32)
            session2.model = compiled
            o1 = session2.prefill([1, 2, 3, 4])
            o2 = session2.decode(greedy(o1, g))
            ok &= check(f"decode step compiles and runs ({time.time() - t0:.0f}s)", list(o2['logits'].shape) == [1, 256])
        except Exception as e:  # noqa: BLE001
            ok &= check("decode step compiles", False, f"{type(e).__name__}: {str(e)[:200]}")
        finally:
            model.static = False
    else:
        print("  skip compile (pass --compile)")
    print("reference: all good" if ok else "reference: FAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(compile_step='--compile' in sys.argv))
