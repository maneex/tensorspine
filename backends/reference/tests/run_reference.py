#!/usr/bin/env python3
"""M0 and M1 of the reference-backend plan (§0).

M0 — random weights, no checkpoint:

  1. a tiny llama document (the corpus document with its quantities shrunk by the generic
     edit helper) derives, builds a module, and runs prefill and decode with every produced
     value checked against D2;
  2. the dump holds exactly the values D2 lists at every layer cut, and every state;
  3. the masked (compiled-form) attention equals the sliced one;
  4. optionally, the decode step compiles (`--compile`).

M1, M2 — for each committed fixture whose checkpoint is on disk (else `skip`): the truncated
document loaded by location, every layer output, every state after prefill (KV; and for Qwen 3.5
the convolution history and the recurrent matrix) and the logits within tolerance of the
`transformers` dump, and the same greedy tokens. With `--full`, the whole Llama 3 8B: the eight
greedy tokens `transformers` produced in bf16 on 29 Aug 2026 (about two minutes on CPU, ~16 GB of
page cache).

    python3 backends/reference/tests/run_reference.py [--compile] [--full]
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
from compare import compare, read_dump  # noqa: E402

CHECKPOINTS = os.path.expanduser('~/work/perso/huggingface')
FIXTURES = [   # (fixture, model document, checkpoint directory)
    ('llama3-8b.3layers.hf.safetensors', 'llama3-8b', 'Meta-Llama-3-8B'),
    ('qwen3.5-4b-text.4layers.hf.safetensors', 'qwen3.5-4b-text', 'Qwen3.5-4B'),
]
CHECKPOINT = os.path.join(CHECKPOINTS, 'Meta-Llama-3-8B')
FULL_IDS = [128000, 791, 6864, 315, 9822, 374]                  # "<|begin_of_text|>The capital of France is"
FULL_TOKENS = [12366, 13, 1102, 374, 7559, 304, 279, 10411]      # " Paris. It is located in the north" — transformers 5.14, bf16

TINY = {'quantities.d.source.value': 64, 'quantities.ffn.source.value': 128, 'quantities.heads.source.value': 4,
        'quantities.kv_heads.source.value': 2, 'quantities.head_dim.source.value': 16,
        'quantities.vocab.source.value': 256, 'quantities.layers.source.value': 3,
        'compositions.decoder.indices.layer.stop.literal': 3}


def check(label, ok, detail=''):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f"\n         {detail}" if detail and not ok else ''))
    return ok


def main(compile_step=False, full=False):
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
    expected = {f"value/{p['value']}" for c in g.cuts for p in c['payload']}
    ok &= check("dump keys = the D2 payload of every cut", set(dump) == expected, f"{sorted(set(dump) ^ expected)[:4]}")
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
    ok &= m1(check)
    if full:
        ok &= m1_full(check)
    print("reference: all good" if ok else "reference: FAILED")
    return 0 if ok else 1


def m1(check):
    ok = True
    for fixture, document, checkpoint in FIXTURES:
        ok &= fixture_case(check, os.path.join(REF, 'fixtures', fixture), document, os.path.join(CHECKPOINTS, checkpoint))
    return ok


def fixture_case(check, fixture, document, checkpoint):
    label = document
    if not (os.path.isdir(checkpoint) and os.path.exists(fixture)):
        print(f"  skip {label} (checkpoint or fixture not on disk)")
        return True
    theirs, header = read_dump(fixture)
    ids = header['ids']
    tmp = tempfile.mkdtemp(prefix='tensorspine-ref-fixture-')
    path, _ = graph_mod.truncated(os.path.join(ROOT, 'data', 'models', f'{document}.json'),
                                  f"decoder.layer={header['layers']}", tmp)
    g = graph_mod.load(path)
    errors, _, stats = loader.verify(g, checkpoint)
    ok = check(f"{label}: the {header['layers']}-layer document verifies against the checkpoint", not errors, errors[:1])
    kernels = registry.load_kernels()
    refused = registry.refusals(g, kernels)
    ok &= check(f"{label}: every contract has a kernel for its arguments", not refused, refused[:2])
    params = loader.load_parameters(g, checkpoint, 'cpu')
    model = TensorspineModel(g, Plan(g, kernels), params, torch.float32, 'cpu')
    session = Session(model, capacity=64, device='cpu', dtype=torch.float32)
    ours = {}
    out = session.prefill(ids, ours)
    for ident, st in session.states.items():
        bufs, length = st.read()
        for c, buf in bufs.items():
            ours[f"state/{ident}/{c}"] = buf[:length].detach().to('cpu', torch.float32).clone()
    logits = out[g.generative[0]]
    ours['logits/last'] = logits[-1].detach().to('cpu', torch.float32).clone()
    ours['logits/argmax'] = logits.argmax(-1).detach().cpu().clone()
    nxt = greedy(out, g)
    tokens = [nxt]
    for _ in range(len(header['tokens']) - 1):
        out = session.decode(nxt)
        nxt = greedy(out, g)
        tokens.append(nxt)
    rows, failures, _ = compare(ours, theirs, atol=1e-3, rtol=1e-2)
    worst = max((r[1] for r in rows if r[1] is not None), default=0.0)
    ok &= check(f"{label}: {len(rows)} values, states and logits within tolerance of transformers (max |d| {worst:.1e})",
                failures == 0, [r for r in rows if 'EXCEEDS' in r[3] or r[1] is None][:2])
    ok &= check(f"{label}: greedy tokens {tokens} equal transformers' {header['tokens']}", tokens == header['tokens'])
    return ok


def m1_full(check):
    if not os.path.isdir(CHECKPOINT):
        print("  skip M1 full (checkpoint not on disk)")
        return True
    g = graph_mod.load(os.path.join(ROOT, 'data', 'models', 'llama3-8b.json'))
    kernels = registry.load_kernels()
    params = loader.load_parameters(g, CHECKPOINT, 'cpu')
    model = TensorspineModel(g, Plan(g, kernels), params, torch.float32, 'cpu')
    session = Session(model, capacity=64, device='cpu', dtype=torch.float32)
    t0 = time.time()
    nxt = greedy(session.prefill(FULL_IDS), g)
    tokens = [nxt]
    for _ in range(len(FULL_TOKENS) - 1):
        nxt = greedy(session.decode(nxt), g)
        tokens.append(nxt)
    return check(f"M1 full: 32 layers, greedy tokens {tokens} equal transformers' ({time.time() - t0:.0f}s)", tokens == FULL_TOKENS)


if __name__ == '__main__':
    sys.exit(main(compile_step='--compile' in sys.argv, full='--full' in sys.argv))
