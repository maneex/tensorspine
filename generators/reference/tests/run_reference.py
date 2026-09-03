#!/usr/bin/env python3
"""M0 and M1 of the reference-generator plan (§0).

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
`transformers` dump, and the same greedy tokens. A fixture's `in/` tensors — the non-token inputs
the delivery received (Whisper's audio frames) — are delivered with the prompt in the one prefill,
the caches on their streams sized by what they deliver; on a document whose token stream joins
the recorded input's stream (Voxtral Realtime) the prompt takes its tokens' frames and every step
a token's, and the prefill is replayed as one fragment per token to measure §5.3's invariance on
the checkpoint. With `--full`, the whole models: the eight greedy tokens `transformers` produced
(minutes on CPU; ~16 GB of page cache for Llama 3 8B, ~8 GB for Qwen 3.5 4B), a `FULL` entry
naming a fixture taking its audio from it, or a sample the artifact's processor turns into a
streaming prefill and fragments, transcribed to its end.

    python3 generators/reference/tests/run_reference.py [--compile] [--full]
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
from compare import compare, read_fixture, tolerance_for  # noqa: E402

# `weights/` under the one runtime directory (`generators/zml/README.md` describes the
# layout; both generators read the same one). No default inside the tree, and no home
# path written down here: unset, every fixture check says `skip`.
ARTIFACTS = os.environ.get('TENSORSPINE_MODEL_ARTIFACTS', '')
CHECKPOINTS = os.path.join(ARTIFACTS, 'weights')
from verified import FIXTURES, FULL   # noqa: E402

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
    out0 = out['logits'].clone()
    ok &= check("prefill: logits [8, vocab] and every value on its D2 shape", list(out['logits'].shape) == [8, 256])
    nxt = greedy(out, g)
    out = session.decode(nxt, dump)
    ok &= check("decode: logits [1, vocab]", list(out['logits'].shape) == [1, 256])
    ok &= check("positions consumed per stream: 9", session.consumed == {'tokens': 9})
    ok &= check("append states hold 9 positions", all(s.length == 9 for s in session.states.values()))
    expected = {f"value/{p['value']}" for c in g.cuts for p in c['payload']}
    ok &= check("dump keys = the D2 payload of every cut", set(dump) == expected, f"{sorted(set(dump) ^ expected)[:4]}")
    ok &= check("finite outputs", bool(torch.isfinite(out['logits']).all()))
    # blocks under a bound: the same outputs, bit for bit, from a partitioned run (M4)
    resident = loader.state_bytes(g, 32, torch.float32) + loader.largest_temporary(g, torch.float32)
    total = g.d3_totals['bytes']
    blocked = Plan(g, kernels, max_bytes=resident + total // 2, elements=32, resident_bytes=resident)
    bmodel = TensorspineModel(g, blocked, None, torch.float32, 'cpu', source=loader.RandomSource(params).materialise)
    bsession = Session(bmodel, capacity=32, device='cpu', dtype=torch.float32)
    bout = bsession.prefill([1, 2, 3, 4, 5, 6, 7, 8])
    bnxt = greedy(bout, g)
    bout2 = bsession.decode(bnxt)
    lines = blocked.summary(32, resident + total // 2, resident)
    ok &= check("blocks: the cut summary has one line per block, opening and closing at D6's cuts",
                len(lines) == len(blocked.blocks) + 2 and all('→' in l for l in lines[1:-1]) and 'start →' in lines[1] and '→ end' in lines[-2])
    ok &= check(f"blocks: {len(blocked.blocks)} blocks at legal cuts give the one-block logits bit for bit",
                len(blocked.blocks) > 1 and bnxt == nxt and torch.equal(bout2['logits'], out['logits'])
                and bmodel.loaded_blocks == 2 * len(blocked.blocks))
    try:
        Plan(g, kernels, max_bytes=resident + 1, elements=32, resident_bytes=resident)
        ok &= check("blocks: a bound below one layer is refused", False)
    except ValueError as e:
        ok &= check("blocks: a bound below one layer is refused", 'exceeds --max-ram' in str(e))
    # the opaque channel (generators/CAPABILITIES.md): parameters reach the primitive beside its arguments
    from module import physical_for
    phys = {'attention.dense@1.0.0': {'backend': 'cpu', 'kernel': 'vanilla'}, 'decoder/attn[layer=*]': {'kernel': 'paged'},
            'decoder/attn[layer=2]': {'block_size': 16}}
    ok &= check("physical parameters resolve contract < pattern < exact, and other occurrences get none",
                physical_for(phys, 'decoder/attn[layer=2]', {'name': 'attention.dense', 'version': '1.0.0'}) == {'backend': 'cpu', 'kernel': 'paged', 'block_size': 16}
                and physical_for(phys, 'decoder/attn[layer=0]', {'name': 'attention.dense', 'version': '1.0.0'}) == {'backend': 'cpu', 'kernel': 'paged'}
                and physical_for(phys, 'decoder/ffn[layer=0]', {'name': 'ffn.gated', 'version': '1.0.0'}) is None)
    pmodel = TensorspineModel(g, Plan(g, kernels), params, torch.float32, 'cpu', physical=phys)
    pout = Session(pmodel, capacity=32, device='cpu', dtype=torch.float32).prefill([1, 2, 3, 4, 5, 6, 7, 8])
    ok &= check("a primitive ignores opaque keys it does not read: same logits", torch.equal(pout['logits'], out0))
    # masked == sliced
    torch.manual_seed(0)
    q = torch.randn(3, 4, 16); K = torch.randn(32, 2, 16); V = torch.randn(32, 2, 16)
    qpos = torch.tensor([6, 7, 8])
    a = attention_dense.attend(q, K, V, 9, qpos, True, static=False)
    b = attention_dense.attend(q, K, V, 9, qpos, True, static=True)
    ok &= check("masked attention over the whole capacity equals the sliced form", torch.allclose(a, b, atol=1e-6))
    # YaRN (finding 10): the kernel's frequencies are transformers' for Shieldstral's record, whose
    # attention factor the document states as 1; without mscale transformers gives the paper's value
    from transformers import Ministral3Config
    from transformers.modeling_rope_utils import _compute_yarn_parameters
    yarn = {'kind': 'yarn', 'factor': 16, 'beta_fast': 32, 'beta_slow': 1, 'attention_factor': 1.0, 'orig_ctx': 16384}
    rp = {'rope_type': 'yarn', 'rope_theta': 1e6, 'factor': 16.0, 'beta_fast': 32.0, 'beta_slow': 1.0, 'original_max_position_embeddings': 16384}
    cfg = Ministral3Config(hidden_size=3072, num_attention_heads=32, head_dim=128, max_position_embeddings=262144,
                           rope_parameters=dict(rp, mscale=1.0, mscale_all_dim=1.0))
    theirs, factor = _compute_yarn_parameters(cfg, 'cpu')
    ours = attention_dense.inv_freq(128, 1e6, yarn, 'cpu')
    ok &= check("YaRN: the kernel's 64 inverse frequencies equal transformers' for Shieldstral's record, whose attention factor is 1",
                torch.allclose(ours, theirs, atol=0, rtol=1e-6) and factor == 1.0)
    _, paper = _compute_yarn_parameters(Ministral3Config(hidden_size=3072, num_attention_heads=32, head_dim=128,
                                                         max_position_embeddings=262144, rope_parameters=dict(rp)), 'cpu')
    ok &= check("YaRN: without mscale transformers' factor is the paper's 0.1·ln 16 + 1, the value deepseek-v4-pro states",
                paper == 1.2772588722239782)
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
    # the committed manifest is what the code generates (generators/CAPABILITIES.md)
    sys.path.insert(0, REF)
    import ref as ref_cli
    import json
    fresh = ref_cli.manifest()
    with open(os.path.join(REF, 'capabilities.json'), encoding='utf-8') as f:
        committed = json.load(f)
    for m in (fresh, committed):
        m['generator'] = {k: v for k, v in m['generator'].items() if k not in ('version', 'generated')}
    ok &= check("the committed capabilities manifest is what the code generates", fresh == committed)
    ok &= witness_case(check)
    ok &= moe_random_case(check, tmp)
    ok &= whisper_random_case(check, tmp)
    ok &= voxtral_random_case(check, tmp)
    ok &= sharing_case(check, tmp)
    ok &= m1(check)
    ok &= composite_case(check)
    ok &= by_source_case(check)
    if full:
        ok &= m1_full(check)
    print("reference: all good" if ok else "reference: FAILED")
    return 0 if ok else 1


TINY_MOE = {'quantities.d.source.value': 64, 'quantities.attn_q.source.value': 4, 'quantities.attn_kv.source.value': 2,
            'quantities.attn_hd.source.value': 16, 'quantities.gdn_k.source.value': 2, 'quantities.gdn_v.source.value': 4,
            'quantities.gdn_hd.source.value': 16, 'quantities.experts.source.value': 8, 'quantities.top_k.source.value': 2,
            'quantities.moe_ffn.source.value': 32, 'quantities.shared_ffn.source.value': 32, 'quantities.vocab.source.value': 256,
            'quantities.d_vis.source.value': 64, 'quantities.vit_heads.source.value': 4, 'quantities.vit_hd.source.value': 16,
            'quantities.vit_ffn.source.value': 128, 'quantities.vit_layers.source.value': 2,
            'compositions.vision.indices.layer.stop.literal': 2}      # the tower is not evaluated on text, but it is partitioned


def witness_case(check):
    """The witness did not change silently (docs/TENSORSPINE-FIXTURE.md §5): every committed unit
    fixture regenerates from its seed and its run repeats within its own tolerance at every dtype
    the kernel declares one for, with the parameters loaded from the fixture as a conformer loads
    them; and every case a kernel declares is recorded."""
    import witness
    kernels = registry.load_kernels()
    ok = True
    ids = witness.committed()
    for fid in ids:
        good, lines = witness.verify(fid, kernels)
        ok &= check(f"witness {fid}: regenerated and repeated within tolerance", good, '\n         '.join(lines))
    declared = {f"{n}@{v}/{c['case']}" for n, v, _k, c in witness.cases(kernels)}
    ok &= check(f"witness: the {len(declared)} cases the kernels declare are the {len(ids)} fixtures committed",
                declared == set(ids), str(sorted(declared ^ set(ids))[:3]))
    return ok


def sharing_case(check, tmp):
    """The sharing granularities as §4.3 defines them, exercised by `Session.fork` on the tiny
    hybrid (an `append` KV cache, `window` convolution rings, `fixed` recurrent matrices) and the
    tiny Llama (`append` alone), on random weights:

      1. a child forked at the end of the prefill and continued gives, bit for bit, what a fresh
         session gives on the concatenation; the parent continued otherwise likewise;
      2. a fork behind the current position is served for `append` (`by_position`: the prefix is
         copied entry by entry) and refused for `window` and `fixed`, naming the granularity;
      3. two sessions with different prefixes and the same last three tokens hold equal conv
         rings at layer 0 and different ones at layer 1: a ring's content depends on the whole
         prefix past the first layer, so a runtime sharing `within_span` proves the prefix, not
         the span."""
    ok = True
    kernels = registry.load_kernels()
    path, _ = graph_mod.truncated(os.path.join(ROOT, 'data', 'models', 'qwen3.5-35b-a3b.json'), 'decoder.layer=4', tmp)
    path, _ = graph_mod.edited(path, TINY_MOE, tmp, 'tiny-fork')
    g = graph_mod.load(path)
    laws = {s['law'] for s in g.states.values()}
    ok &= check("sharing: the tiny hybrid carries all three laws", laws == {'append', 'window', 'fixed'}, str(laws))
    params = loader.random_parameters(g, 'cpu', seed=5)

    def session():
        return Session(TensorspineModel(g, Plan(g, kernels), params, torch.float32, 'cpu'), 64, 'cpu', torch.float32)

    prefix, b, c = [1, 2, 3, 4, 5, 6], [7, 8], [9]
    fresh = session()
    fresh.prefill(prefix)
    wanted_b = [fresh.decode(t)[g.generative[0]].clone() for t in b]
    fresh = session()
    fresh.prefill(prefix)
    wanted_c = fresh.decode(c[0])[g.generative[0]].clone()
    parent = session()
    parent.prefill(prefix)
    child = parent.fork()
    got_b = [child.decode(t)[g.generative[0]].clone() for t in b]
    got_c = parent.decode(c[0])[g.generative[0]].clone()
    ok &= check("sharing: a child forked at the end of the prefill continues as a fresh session on the concatenation, bit for bit",
                all(torch.equal(x, y) for x, y in zip(got_b, wanted_b)) and child.consumed['tokens'] == len(prefix) + len(b))
    ok &= check("sharing: the parent, continued otherwise after the fork, is unaffected by the child",
                torch.equal(got_c, wanted_c) and parent.consumed['tokens'] == len(prefix) + 1)
    try:
        parent.fork(at=3)
        ok &= check("sharing: a fork behind the current position is refused where a ring or a matrix would have to serve it", False)
    except Exception as e:  # noqa: BLE001
        ok &= check("sharing: a fork behind the current position is refused where a ring or a matrix would have to serve it",
                    'within_span' in str(e) or 'at_fork_point' in str(e), str(e)[:160])
    # append alone (the tiny Llama): a fork behind the current position is served by_position
    lpath, _ = graph_mod.edited(os.path.join(ROOT, 'data', 'models', 'llama3-8b.json'), TINY, tmp, 'tiny-fork')
    lg = graph_mod.load(lpath)
    lparams = loader.random_parameters(lg, 'cpu', seed=6)
    lfresh = Session(TensorspineModel(lg, Plan(lg, kernels), lparams, torch.float32, 'cpu'), 32, 'cpu', torch.float32)
    lfresh.prefill(prefix[:3])
    lwant = lfresh.decode(prefix[3])[lg.generative[0]].clone()
    lparent = Session(TensorspineModel(lg, Plan(lg, kernels), lparams, torch.float32, 'cpu'), 32, 'cpu', torch.float32)
    lparent.prefill(prefix)
    lchild = lparent.fork(at=3)
    lgot = lchild.decode(prefix[3])[lg.generative[0]].clone()
    # within f32 rounding, not bit for bit: the parent's prefill of six rows and the fresh session's
    # of three round each row's projections differently, and the copied entries carry that
    ok &= check("sharing: by_position — a child forked behind the parent's position reads the copied prefix and continues as a fresh session would, within f32 rounding",
                torch.allclose(lgot, lwant, atol=1e-5, rtol=1e-5)
                and all(s.length == 4 for s in lchild.states.values()) and all(s.length == 6 for s in lparent.states.values()),
                f"max |d| {float((lgot - lwant).abs().max()):.1e}, child lengths {sorted({s.length for s in lchild.states.values()})}, "
                f"parent lengths {sorted({s.length for s in lparent.states.values()})}")
    # the ring's content depends on the prefix past the first layer
    x, y = session(), session()
    x.prefill([1, 2, 3, 7, 8, 9])
    y.prefill([4, 5, 6, 7, 8, 9])
    ring0 = 'decoder.gdn.conv[layer=0]'
    ring1 = 'decoder.gdn.conv[layer=1]'
    r0x, _ = x.states[ring0].read(); r0y, _ = y.states[ring0].read()
    r1x, _ = x.states[ring1].read(); r1y, _ = y.states[ring1].read()
    ok &= check("sharing: within_span — the same last three tokens give equal rings at layer 0 and different rings at layer 1: a runtime proves the prefix, not the span",
                torch.equal(r0x['w'], r0y['w']) and not torch.equal(r1x['w'], r1y['w']))
    return ok


def moe_random_case(check, tmp):
    """The Qwen 3.5 MoE document on random weights, four layers (three gated-delta, one attention),
    eight experts of which two per token, a gated shared expert: the routing, the fused experts and
    blocks give the one-block logits bit for bit."""
    ok = True
    path, _ = graph_mod.truncated(os.path.join(ROOT, 'data', 'models', 'qwen3.5-35b-a3b.json'), 'decoder.layer=4', tmp)
    path, _ = graph_mod.edited(path, TINY_MOE, tmp, 'tiny')
    g = graph_mod.load(path)
    kernels = registry.load_kernels()
    active = Plan(g, kernels).evaluable({g.feedback_input})
    refused = registry.refusals(g, kernels, active)
    ok &= check("tiny qwen3.5-moe: every contract the text delivery evaluates has a kernel", not refused, refused[:2])
    params = loader.random_parameters(g, 'cpu', seed=3)
    model = TensorspineModel(g, Plan(g, kernels), params, torch.float32, 'cpu')
    session = Session(model, capacity=64, device='cpu', dtype=torch.float32)
    out = session.prefill([1, 2, 3, 4, 5, 6])
    logits = out[g.generative[0]]
    tokens = [greedy(out, g)]
    for _ in range(3):
        tokens.append(greedy(session.decode(tokens[-1]), g))
    ok &= check("tiny qwen3.5-moe: prefill and three decodes give finite logits on their D2 shapes",
                list(logits.shape) == [6, 256] and bool(torch.isfinite(logits).all()))
    resident = loader.state_bytes(g, 64, torch.float32) + loader.largest_temporary(g, torch.float32)
    finest = max(b.bytes + b.payload_bytes_per_element * 64 for b in Plan(g, kernels).minimal)
    blocked = Plan(g, kernels, max_bytes=resident + finest, elements=64, resident_bytes=resident)
    bmodel = TensorspineModel(g, blocked, None, torch.float32, 'cpu', source=loader.RandomSource(params).materialise)
    bsession = Session(bmodel, capacity=64, device='cpu', dtype=torch.float32)
    bl = bsession.prefill([1, 2, 3, 4, 5, 6])[g.generative[0]]
    bt = [greedy({g.generative[0]: bl}, g)]
    for _ in range(3):
        bt.append(greedy(bsession.decode(bt[-1]), g))
    ok &= check(f"tiny qwen3.5-moe: {len(blocked.blocks)} blocks give the same logits and tokens",
                len(blocked.blocks) > 1 and torch.equal(bl, logits) and bt == tokens)
    return ok


TINY_WHISPER = {'quantities.d.source.value': 64, 'quantities.heads.source.value': 4, 'quantities.hd.source.value': 16,
                'quantities.ffn.source.value': 128, 'quantities.vocab.source.value': 256, 'quantities.mels.source.value': 8,
                'quantities.enc_layers.source.value': 2, 'compositions.encoder.indices.layer.stop.literal': 2,
                'quantities.dec_layers.source.value': 2, 'compositions.decoder.indices.layer.stop.literal': 2,
                'occurrences.conv_frontend.arguments.position.literal': 16, 'occurrences.embed.arguments.positions.literal': 16}


def whisper_random_case(check, tmp):
    """The encoder–decoder on random weights, two layers a side, shrunk: a merged domain (24 frames
    make 12 positions behind the strided front end — §5.3's rule in the runtime), a capacity per
    stream (the cross caches hold the source's positions, the self caches the tokens'), the audio
    delivered with the prompt in one prefill, decode steps that deliver nothing on the source and
    append nothing to its caches, an unaligned delivery and a missing input refused, and blocks
    giving the one-block logits bit for bit."""
    ok = True
    path, _ = graph_mod.edited(os.path.join(ROOT, 'data', 'models', 'whisper-large-v3.json'), TINY_WHISPER, tmp, 'tiny')
    g = graph_mod.load(path)
    kernels = registry.load_kernels()
    plan = Plan(g, kernels)
    active = plan.evaluable(g.required_inputs())
    refused = registry.refusals(g, kernels, active)
    ok &= check("tiny whisper: the audio and the prompt evaluate every occurrence, each with a kernel for its arguments",
                not refused and len(active) == len(g.nodes), refused[:2])
    ok &= check("tiny whisper: the prompt alone evaluates five occurrences before any audio is cached (§7)",
                len(plan.evaluable({'tokens'})) == 5, str(sorted(plan.evaluable({'tokens'}))))
    params = loader.random_parameters(g, 'cpu', seed=7)
    model = TensorspineModel(g, plan, params, torch.float32, 'cpu')
    capacity = {'tokens': 16, 'audio': 12}
    session = Session(model, capacity=capacity, device='cpu', dtype=torch.float32)
    audio = torch.randn(24, 8, generator=torch.Generator().manual_seed(7))
    ids = [1, 2, 3, 4]
    dump = {}
    out = session.prefill(ids, dump, inputs={'audio': audio})
    logits = out[g.generative[0]].clone()
    ok &= check("tiny whisper: 24 frames and 4 tokens in one prefill give logits [4, vocab]; consumed per stream in input elements",
                list(logits.shape) == [4, 256] and session.consumed == {'audio': 24, 'tokens': 4}, str(session.consumed))
    cross = [st for ident, st in session.states.items() if 'cross_attn' in ident]
    selfs = [st for ident, st in session.states.items() if 'self_attn' in ident]
    ok &= check("tiny whisper: the cross caches hold the 12 merged source positions, the self caches the 4 tokens",
                len(cross) == 2 and len(selfs) == 2 and all(st.length == 12 for st in cross) and all(st.length == 4 for st in selfs))
    ok &= check("tiny whisper: the encoder runs on 12 positions and its output crosses every decoder cut, dumped once as [12, 64]",
                list(dump['value/enc_final_n.output'].shape) == [12, 64] and list(dump['value/encoder/ffn_r[layer=1].output'].shape) == [12, 64])
    tokens = [greedy(out, g)]
    for _ in range(2):
        tokens.append(greedy(session.decode(tokens[-1]), g))
    ok &= check("tiny whisper: two decode steps deliver nothing on the source: the cross caches keep 12, the self caches reach 6",
                all(st.length == 12 for st in cross) and all(st.length == 6 for st in selfs))
    try:
        Session(model, capacity=capacity, device='cpu', dtype=torch.float32).prefill(ids, inputs={'audio': audio[:23]})
        ok &= check("tiny whisper: 23 frames are refused as unaligned to the stride (§5.3)", False)
    except Exception as e:  # noqa: BLE001
        ok &= check("tiny whisper: 23 frames are refused as unaligned to the stride (§5.3)", 'aligned' in str(e), str(e)[:160])
    try:
        Session(model, capacity=capacity, device='cpu', dtype=torch.float32).prefill(ids)
        ok &= check("tiny whisper: a prefill without the audio is refused, naming the output that needs it (§7)", False)
    except Exception as e:  # noqa: BLE001
        ok &= check("tiny whisper: a prefill without the audio is refused, naming the output that needs it (§7)",
                    'audio' in str(e) and 'main' in str(e), str(e)[:160])
    try:
        Session(model, capacity={'tokens': 16}, device='cpu', dtype=torch.float32)
        ok &= check("tiny whisper: a capacity mapping that omits a stream is refused", False)
    except Exception as e:  # noqa: BLE001
        ok &= check("tiny whisper: a capacity mapping that omits a stream is refused", "'audio'" in str(e), str(e)[:160])
    resident = loader.state_bytes(g, capacity, torch.float32) + loader.largest_temporary(g, torch.float32)
    finest = max(b.bytes + b.payload_bytes_per_element * 24 for b in plan.minimal)
    blocked = Plan(g, kernels, max_bytes=resident + finest, elements=24, resident_bytes=resident)
    bmodel = TensorspineModel(g, blocked, None, torch.float32, 'cpu', source=loader.RandomSource(params).materialise)
    bsession = Session(bmodel, capacity=capacity, device='cpu', dtype=torch.float32)
    bl = bsession.prefill(ids, inputs={'audio': audio})[g.generative[0]]
    bt = [greedy({g.generative[0]: bl}, g)]
    for _ in range(2):
        bt.append(greedy(bsession.decode(bt[-1]), g))
    ok &= check(f"tiny whisper: {len(blocked.blocks)} blocks at legal cuts of both compositions give the same logits and tokens",
                len(blocked.blocks) > 1 and torch.equal(bl, logits) and bt == tokens)
    return ok


TINY_VOXTRAL = {'quantities.d.source.value': 64, 'quantities.enc_d.source.value': 32, 'quantities.enc_heads.source.value': 4,
                'quantities.enc_hd.source.value': 8, 'quantities.heads.source.value': 4, 'quantities.kv_heads.source.value': 2,
                'quantities.hd.source.value': 16, 'quantities.ffn.source.value': 128, 'quantities.enc_ffn.source.value': 64,
                'quantities.vocab.source.value': 256, 'quantities.mels.source.value': 8,
                'quantities.enc_layers.source.value': 2, 'compositions.encoder.indices.layer.stop.literal': 2,
                'quantities.dec_layers.source.value': 2, 'compositions.decoder.indices.layer.stop.literal': 2}


def recorded_states(session):
    """Every state as a dump records it: an append state's positions, a window's valid tail, a
    fixed state's payload — what the delivery implementation's caches hold."""
    out = {}
    for ident, st in session.states.items():
        bufs, length = st.tail() if st.law == 'window' else st.read()
        for c, buf in bufs.items():
            out[f"state/{ident}/{c}"] = (buf[:length] if length is not None else buf).detach().to('cpu', torch.float32).clone()
    return out


def voxtral_random_case(check, tmp):
    """The streaming document on random weights, two layers a side, shrunk: a token stream that
    joins the fragmented audio stream (§2.3, one token per eight frames), a prefill of three tokens
    with twenty-four frames and the delay, the same prefill as three fragments of eight frames with
    one token each giving the same logits and states (§5.3's invariance, measured), the rings'
    lengths after the prefill and after two decode steps that each deliver a token and a fragment,
    deliveries refused when the inputs disagree on the stream's advance or a fragment is unaligned,
    a decode without its fragment refused, a fork after the prefill continuing as a fresh session
    would, and blocks giving the one-block logits bit for bit."""
    ok = True
    path, _ = graph_mod.edited(os.path.join(ROOT, 'data', 'models', 'voxtral-realtime.json'), TINY_VOXTRAL, tmp, 'tiny')
    g = graph_mod.load(path)
    kernels = registry.load_kernels()
    plan = Plan(g, kernels)
    active = plan.evaluable(g.required_inputs())
    refused = registry.refusals(g, kernels, active)
    ok &= check("tiny voxtral: the tokens, the frames and the delay evaluate every occurrence, each with a kernel for its arguments",
                not refused and len(active) == len(g.nodes), refused[:2])
    ok &= check("tiny voxtral: the token input joins the audio stream — the feedback input is `tokens`, eight frames per token (§5.3)",
                g.feedback_input == 'tokens' and g.input_stream['tokens'] == 'audio' and g.elements_per['tokens'] == 8 and g.elements_per['audio'] == 1
                and g.required_inputs() == {'tokens', 'audio', 'delay'}, str((g.feedback_input, g.elements_per)))
    params = loader.random_parameters(g, 'cpu', seed=9)
    model = TensorspineModel(g, plan, params, torch.float32, 'cpu')
    capacity = {'audio': 64, 'delay': 1}
    gen = torch.Generator().manual_seed(9)
    audio = torch.randn(40, 8, generator=gen)
    delay = torch.tensor([6], dtype=torch.int32)
    ids = [1, 2, 3]

    def session():
        return Session(model, capacity=capacity, device='cpu', dtype=torch.float32)

    whole = session()
    dump = {}
    out = whole.prefill(ids, dump, inputs={'audio': audio[:24], 'delay': delay})
    logits = out[g.generative[0]].clone()
    ok &= check("tiny voxtral: 3 tokens, 24 frames and the delay in one prefill give logits [3, vocab]; the stream advanced by 24 frames",
                list(logits.shape) == [3, 256] and whole.consumed == {'audio': 24, 'delay': 1}, str(whole.consumed))
    enc = [st for ident, st in whole.states.items() if ident.startswith('encoder.attn.kv')]
    dec = [st for ident, st in whole.states.items() if ident.startswith('decoder.attn.kv')]
    conds = [st for ident, st in whole.states.items() if 'condition_cache' in ident]
    h1, h2 = whole.states['conv_frontend.conv1_history'], whole.states['conv_frontend.conv2_history']
    ok &= check("tiny voxtral: after the prefill the encoder rings hold 12 positions, the decoder rings 3, the histories 2 and 1 frames, the condition caches 1",
                len(enc) == 2 and all(st.tail()[1] == 12 for st in enc) and len(dec) == 2 and all(st.tail()[1] == 3 for st in dec)
                and h1.tail()[1] == 2 and h2.tail()[1] == 1 and len(conds) == 2 and all(st.length == 1 for st in conds))
    ok &= check("tiny voxtral: the values crossing the cuts are dumped once — the encoder's output on 12 positions, the projector's and the fused embedding on 3 tokens, the time embedding on 1",
                list(dump['value/enc_final_n.output'].shape) == [12, 32] and list(dump['value/audio_projector.output'].shape) == [3, 64]
                and list(dump['value/fuse.output'].shape) == [3, 64] and list(dump['value/time_embed.embedding'].shape) == [1, 64], str(sorted(dump)))
    # §5.3's invariance: the same prefill delivered as three fragments of eight frames with one token each
    parts = session()
    got = []
    for i, tok in enumerate(ids):
        extra = {'audio': audio[8 * i:8 * i + 8]}
        if i == 0:
            extra['delay'] = delay
        got.append(parts.run({'tokens': torch.tensor([tok]), **extra})[g.generative[0]])
    got = torch.cat(got)
    d_logits = float((got - logits).abs().max())
    d_states = max(float((a - b).abs().max()) for k, a in recorded_states(whole).items() for b in [recorded_states(parts)[k]])
    ok &= check(f"tiny voxtral: the prefill delivered as three fragments of eight frames gives the same logits and states — max |d| {d_logits:.1e} on the logits, {d_states:.1e} on the states (f32 rounding across row counts)",
                torch.allclose(got, logits, atol=1e-5, rtol=1e-5) and d_states < 1e-5 and parts.consumed == whole.consumed)
    nxt = greedy(out, g)
    tokens = [nxt]
    for i in range(2):
        out = whole.decode(nxt, inputs={'audio': audio[24 + 8 * i:32 + 8 * i]})
        nxt = greedy(out, g)
        tokens.append(nxt)
    ok &= check("tiny voxtral: two decode steps each deliver a token and eight frames — the encoder rings reach 20, the decoder rings 5, the stream 40 frames",
                all(st.tail()[1] == 20 for st in enc) and all(st.tail()[1] == 5 for st in dec) and whole.consumed == {'audio': 40, 'delay': 1}
                and all(st.length == 1 for st in conds))
    for label, delivered, word in (("2 tokens with 24 frames are refused: the inputs disagree on the stream's advance", {'tokens': [1, 2], 'audio': audio[:24]}, 'disagree'),
                                   ("1 token with 7 frames is refused: eight frames make a token", {'tokens': [1], 'audio': audio[:7]}, 'disagree')):
        try:
            ins = {k: (torch.tensor(v) if isinstance(v, list) else v) for k, v in delivered.items()}
            session().run(dict(ins, delay=delay))
            ok &= check(f"tiny voxtral: {label}", False)
        except Exception as e:  # noqa: BLE001
            ok &= check(f"tiny voxtral: {label}", word in str(e), str(e)[:200])
    try:
        s = session()
        s.prefill(ids, inputs={'audio': audio[:24], 'delay': delay})
        s.run({'audio': audio[24:30]})
        ok &= check("tiny voxtral: 6 frames alone, after the prefill, are refused as unaligned to the projector's stack (§5.3)", False)
    except Exception as e:  # noqa: BLE001
        ok &= check("tiny voxtral: 6 frames alone, after the prefill, are refused as unaligned to the projector's stack (§5.3)", 'aligned' in str(e), str(e)[:200])
    try:
        s = session()
        s.prefill(ids, inputs={'audio': audio[:24], 'delay': delay})
        s.decode(nxt)
        ok &= check("tiny voxtral: a decode without its fragment is refused: the generative output is not evaluated (§7)", False)
    except Exception as e:  # noqa: BLE001
        ok &= check("tiny voxtral: a decode without its fragment is refused: the generative output is not evaluated (§7)",
                    'not evaluated' in str(e) and 'main' in str(e), str(e)[:200])
    try:
        session().prefill(ids, inputs={'audio': audio[:24]})
        ok &= check("tiny voxtral: a first delivery without the delay is refused, naming the output that needs it (§7)", False)
    except Exception as e:  # noqa: BLE001
        ok &= check("tiny voxtral: a first delivery without the delay is refused, naming the output that needs it (§7)",
                    'delay' in str(e) and 'main' in str(e), str(e)[:200])
    # a fork after the prefill: the rings copied whole at the current position, the condition caches by_source
    parent = session()
    parent.prefill(ids, inputs={'audio': audio[:24], 'delay': delay})
    child = parent.fork()
    fresh = session()
    fresh.prefill(ids, inputs={'audio': audio[:24], 'delay': delay})
    want = fresh.decode(tokens[0], inputs={'audio': audio[24:32]})[g.generative[0]]
    got = child.decode(tokens[0], inputs={'audio': audio[24:32]})[g.generative[0]]
    ok &= check("tiny voxtral: a child forked after the prefill continues as a fresh session on the same frames and tokens would, bit for bit; the parent is unaffected",
                torch.equal(got, want) and child.consumed == {'audio': 32, 'delay': 1} and parent.consumed == {'audio': 24, 'delay': 1}
                and all(st.tail()[1] == 12 for i, st in parent.states.items() if i.startswith('encoder.attn.kv')))
    try:
        parent.fork(at=16)
        ok &= check("tiny voxtral: a fork behind the current position is refused by the rings, naming within_span", False)
    except Exception as e:  # noqa: BLE001
        ok &= check("tiny voxtral: a fork behind the current position is refused by the rings, naming within_span", 'within_span' in str(e), str(e)[:160])
    resident = loader.state_bytes(g, capacity, torch.float32) + loader.largest_temporary(g, torch.float32)
    finest = max(b.bytes + b.payload_bytes_per_element * 24 for b in plan.minimal)
    blocked = Plan(g, kernels, max_bytes=resident + finest, elements=24, resident_bytes=resident)
    bmodel = TensorspineModel(g, blocked, None, torch.float32, 'cpu', source=loader.RandomSource(params).materialise)
    bsession = Session(bmodel, capacity=capacity, device='cpu', dtype=torch.float32)
    bl = bsession.prefill(ids, inputs={'audio': audio[:24], 'delay': delay})[g.generative[0]]
    bt = [greedy({g.generative[0]: bl}, g)]
    for i in range(2):
        bt.append(greedy(bsession.decode(bt[-1], inputs={'audio': audio[24 + 8 * i:32 + 8 * i]}), g))
    ok &= check(f"tiny voxtral: {len(blocked.blocks)} blocks at legal cuts of both compositions give the same logits and tokens",
                len(blocked.blocks) > 1 and torch.equal(bl, logits) and bt == tokens)
    return ok


def m1(check):
    ok = True
    for entry in FIXTURES:
        fixture, document, checkpoint = entry[:3]
        ok &= fixture_case(check, os.path.join(REF, 'fixtures', fixture), document, os.path.join(CHECKPOINTS, checkpoint),
                           tolerance=entry[3] if len(entry) > 3 else None)
    ok &= text_only_case(check)
    return ok


def text_only_case(check):
    """§7 (finding 3): the multimodal document run on text alone — `pixels` delivers nothing,
    the vision tower is not evaluated and needs no kernel — gives exactly the text document's
    logits and tokens."""
    ck = os.path.join(CHECKPOINTS, 'Qwen3.8-27B')
    if not os.path.isdir(ck):
        print("  skip text-only equivalence (Qwen 3.8 27B not on disk)")
        return True
    tmp = tempfile.mkdtemp(prefix='tensorspine-ref-textonly-')
    kernels = registry.load_kernels()
    runs = {}
    for document in ('qwen3.8-27b', 'qwen3.8-27b-text'):
        path, _ = graph_mod.truncated(os.path.join(ROOT, 'data', 'models', f'{document}.json'), 'decoder.layer=4', tmp)
        g = graph_mod.load(path)
        plan = Plan(g, kernels)
        active = plan.evaluable({'tokens'})
        refused = registry.refusals(g, kernels, active)
        if document == 'qwen3.8-27b':
            ok = check("text only: the vision tower is not evaluated and needs no kernel",
                       not refused and not any(n.startswith('vision/') for n in active) and 'splice' in active
                       and g.input_values['pixels']['required_for'] == [] and g.input_values['tokens']['required_for'] == ['main'])
        params = loader.load_parameters(g, ck, 'cpu')
        model = TensorspineModel(g, plan, params, torch.float32, 'cpu')
        session = Session(model, capacity=64, device='cpu', dtype=torch.float32)
        ids = [760, 6511, 314, 9338, 369]
        out = session.prefill(ids)
        logits = out[g.generative[0]].clone()
        tokens = [greedy(out, g)]
        for _ in range(2):
            tokens.append(greedy(session.decode(tokens[-1]), g))
        runs[document] = (logits, tokens)
    a, b = runs['qwen3.8-27b'], runs['qwen3.8-27b-text']
    ok &= check(f"text only: the multimodal document gives the text document's logits and tokens {a[1]}",
                torch.equal(a[0], b[0]) and a[1] == b[1])
    return ok


def fixture_case(check, fixture, document, checkpoint, tolerance=None):
    """One integration fixture (docs/TENSORSPINE-FIXTURE.md): the document it names, truncated
    as it says, run on the artifact it names, compared at the tolerance it states for fp32;
    `tolerance` is verified.py's record of that tolerance, which must agree."""
    tag = os.path.basename(fixture).replace('.hf.safetensors', '').rsplit('.', 1)[-1]
    if not os.path.exists(fixture):
        print(f"  skip {document} ({tag}): fixture not on disk")
        return True
    theirs, header = read_fixture(fixture)          # refused when off the fixture schema
    ok = check(f"{document} ({tag}): the fixture is for this document, on {header['artifact']['name']}",
               header['document'] == document and header['artifact']['name'] == os.path.basename(checkpoint))
    atol, rtol = tolerance_for(header, 'f32')
    if tolerance is not None:
        ok &= check(f"{document} ({tag}): verified.py records the fixture's own tolerance", tuple(tolerance) == (atol, rtol),
                    f"verified.py {tolerance}, fixture {(atol, rtol)}")
    label = document if (atol, rtol) == (1e-3, 1e-2) else f"{document} ({tag}, atol {atol:g} rtol {rtol:g})"
    if not os.path.isdir(checkpoint):
        print(f"  skip {label} (checkpoint not on disk)")
        return ok
    ids = header['ids']
    recorded = {k[len('in/'):]: theirs[k] for k in theirs if k.startswith('in/')}   # the non-token inputs the prefill delivered (WL1)
    tmp = tempfile.mkdtemp(prefix='tensorspine-ref-fixture-')
    path, _ = graph_mod.truncated(os.path.join(ROOT, 'data', 'models', f'{document}.json'),
                                  f"{header['truncation']['composition']}.layer={header['truncation']['layers']}", tmp)
    g = graph_mod.load(path)
    errors, _, stats = loader.verify(g, checkpoint)
    ok &= check(f"{label}: the {header['truncation']['layers']}-layer document verifies against the checkpoint", not errors, errors[:1])
    kernels = registry.load_kernels()
    plan = Plan(g, kernels)
    active = plan.evaluable({g.token_input} | set(recorded))      # what the fixture delivers (§7)
    refused = registry.refusals(g, kernels, active)
    ok &= check(f"{label}: every contract the delivery evaluates has a kernel for its arguments", not refused, refused[:2])
    params = loader.load_parameters(g, checkpoint, 'cpu')
    model = TensorspineModel(g, plan, params, torch.float32, 'cpu')
    capacity = stream_capacity(g, recorded)
    session = Session(model, capacity=capacity, device='cpu', dtype=torch.float32)
    ours = {}
    encoder = g.generative is None                # no generative output: one invocation, the exposed outputs compared
    prefill_inputs, step_inputs = schedule(g, ids, recorded, len(header['tokens']) - 1)
    if encoder:
        out = session.run({g.token_input: torch.as_tensor(ids, dtype=torch.long), **recorded}, ours)
        for oname, o in g.interfaces['outputs'].items():
            ours[f"value/{o['node']}.{o['port']}"] = out[oname].detach().to('cpu', torch.float32).clone()
        primary = out[next(iter(g.interfaces['outputs']))]
        tokens = []
    else:
        out = session.prefill(ids, ours, inputs=prefill_inputs)
        ours.update(recorded_states(session))            # a window's valid tail: what the delivery's cache holds
        primary = out[g.generative[0]]
        ours['logits/last'] = primary[-1].detach().to('cpu', torch.float32).clone()
        ours['logits/argmax'] = primary.argmax(-1).detach().cpu().clone()
        nxt = greedy(out, g)
        tokens = [nxt]
        for k in range(len(header['tokens']) - 1):
            out = session.decode(nxt, inputs=step_inputs(k))
            nxt = greedy(out, g)
            tokens.append(nxt)
    rows, failures, _ = compare(ours, theirs, atol=atol, rtol=rtol)
    worst = max((r[1] for r in rows if r[1] is not None), default=0.0)
    if step_inputs(0) is not None:
        # §5.3's invariance on the checkpoint: the prefill the fixture recorded whole, delivered as one
        # fragment per token — the token with its frames, the settings with the first — gives the same
        # logits and states, within f32 rounding across row counts
        parts = Session(model, capacity=capacity, device='cpu', dtype=torch.float32)
        per = {n: t.shape[0] // len(ids) for n, t in prefill_inputs.items() if g.input_stream[n] == g.input_stream[g.feedback_input]}
        got = []
        for i, tok in enumerate(ids):
            extra = {n: t[per[n] * i:per[n] * (i + 1)] for n, t in prefill_inputs.items() if n in per}
            if i == 0:
                extra.update({n: t for n, t in prefill_inputs.items() if n not in per})
            got.append(parts.run({g.feedback_input: torch.tensor([tok]), **extra})[g.generative[0]])
        got = torch.cat(got)
        # the session went on decoding, so the replay is compared with the prefill's own record, `ours`
        d_logits = float((got - primary).abs().max())
        d_states = max(float((ours[k] - v).abs().max()) for k, v in recorded_states(parts).items() if k in ours)
        ok &= check(f"{label}: the prefill replayed as {len(ids)} fragments of {per} gives the same logits and states — "
                    f"max |d| {d_logits:.1e} on the logits, {d_states:.1e} on the states (f32 rounding across row counts)",
                    torch.allclose(got, primary, atol=1e-3, rtol=1e-3) and d_states < 1e-3)
    # the same fixture in blocks: identical logits and tokens, and the traffic is the model (M4)
    resident = loader.state_bytes(g, capacity, torch.float32) + loader.largest_temporary(g, torch.float32)
    finest = max(b.bytes + b.payload_bytes_per_element * 64 for b in Plan(g, kernels).minimal)
    blocked = Plan(g, kernels, max_bytes=resident + finest, elements=64, resident_bytes=resident)
    bmodel = TensorspineModel(g, blocked, None, torch.float32, 'cpu', source=loader.Source(g, checkpoint, 'cpu').materialise)
    bsession = Session(bmodel, capacity=capacity, device='cpu', dtype=torch.float32)
    if encoder:
        bl = bsession.run({g.token_input: torch.as_tensor(ids, dtype=torch.long), **recorded})[next(iter(g.interfaces['outputs']))]
        bt = []
    else:
        bl = bsession.prefill(ids, inputs=prefill_inputs)[g.generative[0]]
        bt = [greedy({g.generative[0]: bl}, g)]
        for k in range(len(header['tokens']) - 1):
            bt.append(greedy(bsession.decode(bt[-1], inputs=step_inputs(k)), g))
    ok &= check(f"{label}: {len(blocked.blocks)} blocks under --max-ram give the same {'outputs' if encoder else 'logits and tokens'}, "
                f"{blocked.traffic_bytes() / 2**30:.2f} GiB of traffic per invocation",
                len(blocked.blocks) > 1 and torch.equal(bl, primary) and bt == tokens)
    ok &= check(f"{label}: {len(rows)} values{'' if encoder else ', states'} and {'outputs' if encoder else 'logits'} within tolerance of transformers (max |d| {worst:.1e})",
                failures == 0, [r for r in rows if 'EXCEEDS' in r[3] or r[1] is None][:2])
    if encoder:
        exposed = [f"value/{o['node']}.{o['port']}" for o in g.interfaces['outputs'].values()]
        ok &= check(f"{label}: the exposed output {exposed} is among the compared values", all(k in theirs for k in exposed))
    else:
        ok &= check(f"{label}: greedy tokens {tokens} equal transformers' {header['tokens']}", tokens == header['tokens'])
    return ok


def schedule(g, ids, recorded, steps):
    """How a recorded delivery is fed (§5.3, D2's counts): on a document whose token stream joins
    a recorded input's stream (Voxtral: eight frames per token), the prompt's tokens take the
    first `len(ids) / count` elements of it and every step one token's worth — `(prefill inputs,
    step -> inputs)`; otherwise everything recorded goes with the prompt and a step delivers the
    token alone (Whisper's audio, a source stream complete after the prefill)."""
    if g.feedback_input is None:
        return dict(recorded), lambda k: None
    stream = g.input_stream[g.feedback_input]
    per_token = int(g.elements_per[g.feedback_input])
    joined = {n: t for n, t in recorded.items() if g.input_stream[n] == stream}
    if not joined:
        return dict(recorded), lambda k: None
    first = len(ids) * per_token
    prefill = {n: (t[:first] if n in joined else t) for n, t in recorded.items()}
    for n, t in joined.items():
        if t.shape[0] < first + steps * per_token:
            raise ValueError(f"{n}: {t.shape[0]} elements recorded, and the prompt with {steps} steps consumes {first + steps * per_token}")
    return prefill, lambda k: {n: t[first + k * per_token:first + (k + 1) * per_token] for n, t in joined.items()}


def stream_capacity(g, recorded, tokens=64):
    """The positions the caches may hold, per stream: `tokens` on the token input's stream, and on
    the stream of every recorded input at most the elements it delivers — a source stream's cache
    holds those positions or fewer (a merge only reduces them), and nothing is delivered on it
    afterwards."""
    capacity = {g.input_stream[g.token_input]: tokens}
    for name, t in recorded.items():
        stream = g.input_stream[name]
        capacity[stream] = max(capacity.get(stream, 0), t.shape[0])
    return capacity


def composite_case(check):
    """The located composite (§3.4): the same model written through a template, loaded from the same
    checkpoint by the prefixed locations of its instance, gives the flat document's logits bit for
    bit and the same greedy tokens, at three layers."""
    ck = os.path.join(CHECKPOINTS, 'Shieldstral-1.0-3B')
    if not os.path.isdir(ck):
        print("  skip composite (Shieldstral-1.0-3B not on disk)")
        return True
    tmp = tempfile.mkdtemp(prefix='tensorspine-ref-composite-')
    kernels = registry.load_kernels()
    ids = [1, 1784, 8961, 1307, 5498, 1395]
    runs = {}
    for document, edit in (('shieldstral-3b', None), ('shieldstral-3b-composite', {'quantities.layers.source.value': 3})):
        source = os.path.join(ROOT, 'data', 'models', f'{document}.json')
        path, _ = graph_mod.truncated(source, 'decoder.layer=3', tmp) if edit is None else graph_mod.edited(source, edit, tmp, '3layers')
        g = graph_mod.load(path)
        errors, _, stats = loader.verify(g, ck)
        ok = check(f"composite: {document} at three layers verifies against the checkpoint ({stats['located']} located)", not errors, errors[:1])
        params = loader.load_parameters(g, ck, 'cpu')
        session = Session(TensorspineModel(g, Plan(g, kernels), params, torch.float32, 'cpu'), 64, 'cpu', torch.float32)
        out = session.prefill(ids)
        logits = out[g.generative[0]].clone()
        tokens = [greedy(out, g)]
        for _ in range(2):
            tokens.append(greedy(session.decode(tokens[-1]), g))
        runs[document] = (logits, tokens)
    a, b = runs['shieldstral-3b'], runs['shieldstral-3b-composite']
    ok &= check(f"composite: the template form gives the flat document's logits bit for bit and its tokens {a[1]}",
                torch.equal(a[0], b[0]) and a[1] == b[1])
    return ok


def by_source_case(check):
    """§4.3's `by_source`, realised: on the three-layer Whisper against the checkpoint, the audio
    delivered whole in the prefill completes the source stream, so the cross-attention caches are
    shared whole — a child forked after the prefill holds them as the parent does and continues,
    bit for bit, as a fresh session on the same audio and prompt would; the parent continued
    otherwise is unaffected; a child forked before any delivery copies empty caches and takes its
    own audio."""
    fixture = os.path.join(REF, 'fixtures', 'whisper-large-v3.3layers.hf.safetensors')
    ck = os.path.join(CHECKPOINTS, 'whisper-large-v3')
    if not os.path.isfile(fixture) or not os.path.isdir(ck):
        print("  skip by_source (the Whisper fixture or checkpoint is not on disk)")
        return True
    theirs, header = read_fixture(fixture)
    recorded = {k[len('in/'):]: v for k, v in theirs.items() if k.startswith('in/')}
    ids = header['ids']
    tmp = tempfile.mkdtemp(prefix='tensorspine-ref-bysource-')
    path, _ = graph_mod.truncated(os.path.join(ROOT, 'data', 'models', 'whisper-large-v3.json'), 'decoder.layer=3', tmp)
    g = graph_mod.load(path)
    kernels = registry.load_kernels()
    params = loader.load_parameters(g, ck, 'cpu')
    model = TensorspineModel(g, Plan(g, kernels), params, torch.float32, 'cpu')
    capacity = stream_capacity(g, recorded)
    cross = [ident for ident, s in g.states.items() if s['sharing'] == 'by_source']
    ok = check(f"by_source: the three-layer Whisper carries {len(cross)} by_source states, on the audio stream",
               len(cross) == 3 and all(g.states[i]['stream']['stream'] == 'audio' for i in cross))

    def session():
        return Session(model, capacity, 'cpu', torch.float32)

    b, c = [header['tokens'][0], header['tokens'][1]], [header['tokens'][0]]
    fresh = session()
    fresh.prefill(ids, inputs=recorded)
    wanted_b = [fresh.decode(t)[g.generative[0]].clone() for t in b]
    fresh = session()
    fresh.prefill(ids, inputs=recorded)
    wanted_c = fresh.decode(c[0])[g.generative[0]].clone()
    parent = session()
    first = parent.prefill(ids, inputs=recorded)[g.generative[0]].clone()
    child = parent.fork()
    ok &= check("by_source: the child's cross caches are the parent's, whole — 1500 source positions each, equal buffers",
                all(child.states[i].length == 1500 and torch.equal(child.states[i].components['k'], parent.states[i].components['k']) for i in cross))
    got_b = [child.decode(t)[g.generative[0]].clone() for t in b]
    got_c = parent.decode(c[0])[g.generative[0]].clone()
    ok &= check("by_source: a child forked after the prefill continues as a fresh session on the same audio and prompt, bit for bit",
                all(torch.equal(x, y) for x, y in zip(got_b, wanted_b)) and child.consumed == {'audio': 3000, 'tokens': len(ids) + len(b)})
    ok &= check("by_source: the parent, continued otherwise, is unaffected by the child",
                torch.equal(got_c, wanted_c) and parent.consumed == {'audio': 3000, 'tokens': len(ids) + 1})
    early = session().fork()
    ok &= check("by_source: a child forked before any delivery copies empty caches and takes its own audio: the prefill's logits",
                all(early.states[i].length == 0 for i in cross)
                and torch.equal(early.prefill(ids, inputs=recorded)[g.generative[0]], first)
                and all(early.states[i].length == 1500 for i in cross))
    return ok


def m1_full(check):
    ok = True
    for entry in FULL:
        document, checkpoint, ids, expected = entry[:4]
        fixture = entry[4] if len(entry) > 4 else None       # the fixture whose in/ tensors the prompt is delivered with
        ck = os.path.join(CHECKPOINTS, checkpoint)
        if not os.path.isdir(ck):
            print(f"  skip {document} full (checkpoint not on disk)")
            continue
        g = graph_mod.load(os.path.join(ROOT, 'data', 'models', f'{document}.json'))
        kernels = registry.load_kernels()
        recorded = {}
        if fixture and fixture.endswith('.wav'):
            # a sample through the artifact's processor (the streaming prefill, the delay, the frames):
            # the whole model transcribes it to its end, one token and eight frames per step
            wav = os.path.join(ARTIFACTS, 'audio', fixture)
            if not os.path.isfile(wav):
                print(f"  skip {document} full (sample {fixture} not on disk)")
                continue
            import ref as ref_cli
            got_ids, prefill_inputs, rest = ref_cli.streaming_delivery(wav, ck, g, 'cpu', torch.float32)
            ok &= check(f"{document} full: the processor's prompt is the {len(ids)} ids recorded", got_ids == ids)
            per_token = int(g.elements_per[g.feedback_input])
            recorded = {n: torch.cat([t, rest]) if g.input_stream[n] == g.input_stream[g.feedback_input] else t for n, t in prefill_inputs.items()}
        elif fixture:
            theirs, _ = read_fixture(os.path.join(REF, 'fixtures', fixture))
            recorded = {k[len('in/'):]: v for k, v in theirs.items() if k.startswith('in/')}
        prefill_inputs, step_inputs = schedule(g, ids, recorded, len(expected) - 1)
        params = loader.load_parameters(g, ck, 'cpu')
        model = TensorspineModel(g, Plan(g, kernels), params, torch.float32, 'cpu')
        session = Session(model, capacity=stream_capacity(g, recorded), device='cpu', dtype=torch.float32)
        t0 = time.time()
        nxt = greedy(session.prefill(ids, inputs=prefill_inputs), g)
        tokens = [nxt]
        for k in range(len(expected) - 1):
            nxt = greedy(session.decode(nxt, inputs=step_inputs(k)), g)
            tokens.append(nxt)
        agree = next((i for i, (a, b) in enumerate(zip(tokens, expected)) if a != b), len(expected))
        shown = tokens if len(tokens) <= 12 else f"{tokens[:12]}… ({len(tokens)})"
        ok &= check(f"{document} full: greedy tokens {shown} equal transformers' ({time.time() - t0:.0f}s)", tokens == expected,
                    f"the first {agree} of {len(expected)} agree; ours {tokens[agree:agree + 6]} vs theirs {expected[agree:agree + 6]} from token {agree}")
        del params, model, session
    return ok


if __name__ == '__main__':
    sys.exit(main(compile_step='--compile' in sys.argv, full='--full' in sys.argv))
