#!/usr/bin/env python3
"""M0 and M1 of the reference-generator plan (§0).

M0 — random weights, no checkpoint:

  1. a tiny llama document (the corpus document with its quantities shrunk by the generic
     edit helper) derives, builds a module, and runs prefill and decode with every produced
     value checked against D2;
  2. the dump holds exactly the values D2 lists at every layer graph_split, and every state;
  3. the masked (compiled-form) attention equals the sliced one;
  4. optionally, the decode step compiles (`--compile`);
  5. the verdict of a comparison (docs/TENSORSPINE-FIXTURE.md §4): an empty comparison and a
     required key absent are failures, integers compare exactly, a dtype disagreement fails.

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

    python3 generators/reference/tests/run_reference.py [--compile] [--full] [--no-strict-provenance]

Every unit fixture must regenerate exactly from its seed on this box, where it was recorded;
`--no-strict-provenance` (CI, another torch) prints the drift instead and keeps the verdict, which is
conformance against the stored tensors.
"""
import contextlib
import io
import os
import re
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
import state as state_mod        # noqa: E402
from kernels import attention_dense  # noqa: E402
from module import TensorspineModel  # noqa: E402
from plan import Plan            # noqa: E402
from session import Batch, Session, greedy  # noqa: E402
import session as session_mod        # noqa: E402
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


def main(compile_step=False, full=False, strict_provenance=True):
    ok = True
    tmp = tempfile.mkdtemp(prefix='tensorspine-ref-test-')
    path, notes = graph_mod.edited(os.path.join(ROOT, 'data', 'models', 'llama3-8b.json'), TINY, tmp, 'tiny')
    g = graph_mod.load(path)
    ok &= check("tiny llama derives: 3 layers, 6 primitives", len(g.nodes) == 3 * 6 + 3 and len(g.layer_graph_splits()) == 2)
    kernels = registry.load_kernels()
    r = registry.refusals(g, kernels)
    ok &= check("no refusal for the six M1 primitives", not r, '; '.join(r[:3]))
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
    expected = {f"value/{p['value']}" for c in g.graph_splits for p in c['payload']}
    ok &= check("dump keys = the D2 payload of every graph_split", set(dump) == expected, f"{sorted(set(dump) ^ expected)[:4]}")
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
    ok &= check("blocks: the graph_split summary has one line per block, opening and closing at D6's graph_splits",
                len(lines) == len(blocked.blocks) + 2 and all('→' in l for l in lines[1:-1]) and 'start →' in lines[1] and '→ end' in lines[-2])
    ok &= check(f"blocks: {len(blocked.blocks)} blocks at valid graph_splits give the one-block logits bit for bit",
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
    ok &= check("physical parameters resolve primitive < pattern < exact, and other instances get none",
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
    # attention factor the document states as 1; without mscale transformers gives the paper's value.
    # `transformers` is the reference peer here, not this generator's dependency (F1): without it the
    # check says skip, as a fixture check does without its checkpoint
    try:
        from transformers import Ministral3Config
        from transformers.modeling_rope_utils import _compute_yarn_parameters
    except ImportError:
        print("  skip YaRN (transformers not importable)")
    else:
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
    ok &= compare_case(check, tmp)
    ok &= witness_case(check, strict_provenance)
    ok &= consistency_case(check, tmp)
    ok &= moe_random_case(check, tmp)
    ok &= multiplicity_case(check, tmp)
    ok &= whisper_random_case(check, tmp)
    ok &= voxtral_random_case(check, tmp)
    ok &= gemma_random_case(check, tmp)
    ok &= gemma_batch_case(check, tmp)
    ok &= sharing_case(check, tmp)
    ok &= batch_case(check, tmp)
    ok &= voxtral_batch_case(check, tmp)
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


def compare_case(check, tmp):
    """The verdict (docs/TENSORSPINE-FIXTURE.md §4, the review's I1 and I3): an unrelated dump
    against a fixture is refused with nothing compared; a dump missing one recorded state is refused
    naming it; a complete dump passes; integers compare exactly whatever the key and the tolerance;
    an integer against a float of the same values is a dtype failure."""
    import witness
    import ref as ref_cli
    from compare import write_dump

    def cli(*argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = ref_cli.main(['compare', *argv])
        return code, out.getvalue()
    ok = True
    fixture = witness.fixture_path('norm.rms@1.0.0/basic')
    unrelated = os.path.join(tmp, 'unrelated.safetensors')
    write_dump(unrelated, {'value/nothing': torch.zeros(2)}, {'compute': 'torch.float32'})
    code, text = cli(unrelated, fixture)
    ok &= check("compare: an unrelated dump against norm.rms@1.0.0/basic exits 1 with 0 keys compared", code == 1 and '0 keys compared' in text, text[-200:])
    fid = next(f for f in witness.committed() if any(k.startswith('state/') for k in read_fixture(witness.fixture_path(f))[0]))
    tensors, _ = read_fixture(witness.fixture_path(fid))
    dump = {k: v for k, v in tensors.items() if not k.startswith(('param/', 'in/'))}
    complete = os.path.join(tmp, 'complete.safetensors')
    write_dump(complete, dump, {'compute': 'torch.float32'})
    code, text = cli(complete, witness.fixture_path(fid))
    ok &= check(f"compare: the fixture {fid}'s own outputs, positions and states as a dump exit 0", code == 0 and 'within tolerance' in text, text[-200:])
    state = next(k for k in dump if k.startswith('state/'))
    partial = os.path.join(tmp, 'partial.safetensors')
    write_dump(partial, {k: v for k, v in dump.items() if k != state}, {'compute': 'torch.float32'})
    code, text = cli(partial, witness.fixture_path(fid))
    ok &= check(f"compare: the same dump without {state} exits 1 naming it", code == 1 and f"missing: {state}" in text, text[-200:])
    big = (torch.tensor([16777216], dtype=torch.int64), torch.tensor([16777217], dtype=torch.int64))
    for key in ('x', 'x/argmax'):
        v = compare({key: big[0]}, {key: big[1]})
        ok &= check(f"compare: int64 16777216 against 16777217 fails under key {key} — no float32 cast", not v.ok and v.failures == 1 and 'unequal' in v.rows[0][3])
    v = compare({'x': torch.tensor([1000])}, {'x': torch.tensor([1001])}, atol=1e-3, rtol=1e-2)
    ok &= check("compare: int64 1000 against 1001 fails under atol 1e-3, rtol 1e-2 — integers are exact", not v.ok and v.failures == 1)
    v = compare({'x': torch.tensor([1000, 7], dtype=torch.int32)}, {'x': torch.tensor([1000, 7], dtype=torch.int64)})
    ok &= check("compare: equal integers of different widths pass, compared in int64", v.ok and v.rows[0][3] == '2/2 equal')
    v = compare({'x': torch.tensor([1000])}, {'x': torch.tensor([1000.0])})
    ok &= check("compare: an int64 against a float32 of the same values fails on dtype", not v.ok and 'dtype int64 vs float32' in v.rows[0][3])
    v = compare({'y': torch.zeros(2)}, {'x': torch.zeros(2)})
    ok &= check("compare: no key in common is a failure with 0 compared, the required key missing and ours unexpected",
                not v.ok and v.compared == 0 and v.missing == ['x'] and v.unexpected == ['y'])
    return ok


def witness_case(check, strict_provenance=True):
    """The witness did not change silently (docs/TENSORSPINE-FIXTURE.md §5): every committed unit
    fixture regenerates from its seed — exactly, under strict provenance — and its run repeats
    within its own tolerance at every dtype the kernel declares one for, with the parameters loaded
    from the fixture as a conformer loads them and every recorded key required; and every case a
    kernel declares is recorded."""
    import witness
    kernels = registry.load_kernels()
    ok = True
    ids = witness.committed()
    for fid in ids:
        good, lines = witness.verify(fid, kernels, strict_provenance)
        ok &= check(f"witness {fid}: regenerated and repeated within tolerance", good, '\n         '.join(lines))
    declared = {f"{n}@{v}/{c['case']}" for n, v, _k, c in witness.cases(kernels)}
    ok &= check(f"witness: the {len(declared)} cases the kernels declare are the {len(ids)} fixtures committed",
                declared == set(ids), str(sorted(declared ^ set(ids))[:3]))
    return ok


def consistency_case(check, tmp):
    """R14 (generators/CAPABILITIES.md): the manifest and the kernels agree — every argument
    combination the validator admits is either refused by supports() and never run, or run once on
    tiny shapes without raising. A raise on an admitted combination is a kernel refusing what the
    manifest does not declare (the review's C3). Over every kernel, from each of its unit fixtures
    (valid bases), every manifest-enumerated value of every top-level argument is overridden in
    turn, and each present record is dropped; each combination is derived (an invalid one is
    counted and skipped), and an admitted one is run for one invocation."""
    import json
    import witness
    import capabilities as cap
    import primitive_library as primitive_library_mod
    kernels = registry.load_kernels()
    cat = primitive_library_mod.load(os.path.join(ROOT, 'data', 'primitive-library'))
    manifest, errs = cap.load(os.path.join(REF, 'capabilities.json'))
    if errs:
        return check("consistency: the manifest loads", False, str(errs[:2]))
    import graph as g_mod
    tried = admitted = refused_validator = refused_supports = structural = 0
    failures = []
    for (name, version), kernel in sorted(kernels.items()):
        entry = manifest['primitives'].get(f"{name}@{version}")
        if entry is None:
            continue
        table = entry['arguments']
        # a combination is (arguments, the invocations and seed of the fixture it varies): the
        # fixture's own delivery is a valid one for the primitive's ports — an insert's source
        # delivering nothing where it must (splice), a merge's groups aligned. `kv_source: shared`
        # is a topological feature (a writer instance a one-instance document cannot hold), so
        # it is not overridden here; the gemma3n random case exercises it on a real topology.
        combos = []
        seen = set()
        for c in getattr(kernel, 'FIXTURES', []):
            base = c['arguments']
            base_ports = {n for inv in c['invocations'] for n in inv}
            variants = [dict(base)]
            for arg, rule in table.items():
                if arg == 'kv_source':                 # a shared reader needs a writer instance
                    continue
                values = rule if isinstance(rule, list) else (rule.get('values') if isinstance(rule, dict) else None)
                for v in (values or []):
                    if v is not None and base.get(arg) != v:
                        variants.append({**base, arg: v})
            for arg in list(base):
                if isinstance(base[arg], dict):
                    variants.append({k: v for k, v in base.items() if k != arg})
            for a in variants:
                key = json.dumps(a, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    combos.append((a, c['invocations'], c['seed'], base_ports))
        for args, invocations, seed, base_ports in combos:
            tried += 1
            base_dir = witness.fixture_dir(name, version)
            try:
                doc = witness.document(name, version, args, cat, witness.primitive_library_base_from(base_dir))
                gpath = witness._materialise(doc, tmp)
                gg = g_mod.load(gpath)
            except (ValueError, KeyError):
                refused_validator += 1
                continue
            # a variant that changes the input-port set (cross adds a source) cannot be delivered by
            # the base fixture's invocations — a structural change, covered by the corpus and the
            # per-model random cases; the pairwise run here holds the port set fixed
            if set(doc['interfaces']['inputs']) != base_ports:
                structural += 1
                continue
            resolved = gg.nodes['unit']['arguments']
            if cap.supports(entry, resolved):
                refused_supports += 1
                continue
            admitted += 1
            try:
                params = witness.parameters(gg, seed)
                witness.run(gg, kernels, params, invocations, torch.float32, seed=seed)
            except Exception as e:  # noqa: BLE001
                failures.append(f"{name}@{version} {args}: {type(e).__name__}: {str(e)[:120]}")
    ok = check(f"consistency: {tried} combinations over {len(kernels)} kernels — {admitted} admitted and run, "
               f"{refused_supports} refused by the manifest, {refused_validator} refused by the validator, "
               f"{structural} structural (port set changed); no admitted combination raised",
               not failures, '\n         '.join(failures[:4]))
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
    evolutions = {s['evolution'] for s in g.states.values()}
    ok &= check("sharing: the tiny hybrid carries all three evolutions", evolutions == {'append', 'window', 'fixed'}, str(evolutions))
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


def multiplicity_case(check, tmp):
    """Finding 30: a slot with a multiplicity is stored with a leading storage axis. Three separately
    named matrices under a `stack` over `multiplicity` and one fused [3, …] tensor load to identical
    values in the same copy order — through the language's own resolver and the loader's assembly."""
    from safetensors.torch import save_file
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    import primitive_library as primitive_library_mod
    import validate
    cat = primitive_library_mod.load(os.path.join(ROOT, 'data', 'primitive-library'))
    slot = cat['primitives']['residual.stream_expand']['parameters']['projection']
    args = {'width': 2, 'streams': 4}
    copies = [torch.arange(4, dtype=torch.float32).reshape(2, 2) + 10 * i for i in range(3)]
    ck = os.path.join(tmp, 'multiplicity')
    os.makedirs(ck, exist_ok=True)
    save_file({**{f'p.{i}.weight': copies[i] for i in range(3)}, 'p.weight': torch.stack(copies)},
              os.path.join(ck, 'model.safetensors'))
    shape = validate._storage_shape(slot)
    logical = [3, 2, 2]
    none = lambda e, env=None: None          # no `{index}` item to evaluate  # noqa: E731
    stacked, p1 = validate.evaluate_location({'stack': {'axis': 'multiplicity', 'part': {'tensor': ['p.', {'coordinate': 'multiplicity'}, '.weight']}}},
                                             {}, shape, args, none)
    fused, p2 = validate.evaluate_location({'tensor': ['p.weight']}, {}, shape, args, none)
    ok = check("multiplicity: the resolver expands a stack over the storage axis into three parts at dim 0 and takes the fused tensor whole",
               not p1 and not p2 and stacked['stack']['dim'] == 0 and len(stacked['stack']['parts']) == 3 and fused == {'tensor': 'p.weight'},
               str(p1[:1] or p2[:1] or stacked))
    src = loader.Source(None, ck, 'cpu')
    a, b = src.assemble(stacked, logical), src.assemble(fused, logical)
    ok &= check("multiplicity: three separately named [2, 2] matrices and one fused [3, 2, 2] tensor load to identical values",
                list(a.shape) == logical and torch.equal(a, b), f"{list(a.shape)} vs {list(b.shape)}")
    ok &= check("multiplicity: copy i of the stack is the matrix named i — the copy order is the coordinate",
                all(torch.equal(a[i], copies[i]) for i in range(3)))
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
    ok &= check("tiny qwen3.5-moe: every primitive the text delivery evaluates has a kernel", not refused, refused[:2])
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


TINY_GEMMA = {'quantities.d.source.value': 64, 'quantities.heads.source.value': 4, 'quantities.kv_heads.source.value': 2,
              'quantities.hd.source.value': 16, 'quantities.ffn.source.value': 128, 'quantities.vocab.source.value': 256,
              'quantities.per_layer_vocab.source.value': 256, 'quantities.per_layer_width.source.value': 16,
              'quantities.laurel_rank.source.value': 8, 'quantities.window.source.value': 8}


def gemma_batch_case(check, tmp):
    """B06 on the shared-KV document (breadth plan G10): the tiny 21-layer Gemma batched on the packed
    layout — two prompts of different lengths prefilled together and decoded together twice, against
    each alone; the split the runner reads from the topology is the two attention sites (own and
    readers, each holding the identity's state) per session, everything else on the union; every
    state's length, the shared ring's included, is its session's own."""
    ok = True
    kernels = registry.load_kernels()
    path, _ = graph_mod.truncated(os.path.join(ROOT, 'data', 'models', 'gemma3n-kvshare.json'), 'decoder.layer=21', tmp)
    path, _ = graph_mod.edited(path, TINY_GEMMA, tmp, 'tiny-batch')
    g = graph_mod.load(path)
    params = loader.random_parameters(g, 'cpu', seed=2)
    model = TensorspineModel(g, Plan(g, kernels), params, torch.float32, 'cpu')
    per = {n.split('[')[0] for n, v in model.per_session.items() if v}
    ok &= check(f"batch (tiny gemma3n): the instances evaluated per session are the two attention sites, own and readers — {sorted(per)}",
                per == {'decoder/attn', 'decoder/attn_full'}, str(sorted(per)))
    A, B = [1, 2, 3, 4, 5, 6], [9, 10, 11]

    def session():
        return Session(model, 64, 'cpu', torch.float32)
    alone = []
    for prompt in (A, B):
        s = session()
        logits = [s.prefill(prompt)[g.generative[0]].clone()]
        tokens = [greedy({g.generative[0]: logits[-1]}, g)]
        for _ in range(2):
            logits.append(s.decode(tokens[-1])[g.generative[0]].clone())
            tokens.append(greedy({g.generative[0]: logits[-1]}, g))
        alone.append((logits, tokens, {i: st.length for i, st in s.states.items()}))
    batch = Batch([session(), session()])
    outs = batch.prefill([A, B])
    logits = [[o[g.generative[0]].clone()] for o in outs]
    nxt = [greedy(o, g) for o in outs]
    tokens = [[n] for n in nxt]
    for _ in range(2):
        outs = batch.decode(nxt)
        nxt = [greedy(o, g) for o in outs]
        for k, o in enumerate(outs):
            logits[k].append(o[g.generative[0]].clone())
            tokens[k].append(nxt[k])
    worst = max(float((x - y).abs().max()) for k in range(2) for x, y in zip(logits[k], alone[k][0]))
    ok &= check(f"batch (tiny gemma3n): two prompts of {len(A)} and {len(B)} tokens prefilled and decoded as one packed batch give each session's own "
                f"logits (max |d| {worst:.1e}) and tokens",
                all(torch.allclose(x, y, atol=1e-5, rtol=1e-4) for k in range(2) for x, y in zip(logits[k], alone[k][0]))
                and tokens == [a[1] for a in alone], f"batched {tokens}, alone {[a[1] for a in alone]}")
    ok &= check("batch (tiny gemma3n): every state's length is its session's own, the shared ring's included",
                all({i: st.length for i, st in s.states.items()} == a[2] for s, a in zip(batch.sessions, alone))
                and [s.states['shared.sliding.kv'].length for s in batch.sessions] == [len(A) + 2, len(B) + 2])
    return ok


def gemma_random_case(check, tmp):
    """The Gemma 3n document on random weights, 21 layers so that layer 20 reads the ring layer 18
    writes, a window of 8: the four-stream residual, the per-layer inputs, the shared ring, the
    shared cache (layer 19 alone once truncated), blocks across the writer and its reader, and the
    refusal of a window reader's multi-position invocation once the ring has wrapped (finding 26)."""
    ok = True
    path, _ = graph_mod.truncated(os.path.join(ROOT, 'data', 'models', 'gemma3n-kvshare.json'), 'decoder.layer=21', tmp)
    path, _ = graph_mod.edited(path, TINY_GEMMA, tmp, 'tiny')
    g = graph_mod.load(path)
    kernels = registry.load_kernels()
    refused = registry.refusals(g, kernels)
    ok &= check("tiny gemma3n: every primitive has a kernel for its arguments", not refused, refused[:2])
    ok &= check("tiny gemma3n: the shared ring is written by layer 18 and read by layer 20; the full cache has layer 19 alone",
                g.states['shared.sliding.kv']['writer'] == 'decoder/attn[layer=18].kv'
                and 'decoder/attn[layer=20].kv' in g.states['shared.sliding.kv']['members']
                and g.states['shared.full.kv']['members'] == ['decoder/attn_full[layer=19].kv'])
    params = loader.random_parameters(g, 'cpu', seed=2)
    ok &= check("tiny gemma3n: a multiplicity slot is one stacked tensor", list(params['expand.projection'].shape) == [3, 64, 64])
    model = TensorspineModel(g, Plan(g, kernels), params, torch.float32, 'cpu')
    session = Session(model, capacity=64, device='cpu', dtype=torch.float32)
    out = session.prefill([1, 2, 3, 4, 5, 6])
    logits = out[g.generative[0]]
    tokens = [greedy(out, g)]
    for _ in range(10):                                              # past the window of 8: the ring wraps
        o = session.decode(tokens[-1])
        tokens.append(greedy(o, g))
    ok &= check("tiny gemma3n: prefill and ten decodes past the window give finite logits on their D2 shapes",
                list(logits.shape) == [6, 256] and bool(torch.isfinite(logits).all()) and session.states['shared.sliding.kv'].length == 16)
    try:
        session.run({g.token_input: torch.tensor([7, 8])})
        ok &= check("tiny gemma3n: a window reader refuses a multi-position invocation once the ring wrapped (finding 26)", False)
    except state_mod.Refusal as e:
        ok &= check("tiny gemma3n: a window reader refuses a multi-position invocation once the ring wrapped (finding 26)", 'finding 26' in str(e))
    resident = loader.state_bytes(g, 64, torch.float32) + loader.largest_temporary(g, torch.float32)
    total = g.d3_totals['bytes']
    blocked = Plan(g, kernels, max_bytes=resident + total // 3, elements=64, resident_bytes=resident)
    bmodel = TensorspineModel(g, blocked, None, torch.float32, 'cpu', source=loader.RandomSource(params).materialise)
    bsession = Session(bmodel, capacity=64, device='cpu', dtype=torch.float32)
    bl = bsession.prefill([1, 2, 3, 4, 5, 6])[g.generative[0]]
    bt = [greedy({g.generative[0]: bl}, g)]
    for _ in range(10):
        bt.append(greedy(bsession.decode(bt[-1]), g))
    ok &= check(f"tiny gemma3n: {len(blocked.blocks)} blocks, the shared ring's writer and reader in different ones, give the same logits and tokens",
                len(blocked.blocks) > 1 and torch.equal(bl, logits) and bt == tokens)
    return ok


TINY_WHISPER = {'quantities.d.source.value': 64, 'quantities.heads.source.value': 4, 'quantities.hd.source.value': 16,
                'quantities.ffn.source.value': 128, 'quantities.vocab.source.value': 256, 'quantities.mels.source.value': 8,
                'quantities.enc_layers.source.value': 2, 'compositions.encoder.indices.layer.stop.literal': 2,
                'quantities.dec_layers.source.value': 2, 'compositions.decoder.indices.layer.stop.literal': 2,
                'instances.conv_frontend.arguments.position.literal': 16, 'instances.embed.arguments.positions.literal': 16}


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
    ok &= check("tiny whisper: the audio and the prompt evaluate every instance, each with a kernel for its arguments",
                not refused and len(active) == len(g.nodes), refused[:2])
    ok &= check("tiny whisper: the prompt alone evaluates five instances before any audio is cached (§7)",
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
    ok &= check("tiny whisper: the encoder runs on 12 positions and its output crosses every decoder graph_split, dumped once as [12, 64]",
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
    ok &= check(f"tiny whisper: {len(blocked.blocks)} blocks at valid graph_splits of both compositions give the same logits and tokens",
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
        bufs, length = st.tail() if st.evolution == 'window' else st.read()
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
    ok &= check("tiny voxtral: the tokens, the frames and the delay evaluate every instance, each with a kernel for its arguments",
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
    ok &= check("tiny voxtral: the values crossing the graph_splits are dumped once — the encoder's output on 12 positions, the projector's and the fused embedding on 3 tokens, the time embedding on 1",
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
    ok &= check(f"tiny voxtral: {len(blocked.blocks)} blocks at valid graph_splits of both compositions give the same logits and tokens",
                len(blocked.blocks) > 1 and torch.equal(bl, logits) and bt == tokens)
    return ok


def batch_case(check, tmp):
    """B06 (batch-plan): a batch of sessions equals the same sessions run alone, on the tiny Llama
    (append) and the tiny hybrid (append, window, fixed; the mixture and the recurrence): two prompts
    of different lengths prefilled as one packed batch and decoded together twice, against each
    prompt alone — logits within f32 rounding across row counts, the same greedy tokens, every
    state's length its own; the split the runner made from the topology is the expected one; a
    batch delivering different inputs, and one over the bound, are refused."""
    ok = True
    kernels = registry.load_kernels()
    A, B = [1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11]
    for label, source, edits, seed, per_expected in (
            ('tiny llama', 'llama3-8b.json', TINY, 1, {'decoder/attn'}),
            ('tiny hybrid', 'qwen3.5-35b-a3b.json', TINY_MOE, 5, {'decoder/attn', 'decoder/gdn', 'splice'})):
        if edits is TINY_MOE:
            path, _ = graph_mod.truncated(os.path.join(ROOT, 'data', 'models', source), 'decoder.layer=4', tmp)
            path, _ = graph_mod.edited(path, edits, tmp, 'tiny-batch')
        else:
            path, _ = graph_mod.edited(os.path.join(ROOT, 'data', 'models', source), edits, tmp, 'tiny-batch')
        g = graph_mod.load(path)
        params = loader.random_parameters(g, 'cpu', seed=seed)
        model = TensorspineModel(g, Plan(g, kernels), params, torch.float32, 'cpu')
        active = model.plan.evaluable({g.feedback_input})          # the text delivery (§7): the vision tower stays out
        per = {n.split('[')[0] for n, v in model.per_session.items() if v and n in active}
        ok &= check(f"batch ({label}): on the text delivery, the instances evaluated per session are those reading across positions or holding a state — {sorted(per)}",
                    per == per_expected, str(sorted(per)))

        def session():
            return Session(model, 64, 'cpu', torch.float32)
        alone = []
        for prompt in (A, B):
            s = session()
            logits = [s.prefill(prompt)[g.generative[0]].clone()]
            tokens = [greedy({g.generative[0]: logits[-1]}, g)]
            for _ in range(2):
                logits.append(s.decode(tokens[-1])[g.generative[0]].clone())
                tokens.append(greedy({g.generative[0]: logits[-1]}, g))
            alone.append((logits, tokens, {i: st.length for i, st in s.states.items()}))
        batch = Batch([session(), session()])
        outs = batch.prefill([A, B])
        logits = [[o[g.generative[0]].clone()] for o in outs]
        nxt = [greedy(o, g) for o in outs]
        tokens = [[n] for n in nxt]
        for _ in range(2):
            outs = batch.decode(nxt)
            nxt = [greedy(o, g) for o in outs]
            for k, o in enumerate(outs):
                logits[k].append(o[g.generative[0]].clone())
                tokens[k].append(nxt[k])
        worst = max(float((x - y).abs().max()) for k in range(2) for x, y in zip(logits[k], alone[k][0]))
        ok &= check(f"batch ({label}): two prompts of {len(A)} and {len(B)} tokens prefilled and decoded as one packed batch give each session's own "
                    f"logits (max |d| {worst:.1e}, f32 rounding across row counts) and tokens",
                    all(torch.allclose(x, y, atol=1e-5, rtol=1e-4) for k in range(2) for x, y in zip(logits[k], alone[k][0]))
                    and [t for t in tokens] == [a[1] for a in alone], f"batched {tokens}, alone {[a[1] for a in alone]}")
        ok &= check(f"batch ({label}): every state's length is its session's own",
                    all({i: st.length for i, st in s.states.items()} == a[2] for s, a in zip(batch.sessions, alone)))
    try:
        Batch([session(), session()]).run([{g.feedback_input: torch.tensor([1])}, {}])
        ok &= check("batch: sessions delivering different inputs are refused", False)
    except Exception as e:  # noqa: BLE001
        ok &= check("batch: sessions delivering different inputs are refused", 'same inputs' in str(e), str(e)[:120])
    try:
        Batch([session() for _ in range(session_mod.SESSIONS_PER_INVOCATION + 1)])
        ok &= check("batch: more sessions than the manifest's bound are refused", False)
    except Exception as e:  # noqa: BLE001
        ok &= check("batch: more sessions than the manifest's bound are refused", 'at most' in str(e), str(e)[:120])
    return ok


def _max_diff(a, b):
    return float((a - b).abs().max()) if a.shape == b.shape and a.numel() else (0.0 if a.shape == b.shape else float('inf'))


def voxtral_batch_case(check, tmp):
    """The tiny streaming document batched on the packed layout (B06 on a merge): two sessions of
    different lengths — three tokens with twenty-four frames and two tokens with sixteen, each
    with its own delay — prefilled together and decoded together twice, a token and eight frames
    each, against each alone. The split is read from the derived document (harness guide §8):
    the stem (it reads neighbouring frames), the attentions and the conditioning scales (a state,
    two streams) per session; the temporal projector — a merge, its groups inside each session —
    on the union with the embeddings, the norms, the feed-forwards, the residual sums and the
    head."""
    ok = True
    path, _ = graph_mod.edited(os.path.join(ROOT, 'data', 'models', 'voxtral-realtime.json'), TINY_VOXTRAL, tmp, 'tiny-batch')
    g = graph_mod.load(path)
    kernels = registry.load_kernels()
    params = loader.random_parameters(g, 'cpu', seed=9)
    model = TensorspineModel(g, Plan(g, kernels), params, torch.float32, 'cpu')
    per = {n for n, v in model.per_session.items() if v}
    want_per = {n for n, e in g.nodes.items() if e['primitive']['name'] in ('conv_frontend', 'attention.dense', 'conditioning.scale')}
    ok &= check("batch (tiny voxtral): per session — the stem, the attentions and the conditioning scales; on the union — the temporal "
                "projector (a merge), the time embedding, the fused embedding, the norms, the feed-forwards and the head",
                per == want_per and not any(model.per_session[n] for n in ('audio_projector', 'time_embed', 'fuse', 'embed', 'lm_head')),
                str(sorted(per ^ want_per)[:4]))
    capacity = {'audio': 64, 'delay': 1}
    gen = torch.Generator().manual_seed(11)
    audio = [torch.randn(48, 8, generator=gen), torch.randn(40, 8, generator=gen)]
    delays = [torch.tensor([6], dtype=torch.int32), torch.tensor([3], dtype=torch.int32)]
    prompts, frames = [[1, 2, 3], [4, 5]], [24, 16]

    def session():
        return Session(model, capacity=capacity, device='cpu', dtype=torch.float32)

    def fragment(k, i):
        return audio[k][frames[k] + 8 * i:frames[k] + 8 * i + 8]
    alone = []
    for k in range(2):
        s = session()
        logits = [s.prefill(prompts[k], inputs={'audio': audio[k][:frames[k]], 'delay': delays[k]})[g.generative[0]].clone()]
        tokens = [greedy({g.generative[0]: logits[-1]}, g)]
        for i in range(2):
            logits.append(s.decode(tokens[-1], inputs={'audio': fragment(k, i)})[g.generative[0]].clone())
            tokens.append(greedy({g.generative[0]: logits[-1]}, g))
        alone.append((logits, tokens, recorded_states(s), dict(s.consumed)))
    batch = Batch([session(), session()])
    outs = batch.prefill(prompts, inputs=[{'audio': audio[k][:frames[k]], 'delay': delays[k]} for k in range(2)])
    logits = [[o[g.generative[0]].clone()] for o in outs]
    nxt = [greedy(o, g) for o in outs]
    tokens = [[n] for n in nxt]
    for i in range(2):
        outs = batch.decode(nxt, inputs=[{'audio': fragment(k, i)} for k in range(2)])
        nxt = [greedy(o, g) for o in outs]
        for k, o in enumerate(outs):
            logits[k].append(o[g.generative[0]].clone())
            tokens[k].append(nxt[k])
    worst = max(_max_diff(x, y) for k in range(2) for x, y in zip(logits[k], alone[k][0]))
    ok &= check(f"batch (tiny voxtral): 3 tokens with 24 frames and 2 tokens with 16, prefilled and decoded together twice, give each "
                f"session its own logits (max |d| {worst:.1e}, f32 rounding across row counts) and tokens — the temporal projector on the union",
                all(x.shape == y.shape and torch.allclose(x, y, atol=1e-5, rtol=1e-4) for k in range(2) for x, y in zip(logits[k], alone[k][0]))
                and tokens == [a[1] for a in alone], f"batched {tokens}, alone {[a[1] for a in alone]}")
    d_states = max(_max_diff(recorded_states(s)[key], a[2][key]) for s, a in zip(batch.sessions, alone) for key in a[2])
    ok &= check(f"batch (tiny voxtral): every ring, history and condition cache is its session's own (max |d| {d_states:.1e}); the audio "
                f"streams advanced by 40 and 32 frames",
                d_states < 1e-5 and [dict(s.consumed) for s in batch.sessions] == [a[3] for a in alone]
                and [s.consumed['audio'] for s in batch.sessions] == [40, 32], str([dict(s.consumed) for s in batch.sessions]))
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
    ok &= check(f"{label}: every primitive the delivery evaluates has a kernel for its arguments", not refused, refused[:2])
    params = loader.load_parameters(g, checkpoint, 'cpu')
    ok &= check(f"{label}: every loaded parameter has D3's stored shape — the shape random parameters draw (a declared multiplicity leading)",
                all(list(params[i].shape) == [a['extent'] for a in t['shape']] for i, t in g.tensors.items()))
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
    composition = header['truncation']['composition']
    expected = expectation(g, composition)
    absent = sorted(expected - set(theirs))
    ok &= check(f"{label}: the fixture holds the value crossing each layer boundary of {composition}, every exposed output and every state "
                f"{composition} writes on its own stream ({len(expected)} keys)", not absent, str(absent[:3]))
    verdict = compare(ours, theirs, atol=atol, rtol=rtol, required=expected)
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
    if not encoder and step_inputs(0) is None:
        # B06 on the checkpoint: the fixture's prompt and its first half as one packed batch, decoded
        # together for the fixture's steps — the first session against the fixture's own record, the
        # second against its run alone; both deliver the prefill's other inputs (Whisper's audio, the
        # cross source of both sessions), since a batch's sessions evaluate the same instances (§7)
        half = ids[:max(1, len(ids) // 2)]
        alone = Session(model, capacity=capacity, device='cpu', dtype=torch.float32)
        a_logits = [alone.prefill(half, inputs=prefill_inputs)[g.generative[0]].clone()]
        a_tokens = [greedy({g.generative[0]: a_logits[-1]}, g)]
        for _ in range(len(header['tokens']) - 1):
            a_logits.append(alone.decode(a_tokens[-1])[g.generative[0]].clone())
            a_tokens.append(greedy({g.generative[0]: a_logits[-1]}, g))
        batch = Batch([Session(model, capacity=capacity, device='cpu', dtype=torch.float32) for _ in range(2)])
        outs = batch.prefill([ids, half], inputs=[prefill_inputs, prefill_inputs])
        b_logits = [[o[g.generative[0]].clone()] for o in outs]
        nxt = [greedy(o, g) for o in outs]
        b_tokens = [[n] for n in nxt]
        for _ in range(len(header['tokens']) - 1):
            outs = batch.decode(nxt)
            nxt = [greedy(o, g) for o in outs]
            for k, o in enumerate(outs):
                b_logits[k].append(o[g.generative[0]].clone())
                b_tokens[k].append(nxt[k])
        d0 = float((b_logits[0][0] - primary).abs().max())
        d1 = max(float((x - y).abs().max()) for x, y in zip(b_logits[1], a_logits))
        ok &= check(f"{label}: the prompt and its first {len(half)} tokens as one packed batch, decoded together, give each session its own "
                    f"logits and tokens (max |d| {d0:.1e} against the fixture's run, {d1:.1e} against the half alone)",
                    torch.allclose(b_logits[0][0], primary, atol=1e-3, rtol=1e-3) and b_tokens[0] == tokens
                    and all(torch.allclose(x, y, atol=1e-3, rtol=1e-3) for x, y in zip(b_logits[1], a_logits)) and b_tokens[1] == a_tokens,
                    f"batched {b_tokens}, fixture {tokens}, half alone {a_tokens}")
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
    ok &= check(f"{label}: every recorded key compared — {verdict.compared} values{'' if encoder else ', states'} and {'outputs' if encoder else 'logits'} "
                f"within tolerance of transformers (max |d| {verdict.worst:.1e})", verdict.ok, verdict.detail())
    if not encoder:
        ok &= check(f"{label}: greedy tokens {tokens} equal transformers' {header['tokens']}", tokens == header['tokens'])
    return ok


LAYER = re.compile(r'\[.*\blayer=(\d+)')


def layer_of(node, composition):
    """The layer index of a node of the composition; None for a node outside it."""
    if not node.startswith(composition + '/'):
        return None
    m = LAYER.search(node)
    return int(m.group(1)) if m else None


def expectation(g, composition):
    """What an integration fixture must hold and the reference must produce, read off the truncated
    derived document (the fixture guide §4): the value crossing each layer boundary of the
    truncated composition (the payload of every `layer`-kind graph_split of it — the output of every layer
    but the last, whose value crosses the composition boundary and is compared when both sides hold
    it), every exposed output, `logits/last` and `logits/argmax` for a generative document, and
    every state an instance of the composition writes on its own stream. A state indexed by a
    source stream (a cross-attention cache, a condition cache) is the source's evidence, which a
    dumper may leave out and its `hook_map` then says so; a fixture value the reference routes
    through a family graph_split instead (Gemma's per-layer inject) is compared when both hold it, never
    required."""
    keys = set()
    for c in g.layer_graph_splits():
        if c['graph_split'].startswith(composition + '['):
            keys |= {f"value/{p['value']}" for p in c['payload']}
    for name, o in g.interfaces['outputs'].items():
        if g.generative and name == g.generative[0]:
            keys |= {'logits/last', 'logits/argmax'}
        else:
            keys.add(f"value/{o['node']}.{o['port']}")
    for ident, st in g.states.items():
        if layer_of(st['writer'].rsplit('.', 1)[0], composition) is not None and not st['indexed_by_source']:
            keys |= {f"state/{ident}/{p['component']}" for p in st['payload']}
    return keys


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
    # R18 (§5.3, source completeness): a stream a reader reads whole (a by_source cache — cross
    # attention's source) must be complete before the reader's first fragment, so it is never the
    # fragmented stream the token input joins. The corpus delivers such a source whole in the
    # prefill (Whisper's audio); a schedule that fragmented it across the steps is refused here,
    # where the runner delivers, rather than left to compute on what has not arrived.
    by_source = {st['stream']['stream'] for st in g.states.values()
                 if st.get('sharing') == 'by_source' and st.get('stream')}
    if stream in by_source:
        raise ValueError(f"{stream}: a source read whole (a by_source cache) cannot be fragmented across the "
                         f"reader's invocations — it must be complete before the reader's first fragment (§5.3, R18)")
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
    sys.exit(main(compile_step='--compile' in sys.argv, full='--full' in sys.argv,
                  strict_provenance='--no-strict-provenance' not in sys.argv))
