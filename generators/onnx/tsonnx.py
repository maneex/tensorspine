#!/usr/bin/env python3
"""The ONNX generator's command line.

    tsonnx.py info    DERIVED                               counts, and the refusals over the instances a delivery evaluates
    tsonnx.py emit    DERIVED --checkpoint DIR --out F.onnx [--dump] [--inputs a,b] [--target onnx|onnxruntime]   the graph of one invocation
    tsonnx.py run     DERIVED --checkpoint DIR --ids 1,2,3 [--steps N] [--dump F] [--target …]                  prefill and greedy decode through onnxruntime
    tsonnx.py run     DERIVED --random [--seed N] …                                  parameters drawn from the D3 shapes
    tsonnx.py run     DERIVED … --ids 1,2,3 --ids 4,5,6 --batch aligned --capacity N    several prompts of one length as one batch of
                                                                                     sessions (a batch axis; the append states buffers of N positions)
    tsonnx.py capabilities [--out FILE] [--check]                                    the manifest, from the emitters' tables

DERIVED is a derived document (`tensorspine --derive MODEL -o DIR`): the generator reads D1–D6 and
the checkpoint, never the model source or the primitive_library. The batch layout (`--batch`) is an argument
too: `none`, one session on the element axis; `aligned`, a batch axis before it (generators/onnx/README.md,
Batching). The target is the runtime the graph is
emitted for: `onnx` (standard operators, runs anywhere) or `onnxruntime` (its fused operators
where a primitive has a fused form: one GroupQueryAttention per attention, SimplifiedLayerNormalization,
the residual sum and its norm as one SkipSimplifiedLayerNormalization, fused activations); the
physical parameters (`--physical`) may still name a backend per instance. See generators/onnx/README.md.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import graph as graph_mod      # noqa: E402
import loader                  # noqa: E402
import registry                # noqa: E402
from emit import Emitter, Refusal, save_model   # noqa: E402
from session import Batch, Session, greedy  # noqa: E402
import session as session_mod        # noqa: E402


def delivery(args, g):
    if args.inputs:
        return set(args.inputs.split(',')) | ({g.feedback_input} if g.feedback_input else set())
    return {g.feedback_input} if g.feedback_input else set(g.interfaces['inputs'])


def build(args, g):
    """The emitter over the checkpoint (V17 checked first) or random parameters; None on a refusal."""
    prims = registry.load_all()
    em = Emitter(g, None, prims, target=args.target, layout=args.batch)
    r = em.refusals(em.evaluable(delivery(args, g)))
    if r:
        print(f"refused: {len(r)} reason(s)")
        for line in r[:20]:
            print("  " + line)
        return None
    if getattr(args, 'random', False):
        source = loader.RandomSource(g, args.seed)
    else:
        errors, _advisories, stats = loader.verify(g, args.checkpoint)
        if errors:
            print(f"refused: {len(errors)} V17 error(s) against {args.checkpoint}")
            for line in errors[:10]:
                print("  " + line)
            return None
        print(f"  verified {stats['located']} located tensors against {stats['physical']} physical ({stats['unnamed']} unnamed)")
        source = loader.Source(g, args.checkpoint)
    return Emitter(g, source, prims, args.compute, physical_of(args), target=args.target, layout=args.batch)


def describe(model, delivered, seconds):
    print(f"  emitted {len(model.graph.node)} ONNX nodes, {len(model.graph.initializer)} initializers, {len(model.graph.input)} inputs, "
          f"{len(model.graph.output)} outputs for the inputs {sorted(delivered)} ({seconds:.1f}s)")


def physical_of(args):
    if not getattr(args, 'physical', None):
        return None
    with open(args.physical, encoding='utf-8') as f:
        return json.load(f)


def save(model, path):
    save_model(model, path)


def parse_capacity(text):
    if '=' not in text:
        return int(text)
    return {k.strip(): int(v) for k, v in (item.split('=', 1) for item in text.split(','))}


def cmd_info(args):
    g = graph_mod.load(args.model)
    prims = registry.load_all()
    em = Emitter(g, None, prims, target=args.target)
    r = em.refusals(em.evaluable(g.required_inputs()))
    o, v, t, s, e, _ = g.counts()
    print(f"{g.model}: {o} instances, {v} values, {t} parameter tensors, {s} states, {e} edges")
    if r:
        print(f"  refusals: {len(r)}")
        for line in r[:20]:
            print("    " + line)
        return 1
    print("  can be emitted")
    return 0


def cmd_emit(args):
    g = graph_mod.load(args.model)
    emitter = build(args, g)
    if emitter is None:
        return 1
    delivered = delivery(args, g)
    t0 = time.time()
    try:
        model = emitter.emit(delivered, dump=args.dump)
    except Refusal as e:
        print(f"refused: {e}")
        return 1
    describe(model, delivered, time.time() - t0)
    save(model, args.out)
    print(f"  -> {args.out}")
    return 0


def cmd_run(args):
    g = graph_mod.load(args.model)
    emitter = build(args, g)
    if emitter is None:
        return 1
    prompts = [[int(x) for x in text.split(',')] for text in args.ids] if args.ids else [[1]]
    if args.batch == 'aligned':
        return run_batch(args, g, emitter, prompts)
    if len(prompts) > 1:
        print(f"refused: {len(prompts)} prompts are several sessions; --batch aligned runs them as one batch")
        return 1
    session = Session(emitter, g, dump=bool(args.dump))
    ids = prompts[0]
    extra = load_inputs(args.input)
    t0 = time.time()
    if g.generative is None:
        out = session.run({g.token_input: np.asarray(ids, dtype=np.int64), **extra})
        for name, t in out.items():
            print(f"  {name}: {list(t.shape)} ({time.time() - t0:.1f}s)")
        return 0
    out = session.prefill(ids, inputs=extra)
    dump = {k: v for k, v in out.items()} if args.dump else None
    nxt = greedy(out, g)
    for delivered, model in session.models.items():
        describe(model, delivered, 0.0)
        if args.out:
            save(model, args.out)
    print(f"  prefill {len(ids)} elements -> next {nxt} ({time.time() - t0:.1f}s)")
    tokens = [nxt]
    for _ in range(args.steps):
        t0 = time.time()
        out = session.decode(nxt)
        nxt = greedy(out, g)
        tokens.append(nxt)
        print(f"  decode -> {nxt} ({time.time() - t0:.2f}s)")
    print("tokens:", tokens)
    if args.dump:
        write_dump(args.dump, g, session, dump, ids, tokens, extra, args.compute)
        print(f"  dumped -> {args.dump}")
    return 0


def run_batch(args, g, emitter, prompts):
    """Several prompts of one length as one batch of sessions on the aligned layout (B04): prefilled
    together, decoded together for --steps, each session's tokens on its own line."""
    if args.dump or args.input:
        print("refused: a batch takes token prompts alone — no --dump or --input")
        return 1
    if any(len(p) != len(prompts[0]) for p in prompts):
        print(f"refused: prompts of {[len(p) for p in prompts]} tokens — an aligned batch's sessions deliver the same count; "
              "prefill unequal prompts apart, or use the reference's packed layout")
        return 1
    batch = Batch(emitter, g, len(prompts), args.capacity)
    print(f"  batch: {len(prompts)} sessions, aligned layout, capacity {args.capacity}")
    t0 = time.time()
    if g.generative is None:
        outs = batch.run([{g.token_input: np.asarray(p, dtype=np.int64)} for p in prompts])
        for k, out in enumerate(outs):
            for name, t in out.items():
                print(f"  session {k} {name}: {list(t.shape)}")
        print(f"  ({time.time() - t0:.1f}s)")
        return 0
    outs = batch.prefill(prompts)
    nxt = [greedy(o, g) for o in outs]
    for delivered, model in batch.models.items():
        describe(model, delivered, 0.0)
        if args.out:
            save(model, args.out)
    print(f"  prefill {len(prompts[0])} elements x {len(prompts)} -> next {nxt} ({time.time() - t0:.1f}s)")
    tokens = [[n] for n in nxt]
    for _ in range(args.steps):
        t0 = time.time()
        outs = batch.decode(nxt)
        nxt = [greedy(o, g) for o in outs]
        for t, n in zip(tokens, nxt):
            t.append(n)
        print(f"  decode -> {nxt} ({time.time() - t0:.2f}s)")
    for k, t in enumerate(tokens):
        print(f"tokens[{k}]:", t)
    return 0


def load_inputs(specs):
    from safetensors.numpy import load_file
    out = {}
    for spec in specs or []:
        name, rest = spec.split('=', 1)
        file, key = rest.rsplit(':', 1) if ':' in rest else (rest, f"in/{name}")
        out[name] = load_file(file)[key]
    return out


def write_dump(path, g, session, prefill_outputs, ids, tokens, extra, compute):
    """A dump in the reference's form: the values at every layer graph_split and the states after the prefill."""
    from safetensors.numpy import save_file
    tensors = {}
    for name, t in prefill_outputs.items():
        tensors[f"value/{name}"] = np.ascontiguousarray(t.astype(np.float32))
    for key, buf in session.states_after_prefill.items():       # the states as the prefill left them
        tensors[f"state/{key}"] = np.ascontiguousarray(buf.astype(np.float32))
    o = g.interfaces['outputs'][g.generative[0]]
    logits = prefill_outputs[f"{o['node']}.{o['port']}"]
    tensors['logits/last'] = np.ascontiguousarray(logits[-1].astype(np.float32))
    tensors['logits/argmax'] = np.ascontiguousarray(logits.argmax(-1).astype(np.int64))
    for name, t in extra.items():
        tensors[f"in/{name}"] = np.ascontiguousarray(t)
    save_file(tensors, path, metadata={k: json.dumps(v) for k, v in {'model': g.model, 'ids': ids, 'tokens': tokens, 'compute': compute}.items()})


def manifest():
    import datetime
    import subprocess
    every = registry.load_all()
    prims = every['onnx']                 # the portable forms: what the generator can run; a target's fused forms cover the same branches
    try:
        version = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=HERE, text=True).strip()
    except Exception:  # noqa: BLE001
        version = 'unknown'
    primitives = {}
    for (name, ver), p in sorted(prims.items()):
        cap = dict(p.CAPABILITIES)
        entry = {'arguments': cap['arguments'], 'states': list(cap.get('states', []))}
        for key in ('excluding', 'conditions', 'transforms', 'notes'):
            if cap.get(key):
                entry[key] = list(cap[key])
        for target, table in every.items():
            fused = table.get((name, ver))
            if target != 'onnx' and fused is not p:
                entry.setdefault('notes', []).extend(f"target {target}: {note}" for note in fused.CAPABILITIES.get('notes', []))
        primitives[f"{name}@{ver}"] = entry
    return {'schema': 'tensorspine-capabilities/1',
            'generator': {'name': 'onnx', 'version': version, 'generator': 'generators/onnx/tsonnx.py capabilities',
                          'generated': datetime.date.today().isoformat()},
            'compute_dtypes': ['f32'],
            'parameter_dtypes': ['bf16', 'f16', 'f32'],
            'state_evolutions': ['append'], 'access': ['logical_position'],
            'sharing': [], 'partition_options': [],
            'domains': {'kinds': ['token'], 'transforms': [], 'fragmented': False},
            'sessions_per_invocation': session_mod.SESSIONS_PER_INVOCATION,     # the aligned layout (B04); what Batch enforces
            'locations': list(loader.FORMS),
            'primitives': primitives}


def cmd_capabilities(args):
    m = manifest()
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(m, f, indent=2)
        f.write('\n')
    print(f"{len(m['primitives'])} primitives -> {args.out}")
    if args.check:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'tools'))
        import capabilities
        return capabilities.run(args.out, [])
    return 0


def common(p):
    p.add_argument('model', help='a derived document')
    p.add_argument('--compute', default='f32', choices=['f32'])
    p.add_argument('--target', default='onnx', choices=registry.targets(),
                   help="the runtime the graph is emitted for: onnx, standard operators only (default); onnxruntime, its fused operators where a primitive has them")
    p.add_argument('--inputs', help='the public inputs delivered, comma-separated (default: the token input)')
    p.add_argument('--batch', default='none', choices=['none', 'aligned'],
                   help="the layout several sessions share an invocation in: none (one session, default); aligned — a batch axis, "
                        "every session delivering the same count per invocation, the append states buffers of --capacity positions")
    p.add_argument('--capacity', type=parse_capacity, default=1024, metavar='N|STREAM=N,…',
                   help='aligned layout: positions every append state may hold, for every stream or per stream')
    p.add_argument('--physical', metavar='FILE', help='opaque parameters for the primitives (generators/CAPABILITIES.md)')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('info'); common(p); p.set_defaults(fn=cmd_info)
    p = sub.add_parser('emit'); common(p)
    p.add_argument('--checkpoint', metavar='DIR'); p.add_argument('--random', action='store_true'); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dump', action='store_true', help='make every value crossing a layer graph_split a graph output')
    p.add_argument('--out', required=True); p.set_defaults(fn=cmd_emit)
    p = sub.add_parser('run'); common(p)
    p.add_argument('--checkpoint', metavar='DIR'); p.add_argument('--random', action='store_true'); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--ids', action='append', help='comma-separated token ids of the prompt; repeated, one session per prompt, under --batch aligned')
    p.add_argument('--steps', type=int, default=4)
    p.add_argument('--input', action='append', default=[], metavar='NAME=FILE[:KEY]')
    p.add_argument('--dump', help='write the values at every layer graph_split and the logits of the prefill to this safetensors file')
    p.add_argument('--out', help='also save the emitted model')
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser('capabilities')
    p.add_argument('--out', default=os.path.join(HERE, 'capabilities.json'))
    p.add_argument('--check', action='store_true')
    p.set_defaults(fn=cmd_capabilities)
    args = ap.parse_args(argv)
    if getattr(args, 'command', None) in ('emit', 'run') and not args.random and not args.checkpoint:
        print("refused: give --checkpoint DIR or --random")
        return 1
    return args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
