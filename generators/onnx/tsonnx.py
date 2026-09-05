#!/usr/bin/env python3
"""The ONNX generator's command line.

    tsonnx.py info    DERIVED                               counts, and the refusals over the occurrences a delivery evaluates
    tsonnx.py emit    DERIVED --checkpoint DIR --out F.onnx [--dump] [--inputs a,b]   the graph of one invocation
    tsonnx.py run     DERIVED --checkpoint DIR --ids 1,2,3 [--steps N] [--dump F]     prefill and greedy decode through onnxruntime
    tsonnx.py run     DERIVED --random [--seed N] …                                  parameters drawn from the D3 shapes
    tsonnx.py capabilities [--out FILE] [--check]                                    the manifest, from the emitters' tables

DERIVED is a derived document (`tensorspine --derive MODEL -o DIR`): the generator reads D1–D6 and
the checkpoint, never the model source or the catalog. See generators/onnx/README.md.
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
from session import Session, greedy  # noqa: E402


def delivery(args, g):
    if args.inputs:
        return set(args.inputs.split(',')) | ({g.feedback_input} if g.feedback_input else set())
    return {g.feedback_input} if g.feedback_input else set(g.interfaces['inputs'])


def build(args, g):
    """The emitter over the checkpoint (V17 checked first) or random parameters; None on a refusal."""
    prims = registry.load_primitives()
    em = Emitter(g, None, prims)
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
    return Emitter(g, source, prims, args.compute, physical_of(args))


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


def cmd_info(args):
    g = graph_mod.load(args.model)
    prims = registry.load_primitives()
    em = Emitter(g, None, prims)
    r = em.refusals(em.evaluable(g.required_inputs()))
    o, v, t, s, e, _ = g.counts()
    print(f"{g.model}: {o} occurrences, {v} values, {t} parameter tensors, {s} states, {e} edges")
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
    session = Session(emitter, g, dump=bool(args.dump))
    ids = [int(x) for x in args.ids.split(',')] if args.ids else [1]
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


def load_inputs(specs):
    from safetensors.numpy import load_file
    out = {}
    for spec in specs or []:
        name, rest = spec.split('=', 1)
        file, key = rest.rsplit(':', 1) if ':' in rest else (rest, f"in/{name}")
        out[name] = load_file(file)[key]
    return out


def write_dump(path, g, session, prefill_outputs, ids, tokens, extra, compute):
    """A dump in the reference's form: the values at every layer cut and the states after the prefill."""
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
    prims = registry.load_primitives()
    try:
        version = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=HERE, text=True).strip()
    except Exception:  # noqa: BLE001
        version = 'unknown'
    contracts = {}
    for (name, ver), p in sorted(prims.items()):
        cap = dict(p.CAPABILITIES)
        entry = {'arguments': cap['arguments'], 'states': list(cap.get('states', []))}
        for key in ('excluding', 'transforms', 'notes'):
            if cap.get(key):
                entry[key] = list(cap[key])
        contracts[f"{name}@{ver}"] = entry
    return {'schema': 'tensorspine-capabilities/1',
            'generator': {'name': 'onnx', 'version': version, 'generator': 'generators/onnx/tsonnx.py capabilities',
                          'generated': datetime.date.today().isoformat()},
            'compute_dtypes': ['f32'],
            'parameter_dtypes': ['bf16', 'f16', 'f32'],
            'state_laws': ['append'], 'access': ['logical_position'],
            'sharing': [], 'partitions': [],
            'domains': {'kinds': ['token'], 'transforms': [], 'fragmented': False},
            'sessions_per_invocation': 1,
            'locations': list(loader.FORMS),
            'contracts': contracts}


def cmd_capabilities(args):
    m = manifest()
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(m, f, indent=2)
        f.write('\n')
    print(f"{len(m['contracts'])} contracts -> {args.out}")
    if args.check:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'tools'))
        import capabilities
        return capabilities.run(args.out, [])
    return 0


def common(p):
    p.add_argument('model', help='a derived document')
    p.add_argument('--compute', default='f32', choices=['f32'])
    p.add_argument('--inputs', help='the public inputs delivered, comma-separated (default: the token input)')
    p.add_argument('--physical', metavar='FILE', help='opaque parameters for the primitives (generators/CAPABILITIES.md)')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('info'); common(p); p.set_defaults(fn=cmd_info)
    p = sub.add_parser('emit'); common(p)
    p.add_argument('--checkpoint', metavar='DIR'); p.add_argument('--random', action='store_true'); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dump', action='store_true', help='make every value crossing a layer cut a graph output')
    p.add_argument('--out', required=True); p.set_defaults(fn=cmd_emit)
    p = sub.add_parser('run'); common(p)
    p.add_argument('--checkpoint', metavar='DIR'); p.add_argument('--random', action='store_true'); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--ids'); p.add_argument('--steps', type=int, default=4)
    p.add_argument('--input', action='append', default=[], metavar='NAME=FILE[:KEY]')
    p.add_argument('--dump', help='write the values at every layer cut and the logits of the prefill to this safetensors file')
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
