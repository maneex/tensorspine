#!/usr/bin/env python3
"""The reference backend's command line.

    ref.py info MODEL [--capacity N] [--compute f32|bf16] [--device cpu|cuda]
    ref.py run  MODEL --random --ids 1,2,3 [--steps N] [--dump F] [--truncate decoder.layer=3]

MODEL is a model document (derived here) or a derived document.
"""
import argparse
import json
import os
import sys
import tempfile
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import graph as graph_mod      # noqa: E402
import loader                  # noqa: E402
import registry                # noqa: E402
from compare import write_dump  # noqa: E402
from module import TensorspineModel  # noqa: E402
from plan import Plan          # noqa: E402
from session import Session, greedy  # noqa: E402

COMPUTE = {'f32': torch.float32, 'bf16': torch.bfloat16, 'f16': torch.float16}


def common(p):
    p.add_argument('model')
    p.add_argument('--truncate', help='shorten one composition index range, e.g. decoder.layer=3')
    p.add_argument('--set', action='append', default=[], metavar='PATH=VALUE',
                   help='edit the document before deriving, e.g. quantities.d.source.value=64 (JSON value)')
    p.add_argument('--capacity', type=int, default=1024)
    p.add_argument('--device', default='cpu')
    p.add_argument('--compute', default=None, help='f32 (CPU default) | bf16 (CUDA default)')


def open_graph(args):
    path = args.model
    if args.truncate or args.set:
        tmp = tempfile.mkdtemp(prefix='tensorspine-ref-')
        if args.truncate:
            path, notes = graph_mod.truncated(path, args.truncate, tmp)
            for n in notes:
                print(f"  edited: {n}")
        if args.set:
            edits = {}
            for item in args.set:
                k, v = item.split('=', 1)
                edits[k] = json.loads(v)
            path, notes = graph_mod.edited(path, edits, tmp, suffix='set')
            for n in notes:
                print(f"  edited: {n}")
    return graph_mod.load(path)


def compute_dtype(args):
    if args.compute:
        return COMPUTE[args.compute]
    return torch.bfloat16 if str(args.device).startswith('cuda') else torch.float32


def cmd_info(args):
    g = open_graph(args)
    dtype = compute_dtype(args)
    i = loader.info(g, args.capacity, dtype, args.device)
    print(f"{g.model}: {len(g.nodes)} nodes, {len(g.tensors)} tensors, {len(g.states)} states")
    print(f"  parameters {loader.gib(i['parameter_bytes'])} at the declared dtypes")
    print(f"  states     {loader.gib(i['state_bytes'])} at {dtype} for a capacity of {args.capacity} "
          f"({i['append_bytes_per_position']} B per cached position declared)")
    print(f"  total      {loader.gib(i['total_bytes'])}; free on {args.device}: "
          f"{loader.gib(i['free_bytes']) if i['free_bytes'] is not None else 'unknown'}")
    print("  activations are not budgeted")
    kernels = registry.load_kernels()
    r = registry.refusals(g, kernels)
    print(f"  refusals: {len(r)}")
    for line in r[:20]:
        print("    " + line)
    return 1 if r else 0


def cmd_compare(args):
    from compare import compare, read_dump
    ours, ho = read_dump(args.ours)
    theirs, ht = read_dump(args.theirs)
    rows, failures, only = compare(ours, theirs, args.atol, args.rtol)
    print(f"{len(rows)} keys compared (atol {args.atol}, rtol {args.rtol}); tokens ours {ho.get('tokens')} theirs {ht.get('tokens')}")
    for key, mabs, mrel, note in rows:
        print(f"  {key:60s} " + (f"max|d| {mabs:.3e}  max rel {mrel:.3e}  {note}" if mabs is not None else note))
    for k in only[:10]:
        print(f"  only on one side: {k}")
    print("compare: within tolerance" if not failures else f"compare: {failures} key(s) exceed")
    return 1 if failures else 0


def compiled(model, args):
    if not getattr(args, 'compile', False):
        return None
    model.static = True                        # masked attention over the whole capacity
    t0 = time.time()
    c = torch.compile(model, dynamic=False)
    print(f"  decode step compiled lazily (torch.compile; first decode pays for it)")
    return c


def build(args, g):
    dtype = compute_dtype(args)
    kernels = registry.load_kernels()
    r = registry.refusals(g, kernels)
    if r:
        print(f"refused: {len(r)} reason(s)")
        for line in r[:20]:
            print("  " + line)
        return None
    i = loader.info(g, args.capacity, dtype, args.device)
    if i['free_bytes'] is not None and i['total_bytes'] > i['free_bytes']:
        print(f"refused: {loader.gib(i['total_bytes'])} needed, {loader.gib(i['free_bytes'])} free on {args.device}")
        return None
    errors, advisories, stats = loader.verify(g, args.checkpoint)
    if errors:
        print(f"refused: {len(errors)} V17 error(s) against {args.checkpoint}")
        for line in errors[:10]:
            print("  " + line)
        return None
    print(f"  verified {stats['located']} located tensors against {stats['physical']} physical ({stats['unnamed']} unnamed)")
    params = loader.load_parameters(g, args.checkpoint, args.device)
    return TensorspineModel(g, Plan(g, kernels), params, dtype, args.device)


def cmd_chat(args):
    from chat import chat
    g = open_graph(args)
    model = build(args, g)
    if model is None:
        return 1
    dtype = compute_dtype(args)
    return chat(model, g, args.checkpoint, args.capacity, args.device, dtype, args.max_new_tokens,
                args.temperature, args.top_p, args.seed, decode_model=compiled(model, args))


def cmd_verify(args):
    g = open_graph(args)
    errors, advisories, stats = loader.verify(g, args.checkpoint)
    print(f"{g.model}: {stats['located']} located, {stats['physical']} physical, {stats['unnamed']} named by no location")
    for line in errors[:20]:
        print("  [V17] " + line)
    for line in advisories[:10]:
        print("  advice: " + line)
    return 1 if errors else 0


def cmd_run(args):
    g = open_graph(args)
    dtype = compute_dtype(args)
    kernels = registry.load_kernels()
    r = registry.refusals(g, kernels)
    if r:
        print(f"refused: {len(r)} reason(s)")
        for line in r[:20]:
            print("  " + line)
        return 1
    i = loader.info(g, args.capacity, dtype, args.device)
    if i['free_bytes'] is not None and i['total_bytes'] > i['free_bytes']:
        print(f"refused: {loader.gib(i['total_bytes'])} needed, {loader.gib(i['free_bytes'])} free on {args.device}")
        return 1
    if not args.random and not args.checkpoint:
        print("refused: give --checkpoint DIR (the document locates its weights) or --random")
        return 1
    t0 = time.time()
    if args.random:
        params = loader.random_parameters(g, args.device, args.seed)
    else:
        errors, advisories, stats = loader.verify(g, args.checkpoint)
        if errors:
            print(f"refused: {len(errors)} V17 error(s) against {args.checkpoint}")
            for line in errors[:10]:
                print("  " + line)
            return 1
        print(f"  verified {stats['located']} located tensors against {stats['physical']} physical "
              f"({stats['unnamed']} unnamed)")
        params = loader.load_parameters(g, args.checkpoint, args.device)
    plan = Plan(g, kernels)
    model = TensorspineModel(g, plan, params, dtype, args.device)
    session = Session(model, args.capacity, args.device, dtype, decode_model=compiled(model, args))
    print(f"{g.model}: {len(plan.steps)} steps, {len(params)} tensors, {len(session.states)} states, "
          f"{'random parameters' if args.random else 'loaded from ' + args.checkpoint}, "
          f"{loader.gib(i['parameter_bytes'])} ({time.time() - t0:.1f}s)")
    ids = [int(x) for x in args.ids.split(',')] if args.ids else [1]
    dump = {} if args.dump else None
    t0 = time.time()
    out = session.prefill(ids, dump)
    first_logits = out[g.generative[0]]
    nxt = greedy(out, g)
    print(f"  prefill {len(ids)} elements -> next {nxt} ({time.time() - t0:.1f}s)")
    if dump is not None:                          # the states as prefill left them (R07)
        for ident, st in session.states.items():
            bufs, length = st.read()
            for c, buf in bufs.items():
                dump[f"state/{ident}/{c}"] = (buf[:length] if length is not None else buf).detach().to('cpu', torch.float32)
    tokens = [nxt]
    for _ in range(args.steps):
        t0 = time.time()
        out = session.decode(nxt)
        nxt = greedy(out, g)
        tokens.append(nxt)
        print(f"  decode -> {nxt} ({time.time() - t0:.2f}s)")
    print("tokens:", tokens)
    if args.dump:
        dump['logits/last'] = first_logits[-1].detach().to('cpu', torch.float32)
        dump['logits/argmax'] = first_logits.argmax(-1).detach().cpu()
        write_dump(args.dump, dump, {'model': g.model, 'ids': ids, 'tokens': tokens, 'capacity': args.capacity,
                                     'compute': str(dtype), 'random_seed': args.seed if args.random else None,
                                     'cuts': [c['cut'] for c in g.layer_cuts()]})
        print(f"  dumped {len(dump)} tensors -> {args.dump}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('info'); common(p); p.set_defaults(fn=cmd_info)
    p = sub.add_parser('compare')
    p.add_argument('ours'); p.add_argument('theirs')
    p.add_argument('--atol', type=float, default=1e-3); p.add_argument('--rtol', type=float, default=1e-2)
    p.set_defaults(fn=cmd_compare)
    p = sub.add_parser('verify'); common(p)
    p.add_argument('--checkpoint', metavar='DIR', required=True)
    p.set_defaults(fn=cmd_verify)
    p = sub.add_parser('run'); common(p)
    p.add_argument('--random', action='store_true', help='parameters drawn from the D3 shapes, no checkpoint')
    p.add_argument('--checkpoint', metavar='DIR', help='the safetensors checkpoint the document locates its weights in')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--ids', help='comma-separated token ids of the prompt')
    p.add_argument('--steps', type=int, default=4)
    p.add_argument('--dump', help='write the values at every layer cut and the states to this safetensors file')
    p.add_argument('--compile', action='store_true', help='torch.compile the decode step (prefill stays eager)')
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser('chat'); common(p)
    p.add_argument('--checkpoint', metavar='DIR', required=True)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--top-p', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--compile', action='store_true')
    p.set_defaults(fn=cmd_chat)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
