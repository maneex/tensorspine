#!/usr/bin/env python3
"""The reference backend's command line.

    ref.py info    MODEL [--capacity N]                      bytes from D3/D4, free memory, refusals
    ref.py verify  MODEL --checkpoint DIR                    V17 against the file headers; nothing is read
    ref.py run     MODEL --checkpoint DIR --ids 1,2,3 [--steps N] [--dump F] [--compile]
    ref.py run     MODEL --random [--seed N] …               parameters drawn from the D3 shapes
    ref.py chat    MODEL --checkpoint DIR [--max-new-tokens N] [--temperature T --top-p P --seed N]
    ref.py compare OURS THEIRS [--atol A --rtol R]           two dumps, at every cut and state

Common options: --device cpu|cuda[:i], --compute f32|bf16, --capacity N, --truncate decoder.layer=N,
--set path=value. MODEL is a model document (derived here) or a derived document. In a chat, an
empty line quits; the session persists across turns. See backends/reference/README.md.
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
    p.add_argument('--max-ram', type=float, default=None, metavar='GIB',
                   help='run in blocks of layers at legal cuts so that the parameters held at once, the '
                        'payload crossing into a block, the states and the largest temporary stay under this bound')


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


def make_plan(g, kernels, args, dtype):
    resident = loader.state_bytes(g, args.capacity, dtype) + loader.largest_temporary(g, dtype)
    max_bytes = int(args.max_ram * 2**30) if args.max_ram else None
    return Plan(g, kernels, max_bytes=max_bytes, elements=args.capacity, resident_bytes=resident)


def report(g, plan, i, args, dtype):
    print(f"  parameters {loader.gib(i['parameter_bytes'])} at the declared dtypes; states {loader.gib(i['state_bytes'])} "
          f"at {dtype} for a capacity of {args.capacity}; largest per-operation temporary {loader.gib(i['temporary_bytes'])}")
    print(f"  resident   {loader.gib(i['resident_bytes'])} — {i['mode']}; free on {args.device}: "
          f"{loader.gib(i['free_bytes']) if i['free_bytes'] is not None else 'unknown'}; activations not budgeted")
    if len(plan.blocks) > 1:
        print(f"  blocks     " + ", ".join(f"{b.name} {loader.gib(b.bytes)}" for b in plan.blocks[:8])
              + (" …" if len(plan.blocks) > 8 else ""))
        print(f"  traffic    {loader.gib(i['traffic_bytes'])} of parameters loaded and released per invocation")


def cmd_info(args):
    g = open_graph(args)
    dtype = compute_dtype(args)
    kernels = registry.load_kernels()
    r = registry.refusals(g, kernels)
    print(f"{g.model}: {len(g.nodes)} nodes, {len(g.tensors)} tensors, {len(g.states)} states")
    if r:
        print(f"  refusals: {len(r)}")
        for line in r[:20]:
            print("    " + line)
        return 1
    try:
        plan = make_plan(g, kernels, args, dtype)
    except ValueError as e:
        print(f"  refused: {e}")
        return 1
    report(g, plan, loader.info(g, args.capacity, dtype, args.device, plan, args.capacity), args, dtype)
    return 0


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
    try:
        plan = make_plan(g, kernels, args, dtype)
    except ValueError as e:
        print(f"refused: {e}")
        return None
    i = loader.info(g, args.capacity, dtype, args.device, plan, args.capacity)
    if i['free_bytes'] is not None and i['resident_bytes'] > i['free_bytes']:
        print(f"refused: {loader.gib(i['resident_bytes'])} resident ({i['mode']}), {loader.gib(i['free_bytes'])} free on {args.device}")
        return None
    errors, advisories, stats = loader.verify(g, args.checkpoint)
    if errors:
        print(f"refused: {len(errors)} V17 error(s) against {args.checkpoint}")
        for line in errors[:10]:
            print("  " + line)
        return None
    print(f"  verified {stats['located']} located tensors against {stats['physical']} physical ({stats['unnamed']} unnamed)")
    if len(plan.blocks) > 1:
        print(f"  {i['mode']}; traffic {loader.gib(i['traffic_bytes'])} of parameters per invocation")
        return TensorspineModel(g, plan, None, dtype, args.device, source=loader.Source(g, args.checkpoint, args.device).materialise)
    return TensorspineModel(g, plan, loader.load_parameters(g, args.checkpoint, args.device), dtype, args.device)


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
    if not args.random and not args.checkpoint:
        print("refused: give --checkpoint DIR (the document locates its weights) or --random")
        return 1
    try:
        plan = make_plan(g, kernels, args, dtype)
    except ValueError as e:
        print(f"refused: {e}")
        return 1
    i = loader.info(g, args.capacity, dtype, args.device, plan, args.capacity)
    if i['free_bytes'] is not None and i['resident_bytes'] > i['free_bytes']:
        print(f"refused: {loader.gib(i['resident_bytes'])} resident ({i['mode']}), {loader.gib(i['free_bytes'])} free on {args.device}")
        return 1
    t0 = time.time()
    blocks = len(plan.blocks) > 1
    if args.random:
        params = loader.random_parameters(g, args.device, args.seed)
        model = TensorspineModel(g, plan, None if blocks else params, dtype, args.device,
                                 source=loader.RandomSource(params).materialise if blocks else None)
    else:
        errors, advisories, stats = loader.verify(g, args.checkpoint)
        if errors:
            print(f"refused: {len(errors)} V17 error(s) against {args.checkpoint}")
            for line in errors[:10]:
                print("  " + line)
            return 1
        print(f"  verified {stats['located']} located tensors against {stats['physical']} physical "
              f"({stats['unnamed']} unnamed)")
        if blocks:
            model = TensorspineModel(g, plan, None, dtype, args.device, source=loader.Source(g, args.checkpoint, args.device).materialise)
        else:
            model = TensorspineModel(g, plan, loader.load_parameters(g, args.checkpoint, args.device), dtype, args.device)
    session = Session(model, args.capacity, args.device, dtype, decode_model=compiled(model, args))
    print(f"{g.model}: {len(plan.steps)} steps, {len(g.tensors)} tensors, {len(session.states)} states, "
          f"{'random parameters' if args.random else 'loaded from ' + args.checkpoint}, "
          f"{loader.gib(i['parameter_bytes'])}, {i['mode']} ({time.time() - t0:.1f}s)")
    if blocks:
        print(f"  traffic {loader.gib(i['traffic_bytes'])} of parameters per invocation")
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
                dump[f"state/{ident}/{c}"] = (buf[:length] if length is not None else buf).detach().to('cpu', torch.float32).clone()
    tokens = [nxt]
    for _ in range(args.steps):
        t0 = time.time()
        out = session.decode(nxt)
        nxt = greedy(out, g)
        tokens.append(nxt)
        print(f"  decode -> {nxt} ({time.time() - t0:.2f}s)")
    print("tokens:", tokens)
    if args.dump:
        dump['logits/last'] = first_logits[-1].detach().to('cpu', torch.float32).clone()
        dump['logits/argmax'] = first_logits.argmax(-1).detach().cpu().clone()
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
