"""The model as a torch.nn.Module (R03): parameters registered per D3 identity — or
materialised block by block from a source under `--max-ram` (R13) — and `forward`
walking the plan's blocks and calling the kernels; `step` does the work with the
states passed explicitly.
"""
from fractions import Fraction

import torch
import torch.nn as nn

ESCAPE = {'.': '__', '/': '_S_', '[': '_L_', ']': '_R_', '=': '_E_', ',': '_C_', '-': '_D_'}


def escape(identity):
    return ''.join(ESCAPE.get(c, c) for c in identity)


class ShapeError(Exception):
    pass


class Ctx:
    def __init__(self, dtype, device, static=False):
        self.dtype = dtype
        self.device = device
        self.static = static
        self.positions = None


def elements(n, count):
    """`n` elements of a stream seen through a D2 `count`: n·count, an integer because a delivery
    is aligned to every merge on its stream (§5.3, D2 `fragment_alignment`); an unaligned
    delivery is refused, never rounded."""
    m = n * Fraction(count).limit_denominator(1 << 20)
    if m.denominator != 1:
        raise ShapeError(f"a delivery of {n} elements is not aligned: every fragment delivers a multiple of {m.denominator} (§5.3)")
    return int(m)


def scaled(positions, count):
    """The positions of a value with `count` elements per element of its stream: a delivery of n
    elements from stream position s is the positions s·count … (s + n)·count − 1 — §5.3's merge,
    applied where the runtime needs it (an encoder behind a strided front end works on n/stride
    positions for n frames). Every value delivered whole has count 1 and keeps its positions."""
    if positions is None or count == 1:
        return positions
    n = positions.shape[0]
    start = elements(int(positions[0]) if n else 0, count)
    return torch.arange(start, start + elements(n, count), device=positions.device)


def physical_for(physical, node, contract):
    """The opaque parameters addressed to an occurrence: by its exact identifier, by a site
    pattern where `*` alone is a wildcard (`decoder/attn[layer=*]`), or by its contract
    version; more specific entries override more general ones (contract < pattern < exact)."""
    import re
    if not physical:
        return None
    out = {}
    cid = f"{contract['name']}@{contract['version']}"
    for key, value in physical.items():
        if key == cid:
            out.update(value)
    for key, value in physical.items():
        if key != cid and key != node and '*' in key and re.fullmatch(re.escape(key).replace(r'\*', '.*'), node):
            out.update(value)
    if node in physical:
        out.update(physical[node])
    return out or None


def node_streams(graph, node):
    """The streams of the values feeding the node (D2), a public input's included."""
    streams = set()
    for (n, _port), vname in graph.sources.items():
        if n == node:
            streams.add(graph.values[vname]['domain']['stream'])
    for (n, _port), name in graph.fed_by_input.items():
        if n == node:
            streams.add(graph.input_values[name]['domain']['stream'])
    return streams


def evaluated_per_session(graph, step):
    """Whether a batch evaluates the occurrence per session (B03; harness guide §8): it reads
    across positions of its stream (D1's `across_positions`, the contract's condition evaluated on
    the occurrence's arguments — the kernels declare nothing), it carries a state (per session by
    its instance key, §4.4), or it reads values of several streams (a broadcast from a per-session
    value). A derived document without the field is refused: the split is read, never guessed
    from the states."""
    entry = graph.nodes[step.node]
    if 'across_positions' not in entry:
        raise ShapeError(f"{step.node}: the derived document states no across_positions — derive it again "
                         f"(tensorspine --derive MODEL -o DIR)")
    return bool(entry['across_positions'] or step.states or len(node_streams(graph, step.node)) > 1)


class TensorspineModel(nn.Module):
    def __init__(self, graph, plan, params=None, compute_dtype=torch.float32, device='cpu', source=None, physical=None):
        """`params`: every identity resident (one block). `source(identity) -> tensor` on the
        device: materialised per block and released after it (several blocks)."""
        super().__init__()
        self.graph = graph
        self.plan = plan
        self.keys = {ident: escape(ident) for ident in graph.tensors}
        self.params = nn.ParameterDict({self.keys[i]: nn.Parameter(t, requires_grad=False) for i, t in (params or {}).items()})
        self.source = source
        self.compute = compute_dtype
        self.device = device
        self.check = True          # every produced value against its D2 shape (eager)
        self.static = False        # masked attention over the whole capacity (compiled form)
        self.loaded_blocks = 0     # blocks materialised so far (the traffic, in blocks)
        self.physical = {s.node: physical_for(physical, s.node, s.contract) for s in plan.steps}
        self.per_session = {s.node: evaluated_per_session(graph, s) for s in plan.steps}   # a batch's split (B03)

    def block_params(self, block):
        if self.source is None:
            return {ident: self.params[self.keys[ident]] for ident in block.identities}
        return {ident: self.source(ident) for ident in block.identities}

    def release(self, block, params):
        if self.source is not None:
            params.clear()
            if str(self.device).startswith('cuda'):
                torch.cuda.empty_cache()

    def forward(self, inputs, positions, states, dump=None):
        return step(self, inputs, positions, states, dump)


def feed(model, s, inputs, values):
    """The tensors on a step's input ports: a delivered input, an evaluated value, or — nothing
    delivered — an empty value on its D2 shape."""
    graph = model.graph
    ins = {}
    for port, (kind, ref) in s.inputs.items():
        if kind == 'input' and ref in inputs:
            ins[port] = inputs[ref]
        elif kind == 'value' and ref in values:
            ins[port] = values[ref]
        else:
            shape = [a['extent'] for a in (graph.values.get(ref) or graph.input_values.get(ref) or {}).get('shape', [])]
            ins[port] = torch.empty((0, *shape), dtype=model.compute, device=model.device)
    return ins


def evaluate(model, s, ctx, ins, params, sts, stream_positions):
    """One occurrence: the kernel on its inputs, parameters and states, at its positions (the
    stream's, scaled by its D2 count, §5.3); every output checked against its D2 shape. On the
    packed layout `stream_positions` is a list, one entry per session: each session's positions
    are scaled on their own and then concatenated — a merge's block per session, never one arange
    across the sessions."""
    parts = stream_positions if isinstance(stream_positions, list) else [stream_positions]
    if parts[0] is None:
        ctx.positions, n = None, None
    else:
        ctx.positions = torch.cat([scaled(p, s.factor) for p in parts]) if len(parts) > 1 else scaled(parts[0], s.factor)
        n = sum(p.shape[0] for p in parts)
    rows = {} if n is None else {port: elements(n, s.counts[port]) for port in s.outputs}   # refused before the kernel runs
    outs = s.kernel.run(ctx, s.arguments, ins, params, sts, model.physical.get(s.node))
    for port, t in outs.items():
        if model.check and port in s.outputs:
            expect = [a['extent'] for a in s.outputs[port]['shape']]
            if list(t.shape[1:]) != expect or (port in rows and t.shape[0] != rows[port]):
                raise ShapeError(f"{s.node}.{port}: D2 says {expect} per element for {rows.get(port)} elements, got {list(t.shape)}")
    return outs


def consume(plan, s, values, remaining, needed):
    """A value read by its last consumer is released, unless it is exposed."""
    for port, (kind, ref) in s.inputs.items():
        if kind == 'value' and ref in values:
            remaining[ref] -= 1
            if remaining[ref] == 0 and ref not in needed:
                del values[ref]


def exposed(graph, values):
    return {name: values[f"{o['node']}.{o['port']}"] for name, o in graph.interfaces['outputs'].items()
            if f"{o['node']}.{o['port']}" in values}


def step(model, inputs, positions, states, dump=None):
    plan, graph = model.plan, model.graph
    needed = {f"{o['node']}.{o['port']}" for o in graph.interfaces['outputs'].values()}
    values = {}
    remaining = dict(plan.remaining)
    ctx = Ctx(model.compute, model.device, model.static)
    active = plan.evaluable(set(inputs), states)
    for block in plan.blocks:
        block_params = model.block_params(block)
        model.loaded_blocks += 1
        for si in block.steps:
            s = plan.steps[si]
            if s.node not in active:
                continue
            if s.kernel is None:
                raise ShapeError(f"{s.node}: no kernel for {s.contract['name']}@{s.contract['version']}, yet evaluated")
            ins = feed(model, s, inputs, values)
            params = {slot: block_params[ident] for slot, ident in s.params.items()}
            sts = {name: states[ident] for name, ident in s.states.items()}
            outs = evaluate(model, s, ctx, ins, params, sts, positions.get(s.stream) if s.stream else None)
            for port, t in outs.items():
                vname = f"{s.node}.{port}"
                values[vname] = t
                if dump is not None and vname in plan.dump_values:
                    dump[f"value/{vname}"] = t.detach().to('cpu', torch.float32).clone()
            consume(plan, s, values, remaining, needed)
        model.release(block, block_params)
    return exposed(graph, values)


def step_batch(model, inputs, positions, states):
    """One invocation for several sessions — the packed layout (B03): `inputs`, `positions` and
    `states` hold one entry per session. An occurrence the model evaluates per session
    (`per_session`: it reads across positions, it carries a state, or it reads several streams)
    runs on each session's elements and states in turn; every other occurrence runs once on the
    sessions' elements concatenated along the element axis — the language's own axis, its
    positions per element — and its outputs are split back, each session's rows being its
    elements of the stream through the value's D2 count (a merge, the temporal projector, makes
    fewer rows than it reads, and every aligned delivery keeps its groups inside a session,
    §5.3). A session gets what it would get alone, up to the rounding of a matrix product over
    more rows. The sessions must evaluate the same occurrences (§7: one delivery pattern per
    invocation)."""
    k = len(inputs)
    if k == 1:
        return [step(model, inputs[0], positions[0], states[0])]
    plan, graph = model.plan, model.graph
    needed = {f"{o['node']}.{o['port']}" for o in graph.interfaces['outputs'].values()}
    actives = [plan.evaluable(set(i), st) for i, st in zip(inputs, states)]
    for i, a in enumerate(actives[1:], 1):
        if a != actives[0]:
            raise ShapeError(f"a batch's sessions evaluate different occurrences (§7): session {i} differs on "
                             f"{sorted(a ^ actives[0])[:3]}")
    active = actives[0]
    values = [{} for _ in range(k)]
    remaining = [dict(plan.remaining) for _ in range(k)]
    ctx = Ctx(model.compute, model.device, False)
    for block in plan.blocks:
        block_params = model.block_params(block)
        model.loaded_blocks += 1
        for si in block.steps:
            s = plan.steps[si]
            if s.node not in active:
                continue
            if s.kernel is None:
                raise ShapeError(f"{s.node}: no kernel for {s.contract['name']}@{s.contract['version']}, yet evaluated")
            params = {slot: block_params[ident] for slot, ident in s.params.items()}
            per = [feed(model, s, inputs[i], values[i]) for i in range(k)]
            spos = [positions[i].get(s.stream) if s.stream else None for i in range(k)]
            if model.per_session[s.node]:
                outs = [evaluate(model, s, ctx, per[i], params, {name: states[i][ident] for name, ident in s.states.items()}, spos[i])
                        for i in range(k)]
            else:
                first = next(iter(per[0]), None)
                delivered = [per[i][first].shape[0] if first is not None else 0 for i in range(k)]
                joint = {port: torch.cat([per[i][port] for i in range(k)], dim=0) for port in per[0]}
                out = evaluate(model, s, ctx, joint, params, {}, None if spos[0] is None else spos)
                outs = [{} for _ in range(k)]
                for port, t in out.items():
                    rows = ([elements(spos[i].shape[0], s.counts[port]) for i in range(k)]
                            if spos[0] is not None and port in s.counts else delivered)
                    if t.shape[0] != sum(rows):
                        raise ShapeError(f"{s.node}.{port}: {t.shape[0]} rows for {rows} per session — not the per-element "
                                         f"occurrence the derived document describes")
                    for i, part in enumerate(torch.split(t, rows, dim=0)):
                        outs[i][port] = part
            for i in range(k):
                for port, t in outs[i].items():
                    values[i][f"{s.node}.{port}"] = t
                consume(plan, s, values[i], remaining[i], needed)
        model.release(block, block_params)
    return [exposed(graph, values[i]) for i in range(k)]
