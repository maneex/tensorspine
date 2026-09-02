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
            ins = {}
            for port, (kind, ref) in s.inputs.items():
                if kind == 'input' and ref in inputs:
                    ins[port] = inputs[ref]
                elif kind == 'value' and ref in values:
                    ins[port] = values[ref]
                else:                                       # nothing delivered: an empty value
                    shape = [a['extent'] for a in (graph.values.get(ref) or graph.input_values.get(ref) or {}).get('shape', [])]
                    ins[port] = torch.empty((0, *shape), dtype=model.compute, device=model.device)
            params = {slot: block_params[ident] for slot, ident in s.params.items()}
            sts = {name: states[ident] for name, ident in s.states.items()}
            stream_positions = positions.get(s.stream) if s.stream else None
            ctx.positions = scaled(stream_positions, s.factor)
            n = None if stream_positions is None else stream_positions.shape[0]
            rows = {} if n is None else {port: elements(n, s.counts[port]) for port in s.outputs}   # refused before the kernel runs
            outs = s.kernel.run(ctx, s.arguments, ins, params, sts, model.physical.get(s.node))
            for port, t in outs.items():
                vname = f"{s.node}.{port}"
                if model.check and port in s.outputs:
                    expect = [a['extent'] for a in s.outputs[port]['shape']]
                    if list(t.shape[1:]) != expect or (port in rows and t.shape[0] != rows[port]):
                        raise ShapeError(f"{vname}: D2 says {expect} per element for {rows.get(port)} elements, got {list(t.shape)}")
                values[vname] = t
                if dump is not None and vname in plan.dump_values:
                    dump[f"value/{vname}"] = t.detach().to('cpu', torch.float32).clone()
            for port, (kind, ref) in s.inputs.items():
                if kind == 'value' and ref in values:
                    remaining[ref] -= 1
                    if remaining[ref] == 0 and ref not in needed:
                        del values[ref]
        model.release(block, block_params)
    return {name: values[f"{o['node']}.{o['port']}"] for name, o in graph.interfaces['outputs'].items()
            if f"{o['node']}.{o['port']}" in values}
