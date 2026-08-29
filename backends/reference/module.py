"""The model as a torch.nn.Module (R03): parameters registered per D3 identity,
`forward` walking the plan and calling the kernels; `step` does the work with
the states passed explicitly.
"""
import torch
import torch.nn as nn

ESCAPE = {'.': '__', '/': '_S_', '[': '_L_', ']': '_R_', '=': '_E_', ',': '_C_', '-': '_D_'}


def escape(identity):
    return ''.join(ESCAPE.get(c, c) for c in identity)


class ShapeError(Exception):
    pass


class Ctx:
    def __init__(self, dtype, device, static=False, eps=1e-6):
        self.dtype = dtype
        self.device = device
        self.static = static
        self.positions = None
        self.eps = eps            # finding: the Q/K normalisation has no `eps` of its own in the contract


class TensorspineModel(nn.Module):
    def __init__(self, graph, plan, params, compute_dtype, device):
        super().__init__()
        self.graph = graph
        self.plan = plan
        self.keys = {ident: escape(ident) for ident in graph.tensors}
        self.params = nn.ParameterDict({self.keys[i]: nn.Parameter(t, requires_grad=False) for i, t in params.items()})
        self.compute = compute_dtype
        self.device = device
        self.check = True          # every produced value against its D2 shape (eager)
        self.static = False        # masked attention over the whole capacity (compiled form)

    def forward(self, inputs, positions, states, dump=None):
        return step(self, inputs, positions, states, dump)


def step(model, inputs, positions, states, dump=None):
    plan, graph = model.plan, model.graph
    needed = {f"{o['node']}.{o['port']}" for o in graph.interfaces['outputs'].values()}
    values = {}
    remaining = dict(plan.remaining)
    ctx = Ctx(model.compute, model.device, model.static)
    for s in plan.steps:
        ins = {}
        for port, (kind, ref) in s.inputs.items():
            ins[port] = inputs[ref] if kind == 'input' else values[ref]
        params = {slot: model.params[model.keys[ident]] for slot, ident in s.params.items()}
        sts = {name: states[ident] for name, ident in s.states.items()}
        ctx.positions = positions.get(s.stream) if s.stream else None
        outs = s.kernel.run(ctx, s.arguments, ins, params, sts)
        n = None if ctx.positions is None else ctx.positions.shape[0]
        for port, t in outs.items():
            vname = f"{s.node}.{port}"
            if model.check and port in s.outputs:
                expect = [a['extent'] for a in s.outputs[port]['shape']]
                if list(t.shape[1:]) != expect or (n is not None and t.shape[0] != n):
                    raise ShapeError(f"{vname}: D2 says {expect} per element for {n} elements, got {list(t.shape)}")
            values[vname] = t
            if dump is not None and vname in plan.dump_values:
                dump[f"value/{vname}"] = t.detach().to('cpu', torch.float32).clone()
        for port, (kind, ref) in s.inputs.items():
            if kind == 'value':
                remaining[ref] -= 1
                if remaining[ref] == 0 and ref not in needed:
                    del values[ref]
    return {name: values[f"{o['node']}.{o['port']}"] for name, o in graph.interfaces['outputs'].items()}
