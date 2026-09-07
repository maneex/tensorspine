"""residual.stream_expand@1.0.0 — one `width` vector becomes `streams` of them: the input itself,
then `streams - 1` projections of it, each rescaled to the input's root-mean-square magnitude
(the projected mean square floored at 1e-5 before the root), as Gemma 3n's AltUp does.

| branch / record | status      |
|-----------------|-------------|
| streams ≥ 2     | implemented |
"""
import torch
from kernels._common import refuse_unknown, supports_from, w

PRIMITIVE = ("residual.stream_expand", "1.0.0")


CAPABILITIES = {"arguments": {"width": "any", "streams": "any"}, "states": []}


# What a conformer must meet against this kernel's unit fixtures, per compute dtype (§4.2):
# `|a − b| ≤ atol + rtol·|b|`. The manifest's witness block is written from it.
TOLERANCE = {'f32': {'atol': 1e-05, 'rtol': 0.0001}, 'bf16': {'atol': 0.1, 'rtol': 0.02}}

# The unit fixtures this kernel produces (docs/TENSORSPINE-FIXTURE.md): the slot `projection` has a
# multiplicity (one matrix per auxiliary stream), stored whole as [streams − 1, width, width] — the
# storage axis of finding 30 — which the fixture, its own checkpoint, holds under one name.
FIXTURES = [
    {"case": "basic", "seed": 121, "invocations": [{"input": 5}, {"input": 2}], "arguments": {"width": 16, "streams": 3}},
]


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)

def run(ctx, arguments, inputs, params, states, physical=None):
    x = inputs['input']
    P = w(ctx, params['projection'])                                  # [streams - 1, width, width]
    target = x.to(torch.float32).pow(2).mean(-1, keepdim=True).sqrt()
    out = [x]
    for i in range(arguments['streams'] - 1):
        y = x @ P[i].T
        mag = y.to(torch.float32).pow(2).mean(-1, keepdim=True).clamp(min=1e-5).sqrt()
        out.append((y.to(torch.float32) * target / mag).to(x.dtype))
    return {'output': torch.stack(out, dim=1)}
