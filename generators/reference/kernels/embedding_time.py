"""embedding.time@1.0.0 — the sinusoidal embedding of a count: `[cos(t · f), sin(t · f)]` with
`f_i = theta^(−i / (width / 2))`, the width/2 cosines first, then the sines (Voxtral Realtime's
time embedding of the number of delay tokens).

| branch / record | status      |
|-----------------|-------------|
| width, theta    | implemented |

The frequencies and the products are computed in f32 whatever the compute dtype, and the
embedding is delivered in the compute dtype; the count arrives as an integer tensor, one per
element of its stream (a `sequence` stream carries one).
"""
import torch
from kernels._common import supports_from

CONTRACT = ("embedding.time", "1.0.0")


CAPABILITIES = {"arguments": {"width": "any", "theta": "any"}, "states": []}


# What a conformer must meet against this kernel's unit fixtures, per compute dtype (§4.2):
# `|a − b| ≤ atol + rtol·|b|`. The manifest's witness block is written from it.
TOLERANCE = {'f32': {'atol': 1e-6, 'rtol': 1e-5}, 'bf16': {'atol': 1e-2, 'rtol': 1e-2}}

# The unit fixtures this kernel produces (docs/TENSORSPINE-FIXTURE.md): one case, one count.
FIXTURES = [
    {"case": "basic", "seed": 73, "invocations": [{"steps": 1}], "arguments": {"width": 64, "theta": 10000.0}},
]


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)


def embed(t, width, theta):
    """[n, width] for the counts t [n]: cosines then sines of t · f, in f32."""
    half = width // 2
    inv = theta ** (-torch.arange(half, device=t.device, dtype=torch.float32) / half)
    angles = t.to(torch.float32)[:, None] * inv[None, :]
    return torch.cat([angles.cos(), angles.sin()], dim=-1)


def run(ctx, arguments, inputs, params, states, physical=None):
    t = inputs['steps']
    return {'embedding': embed(t.reshape(-1), arguments['width'], float(arguments['theta'])).to(ctx.dtype)}
