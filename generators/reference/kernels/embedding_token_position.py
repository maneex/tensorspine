"""embedding.token_position@1.0.0 — Whisper's decoder embedding: the token's row of `weight`
plus the row of `position` for its stream position (0 upwards).

| branch / record          | status      |
|--------------------------|-------------|
| token + position sum     | implemented |

A position at or beyond `positions` has no row and is refused at run time, as the delivery
implementation would index out of its table.
"""
from kernels._common import supports_from

CONTRACT = ("embedding.token_position", "1.0.0")


CAPABILITIES = {"arguments": {"width": "any", "vocabulary": "any", "positions": "any"}, "states": []}


# What a conformer must meet against this kernel's unit fixtures, per compute dtype (§4.2):
# `|a − b| ≤ atol + rtol·|b|`. The manifest's witness block is written from it.
TOLERANCE = {'f32': {'atol': 1e-6, 'rtol': 1e-5}, 'bf16': {'atol': 1e-2, 'rtol': 1e-2}}

# The unit fixtures this kernel produces (docs/TENSORSPINE-FIXTURE.md): one case per branch
# worth its own evidence, at small quantities; two invocations, so the positions continue.
FIXTURES = [
    {"case": "basic", "seed": 23, "invocations": [{"tokens": 6}, {"tokens": 2}],
     "arguments": {"width": 64, "vocabulary": 256, "positions": 64}},
]


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)


def run(ctx, arguments, inputs, params, states, physical=None):
    ids = inputs['tokens']
    pos = ctx.positions
    if pos.numel() and int(pos.max()) >= arguments['positions']:
        raise ValueError(f"position {int(pos.max())} has no row: the table holds {arguments['positions']} positions")
    return {'output': params['weight'][ids].to(ctx.dtype) + params['position'][pos].to(ctx.dtype)}   # gathers: no upcast of a table
