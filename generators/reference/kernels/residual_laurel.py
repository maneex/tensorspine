"""residual.laurel@2.0.0 — output = input + norm(right(left(input))): a learned low-rank term,
RMS-normalized with `norm` and `eps`, added to the input.

| branch / record | status      |
|-----------------|-------------|
| rank, eps       | implemented |
"""
from kernels._common import refuse_unknown, rms_norm, supports_from, w

PRIMITIVE = ("residual.laurel", "2.0.0")


CAPABILITIES = {"arguments": {"width": "any", "rank": "any", "eps": "any"}, "states": []}


# What a conformer must meet against this kernel's unit fixtures, per compute dtype (§4.2):
# `|a − b| ≤ atol + rtol·|b|`. The manifest's witness block is written from it.
TOLERANCE = {'f32': {'atol': 1e-05, 'rtol': 0.0001}, 'bf16': {'atol': 0.1, 'rtol': 0.02}}

# The unit fixtures this kernel produces (docs/TENSORSPINE-FIXTURE.md): a rank-4 term on a width of 16.
FIXTURES = [
    {"case": "basic", "seed": 114, "invocations": [{"input": 5}, {"input": 3}], "arguments": {"width": 16, "rank": 4, "eps": 1e-6}},
]


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)

def run(ctx, arguments, inputs, params, states, physical=None):
    x = inputs['input']
    y = (x @ w(ctx, params['left']).T) @ w(ctx, params['right']).T
    return {'output': x + rms_norm(y, w(ctx, params['norm']), arguments['eps'])}
