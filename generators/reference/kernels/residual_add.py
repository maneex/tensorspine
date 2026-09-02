"""residual.add@1.0.0 — a + b."""
from kernels._common import refuse_unknown, supports_from

CONTRACT = ("residual.add", "1.0.0")


CAPABILITIES = {"arguments": {"width": "any"}, "states": []}


# What a conformer must meet against this kernel's unit fixtures, per compute dtype (§4.2):
# `|a − b| ≤ atol + rtol·|b|`. The manifest's witness block is written from it.
TOLERANCE = {'f32': {'atol': 1e-6, 'rtol': 1e-5}, 'bf16': {'atol': 1e-2, 'rtol': 1e-2}}


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)

def run(ctx, arguments, inputs, params, states, physical=None):
    return {'output': inputs['a'] + inputs['b']}
