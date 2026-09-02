"""embed@1.0.0 — one `width` vector per token identifier."""
from kernels._common import refuse_unknown, supports_from

CONTRACT = ("embed", "1.0.0")


CAPABILITIES = {"arguments": {"width": "any", "vocabulary": "any"}, "states": []}


# What a conformer must meet against this kernel's unit fixtures, per compute dtype (§4.2):
# `|a − b| ≤ atol + rtol·|b|`. The manifest's witness block is written from it.
TOLERANCE = {'f32': {'atol': 1e-6, 'rtol': 1e-5}, 'bf16': {'atol': 1e-2, 'rtol': 1e-2}}


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)

def run(ctx, arguments, inputs, params, states, physical=None):
    ids = inputs['tokens']
    return {'output': params['weight'][ids].to(ctx.dtype)}      # gather first: no upcast of the whole table
