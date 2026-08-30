"""embed@1.0.0 — one `width` vector per token identifier."""
from kernels._common import refuse_unknown, supports_from

CONTRACT = ("embed", "1.0.0")


CAPABILITIES = {"arguments": {"width": "any", "vocabulary": "any"}, "states": []}


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)

def run(ctx, arguments, inputs, params, states, physical=None):
    ids = inputs['tokens']
    return {'output': params['weight'][ids].to(ctx.dtype)}      # gather first: no upcast of the whole table
