"""residual.add@1.0.0 — a + b."""
from kernels._common import refuse_unknown, supports_from

CONTRACT = ("residual.add", "1.0.0")


CAPABILITIES = {"arguments": {"width": "any"}, "states": []}


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)

def run(ctx, arguments, inputs, params, states):
    return {'output': inputs['a'] + inputs['b']}
