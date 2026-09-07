"""residual.add@2.0.0 — a + b."""
from primitives._common import supports_from

PRIMITIVE = ("residual.add", "2.0.0")
CAPABILITIES = {"arguments": {"width": "any"}, "states": []}


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)


def emit(ctx, arguments, inputs, params, states):
    return {'output': ctx.b.node('Add', [inputs['a'], inputs['b']], hint=f"{ctx.node}.add")}
