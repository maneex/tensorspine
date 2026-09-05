"""lm_head@1.0.0 — one logit per vocabulary entry."""
from primitives._common import linear, supports_from

CONTRACT = ("lm_head", "1.0.0")
CAPABILITIES = {"arguments": {"width": "any", "vocabulary": "any"}, "states": []}


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)


def emit(ctx, arguments, inputs, params, states):
    return {'logits': linear(ctx, inputs['input'], ctx.param(params['weight']))}
