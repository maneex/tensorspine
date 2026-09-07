"""lm_head@1.0.0 — one logit per vocabulary entry; `softcap` refused."""
from primitives._common import linear, supports_from

PRIMITIVE = ("lm_head", "1.0.0")
CAPABILITIES = {"arguments": {"width": "any", "vocabulary": "any", "softcap": "absent"}, "states": []}


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)


def emit(ctx, arguments, inputs, params, states):
    return {'logits': linear(ctx, inputs['input'], ctx.param(params['weight']))}
