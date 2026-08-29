"""lm_head@1.0.0 — one logit per vocabulary entry."""
from kernels._common import refuse_unknown, w

CONTRACT = ("lm_head", "1.0.0")
KNOWN = {'width', 'vocabulary'}


def supports(arguments):
    reasons = []
    refuse_unknown(arguments, KNOWN, reasons)
    return reasons


def run(ctx, arguments, inputs, params, states):
    return {'logits': inputs['input'] @ w(ctx, params['weight']).T}
