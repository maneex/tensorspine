"""lm_head@1.0.0 — one logit per vocabulary entry."""
from kernels._common import chunked_matmul, refuse_unknown

CONTRACT = ("lm_head", "1.0.0")
KNOWN = {'width', 'vocabulary'}


def supports(arguments):
    reasons = []
    refuse_unknown(arguments, KNOWN, reasons)
    return reasons


def run(ctx, arguments, inputs, params, states):
    return {'logits': chunked_matmul(ctx, inputs['input'], params['weight'])}   # the head upcast in bounded chunks
