"""lm_head@1.0.0 — one logit per vocabulary entry."""
from kernels._common import chunked_matmul, refuse_unknown, supports_from

CONTRACT = ("lm_head", "1.0.0")


CAPABILITIES = {"arguments": {"width": "any", "vocabulary": "any"}, "states": []}


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)

def run(ctx, arguments, inputs, params, states):
    return {'logits': chunked_matmul(ctx, inputs['input'], params['weight'])}   # the head upcast in bounded chunks
