"""lm_head@1.0.0 — one logit per vocabulary entry."""
from kernels._common import chunked_matmul, refuse_unknown, supports_from

CONTRACT = ("lm_head", "1.0.0")


CAPABILITIES = {"arguments": {"width": "any", "vocabulary": "any"}, "states": []}


# What a conformer must meet against this kernel's unit fixtures, per compute dtype (§4.2):
# `|a − b| ≤ atol + rtol·|b|`. The manifest's witness block is written from it.
TOLERANCE = {'f32': {'atol': 1e-5, 'rtol': 1e-4}, 'bf16': {'atol': 1e-1, 'rtol': 2e-2}}


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)

def run(ctx, arguments, inputs, params, states, physical=None):
    return {'logits': chunked_matmul(ctx, inputs['input'], params['weight'])}   # the head upcast in bounded chunks
