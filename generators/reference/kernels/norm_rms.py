"""norm.rms@1.0.0 — x · rsqrt(mean(x²) + eps) · weight; `zero_centered` stores the
scale as an offset from one (Qwen 3.5, Gemma)."""
import torch
from kernels._common import refuse_unknown, supports_from, w

CONTRACT = ("norm.rms", "1.0.0")


CAPABILITIES = {"arguments": {"width": "any", "eps": "any", "zero_centered": "any"}, "states": []}


# What a conformer must meet against this kernel's unit fixtures, per compute dtype (§4.2):
# `|a − b| ≤ atol + rtol·|b|`. The manifest's witness block is written from it.
TOLERANCE = {'f32': {'atol': 1e-5, 'rtol': 1e-4}, 'bf16': {'atol': 1e-1, 'rtol': 2e-2}}


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)

def run(ctx, arguments, inputs, params, states, physical=None):
    x = inputs['input']
    scale = w(ctx, params['weight'])
    if arguments.get('zero_centered'):
        scale = scale + 1
    var = x.pow(2).mean(-1, keepdim=True)
    return {'output': x * torch.rsqrt(var + arguments['eps']) * scale}
