"""norm.rms@1.0.0 — x · rsqrt(mean(x²) + eps) · weight; `zero_centered` stores the
scale as an offset from one (Qwen 3.5, Gemma)."""
import torch
from kernels._common import refuse_unknown, w

CONTRACT = ("norm.rms", "1.0.0")
KNOWN = {'width', 'eps', 'zero_centered'}


def supports(arguments):
    """Implemented: eps, zero_centered. Refused: unknown arguments."""
    reasons = []
    refuse_unknown(arguments, KNOWN, reasons)
    return reasons


def run(ctx, arguments, inputs, params, states):
    x = inputs['input']
    scale = w(ctx, params['weight'])
    if arguments.get('zero_centered'):
        scale = scale + 1
    var = x.pow(2).mean(-1, keepdim=True)
    return {'output': x * torch.rsqrt(var + arguments['eps']) * scale}
