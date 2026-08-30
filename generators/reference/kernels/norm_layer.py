"""norm.layer@1.0.0 — (x − mean) · rsqrt(var + eps) · weight + bias over `width`, the variance
biased, as torch's LayerNorm and BERT compute it.

| branch / record | status      |
|-----------------|-------------|
| eps, weight, bias | implemented |
"""
import torch.nn.functional as F
from kernels._common import refuse_unknown, supports_from, w

CONTRACT = ("norm.layer", "1.0.0")


CAPABILITIES = {"arguments": {"width": "any", "eps": "any"}, "states": []}


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)

def run(ctx, arguments, inputs, params, states, physical=None):
    x = inputs['input']
    return {'output': F.layer_norm(x, (x.shape[-1],), w(ctx, params['weight']), w(ctx, params['bias']), arguments['eps'])}
