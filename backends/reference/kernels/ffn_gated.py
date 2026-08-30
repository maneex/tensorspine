"""ffn.gated@1.0.0 — act(x·gateᵀ) ⊙ (x·upᵀ) · outᵀ, with optional biases.

| branch / record        | status                       |
|------------------------|------------------------------|
| activation silu        | implemented                  |
| activation gelu        | implemented (erf)            |
| activation gelu_tanh   | implemented                  |
| activation relu2       | implemented                  |
| in_bias, out_bias      | implemented                  |
| condition_dim / condition | refused (M2+)             |
| activation_sparsity    | refused when > 0             |
"""
import torch
import torch.nn.functional as F
from kernels._common import present, refuse_unknown, supports_from, w

CONTRACT = ("ffn.gated", "1.0.0")
ACT = {'silu': F.silu, 'gelu': F.gelu, 'gelu_tanh': lambda x: F.gelu(x, approximate='tanh'),
       'relu2': lambda x: F.relu(x).pow(2)}


CAPABILITIES = {"arguments": {"width": "any", "inner": "any", "activation": ["silu", "gelu", "gelu_tanh", "relu2"],
                              "in_bias": "any", "out_bias": "any", "condition_dim": "absent", "condition": "absent",
                              "activation_sparsity": {"absent": True, "values": [0, 0.0]}},
                "states": []}


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)

def run(ctx, arguments, inputs, params, states):
    x = inputs['input']
    g = x @ w(ctx, params['gate']).T
    u = x @ w(ctx, params['up']).T
    if arguments.get('in_bias'):
        g = g + w(ctx, params['gate_bias'])
        u = u + w(ctx, params['up_bias'])
    h = ACT[arguments['activation']](g) * u
    y = h @ w(ctx, params['out']).T
    if arguments.get('out_bias'):
        y = y + w(ctx, params['out_bias'])
    return {'output': y}
