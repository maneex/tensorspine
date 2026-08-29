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
from kernels._common import present, refuse_unknown, w

CONTRACT = ("ffn.gated", "1.0.0")
KNOWN = {'width', 'inner', 'activation', 'in_bias', 'out_bias', 'condition_dim', 'condition', 'activation_sparsity'}
ACT = {'silu': F.silu, 'gelu': F.gelu, 'gelu_tanh': lambda x: F.gelu(x, approximate='tanh'),
       'relu2': lambda x: F.relu(x).pow(2)}


def supports(arguments):
    reasons = []
    refuse_unknown(arguments, KNOWN, reasons)
    if arguments.get('activation') not in ACT:
        reasons.append(f"activation={arguments.get('activation')}")
    if present(arguments, 'condition_dim'):
        reasons.append(f"condition_dim={arguments['condition_dim']}")
    if present(arguments, 'condition'):
        reasons.append(f"condition={arguments['condition']}")
    if arguments.get('activation_sparsity'):
        reasons.append(f"activation_sparsity={arguments['activation_sparsity']}")
    return reasons


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
