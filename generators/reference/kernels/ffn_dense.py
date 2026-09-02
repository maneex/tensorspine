"""ffn.dense@1.0.0 — act(x·inᵀ + in_bias) · outᵀ + out_bias: the up projection, the activation,
the down projection — `ffn.gated` without its gate half.

| branch / record           | status                 |
|---------------------------|------------------------|
| activation silu           | implemented            |
| activation gelu           | implemented (erf)      |
| activation gelu_tanh      | implemented            |
| activation relu2          | implemented            |
| in_bias, out_bias         | implemented            |
| condition_dim / condition | refused                |
| activation_sparsity       | refused when > 0       |
"""
import torch.nn.functional as F
from kernels._common import refuse_unknown, supports_from, w

CONTRACT = ("ffn.dense", "1.0.0")
ACT = {'silu': F.silu, 'gelu': F.gelu, 'gelu_tanh': lambda x: F.gelu(x, approximate='tanh'),
       'relu2': lambda x: F.relu(x).pow(2)}


CAPABILITIES = {"arguments": {"width": "any", "inner": "any", "activation": ["silu", "gelu", "gelu_tanh", "relu2"],
                              "in_bias": "any", "out_bias": "any", "condition_dim": "absent", "condition": "absent",
                              "activation_sparsity": {"absent": True, "values": [0, 0.0]}},
                "states": []}


# What a conformer must meet against this kernel's unit fixtures, per compute dtype (§4.2):
# `|a − b| ≤ atol + rtol·|b|`. The manifest's witness block is written from it.
TOLERANCE = {'f32': {'atol': 1e-5, 'rtol': 1e-4}, 'bf16': {'atol': 1e-1, 'rtol': 2e-2}}

# The unit fixtures this kernel produces (docs/TENSORSPINE-FIXTURE.md): one case per branch
# worth its own evidence, at small quantities.
FIXTURES = [
    {"case": "gelu-biased", "seed": 31, "invocations": [{"input": 5}, {"input": 3}],
     "arguments": {"width": 64, "inner": 128, "activation": "gelu", "in_bias": True, "out_bias": True}},
    {"case": "silu", "seed": 32, "invocations": [{"input": 5}], "arguments": {"width": 64, "inner": 128, "activation": "silu"}},
]


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)

def run(ctx, arguments, inputs, params, states, physical=None):
    x = inputs['input']
    h = x @ w(ctx, params['in']).T
    if arguments.get('in_bias'):
        h = h + w(ctx, params['in_bias'])
    y = ACT[arguments['activation']](h) @ w(ctx, params['out']).T
    if arguments.get('out_bias'):
        y = y + w(ctx, params['out_bias'])
    return {'output': y}
