"""ffn.gated@1.0.0 — act(x·gateᵀ) ⊙ (x·upᵀ) · outᵀ, with optional biases.

| branch / record        | status                       |
|------------------------|------------------------------|
| activation silu        | implemented                  |
| activation gelu        | implemented (erf)            |
| activation gelu_tanh   | implemented                  |
| activation relu2       | implemented                  |
| in_bias, out_bias      | implemented                  |
| activation_sparsity    | implemented: Gaussian top-k on the gate before the activation — cutoff = mean + std (biased) · Φ⁻¹(fraction) per element over the inner axis, relu(gate − cutoff) |
"""
import torch
import torch.nn.functional as F
from kernels._common import present, refuse_unknown, supports_from, w

PRIMITIVE = ("ffn.gated", "1.0.0")
ACT = {'silu': F.silu, 'gelu': F.gelu, 'gelu_tanh': lambda x: F.gelu(x, approximate='tanh'),
       'relu2': lambda x: F.relu(x).pow(2)}


CAPABILITIES = {"arguments": {"width": "any", "inner": "any", "activation": ["silu", "gelu", "gelu_tanh", "relu2"],
                              "in_bias": "any", "out_bias": "any",
                              "activation_sparsity": "any"},
                "states": []}


# What a conformer must meet against this kernel's unit fixtures, per compute dtype (§4.2):
# `|a − b| ≤ atol + rtol·|b|`. The manifest's witness block is written from it.
TOLERANCE = {'f32': {'atol': 1e-5, 'rtol': 1e-4}, 'bf16': {'atol': 1e-1, 'rtol': 2e-2}}

# The unit fixtures this kernel produces (docs/TENSORSPINE-FIXTURE.md): one case per branch
# worth its own evidence, at small quantities.
FIXTURES = [
    {"case": "silu", "seed": 33, "invocations": [{"input": 5}, {"input": 3}], "arguments": {"width": 64, "inner": 128, "activation": "silu"}},
    {"case": "gelu-tanh-biased", "seed": 34, "invocations": [{"input": 5}],
     "arguments": {"width": 64, "inner": 128, "activation": "gelu_tanh", "in_bias": True, "out_bias": True}},
    {"case": "relu2", "seed": 35, "invocations": [{"input": 5}], "arguments": {"width": 64, "inner": 128, "activation": "relu2"}},
]


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)

def run(ctx, arguments, inputs, params, states, physical=None):
    x = inputs['input']
    g = x @ w(ctx, params['gate']).T
    u = x @ w(ctx, params['up']).T
    if arguments.get('in_bias'):
        g = g + w(ctx, params['gate_bias'])
        u = u + w(ctx, params['up_bias'])
    sparsity = arguments.get('activation_sparsity') or 0
    if sparsity > 0:
        z = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(float(sparsity)))
        cutoff = g.mean(-1, keepdim=True) + g.std(-1, keepdim=True, unbiased=False) * z.to(g.dtype)
        g = F.relu(g - cutoff)
    h = ACT[arguments['activation']](g) * u
    y = h @ w(ctx, params['out']).T
    if arguments.get('out_bias'):
        y = y + w(ctx, params['out_bias'])
    return {'output': y}
