"""conditioning.multiplicative@2.0.0 — act(input · gateᵀ) ⊙ condition, projected back to `width`
and RMS-normalized (`norm`, `eps`): Gemma 3n's per-layer input term.

| branch / record        | status      |
|------------------------|-------------|
| activation silu / gelu / gelu_tanh / relu2 | implemented |
"""
import torch.nn.functional as F
from kernels._common import refuse_unknown, rms_norm, supports_from, w

PRIMITIVE = ("conditioning.multiplicative", "2.0.0")
ACT = {'silu': F.silu, 'gelu': F.gelu, 'gelu_tanh': lambda x: F.gelu(x, approximate='tanh'),
       'relu2': lambda x: F.relu(x).pow(2)}


CAPABILITIES = {"arguments": {"width": "any", "condition_width": "any", "eps": "any",
                              "activation": ["silu", "gelu", "gelu_tanh", "relu2"]}, "states": []}


# What a conformer must meet against this kernel's unit fixtures, per compute dtype (§4.2):
# `|a − b| ≤ atol + rtol·|b|`. The manifest's witness block is written from it.
TOLERANCE = {'f32': {'atol': 1e-05, 'rtol': 0.0001}, 'bf16': {'atol': 0.1, 'rtol': 0.02}}

# The unit fixtures this kernel produces (docs/TENSORSPINE-FIXTURE.md): a gated product on one stream, normalised.
FIXTURES = [
    {"case": "gelu-tanh", "seed": 119, "invocations": [{"input": 5, "condition": 5}],
     "arguments": {"width": 16, "condition_width": 8, "eps": 1e-6, "activation": "gelu_tanh"}},
]


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)

def run(ctx, arguments, inputs, params, states, physical=None):
    g = ACT[arguments['activation']](inputs['input'] @ w(ctx, params['gate']).T)
    y = (g * inputs['condition']) @ w(ctx, params['projection']).T
    return {'output': rms_norm(y, w(ctx, params['norm']), arguments['eps'])}
