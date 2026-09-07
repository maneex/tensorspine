"""embedding.token_auxiliary@2.0.0 — Gemma 3n's embedding with per-layer inputs: `output` is the
token's row of `weight` scaled by √width; `auxiliary` [layers, per_layer_width] per token is the
sum of the token's per-layer lookup (scaled by √per_layer_width) and the main embedding projected
by `per_layer_model_projection` (scaled by width⁻½ and RMS-normalized per layer with
`per_layer_projection_norm`, `eps`), the sum scaled by 1/√2.

| branch / record | status      |
|-----------------|-------------|
| the two tables, the projection, the scales | implemented |
"""
import math
import torch
from kernels._common import refuse_unknown, rms_norm, supports_from, w

PRIMITIVE = ("embedding.token_auxiliary", "2.0.0")


CAPABILITIES = {"arguments": {"width": "any", "vocabulary": "any", "layers": "any", "per_layer_width": "any",
                              "per_layer_vocabulary": "any", "eps": "any"}, "states": []}


# What a conformer must meet against this kernel's unit fixtures, per compute dtype (§4.2):
# `|a − b| ≤ atol + rtol·|b|`. The manifest's witness block is written from it.
TOLERANCE = {'f32': {'atol': 1e-05, 'rtol': 0.0001}, 'bf16': {'atol': 0.1, 'rtol': 0.02}}

# The unit fixtures this kernel produces (docs/TENSORSPINE-FIXTURE.md): the embedding and three per-layer vectors per token.
FIXTURES = [
    {"case": "basic", "seed": 120, "invocations": [{"tokens": 5}, {"tokens": 2}],
     "arguments": {"width": 16, "vocabulary": 64, "layers": 3, "per_layer_width": 8, "per_layer_vocabulary": 64, "eps": 1e-6}},
]


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)

def run(ctx, arguments, inputs, params, states, physical=None):
    ids = inputs['tokens']
    D, L, W = arguments['width'], arguments['layers'], arguments['per_layer_width']
    x = params['weight'][ids].to(ctx.dtype) * math.sqrt(D)                       # gather first, then upcast
    lookup = params['per_layer_embed'][ids].to(ctx.dtype).reshape(-1, L, W) * math.sqrt(W)
    proj = (x @ w(ctx, params['per_layer_model_projection']).T) * (D ** -0.5)
    proj = rms_norm(proj.reshape(-1, L, W), w(ctx, params['per_layer_projection_norm']), arguments['eps'])
    return {'output': x, 'auxiliary': (proj + lookup) * (1.0 / math.sqrt(2.0))}
