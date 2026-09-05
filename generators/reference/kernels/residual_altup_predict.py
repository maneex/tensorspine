"""residual.altup_predict@1.0.0 — AltUp's prediction: the router reads the active stream after an
RMS norm (`router_norm`, `eps`) scaled by width⁻¹, its output through `tanh` gives `streams`
modalities; `prediction` maps them to a `streams × streams` mixing matrix applied across the
streams, and the input streams are added back. `active` is the predicted active stream (index 0).

| branch / record | status      |
|-----------------|-------------|
| streams ≥ 2     | implemented |
"""
import torch
from kernels._common import refuse_unknown, rms_norm, supports_from, w

CONTRACT = ("residual.altup_predict", "1.0.0")


CAPABILITIES = {"arguments": {"width": "any", "streams": "any", "eps": "any"}, "states": []}


# What a conformer must meet against this kernel's unit fixtures, per compute dtype (§4.2):
# `|a − b| ≤ atol + rtol·|b|`. The manifest's witness block is written from it.
TOLERANCE = {'f32': {'atol': 1e-05, 'rtol': 0.0001}, 'bf16': {'atol': 0.1, 'rtol': 0.02}}

# The unit fixtures this kernel produces (docs/TENSORSPINE-FIXTURE.md): three streams predicted.
FIXTURES = [
    {"case": "basic", "seed": 112, "invocations": [{"input": 5}, {"input": 2}], "arguments": {"width": 16, "streams": 3, "eps": 1e-6}},
]


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)

def modalities(ctx, x, params, eps, width):
    routed = rms_norm(x, w(ctx, params['router_norm']), eps) * (1.0 / width)
    return torch.tanh((routed @ w(ctx, params['router']).T).to(torch.float32)).to(x.dtype)

def run(ctx, arguments, inputs, params, states, physical=None):
    x = inputs['input']                                               # [n, streams, width]
    n, S, D = x.shape
    m = modalities(ctx, x[:, 0], params, arguments['eps'], D)         # [n, streams]
    coefs = (m @ w(ctx, params['prediction']).T).reshape(n, S, S)     # [n, out stream, in stream]
    pred = torch.einsum('nji,nid->njd', coefs, x) + x
    return {'predictions': pred, 'active': pred[:, 0]}
