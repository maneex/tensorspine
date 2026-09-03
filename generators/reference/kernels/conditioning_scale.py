"""conditioning.scale@1.0.0 — input ⊙ (1 + out · act(in · condition)): the adaptive scale Voxtral
Realtime applies to the feed-forward's normed input, its condition the time embedding of the delay.

| branch / record            | status                                                      |
|----------------------------|-------------------------------------------------------------|
| activation silu, gelu, gelu_tanh, relu2 | implemented (gelu is the erf form)                  |
| condition delivered        | implemented: appended to `condition_cache` along the port's stream (an `align` transform: another domain than the input's) |
| condition held             | implemented: an invocation delivering nothing on the port — every decode step — reads the cache (§7's exemption) |

Conventions the contract leaves open, as read here: the condition applied is the last element
held — one per sequence for a time embedding, which is what the held stream carries; the rank
path is computed once per invocation from that element and broadcast over the input's rows. An
invocation with nothing delivered and nothing held is refused: there is no condition to scale by.
"""
import torch
import torch.nn.functional as F
from kernels._common import supports_from, w

CONTRACT = ("conditioning.scale", "1.0.0")
ACT = {'silu': F.silu, 'gelu': F.gelu, 'gelu_tanh': lambda x: F.gelu(x, approximate='tanh'),
       'relu2': lambda x: F.relu(x).pow(2)}


CAPABILITIES = {"arguments": {"width": "any", "condition_width": "any", "rank": "any",
                              "activation": ["silu", "gelu", "gelu_tanh", "relu2"]},
                "states": ["append"],
                "transforms": ["align"],
                "notes": ["the condition applied is the last element held by condition_cache: one per sequence"]}


# What a conformer must meet against this kernel's unit fixtures, per compute dtype (§4.2):
# `|a − b| ≤ atol + rtol·|b|`. The manifest's witness block is written from it.
TOLERANCE = {'f32': {'atol': 1e-5, 'rtol': 1e-4}, 'bf16': {'atol': 1e-1, 'rtol': 2e-2}}

# The unit fixtures this kernel produces (docs/TENSORSPINE-FIXTURE.md): one case per branch
# worth its own evidence, at small quantities. The second invocation delivers no condition and
# reads the one held — a decode step.
FIXTURES = [
    {"case": "basic", "seed": 71, "invocations": [{"input": 5, "condition": 1}, {"input": 3}],
     "arguments": {"width": 64, "condition_width": 48, "rank": 8, "activation": "gelu"}},
]


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)


def run(ctx, arguments, inputs, params, states, physical=None):
    x = inputs['input']
    c = inputs.get('condition')
    st = states['condition_cache']
    if c is not None and c.shape[0]:
        st.append({'c': c})
    bufs, length = st.read()
    if not length:
        raise ValueError("conditioning.scale: no condition delivered on the port and none held by condition_cache")
    e = bufs['c'][length - 1].to(ctx.dtype)                         # the held condition: one per sequence
    h = ACT[arguments['activation']](e @ w(ctx, params['in']).T) @ w(ctx, params['out']).T
    return {'output': x * (1 + h)}
