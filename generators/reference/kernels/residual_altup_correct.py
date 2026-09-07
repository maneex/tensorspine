"""residual.altup_correct@1.0.0 — AltUp's correction: the layer's output on the active stream is
routed as the input was (`router` after `router_norm`, `altup_predict`'s, tied); `correction` maps
the modalities to one coefficient per stream, plus one; the innovation `activated − predicted
active` times each stream's coefficient is added to the predictions. `active` is the corrected
active stream multiplied channel-wise by `correct_scale`; `streams` keeps it unscaled.

| branch / record | status      |
|-----------------|-------------|
| streams ≥ 2     | implemented |
"""
import torch
from kernels._common import refuse_unknown, supports_from, w
from kernels.residual_altup_predict import modalities

PRIMITIVE = ("residual.altup_correct", "1.0.0")


CAPABILITIES = {"arguments": {"width": "any", "streams": "any", "eps": "any"}, "states": []}


# What a conformer must meet against this kernel's unit fixtures, per compute dtype (§4.2):
# `|a − b| ≤ atol + rtol·|b|`. The manifest's witness block is written from it.
TOLERANCE = {'f32': {'atol': 1e-05, 'rtol': 0.0001}, 'bf16': {'atol': 0.1, 'rtol': 0.02}}

# The unit fixtures this kernel produces (docs/TENSORSPINE-FIXTURE.md): the predictions and the
# activated stream on one stream, corrected.
FIXTURES = [
    {"case": "basic", "seed": 113, "invocations": [{"predictions": 5, "activated": 5}], "arguments": {"width": 16, "streams": 3, "eps": 1e-6}},
]


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)

def run(ctx, arguments, inputs, params, states, physical=None):
    pred, activated = inputs['predictions'], inputs['activated']
    m = modalities(ctx, activated, params, arguments['eps'], pred.shape[-1])
    coefs = m @ w(ctx, params['correction']).T + 1.0                  # [n, streams]
    innovation = activated - pred[:, 0]
    corrected = pred + coefs[:, :, None] * innovation[:, None, :]
    return {'streams': corrected, 'active': corrected[:, 0] * w(ctx, params['correct_scale'])}
