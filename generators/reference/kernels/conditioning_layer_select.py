"""conditioning.layer_select@1.0.0 — the `layer`-th of the `layers` vectors the input carries.

| branch / record | status      |
|-----------------|-------------|
| layer index     | implemented |
"""
from kernels._common import refuse_unknown, supports_from

PRIMITIVE = ("conditioning.layer_select", "1.0.0")


CAPABILITIES = {"arguments": {"layers": "any", "width": "any", "layer": "any"}, "states": []}


# What a conformer must meet against this kernel's unit fixtures, per compute dtype (§4.2):
# `|a − b| ≤ atol + rtol·|b|`. The manifest's witness block is written from it.
TOLERANCE = {'f32': {'atol': 0.0, 'rtol': 0.0}, 'bf16': {'atol': 0.01, 'rtol': 0.01}}

# The unit fixtures this kernel produces (docs/TENSORSPINE-FIXTURE.md): the third of four per-layer vectors.
FIXTURES = [
    {"case": "basic", "seed": 118, "invocations": [{"input": 5}], "arguments": {"layers": 4, "width": 8, "layer": 2}},
]


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)

def run(ctx, arguments, inputs, params, states, physical=None):
    return {'output': inputs['input'][:, arguments['layer']]}
