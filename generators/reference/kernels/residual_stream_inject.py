"""residual.stream_inject@1.0.0 — adds `active` to every stream but the active one (index 0),
which passes through unchanged.

| branch / record | status      |
|-----------------|-------------|
| streams ≥ 2     | implemented |
"""
from kernels._common import refuse_unknown, supports_from

CONTRACT = ("residual.stream_inject", "1.0.0")


CAPABILITIES = {"arguments": {"width": "any", "streams": "any"}, "states": []}


# What a conformer must meet against this kernel's unit fixtures, per compute dtype (§4.2):
# `|a − b| ≤ atol + rtol·|b|`. The manifest's witness block is written from it.
TOLERANCE = {'f32': {'atol': 1e-06, 'rtol': 1e-05}, 'bf16': {'atol': 0.01, 'rtol': 0.01}}

# The unit fixtures this kernel produces (docs/TENSORSPINE-FIXTURE.md): the active vector added to the other streams.
FIXTURES = [
    {"case": "basic", "seed": 116, "invocations": [{"streams": 5, "active": 5}], "arguments": {"width": 16, "streams": 3}},
]


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)

def run(ctx, arguments, inputs, params, states, physical=None):
    out = inputs['streams'].clone()
    out[:, 1:] = out[:, 1:] + inputs['active'][:, None, :]
    return {'output': out}
