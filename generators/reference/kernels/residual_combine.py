"""residual.combine@2.0.0 — output_scale · (left_scale · left + right_scale · right).

| branch / record | status      |
|-----------------|-------------|
| the three scales | implemented |
"""
from kernels._common import refuse_unknown, supports_from

PRIMITIVE = ("residual.combine", "2.0.0")


CAPABILITIES = {"arguments": {"width": "any", "left_scale": "any", "right_scale": "any", "output_scale": "any"}, "states": []}


# What a conformer must meet against this kernel's unit fixtures, per compute dtype (§4.2):
# `|a − b| ≤ atol + rtol·|b|`. The manifest's witness block is written from it.
TOLERANCE = {'f32': {'atol': 1e-06, 'rtol': 1e-05}, 'bf16': {'atol': 0.01, 'rtol': 0.01}}

# The unit fixtures this kernel produces (docs/TENSORSPINE-FIXTURE.md): Gemma 3n's LAUREL sum at 1/√2.
FIXTURES = [
    {"case": "basic", "seed": 115, "invocations": [{"left": 5, "right": 5}],
     "arguments": {"width": 16, "left_scale": 1.0, "right_scale": 1.0, "output_scale": 0.7071067811865476}},
]


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)

def run(ctx, arguments, inputs, params, states, physical=None):
    y = arguments['left_scale'] * inputs['left'] + arguments['right_scale'] * inputs['right']
    return {'output': arguments['output_scale'] * y}
