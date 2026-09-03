"""projector.temporal_stack@1.0.0 — `merge_count` consecutive positions concatenated earliest
first, `input` projects to `width`, an activation, `output` projects again; no biases. n positions
make n / merge_count tokens of the same stream (the contract's `merge`, §5.3).

| branch / record       | status                                                        |
|-----------------------|---------------------------------------------------------------|
| activation gelu       | implemented (erf)                                             |
| activation gelu_tanh, relu, silu | refused until a document brings one (B02)          |

A delivery that is not a multiple of `merge_count` positions is refused before the projection:
the runtime's alignment rule (§5.3) makes it unreachable from a fragmented stream, and a whole
delivery that breaks it has no token for its remainder.
"""
import torch.nn.functional as F
from kernels._common import supports_from, w

CONTRACT = ("projector.temporal_stack", "1.0.0")


CAPABILITIES = {"arguments": {"width": "any", "source_width": "any", "merge_count": "any", "activation": ["gelu"]},
                "states": [],
                "transforms": ["merge"]}


# What a conformer must meet against this kernel's unit fixtures, per compute dtype (§4.2):
# `|a − b| ≤ atol + rtol·|b|`. The manifest's witness block is written from it.
TOLERANCE = {'f32': {'atol': 1e-5, 'rtol': 1e-4}, 'bf16': {'atol': 1e-1, 'rtol': 2e-2}}

# The unit fixtures this kernel produces (docs/TENSORSPINE-FIXTURE.md): eight positions make two tokens.
FIXTURES = [
    {"case": "basic", "seed": 75, "invocations": [{"input": 8}],
     "arguments": {"width": 16, "source_width": 8, "merge_count": 4, "activation": "gelu"}},
]


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)


def run(ctx, arguments, inputs, params, states, physical=None):
    x = inputs['input']
    n, merge = x.shape[0], arguments['merge_count']
    if n % merge:
        raise ValueError(f"projector.temporal_stack: {n} positions are not a multiple of merge_count {merge}")
    stacked = x.reshape(n // merge, merge * x.shape[1])            # consecutive positions, earliest first
    h = F.gelu(stacked @ w(ctx, params['input']).T)
    return {'output': h @ w(ctx, params['output']).T}
