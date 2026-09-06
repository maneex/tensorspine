"""pooler@1.0.0 — an embedding rather than logits: `weight` projects to `project_to`; `normalize`
l2 divides each vector by its norm (torch's floor of 1e-12, as ColBERT does); `reduce` none keeps
one vector per token.

| branch / record   | status                        |
|-------------------|-------------------------------|
| normalize l2/none | implemented                   |
| reduce none       | implemented                   |
| reduce mean / cls | refused (no document uses them) |
"""
import torch.nn.functional as F
from kernels._common import refuse_unknown, supports_from, w

CONTRACT = ("pooler", "1.0.0")


CAPABILITIES = {"arguments": {"width": "any", "project_to": "any", "normalize": ["l2", "none"], "reduce": ["none"]},
                "states": []}


# What a conformer must meet against this kernel's unit fixtures, per compute dtype (§4.2):
# `|a − b| ≤ atol + rtol·|b|`. The manifest's witness block is written from it.
TOLERANCE = {'f32': {'atol': 1e-5, 'rtol': 1e-4}, 'bf16': {'atol': 1e-1, 'rtol': 2e-2}}

# The unit fixtures this kernel produces (docs/TENSORSPINE-FIXTURE.md): one case per branch
# worth its own evidence, at small quantities.
FIXTURES = [
    {"case": "l2", "seed": 71, "invocations": [{"input": 5}, {"input": 3}], "arguments": {"width": 64, "project_to": 32, "normalize": "l2"}},
    {"case": "plain", "seed": 72, "invocations": [{"input": 5}], "arguments": {"width": 64, "project_to": 32, "normalize": "none"}},
]


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)

def run(ctx, arguments, inputs, params, states, physical=None):
    y = inputs['input'] @ w(ctx, params['weight']).T
    if arguments['normalize'] == 'l2':
        y = F.normalize(y, dim=-1)
    return {'output': y}
