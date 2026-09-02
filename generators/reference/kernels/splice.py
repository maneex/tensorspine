"""splice@1.0.0 — inserts an already-projected stream into the token sequence.

| branch                     | status                                                        |
|----------------------------|---------------------------------------------------------------|
| empty `source`             | implemented: the output is `text`                             |
| non-empty `source`         | refused: the language does not yet say where the elements go  |
"""
from kernels._common import refuse_unknown, supports_from

CONTRACT = ("splice", "1.0.0")


class Unplaced(Exception):
    pass


CAPABILITIES = {"arguments": {"width": "any"}, "states": [], "transforms": ["insert"],
                "notes": ["source must deliver nothing: the language does not yet say where inserted elements go"]}


# What a conformer must meet against this kernel's unit fixtures, per compute dtype (§4.2):
# `|a − b| ≤ atol + rtol·|b|`. The manifest's witness block is written from it.
TOLERANCE = {'f32': {'atol': 0.0, 'rtol': 0.0}, 'bf16': {'atol': 0.0, 'rtol': 0.0}}


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)

def run(ctx, arguments, inputs, params, states, physical=None):
    source = inputs.get('source')
    if source is not None and source.shape[0] > 0:
        raise Unplaced("splice: the placement of the inserted elements in the token stream is not declared by the language")
    return {'output': inputs['text']}
