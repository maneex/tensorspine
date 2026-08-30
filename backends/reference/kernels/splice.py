"""splice@1.0.0 — inserts an already-projected stream into the token sequence.

| branch                     | status                                                        |
|----------------------------|---------------------------------------------------------------|
| empty `source`             | implemented: the output is `text`                             |
| non-empty `source`         | refused: the language does not yet say where the elements go  |
"""
from kernels._common import refuse_unknown

CONTRACT = ("splice", "1.0.0")
KNOWN = {'width'}


class Unplaced(Exception):
    pass


def supports(arguments):
    reasons = []
    refuse_unknown(arguments, KNOWN, reasons)
    return reasons


def run(ctx, arguments, inputs, params, states):
    source = inputs.get('source')
    if source is not None and source.shape[0] > 0:
        raise Unplaced("splice: the placement of the inserted elements in the token stream is not declared by the language")
    return {'output': inputs['text']}
