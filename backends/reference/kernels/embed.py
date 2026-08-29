"""embed@1.0.0 — one `width` vector per token identifier."""
from kernels._common import refuse_unknown, w

CONTRACT = ("embed", "1.0.0")
KNOWN = {'width', 'vocabulary'}


def supports(arguments):
    """Implemented: the lookup. Refused: any argument this kernel does not know."""
    reasons = []
    refuse_unknown(arguments, KNOWN, reasons)
    return reasons


def run(ctx, arguments, inputs, params, states):
    ids = inputs['tokens']
    return {'output': w(ctx, params['weight'])[ids]}
