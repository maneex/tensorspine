"""residual.add@1.0.0 — a + b."""
from kernels._common import refuse_unknown

CONTRACT = ("residual.add", "1.0.0")
KNOWN = {'width'}


def supports(arguments):
    reasons = []
    refuse_unknown(arguments, KNOWN, reasons)
    return reasons


def run(ctx, arguments, inputs, params, states):
    return {'output': inputs['a'] + inputs['b']}
