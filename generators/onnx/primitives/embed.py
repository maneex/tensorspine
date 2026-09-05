"""embed@1.0.0 — the token's row of the table, gathered at the stored dtype and cast once gathered."""
from primitives._common import supports_from

CONTRACT = ("embed", "1.0.0")
CAPABILITIES = {"arguments": {"width": "any", "vocabulary": "any"}, "states": []}


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)


def emit(ctx, arguments, inputs, params, states):
    rows = ctx.b.node('Gather', [ctx.param(params['weight'], cast=False), inputs['tokens']], hint=f"{ctx.node}.rows")
    return {'output': ctx.b.node('Cast', [rows], hint=f"{ctx.node}.out", to=ctx.b.ctype)}
