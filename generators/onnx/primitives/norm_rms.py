"""norm.rms@1.0.0 — x · rsqrt(mean(x²) + eps) · weight; `zero_centered` adds one to the stored scale."""
from primitives._common import rms_norm, supports_from

CONTRACT = ("norm.rms", "1.0.0")
CAPABILITIES = {"arguments": {"width": "any", "eps": "any", "zero_centered": "any"}, "states": []}


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)


def emit(ctx, arguments, inputs, params, states):
    return {'output': rms_norm(ctx, inputs['input'], ctx.param(params['weight']), arguments['eps'], bool(arguments.get('zero_centered')))}
