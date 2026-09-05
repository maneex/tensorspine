"""norm.rms@1.0.0 — x · rsqrt(mean(x²) + eps) · weight; `zero_centered` adds one to the stored scale.
Standard operators: Mul, ReduceMean, Add, Sqrt, Reciprocal, Mul, Mul."""
import numpy as np

from primitives._common import rms_norm, supports_from

CONTRACT = ("norm.rms", "1.0.0")
CAPABILITIES = {"arguments": {"width": "any", "eps": "any", "zero_centered": "any"}, "states": []}


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)


def scale_of(ctx, arguments, params):
    scale = ctx.param(params['weight'])
    if arguments.get('zero_centered'):
        scale = ctx.b.node('Add', [scale, ctx.b.const(np.float32(1.0), 'one')], hint=f"{ctx.node}.scale1")
    return scale


def emit(ctx, arguments, inputs, params, states):
    return {'output': rms_norm(ctx, inputs['input'], scale_of(ctx, arguments, params), arguments['eps'])}
