"""ffn.gated@1.0.0 for the onnxruntime target — the portable projections with the activation as one
fused operator: QuickGelu with alpha 1 (SiLU exactly), Gelu (the erf form), FastGelu (the tanh
form); relu2 stays composed."""
from primitives import ffn_gated as portable
from primitives._common import activation

PRIMITIVE = portable.PRIMITIVE
CAPABILITIES = dict(portable.CAPABILITIES, notes=["the activation as QuickGelu (silu), Gelu or FastGelu"])


def supports(arguments):
    return portable.supports(arguments)


def fused_activation(ctx, x, kind):
    if kind == 'silu':
        return ctx.b.node('QuickGelu', [x], hint=f"{ctx.node}.silu", domain='com.microsoft', alpha=1.0)
    if kind == 'gelu':
        return ctx.b.node('Gelu', [x], hint=f"{ctx.node}.gelu", domain='com.microsoft')
    if kind == 'gelu_tanh':
        return ctx.b.node('FastGelu', [x], hint=f"{ctx.node}.gelu_tanh", domain='com.microsoft')
    return activation(ctx, x, kind)


def emit(ctx, arguments, inputs, params, states):
    return portable.emit(ctx, arguments, inputs, params, states, act=fused_activation)
