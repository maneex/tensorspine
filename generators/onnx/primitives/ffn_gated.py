"""ffn.gated@2.0.0 — act(x·gateᵀ) ⊙ (x·upᵀ) · outᵀ, with optional biases.

| branch / record        | status                       |
|------------------------|------------------------------|
| activation silu, gelu, gelu_tanh, relu2 | emitted (gelu in its erf form) |
| in_bias, out_bias      | emitted                      |
| activation_sparsity    | refused when > 0             |
"""
from primitives._common import activation, linear, supports_from

PRIMITIVE = ("ffn.gated", "2.0.0")
CAPABILITIES = {"arguments": {"width": "any", "inner": "any", "activation": ["silu", "gelu", "gelu_tanh", "relu2"],
                              "in_bias": "any", "out_bias": "any",
                              "activation_sparsity": {"absent": True, "values": [0, 0.0]}},
                "states": []}


def supports(arguments):
    return supports_from(CAPABILITIES, arguments)


def emit(ctx, arguments, inputs, params, states, act=activation):
    """`act`: the activation's emitter — a target's fused one may be passed in."""
    x = inputs['input']
    bias = bool(arguments.get('in_bias'))
    g = linear(ctx, x, ctx.param(params['gate']), ctx.param(params['gate_bias']) if bias else None)
    u = linear(ctx, x, ctx.param(params['up']), ctx.param(params['up_bias']) if bias else None)
    h = ctx.b.node('Mul', [act(ctx, g, arguments['activation']), u], hint=f"{ctx.node}.h")
    return {'output': linear(ctx, h, ctx.param(params['out']), ctx.param(params['out_bias']) if arguments.get('out_bias') else None)}
