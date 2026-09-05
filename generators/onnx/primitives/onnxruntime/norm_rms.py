"""norm.rms@1.0.0 for the onnxruntime target — one `SimplifiedLayerNormalization`; when the input
is the sum a `residual.add` occurrence just produced, the two occurrences become one
`SkipSimplifiedLayerNormalization` whose sum output replaces the Add for every consumer: a fusion
across occurrences, reasoned from the topology (the value's origin) rather than from the model."""
from primitives import norm_rms as portable

CONTRACT = portable.CONTRACT
CAPABILITIES = dict(portable.CAPABILITIES, notes=["SimplifiedLayerNormalization; fused with a preceding residual.add into SkipSimplifiedLayerNormalization"])


def supports(arguments):
    return portable.supports(arguments)


def emit(ctx, arguments, inputs, params, states):
    b, x = ctx.b, inputs['input']
    scale = portable.scale_of(ctx, arguments, params)
    src = ctx.origin(x)
    if src is not None and src[0] == 'residual.add':
        add = b.by_output.get(src[1])
        if add is not None and add.op_type == 'Add' and add in b.nodes:
            # the fused node takes [b, n, d]: the aligned layout's values as they are, the `none` layout's behind a batch axis of one
            aligned = ctx.layout == 'aligned'
            a3 = add.input[0] if aligned else b.unsqueeze0(add.input[0], f"{ctx.node}.a3")
            c3 = add.input[1] if aligned else b.unsqueeze0(add.input[1], f"{ctx.node}.b3")
            y3, _m, _v, sum3 = b.node('SkipSimplifiedLayerNormalization', [a3, c3, scale], outputs=4, hint=f"{ctx.node}.skipnorm",
                                      domain='com.microsoft', epsilon=float(arguments['eps']))
            src[2].input[0] = sum3 if aligned else b.squeeze0(sum3, f"{ctx.node}.sum")       # the Add's value, now the fused node's sum
            b.remove(add)
            return {'output': y3 if aligned else b.squeeze0(y3, f"{ctx.node}.normed")}
    return {'output': b.node('SimplifiedLayerNormalization', [x, scale], hint=f"{ctx.node}.rmsnorm", axis=-1, epsilon=float(arguments['eps']))}
