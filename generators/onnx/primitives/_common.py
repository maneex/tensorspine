"""Helpers shared by the emitters: the manifest rules evaluated by the language's own reader, a
projection against a stored weight, an RMS norm, the activations."""
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if os.path.join(_ROOT, 'tools') not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, 'tools'))
from capabilities import supports as supports_from   # noqa: E402,F401


def transposed(ctx, weight):
    """The stored [out, in] weight as [in, out], once per weight (the runtime folds the constant)."""
    b = ctx.b
    if weight not in b.transposed:
        b.transposed[weight] = b.node('Transpose', [weight], hint=f"{weight}.T", perm=[1, 0])
    return b.transposed[weight]


def linear(ctx, x, weight, bias=None):
    """x @ weightᵀ (+ bias), the weight stored [out, in] as the file stores it: `Gemm` on the 2-D
    activation of the `none` layout; `MatMul` against the transposed weight on the aligned
    layout's `[b, n, in]` (any rank)."""
    b = ctx.b
    if ctx.layout == 'none':
        inputs = [x, weight] + ([bias] if bias is not None else [])
        return b.node('Gemm', inputs, hint=f"{ctx.node}.gemm", transB=1)
    y = b.node('MatMul', [x, transposed(ctx, weight)], hint=f"{ctx.node}.matmul")
    return b.node('Add', [y, bias], hint=f"{ctx.node}.biased") if bias is not None else y


def rms_norm(ctx, x, scale, eps, zero_centered=False):
    """x · rsqrt(mean(x²) + eps) · scale, the last axis normalised."""
    b = ctx.b
    sq = b.node('Mul', [x, x], hint=f"{ctx.node}.sq")
    mean = b.node('ReduceMean', [sq], hint=f"{ctx.node}.ms", axes=[-1], keepdims=1)
    inv = b.node('Reciprocal', [b.node('Sqrt', [b.node('Add', [mean, b.const(np.float32(eps), 'eps')], hint=f"{ctx.node}.var")], hint=f"{ctx.node}.std")], hint=f"{ctx.node}.rstd")
    y = b.node('Mul', [x, inv], hint=f"{ctx.node}.normed")
    if scale is None:
        return y
    if zero_centered:
        scale = b.node('Add', [scale, b.const(np.float32(1.0), 'one')], hint=f"{ctx.node}.scale1")
    return b.node('Mul', [y, scale], hint=f"{ctx.node}.scaled")


def activation(ctx, x, kind):
    b = ctx.b
    if kind == 'silu':
        return b.node('Mul', [x, b.node('Sigmoid', [x], hint=f"{ctx.node}.sig")], hint=f"{ctx.node}.silu")
    if kind == 'gelu':                    # the erf form
        h = b.node('Mul', [x, b.const(np.float32(1 / np.sqrt(2.0)), 'rsqrt2')], hint=f"{ctx.node}.h")
        e = b.node('Add', [b.node('Erf', [h], hint=f"{ctx.node}.erf"), b.const(np.float32(1.0), 'one')], hint=f"{ctx.node}.e")
        return b.node('Mul', [b.node('Mul', [x, e], hint=f"{ctx.node}.xe"), b.const(np.float32(0.5), 'half')], hint=f"{ctx.node}.gelu")
    if kind == 'gelu_tanh':
        kx2 = b.node('Mul', [b.node('Mul', [x, x], hint=f"{ctx.node}.x2"), b.const(np.float32(0.044715), 'k')], hint=f"{ctx.node}.kx2")
        inner = b.node('Mul', [x, b.node('Add', [b.const(np.float32(1.0), 'one'), kx2], hint=f"{ctx.node}.poly")], hint=f"{ctx.node}.inner")   # x (1 + 0.044715 x²)
        t = b.node('Tanh', [b.node('Mul', [inner, b.const(np.float32(np.sqrt(2.0 / np.pi)), 's2pi')], hint=f"{ctx.node}.arg")], hint=f"{ctx.node}.tanh")
        e = b.node('Add', [t, b.const(np.float32(1.0), 'one')], hint=f"{ctx.node}.e")
        return b.node('Mul', [b.node('Mul', [x, e], hint=f"{ctx.node}.xe"), b.const(np.float32(0.5), 'half')], hint=f"{ctx.node}.gelu_tanh")
    if kind == 'relu2':
        r = b.node('Relu', [x], hint=f"{ctx.node}.relu")
        return b.node('Mul', [r, r], hint=f"{ctx.node}.relu2")
    raise ValueError(f"activation {kind} is not emitted")
