//! The contract's activations, by the contract's names.
//!
//! The names matter more than they look. ZML's `Tensor.gelu` is the **tanh
//! approximation**, which the catalog calls `gelu_tanh` and distinguishes from the exact
//! `gelu`; mapping either by name would compute the wrong function silently, and a
//! retriever's embeddings would be quietly wrong rather than loudly absent.
//!
//! The exact one needs `erf`, which neither StableHLO nor ZML gives us: no `erf` op is
//! bound, and CHLO is not registered. It is computed here from public tensor operations
//! instead — Abramowitz & Stegun 7.1.26, whose error is bounded by 1.5e-7 on erf itself.
//! That is an implementation of the exact gelu, in the same sense a libm's is; the
//! fixture comparison is what decides whether it is close enough, and it is recorded as a
//! note in the manifest rather than passed off as a hardware erf.

const std = @import("std");

const zml = @import("zml");

pub const Error = error{UnknownActivation};

pub fn apply(name: []const u8, x: zml.Tensor) !zml.Tensor {
    if (std.mem.eql(u8, name, "silu")) return x.silu();
    if (std.mem.eql(u8, name, "gelu_tanh")) return x.gelu();
    if (std.mem.eql(u8, name, "relu2")) return x.relu().powByConst(2);
    if (std.mem.eql(u8, name, "gelu")) return gelu(x);
    return Error.UnknownActivation;
}

/// gelu(x) = x · Φ(x) = x · ½ · (1 + erf(x/√2)).
fn gelu(x: zml.Tensor) zml.Tensor {
    const half = x.scale(0.5);
    return half.mul(erf(x.scale(1.0 / std.math.sqrt2)).addConstant(1));
}

/// erf, by Abramowitz & Stegun 7.1.26 on |x|, carried back by the odd symmetry
/// erf(−x) = −erf(x). |ε| ≤ 1.5e-7.
fn erf(x: zml.Tensor) zml.Tensor {
    const dt = x.dtype();
    const ax = x.abs();

    const one: zml.Tensor = .scalar(1, dt);
    const t = one.broad(ax.shape()).div(ax.scale(0.3275911).addConstant(1));

    // Horner, outermost coefficient first.
    var poly = t.scale(1.061405429).addConstant(-1.453152027);
    poly = poly.mul(t).addConstant(1.421413741);
    poly = poly.mul(t).addConstant(-0.284496736);
    poly = poly.mul(t).addConstant(0.254829592);
    poly = poly.mul(t);

    const decay = ax.mul(ax).scale(-1).exp();
    return x.sign().mul(one.broad(ax.shape()).sub(poly.mul(decay)));
}
