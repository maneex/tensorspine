//! norm.layer@1.0.0 — (x − mean) · rsqrt(var + eps) · weight + bias over `width`.
//!
//! The variance is biased, as torch's LayerNorm and BERT compute it — which is what
//! `zml.nn.normalizeVariance` does too.

const zml = @import("zml");

const p = @import("../primitive.zig");

pub const primitive: p.Primitive = .{
    .name = "norm.layer",
    .version = "1.0.0",
    .run = run,
    .capabilities =
    \\{"arguments": {"width": "any", "eps": "any"}, "states": []}
    ,
};

fn run(ctx: *p.Ctx, call: p.Call) ![]const p.Binding {
    const eps: f32 = @floatCast(try call.requireFloat("eps"));
    const x = p.features(call.inputs.must("input"));

    const norm: zml.nn.LayerNorm = .{
        .weight = p.features(call.params.must("weight")).convert(x.dtype()),
        .bias = p.features(call.params.must("bias")).convert(x.dtype()),
        .eps = eps,
    };
    return p.one(ctx, "output", norm.forward(x));
}
