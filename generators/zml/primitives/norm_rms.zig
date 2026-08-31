//! norm.rms@1.0.0 — x · rsqrt(mean(x²) + eps) · weight.
//!
//! `zero_centered` stores the scale as an offset from one (Qwen 3.5, Gemma), so the
//! effective scale is `weight + 1`. ZML's own models spell that convention out per
//! model; here it is the contract's argument, read once.

const zml = @import("zml");

const p = @import("../primitive.zig");

pub const primitive: p.Primitive = .{
    .name = "norm.rms",
    .version = "1.0.0",
    .run = run,
    .capabilities =
    \\{"arguments": {"width": "any", "eps": "any", "zero_centered": "any"}, "states": []}
    ,
};

fn run(ctx: *p.Ctx, call: p.Call) ![]const p.Binding {
    const eps: f32 = @floatCast(try call.requireFloat("eps"));

    const x = call.inputs.must("input").withPartialTags(.{.d});
    var scale = call.params.must("weight").withTags(.{.d}).convert(x.dtype());
    if (call.argBool("zero_centered")) scale = scale.addConstant(1);

    const y = zml.nn.rmsNorm(x, .d, eps).mul(scale.broad(x.shape()));
    return p.one(ctx, "output", y);
}
