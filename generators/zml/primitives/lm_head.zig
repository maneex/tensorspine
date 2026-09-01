//! lm_head@1.0.0 — one logit per vocabulary entry.

const p = @import("../primitive.zig");

pub const primitive: p.Primitive = .{
    .name = "lm_head",
    .version = "1.0.0",
    .run = run,
    .capabilities =
    \\{"arguments": {"width": "any", "vocabulary": "any"}, "states": []}
    ,
};

fn run(ctx: *p.Ctx, call: p.Call) ![]const p.Binding {
    return p.one(ctx, "logits", p.linear(call.inputs.must("input"), call.params.must("weight")));
}
