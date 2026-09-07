//! residual.add@2.0.0 — a + b.

const p = @import("../primitive.zig");

pub const primitive: p.Primitive = .{
    .name = "residual.add",
    .version = "2.0.0",
    .run = run,
    .capabilities =
    \\{"arguments": {"width": "any"}, "states": []}
    ,
};

fn run(ctx: *p.Ctx, call: p.Call) ![]const p.Binding {
    return p.one(ctx, "output", call.inputs.must("a").add(call.inputs.must("b")));
}
