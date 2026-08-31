//! embed@1.0.0 — one `width` vector per token identifier.

const zml = @import("zml");

const p = @import("../primitive.zig");

pub const primitive: p.Primitive = .{
    .name = "embed",
    .version = "1.0.0",
    .run = run,
    .capabilities =
    \\{"arguments": {"width": "any", "vocabulary": "any"}, "states": []}
    ,
};

fn run(ctx: *p.Ctx, call: p.Call) ![]const p.Binding {
    const ids = call.inputs.must("tokens");
    const table = call.params.must("weight");
    // Gather first, then convert: the whole table is never upcast.
    const rows = zml.nn.TokenEmbedding.forward(.{ .weight = table }, ids);
    return p.one(ctx, "output", rows.convert(ctx.compute));
}
