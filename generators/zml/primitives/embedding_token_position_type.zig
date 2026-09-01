//! embedding.token_position_type@1.0.0 — BERT's embedding.
//!
//! The token's row of `weight`, the position's row of `position` and the segment type's
//! row of `token_type`, summed, then a LayerNorm.
//!
//! The segment type is **0 for every token**, as the contract states for this version —
//! so no second input is needed, which is the question this contract looked like it would
//! raise and does not.
//!
//! A position at or beyond `positions` has no row. The contract says so; here the table's
//! own extent says it, and a gather past it is the runtime's business rather than a check
//! this primitive can make on a traced index.

const zml = @import("zml");

const p = @import("../primitive.zig");

pub const primitive: p.Primitive = .{
    .name = "embedding.token_position_type",
    .version = "1.0.0",
    .run = run,
    .needs_positions = true,
    .capabilities =
    \\{"arguments": {"width": "any", "vocabulary": "any", "positions": "any",
    \\               "token_types": "any", "eps": "any"},
    \\ "states": [],
    \\ "notes": ["the segment type is 0 for every token, as the contract states for this version"]}
    ,
};

fn run(ctx: *p.Ctx, call: p.Call) ![]const p.Binding {
    const eps: f32 = @floatCast(try call.requireFloat("eps"));
    const ids = call.inputs.must("tokens");
    const positions = ctx.positions orelse return p.Error.MissingArgument;

    // Gather first, convert after: no table is ever upcast whole.
    const tokens = row(call.params.must("weight"), ids).convert(ctx.compute);
    const places = row(call.params.must("position"), positions).convert(ctx.compute);
    const kind = call.params.must("token_type").slice(0, .single(0)).convert(ctx.compute);

    const x = p.features(tokens.add(places));
    const summed = x.add(p.features(kind).convert(x.dtype()).broad(x.shape()));

    const norm: zml.nn.LayerNorm = .{
        .weight = p.features(call.params.must("norm")).convert(summed.dtype()),
        .bias = p.features(call.params.must("norm_bias")).convert(summed.dtype()),
        .eps = eps,
    };
    return p.one(ctx, "output", norm.forward(summed));
}

/// One row of a `[entries, width]` table per index.
fn row(table: zml.Tensor, index: zml.Tensor) zml.Tensor {
    return table.withTags(.{ .entry, .d }).gather(.{ .entry = index }, .{});
}
