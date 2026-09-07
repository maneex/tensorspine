//! pooler@1.0.0 — an embedding rather than logits.
//!
//! `weight` projects to `project_to`; `normalize: l2` divides each vector by its norm;
//! `reduce: none` keeps one vector per element, which is what makes a late-interaction
//! retriever a retriever — the document's output is a vector per token, not per document.
//!
//! | branch            | status                          |
//! |-------------------|---------------------------------|
//! | normalize l2, none| implemented                     |
//! | reduce none       | implemented                     |
//! | reduce mean, cls  | refused — no document uses them |

const std = @import("std");

const zml = @import("zml");

const p = @import("../primitive.zig");

pub const primitive: p.Primitive = .{
    .name = "pooler",
    .version = "1.0.0",
    .run = run,
    .capabilities =
    \\{"arguments": {"width": "any", "project_to": "any",
    \\               "normalize": ["l2", "none"], "reduce": ["none"]},
    \\ "states": []}
    ,
};

fn run(ctx: *p.Ctx, call: p.Call) ![]const p.Binding {
    if (!std.mem.eql(u8, call.argStr("reduce") orelse "none", "none")) return p.Error.Unimplemented;

    var y = p.linear(call.inputs.must("input"), call.params.must("weight"));
    if (std.mem.eql(u8, call.argStr("normalize") orelse "none", "l2")) {
        // torch's floor, as ColBERT uses it.
        y = zml.nn.normalizeL2(y, 1e-12);
    }
    return p.one(ctx, "output", y);
}
