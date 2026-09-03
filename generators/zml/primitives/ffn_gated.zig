//! ffn.gated@1.0.0 — act(x·gateᵀ) ⊙ (x·upᵀ) · outᵀ, with optional biases.
//!
//! | branch                   | status                                  |
//! |--------------------------|-----------------------------------------|
//! | activation silu          | implemented                             |
//! | activation gelu_tanh     | implemented                             |
//! | activation relu2         | implemented                             |
//! | activation gelu (erf)    | refused — ZML's `gelu` is the tanh approximation, which the contract names `gelu_tanh` and distinguishes from the exact one; mapping by name would compute the wrong function silently |
//! | in_bias, out_bias        | implemented                             |
//! | activation_sparsity      | refused when > 0                        |

const std = @import("std");

const zml = @import("zml");

const p = @import("../primitive.zig");

pub const primitive: p.Primitive = .{
    .name = "ffn.gated",
    .version = "1.0.0",
    .run = run,
    .capabilities =
    \\{"arguments": {"width": "any", "inner": "any",
    \\               "activation": ["silu", "gelu_tanh", "relu2"],
    \\               "in_bias": "any", "out_bias": "any",
    \\               "activation_sparsity": {"absent": true, "values": [0, 0.0]}},
    \\ "states": [],
    \\ "notes": ["activation gelu (erf) is refused: ZML's gelu is the tanh approximation"]}
    ,
};

fn activate(name: []const u8, x: zml.Tensor) !zml.Tensor {
    if (std.mem.eql(u8, name, "silu")) return x.silu();
    if (std.mem.eql(u8, name, "gelu_tanh")) return x.gelu();
    if (std.mem.eql(u8, name, "relu2")) return x.relu().powByConst(2);
    return p.Error.Unimplemented;
}

fn run(ctx: *p.Ctx, call: p.Call) ![]const p.Binding {
    const x = p.features(call.inputs.must("input"));
    const activation = call.argStr("activation") orelse return p.Error.MissingArgument;

    var g = p.linear(x, call.params.must("gate"));
    var u = p.linear(x, call.params.must("up"));
    if (call.argBool("in_bias")) {
        g = g.add(p.features(call.params.must("gate_bias")).convert(g.dtype()).broad(g.shape()));
        u = u.add(p.features(call.params.must("up_bias")).convert(u.dtype()).broad(u.shape()));
    }

    var y = p.linear((try activate(activation, g)).mul(u), call.params.must("out"));
    if (call.argBool("out_bias")) {
        y = y.add(p.features(call.params.must("out_bias")).convert(y.dtype()).broad(y.shape()));
    }
    return p.one(ctx, "output", y);
}
