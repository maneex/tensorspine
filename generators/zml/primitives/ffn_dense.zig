//! ffn.dense@1.0.0 — act(x·inᵀ + in_bias) · outᵀ + out_bias.
//!
//! `ffn.gated` without its gate half: the up projection, the activation, the down
//! projection.
//!
//! | branch                   | status                        |
//! |--------------------------|-------------------------------|
//! | activation silu          | implemented                   |
//! | activation gelu (erf)    | implemented — see activation.zig |
//! | activation gelu_tanh     | implemented                   |
//! | activation relu2         | implemented                   |
//! | in_bias, out_bias        | implemented                   |
//! | activation_sparsity      | refused when > 0              |

const activation = @import("../activation.zig");
const p = @import("../primitive.zig");

pub const primitive: p.Primitive = .{
    .name = "ffn.dense",
    .version = "1.0.0",
    .run = run,
    .capabilities =
    \\{"arguments": {"width": "any", "inner": "any",
    \\               "activation": ["silu", "gelu", "gelu_tanh", "relu2"],
    \\               "in_bias": "any", "out_bias": "any",
    \\               "activation_sparsity": {"absent": true, "values": [0, 0.0]}},
    \\ "states": [],
    \\ "notes": ["activation gelu computes erf by Abramowitz & Stegun 7.1.26 (|e| <= 1.5e-7): neither StableHLO nor ZML binds an erf op"]}
    ,
};

fn run(ctx: *p.Ctx, call: p.Call) ![]const p.Binding {
    const x = p.features(call.inputs.must("input"));
    const name = call.argStr("activation") orelse return p.Error.MissingArgument;

    var h = p.linear(x, call.params.must("in"));
    if (call.argBool("in_bias")) {
        h = h.add(p.features(call.params.must("in_bias")).convert(h.dtype()).broad(h.shape()));
    }

    var y = p.linear(activation.apply(name, h) catch return p.Error.Unimplemented, call.params.must("out"));
    if (call.argBool("out_bias")) {
        y = y.add(p.features(call.params.must("out_bias")).convert(y.dtype()).broad(y.shape()));
    }
    return p.one(ctx, "output", y);
}
