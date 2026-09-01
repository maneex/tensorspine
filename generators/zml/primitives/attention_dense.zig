//! attention.dense@1.0.0 — dense or grouped-query attention over a KV state.
//!
//! | branch                          | status                                    |
//! |---------------------------------|-------------------------------------------|
//! | mask causal                     | implemented                               |
//! | mask none (stateless encoder)   | implemented                               |
//! | mask chunked, window            | refused                                   |
//! | cross, streaming                | refused                                   |
//! | rope theta, layout split        | implemented (rotate-half)                 |
//! | rope layout interleaved / 2d    | refused                                   |
//! | rope partial, mrope, scaling    | refused (later; the contract carries them) |
//! | qk_norm                         | refused (later)                           |
//! | temperature, output_gate        | refused                                   |
//! | q/k/v/out biases                | implemented                               |
//!
//! Conventions, as the contract now states them: the keys of the current elements are
//! appended to the state before the queries attend, so a query sees itself; the scale
//! is head_dim^-1/2; `split` pairs channel i with i + rotary/2 (rotate-half).
//!
//! ZML's `real_im_pass` RoPE layout is that same pairing — its own documentation says
//! HF models do not use `interleaved` — so the contract's `split` maps to it and to
//! nothing else.

const std = @import("std");

const zml = @import("zml");

const p = @import("../primitive.zig");

pub const primitive: p.Primitive = .{
    .name = "attention.dense",
    .version = "1.0.0",
    .run = run,
    .needs_positions = true,
    .capabilities =
    \\{"arguments": {"width": "any", "heads": "any", "head_dim": "any", "kv_heads": "any",
    \\               "mask": ["causal", "none"], "window": "absent", "chunk": "absent",
    \\               "cross": [false], "streaming": [false], "temperature": "absent",
    \\               "output_gate": [false], "qk_norm": "absent",
    \\               "rope": {"absent": true, "fields": {"theta": "any", "layout": ["split"],
    \\                        "partial": "absent", "mrope": "absent", "scaling": "absent"}},
    \\               "q_bias": "any", "k_bias": "any", "v_bias": "any", "out_bias": "any"},
    \\ "states": ["append"],
    \\ "notes": ["rope layout split is emitted as ZML's real_im_pass, the same rotate-half pairing"]}
    ,
};

/// q, k or v: projected, then split into heads — [.s, .h, .hd].
fn project(x: zml.Tensor, w: zml.Tensor, bias: ?zml.Tensor, heads: i64) zml.Tensor {
    var y = p.linear(x, w);
    if (bias) |b| y = y.add(p.features(b).convert(y.dtype()).broad(y.shape()));
    return y.splitAxis(.d, .{ .h = heads, .hd = .auto });
}

/// Which keys each query may attend to, as the additive mask `sdpa` takes: the
/// positions the state holds, and — when causal — those at or before the query's own
/// position. Masking over the whole capacity rather than slicing to the length is what
/// gives one code path whatever the capacity, which a compiled graph needs.
fn mask(positions: zml.Tensor, length: zml.Tensor, keys: i64, dt: zml.DataType) zml.Tensor {
    const qk = zml.Shape.init(.{ .q = positions.dim(0), .k = keys }, .i32);
    const kidx = zml.Tensor.iota(qk, .k);

    var allowed = kidx.cmp(.LT, length.convert(.i32).broad(qk));
    allowed = allowed.logical(.AND, kidx.cmp(.LE, positions.convert(.i32).withTags(.{.q}).broad(qk)));

    return zml.Tensor.select(allowed, .scalar(0, dt), .scalar(-std.math.inf(f32), dt));
}

fn run(ctx: *p.Ctx, call: p.Call) ![]const p.Binding {
    const x = call.inputs.must("input").withTags(.{ .s, .d });
    const heads = try call.requireInt("heads");
    const kv_heads = try call.requireInt("kv_heads");
    const positions = ctx.positions orelse return p.Error.MissingArgument;

    if (call.argBool("output_gate")) return p.Error.Unimplemented;
    if (call.arg("qk_norm") != null) return p.Error.Unimplemented;
    if (!std.mem.eql(u8, call.argStr("mask") orelse "causal", "causal")) return p.Error.Unimplemented;

    var q = project(x, call.params.must("q"), call.params.get("q_bias"), heads);
    var k = project(x, call.params.must("k"), call.params.get("k_bias"), kv_heads);
    const v = project(x, call.params.must("v"), call.params.get("v_bias"), kv_heads);

    if (call.argObj("rope")) |rope| {
        const layout = switch (rope.get("layout") orelse return p.Error.MissingArgument) {
            .string => |str| str,
            else => return p.Error.MissingArgument,
        };
        if (!std.mem.eql(u8, layout, "split")) return p.Error.Unimplemented;
        if (rope.get("partial") != null or rope.get("mrope") != null or rope.get("scaling") != null) {
            return p.Error.Unimplemented;
        }
        const theta: f32 = switch (rope.get("theta") orelse return p.Error.MissingArgument) {
            .integer => |i| @floatFromInt(i),
            .float => |f| @floatCast(f),
            else => return p.Error.MissingArgument,
        };
        const opts: zml.nn.RopeOpts = .{
            .layout = .real_im_pass,
            .scaling = .{ .default = .{ .rope_theta = theta } },
        };
        q = zml.nn.rope(q, positions.withTags(.{.s}), opts);
        k = zml.nn.rope(k, positions.withTags(.{.s}), opts);
    }

    // The keys and values of this invocation join the state before the queries attend,
    // so a query sees itself — the convention the contract states.
    const kv = call.state("kv") orelse return p.Error.MissingArgument;
    const updated = try kv.append(ctx.allocator, &.{ k, v });

    // This occurrence's own portion of the state, whatever layout holds it.
    const written = kv.after(updated);
    const k_view = written.get("k").?.withTags(.{ .k, .h, .hd }).convert(q.dtype());
    const v_view = written.get("v").?.withTags(.{ .k, .h, .hd }).convert(q.dtype());

    const out = zml.nn.sdpa(
        q.rename(.{ .s = .q }),
        k_view,
        v_view,
        .{ .attn_mask = mask(positions, kv.length(), k_view.dim(.k), q.dtype()) },
    );

    var y = p.linear(out.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s }), call.params.must("out"));
    if (call.params.get("out_bias")) |b| {
        y = y.add(p.features(b).convert(y.dtype()).broad(y.shape()));
    }

    const results = try ctx.allocator.alloc(p.Binding, 3);
    results[0] = .{ .name = "output", .tensor = y };
    results[1] = .{ .name = "kv.k", .tensor = updated[0] };
    results[2] = .{ .name = "kv.v", .tensor = updated[1] };
    return results;
}
