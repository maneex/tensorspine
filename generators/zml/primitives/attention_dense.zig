//! attention.dense@1.0.0 — dense or grouped-query attention over a KV state.
//!
//! | branch                          | status                                     |
//! |---------------------------------|--------------------------------------------|
//! | mask causal                     | implemented                                |
//! | mask none (stateless encoder)   | implemented, with or without a state       |
//! | mask chunked, window            | refused                                    |
//! | cross, streaming                | refused                                    |
//! | rope theta, layout split        | implemented (rotate-half)                  |
//! | rope layout interleaved / 2d    | refused                                    |
//! | rope partial                    | implemented: the first `partial · head_dim` channels |
//! | rope mrope                      | implemented for one position stream: every section is rotated by the same position, which is plain RoPE whatever `sections` says |
//! | rope scaling (yarn, llama3, …)  | refused                                    |
//! | qk_norm kind rms                | implemented, over head_dim and before RoPE |
//! | qk_norm kind layer              | refused                                    |
//! | temperature                     | refused                                    |
//! | output_gate                     | implemented: per head, query rows then gate rows of `q_gated` |
//! | q/k/v/out biases                | implemented                                |
//!
//! Conventions, as the contract now states them: the keys of the current elements are
//! appended to the state before the queries attend, so a query sees itself; the scale
//! is head_dim^-1/2; `split` pairs channel i with i + rotary/2 (rotate-half) over the
//! rotated channels only, whose base frequencies are computed on the rotated width.
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
    \\               "output_gate": "any",
    \\               "qk_norm": {"absent": true, "fields": {"kind": ["rms"], "eps": "any",
    \\                           "scale": {"absent": true, "fields": {"zero_centered": "any"}}}},
    \\               "rope": {"absent": true, "fields": {"theta": "any", "layout": ["split"],
    \\                        "partial": "any", "scaling": "absent",
    \\                        "mrope": {"absent": true, "fields": {"t": "any", "h": "any", "w": "any",
    \\                                  "sections": ["contiguous", "interleaved"]}}}},
    \\               "q_bias": "any", "k_bias": "any", "v_bias": "any", "out_bias": "any"},
    \\ "states": ["append"],
    \\ "notes": ["rope layout split is emitted as ZML's real_im_pass, the same rotate-half pairing",
    \\           "mrope for one position stream only: an image would need the sections to differ"]}
    ,
};

/// q, k or v: projected, then split into heads — [.s, .h, .hd].
fn project(x: zml.Tensor, w: zml.Tensor, heads: i64) zml.Tensor {
    return p.linear(x, w).splitAxis(.d, .{ .h = heads, .hd = .auto });
}

/// A `[heads · head_dim]` bias, added per head. The contract declares it flat and the
/// factors it names are `heads` and `head_dim`, so it is the same addition whether the
/// query rows came from `q` or from `q_gated`.
fn bias(x: zml.Tensor, b: zml.Tensor) zml.Tensor {
    const per_head = p.features(b).splitAxis(.d, .{ .h = x.dim(.h), .hd = .auto });
    return x.add(per_head.convert(x.dtype()).broad(x.shape()));
}

/// The RMS norm the contract puts on queries and keys, over head_dim, before RoPE.
/// `qk_norm.scale` present means the occurrence declares `q_norm` and `k_norm`;
/// `zero_centered` stores them as an offset from one, as `norm.rms` does.
fn qkNorm(x: zml.Tensor, weight: ?zml.Tensor, eps: f32, zero_centered: bool) zml.Tensor {
    const y = zml.nn.rmsNorm(x, .hd, eps);
    const w = weight orelse return y;
    var scale = p.features(w).rename(.{ .d = .hd }).convert(y.dtype());
    if (zero_centered) scale = scale.addConstant(1);
    return y.mul(scale.broad(y.shape()));
}

/// Rotate-half over the first `rotary` channels of each head, the rest passed through.
///
/// The split is made here rather than left to ZML because ZML carries a partial factor
/// only on its `yarn` and `proportional` scalings, and `proportional` computes the base
/// frequencies on the *full* head. The contract's are computed on the rotated width — so
/// handing ZML a whole, narrower head is what makes `.default` compute the frequencies
/// the document asks for.
fn rotate(x: zml.Tensor, positions: zml.Tensor, theta: f32, rotary: i64) zml.Tensor {
    const opts: zml.nn.RopeOpts = .{
        .layout = .real_im_pass,
        .scaling = .{ .default = .{ .rope_theta = theta } },
    };
    const idx = positions.withTags(.{.s});
    if (rotary == x.dim(.hd)) return zml.nn.rope(x, idx, opts);
    return zml.Tensor.concatenate(&.{
        zml.nn.rope(x.slice(.hd, .{ .end = rotary }), idx, opts),
        x.slice(.hd, .{ .start = rotary }),
    }, .hd);
}

/// Which keys each query may attend to, as the additive mask `sdpa` takes: the
/// positions the state holds, and — when causal — those at or before the query's own
/// position. Masking over the whole capacity rather than slicing to the length is what
/// gives one code path whatever the capacity, which a compiled graph needs.
fn mask(positions: zml.Tensor, length: zml.Tensor, keys: i64, dt: zml.DataType, causal: bool) zml.Tensor {
    const qk = zml.Shape.init(.{ .q = positions.dim(0), .k = keys }, .i32);
    const kidx = zml.Tensor.iota(qk, .k);

    var allowed = kidx.cmp(.LT, length.convert(.i32).broad(qk));
    if (causal) {
        allowed = allowed.logical(.AND, kidx.cmp(.LE, positions.convert(.i32).withTags(.{.q}).broad(qk)));
    }
    return zml.Tensor.select(allowed, .scalar(0, dt), .scalar(-std.math.inf(f32), dt));
}

fn float(value: std.json.Value) !f64 {
    return switch (value) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => p.Error.MissingArgument,
    };
}

fn string(value: std.json.Value) ![]const u8 {
    return switch (value) {
        .string => |s| s,
        else => p.Error.MissingArgument,
    };
}

fn run(ctx: *p.Ctx, call: p.Call) ![]const p.Binding {
    const x = call.inputs.must("input").withTags(.{ .s, .d });
    const heads = try call.requireInt("heads");
    const kv_heads = try call.requireInt("kv_heads");
    const head_dim = try call.requireInt("head_dim");
    const positions = ctx.positions orelse return p.Error.MissingArgument;

    // The queries, and the per-head gate when the projection carries its rows: the
    // contract puts them in `q_gated`, per head, query rows then gate rows.
    var gate: ?zml.Tensor = null;
    var q = q: {
        if (!call.argBool("output_gate")) break :q project(x, call.params.must("q"), heads);
        const both = p.linear(x, call.params.must("q_gated"))
            .splitAxis(.d, .{ .h = heads, .rows = 2, .hd = .auto });
        gate = both.slice(.rows, .single(1));
        break :q both.slice(.rows, .single(0));
    };
    var k = project(x, call.params.must("k"), kv_heads);
    var v = project(x, call.params.must("v"), kv_heads);

    if (call.params.get("q_bias")) |b| q = bias(q, b);
    if (call.params.get("k_bias")) |b| k = bias(k, b);
    if (call.params.get("v_bias")) |b| v = bias(v, b);

    if (call.argObj("qk_norm")) |qk| {
        if (!std.mem.eql(u8, try string(qk.get("kind") orelse return p.Error.MissingArgument), "rms")) {
            return p.Error.Unimplemented;
        }
        const eps: f32 = @floatCast(try float(qk.get("eps") orelse return p.Error.MissingArgument));
        // `scale` present is what declares q_norm and k_norm; absent, the norm is bare.
        const zero_centered = switch (qk.get("scale") orelse std.json.Value.null) {
            .object => |scale| switch (scale.get("zero_centered") orelse std.json.Value{ .bool = false }) {
                .bool => |b| b,
                else => false,
            },
            else => false,
        };
        q = qkNorm(q, call.params.get("q_norm"), eps, zero_centered);
        k = qkNorm(k, call.params.get("k_norm"), eps, zero_centered);
    }

    if (call.argObj("rope")) |rope| {
        if (!std.mem.eql(u8, try string(rope.get("layout") orelse return p.Error.MissingArgument), "split")) {
            return p.Error.Unimplemented;
        }
        if (rope.get("scaling") != null) return p.Error.Unimplemented;
        const theta: f32 = @floatCast(try float(rope.get("theta") orelse return p.Error.MissingArgument));

        // `mrope` indexes one occurrence by several position streams. This generator
        // has one — the manifest says so, `domains.kinds` naming `token` alone — and
        // with one stream every section is rotated by the same position, which is plain
        // RoPE whatever `sections` says. An image would tell the two layouts apart; a
        // text document cannot, and this is the reading the reference generator records.
        const rotary: i64 = if (rope.get("partial")) |part|
            @intFromFloat(@as(f64, @floatFromInt(head_dim)) * try float(part))
        else
            head_dim;
        q = rotate(q, positions, theta, rotary);
        k = rotate(k, positions, theta, rotary);
    }

    const causal = std.mem.eql(u8, call.argStr("mask") orelse "causal", "causal");
    if (!causal and !std.mem.eql(u8, call.argStr("mask").?, "none")) return p.Error.Unimplemented;

    // With a state, the keys and values of this invocation join it before the queries
    // attend, so a query sees itself — the convention the contract states. Without one —
    // a stateless encoder, which D4 gives no state at all — they attend over this
    // invocation's own elements and nothing else.
    var k_view = k.rename(.{ .s = .k });
    var v_view = v.rename(.{ .s = .k });
    var updated: []zml.Tensor = &.{};
    var attn_mask: ?zml.Tensor = null;

    if (call.state("kv")) |kv| {
        updated = try kv.append(ctx.allocator, &.{ k, v });
        const written = kv.after(updated);
        k_view = written.get("k").?.withTags(.{ .k, .h, .hd });
        v_view = written.get("v").?.withTags(.{ .k, .h, .hd });
        attn_mask = mask(positions, kv.length(), k_view.dim(.k), q.dtype(), causal);
    } else if (causal) {
        attn_mask = mask(positions, zml.Tensor.scalar(k_view.dim(.k), .i32), k_view.dim(.k), q.dtype(), true);
    }

    const out = zml.nn.sdpa(
        q.rename(.{ .s = .q }),
        k_view.convert(q.dtype()),
        v_view.convert(q.dtype()),
        .{ .attn_mask = attn_mask },
    );

    var merged = out.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
    if (gate) |g| merged = merged.mul(g.merge(.{ .d = .{ .h, .hd } }).sigmoid().convert(merged.dtype()));

    var y = p.linear(merged, call.params.must("out"));
    if (call.params.get("out_bias")) |b| {
        y = y.add(p.features(b).convert(y.dtype()).broad(y.shape()));
    }

    if (updated.len == 0) return p.one(ctx, "output", y);

    const results = try ctx.allocator.alloc(p.Binding, 3);
    results[0] = .{ .name = "output", .tensor = y };
    results[1] = .{ .name = "kv.k", .tensor = updated[0] };
    results[2] = .{ .name = "kv.v", .tensor = updated[1] };
    return results;
}
