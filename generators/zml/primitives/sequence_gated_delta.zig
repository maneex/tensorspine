//! sequence.gated_delta@1.0.0 — a per-head matrix state updated by the delta rule,
//! behind a short causal convolution (Qwen 3.5's linear-attention layers).
//!
//! | branch / record             | status                                        |
//! |-----------------------------|-----------------------------------------------|
//! | conv (kernel, history)      | implemented, depthwise causal, a `window` state |
//! | no conv                     | refused                                       |
//! | out_gate silu               | implemented                                   |
//! | out_gate none/sigmoid/swish | refused                                       |
//! | value_heads > key_heads     | implemented: each q/k head serves `value_heads / key_heads` value heads, in order |
//!
//! Two states, and they are why this contract is worth having in a second generator:
//! `recurrent` is `fixed`/`aggregate` — read whole, written whole, zero bytes per token —
//! and `conv` is `window`/`ring`, a three-position history. Neither grows with the
//! sequence, so a document made of these has no KV cache to page at all.
//!
//! The recurrence is written position by position, as the reference generator writes it:
//! exact, and unrolled by the tracer into one block per element of the invocation. That
//! is affordable because this generator compiles one arity — a chat feeds its prompt a
//! token at a time (`chat.zig`) — and it is the form the contract's description states.
//! A chunked form would be the same mathematics reassociated, and would have to be
//! justified against this one rather than assumed.
//!
//! Conventions the contract states and this reads out of it: β = σ(b);
//! g = −exp(A_log) · softplus(a + dt_bias); q and k are L2-normalised over `head_dim`
//! (epsilon 1e-6) and q scaled by head_dim^-1/2; the state decays by exp(g), then takes
//! k ⊗ ((v − Sᵀk) β), and is read as Sᵀq; the read-out is RMS-normalised per head, scaled
//! by `norm`, and gated by silu(z). The read-out norm's epsilon is the one the reference
//! implementation uses, 1e-6; the contract does not state it.

const std = @import("std");

const zml = @import("zml");

const p = @import("../primitive.zig");

pub const primitive: p.Primitive = .{
    .name = "sequence.gated_delta",
    .version = "1.0.0",
    .run = run,
    .capabilities =
    \\{"arguments": {"width": "any", "key_heads": "any", "value_heads": "any", "head_dim": "any",
    \\               "out_gate": ["silu"],
    \\               "conv": {"absent": false, "fields": {"width": "any", "kernel": "any", "history": "any"}}},
    \\ "states": ["window", "fixed"],
    \\ "notes": ["the recurrence position by position, unrolled by the tracer: exact, and one block per element"]}
    ,
};

/// softplus(x) = log(1 + eˣ), by the identity max(x, 0) + log(1 + e^−|x|) — neither
/// StableHLO nor ZML binds one, and the naive form overflows on the positive side.
fn softplus(x: zml.Tensor) zml.Tensor {
    const one: zml.Tensor = .scalar(1, x.dtype());
    return x.relu().add(one.broad(x.shape()).add(x.abs().scale(-1).exp()).log());
}

/// x · rsqrt(Σx² + eps) over `axis`. A true L2 normalisation, not an RMS one: the sum of
/// squares, where `norm.rms` would take their mean.
fn l2norm(x: zml.Tensor, axis: anytype) zml.Tensor {
    const sq = x.mul(x).sum(axis).addConstant(1e-6);
    return x.mul(zml.Tensor.rsqrt(sq).broad(x.shape()));
}

/// `.hq` query/key heads spread over `.h` value heads, `rep` value heads to each, in
/// order — the contract's rule when `value_heads` exceeds `key_heads`.
fn spread(x: zml.Tensor, rep: i64) zml.Tensor {
    if (rep == 1) return x.rename(.{ .hq = .h });
    const wide = x.insertAxes(.hd, .{.rep});
    return wide.broad(wide.shape().setDim(.rep, rep)).merge(.{ .h = .{ .hq, .rep } });
}

fn run(ctx: *p.Ctx, call: p.Call) ![]const p.Binding {
    const x = call.inputs.must("input").withTags(.{ .s, .d });
    const kh = try call.requireInt("key_heads");
    const vh = try call.requireInt("value_heads");
    const hd = try call.requireInt("head_dim");
    const n = x.dim(.s);

    if (!std.mem.eql(u8, call.argStr("out_gate") orelse "none", "silu")) return p.Error.Unimplemented;
    const conv_arg = call.argObj("conv") orelse return p.Error.Unimplemented;
    const kernel: i64 = switch (conv_arg.get("kernel") orelse return p.Error.MissingArgument) {
        .integer => |i| i,
        else => return p.Error.MissingArgument,
    };
    const conv = call.state("conv") orelse return p.Error.Unimplemented;
    const recurrent = call.state("recurrent") orelse return p.Error.Unimplemented;

    // The projections: queries, keys and values fused, the output gate, and the two
    // per-head decay controls.
    const mixed = p.linear(x, call.params.must("qkv"));
    const z = p.linear(x, call.params.must("z"));
    const beta = p.linear(x, call.params.must("b")).sigmoid().rename(.{ .d = .h });
    const a = p.linear(x, call.params.must("a")).rename(.{ .d = .h }).convert(.f32);

    // The causal depthwise convolution: the positions the state holds, then this
    // invocation's own, weighted over the kernel's support. A state that has held fewer
    // than `history` positions is still zero in front, which is the padding a causal
    // convolution wants at the start of a sequence.
    const history = conv.get("w").?.withTags(.{ .s, .d });
    const seq = zml.Tensor.concatenate(&.{ history, mixed.convert(history.dtype()) }, .s);
    const weight = call.params.must("conv_weight");
    var acc: ?zml.Tensor = null;
    for (0..@intCast(kernel)) |j| {
        const at: i64 = @intCast(j);
        const window = seq.slice(.s, .{ .start = at, .end = at + n });
        const tap = p.features(weight.slice(1, .single(at))).convert(window.dtype());
        const term = window.mul(tap.broad(window.shape()));
        acc = if (acc) |sum| sum.add(term) else term;
    }
    const gated = acc.?.silu();
    const conv_state = try conv.append(ctx.allocator, &.{mixed});

    const qk = kh * hd;
    const rep = @divExact(vh, kh);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    const q = spread(l2norm(gated.slice(.d, .{ .start = 0, .end = qk })
        .splitAxis(.d, .{ .hq = kh, .hd = .auto }), .hd).scale(scale), rep);
    const k = spread(l2norm(gated.slice(.d, .{ .start = qk, .end = 2 * qk })
        .splitAxis(.d, .{ .hq = kh, .hd = .auto }), .hd), rep);
    const v = gated.slice(.d, .{ .start = 2 * qk, .end = 2 * qk + vh * hd })
        .splitAxis(.d, .{ .h = vh, .hd = .auto });

    // The log decay, per position and value head.
    const dt_bias = p.features(call.params.must("dt_bias")).convert(.f32).rename(.{ .d = .h });
    const a_log = p.features(call.params.must("A_log")).convert(.f32).rename(.{ .d = .h });
    const g = softplus(a.add(dt_bias.broad(a.shape())))
        .mul(a_log.exp().broad(a.shape()))
        .scale(-1);

    // The rule itself, in f32: rows of the state are indexed by the key, columns by the
    // value, so a read is Sᵀq and the update is the outer product k ⊗ delta.
    var state = recurrent.get("s").?.withTags(.{ .h, .row, .col }).convert(.f32);
    const shape = state.shape();
    const reads = try ctx.allocator.alloc(zml.Tensor, @intCast(n));
    for (reads, 0..) |*read, i| {
        const at: i64 = @intCast(i);
        const gi = g.slice(.s, .single(at)).insertAxes(.last, .{ .row, .col });
        const bi = beta.slice(.s, .single(at)).insertAxes(.last, .{ .row, .col }).convert(.f32);
        const ki = k.slice(.s, .single(at)).rename(.{ .hd = .row }).insertAxes(.last, .{.col}).convert(.f32);
        const qi = q.slice(.s, .single(at)).rename(.{ .hd = .row }).insertAxes(.last, .{.col}).convert(.f32);
        const vi = v.slice(.s, .single(at)).rename(.{ .hd = .col }).insertAxes(.col, .{.row}).convert(.f32);

        state = state.mul(gi.exp().broad(shape));
        const memory = state.mul(ki.broad(shape)).sum(.row);
        const delta = vi.sub(memory).mul(bi.broad(memory.shape()));
        state = state.add(ki.broad(shape).mul(delta.broad(shape)));
        read.* = state.mul(qi.broad(shape)).sum(.row)
            .slice(.row, .single(0)).rename(.{ .col = .hd }).insertAxes(0, .{.s});
    }
    const recurrent_state = try recurrent.write(ctx.allocator, &.{state});

    // The read-out: normalised per head, gated, projected back.
    var out = zml.Tensor.concatenate(reads, .s).convert(x.dtype());
    const norm = p.features(call.params.must("norm")).convert(out.dtype()).rename(.{ .d = .hd });
    out = zml.nn.rmsNorm(out, .hd, 1e-6).mul(norm.broad(out.shape()));
    out = out.mul(z.splitAxis(.d, .{ .h = vh, .hd = .auto }).silu());

    const results = try ctx.allocator.alloc(p.Binding, 3);
    results[0] = .{ .name = "output", .tensor = p.linear(out.merge(.{ .d = .{ .h, .hd } }), call.params.must("out")) };
    results[1] = .{ .name = "conv.w", .tensor = conv_state[0] };
    results[2] = .{ .name = "recurrent.s", .tensor = recurrent_state[0] };
    return results;
}
