//! A session: the weights resident, the states carried, and one or more compiled
//! arities over them.
//!
//! A compiled graph has static shapes, so an invocation of n elements is its own
//! program; and a long graph is cut into several programs run in sequence, because
//! XLA's scratch for one program holds an f32 copy of every weight that program's
//! matmuls touch. Both are serving choices — the numbers do not move — so both are
//! arguments.

const std = @import("std");

const zml = @import("zml");

const emit = @import("emit.zig");
const graph = @import("graph.zig");
const loader = @import("loader.zig");
const plan = @import("plan.zig");

const log = std.log.scoped(.tspl);

pub const Compiled = struct {
    plan: plan.Plan,
    exes: []zml.Exe,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        g: *const graph.Graph,
        target: []const u8,
        elements: i64,
        capacity: i64,
        compute: zml.DataType,
        packed_states: bool,
        split: usize,
        params: []const zml.Tensor,
    ) !Compiled {
        var p = try plan.until(allocator, g, target, elements, capacity, compute, packed_states);
        errdefer p.deinit();
        try p.split(split);

        const publics = try allocator.alloc(zml.Tensor, p.publics.len);
        defer allocator.free(publics);
        for (p.public_shapes, publics) |shape, *t| t.* = .fromShape(shape);

        const states = try allocator.alloc(zml.Tensor, p.state_shapes.len);
        defer allocator.free(states);
        for (p.state_shapes, states) |shape, *t| t.* = .fromShape(shape);

        const start: zml.Tensor = .fromShape(zml.Shape.init(.{}, .i32));

        const exes = try allocator.alloc(zml.Exe, p.groups.len);
        errdefer allocator.free(exes);
        for (p.groups, exes, 0..) |group, *exe, i| {
            const group_params = try allocator.alloc(zml.Tensor, group.params.len);
            defer allocator.free(group_params);
            for (group.params, group_params) |at, *t| t.* = params[at];

            const carried = try allocator.alloc(zml.Tensor, group.inputs.len);
            defer allocator.free(carried);
            for (group.inputs, carried) |b, *t| t.* = .fromShape(b.shape);

            log.info("compiling group {d}/{d}: steps {d}..{d}, {d} param(s), {d} carried, {d} out", .{
                i + 1, p.groups.len, group.first, group.last, group.params.len, group.inputs.len, group.outputs.len,
            });
            exe.* = try zml.module.compile(allocator, io, emit.forward, .{
                plan.Handle.ofGroup(&p, i), group_params, publics, carried, start, states,
            }, platform, .{ .program_name = g.model() });
        }
        return .{ .plan = p, .exes = exes };
    }

    pub fn deinit(self: *Compiled, allocator: std.mem.Allocator) void {
        for (self.exes) |exe| exe.deinit();
        allocator.free(self.exes);
        self.plan.deinit();
    }
};
pub fn invoke(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    c: *const Compiled,
    params: []zml.Buffer,
    ids: []const i32,
    start: i32,
    states: []zml.Buffer,
    out: *zml.Buffer,
) !void {
    var tokens = try zml.Buffer.fromSlice(
        io,
        platform,
        .init(c.plan.public_shapes[0], std.mem.sliceAsBytes(ids)),
        platform.replicated_sharding,
    );
    defer tokens.deinit();

    var start_buffer = try zml.Buffer.scalar(io, platform, start, .i32);
    defer start_buffer.deinit();

    // Values that have crossed a boundary and are still needed, keyed as the plan keys
    // them. A program's scratch is gone by the time the next one starts; these are not.
    var live: std.ArrayList(struct { step: usize, out: usize, buffer: zml.Buffer }) = .empty;
    defer live.deinit(allocator);
    defer for (live.items) |*item| item.buffer.deinit();

    for (c.plan.groups, c.exes) |group, exe| {
        const group_params = try allocator.alloc(zml.Buffer, group.params.len);
        defer allocator.free(group_params);
        for (group.params, group_params) |at, *b| b.* = params[at];

        const carried = try allocator.alloc(zml.Buffer, group.inputs.len);
        defer allocator.free(carried);
        for (group.inputs, carried) |want, *b| {
            b.* = for (live.items) |item| {
                if (item.step == want.step and item.out == want.out) break item.buffer;
            } else return error.MissingBoundaryValue;
        }

        var call_args = try exe.args(allocator);
        defer call_args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);

        call_args.set(.{ group_params, tokens, carried, start_buffer, states });
        exe.call(call_args, &results);

        const produced = try allocator.alloc(zml.Buffer, group.outputs.len);
        defer allocator.free(produced);
        const after = try allocator.alloc(zml.Buffer, states.len);
        defer allocator.free(after);
        results.fill(.{ &produced, &after });

        for (states, after) |*before, next| {
            before.deinit();
            before.* = next;
        }
        for (group.outputs, produced) |b, buffer| {
            try live.append(allocator, .{ .step = b.step, .out = b.out, .buffer = buffer });
        }
    }

    // The plan's result is the last value the last group produced.
    const last = c.plan.groups[c.plan.groups.len - 1];
    const wanted = last.outputs[last.outputs.len - 1];
    for (live.items, 0..) |item, i| {
        if (item.step == wanted.step and item.out == wanted.out) {
            out.* = item.buffer;
            _ = live.swapRemove(i);
            return;
        }
    }
    return error.MissingResult;
}

/// The identifier of the largest logit of the last element — greedy decoding. Sampling
/// is the serving application's, not the document's: nothing in D1–D6 mentions it. The
/// logits arrive in whatever the compute dtype is, so the comparison decodes rather than
/// assumes.
pub fn argmaxLast(bytes: []const u8, dt: zml.DataType, vocabulary: usize) !i32 {
    const width = dt.sizeOf();
    const start = bytes.len - vocabulary * width;
    var best: usize = 0;
    var best_value: f32 = -std.math.inf(f32);
    for (0..vocabulary) |i| {
        const at = bytes[start + i * width ..];
        const value: f32 = switch (dt) {
            .f32 => @bitCast(std.mem.readInt(u32, at[0..4], .little)),
            .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, at[0..2], .little)) << 16),
            else => return error.UnsupportedComputeDtype,
        };
        if (value > best_value) {
            best_value = value;
            best = i;
        }
    }
    return @intCast(best);
}
