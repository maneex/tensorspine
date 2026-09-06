//! The traced function: the plan walked once, emitting MLIR (Z06).
//!
//! Every occurrence becomes a `stablehlo.composite` named for its contract, with the
//! primitive's output as its decomposition. That seam is in from the first commit
//! even though every body is local: without it primitives inline into an
//! undifferentiated blob, and there is nothing for a fetched body or an optimised
//! kernel to plug into later.
//!
//! `zml.ops.composite` takes its decomposition at comptime but its context, name and
//! attributes at run time, which is exactly what a registry-dispatched generator
//! needs: one comptime dispatcher closing over a runtime primitive pointer.
//!
//! Several sessions in one invocation (batch-plan B05, the aligned layout): every value
//! the walk carries is `[sessions, elements, …]`. An occurrence that reads across positions
//! (D1's `across_positions`), holds a state or reads several streams is evaluated once per
//! session, on that session's slice, positions and state view — one composite per session,
//! the state buffers flowing from one to the next; every other occurrence is evaluated once
//! on all the sessions' elements merged into one element axis — its positions merged the
//! same way when its primitive takes them — and its outputs are split back. The primitives
//! see rank-2 values either way and know nothing of the batch — the same split the
//! reference generator's runner makes on its packed layout (harness guide §8).

const std = @import("std");

const zml = @import("zml");

const plan_mod = @import("plan.zig");
const primitive = @import("primitive.zig");
const state_mod = @import("state.zig");

/// What one operand of a composite is bound to inside the primitive. The host fixes
/// this order and the decomposition rebuilds the bindings from it — the same rule the
/// primitive ABI states for a body that arrives from outside.
const Operand = union(enum) {
    input: []const u8,
    param: []const u8,
    state: struct { index: usize, component: []const u8 },
    positions,
    start,
};

const Decomposition = struct {
    step: *const plan_mod.Step,
    plan: *const plan_mod.Plan,
    ctx: *primitive.Ctx,
    operands: []const Operand,
    /// The session this composite evaluates, for an occurrence evaluated per session.
    session: i64 = 0,
};

/// The body of one composite: rebuild the bindings from the block arguments, run the
/// primitive, and return its outputs in the order the host declared them.
fn decompose(args: []zml.Tensor, c: Decomposition) []zml.Tensor {
    const a = c.ctx.allocator;

    var inputs: std.ArrayList(primitive.Binding) = .empty;
    var params: std.ArrayList(primitive.Binding) = .empty;
    var buffers: std.ArrayList(zml.Tensor) = .empty;
    var names: std.ArrayList([]const u8) = .empty;
    var positions: ?zml.Tensor = null;
    var start: ?zml.Tensor = null;

    for (args, c.operands) |tensor, operand| {
        switch (operand) {
            .input => |name| inputs.append(a, .{ .name = name, .tensor = tensor }) catch @panic("out of memory"),
            .param => |name| params.append(a, .{ .name = name, .tensor = tensor }) catch @panic("out of memory"),
            .state => |s| {
                buffers.append(a, tensor) catch @panic("out of memory");
                names.append(a, s.component) catch @panic("out of memory");
            },
            .positions => positions = tensor,
            .start => start = tensor,
        }
    }

    var states: std.ArrayList(primitive.State) = .empty;
    var at: usize = 0;
    for (c.step.states) |binding| {
        const instance = c.plan.states[binding.instance];
        const n = instance.components.len;
        states.append(a, .{
            .name = binding.name,
            .handle = .{
                .law = instance.law,
                .access = instance.access,
                .buffers = buffers.items[at .. at + n],
                .names = names.items[at .. at + n],
                .member = binding.member,
                .session = c.session,
                .start = start.?,
                .elements = c.plan.elements,
            },
        }) catch @panic("out of memory");
        at += n;
    }

    c.ctx.positions = positions;
    const produced = c.step.prim.run(c.ctx, .{
        .occurrence = c.step.node,
        .arguments = c.step.arguments,
        .inputs = .{ .items = inputs.items },
        .params = .{ .items = params.items },
        .states = states.items,
    }) catch |err| std.debug.panic("{s}: {s}@{s} failed: {s}", .{
        c.step.node, c.step.prim.name, c.step.prim.version, @errorName(err),
    });

    // The results, in the order the host declared them: the ports D2 lists, then each
    // state's components. A primitive that produced none of a name it was asked for is
    // a programming error in the primitive, and says so by name.
    const out = a.alloc(zml.Tensor, c.step.outputs.len + buffers.items.len) catch @panic("out of memory");
    var k: usize = 0;
    for (c.step.outputs) |port| {
        out[k] = find(produced, port, c.step);
        k += 1;
    }
    for (c.step.states) |binding| {
        const instance = c.plan.states[binding.instance];
        for (instance.components) |component| {
            const name = std.fmt.allocPrint(a, "{s}.{s}", .{ binding.name, component.name }) catch @panic("out of memory");
            out[k] = find(produced, name, c.step);
            k += 1;
        }
    }
    return out;
}

fn find(produced: []const primitive.Binding, name: []const u8, step: *const plan_mod.Step) zml.Tensor {
    for (produced) |b| {
        if (std.mem.eql(u8, b.name, name)) return b.tensor;
    }
    std.debug.panic("{s}: {s} produced no '{s}'", .{ step.node, step.prim.name, name });
}

/// What one program returns: the values crossing its outgoing boundary (the last
/// group's carries the plan's result), and every state buffer as it stands afterwards.
/// States are functional here — in as operands, out as results.
pub const Result = struct {
    outputs: []zml.Tensor,
    states: []zml.Tensor,
};

/// A `[sessions, elements, …]` shape with its first two axes merged: what a union step's
/// primitive sees.
fn merged(sh: zml.Shape) zml.Shape {
    var out = zml.Shape.init(.{}, sh.dtype());
    out = out.appendDim(sh.dim(0) * sh.dim(1), null);
    for (sh.dims()[2..]) |d| out = out.appendDim(d, null);
    return out;
}

/// A `[sessions, elements, …]` shape for one session: what a per-session step's primitive sees.
fn one(sh: zml.Shape) zml.Shape {
    return sh.remove(0);
}

/// Walk the plan. Traced by `zml.module.compile`, so the slices' run-time lengths fix
/// the arity of the MLIR function — which is what lets one Zig type serve every
/// document.
pub fn forward(
    handle: plan_mod.Handle,
    params: []const zml.Tensor,
    publics: []const zml.Tensor,
    carried: []const zml.Tensor,
    start: zml.Tensor,
    states_in: []const zml.Tensor,
) Result {
    const p = handle.plan();
    const group = p.groups[handle.group];
    const cc = zml.module.CompilationContext.current();

    // Everything the walk allocates dies with the walk: the emitted MLIR holds the
    // values, not these bindings. An arena is both the freeing strategy and the
    // statement that a primitive may allocate nothing longer-lived.
    var arena: std.heap.ArenaAllocator = .init(cc.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var ctx: primitive.Ctx = .{ .allocator = a, .compute = p.compute };
    const sessions: usize = @intCast(p.batch);
    const elements_shape = zml.Shape.init(.{p.elements}, .i32);

    // States live across the whole walk: written by one step, read by the next. They
    // are returned, so they outlive the arena — `collectOutputInfo` reads the result
    // after this function returns, and the compilation context's own arena is what
    // lasts that long.
    const states = cc.alloc(zml.Tensor, states_in.len);
    @memcpy(states, states_in);

    // Every value the walk carries is `[sessions, elements, …]` — a public input, a value
    // carried from an earlier program, a step's outputs.
    const produced = a.alloc([]zml.Tensor, p.steps.len) catch @panic("out of memory");
    for (p.steps[group.first..group.last], produced[group.first..group.last]) |*step, *results| {
        const sources = a.alloc(zml.Tensor, step.inputs.len) catch @panic("out of memory");
        for (step.inputs, sources) |in, *src| {
            src.* = switch (in.source) {
                // Inside the group it is a value; from an earlier group it entered as an
                // argument of this program.
                .value => |v| if (v.step >= group.first) produced[v.step][v.out] else blk: {
                    for (group.inputs, carried) |b, tensor| {
                        if (b.step == v.step and b.out == v.out) break :blk tensor;
                    }
                    std.debug.panic("{s}: nothing carries {d}.{d} into group {d}", .{ step.node, v.step, v.out, handle.group });
                },
                .public => |i| publics[i],
            };
        }
        results.* = a.alloc(zml.Tensor, step.outputs.len) catch @panic("out of memory");

        if (!step.per_session) {
            // The union: every session's elements on one element axis, one composite.
            var operands: std.ArrayList(Operand) = .empty;
            var tensors: std.ArrayList(zml.Tensor) = .empty;
            var shapes: std.ArrayList(zml.Shape) = .empty;
            for (step.inputs, sources) |in, src| {
                operands.append(a, .{ .input = in.port }) catch @panic("out of memory");
                tensors.append(a, src.reshape(merged(src.shape()))) catch @panic("out of memory");
            }
            appendParams(a, step, group, params, handle.group, &operands, &tensors);
            if (step.prim.needs_positions) {
                // The positions of every session's elements, one session after the other:
                // the union's element axis is the sessions' elements concatenated, and so
                // are their positions (a position embedding on the union).
                const parts = a.alloc(zml.Tensor, sessions) catch @panic("out of memory");
                for (parts, 0..) |*t, session| {
                    const start_s = start.slice(0, .single(@as(i64, @intCast(session))));
                    t.* = start_s.convert(.i32).broad(elements_shape).add(zml.Tensor.iota(elements_shape, 0));
                }
                operands.append(a, .positions) catch @panic("out of memory");
                tensors.append(a, zml.Tensor.concatenate(parts, 0)) catch @panic("out of memory");
            }
            for (step.shapes) |shape| shapes.append(a, merged(shape)) catch @panic("out of memory");
            const emitted = zml.ops.composite(step.composite, tensors.items, shapes.items, decompose, Decomposition{
                .step = step,
                .plan = p,
                .ctx = &ctx,
                .operands = operands.items,
            }, .{});
            for (results.*, emitted, step.shapes) |*r, t, shape| r.* = t.reshape(shape);
            continue;
        }

        // Per session: its slice of every input, its positions and start, its view of the
        // states — the buffers flowing from one session's composite to the next.
        const parts = a.alloc([]zml.Tensor, sessions) catch @panic("out of memory");
        for (0..sessions) |session| {
            const s_i64: i64 = @intCast(session);
            const start_s = start.slice(0, .single(s_i64));
            const positions = start_s.convert(.i32).broad(elements_shape).add(zml.Tensor.iota(elements_shape, 0));

            var operands: std.ArrayList(Operand) = .empty;
            var tensors: std.ArrayList(zml.Tensor) = .empty;
            var shapes: std.ArrayList(zml.Shape) = .empty;
            for (step.inputs, sources) |in, src| {
                operands.append(a, .{ .input = in.port }) catch @panic("out of memory");
                tensors.append(a, src.slice(0, .single(s_i64))) catch @panic("out of memory");
            }
            appendParams(a, step, group, params, handle.group, &operands, &tensors);
            for (step.shapes) |shape| shapes.append(a, one(shape)) catch @panic("out of memory");
            for (step.states) |binding| {
                const instance = p.states[binding.instance];
                for (instance.components, 0..) |component, i| {
                    operands.append(a, .{ .state = .{ .index = binding.base + i, .component = component.name } }) catch @panic("out of memory");
                    tensors.append(a, states[binding.base + i]) catch @panic("out of memory");
                    shapes.append(a, component.shape) catch @panic("out of memory");
                }
            }
            operands.append(a, .positions) catch @panic("out of memory");
            tensors.append(a, positions) catch @panic("out of memory");
            operands.append(a, .start) catch @panic("out of memory");
            tensors.append(a, start_s) catch @panic("out of memory");

            const emitted = zml.ops.composite(step.composite, tensors.items, shapes.items, decompose, Decomposition{
                .step = step,
                .plan = p,
                .ctx = &ctx,
                .operands = operands.items,
                .session = s_i64,
            }, .{});
            parts[session] = emitted[0..step.outputs.len];
            var k = step.outputs.len;
            for (step.states) |binding| {
                const instance = p.states[binding.instance];
                for (instance.components, 0..) |_, i| {
                    states[binding.base + i] = emitted[k];
                    k += 1;
                }
            }
        }
        for (results.*, step.shapes, 0..) |*r, shape, out| {
            const stacked = a.alloc(zml.Tensor, sessions) catch @panic("out of memory");
            for (stacked, 0..) |*t, session| t.* = parts[session][out].reshape(shape.setDim(0, 1));
            r.* = zml.Tensor.concatenate(stacked, 0);
        }
    }

    // Returned, so from the compilation context's arena and not the walk's.
    const outputs = cc.alloc(zml.Tensor, group.outputs.len);
    for (group.outputs, outputs) |b, *t| t.* = produced[b.step][b.out];

    return .{ .outputs = outputs, .states = states };
}

fn appendParams(
    a: std.mem.Allocator,
    step: *const plan_mod.Step,
    group: plan_mod.Group,
    params: []const zml.Tensor,
    which: usize,
    operands: *std.ArrayList(Operand),
    tensors: *std.ArrayList(zml.Tensor),
) void {
    for (step.params) |slot| {
        operands.append(a, .{ .param = slot.slot }) catch @panic("out of memory");
        const at = std.mem.indexOfScalar(usize, group.params, slot.param) orelse
            std.debug.panic("{s}: group {d} was not given parameter {d}", .{ step.node, which, slot.param });
        tensors.append(a, params[at]) catch @panic("out of memory");
    }
}
