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
    for (c.step.states) |binding| {
        const instance = c.plan.states[binding.instance];
        const n = instance.components.len;
        states.append(a, .{
            .name = binding.name,
            .handle = .{
                .law = instance.law,
                .access = instance.access,
                .buffers = buffers.items[0..n],
                .names = names.items[0..n],
                .start = start.?,
                .elements = c.plan.elements,
            },
        }) catch @panic("out of memory");
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

/// What the traced function returns: the value asked for, and every state buffer as it
/// stands afterwards. States are functional here — in as operands, out as results —
/// which is what lets the caller donate them.
pub const Result = struct {
    value: zml.Tensor,
    states: []zml.Tensor,
};

/// Walk the plan. Traced by `zml.module.compile`, so the slices' run-time lengths fix
/// the arity of the MLIR function — which is what lets one Zig type serve every
/// document.
pub fn forward(
    handle: plan_mod.Handle,
    params: []const zml.Tensor,
    publics: []const zml.Tensor,
    start: zml.Tensor,
    states_in: []const zml.Tensor,
) Result {
    const p = handle.plan();
    const cc = zml.module.CompilationContext.current();

    // Everything the walk allocates dies with the walk: the emitted MLIR holds the
    // values, not these bindings. An arena is both the freeing strategy and the
    // statement that a primitive may allocate nothing longer-lived.
    var arena: std.heap.ArenaAllocator = .init(cc.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var ctx: primitive.Ctx = .{ .allocator = a, .compute = p.compute };

    // The positions of this invocation's elements: where the state already reaches,
    // plus one per element. One stream today; a document with several would index this
    // by the stream each occurrence belongs to.
    const positions = start.convert(.i32).broad(zml.Shape.init(.{p.elements}, .i32))
        .add(zml.Tensor.iota(zml.Shape.init(.{p.elements}, .i32), 0));

    // States live across the whole walk: written by one step, read by the next.
    const states = a.alloc(zml.Tensor, states_in.len) catch @panic("out of memory");
    @memcpy(states, states_in);

    const produced = a.alloc([]zml.Tensor, p.steps.len) catch @panic("out of memory");
    for (p.steps, produced) |*step, *results| {
        var operands: std.ArrayList(Operand) = .empty;
        var tensors: std.ArrayList(zml.Tensor) = .empty;
        var shapes: std.ArrayList(zml.Shape) = .empty;

        for (step.inputs) |in| {
            operands.append(a, .{ .input = in.port }) catch @panic("out of memory");
            tensors.append(a, switch (in.source) {
                .value => |v| produced[v.step][v.out],
                .public => |i| publics[i],
            }) catch @panic("out of memory");
        }
        for (step.params) |slot| {
            operands.append(a, .{ .param = slot.slot }) catch @panic("out of memory");
            tensors.append(a, params[slot.param]) catch @panic("out of memory");
        }
        for (step.outputs, step.shapes) |_, shape| {
            shapes.append(a, shape) catch @panic("out of memory");
        }
        for (step.states) |binding| {
            const instance = p.states[binding.instance];
            for (instance.components, 0..) |component, i| {
                operands.append(a, .{ .state = .{ .index = binding.base + i, .component = component.name } }) catch @panic("out of memory");
                tensors.append(a, states[binding.base + i]) catch @panic("out of memory");
                shapes.append(a, component.shape) catch @panic("out of memory");
            }
        }
        if (step.prim.needs_positions or step.states.len > 0) {
            operands.append(a, .positions) catch @panic("out of memory");
            tensors.append(a, positions) catch @panic("out of memory");
            operands.append(a, .start) catch @panic("out of memory");
            tensors.append(a, start) catch @panic("out of memory");
        }

        const emitted = zml.ops.composite(step.composite, tensors.items, shapes.items, decompose, Decomposition{
            .step = step,
            .plan = p,
            .ctx = &ctx,
            .operands = operands.items,
        }, .{});

        results.* = emitted[0..step.outputs.len];
        var k = step.outputs.len;
        for (step.states) |binding| {
            const instance = p.states[binding.instance];
            for (instance.components, 0..) |_, i| {
                states[binding.base + i] = emitted[k];
                k += 1;
            }
        }
    }

    return .{
        .value = switch (p.result) {
            .value => |v| produced[v.step][v.out],
            .public => |i| publics[i],
        },
        .states = states,
    };
}
