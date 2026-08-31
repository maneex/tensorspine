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

/// One operand of a composite: the name it is bound to inside the primitive, and
/// whether that name is a port or a parameter slot.
const Operand = struct {
    name: []const u8,
    is_param: bool,
};

const Decomposition = struct {
    step: *const plan_mod.Step,
    ctx: *primitive.Ctx,
    operands: []const Operand,
};

/// The body of one composite: rebuild the bindings from the block arguments, run the
/// primitive, and return its outputs in the order the host declared them.
fn decompose(args: []zml.Tensor, c: Decomposition) []zml.Tensor {
    const a = c.ctx.allocator;

    var inputs: std.ArrayList(primitive.Binding) = .empty;
    var params: std.ArrayList(primitive.Binding) = .empty;
    for (args, c.operands) |tensor, operand| {
        const binding: primitive.Binding = .{ .name = operand.name, .tensor = tensor };
        (if (operand.is_param) &params else &inputs).append(a, binding) catch @panic("out of memory");
    }

    const produced = c.step.prim.run(c.ctx, .{
        .occurrence = c.step.node,
        .arguments = c.step.arguments,
        .inputs = .{ .items = inputs.items },
        .params = .{ .items = params.items },
    }) catch |err| std.debug.panic("{s}: {s}@{s} failed: {s}", .{
        c.step.node, c.step.prim.name, c.step.prim.version, @errorName(err),
    });

    const out = a.alloc(zml.Tensor, c.step.outputs.len) catch @panic("out of memory");
    for (c.step.outputs, out) |port, *slot| {
        slot.* = blk: {
            for (produced) |b| {
                if (std.mem.eql(u8, b.name, port)) break :blk b.tensor;
            }
            std.debug.panic("{s}: {s} produced no '{s}'", .{ c.step.node, c.step.prim.name, port });
        };
    }
    return out;
}

/// Walk the plan. Traced by `zml.module.compile`, so the slices' run-time lengths fix
/// the arity of the MLIR function — which is what lets one Zig type serve every
/// document.
pub fn forward(handle: plan_mod.Handle, params: []const zml.Tensor, publics: []const zml.Tensor) zml.Tensor {
    const p = handle.plan();
    const cc = zml.module.CompilationContext.current();

    // Everything the walk allocates dies with the walk: the emitted MLIR holds the
    // values, not these bindings. An arena is both the freeing strategy and the
    // statement that a primitive may allocate nothing longer-lived.
    var arena: std.heap.ArenaAllocator = .init(cc.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var ctx: primitive.Ctx = .{ .allocator = a, .compute = p.compute };

    const produced = a.alloc([]zml.Tensor, p.steps.len) catch @panic("out of memory");
    for (p.steps, produced) |*step, *results| {
        const n = step.inputs.len + step.params.len;
        const operands = a.alloc(Operand, n) catch @panic("out of memory");
        const tensors = a.alloc(zml.Tensor, n) catch @panic("out of memory");

        var k: usize = 0;
        for (step.inputs) |in| {
            operands[k] = .{ .name = in.port, .is_param = false };
            tensors[k] = switch (in.source) {
                .value => |v| produced[v.step][v.out],
                .public => |i| publics[i],
            };
            k += 1;
        }
        for (step.params) |slot| {
            operands[k] = .{ .name = slot.slot, .is_param = true };
            tensors[k] = params[slot.param];
            k += 1;
        }

        results.* = zml.ops.composite(step.composite, tensors, step.shapes, decompose, Decomposition{
            .step = step,
            .ctx = &ctx,
            .operands = operands,
        }, .{});
    }

    return switch (p.result) {
        .value => |v| produced[v.step][v.out],
        .public => |i| publics[i],
    };
}
