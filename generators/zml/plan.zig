//! What to evaluate, in what order, with what bound to what.
//!
//! Everything here is computed at Zig run time, before tracing: the traced function
//! walks a plan whose sources and shapes are already resolved, so nothing in the
//! emitter has to search. The plan holds no `zml.Tensor` (Z04) — only `Shape`s,
//! indices and names — so `meta.visit` prunes it and it never becomes a block
//! argument.

const std = @import("std");

const zml = @import("zml");

const dtypes = @import("dtypes.zig");
const graph = @import("graph.zig");
const primitive = @import("primitive.zig");
const registry = @import("registry.zig");

const log = std.log.scoped(.tspl);

pub const Error = error{
    UnknownValue,
    NoPrimitive,
    UnfedPort,
};

/// Where one of a step's inputs comes from.
pub const Source = union(enum) {
    /// Produced by an earlier step: its index, and which of its outputs.
    value: struct { step: usize, out: usize },
    /// Delivered by a public input: its index among `publics`.
    public: usize,
};

pub const PortBinding = struct {
    port: []const u8,
    source: Source,
};

pub const SlotBinding = struct {
    slot: []const u8,
    /// Index into `Model.params`, which holds only the identities this plan uses —
    /// `Plan.params_used` maps it back to D3's order.
    param: usize,
};

pub const Step = struct {
    node: []const u8,
    prim: *const primitive.Primitive,
    arguments: std.json.Value,
    inputs: []const PortBinding,
    params: []const SlotBinding,
    /// The ports this step produces, and the shape each is declared to have. The host
    /// fixes the signature; the primitive fills it (PRIMITIVE-ABI.md, the same rule).
    outputs: []const []const u8,
    shapes: []const zml.Shape,
    /// The name of the `stablehlo.composite` this occurrence becomes.
    composite: [:0]const u8,
};

/// How the plan reaches the traced function.
///
/// Z04 said the plan must hold no `zml.Tensor`; it must also not be *walked*. Passed
/// as a typed pointer it is, and `meta.Contains` recurses through `std.json.Value` —
/// a recursive union of maps and arrays — until the comptime branch quota is
/// exhausted and compilation fails. `Contains` returns false for `anyopaque` at once,
/// so an opaque handle is both the fix and the statement of intent: the plan is data
/// the primitive side reads, and ZML never looks inside it.
pub const Handle = struct {
    ptr: *const anyopaque,

    pub fn of(p: *const Plan) Handle {
        return .{ .ptr = p };
    }

    pub fn plan(self: Handle) *const Plan {
        return @ptrCast(@alignCast(self.ptr));
    }
};

pub const Plan = struct {
    arena: std.heap.ArenaAllocator,
    steps: []const Step,
    publics: []const []const u8,
    public_shapes: []const zml.Shape,
    /// The D3 identities this plan needs, in the order `Model.params` holds them.
    /// Evaluating one value must not read a whole checkpoint.
    params_used: []const usize,
    result: Source,
    compute: zml.DataType,

    pub fn deinit(self: *Plan) void {
        self.arena.deinit();
    }
};

fn splitLast(name: []const u8) struct { []const u8, []const u8 } {
    const i = std.mem.lastIndexOfScalar(u8, name, '.') orelse return .{ name, "" };
    return .{ name[0..i], name[i + 1 ..] };
}

/// The shape of a value for this invocation: D2's per-element extents behind the
/// element count. D2 sizes payloads per element; how many elements an invocation
/// carries is deployment intent, so it arrives here as `elements`.
fn shapeOf(v: graph.Value, elements: i64, dt: zml.DataType) zml.Shape {
    var sh = zml.Shape.init(.{}, dt);
    sh = sh.appendDim(elements, null);
    for (v.shape) |axis| sh = sh.appendDim(axis.extent, null);
    return sh;
}

/// A plan evaluating exactly what `target` (a `node.port` value) needs — its ancestor
/// closure, in the document's topological order.
pub fn until(
    allocator: std.mem.Allocator,
    g: *const graph.Graph,
    target: []const u8,
    elements: i64,
    compute: zml.DataType,
) !Plan {
    var arena: std.heap.ArenaAllocator = .init(allocator);
    errdefer arena.deinit();
    const a = arena.allocator();
    const d = g.doc();

    if (g.valueNamed(target) == null) {
        log.err("no value named '{s}' in D2", .{target});
        return Error.UnknownValue;
    }
    const target_node, const target_port = splitLast(target);

    // --- the ancestor closure, back along the edges ---
    var needed: std.StringHashMapUnmanaged(void) = .empty;
    var stack: std.ArrayList([]const u8) = .empty;
    defer stack.deinit(a);
    try needed.put(a, target_node, {});
    try stack.append(a, target_node);
    while (stack.pop()) |node| {
        for (d.d1.edges) |e| {
            if (!std.mem.eql(u8, e.to.node, node)) continue;
            if (needed.contains(e.from.node)) continue;
            try needed.put(a, e.from.node, {});
            try stack.append(a, e.from.node);
        }
    }

    // --- the steps, in the document's order ---
    var steps: std.ArrayList(Step) = .empty;
    var publics: std.ArrayList([]const u8) = .empty;
    var public_shapes: std.ArrayList(zml.Shape) = .empty;
    // value name -> (step, out)
    var produced: std.StringHashMapUnmanaged(struct { step: usize, out: usize }) = .empty;
    // D3 index -> index into params_used
    var used: std.AutoHashMapUnmanaged(usize, usize) = .empty;
    var params_used: std.ArrayList(usize) = .empty;

    for (d.d1.topological_order) |node_id| {
        if (!needed.contains(node_id)) continue;
        const node = g.node(node_id) orelse continue;

        const prim = registry.find(node.contract.name, node.contract.version) orelse {
            log.err("{s}: no primitive for {s}@{s}", .{ node_id, node.contract.name, node.contract.version });
            return Error.NoPrimitive;
        };

        // inputs: every edge landing on this node, then every public input feeding it
        var inputs: std.ArrayList(PortBinding) = .empty;
        for (d.d1.edges) |e| {
            if (!std.mem.eql(u8, e.to.node, node_id)) continue;
            const from = try std.fmt.allocPrint(a, "{s}.{s}", .{ e.from.node, e.from.port });
            const p = produced.get(from) orelse {
                log.err("{s}.{s} is fed by {s}, which no earlier step produced", .{ node_id, e.to.port, from });
                return Error.UnfedPort;
            };
            try inputs.append(a, .{ .port = e.to.port, .source = .{ .value = .{ .step = p.step, .out = p.out } } });
        }
        for (d.d1.interfaces.inputs.map.keys(), d.d1.interfaces.inputs.map.values()) |name, entry| {
            for (entry.to) |t| {
                if (!std.mem.eql(u8, t.node, node_id)) continue;
                var index: ?usize = null;
                for (publics.items, 0..) |p, i| {
                    if (std.mem.eql(u8, p, name)) index = i;
                }
                if (index == null) {
                    const v = g.valueNamed(name) orelse return Error.UnknownValue;
                    index = publics.items.len;
                    try publics.append(a, name);
                    try public_shapes.append(a, shapeOf(v, elements, try dtypes.of(v.dtype)));
                }
                try inputs.append(a, .{ .port = t.port, .source = .{ .public = index.? } });
            }
        }

        // parameters: D3's members are already `node.slot`. A tied identity used by two
        // steps is added once — which is also what keeps `emitMlir` from seeing the same
        // tensor id twice among the arguments (Z03).
        var params: std.ArrayList(SlotBinding) = .empty;
        for (d.d3.tensors, 0..) |t, ti| {
            for (t.members) |m| {
                const owner, const slot = splitLast(m);
                if (!std.mem.eql(u8, owner, node_id)) continue;
                const gop = try used.getOrPut(a, ti);
                if (!gop.found_existing) {
                    gop.value_ptr.* = params_used.items.len;
                    try params_used.append(a, ti);
                }
                try params.append(a, .{ .slot = slot, .param = gop.value_ptr.* });
            }
        }

        // outputs: the D2 values this node produces, in D2's order
        var outputs: std.ArrayList([]const u8) = .empty;
        var shapes: std.ArrayList(zml.Shape) = .empty;
        for (d.d2.values) |v| {
            if (v.input != null) continue;
            const owner, const port = splitLast(v.value);
            if (!std.mem.eql(u8, owner, node_id)) continue;
            try produced.put(a, v.value, .{ .step = steps.items.len, .out = outputs.items.len });
            try outputs.append(a, port);
            try shapes.append(a, shapeOf(v, elements, compute));
        }

        try steps.append(a, .{
            .node = node_id,
            .prim = prim,
            .arguments = node.arguments,
            .inputs = try inputs.toOwnedSlice(a),
            .params = try params.toOwnedSlice(a),
            .outputs = try outputs.toOwnedSlice(a),
            .shapes = try shapes.toOwnedSlice(a),
            .composite = try std.fmt.allocPrintSentinel(
                a,
                "tensorspine.{s}",
                .{node.contract.name},
                0,
            ),
        });
    }

    const result = produced.get(target) orelse {
        log.err("'{s}' is not produced by the closure", .{target});
        return Error.UnknownValue;
    };
    _ = target_port;

    return .{
        .arena = arena,
        .steps = try steps.toOwnedSlice(a),
        .publics = try publics.toOwnedSlice(a),
        .public_shapes = try public_shapes.toOwnedSlice(a),
        .params_used = try params_used.toOwnedSlice(a),
        .result = .{ .value = .{ .step = result.step, .out = result.out } },
        .compute = compute,
    };
}
