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
const state_mod = @import("state.zig");
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

/// One state an occurrence holds: the contract's name for it, which layout holds it,
/// where that layout's components begin in the flat state array, and which portion of
/// them this occurrence owns.
///
/// `member` is a **physical parameter**, not a contract argument: D4 declares one
/// identity per layer, and packing a family into one buffer is the serving
/// application's layout choice (`state.zig`).
pub const StateBinding = struct {
    name: []const u8,
    instance: usize,
    base: usize,
    member: i64,
};

pub const Step = struct {
    node: []const u8,
    prim: *const primitive.Primitive,
    arguments: std.json.Value,
    inputs: []const PortBinding,
    params: []const SlotBinding,
    states: []const StateBinding,
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
    /// Which of the plan's groups this program is.
    group: usize = 0,

    pub fn of(p: *const Plan) Handle {
        return .{ .ptr = p };
    }

    pub fn ofGroup(p: *const Plan, group: usize) Handle {
        return .{ .ptr = p, .group = group };
    }

    pub fn plan(self: Handle) *const Plan {
        return @ptrCast(@alignCast(self.ptr));
    }
};

/// One value crossing a group boundary: produced in one program, consumed in a later
/// one, so it leaves the first as a result and enters the second as an argument.
pub const Boundary = struct {
    step: usize,
    out: usize,
    shape: zml.Shape,
};

/// A contiguous run of steps compiled as one program. Several programs run in sequence
/// over the same weights and states, and XLA frees each one's scratch before the next
/// begins — which is what bounds a run whose scratch would otherwise hold every layer's
/// weights at once.
pub const Group = struct {
    first: usize,
    last: usize,
    inputs: []const Boundary,
    outputs: []const Boundary,
    /// Indices into `Plan.params_used`: only what these steps read.
    params: []const usize,
};

pub const Plan = struct {
    arena: std.heap.ArenaAllocator,
    steps: []const Step,
    publics: []const []const u8,
    public_shapes: []const zml.Shape,
    /// The D3 identities this plan needs, in the order `Model.params` holds them.
    /// Evaluating one value must not read a whole checkpoint.
    params_used: []const usize,
    /// The state instances the plan touches, and their buffers flattened in the order
    /// the traced function takes and returns them.
    states: []const state_mod.Instance,
    state_shapes: []const zml.Shape,
    /// How many elements this invocation carries — deployment intent, fixed before the
    /// plan so every shape in it is concrete.
    elements: i64,
    result: Source,
    compute: zml.DataType,
    /// One group when the whole plan is one program.
    groups: []const Group = &.{},

    pub fn deinit(self: *Plan) void {
        self.arena.deinit();
    }

    /// Cut the steps into `count` contiguous programs of roughly equal length, and work
    /// out what crosses each boundary. Splitting is a **serving choice** — the graph and
    /// its numbers are the same either way — so it is a run-time argument, like the
    /// reference generator's `--max-ram`.
    pub fn split(self: *Plan, count: usize) !void {
        const a = self.arena.allocator();
        const n = @max(1, @min(count, self.steps.len));
        const groups = try a.alloc(Group, n);

        var first: usize = 0;
        for (groups, 0..) |*group, g| {
            const last = if (g + 1 == n) self.steps.len else (self.steps.len * (g + 1)) / n;

            var inputs: std.ArrayList(Boundary) = .empty;
            var outputs: std.ArrayList(Boundary) = .empty;
            var params: std.ArrayList(usize) = .empty;

            for (self.steps[first..last]) |step| {
                for (step.inputs) |in| {
                    const v = switch (in.source) {
                        .value => |v| v,
                        .public => continue,
                    };
                    if (v.step >= first) continue;            // produced inside this group
                    if (has(inputs.items, v.step, v.out)) continue;
                    try inputs.append(a, .{ .step = v.step, .out = v.out, .shape = self.steps[v.step].shapes[v.out] });
                }
                for (step.params) |slot| {
                    if (std.mem.indexOfScalar(usize, params.items, slot.param) == null) {
                        try params.append(a, slot.param);
                    }
                }
            }

            // What leaves: anything a later step reads, and the plan's own result.
            for (first..last) |si| {
                for (self.steps[si].outputs, 0..) |_, out| {
                    var needed = switch (self.result) {
                        .value => |v| v.step == si and v.out == out,
                        .public => false,
                    };
                    if (!needed) {
                        for (self.steps[last..]) |later| {
                            for (later.inputs) |in| switch (in.source) {
                                .value => |v| if (v.step == si and v.out == out) {
                                    needed = true;
                                },
                                .public => {},
                            };
                        }
                    }
                    if (needed) try outputs.append(a, .{ .step = si, .out = out, .shape = self.steps[si].shapes[out] });
                }
            }

            group.* = .{
                .first = first,
                .last = last,
                .inputs = try inputs.toOwnedSlice(a),
                .outputs = try outputs.toOwnedSlice(a),
                .params = try params.toOwnedSlice(a),
            };
            first = last;
        }
        self.groups = groups;
    }
};

fn has(items: []const Boundary, step: usize, out: usize) bool {
    for (items) |b| {
        if (b.step == step and b.out == out) return true;
    }
    return false;
}

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
    capacity: i64,
    compute: zml.DataType,
    /// Pack states whose law, access and payload agree into one buffer each. A serving
    /// layout choice: one buffer per identity is legal and simpler, one buffer per
    /// family is what a deep model needs and what a paged cache would want.
    packed_states: bool,
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
    var states: std.ArrayList(state_mod.Instance) = .empty;
    var state_shapes: std.ArrayList(zml.Shape) = .empty;
    var bases: std.ArrayList(usize) = .empty;

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

        // states: D4's members are `node.state`, as D3's are `node.slot`. States whose
        // law, access and payload agree share one allocation — the serving layout — and
        // each occurrence is told which portion is its own.
        var step_states: std.ArrayList(StateBinding) = .empty;
        for (d.d4.states) |st| {
            for (st.members) |m| {
                const owner, const name = splitLast(m);
                if (!std.mem.eql(u8, owner, node_id)) continue;

                var into: ?usize = null;
                if (packed_states) {
                    for (states.items, 0..) |*existing, i| {
                        if (existing.packableWith(st, compute)) {
                            into = i;
                            break;
                        }
                    }
                }
                if (into) |i| {
                    const member: i64 = @intCast(states.items[i].members);
                    try state_mod.addMember(a, &states.items[i], st.identity);
                    for (states.items[i].components, 0..) |c, k| state_shapes.items[bases.items[i] + k] = c.shape;
                    try step_states.append(a, .{ .name = name, .instance = i, .base = bases.items[i], .member = member });
                } else {
                    const instance = try state_mod.instanceOf(a, st, capacity, compute);
                    const base = state_shapes.items.len;
                    for (instance.components) |c| try state_shapes.append(a, c.shape);
                    try step_states.append(a, .{ .name = name, .instance = states.items.len, .base = base, .member = 0 });
                    try bases.append(a, base);
                    try states.append(a, instance);
                }
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
            .states = try step_states.toOwnedSlice(a),
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
        .states = try states.toOwnedSlice(a),
        .state_shapes = try state_shapes.toOwnedSlice(a),
        .elements = elements,
        .result = .{ .value = .{ .step = result.step, .out = result.out } },
        .compute = compute,
    };
}
