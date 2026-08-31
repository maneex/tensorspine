//! The derived document as Zig data — and nothing else (Z01).
//!
//! A `Graph` is built from a `.derived.json`. The generator never reads the model
//! document, the catalog, or the language's tools: everything it needs must be in
//! D1–D6. Deriving here would be a second implementation of the language.
//!
//! Nothing in this file imports `zml`: the document is data, and a `zml.Tensor`
//! reaching it would silently become an MLIR block argument (Z04).

const std = @import("std");

pub const schema_id = "tensorspine-derived/2.0";

pub const Error = error{
    /// The file parsed, but it is not a derived document.
    NotADerivedDocument,
};

// --- D1: the expanded graph -------------------------------------------------

pub const Contract = struct {
    name: []const u8,
    version: []const u8,

    pub fn is(self: Contract, name: []const u8, version: []const u8) bool {
        return std.mem.eql(u8, self.name, name) and std.mem.eql(u8, self.version, version);
    }
};

/// One occurrence. `arguments` stays a `std.json.Value`: the contract owns their
/// grammar, and a primitive reads the ones it knows.
pub const Node = struct {
    contract: Contract,
    arguments: std.json.Value = .null,
};

pub const PortRef = struct {
    node: []const u8,
    port: []const u8,
};

pub const Edge = struct {
    from: PortRef,
    to: PortRef,
};

/// A public input: the stream it introduces or joins, and the ports it feeds.
pub const Input = struct {
    to: []const PortRef = &.{},
    kind: []const u8 = "",
    stream: ?[]const u8 = null,
};

/// A public output: the value it exposes, and whether it is fed back at decode.
pub const Output = struct {
    node: []const u8,
    port: []const u8,
    generative: bool = false,
};

pub const Interfaces = struct {
    inputs: std.json.ArrayHashMap(Input) = .{},
    outputs: std.json.ArrayHashMap(Output) = .{},
};

pub const D1 = struct {
    nodes: std.json.ArrayHashMap(Node) = .{},
    edges: []const Edge = &.{},
    interfaces: Interfaces = .{},
    topological_order: []const []const u8 = &.{},
};

// --- D2: values -------------------------------------------------------------

/// One axis of a shape. `extent` is per element for a value, absolute for a tensor.
pub const Axis = struct {
    axis: []const u8 = "",
    extent: i64 = 0,
};

pub const Domain = struct {
    kind: []const u8 = "",
    stream: []const u8 = "",
};

pub const Value = struct {
    value: []const u8,
    shape: []const Axis = &.{},
    dtype: []const u8 = "",
    role: []const u8 = "",
    /// Present when a public input delivers this value, naming that input.
    input: ?[]const u8 = null,
    domain: ?Domain = null,
};

pub const D2 = struct {
    values: []const Value = &.{},
};

// --- D3: parameter tensors --------------------------------------------------

/// Where a tensor lives in the checkpoint. The form is the key that is present;
/// the corpus uses `tensor` and `slice` today.
pub const Location = struct {
    tensor: ?[]const u8 = null,
    slice: ?Slice = null,

    pub const Slice = struct {
        tensor: []const u8,
        axis: []const u8 = "",
        dim: usize = 0,
        offset: i64 = 0,
        extent: i64 = 0,
    };
};

pub const Tensor = struct {
    identity: []const u8,
    /// `node.slot`, one per occurrence sharing this identity. A tied identity has
    /// several; it is loaded once, which is why `params` is one element per identity (Z03).
    members: []const []const u8 = &.{},
    slot: []const u8 = "",
    dtype: []const u8,
    shape: []const Axis = &.{},
    tied: bool = false,
    bytes: i64 = 0,
    location: ?Location = null,
};

pub const D3 = struct {
    tensors: []const Tensor = &.{},
};

// --- D4: states -------------------------------------------------------------

pub const PayloadComponent = struct {
    component: []const u8,
    dtype: []const u8,
    shape: []const Axis = &.{},
};

pub const State = struct {
    identity: []const u8,
    /// `node.state`, as D3's members are `node.slot`.
    members: []const []const u8 = &.{},
    state: []const u8 = "",
    law: []const u8,
    access: []const u8,
    sharing: []const u8 = "",
    span: ?i64 = null,
    payload: []const PayloadComponent = &.{},
};

pub const D4 = struct {
    states: []const State = &.{},
};

// --- the document -----------------------------------------------------------

pub const Document = struct {
    schema: []const u8 = "",
    model: []const u8 = "",
    d1: D1 = .{},
    d2: D2 = .{},
    d3: D3 = .{},
    d4: D4 = .{},
};

/// One derived document, parsed, owned, and indexed for the walks the generator makes.
pub const Graph = struct {
    parsed: std.json.Parsed(Document),
    arena: std.heap.ArenaAllocator,

    /// `node.slot` -> index into d3.tensors. The key is D3's own member string.
    tensor_of: std.StringHashMapUnmanaged(usize) = .empty,
    /// `node.state` -> index into d4.states.
    state_of: std.StringHashMapUnmanaged(usize) = .empty,
    /// value name -> index into d2.values.
    value_of: std.StringHashMapUnmanaged(usize) = .empty,
    /// `node.port` of an input port -> the value name feeding it.
    fed_by: std.StringHashMapUnmanaged([]const u8) = .empty,
    /// `node.port` of an input port -> the public input delivering it.
    from_input: std.StringHashMapUnmanaged([]const u8) = .empty,

    pub fn openFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !Graph {
        const source = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
        defer allocator.free(source);
        return parse(allocator, source);
    }

    pub fn parse(allocator: std.mem.Allocator, source: []const u8) !Graph {
        const parsed = try std.json.parseFromSlice(Document, allocator, source, .{
            .ignore_unknown_fields = true,
            .allocate = .alloc_always,
        });
        errdefer parsed.deinit();

        if (!std.mem.eql(u8, parsed.value.schema, schema_id)) return Error.NotADerivedDocument;

        var self: Graph = .{
            .parsed = parsed,
            .arena = .init(allocator),
        };
        errdefer self.arena.deinit();
        try self.index();
        return self;
    }

    pub fn deinit(self: *Graph) void {
        self.arena.deinit();
        self.parsed.deinit();
    }

    pub fn doc(self: *const Graph) *const Document {
        return &self.parsed.value;
    }

    fn index(self: *Graph) !void {
        const a = self.arena.allocator();
        const d = self.doc();

        for (d.d3.tensors, 0..) |t, i| {
            for (t.members) |m| try self.tensor_of.put(a, m, i);
        }
        for (d.d4.states, 0..) |s, i| {
            for (s.members) |m| try self.state_of.put(a, m, i);
        }
        for (d.d2.values, 0..) |v, i| {
            try self.value_of.put(a, v.value, i);
        }
        for (d.d1.edges) |e| {
            const to = try join(a, e.to.node, e.to.port);
            const from = try join(a, e.from.node, e.from.port);
            try self.fed_by.put(a, to, from);
        }
        for (d.d1.interfaces.inputs.map.keys(), d.d1.interfaces.inputs.map.values()) |name, entry| {
            for (entry.to) |t| {
                try self.from_input.put(a, try join(a, t.node, t.port), name);
            }
        }
    }

    fn join(a: std.mem.Allocator, left: []const u8, right: []const u8) ![]const u8 {
        return std.fmt.allocPrint(a, "{s}.{s}", .{ left, right });
    }

    // --- accessors ----------------------------------------------------------

    pub fn model(self: *const Graph) []const u8 {
        return self.doc().model;
    }

    pub fn nodes(self: *const Graph) *const std.json.ArrayHashMap(Node) {
        return &self.doc().d1.nodes;
    }

    pub fn node(self: *const Graph, id: []const u8) ?Node {
        return self.doc().d1.nodes.map.get(id);
    }

    /// The parameter tensor a `node.slot` member names.
    pub fn tensorOf(self: *const Graph, member: []const u8) ?Tensor {
        const i = self.tensor_of.get(member) orelse return null;
        return self.doc().d3.tensors[i];
    }

    /// The index into `params` of the tensor a member names — D3 order, one per identity.
    pub fn tensorIndexOf(self: *const Graph, member: []const u8) ?usize {
        return self.tensor_of.get(member);
    }

    pub fn stateOf(self: *const Graph, member: []const u8) ?State {
        const i = self.state_of.get(member) orelse return null;
        return self.doc().d4.states[i];
    }

    pub fn valueNamed(self: *const Graph, name: []const u8) ?Value {
        const i = self.value_of.get(name) orelse return null;
        return self.doc().d2.values[i];
    }

    /// The value feeding an input port, as `node.port`.
    pub fn fedBy(self: *const Graph, port: []const u8) ?[]const u8 {
        return self.fed_by.get(port);
    }

    /// The public input delivering an input port, when no edge does.
    pub fn fromInput(self: *const Graph, port: []const u8) ?[]const u8 {
        return self.from_input.get(port);
    }

    /// The generative output — the one fed back at decode — if the document has one.
    pub fn generative(self: *const Graph) ?Output {
        for (self.doc().d1.interfaces.outputs.map.values()) |o| {
            if (o.generative) return o;
        }
        return null;
    }
};
