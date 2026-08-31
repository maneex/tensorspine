//! What a primitive is, and everything it is given (Z05).
//!
//! One file per contract version, each a pure function from the occurrence's
//! arguments, inputs, parameters and states to its outputs. A primitive never
//! reaches into the graph, the loader or the session: that narrowness is what
//! makes a linked primitive and one arriving through `PRIMITIVE-ABI.md`
//! interchangeable.

const std = @import("std");

const zml = @import("zml");

/// A value bound to the port, slot, state component or stream that names it.
pub const Binding = struct {
    name: []const u8,
    tensor: zml.Tensor,
};

/// A small ordered set of bindings — never more than a handful, so a scan is the
/// lookup. Missing names are a programming error in the caller, not a run-time
/// condition: D1 fed every port, and D3 and D4 named every slot and state.
pub const Bindings = struct {
    items: []const Binding = &.{},

    pub fn get(self: Bindings, name: []const u8) ?zml.Tensor {
        for (self.items) |b| {
            if (std.mem.eql(u8, b.name, name)) return b.tensor;
        }
        return null;
    }

    pub fn must(self: Bindings, name: []const u8) zml.Tensor {
        return self.get(name) orelse std.debug.panic("no binding named '{s}'", .{name});
    }

    pub fn has(self: Bindings, name: []const u8) bool {
        return self.get(name) != null;
    }
};

/// What holds for the whole invocation, not for one occurrence.
pub const Ctx = struct {
    /// Freed by the emitter after the graph is built; a primitive may allocate its
    /// results here and nothing longer-lived.
    allocator: std.mem.Allocator,
    compute: zml.DataType = .f32,
    /// The positions of the stream indexing this occurrence's elements, when it has one.
    positions: ?zml.Tensor = null,
};

pub const Error = error{
    /// The contract requires an argument the occurrence does not carry, or carries
    /// with a type the contract does not allow.
    MissingArgument,
};

/// One occurrence, as a primitive sees it.
pub const Call = struct {
    occurrence: []const u8 = "",
    /// D1's arguments, resolved, defaults applied. The contract owns their grammar.
    arguments: std.json.Value = .null,
    inputs: Bindings = .{},
    params: Bindings = .{},
    states: Bindings = .{},
    /// The opaque physical parameters addressed to this occurrence — `backend` among
    /// them. Neither typed nor validated: a primitive reads what it knows and ignores
    /// the rest (generators/CAPABILITIES.md).
    physical: ?std.json.Value = null,

    pub fn arg(self: Call, name: []const u8) ?std.json.Value {
        return switch (self.arguments) {
            .object => |o| o.get(name),
            else => null,
        };
    }

    pub fn argInt(self: Call, name: []const u8) ?i64 {
        return switch (self.arg(name) orelse return null) {
            .integer => |i| i,
            .float => |f| @intFromFloat(f),
            else => null,
        };
    }

    pub fn argFloat(self: Call, name: []const u8) ?f64 {
        return switch (self.arg(name) orelse return null) {
            .float => |f| f,
            .integer => |i| @floatFromInt(i),
            else => null,
        };
    }

    /// An absent boolean is false: D1 applies the contract's defaults, so an argument
    /// still absent here is one the contract does not carry for this occurrence.
    pub fn argBool(self: Call, name: []const u8) bool {
        return switch (self.arg(name) orelse return false) {
            .bool => |b| b,
            else => false,
        };
    }

    pub fn argStr(self: Call, name: []const u8) ?[]const u8 {
        return switch (self.arg(name) orelse return null) {
            .string => |s| s,
            else => null,
        };
    }

    pub fn argObj(self: Call, name: []const u8) ?std.json.ObjectMap {
        return switch (self.arg(name) orelse return null) {
            .object => |o| o,
            else => null,
        };
    }

    pub fn requireFloat(self: Call, name: []const u8) !f64 {
        return self.argFloat(name) orelse Error.MissingArgument;
    }

    pub fn requireInt(self: Call, name: []const u8) !i64 {
        return self.argInt(name) orelse Error.MissingArgument;
    }
};

pub const Run = *const fn (ctx: *Ctx, call: Call) anyerror![]const Binding;

pub const Primitive = struct {
    name: []const u8,
    version: []const u8,
    run: Run,
    /// This primitive's entry in the manifest's rule grammar
    /// (`generators/CAPABILITIES.md`), declared by the code that implements it and
    /// assembled by `tspl --capabilities`. Never a hand-maintained list elsewhere.
    capabilities: []const u8,

    pub fn is(self: Primitive, name: []const u8, version: []const u8) bool {
        return std.mem.eql(u8, self.name, name) and std.mem.eql(u8, self.version, version);
    }
};

/// The common case: a primitive producing one value.
pub fn one(ctx: *Ctx, name: []const u8, tensor: zml.Tensor) ![]const Binding {
    const out = try ctx.allocator.alloc(Binding, 1);
    out[0] = .{ .name = name, .tensor = tensor };
    return out;
}
