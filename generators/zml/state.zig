//! D4's state laws, implemented once (Z08).
//!
//! Every ZML model today hand-writes its own KV cache. D4 declares the law, the
//! access geometry and the payload of every state, so the three laws are three
//! implementations for the whole corpus — a primitive never allocates or shapes a
//! state, it reads and appends.
//!
//! In a traced graph a state is functional: buffers arrive as operands and leave as
//! results, donated by the caller. `length` is not stored — it is the first position
//! of the invocation plus the number of elements it carries, which the emitter knows.

const std = @import("std");

const zml = @import("zml");

const dtypes = @import("dtypes.zig");
const graph = @import("graph.zig");

pub const Error = error{
    /// A law D4 declares and this generator does not implement. A finding, not a
    /// workaround: the enum is closed, so a fourth law is a language matter.
    UnimplementedLaw,
    UnimplementedAccess,
    /// `window` needs the span D4 gives it.
    MissingSpan,
};

pub const Law = enum { append, window, fixed };
pub const Access = enum { logical_position, ring, aggregate, selected };

pub fn lawOf(name: []const u8) Error!Law {
    return std.meta.stringToEnum(Law, name) orelse Error.UnimplementedLaw;
}

pub fn accessOf(name: []const u8) Error!Access {
    return std.meta.stringToEnum(Access, name) orelse Error.UnimplementedAccess;
}

/// One payload component's buffer: `[positions, *payload]` for a growing or bounded
/// law, the payload alone for `fixed` (§4.3: a payload is per position for
/// append/window, the whole state for fixed).
pub const Component = struct {
    name: []const u8,
    shape: zml.Shape,
};

/// One state identity, as the generator holds it for one (session, branch).
pub const Instance = struct {
    identity: []const u8,
    law: Law,
    access: Access,
    components: []const Component,

    /// The buffers this instance needs, in component order.
    pub fn shapes(self: Instance, allocator: std.mem.Allocator) ![]zml.Shape {
        const out = try allocator.alloc(zml.Shape, self.components.len);
        for (self.components, out) |c, *s| s.* = c.shape;
        return out;
    }
};

/// The buffer shapes D4 asks for, under a capacity the deployment fixed. Capacity is
/// deployment intent, not a document fact: D4 gives the bytes per position, §7 says
/// how many positions is the runtime's business.
pub fn instanceOf(
    allocator: std.mem.Allocator,
    s: graph.State,
    capacity: i64,
    compute: zml.DataType,
) !Instance {
    const law = try lawOf(s.law);
    const access = try accessOf(s.access);

    const positions: ?i64 = switch (law) {
        .append => capacity,
        .window => s.span orelse return Error.MissingSpan,
        .fixed => null,
    };

    const components = try allocator.alloc(Component, s.payload.len);
    for (s.payload, components) |p, *c| {
        var sh = zml.Shape.init(.{}, compute);
        if (positions) |n| sh = sh.appendDim(n, null);
        for (p.shape) |axis| sh = sh.appendDim(axis.extent, null);
        c.* = .{ .name = p.component, .shape = sh };
    }
    return .{ .identity = s.identity, .law = law, .access = access, .components = components };
}

/// What a primitive is handed for one state: its buffers, and where this invocation's
/// elements sit in them.
pub const Handle = struct {
    law: Law,
    access: Access,
    /// One per payload component, in D4's order.
    buffers: []const zml.Tensor,
    names: []const []const u8,
    /// The logical position of this invocation's first element.
    start: zml.Tensor,
    /// How many elements this invocation carries.
    elements: i64,

    pub fn get(self: Handle, name: []const u8) ?zml.Tensor {
        for (self.names, self.buffers) |n, b| {
            if (std.mem.eql(u8, n, name)) return b;
        }
        return null;
    }

    /// The buffers with this invocation's values written in, and the number of
    /// positions readable afterwards. For `append`, elements land at `start`,
    /// contiguously — which is what the reference's cursor does.
    pub fn append(self: Handle, allocator: std.mem.Allocator, values: []const zml.Tensor) ![]zml.Tensor {
        std.debug.assert(values.len == self.buffers.len);
        const out = try allocator.alloc(zml.Tensor, self.buffers.len);
        for (self.buffers, values, out) |buffer, value, *updated| {
            updated.* = switch (self.law) {
                .append => buffer.dynamicUpdateSlice1d(value.convert(buffer.dtype()), 0, self.start),
                .window, .fixed => return Error.UnimplementedLaw,
            };
        }
        return out;
    }

    /// How many positions of the buffer hold written values, as a scalar tensor.
    pub fn length(self: Handle) zml.Tensor {
        return self.start.addConstant(self.elements);
    }
};
