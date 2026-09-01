//! D4's state laws, implemented once (Z08).
//!
//! Every ZML model today hand-writes its own KV cache. D4 declares the law, the access
//! geometry and the payload of every state, so the three laws are three implementations
//! for the whole corpus — a primitive never allocates or shapes a state, it reads and
//! appends.
//!
//! In a traced graph a state is functional: buffers arrive as operands and leave as
//! results.
//!
//! **The model exposes a state; the serving application decides how to hold it.**
//!
//! The line between the two is the whole point, so it is worth stating exactly:
//!
//! | From the document (D4) | From the serving application |
//! |---|---|
//! | the law, the access geometry, the span | how many positions (capacity) |
//! | the payload: components, axes, extents, dtype | one buffer per identity, one per family, or pages |
//! | that there is one identity per layer | which portion of a shared buffer each occurrence consumes |
//!
//! **Geometry is inferred from the model and nowhere else** — `instanceOf` reads it out
//! of D4's `payload`, `law` and `span`, and `packableWith` refuses to put two states in
//! one buffer unless the document says their payloads agree. A serving choice may
//! rearrange states; it may never decide their shape.
//!
//! The layout, by contrast, is the serving application's, exactly like a block size or a
//! kernel selection — so the portion an occurrence consumes reaches its primitive
//! through the **opaque physical channel** (`generators/CAPABILITIES.md`), never through
//! the contract's arguments. A primitive asks this handle for its own view and never
//! learns how many neighbours share the buffer, nor in what order, nor whether the view
//! it gets is a slice of one allocation or gathered from pages. That is what lets the
//! same primitive serve a naive contiguous cache and a paged one.

const std = @import("std");

const zml = @import("zml");

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

/// One payload component's buffer. `[positions, *payload]` for a growing or bounded law
/// (§4.3: a payload is per position for append/window, the whole state for fixed), with
/// a leading `[members]` axis when the serving layout packs a family together.
pub const Component = struct {
    name: []const u8,
    shape: zml.Shape,
};

/// The buffers one *layout* needs: the D4 identities it packs, and their components.
/// One identity per member when nothing is packed.
pub const Instance = struct {
    /// The D4 identities this layout holds, in member order. A dump names its files by
    /// these, never by the layout: what the runtime chose to pack is its own business,
    /// and what leaves the generator is what the document declared.
    identities: []const []const u8,
    law: Law,
    access: Access,
    /// How many D4 identities share these buffers. 1 when each has its own.
    members: usize,
    components: []const Component,

    /// Whether two D4 states can share one buffer: same law, same access, same payload.
    /// Identity is not consulted — that they are one family is D4's business, and this
    /// is only asking whether one allocation can hold both.
    pub fn packableWith(self: Instance, s: graph.State, compute: zml.DataType) bool {
        if (self.law != (lawOf(s.law) catch return false)) return false;
        if (self.access != (accessOf(s.access) catch return false)) return false;
        if (self.components.len != s.payload.len) return false;
        for (self.components, s.payload) |c, p| {
            if (!std.mem.eql(u8, c.name, p.component)) return false;
            if (p.shape.len + 2 != c.shape.rank()) return false;
            for (p.shape, 0..) |axis, i| {
                if (c.shape.dim(@as(i64, @intCast(i)) + 2) != axis.extent) return false;
            }
            if (c.shape.dtype() != compute) return false;
        }
        return true;
    }
};

/// The buffers D4 asks for, under a capacity the deployment fixed and a layout the
/// serving application chose. Capacity is deployment intent, not a document fact: D4
/// gives the bytes per position, §7 says how many positions is the runtime's business.
pub fn instanceOf(
    allocator: std.mem.Allocator,
    s: graph.State,
    capacity: i64,
    compute: zml.DataType,
) !Instance {
    const law = try lawOf(s.law);
    const access = try accessOf(s.access);

    const positions: i64 = switch (law) {
        .append => capacity,
        .window => s.span orelse return Error.MissingSpan,
        .fixed => 1,
    };

    const components = try allocator.alloc(Component, s.payload.len);
    for (s.payload, components) |p, *c| {
        // [members, positions, *payload]; the members axis grows as identities join.
        var sh = zml.Shape.init(.{}, compute);
        sh = sh.appendDim(1, null);
        sh = sh.appendDim(positions, null);
        for (p.shape) |axis| sh = sh.appendDim(axis.extent, null);
        c.* = .{ .name = p.component, .shape = sh };
    }
    const identities = try allocator.alloc([]const u8, 1);
    identities[0] = s.identity;
    return .{ .identities = identities, .law = law, .access = access, .members = 1, .components = components };
}

/// One more identity joins this layout's buffers.
pub fn addMember(allocator: std.mem.Allocator, instance: *Instance, identity: []const u8) !void {
    const grown = try allocator.alloc(Component, instance.components.len);
    for (instance.components, grown) |c, *g| {
        g.* = .{ .name = c.name, .shape = c.shape.setDim(0, c.shape.dim(0) + 1) };
    }
    instance.components = grown;

    const identities = try allocator.alloc([]const u8, instance.identities.len + 1);
    @memcpy(identities[0..instance.identities.len], instance.identities);
    identities[instance.identities.len] = identity;
    instance.identities = identities;
    instance.members += 1;
}

/// What a primitive is handed for one state: its buffers, where this invocation's
/// elements sit in them, and — from the opaque channel — which portion it owns.
pub const Handle = struct {
    law: Law,
    access: Access,
    /// The whole buffers, members included. A primitive never indexes these directly.
    buffers: []const zml.Tensor,
    names: []const []const u8,
    /// The portion of the packed buffers this occurrence consumes. A **physical
    /// parameter**: which layer sits where is the serving layout's business, opaque to
    /// the language and passed beside the contract's arguments, never merged into them.
    member: i64,
    /// The logical position of this invocation's first element.
    start: zml.Tensor,
    /// How many elements this invocation carries.
    elements: i64,

    /// This occurrence's own view of a component — `[positions, *payload]`, with the
    /// packing gone.
    pub fn get(self: Handle, name: []const u8) ?zml.Tensor {
        for (self.names, self.buffers) |n, b| {
            // `.single` drops the axis: the primitive sees `[positions, *payload]` and
            // nothing of the layout that produced it.
            if (std.mem.eql(u8, n, name)) return b.slice(0, .single(self.member));
        }
        return null;
    }

    /// The buffers with this invocation's values written into this occurrence's
    /// portion. For `append`, elements land at `start`, contiguously — which is what
    /// the reference's cursor does.
    pub fn append(self: Handle, allocator: std.mem.Allocator, values: []const zml.Tensor) ![]zml.Tensor {
        std.debug.assert(values.len == self.buffers.len);
        if (self.law != .append) return Error.UnimplementedLaw;

        const out = try allocator.alloc(zml.Tensor, self.buffers.len);
        for (self.buffers, values, out) |buffer, value, *updated| {
            const offsets = try allocator.alloc(zml.Tensor, buffer.rank());
            defer allocator.free(offsets);
            offsets[0] = zml.Tensor.scalar(self.member, .i32);
            offsets[1] = self.start.convert(.i32);
            for (offsets[2..]) |*o| o.* = zml.Tensor.scalar(0, .i32);

            // The update carries the members axis as a single position of its own.
            const update = value.convert(buffer.dtype()).insertAxes(0, .{.portion});
            updated.* = buffer.dynamicUpdateSlice(offsets, update);
        }
        return out;
    }

    /// The same handle over buffers that have just been written — so a primitive reads
    /// its own portion of what it appended without knowing the layout.
    pub fn after(self: Handle, buffers: []const zml.Tensor) Handle {
        var next = self;
        next.buffers = buffers;
        return next;
    }

    /// How many positions of this occurrence's portion hold written values.
    pub fn length(self: Handle) zml.Tensor {
        return self.start.addConstant(self.elements);
    }
};
