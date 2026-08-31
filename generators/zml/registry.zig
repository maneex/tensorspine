//! {name, version} -> primitive (Z05).
//!
//! Zig has no runtime import, so the table is an explicit list: a new primitive is
//! its own file plus one line here. A lookup, never a switch woven through the
//! emitter — so a second, dynamic tier can be chained behind this one without
//! touching a single call site.

const std = @import("std");

const primitive = @import("primitive.zig");

pub const all = [_]primitive.Primitive{
    @import("primitives/embed.zig").primitive,
    @import("primitives/norm_rms.zig").primitive,
    @import("primitives/residual_add.zig").primitive,
};

pub fn find(name: []const u8, version: []const u8) ?*const primitive.Primitive {
    for (&all) |*p| {
        if (p.is(name, version)) return p;
    }
    return null;
}
