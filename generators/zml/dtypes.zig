//! The catalog's dtype names as ZML's.
//!
//! The two vocabularies agree everywhere but one name, so this is a lookup with a
//! single exception rather than a table — and an unknown name is refused, never
//! guessed.

const std = @import("std");

const zml = @import("zml");

pub const Error = error{UnknownDtype};

pub fn of(name: []const u8) Error!zml.DataType {
    if (std.mem.eql(u8, name, "fp4")) return .f4e2m1;
    return std.meta.stringToEnum(zml.DataType, name) orelse Error.UnknownDtype;
}
