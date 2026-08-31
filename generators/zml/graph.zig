//! The derived document as Zig data — and nothing else (Z01).
//!
//! A `Graph` is built from a `.derived.json`. The generator never reads the model
//! document, the catalog, or the language's tools: everything it needs must be in
//! D1–D6. Deriving here would be a second implementation of the language.

const std = @import("std");

pub const schema_id = "tensorspine-derived/2.0";

pub const Error = error{
    /// The file parsed, but it is not a derived document.
    NotADerivedDocument,
    /// A product the caller asked for is absent or not of the shape D1–D6 declares.
    MissingProduct,
};

fn asObject(v: std.json.Value) Error!std.json.ObjectMap {
    return switch (v) {
        .object => |o| o,
        else => Error.MissingProduct,
    };
}

fn asArray(v: std.json.Value) Error!std.json.Array {
    return switch (v) {
        .array => |a| a,
        else => Error.MissingProduct,
    };
}

fn asString(v: std.json.Value) ?[]const u8 {
    return switch (v) {
        .string => |s| s,
        else => null,
    };
}

/// One derived document, parsed and owned.
pub const Graph = struct {
    parsed: std.json.Parsed(std.json.Value),

    pub fn openFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !Graph {
        const source = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
        defer allocator.free(source);
        return parse(allocator, source);
    }

    pub fn parse(allocator: std.mem.Allocator, source: []const u8) !Graph {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, source, .{});
        errdefer parsed.deinit();

        const root = asObject(parsed.value) catch return Error.NotADerivedDocument;
        const schema = root.get("schema") orelse return Error.NotADerivedDocument;
        const name = asString(schema) orelse return Error.NotADerivedDocument;
        if (!std.mem.eql(u8, name, schema_id)) return Error.NotADerivedDocument;

        return .{ .parsed = parsed };
    }

    pub fn deinit(self: *Graph) void {
        self.parsed.deinit();
    }

    /// The model identifier the document names. Provenance stops here: a derived
    /// document carries this name and a catalog *path*, no version and no hash, so
    /// it cannot be checked against what produced it (plan §7, finding 1).
    pub fn model(self: Graph) []const u8 {
        const root = asObject(self.parsed.value) catch return "";
        const v = root.get("model") orelse return "";
        return asString(v) orelse "";
    }

    /// One of the products, by key: "d1" … "d6".
    fn product(self: Graph, key: []const u8) Error!std.json.ObjectMap {
        const root = try asObject(self.parsed.value);
        return asObject(root.get(key) orelse return Error.MissingProduct);
    }

    fn productArray(self: Graph, key: []const u8, member: []const u8) Error!std.json.Array {
        const p = try self.product(key);
        return asArray(p.get(member) orelse return Error.MissingProduct);
    }

    /// D1's occurrences, keyed by node identifier.
    pub fn nodes(self: Graph) Error!std.json.ObjectMap {
        const d1 = try self.product("d1");
        return asObject(d1.get("nodes") orelse return Error.MissingProduct);
    }

    /// D2's values: every value produced, and those the interfaces deliver and expose.
    pub fn values(self: Graph) Error!std.json.Array {
        return self.productArray("d2", "values");
    }

    /// D3's parameter tensors, one entry per identity — the order `params` follows (Z03).
    pub fn tensors(self: Graph) Error!std.json.Array {
        return self.productArray("d3", "tensors");
    }

    /// D4's states, one entry per identity.
    pub fn states(self: Graph) Error!std.json.Array {
        return self.productArray("d4", "states");
    }
};
