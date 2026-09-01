//! This generator's capabilities manifest, written from its own code (Z13).
//!
//! `generators/CAPABILITIES.md` is the grammar and the rule: *every generator writes its
//! manifest from its code — the tables its kernels are written against, never a
//! hand-maintained list — and commits the result beside a test that regenerates it.*
//! So every field here is read off something that decides behaviour: the contract
//! entries are the primitives' own declared tables, the state laws are the ones
//! `state.zig` actually appends under, the locations are the forms `loader.zig`
//! assembles. Declaring more would be a lie the reader would believe.
//!
//! The reader is the language's, not ours: `tensorspine --capabilities MANIFEST MODEL…`.

const std = @import("std");

const zml = @import("zml");

const primitive = @import("primitive.zig");
const registry = @import("registry.zig");

/// The dtypes a primitive computes in. Both are exercised: f32 against the reference's
/// fixtures, bf16 to run a whole model.
const compute_dtypes = [_][]const u8{ "f32", "bf16" };

/// Storage dtypes the loader reads and a primitive converts from. The packed forms
/// (`fp4`, the f8s, the sub-byte integers) would need unpacking that nothing here does.
const parameter_dtypes = [_][]const u8{ "bf16", "f16", "f32" };

/// What `state.zig` implements, not what it names. All three laws are written now —
/// `append` at a cursor, `window` as a chronological slide, `fixed` written whole.
/// `Access` has four values and nothing consumes a `selected` state.
const state_laws = [_][]const u8{ "append", "window", "fixed" };
const access = [_][]const u8{"logical_position"};

/// What `loader.zig` assembles. D3 also carries `slice`, which it refuses by name.
const locations = [_][]const u8{"tensor"};

/// The indexing domains the emitter handles. One stream of token-indexed elements,
/// positioned by the invocation's own offset; no transform, and no fragmented input.
const domain_kinds = [_][]const u8{"token"};

/// `version` and `generated` are given rather than discovered: the manifest is committed
/// and a test regenerates it and diffs, so anything the code cannot derive twice — a
/// commit, a date — has to come in from outside or the diff fails on the calendar.
pub fn write(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    version: []const u8,
    generated: []const u8,
) !usize {
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var out: std.Io.Writer.Allocating = .init(a);
    var s: std.json.Stringify = .{ .writer = &out.writer, .options = .{ .whitespace = .indent_2 } };

    try s.beginObject();
    try s.objectField("schema");
    try s.write("tensorspine-capabilities/1");

    try s.objectField("generator");
    try s.beginObject();
    try s.objectField("name");
    try s.write("zml");
    try s.objectField("version");
    try s.write(version);
    try s.objectField("generator");
    try s.write("tspl --capabilities");
    try s.objectField("generated");
    try s.write(generated);
    try s.endObject();

    try s.objectField("compute_dtypes");
    try s.write(compute_dtypes);
    try s.objectField("parameter_dtypes");
    try s.write(parameter_dtypes);
    try s.objectField("state_laws");
    try s.write(state_laws);
    try s.objectField("access");
    try s.write(access);
    try s.objectField("sharing");
    try s.write([0][]const u8{});          // no cross-session sharing
    try s.objectField("partitions");
    try s.write([0][]const u8{});          // one machine, one device: nothing to communicate

    try s.objectField("domains");
    try s.beginObject();
    try s.objectField("kinds");
    try s.write(domain_kinds);
    try s.objectField("transforms");
    try s.write([0][]const u8{});
    try s.objectField("fragmented");
    try s.write(false);
    try s.endObject();

    try s.objectField("sessions_per_invocation");
    try s.write(1);
    try s.objectField("locations");
    try s.write(locations);

    // The contracts, from the primitives' own tables — parsed here only to prove they
    // are the grammar the reader expects, and re-emitted as they were written.
    try s.objectField("contracts");
    try s.beginObject();
    for (sorted(a)) |p| {
        const parsed = std.json.parseFromSlice(std.json.Value, a, p.capabilities, .{}) catch |err| {
            std.log.err("{s}@{s}: its capability table is not JSON: {s}", .{ p.name, p.version, @errorName(err) });
            return err;
        };
        defer parsed.deinit();

        const key = try std.fmt.allocPrint(a, "{s}@{s}", .{ p.name, p.version });
        try s.objectField(key);
        try s.write(parsed.value);
    }
    try s.endObject();
    try s.endObject();
    try out.writer.writeByte('\n');

    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buffer: [4096]u8 = undefined;
    var writer = file.writer(io, &buffer);
    try writer.interface.writeAll(out.written());
    try writer.interface.flush();

    return registry.all.len;
}

/// The primitives by contract key, so the manifest does not change when the registry's
/// import order does.
fn sorted(a: std.mem.Allocator) []const primitive.Primitive {
    const out = a.alloc(primitive.Primitive, registry.all.len) catch @panic("out of memory");
    @memcpy(out, &registry.all);
    std.mem.sort(primitive.Primitive, out, {}, struct {
        fn less(_: void, x: primitive.Primitive, y: primitive.Primitive) bool {
            return std.mem.order(u8, x.name, y.name) == .lt;
        }
    }.less);
    return out;
}
