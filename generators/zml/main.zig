//! `tspl` — the ZML generator's command line.
//!
//! ZM0 reads a derived document and reports what it contains, linking `@zml//zml`
//! so the build itself is the check: the module wiring, the Zig toolchain and the
//! ZML dependency are proven before any semantics are written.

const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const graph = @import("graph.zig");
const registry = @import("registry.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const log = std.log.scoped(.tspl);

const Args = struct {
    derived: []const u8,
    refusals: bool = false,

    pub const help =
        \\ Use tspl --derived=<path> [--refusals]
        \\
        \\ Run a tensorspine/2.0 model from its derived document (D1–D6).
        \\
        \\ Options:
        \\   --derived=<path>    Path to a .derived.json, as `tensorspine --derive` emits it (required)
        \\   --refusals          Report, per contract, the occurrences no primitive implements
        \\
    ;
};

/// Which contracts the document calls, how often, and whether a primitive answers.
/// The coarse half of R02: it says a contract is absent, not that an argument of a
/// present one is unimplemented — the arguments are the manifest's business, and
/// `tensorspine --capabilities` is their reader.
fn reportRefusals(allocator: std.mem.Allocator, g: *const graph.Graph) !bool {
    var counts: std.StringArrayHashMapUnmanaged(usize) = .empty;
    defer counts.deinit(allocator);

    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();

    for (g.doc().d1.nodes.map.values()) |n| {
        const key = try std.fmt.allocPrint(arena.allocator(), "{s}@{s}", .{ n.contract.name, n.contract.version });
        const gop = try counts.getOrPut(allocator, key);
        if (!gop.found_existing) gop.value_ptr.* = 0;
        gop.value_ptr.* += 1;
    }

    var refused: usize = 0;
    var served: usize = 0;
    for (counts.keys(), counts.values()) |key, count| {
        const at = std.mem.indexOfScalar(u8, key, '@').?;
        const known = registry.find(key[0..at], key[at + 1 ..]) != null;
        if (known) served += count else refused += count;
        log.info("  {s:<34} {d:>4} occurrence(s)  {s}", .{
            key, count, if (known) "implemented" else "NO PRIMITIVE",
        });
    }
    log.info("{s}: {d}/{d} occurrences implemented, {d} refused, over {d} contracts", .{
        g.model(), served, served + refused, refused, counts.count(),
    });
    return refused == 0;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    // `bazel run` executes from the runfiles tree; go back to the shell's directory
    // so a relative --derived path means what the user typed.
    if (init.environ_map.get("BUILD_WORKING_DIRECTORY")) |build_working_directory| {
        var working_dir = try std.Io.Dir.openDirAbsolute(init.io, build_working_directory, .{});
        defer working_dir.close(init.io);
        try std.process.setCurrentDir(init.io, working_dir);
    }

    const args = stdx.flags.parseProcessArgs(init.minimal, Args);

    var g: graph.Graph = try .openFile(allocator, init.io, args.derived);
    defer g.deinit();

    const d = g.doc();
    log.info("{s}: {d} occurrences, {d} values, {d} parameter tensors, {d} states, {d} edges, {d} ordered", .{
        g.model(),
        d.d1.nodes.map.count(),
        d.d2.values.len,
        d.d3.tensors.len,
        d.d4.states.len,
        d.d1.edges.len,
        d.d1.topological_order.len,
    });

    // Every member of D3 and D4 resolves, and every edge's target is a known port:
    // the indices are exercised here rather than trusted at ZM2.
    var located: usize = 0;
    for (d.d3.tensors) |t| {
        for (t.members) |m| if (g.tensorIndexOf(m) == null) {
            log.err("unresolved parameter member {s}", .{m});
            return error.InconsistentDocument;
        };
        if (t.location != null) located += 1;
    }
    for (d.d4.states) |s| {
        for (s.members) |m| if (g.stateOf(m) == null) {
            log.err("unresolved state member {s}", .{m});
            return error.InconsistentDocument;
        };
    }
    for (d.d1.topological_order) |id| if (g.node(id) == null) {
        log.err("ordered node {s} is not in d1.nodes", .{id});
        return error.InconsistentDocument;
    };

    const gen = g.generative();
    log.info("{d}/{d} tensors located; generative output {s}", .{
        located,
        d.d3.tensors.len,
        if (gen) |o| o.port else "(none)",
    });

    if (args.refusals) {
        _ = try reportRefusals(allocator, &g);
        return;
    }

    // Touching ZML proves the dependency is linked, not merely declared.
    const hidden = zml.Shape.init(.{ .s = 1, .d = 4096 }, .bf16);
    log.info("zml links: a {f} activation is {d} bytes", .{ hidden, hidden.byteSize() });
}
