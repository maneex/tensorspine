//! `tspl` — the ZML generator's command line.
//!
//! ZM0 reads a derived document and reports what it contains, linking `@zml//zml`
//! so the build itself is the check: the module wiring, the Zig toolchain and the
//! ZML dependency are proven before any semantics are written.

const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const graph = @import("graph.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const log = std.log.scoped(.tspl);

const Args = struct {
    derived: []const u8,

    pub const help =
        \\ Use tspl --derived=<path>
        \\
        \\ Run a tensorspine/2.0 model from its derived document (D1–D6).
        \\
        \\ Options:
        \\   --derived=<path>    Path to a .derived.json, as `tensorspine --derive` emits it (required)
        \\
    ;
};

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

    const nodes = try g.nodes();
    const values = try g.values();
    const tensors = try g.tensors();
    const states = try g.states();

    log.info("{s}: {d} occurrences, {d} values, {d} parameter tensors, {d} states", .{
        g.model(),
        nodes.count(),
        values.items.len,
        tensors.items.len,
        states.items.len,
    });

    // Touching ZML proves the dependency is linked, not merely declared.
    const hidden = zml.Shape.init(.{ .s = 1, .d = 4096 }, .bf16);
    log.info("zml links: a {f} activation is {d} bytes", .{ hidden, hidden.byteSize() });
}
