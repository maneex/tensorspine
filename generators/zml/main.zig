//! `tspl` — the ZML generator's command line.

const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const dtypes = @import("dtypes.zig");
const emit = @import("emit.zig");
const graph = @import("graph.zig");
const loader = @import("loader.zig");
const plan = @import("plan.zig");
const registry = @import("registry.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const log = std.log.scoped(.tspl);

const Args = struct {
    derived: []const u8,
    refusals: bool = false,
    checkpoint: ?[]const u8 = null,
    until: ?[]const u8 = null,
    ids: ?[]const u8 = null,
    out: ?[]const u8 = null,
    @"dump-mlir": ?[]const u8 = null,

    pub const help =
        \\ Use tspl --derived=<path> [--refusals] [--checkpoint=<dir> --until=<value>]
        \\
        \\ Run a tensorspine/2.0 model from its derived document (D1–D6).
        \\
        \\ Options:
        \\   --derived=<path>      Path to a .derived.json, as `tensorspine --derive` emits it (required)
        \\   --refusals            Report, per contract, the occurrences no primitive implements
        \\   --checkpoint=<path>   The safetensors repository or file D3's locations name
        \\   --until=<value>       Evaluate the ancestor closure of one D2 value, e.g. embed.output
        \\   --ids=<n,n,...>       The token identifiers to run (default: the llama3-8b fixture's)
        \\   --out=<path>          Write the result's raw bytes here
        \\   --dump-mlir=<dir>     Ask XLA to dump the emitted IR here
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

/// The llama3-8b fixture's prompt: "<|begin_of_text|>The capital of France is".
const default_ids = [_]i32{ 128000, 791, 6864, 315, 9822, 374 };

fn parseIds(allocator: std.mem.Allocator, spec: ?[]const u8) ![]i32 {
    const text = spec orelse return allocator.dupe(i32, &default_ids);
    var list: std.ArrayList(i32) = .empty;
    errdefer list.deinit(allocator);
    var it = std.mem.tokenizeAny(u8, text, ", ");
    while (it.next()) |token| {
        try list.append(allocator, try std.fmt.parseInt(i32, token, 10));
    }
    return list.toOwnedSlice(allocator);
}

fn evaluate(allocator: std.mem.Allocator, io: std.Io, args: Args, g: *const graph.Graph) !void {
    const checkpoint = args.checkpoint orelse {
        log.err("--until needs --checkpoint: the parameters come from where D3 locates them", .{});
        return error.MissingCheckpoint;
    };

    const ids = try parseIds(allocator, args.ids);
    defer allocator.free(ids);

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);
    log.info("platform: {s}", .{@tagName(platform.target)});

    var tensors: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, checkpoint);
    defer tensors.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &tensors);
    defer store.deinit();

    // What to evaluate. Compute is f32 on CPU, as the reference does, so a comparison
    // against it is a comparison of the mathematics and not of two roundings.
    var p = try plan.until(allocator, g, args.until.?, @intCast(ids.len), .f32);
    defer p.deinit();
    log.info("{d} step(s) to reach {s}, {d} public input(s), {d} parameter tensor(s)", .{
        p.steps.len, args.until.?, p.publics.len, p.params_used.len,
    });

    // The parameters the plan needs, by D3's locations — nothing per model.
    var model = try loader.locate(allocator, g, store.view(), p.params_used);
    defer model.deinit(allocator);

    const publics = try allocator.alloc(zml.Tensor, p.publics.len);
    defer allocator.free(publics);
    for (p.public_shapes, publics) |shape, *t| t.* = .fromShape(shape);

    var exe = try zml.module.compile(allocator, io, emit.forward, .{ plan.Handle.of(&p), model.params, publics }, platform, .{
        .program_name = g.model(),
        .xla_dump_to = args.@"dump-mlir",
    });
    defer exe.deinit();
    log.info("compiled", .{});

    // The weights, into buffers ZML pairs with the model by visit order.
    var buffers = try zml.mem.bufferize(allocator, loader.Model, &model);
    defer allocator.free(buffers.params);
    var weights: zml.io.Loader = try .init(allocator, platform, .default);
    defer weights.deinit();
    weights.load(io, loader.Model, &model, &buffers, &store, &.{}, .{});
    try weights.await(io);
    log.info("{Bi:.2} of weights loaded", .{weights.bytes_loaded.raw});

    var token_buffer = try zml.Buffer.fromSlice(
        io,
        platform,
        .init(p.public_shapes[0], std.mem.sliceAsBytes(ids)),
        platform.replicated_sharding,
    );
    defer token_buffer.deinit();

    var call_args = try exe.args(allocator);
    defer call_args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    call_args.set(.{ buffers.params, token_buffer });
    exe.call(call_args, &results);

    var out = results.get(zml.Buffer);
    defer out.deinit();
    const slice = try out.toSliceAlloc(allocator, io);
    defer allocator.free(slice.bytes);

    log.info("{s} = {f}", .{ args.until.?, out.shape() });
    if (args.out) |path| {
        const file = try std.Io.Dir.cwd().createFile(io, path, .{});
        defer file.close(io);
        var buffer: [4096]u8 = undefined;
        var writer = file.writer(io, &buffer);
        try writer.interface.writeAll(slice.bytes);
        try writer.interface.flush();
        log.info("{d} bytes written to {s}", .{ slice.bytes.len, path });
    }
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    // `bazel run` executes from the runfiles tree; go back to the shell's directory
    // so a relative path means what the user typed.
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

    if (args.refusals) {
        _ = try reportRefusals(allocator, &g);
        return;
    }
    if (args.until != null) return evaluate(allocator, init.io, args, &g);

    // Every member of D3 and D4 resolves, and every ordered node is known.
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
}
