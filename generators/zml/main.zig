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
const capabilities = @import("capabilities.zig");
const chat = @import("chat.zig");
const session = @import("session.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const log = std.log.scoped(.tspl);

const Args = struct {
    derived: []const u8 = "",
    refusals: bool = false,
    checkpoint: ?[]const u8 = null,
    until: ?[]const u8 = null,
    ids: ?[]const u8 = null,
    capacity: ?u32 = null,
    @"max-tokens": u32 = 8,
    split: u32 = 1,
    chat: bool = false,
    tokenizer: ?[]const u8 = null,
    capabilities: ?[]const u8 = null,
    version: []const u8 = "unknown",
    generated: []const u8 = "unknown",
    compute: []const u8 = "f32",
    @"separate-states": bool = false,
    out: ?[]const u8 = null,
    dump: ?[]const u8 = null,
    @"dump-mlir": ?[]const u8 = null,

    pub const help =
        \\ Use tspl --derived=<path> [--refusals] [--checkpoint=<dir> --until=<value>]
        \\
        \\ Run a tensorspine/2.0 model from its derived document (D1–D6).
        \\
        \\ Options:
        \\   --derived=<path>      Path to a .derived.json, as `tensorspine --derive` emits it (required)
        \\   --capabilities=<path> Write this generator's manifest, from the primitives' own tables
        \\   --version=<text>      What the manifest records as this generator's version
        \\   --generated=<date>    What the manifest records as its date
        \\   --refusals            Report, per contract, the occurrences no primitive implements
        \\   --checkpoint=<path>   The safetensors repository or file D3's locations name
        \\   --until=<value>       Evaluate the ancestor closure of one D2 value, e.g. embed.output
        \\   --ids=<n,n,...>       The token identifiers to run (default: the llama3-8b fixture's)
        \\   --capacity=<n>        Positions a growing state holds (default: the prompt plus --max-tokens)
        \\   --chat                Converse: a turn is tokenised, fed, and answered until a stop
        \\                         token, the states carried from one turn to the next
        \\   --tokenizer=<path>    tokenizer.json (default: the checkpoint's own)
        \\   --max-tokens=<n>      Tokens to answer with: exactly that many when generating,
        \\                         which has no stopping rule, and at most that many in a chat
        \\                         turn, which also stops on a stop token (default: 8)
        \\   --split=<n>           Compile and run the graph as n programs in sequence; XLA's
        \\                         scratch holds an f32 copy of every weight one program's
        \\                         matmuls touch, so cutting bounds a run's memory
        \\   --separate-states     One buffer per D4 identity instead of one per family; the
        \\                         packed layout is the default and is a serving choice
        \\   --compute=<dtype>     f32 (default) or bf16. f32 upcasts every weight inside the graph,
        \\                         which doubles what a run holds; bf16 computes at the checkpoint's
        \\                         own precision, as ZML's hand-written models do
        \\   --out=<path>          Write the result's raw bytes here
        \\   --dump=<dir>          Also write every state buffer, named by its D4 identity
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

/// Resident set size, in bytes — what the process actually holds. Reported at each
/// phase because a generator that cannot say where its memory went cannot be trusted
/// with a model that barely fits.
fn rss(io: std.Io) u64 {
    var buffer: [256]u8 = undefined;
    const file = std.Io.Dir.cwd().openFile(io, "/proc/self/statm", .{}) catch return 0;
    defer file.close(io);
    var reader = file.reader(io, &buffer);
    const line = reader.interface.takeDelimiterExclusive('\n') catch return 0;
    var it = std.mem.tokenizeScalar(u8, line, ' ');
    _ = it.next() orelse return 0;                       // total program size
    const pages = std.fmt.parseInt(u64, it.next() orelse return 0, 10) catch return 0;
    return pages * std.heap.pageSize();
}

fn reportRss(io: std.Io, what: []const u8) void {
    log.info("  rss after {s}: {Bi:.2}", .{ what, rss(io) });
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

/// One compiled arity, as one or more programs run in sequence.
///
/// A compiled graph has static shapes, so prefill and decode are two arities; and a
/// long graph is cut into several programs because XLA's scratch for one program holds
/// an f32 copy of every weight that program's matmuls touch — it upcasts bf16 dots on
/// CPU — so a whole model in one program needs about three times its own weights.
/// Cutting bounds that to the largest group, and the numbers do not move.
fn generate(allocator: std.mem.Allocator, io: std.Io, args: Args, g: *const graph.Graph) !void {
    const checkpoint = args.checkpoint orelse {
        log.err("generating needs --checkpoint: the parameters come from where D3 locates them", .{});
        return error.MissingCheckpoint;
    };
    const output = g.generative() orelse {
        log.err("{s} has no generative output; use --until to evaluate one value", .{g.model()});
        return error.NotGenerative;
    };
    const target = try std.fmt.allocPrint(allocator, "{s}.{s}", .{ output.node, output.port });
    defer allocator.free(target);

    const ids = try parseIds(allocator, args.ids);
    defer allocator.free(ids);

    // One device. ZML's CPU default is four, and a replicated parameter is copied to
    // each of them — four times the weights resident, before anything is computed.
    // Sharding is a non-goal here (the manifest declares no partitions), so one device
    // is both what this generator means and what fits.
    const platform: *zml.Platform = try .auto(allocator, io, .{ .cpu = .{ .device_count = 1 } });
    defer platform.deinit(allocator, io);
    log.info("platform: {s}, {d} device(s)", .{ @tagName(platform.target), platform.devices.len });

    var tensors: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, checkpoint);
    defer tensors.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &tensors);
    defer store.deinit();

    // Capacity holds the prompt and everything generated: deployment intent, not a
    // document fact (§7).
    const capacity: i64 = if (args.capacity) |c| @intCast(c) else @intCast(ids.len + args.@"max-tokens");
    const compute = try dtypes.of(args.compute);
    log.info("computing in {s}", .{@tagName(compute)});

    // The parameters, by D3's locations. Both arities reach the same value, so both
    // need the same identities in the same order — asserted rather than assumed.
    var shape_plan = try plan.until(allocator, g, target, @intCast(ids.len), capacity, compute, !args.@"separate-states");
    const params_used = try allocator.dupe(usize, shape_plan.params_used);
    defer allocator.free(params_used);
    shape_plan.deinit();

    var model = try loader.locate(allocator, g, store.view(), params_used);
    defer model.deinit(allocator);
    reportRss(io, "locate");

    var prefill = try session.Compiled.init(allocator, io, platform, g, target, @intCast(ids.len), capacity, compute, !args.@"separate-states", args.split, args.@"dump-mlir", model.params);
    defer prefill.deinit(allocator);
    var decode = try session.Compiled.init(allocator, io, platform, g, target, 1, capacity, compute, !args.@"separate-states", args.split, args.@"dump-mlir", model.params);
    defer decode.deinit(allocator);
    if (!std.mem.eql(usize, prefill.plan.params_used, decode.plan.params_used)) {
        log.err("the two arities disagree on which parameters they need", .{});
        return error.InconsistentPlan;
    }
    log.info("{d} occurrences in {d} program(s), {d} parameter tensors, {d} state buffers, capacity {d}", .{
        prefill.plan.steps.len, prefill.plan.groups.len, prefill.plan.params_used.len,
        prefill.plan.state_shapes.len, capacity,
    });
    reportRss(io, "compile");

    var buffers = try zml.mem.bufferize(allocator, loader.Model, &model);
    defer allocator.free(buffers.params);
    defer for (buffers.params) |*b| b.deinit();
    {
        // The loader stages each tensor in host memory before handing it to the device,
        // one at a time, and frees it again — allocations as large as the largest tensor,
        // with no reuse between them. The page allocator returns them to the system at
        // once; a general-purpose allocator retains them, and a debug build's retains
        // them with leak checking on top, which is how 3.18 GiB of weights came to hold
        // 12.93 GiB resident.
        var weights: zml.io.Loader = try .init(std.heap.page_allocator, platform, .default);
        defer weights.deinit();
        weights.load(io, loader.Model, &model, &buffers, &store, &.{}, .{});
        try weights.await(io);
        log.info("{Bi:.2} of weights loaded", .{weights.bytes_loaded.raw});
    }
    reportRss(io, "load");

    const states = try allocator.alloc(zml.Buffer, prefill.plan.state_shapes.len);
    defer allocator.free(states);
    for (prefill.plan.state_shapes, states) |shape, *b| {
        const zeros = try allocator.alloc(u8, shape.byteSize());
        defer allocator.free(zeros);
        @memset(zeros, 0);
        b.* = try .fromBytes(io, platform, shape, platform.replicated_sharding, zeros);
    }
    defer for (states) |*b| b.deinit();

    var generated: std.ArrayList(i32) = .empty;
    defer generated.deinit(allocator);

    var out: zml.Buffer = undefined;
    var start: i32 = 0;
    var next: [1]i32 = undefined;

    // Prefill, then one element at a time.
    var step: usize = 0;
    while (step <= args.@"max-tokens") : (step += 1) {
        const elements: []const i32 = if (step == 0) ids else next[0..1];
        const c = if (step == 0) &prefill else &decode;
        try session.invoke(allocator, io, platform, c, buffers.params, elements, start, states, &out);
        start += @intCast(elements.len);

        const logits = try out.toSliceAlloc(allocator, io);
        defer logits.free(allocator);
        const vocab: usize = @intCast(out.shape().dim(-1));
        next[0] = try session.argmaxLast(logits.bytes, compute, vocab);
        out.deinit();

        if (step < args.@"max-tokens") try generated.append(allocator, next[0]);
        reportRss(io, if (step == 0) "prefill" else "decode");
    }

    log.info("{d} greedy token(s): {any}", .{ generated.items.len, generated.items });
    if (args.out) |path| try write(io, path, std.mem.sliceAsBytes(generated.items));
}

fn evaluate(allocator: std.mem.Allocator, io: std.Io, args: Args, g: *const graph.Graph) !void {
    const checkpoint = args.checkpoint orelse {
        log.err("--until needs --checkpoint: the parameters come from where D3 locates them", .{});
        return error.MissingCheckpoint;
    };

    const ids = try parseIds(allocator, args.ids);
    defer allocator.free(ids);

    const platform: *zml.Platform = try .auto(allocator, io, .{ .cpu = .{ .device_count = 1 } });
    defer platform.deinit(allocator, io);
    log.info("platform: {s}, {d} device(s)", .{ @tagName(platform.target), platform.devices.len });

    var tensors: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, checkpoint);
    defer tensors.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &tensors);
    defer store.deinit();

    const compute = try dtypes.of(args.compute);
    const capacity: i64 = if (args.capacity) |c| @intCast(c) else @intCast(ids.len);

    var shape_plan = try plan.until(allocator, g, args.until.?, @intCast(ids.len), capacity, compute, !args.@"separate-states");
    const params_used = try allocator.dupe(usize, shape_plan.params_used);
    defer allocator.free(params_used);
    shape_plan.deinit();

    var model = try loader.locate(allocator, g, store.view(), params_used);
    defer model.deinit(allocator);

    var c = try session.Compiled.init(allocator, io, platform, g, args.until.?, @intCast(ids.len), capacity, compute, !args.@"separate-states", args.split, args.@"dump-mlir", model.params);
    defer c.deinit(allocator);
    log.info("{d} step(s) to reach {s} in {d} program(s), {d} parameter tensor(s), {d} state buffer(s)", .{
        c.plan.steps.len, args.until.?, c.plan.groups.len, c.plan.params_used.len, c.plan.state_shapes.len,
    });

    var buffers = try zml.mem.bufferize(allocator, loader.Model, &model);
    defer allocator.free(buffers.params);
    defer for (buffers.params) |*b| b.deinit();
    {
        var weights: zml.io.Loader = try .init(std.heap.page_allocator, platform, .default);
        defer weights.deinit();
        weights.load(io, loader.Model, &model, &buffers, &store, &.{}, .{});
        try weights.await(io);
        log.info("{Bi:.2} of weights loaded", .{weights.bytes_loaded.raw});
    }

    const states = try allocator.alloc(zml.Buffer, c.plan.state_shapes.len);
    defer allocator.free(states);
    for (c.plan.state_shapes, states) |shape, *b| {
        const zeros = try allocator.alloc(u8, shape.byteSize());
        defer allocator.free(zeros);
        @memset(zeros, 0);
        b.* = try .fromBytes(io, platform, shape, platform.replicated_sharding, zeros);
    }
    defer for (states) |*b| b.deinit();

    var out: zml.Buffer = undefined;
    try session.invoke(allocator, io, platform, &c, buffers.params, ids, 0, states, &out);
    defer out.deinit();

    const slice = try out.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    log.info("{s} = {f}", .{ args.until.?, out.shape() });
    if (args.out) |path| {
        try write(io, path, slice.bytes);
        log.info("{d} bytes written to {s}", .{ slice.bytes.len, path });
    }

    // A dump is written in the document's terms: one file per D4 identity, whatever
    // layout the run chose to hold them in.
    if (args.dump) |dir| {
        var k: usize = 0;
        for (c.plan.states) |instance| {
            for (instance.components) |component| {
                const bytes = try states[k].toSliceAlloc(allocator, io);
                defer bytes.free(allocator);
                const portion = bytes.bytes.len / instance.members;
                for (instance.identities, 0..) |identity, m| {
                    const name = try std.fmt.allocPrint(allocator, "{s}/{s}.{s}.bin", .{ dir, identity, component.name });
                    defer allocator.free(name);
                    try write(io, name, bytes.bytes[m * portion ..][0..portion]);
                }
                k += 1;
            }
        }
    }
}

fn write(io: std.Io, path: []const u8, bytes: []const u8) !void {
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buffer: [4096]u8 = undefined;
    var writer = file.writer(io, &buffer);
    try writer.interface.writeAll(bytes);
    try writer.interface.flush();
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

    if (args.capabilities) |path| {
        const n = try capabilities.write(allocator, init.io, path, args.version, args.generated);
        log.info("{d} contracts -> {s}", .{ n, path });
        return;
    }

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
    if (args.chat) {
        const checkpoint = args.checkpoint orelse {
            log.err("--chat needs --checkpoint", .{});
            return error.MissingCheckpoint;
        };
        // A chat reads its prompt from the terminal and stops on a stop token, so it has
        // no use for these four. Taking one and ignoring it is how `--until` came to be
        // read as a length, accepted, and silently dropped.
        const unused: ?[]const u8 = if (args.until != null) "--until"
            else if (args.ids != null) "--ids"
            else if (args.out != null) "--out"
            else if (args.dump != null) "--dump"
            else null;
        if (unused) |option| {
            log.err("--chat does not use {s}; the length of a turn is --max-tokens", .{option});
            return error.UnusedOption;
        }
        return chat.run(allocator, init.io, &g, .{
            .checkpoint = checkpoint,
            .tokenizer = args.tokenizer,
            .capacity = if (args.capacity) |c| @intCast(c) else 512,
            .compute = try dtypes.of(args.compute),
            .split = args.split,
            .packed_states = !args.@"separate-states",
            .max_tokens = args.@"max-tokens",
            .dump_mlir = args.@"dump-mlir",
        });
    }
    if (args.until != null) return evaluate(allocator, init.io, args, &g);
    if (args.checkpoint != null) return generate(allocator, init.io, args, &g);

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
