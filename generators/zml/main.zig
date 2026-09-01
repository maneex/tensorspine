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
    capacity: ?u32 = null,
    steps: u32 = 8,
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
        \\   --refusals            Report, per contract, the occurrences no primitive implements
        \\   --checkpoint=<path>   The safetensors repository or file D3's locations name
        \\   --until=<value>       Evaluate the ancestor closure of one D2 value, e.g. embed.output
        \\   --ids=<n,n,...>       The token identifiers to run (default: the llama3-8b fixture's)
        \\   --capacity=<n>        Positions a growing state holds (default: the prompt plus --steps)
        \\   --steps=<n>           Tokens to generate when --until is absent (default: 8)
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

/// One compiled arity: prefill carries the prompt, decode carries one element. A
/// compiled graph has static shapes, so the two are two executables over the same
/// weights and the same states — which is what a session is.
const Compiled = struct {
    plan: plan.Plan,
    exe: zml.Exe,

    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        g: *const graph.Graph,
        target: []const u8,
        elements: i64,
        capacity: i64,
        compute: zml.DataType,
        packed_states: bool,
        params: []const zml.Tensor,
    ) !Compiled {
        var p = try plan.until(allocator, g, target, elements, capacity, compute, packed_states);
        errdefer p.deinit();

        const publics = try allocator.alloc(zml.Tensor, p.publics.len);
        defer allocator.free(publics);
        for (p.public_shapes, publics) |shape, *t| t.* = .fromShape(shape);

        const states = try allocator.alloc(zml.Tensor, p.state_shapes.len);
        defer allocator.free(states);
        for (p.state_shapes, states) |shape, *t| t.* = .fromShape(shape);

        const start: zml.Tensor = .fromShape(zml.Shape.init(.{}, .i32));
        const exe = try zml.module.compile(allocator, io, emit.forward, .{
            plan.Handle.of(&p), params, publics, start, states,
        }, platform, .{ .program_name = g.model() });
        return .{ .plan = p, .exe = exe };
    }

    fn deinit(self: *Compiled) void {
        self.exe.deinit();
        self.plan.deinit();
    }
};


/// One invocation of a compiled arity: the elements in, the states as they stand, the
/// value out and the states after. States are functional in the graph, so a session is
/// just the buffers carried from one call to the next.
fn invoke(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    c: *const Compiled,
    params: []zml.Buffer,
    ids: []const i32,
    start: i32,
    states: []zml.Buffer,
    out: *zml.Buffer,
) !void {
    var tokens = try zml.Buffer.fromSlice(
        io,
        platform,
        .init(c.plan.public_shapes[0], std.mem.sliceAsBytes(ids)),
        platform.replicated_sharding,
    );
    defer tokens.deinit();

    var start_buffer = try zml.Buffer.scalar(io, platform, start, .i32);
    defer start_buffer.deinit();

    var call_args = try c.exe.args(allocator);
    defer call_args.deinit(allocator);
    var results = try c.exe.results(allocator);
    defer results.deinit(allocator);

    call_args.set(.{ params, tokens, start_buffer, states });
    c.exe.call(call_args, &results);

    const after = try allocator.alloc(zml.Buffer, states.len);
    defer allocator.free(after);
    results.fill(.{ out, &after });

    // The states this call produced replace the ones it read.
    for (states, after) |*before, next| {
        before.deinit();
        before.* = next;
    }
}

/// The identifier of the largest logit of the last element — greedy decoding. Sampling
/// is the serving application's, not the document's: nothing in D1–D6 mentions it. The
/// logits arrive in whatever the compute dtype is, so the comparison decodes rather than
/// assumes.
fn argmaxLast(bytes: []const u8, dt: zml.DataType, vocabulary: usize) !i32 {
    const width = dt.sizeOf();
    const start = bytes.len - vocabulary * width;
    var best: usize = 0;
    var best_value: f32 = -std.math.inf(f32);
    for (0..vocabulary) |i| {
        const at = bytes[start + i * width ..];
        const value: f32 = switch (dt) {
            .f32 => @bitCast(std.mem.readInt(u32, at[0..4], .little)),
            .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, at[0..2], .little)) << 16),
            else => return error.UnsupportedComputeDtype,
        };
        if (value > best_value) {
            best_value = value;
            best = i;
        }
    }
    return @intCast(best);
}

/// Prefill the prompt, then decode one element at a time, feeding each generated
/// identifier back into the stream its generative output belongs to (§7).
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
    const capacity: i64 = if (args.capacity) |c| @intCast(c) else @intCast(ids.len + args.steps);
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

    var prefill = try Compiled.init(allocator, io, platform, g, target, @intCast(ids.len), capacity, compute, !args.@"separate-states", model.params);
    defer prefill.deinit();
    var decode = try Compiled.init(allocator, io, platform, g, target, 1, capacity, compute, !args.@"separate-states", model.params);
    defer decode.deinit();
    if (!std.mem.eql(usize, prefill.plan.params_used, decode.plan.params_used)) {
        log.err("the two arities disagree on which parameters they need", .{});
        return error.InconsistentPlan;
    }
    log.info("{d} occurrences, {d} parameter tensors, {d} state buffers, capacity {d}", .{
        prefill.plan.steps.len, prefill.plan.params_used.len, prefill.plan.state_shapes.len, capacity,
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
    while (step <= args.steps) : (step += 1) {
        const elements: []const i32 = if (step == 0) ids else next[0..1];
        const c = if (step == 0) &prefill else &decode;
        try invoke(allocator, io, platform, c, buffers.params, elements, start, states, &out);
        start += @intCast(elements.len);

        const logits = try out.toSliceAlloc(allocator, io);
        defer allocator.free(logits.bytes);
        const vocab: usize = @intCast(out.shape().dim(-1));
        next[0] = try argmaxLast(logits.bytes, compute, vocab);
        out.deinit();

        if (step < args.steps) try generated.append(allocator, next[0]);
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

    // What to evaluate. Compute is f32 on CPU, as the reference does, so a comparison
    // against it is a comparison of the mathematics and not of two roundings.
    // Capacity is deployment intent, not a document fact: D4 gives the bytes per
    // position, how many positions is the runtime's business (§7).
    const capacity: i64 = if (args.capacity) |c| @intCast(c) else @intCast(ids.len);
    var p = try plan.until(allocator, g, args.until.?, @intCast(ids.len), capacity, try dtypes.of(args.compute), !args.@"separate-states");
    defer p.deinit();
    log.info("{d} step(s) to reach {s}, {d} public input(s), {d} parameter tensor(s), {d} state buffer(s)", .{
        p.steps.len, args.until.?, p.publics.len, p.params_used.len, p.state_shapes.len,
    });

    // The parameters the plan needs, by D3's locations — nothing per model.
    var model = try loader.locate(allocator, g, store.view(), p.params_used);
    defer model.deinit(allocator);

    const publics = try allocator.alloc(zml.Tensor, p.publics.len);
    defer allocator.free(publics);
    for (p.public_shapes, publics) |shape, *t| t.* = .fromShape(shape);

    const state_tensors = try allocator.alloc(zml.Tensor, p.state_shapes.len);
    defer allocator.free(state_tensors);
    for (p.state_shapes, state_tensors) |shape, *t| t.* = .fromShape(shape);

    const start: zml.Tensor = .fromShape(zml.Shape.init(.{}, .i32));

    var exe = try zml.module.compile(allocator, io, emit.forward, .{
        plan.Handle.of(&p), model.params, publics, start, state_tensors,
    }, platform, .{
        .program_name = g.model(),
        .xla_dump_to = args.@"dump-mlir",
    });
    defer exe.deinit();
    log.info("compiled", .{});

    // The weights, into buffers ZML pairs with the model by visit order.
    var buffers = try zml.mem.bufferize(allocator, loader.Model, &model);
    defer allocator.free(buffers.params);
    var weights: zml.io.Loader = try .init(std.heap.page_allocator, platform, .default);
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

    // The states begin zeroed and `start` says where this invocation's elements land.
    // Zeroed, not uninitialised: masked positions still reach the weighted sum with a
    // zero weight, and a NaN there would poison it.
    const state_buffers = try allocator.alloc(zml.Buffer, p.state_shapes.len);
    defer allocator.free(state_buffers);
    for (p.state_shapes, state_buffers) |shape, *b| {
        const zeros = try allocator.alloc(u8, shape.byteSize());
        defer allocator.free(zeros);
        @memset(zeros, 0);
        b.* = try .fromBytes(io, platform, shape, platform.replicated_sharding, zeros);
    }
    defer for (state_buffers) |*b| b.deinit();

    var start_buffer = try zml.Buffer.scalar(io, platform, @as(i32, 0), .i32);
    defer start_buffer.deinit();

    call_args.set(.{ buffers.params, token_buffer, start_buffer, state_buffers });
    exe.call(call_args, &results);

    // `fill` writes into buffers already allocated, which is what a result carrying a
    // runtime-length slice of states needs.
    const state_out = try allocator.alloc(zml.Buffer, p.state_shapes.len);
    defer allocator.free(state_out);
    var out: zml.Buffer = undefined;
    results.fill(.{ &out, &state_out });
    defer out.deinit();
    const slice = try out.toSliceAlloc(allocator, io);
    defer allocator.free(slice.bytes);

    log.info("{s} = {f}", .{ args.until.?, out.shape() });

    // A dump is written in the document's terms: one file per D4 identity, whatever
    // layout the run chose to hold them in. What the serving application packed is its
    // own business and does not leave the generator.
    if (args.dump) |dir| {
        var k: usize = 0;
        for (p.states) |instance| {
            for (instance.components) |component| {
                const bytes = try state_out[k].toSliceAlloc(allocator, io);
                defer allocator.free(bytes.bytes);
                const portion = bytes.bytes.len / instance.members;
                for (instance.identities, 0..) |identity, m| {
                    const name = try std.fmt.allocPrint(allocator, "{s}/{s}.{s}.bin", .{ dir, identity, component.name });
                    defer allocator.free(name);
                    try write(io, name, bytes.bytes[m * portion ..][0..portion]);
                }
                log.info("state {s} x{d} = {f}", .{ component.name, instance.members, state_out[k].shape() });
                k += 1;
            }
        }
    }
    if (args.out) |path| {
        try write(io, path, slice.bytes);
        log.info("{d} bytes written to {s}", .{ slice.bytes.len, path });
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
