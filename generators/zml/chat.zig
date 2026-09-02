//! A chat over a Tensorspine document, as ZML's own example is a chat over a
//! hand-written model.
//!
//! Nothing here knows which model it is talking to: the graph, the parameters and the
//! states all come from the derived document and the checkpoint. What this file adds is
//! everything the language deliberately does not describe — a tokenizer, a stopping
//! rule, a turn — which is the serving application's, not the document's.
//!
//! One arity is compiled, for a single element. The prompt is fed through it one token
//! at a time rather than as one wide invocation: a compiled graph has static shapes, so
//! a prefill of n tokens is a different program for every n a conversation happens to
//! produce. Feeding one at a time costs the same forward passes and compiles once. A
//! serving application that cared would compile a few prefill widths and pad to them.

const std = @import("std");

const zml = @import("zml");

const graph = @import("graph.zig");
const loader = @import("loader.zig");
const plan = @import("plan.zig");
const session = @import("session.zig");

const log = std.log.scoped(.tspl);

/// The identifiers a turn stops on. A base model has only end-of-text and will run to
/// the limit; an instruction-tuned one ends its turn with `<|eot_id|>`.
const stop_tokens = [_][]const u8{ "<|end_of_text|>", "<|eot_id|>", "</s>", "<|im_end|>" };

pub const Options = struct {
    checkpoint: []const u8,
    tokenizer: ?[]const u8,
    capacity: i64,
    compute: zml.DataType,
    split: u32,
    packed_states: bool,
    max_tokens: u32,
    dump_mlir: ?[]const u8,
    /// One turn, given on the command line, with no terminal to read from. The same
    /// feed and the same stopping rule as a typed turn — which is what the interactive
    /// form is made of, and what a script on a remote machine can actually use.
    prompt: ?[]const u8,
};

fn tokenizerBytes(allocator: std.mem.Allocator, io: std.Io, opts: Options) ![]u8 {
    const path = if (opts.tokenizer) |p|
        try allocator.dupe(u8, p)
    else
        try std.fmt.allocPrint(allocator, "{s}/tokenizer.json", .{opts.checkpoint});
    defer allocator.free(path);

    const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch |err| {
        log.err("no tokenizer at {s}: {s} (use --tokenizer=PATH)", .{ path, @errorName(err) });
        return err;
    };
    defer file.close(io);
    var reader = file.reader(io, &.{});
    return reader.interface.readAlloc(allocator, try file.length(io));
}

pub fn run(allocator: std.mem.Allocator, io: std.Io, g: *const graph.Graph, opts: Options) !void {
    const output = g.generative() orelse {
        log.err("{s} has no generative output: there is nothing to chat with", .{g.model()});
        return error.NotGenerative;
    };
    const target = try std.fmt.allocPrint(allocator, "{s}.{s}", .{ output.node, output.port });
    defer allocator.free(target);

    // The tokenizer is the serving application's: the language says a token stream, not
    // what a token is.
    var tokenizer = b: {
        const bytes = try tokenizerBytes(allocator, io, opts);
        defer allocator.free(bytes);
        break :b try zml.tokenizer.Tokenizer.fromBytes(allocator, bytes);
    };
    defer tokenizer.deinit();

    var stops: std.ArrayList(u32) = .empty;
    defer stops.deinit(allocator);
    for (stop_tokens) |name| {
        if (tokenizer.tokenId(name)) |id| try stops.append(allocator, id);
    }

    const platform: *zml.Platform = try .auto(allocator, io, .{ .cpu = .{ .device_count = 1 } });
    defer platform.deinit(allocator, io);
    // `auto` falls back to the CPU without being asked, and the other two entry points
    // say which backend they got. A chat that stays silent lets a machine with a device
    // answer for an hour off its processor.
    log.info("platform: {s}, {d} device(s)", .{ @tagName(platform.target), platform.devices.len });

    var tensors: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, opts.checkpoint);
    defer tensors.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &tensors);
    defer store.deinit();

    var shape_plan = try plan.until(allocator, g, target, 1, opts.capacity, opts.compute, opts.packed_states);
    const params_used = try allocator.dupe(usize, shape_plan.params_used);
    defer allocator.free(params_used);
    shape_plan.deinit();

    var model = try loader.locate(allocator, g, store.view(), params_used);
    defer model.deinit(allocator);

    var step = try session.Compiled.init(
        allocator, io, platform, g, target, 1,
        opts.capacity, opts.compute, opts.packed_states, opts.split, opts.dump_mlir, model.params,
    );
    defer step.deinit(allocator);

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

    // One session's states, for one (session, branch) key, carried across every turn:
    // the conversation is the growing state, which is the point.
    const states = try allocator.alloc(zml.Buffer, step.plan.state_shapes.len);
    defer allocator.free(states);
    for (step.plan.state_shapes, states) |shape, *b| {
        const zeros = try allocator.alloc(u8, shape.byteSize());
        defer allocator.free(zeros);
        @memset(zeros, 0);
        b.* = try .fromBytes(io, platform, shape, platform.replicated_sharding, zeros);
    }
    defer for (states) |*b| b.deinit();

    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    var in_buffer: [4096]u8 = undefined;
    var out_buffer: [4096]u8 = undefined;
    var stdin = std.Io.File.stdin().reader(io, &in_buffer);
    var stdout = std.Io.File.stdout().writer(io, &out_buffer);
    const out = &stdout.interface;

    const interactive = opts.prompt == null;
    if (interactive) {
        try out.print(
            "\n{s}: {d} occurrences in {d} program(s), capacity {d}, computing in {s}.\n" ++
                "Type a prompt, or an empty line to leave.\n\n",
            .{ g.model(), step.plan.steps.len, step.plan.groups.len, opts.capacity, @tagName(opts.compute) },
        );
        try out.flush();
    }

    // Where the states reach: the conversation so far, in positions.
    var position: i32 = 0;

    while (true) {
        const line = opts.prompt orelse line: {
            try out.writeAll("> ");
            try out.flush();
            break :line stdin.interface.takeDelimiterExclusive('\n') catch |err| switch (err) {
                error.EndOfStream => break,
                else => return err,
            };
        };
        const prompt = std.mem.trim(u8, line, " \t\r");
        if (prompt.len == 0) break;

        const ids = try encoder.encodeAlloc(allocator, prompt);
        defer allocator.free(ids);
        if (position + @as(i32, @intCast(ids.len)) >= opts.capacity) {
            try out.print("(the capacity of {d} positions is full; restart with a larger --capacity)\n", .{opts.capacity});
            try out.flush();
            break;
        }

        // The prompt, one token at a time; only the last invocation's logits matter.
        var next: i32 = 0;
        for (ids) |id| {
            next = try feed(allocator, io, platform, &step, buffers.params, @intCast(id), position, states, opts.compute);
            position += 1;
        }

        // A decoder per turn. `Decoder.reset` would be the way to reuse one, but it does
        // not compile in ZML today: the wrapper is declared `void` over a variant whose
        // `reset` returns an error union (`zml/tokenizer/tokenizer.zig:71`).
        var decoder = try tokenizer.decoder();
        defer decoder.deinit();

        // Then its own output, fed back into the stream it belongs to (§7).
        var produced: u32 = 0;
        while (produced < opts.max_tokens and position < opts.capacity) : (produced += 1) {
            if (std.mem.indexOfScalar(u32, stops.items, @intCast(next)) != null) break;
            var piece_buffer: [256]u8 = undefined;
            const piece = try decoder.feedOne(@intCast(next), &piece_buffer);
            try out.writeAll(piece);
            try out.flush();

            next = try feed(allocator, io, platform, &step, buffers.params, next, position, states, opts.compute);
            position += 1;
        }
        try out.writeAll(if (interactive) "\n\n" else "\n");
        try out.flush();
        if (!interactive) return;
    }

    try out.writeAll("\n");
    try out.flush();
}

/// One element through the graph: the identifier in, the next identifier out, the states
/// advanced in place.
fn feed(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    step: *const session.Compiled,
    params: []zml.Buffer,
    id: i32,
    position: i32,
    states: []zml.Buffer,
    compute: zml.DataType,
) !i32 {
    var logits: zml.Buffer = undefined;
    try session.invoke(allocator, io, platform, step, params, &.{id}, position, states, &logits);
    defer logits.deinit();

    const slice = try logits.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    return session.argmaxLast(slice.bytes, compute, @intCast(logits.shape().dim(-1)));
}
