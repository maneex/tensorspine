//! Parameters, from D3's locations (Z02, Z03).
//!
//! One `Model` type serves every document. `params` holds one tensor per D3
//! *identity*, in D3 order — never one per slot: `emitMlir` panics when a tensor id
//! appears twice among a function's arguments, and a tied identity (embed and
//! lm_head sharing a table) would do exactly that. D3 already deduplicates, so the
//! language's own product is what keeps the invariant.
//!
//! Tensors are created untagged. Axis names travel as extents, not as tags: at run
//! time ZML matches tags by pointer identity, so a tag interned from a document
//! would never equal a primitive's comptime `.d` (Z07).

const std = @import("std");

const zml = @import("zml");

const graph = @import("graph.zig");

const log = std.log.scoped(.tspl);

pub const Error = error{
    /// D3 gives no location for a tensor — the document was written before locations,
    /// or the checkpoint is not the one it describes.
    NotLocated,
    /// A location form this loader does not assemble yet. A finding, not a workaround.
    UnsupportedLocation,
    /// The checkpoint has no tensor by that name.
    NotInCheckpoint,
    /// The checkpoint's tensor is not the shape D3 declares.
    ShapeMismatch,
};

/// The model, for every document. ZML's machinery is structural, not nominal:
/// `bufferize`, `Loader.load` and `emitMlir` are `meta` walks that recurse into
/// runtime-length slices, so this is indistinguishable from a hand-written model.
pub const Model = struct {
    params: []zml.Tensor,

    pub fn deinit(self: *Model, allocator: std.mem.Allocator) void {
        allocator.free(self.params);
    }
};

/// Create one tensor per identity the plan uses, in the order it expects them, from
/// the checkpoint by location. `used` holds D3 indices: evaluating one value must not
/// read a whole checkpoint.
pub fn locate(
    allocator: std.mem.Allocator,
    g: *const graph.Graph,
    store: zml.io.TensorStore.View,
    used: []const usize,
) !Model {
    const tensors = g.doc().d3.tensors;
    const params = try allocator.alloc(zml.Tensor, used.len);
    errdefer allocator.free(params);

    for (used, 0..) |ti, i| {
        const t = tensors[ti];
        const loc = t.location orelse {
            log.err("{s}: D3 gives no location", .{t.identity});
            return Error.NotLocated;
        };
        const name = loc.tensor orelse {
            log.err("{s}: location form not implemented (only `tensor` today)", .{t.identity});
            return Error.UnsupportedLocation;
        };
        params[i] = try create(store, t, name);
    }
    return .{ .params = params };
}

/// One tensor, at the shape D3 declares.
///
/// D3 is the authority on the shape; the checkpoint is asked only to hold the same
/// elements. A checkpoint that writes them under a different rank — a depthwise
/// convolution kernel stored `[channels, 1, kernel]` where the contract declares
/// `[channels, kernel]` — is bound at D3's, because the bytes are identical and a
/// primitive is written against the document, never against a framework's habits.
/// The reference generator reshapes for the same reason, so the two agree.
fn create(store: zml.io.TensorStore.View, t: graph.Tensor, name: []const u8) !zml.Tensor {
    const source = store.getShape(name) orelse {
        log.err("{s}: checkpoint has no tensor '{s}'", .{ t.identity, name });
        return Error.NotInCheckpoint;
    };

    var declared: zml.Shape = .init(.{}, source.dtype());
    for (t.shape) |axis| declared = declared.appendDim(axis.extent, null);

    if (std.mem.eql(i64, source.dims(), declared.dims())) {
        return store.maybeCreateTensor(name, null, .replicated).?;
    }
    if (source.count() != declared.count()) {
        log.err("{s}: D3 says {any}, checkpoint '{s}' has {any} — not the same elements", .{
            t.identity, declared.dims(), name, source.dims(),
        });
        return Error.ShapeMismatch;
    }
    return store.maybeCreateBinding(&.{name}, declared.withReplicatedPartitioning()).?;
}
