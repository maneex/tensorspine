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
        params[i] = store.maybeCreateTensor(name, null, .replicated) orelse {
            log.err("{s}: checkpoint has no tensor '{s}'", .{ t.identity, name });
            return Error.NotInCheckpoint;
        };
        try check(t, params[i]);
    }
    return .{ .params = params };
}

/// The checkpoint agrees with what the document declares. D3 is the authority on the
/// shape; the checkpoint is only asked to match it.
fn check(t: graph.Tensor, tensor: zml.Tensor) !void {
    const dims = tensor.dims();
    if (dims.len != t.shape.len) {
        log.err("{s}: D3 says rank {d}, checkpoint has {d}", .{ t.identity, t.shape.len, dims.len });
        return Error.ShapeMismatch;
    }
    for (t.shape, dims) |axis, dim| {
        if (axis.extent != dim) {
            log.err("{s}: D3 says {d} on {s}, checkpoint has {d}", .{ t.identity, axis.extent, axis.axis, dim });
            return Error.ShapeMismatch;
        }
    }
}
