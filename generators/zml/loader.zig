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
/// D3 is the authority on the shape, and V17 says exactly how far a checkpoint may
/// differ from it: *"unit axes the physical tensor has and the logical shape lacks
/// being dropped"*. `torch.nn.Conv1d` stores a `[channels, kernel]` kernel as
/// `[channels, 1, kernel]`; that is the whole latitude. The tensor is then bound at
/// D3's shape, because the bytes are identical and a primitive is written against the
/// document, never against a framework's habits — which is why the reference generator
/// reshapes too, and `tools/artifact.py` compares the same way.
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
    if (!squeezedEqual(source.dims(), declared.dims())) {
        log.err("{s}: D3 says {any}, checkpoint '{s}' has {any}", .{
            t.identity, declared.dims(), name, source.dims(),
        });
        return Error.ShapeMismatch;
    }
    return store.maybeCreateBinding(&.{name}, declared.withReplicatedPartitioning()).?;
}

/// V17's comparison: both shapes with their unit axes dropped. Equal element counts
/// would not do — `[16384, 2]` holds as many as `[8192, 4]` and is a different tensor,
/// which a loader that only counted would accept and silently reinterpret.
fn squeezedEqual(physical: []const i64, logical: []const i64) bool {
    var i: usize = 0;
    var j: usize = 0;
    while (true) {
        while (i < physical.len and physical[i] == 1) i += 1;
        while (j < logical.len and logical[j] == 1) j += 1;
        if (i == physical.len or j == logical.len) return i == physical.len and j == logical.len;
        if (physical[i] != logical[j]) return false;
        i += 1;
        j += 1;
    }
}
