/**
 * The checkpoint: safetensors headers, and V17 against them — the port of `tools/artifact.py`
 * (plan §5.3, `readHeader` and `checkCheckpoint`; feature 1.9).
 *
 * Two things, and the table between them:
 *
 * - `header.ts` reads a safetensors header from bytes — the eight-byte length, the JSON object,
 *   `name → {dtype, shape, file}` — and says which files an index names and how several shards'
 *   headers make one map. No file is opened: §5.2's `CheckpointSource` hands the core the bytes,
 *   which is what keeps `packages/lang` free of a platform.
 * - `check.ts` is `artifact.check`: every located tensor of a derived document's D3 against those
 *   headers, with the errors word for word as the tools write them, the warnings §4.19 asks for
 *   (a physical tensor no location names; an identity with no location while others have one) and
 *   the three counters the status bar reads.
 * - `dtypes.ts` is the one table of the core whose keys are a *file format's* vocabulary rather
 *   than a schema's — the case plan §1 states as the exception — mapped onto the schema's `dtype`
 *   enumeration and audited against it. Its header explains what it does with a name the format
 *   has and the language has not, which is the decision feature 0.5 recorded as this feature's.
 *
 * V17's other half — locations total or absent, a physical name bound once, slices that neither
 * overlap nor coexist with a whole binding, a `stack` over an axis of the slot, a `slice` offset
 * that resolves — is the *validator's* (`validate/bindings/`, feature 1.6c), and it decides the
 * document alone, without a checkpoint. This module never re-decides any of it.
 */
export { dtypeOf, SAFETENSORS_DTYPES, type HeaderDtype, type SafetensorsDtype } from './dtypes.js';
export {
  headerAt,
  headerLength,
  HeaderError,
  LENGTH_PREFIX_BYTES,
  MAX_HEADER_BYTES,
  mergeHeaders,
  readHeader,
  shardFiles,
  type CheckpointHeaders,
  type HeaderEntry,
  type HeaderRead,
} from './header.js';
export {
  checkCheckpoint,
  formatCheckpointProblem,
  squeeze,
  type CheckpointCheck,
  type CheckpointProblem,
  type CheckpointProblemKind,
  type CheckpointStats,
} from './check.js';
