/**
 * The safetensors header, read from bytes: the port of `artifact.read_header` and `read_headers`
 * (plan §5.3, `readHeader`).
 *
 * A safetensors file is an eight-byte little-endian length, a JSON header of exactly that many
 * bytes, and then the weights. `tools/artifact.py` reads it with two `read`s and never touches the
 * payload — "headers only — no weight is read" — which is what makes §4.19's claim true: a 16 GB
 * checkpoint costs a few kilobytes of reading (feature 0.5 measured 33 880 bytes and 15 ms for the
 * whole four-shard `Meta-Llama-3-8B`).
 *
 * ## What is here and what is not (the line feature 0.5 left for this feature to draw)
 *
 * The core is pure: "the UI reads files through `Platform` and hands the core texts and trees"
 * (§5.3), and §5.2's `CheckpointSource` says the same in its own signature —
 * `headerBytes(file): Promise<Uint8Array>`. So **the bytes are the caller's and the format is the
 * core's**:
 *
 * | The core's | The source's (feature 4.1, `LocalCheckpoint` / `HubCheckpoint`) |
 * |---|---|
 * | {@link headerLength}: the eight-byte prefix, and the guards on it | reading the prefix |
 * | {@link readHeader}: the header's JSON, its `__metadata__`, the dtype table, the fold into `name → {dtype, shape, file}` | reading `8 + n` bytes and no more |
 * | {@link shardFiles}: which files an index names, and in which order | listing a directory, globbing, ranging over HTTP |
 * | {@link mergeHeaders}: how several shards' headers make one map | holding them |
 *
 * Feature 0.5's spike (`editor/spikes/headers/`) proved the two-slice reading in a browser over a
 * structural `ByteSource` — a `Blob`, a `File`, an OPFS file, `@huggingface/hub`'s range-backed
 * blob. That abstraction stays spike code on purpose: it is about *where bytes come from*, which
 * is `Platform`'s business, and naming it here would put an asynchronous, source-shaped API into a
 * package whose whole claim is that it computes and does not read. What a source needs from the
 * core to do the two reads is {@link LENGTH_PREFIX_BYTES} and {@link headerLength}, and they are
 * exported for that.
 *
 * ## Where this reads differently from the tools, and why
 *
 * Three places, all of them guards a browser needs and a command line does not — "a browser is
 * handed files nobody checked" (feature 0.5):
 *
 * 1. A length of zero, a length past the end of what the caller read, and a buffer shorter than
 *    the prefix are refused with a {@link HeaderError} instead of being allocated or read past.
 * 2. A length beyond {@link MAX_HEADER_BYTES} is refused, so a corrupt or hostile prefix cannot
 *    ask the page for a gigabyte. `@huggingface/hub` guards at the same number.
 * 3. A header that is not JSON, that is not an object, or whose member is not a tensor entry is a
 *    {@link HeaderError} naming the file. `read_header` lets CPython's `JSONDecodeError` escape and
 *    would fail later on a missing key; the editor opens many files and names the one that failed.
 *
 * The header's JSON is read by the core's own parser (feature 0.3), not by `JSON.parse`: that is
 * what keeps a shape's integers Python's `int` — `bigint` here — so that a shape read from a
 * checkpoint and a shape derived from the document are compared in one numeric world, the way
 * `artifact.check` compares them. Duplicate member names are folded as `dict(pairs)` folds them,
 * which is what `json.loads` does and therefore what `read_header` does.
 */
import { PyTypeError } from '../expr/errors.js';
import { isRecord, toPython, type PyValue } from '../expr/value.js';
import { parse } from '../json/parse.js';
import { put } from '../json/tree.js';
import { comparePythonStrings } from '../schema/repr.js';
import { demand, entries } from '../library/access.js';
import { dtypeOf } from './dtypes.js';

/** The length prefix: eight bytes, little-endian, unsigned — `struct.unpack('<Q', f.read(8))`. */
export const LENGTH_PREFIX_BYTES = 8;

/**
 * The largest header this reader will accept, and the number `@huggingface/hub` guards at.
 *
 * A header is a few kilobytes per shard across the repository's checkpoints (9 512 bytes for
 * `Meta-Llama-3-8B`'s first shard, 88 120 for Voxtral's single file). The cap is not a property of
 * the format; it is what stops a corrupt prefix from being believed.
 */
export const MAX_HEADER_BYTES = 25_000_000;

/** One tensor as `artifact.read_headers` yields it: the language's dtype, the shape, the file. */
export interface HeaderEntry {
  /**
   * The language's `dtype` when {@link known}; the header's own spelling when not — never a
   * lower-cased invention (`artifact/dtypes.ts` states the decision).
   */
  readonly dtype: string;
  /** Whether {@link dtype} is a value of the schema's `dtype` enumeration. */
  readonly known: boolean;
  /**
   * The shape as the header writes it, read as Python reads it: a whole number is a `bigint`.
   *
   * It is not narrowed to integers, because `_check_part` compares it with `==` against a shape
   * D3 evaluated and a malformed header is the caller's to see rather than the reader's to hide.
   */
  readonly shape: readonly PyValue[];
  /** The file the tensor is in, as the caller named it (`os.path.basename` in the tools). */
  readonly file: string;
}

/** `name → {dtype, shape, file}`: the map `artifact.read_headers` answers and V17 is checked against. */
export type CheckpointHeaders = Readonly<Record<string, HeaderEntry>>;

/** What one file's header holds: its tensors, its `__metadata__`, and the header's own length. */
export interface HeaderRead {
  /** The tensors, folded as `read_headers` folds them; `__metadata__` is not one of them. */
  readonly entries: CheckpointHeaders;
  /** The JSON header's length in bytes, from the prefix. */
  readonly headerBytes: number;
  /**
   * The file's `__metadata__`, which `read_header` pops and discards.
   *
   * It is handed back rather than dropped because it is where a unit fixture keeps its document
   * (`docs/TENSORSPINE-FIXTURE.md`). `artifact.read_metadata`'s further step — JSON-decoding each
   * value, `str` on a failure — is not ported: the fixture format is the generators' and no
   * screen of this plan reads one, so the core would be carrying a rule with no reader.
   */
  readonly metadata?: PyValue;
}

/** A file whose header is not one. The message names the file, as the loader's refusals do. */
export class HeaderError extends Error {
  /** The file the caller named. */
  readonly file: string;

  constructor(file: string, reason: string, options?: { cause: unknown }) {
    super(`${file}: ${reason}`, options);
    this.name = 'HeaderError';
    this.file = file;
  }
}

/** The bytes of a buffer, whichever of the two forms the caller holds them in. */
function bytesOf(source: Uint8Array | ArrayBuffer): Uint8Array {
  return source instanceof Uint8Array ? source : new Uint8Array(source);
}

/**
 * The length of the JSON header, from a file's first eight bytes.
 *
 * A `CheckpointSource` reads the prefix, asks this, reads exactly that many further bytes, and
 * hands the whole thing to {@link readHeader} — two reads and no weight, which is the discipline
 * feature 0.5 measured and the reason this is exported at all.
 */
export function headerLength(prefix: Uint8Array | ArrayBuffer, file: string): number {
  const bytes = bytesOf(prefix);
  if (bytes.byteLength < LENGTH_PREFIX_BYTES) {
    throw new HeaderError(
      file,
      `${String(bytes.byteLength)} bytes is shorter than a header length`,
    );
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, LENGTH_PREFIX_BYTES);
  const length = view.getBigUint64(0, true);
  if (length <= 0n) throw new HeaderError(file, 'the header length is zero');
  if (length > BigInt(MAX_HEADER_BYTES)) {
    throw new HeaderError(
      file,
      `the header length ${String(length)} is beyond ${String(MAX_HEADER_BYTES)}`,
    );
  }
  return Number(length);
}

/**
 * One safetensors file's header, from its leading bytes: `artifact.read_header`, folded as
 * `read_headers` folds it.
 *
 * `bytes` is the file's beginning — at least the eight-byte prefix and the header it announces.
 * A caller that read exactly `8 + headerLength(prefix)` bytes passes; so does one holding the whole
 * file, and nothing past the header is looked at either way.
 */
export function readHeader(bytes: Uint8Array | ArrayBuffer, file: string): HeaderRead {
  const all = bytesOf(bytes);
  const length = headerLength(all, file);
  const end = LENGTH_PREFIX_BYTES + length;
  if (end > all.byteLength) {
    throw new HeaderError(
      file,
      `the header length ${String(length)} runs past the end of ${String(all.byteLength)} byte(s)`,
    );
  }
  // `read_header` is `json.loads(f.read(n))` — **bytes**, so CPython decodes them itself, and it
  // raises `UnicodeDecodeError` on a sequence that is not UTF-8 rather than reading a name with a
  // replacement character in it. `fatal` is what makes this refuse the same file: without it a
  // corrupted shard decodes to a name spelled with U+FFFD, parses, and V17 then reports the
  // document's tensor absent and the garbled one named by no location — two refusals about the
  // document where the truth is one about the file.
  //
  // The byte-order mark is **kept stripped**, which is the default: `json.loads` of bytes runs
  // `detect_encoding` first and reads a BOM-prefixed header as `utf-8-sig`, so a header written
  // with one is a header CPython accepts.
  let text: string;
  try {
    text = new TextDecoder('utf-8', { fatal: true }).decode(all.subarray(LENGTH_PREFIX_BYTES, end));
  } catch (error) {
    throw new HeaderError(file, `the header is not UTF-8 (${messageOf(error)})`, { cause: error });
  }
  let header: PyValue;
  try {
    // `json.loads` takes the last of two members of one name; the core's parser refuses one by
    // default (V12), so the plain reading is asked for here, as the loader asks for it.
    header = toPython(parse(text, { duplicates: 'last' }));
  } catch (error) {
    throw new HeaderError(file, `the header is not JSON (${messageOf(error)})`, { cause: error });
  }
  if (!isRecord(header)) throw new HeaderError(file, 'the header is not a JSON object');

  const folded: Record<string, HeaderEntry> = {};
  let metadata: PyValue | undefined;
  for (const [name, value] of entries(header)) {
    if (name === '__metadata__') {
      metadata = value;
      continue;
    }
    put(folded, name, entryOf(value, name, file));
  }
  return {
    entries: folded,
    headerBytes: length,
    ...(metadata === undefined ? {} : { metadata }),
  };
}

/** One member of a header, as `read_headers` reads it: `{dtype, shape}` and nothing else is used. */
function entryOf(value: PyValue, name: string, file: string): HeaderEntry {
  if (!isRecord(value)) throw new HeaderError(file, `'${name}' is not a tensor entry`);
  const dtype = value['dtype'];
  const shape = value['shape'];
  if (typeof dtype !== 'string') {
    throw new HeaderError(file, `'${name}' declares no dtype`);
  }
  if (!Array.isArray(shape)) {
    throw new HeaderError(file, `'${name}' declares no shape`);
  }
  return { ...dtypeOf(dtype), shape: [...(shape as readonly PyValue[])], file };
}

/** An exception's own words, for a message that quotes it. */
function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * The files a `model.safetensors.index.json` names, in the order `read_headers` reads them.
 *
 * `sorted({os.path.join(checkpoint, v) for v in json.load(f)['weight_map'].values()})`: the
 * distinct values of the weight map, sorted — a code-point sort, since Python's is. The order is
 * not decoration: {@link mergeHeaders} lets a later shard's entry replace an earlier one's, so two
 * shards that declared the same tensor name would be resolved by it.
 *
 * This is the pure half of `read_headers`; choosing between an index, a glob and a single file,
 * and turning a name into bytes, is the `CheckpointSource`'s (§5.2).
 */
export function shardFiles(index: PyValue): string[] {
  const seen = new Set<string>();
  for (const [name, value] of entries(demand(index, 'weight_map'))) {
    if (typeof value !== 'string') {
      throw new PyTypeError(`weight_map['${name}'] is not a file name`);
    }
    seen.add(value);
  }
  return [...seen].sort(comparePythonStrings);
}

/**
 * Several files' headers as one map: `read_headers`' loop, where a later file's entry wins.
 *
 * `out[name] = {…}` over the files in order, so the caller's order is the rule — which is why
 * {@link shardFiles} sorts.
 */
export function mergeHeaders(reads: readonly CheckpointHeaders[]): CheckpointHeaders {
  const merged: Record<string, HeaderEntry> = {};
  for (const one of reads) {
    for (const name of Object.keys(one)) put(merged, name, headerAt(one, name) as HeaderEntry);
  }
  return merged;
}

/** `headers.get(name)`: the entry, or `undefined` — never something off the prototype chain. */
export function headerAt(headers: CheckpointHeaders, name: string): HeaderEntry | undefined {
  return Object.hasOwn(headers, name) ? headers[name] : undefined;
}
