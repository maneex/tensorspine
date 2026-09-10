/**
 * The safetensors header, read from a slice — the spike's half of feature 0.5.
 *
 * A safetensors file is an 8-byte little-endian length, a JSON header of exactly that many
 * bytes, and then the weights. `tools/artifact.py` reads it with two `read`s and never touches
 * the payload:
 *
 *     n = struct.unpack('<Q', f.read(8))[0]
 *     header = json.loads(f.read(n))
 *     header.pop('__metadata__', None)
 *
 * This module is the same two reads over anything that can be sliced — a `Blob`, a `File` the
 * user picked, an OPFS file, or `@huggingface/hub`'s `WebBlob`/`XetBlob`, which turn a slice
 * into an HTTP range request. That is the whole point of the spike: **one reader, and the source
 * decides where the bytes come from** (plan §2 D10, §4.19 "Sources").
 *
 * It is spike code, not the core. Feature 1.9 ports `artifact.py` into `packages/lang` with the
 * dtype table audited against the schema's `dtype` enum (plan §1, "what the rule does not
 * cover"); what is here exists to prove the browser can do it at all, and to measure what it
 * costs. Two things it does that the tools do not, because a browser is handed files nobody
 * checked: it refuses a length that cannot be a header instead of allocating it, and it reports
 * how many bytes it read, so a test can assert that no weight was.
 */

/**
 * What the reader needs of a byte source: a size, a slice, and a way to get the slice's bytes.
 *
 * `Blob`, `File`, and the blobs `@huggingface/hub` returns all satisfy it structurally, so the
 * reader never names one of them.
 */
export interface ByteSource {
  readonly size: number;
  slice(start: number, end: number): ByteSource;
  arrayBuffer(): Promise<ArrayBuffer>;
}

/** The length prefix: eight bytes, little-endian, unsigned (`<Q`). */
export const LENGTH_PREFIX_BYTES = 8;

/**
 * The largest header this reader will fetch, and the number `@huggingface/hub` uses for the same
 * guard. A header is a few kilobytes per shard in the corpus's checkpoints; the cap is there so
 * that a corrupt or hostile length prefix cannot ask the page for a gigabyte.
 */
export const MAX_HEADER_BYTES = 25_000_000;

/**
 * The safetensors dtype names, mapped onto the language's, exactly as `tools/artifact.py`'s
 * `DTYPES` maps them. The keys are the file format's vocabulary — not the schemas' — which is
 * why the language's own names are the values: `read_headers` stores
 * `DTYPES.get(name, name.lower())`, and this table reproduces that, fallback included.
 */
export const DTYPES: Readonly<Record<string, string>> = {
  BF16: 'bf16',
  BOOL: 'bool',
  F16: 'f16',
  F32: 'f32',
  F64: 'f64',
  F8_E4M3: 'f8e4m3',
  F8_E5M2: 'f8e5m2',
  I16: 'i16',
  I32: 'i32',
  I64: 'i64',
  I8: 'i8',
  U8: 'u8',
};

/** The tools' `DTYPES.get(name, name.lower())`, and nothing more. */
export function dtypeOf(name: string): string {
  return DTYPES[name] ?? name.toLowerCase();
}

/** One tensor as the file's header writes it. */
export interface RawTensor {
  readonly dtype: string;
  readonly shape: readonly number[];
  readonly data_offsets: readonly [number, number];
}

/** One tensor as `artifact.read_headers` yields it: the language's dtype, the shape, the file. */
export interface HeaderEntry {
  readonly dtype: string;
  readonly shape: number[];
  readonly file: string;
}

/** What one file's header says, and what reading it cost. */
export interface HeaderRead {
  /** `name -> {dtype, shape, file}`, the shape of `artifact.read_headers`'s map. */
  readonly entries: Record<string, HeaderEntry>;
  /** The JSON header's own length, from the prefix. */
  readonly headerBytes: number;
  /** Bytes the reader asked the source for: the prefix plus the header, never a weight. */
  readonly bytesRead: number;
  /** The file's `__metadata__`, dropped from `entries` as the tools drop it. */
  readonly metadata: Record<string, string> | undefined;
}

/** A header that is not one. The message names the file, as the loader's messages do. */
export class HeaderError extends Error {
  constructor(file: string, reason: string) {
    super(`${file}: ${reason}`);
    this.name = 'HeaderError';
  }
}

/** The length prefix of a safetensors file, from its first eight bytes. */
export function headerLength(prefix: ArrayBuffer, file: string): number {
  if (prefix.byteLength < LENGTH_PREFIX_BYTES) {
    throw new HeaderError(file, `${String(prefix.byteLength)} bytes is shorter than a header length`);
  }
  const length = new DataView(prefix).getBigUint64(0, true);
  if (length <= 0n) throw new HeaderError(file, 'the header length is zero');
  if (length > BigInt(MAX_HEADER_BYTES)) {
    throw new HeaderError(file, `the header length ${String(length)} is beyond ${String(MAX_HEADER_BYTES)}`);
  }
  return Number(length);
}

/**
 * The header's tensors and its `__metadata__`, from the header's own bytes.
 *
 * `__metadata__` is the one member of a safetensors header that is not a tensor; `read_header`
 * pops it, and `read_metadata` reads it on its own for a unit fixture's document
 * (`docs/TENSORSPINE-FIXTURE.md`). Both are returned here, separately.
 */
export function parseHeader(
  bytes: ArrayBuffer,
  file: string,
): { tensors: Record<string, RawTensor>; metadata: Record<string, string> | undefined } {
  let parsed: unknown;
  try {
    parsed = JSON.parse(new TextDecoder().decode(bytes));
  } catch (error) {
    throw new HeaderError(file, `the header is not JSON (${String(error)})`);
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    throw new HeaderError(file, 'the header is not a JSON object');
  }
  const members = parsed as Record<string, unknown>;
  const metadata = members['__metadata__'] as Record<string, string> | undefined;
  const tensors: Record<string, RawTensor> = {};
  for (const [name, value] of Object.entries(members)) {
    if (name === '__metadata__') continue;
    const tensor = value as Partial<RawTensor>;
    if (typeof tensor.dtype !== 'string' || !Array.isArray(tensor.shape)) {
      throw new HeaderError(file, `'${name}' is not a tensor entry`);
    }
    tensors[name] = tensor as RawTensor;
  }
  return { tensors, metadata };
}

/**
 * The header of one safetensors file, from two slices of it: the eight-byte prefix, then the
 * header itself. Nothing else of the file is touched — which is what `bytesRead` records, and
 * what the tests assert on a file whose payload is hundreds of megabytes.
 */
export async function readHeader(source: ByteSource, file: string): Promise<HeaderRead> {
  if (source.size < LENGTH_PREFIX_BYTES) {
    throw new HeaderError(file, `${String(source.size)} bytes is shorter than a header length`);
  }
  const prefix = await source.slice(0, LENGTH_PREFIX_BYTES).arrayBuffer();
  const headerBytes = headerLength(prefix, file);
  const end = LENGTH_PREFIX_BYTES + headerBytes;
  if (end > source.size) {
    throw new HeaderError(
      file,
      `the header length ${String(headerBytes)} runs past the end of a ${String(source.size)}-byte file`,
    );
  }
  const body = await source.slice(LENGTH_PREFIX_BYTES, end).arrayBuffer();
  const { tensors, metadata } = parseHeader(body, file);
  return {
    entries: entriesOf(tensors, file),
    headerBytes,
    bytesRead: LENGTH_PREFIX_BYTES + headerBytes,
    metadata,
  };
}

/**
 * A parsed header folded into `artifact.read_headers`'s map: the language's dtype names, the
 * shape as a list, and the file the tensor is in. The same fold applies to a header
 * `@huggingface/hub` parsed for us, so both sources answer in one vocabulary.
 */
export function entriesOf(tensors: Record<string, RawTensor>, file: string): Record<string, HeaderEntry> {
  const entries: Record<string, HeaderEntry> = {};
  for (const [name, tensor] of Object.entries(tensors)) {
    entries[name] = { dtype: dtypeOf(tensor.dtype), shape: [...tensor.shape], file };
  }
  return entries;
}

/**
 * A source that remembers what was asked of it, so a test can prove the payload was never read.
 *
 * It is a wrapper, not a replacement: every byte still comes from the `Blob` underneath, so what
 * it counts is what the browser was asked to produce.
 */
export function counting(source: ByteSource): { source: ByteSource; reads: () => readonly ReadSpan[] } {
  const spans: ReadSpan[] = [];
  const wrap = (inner: ByteSource, offset: number): ByteSource => ({
    get size() {
      return inner.size;
    },
    slice: (start: number, end: number) => wrap(inner.slice(start, end), offset + start),
    arrayBuffer: async () => {
      spans.push({ start: offset, bytes: inner.size });
      return inner.arrayBuffer();
    },
  });
  return { source: wrap(source, 0), reads: () => spans };
}

/** One read the source was asked for: where it started and how many bytes it wanted. */
export interface ReadSpan {
  readonly start: number;
  readonly bytes: number;
}
