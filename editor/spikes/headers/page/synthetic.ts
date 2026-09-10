/**
 * A synthetic `.safetensors`, built in the page — feature 0.5's first test material.
 *
 * The feature asks for "a synthetic `.safetensors` built in the test (8-byte length, JSON
 * header, no payload needed beyond the header) parsed from a `Blob` slice". What is built here
 * is the whole file rather than the header alone, because the interesting measurement is what
 * reading the header of a *large* file costs: the payload is written as a run of identical
 * one-megabyte chunks, so a 256 MiB file costs one megabyte of source data to make, and the
 * reader still has to prove it never asked for any of it.
 *
 * The dtype widths below are the file format's own, not the language's: they exist to place
 * `data_offsets` correctly so that the file is a real safetensors file and not a shape that only
 * this reader would accept.
 */

/** What the caller asks for: a tensor's name, its safetensors dtype, its shape. */
export interface SyntheticTensor {
  readonly name: string;
  readonly dtype: string;
  readonly shape: readonly number[];
}

/** A built file: the bytes before the payload, and how long the payload is. */
export interface Synthetic {
  /** The eight-byte length prefix followed by the JSON header. */
  readonly prefix: Uint8Array<ArrayBuffer>;
  /** The JSON header's length, as the prefix states it. */
  readonly headerBytes: number;
  /** The payload's length, from the last tensor's end offset. */
  readonly payloadBytes: number;
  /** The header as an object, for a test to compare against what the reader returns. */
  readonly header: Record<string, unknown>;
}

/** Bytes per element of the safetensors dtypes this spike writes. */
const WIDTHS: Readonly<Record<string, number>> = {
  BF16: 2,
  BOOL: 1,
  F16: 2,
  F32: 4,
  F64: 8,
  F8_E4M3: 1,
  F8_E5M2: 1,
  I16: 2,
  I32: 4,
  I64: 8,
  I8: 1,
  U16: 2,
  U32: 4,
  U64: 8,
  U8: 1,
};

/**
 * The header of a synthetic file, with `data_offsets` laid out in the order the tensors are
 * given. `padding` appends that many spaces inside the header's declared length, which is what
 * the reference writer does to align the payload — a header a reader must still parse.
 */
export function synthesise(
  tensors: readonly SyntheticTensor[],
  options: { readonly metadata?: Record<string, string>; readonly padding?: number } = {},
): Synthetic {
  const header: Record<string, unknown> = {};
  if (options.metadata !== undefined) header['__metadata__'] = options.metadata;
  let offset = 0;
  for (const tensor of tensors) {
    const width = WIDTHS[tensor.dtype];
    if (width === undefined) throw new Error(`no width for the synthetic dtype ${tensor.dtype}`);
    const bytes = tensor.shape.reduce((product, extent) => product * extent, width);
    header[tensor.name] = { dtype: tensor.dtype, shape: [...tensor.shape], data_offsets: [offset, offset + bytes] };
    offset += bytes;
  }
  const json = JSON.stringify(header) + ' '.repeat(options.padding ?? 0);
  const body = new TextEncoder().encode(json);
  const prefix = new Uint8Array(8 + body.byteLength);
  new DataView(prefix.buffer).setBigUint64(0, BigInt(body.byteLength), true);
  prefix.set(body, 8);
  return { prefix, headerBytes: body.byteLength, payloadBytes: offset, header };
}

/** One megabyte of payload, shared by every chunk of every synthetic blob. */
const CHUNK: Uint8Array<ArrayBuffer> = new Uint8Array(1 << 20);

/**
 * The file as a `Blob`: the prefix and header, then the payload in whole chunks and a remainder.
 *
 * A `Blob` copies what it is given, so the payload costs its own size in the browser's storage
 * and nothing more in this page's heap.
 */
export function syntheticBlob(file: Synthetic): Blob {
  const parts: BlobPart[] = [file.prefix];
  let remaining = file.payloadBytes;
  while (remaining >= CHUNK.byteLength) {
    parts.push(CHUNK);
    remaining -= CHUNK.byteLength;
  }
  if (remaining > 0) parts.push(new Uint8Array(remaining));
  return new Blob(parts, { type: 'application/octet-stream' });
}
