import { describe, expect, it } from 'vitest';

import {
  dtypeOf,
  headerAt,
  headerLength,
  HeaderError,
  LENGTH_PREFIX_BYTES,
  MAX_HEADER_BYTES,
  mergeHeaders,
  parse,
  readHeader,
  SAFETENSORS_DTYPES,
  shardFiles,
  toPython,
  type CheckpointHeaders,
} from '../../src/index.js';
import { PyKeyError, PyTypeError } from '../../src/expr/errors.js';

// The safetensors header (feature 1.9), read from bytes: `artifact.read_header` and the pure half
// of `read_headers`.
//
// What is proved here is the *format*: the eight-byte prefix, the JSON object, `__metadata__`
// popped, the dtype table, the shape read as Python reads it, and the guards a browser needs that
// a command line does not. That a browser can do the two reads over a `Blob`, an OPFS file and the
// Hub was proved by feature 0.5 and is held to account in the browser layer
// (`apps/web/e2e/headers.spec.ts`); what the core owns is everything after the bytes arrive.

/** A safetensors file's leading bytes: the little-endian length, then the header's own text. */
function fileOf(text: string, trailing = 0): Uint8Array {
  const json = new TextEncoder().encode(text);
  const bytes = new Uint8Array(LENGTH_PREFIX_BYTES + json.byteLength + trailing);
  new DataView(bytes.buffer).setBigUint64(0, BigInt(json.byteLength), true);
  bytes.set(json, LENGTH_PREFIX_BYTES);
  return bytes;
}

/**
 * The same from a value — for the headers whose text `JSON.stringify` writes faithfully.
 *
 * Two of the cases below write their text by hand instead, because a JavaScript object literal
 * cannot express what a safetensors header can: `4.0` comes back out of `JSON.stringify` as `4`,
 * and `{__proto__: …}` sets a prototype rather than declaring a member.
 */
function file(header: unknown, trailing = 0): Uint8Array {
  return fileOf(JSON.stringify(header), trailing);
}

/** The same, with the length prefix saying something else than the header's own length. */
function withLength(header: unknown, length: bigint): Uint8Array {
  const bytes = file(header);
  new DataView(bytes.buffer).setBigUint64(0, length, true);
  return bytes;
}

const ONE = { 'model.norm.weight': { dtype: 'BF16', shape: [4096], data_offsets: [0, 8192] } };

describe('the header, from the bytes a source read', () => {
  it('reads the prefix, the JSON object, and the entries read_headers folds', () => {
    const answer = readHeader(file(ONE), 'model-00001-of-00004.safetensors');
    expect(answer.headerBytes).toBe(JSON.stringify(ONE).length);
    expect(answer.entries).toEqual({
      'model.norm.weight': {
        dtype: 'bf16',
        known: true,
        shape: [4096n],
        file: 'model-00001-of-00004.safetensors',
      },
    });
    // `data_offsets` is the format's and no part of what V17 compares: `read_headers` keeps the
    // dtype, the shape and the file, and so does this.
    expect(Object.keys(answer.entries['model.norm.weight'] ?? {}).sort()).toEqual([
      'dtype',
      'file',
      'known',
      'shape',
    ]);
  });

  it('reads nothing past the header, whatever the caller handed it', () => {
    // A caller holding the whole file and one holding `8 + n` bytes answer the same thing, which
    // is what lets a `CheckpointSource` read two slices and stop (§4.19: a 16 GB checkpoint costs
    // a few kilobytes).
    const exact = file(ONE);
    const padded = file(ONE, 4096);
    padded.fill(0xff, exact.byteLength);
    expect(readHeader(padded, 'x').entries).toEqual(readHeader(exact, 'x').entries);
  });

  it('pops `__metadata__` and hands it back', () => {
    // `read_header` drops it so that it is not read as a tensor; `read_metadata` reads it for a
    // unit fixture's document. The entries never carry it, and the caller can still see it.
    const answer = readHeader(file({ __metadata__: { format: 'pt' }, ...ONE }), 'x');
    expect(Object.keys(answer.entries)).toEqual(['model.norm.weight']);
    expect(answer.metadata).toEqual({ format: 'pt' });
    expect(readHeader(file(ONE), 'x').metadata).toBeUndefined();
  });

  it('reads a shape as Python reads it: a whole number is an integer', () => {
    // The comparison `_check_part` makes is against a shape D3 evaluated, where an extent is a
    // Python `int` — feature 1.2's bigint. Reading the header with `JSON.parse` would make both
    // sides `number` and lose the distinction the rest of the core keeps.
    const answer = readHeader(
      fileOf('{"w": {"dtype": "F32", "shape": [1, 4096, 4.0], "data_offsets": [0, 4]}}'),
      'x',
    );
    expect(answer.entries['w']?.shape).toEqual([1n, 4096n, 4]);
  });

  it('takes the last of two members of one name, as `json.loads` does', () => {
    // `read_header` parses with a plain `json.load`, which installs no duplicate hook: the last
    // wins. The core's parser refuses a duplicate by default (V12), so this caller asks for the
    // plain reading — the one thing `duplicates: 'last'` exists for beside the loader.
    const bytes = fileOf('{"w": {"dtype": "BF16", "shape": [2]}, "w": {"dtype": "F32", "shape": [3]}}');
    expect(readHeader(bytes, 'x').entries['w']).toEqual({
      dtype: 'f32',
      known: true,
      shape: [3n],
      file: 'x',
    });
  });

  it('keeps a tensor named `__proto__` a member of the map', () => {
    // The defence features 0.3, 1.1 and 1.6a each found in its own reading: plain assignment
    // would set the prototype and the tensor would vanish from a map V17 walks.
    const answer = readHeader(fileOf('{"__proto__": {"dtype": "U8", "shape": [1]}}'), 'x');
    expect(Object.keys(answer.entries)).toEqual(['__proto__']);
    expect(headerAt(answer.entries, '__proto__')?.dtype).toBe('u8');
    expect(headerAt(answer.entries, 'absent')).toBeUndefined();
  });
});

describe('the guards a browser needs and the tools do not', () => {
  it('refuses a length of zero, one past the end, and a buffer too short for the prefix', () => {
    expect(() => readHeader(withLength(ONE, 0n), 'a.safetensors')).toThrowError(HeaderError);
    expect(() => readHeader(withLength(ONE, 0n), 'a.safetensors')).toThrowError(
      'a.safetensors: the header length is zero',
    );
    expect(() => readHeader(withLength(ONE, 4096n), 'a.safetensors')).toThrowError(
      'runs past the end of',
    );
    expect(() => readHeader(new Uint8Array(4), 'a.safetensors')).toThrowError(
      'a.safetensors: 4 bytes is shorter than a header length',
    );
  });

  it('refuses a length beyond the cap instead of allocating it', () => {
    const beyond = BigInt(MAX_HEADER_BYTES) + 1n;
    expect(() => headerLength(withLength(ONE, beyond), 'a.safetensors')).toThrowError(
      `the header length ${String(beyond)} is beyond ${String(MAX_HEADER_BYTES)}`,
    );
    expect(headerLength(withLength(ONE, BigInt(MAX_HEADER_BYTES)), 'x')).toBe(MAX_HEADER_BYTES);
  });

  it('names the file when the header is not JSON, not an object, or not made of tensors', () => {
    const broken = new Uint8Array(LENGTH_PREFIX_BYTES + 3);
    new DataView(broken.buffer).setBigUint64(0, 3n, true);
    broken.set(new TextEncoder().encode('{,}'), LENGTH_PREFIX_BYTES);
    expect(() => readHeader(broken, 'a.safetensors')).toThrowError(
      /^a\.safetensors: the header is not JSON \(/,
    );
    expect(() => readHeader(file([1, 2]), 'a.safetensors')).toThrowError(
      'a.safetensors: the header is not a JSON object',
    );
    expect(() => readHeader(file({ w: 3 }), 'a.safetensors')).toThrowError(
      "a.safetensors: 'w' is not a tensor entry",
    );
    expect(() => readHeader(file({ w: { shape: [1] } }), 'a.safetensors')).toThrowError(
      "a.safetensors: 'w' declares no dtype",
    );
    expect(() => readHeader(file({ w: { dtype: 'BF16' } }), 'a.safetensors')).toThrowError(
      "a.safetensors: 'w' declares no shape",
    );
  });

  it('carries the parser’s own refusal as the cause', () => {
    const broken = new Uint8Array(LENGTH_PREFIX_BYTES + 3);
    new DataView(broken.buffer).setBigUint64(0, 3n, true);
    broken.set(new TextEncoder().encode('{,}'), LENGTH_PREFIX_BYTES);
    let caught: unknown;
    try {
      readHeader(broken, 'a.safetensors');
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(HeaderError);
    expect((caught as HeaderError).file).toBe('a.safetensors');
    expect((caught as { cause?: unknown }).cause).toBeInstanceOf(Error);
  });
});

describe('the dtype table', () => {
  it('maps the twelve `tools/artifact.py` maps, to the names it maps them to', () => {
    // The audit (`editor/tests/audit/semantic-tables.test.ts`) is what holds the table to
    // `tools/artifact.py`'s own dictionary and to the schema's enumeration; here is what a reader
    // gets out of it.
    expect(dtypeOf('BF16')).toEqual({ dtype: 'bf16', known: true });
    expect(dtypeOf('F8_E4M3')).toEqual({ dtype: 'f8e4m3', known: true });
    expect(dtypeOf('BOOL')).toEqual({ dtype: 'bool', known: true });
  });

  it('maps `FP4`, whose lower-case form the tools’ fallback already answers', () => {
    expect('FP4'.toLowerCase()).toBe('fp4');
    expect(dtypeOf('FP4')).toEqual({ dtype: 'fp4', known: true });
    expect(Object.keys(SAFETENSORS_DTYPES)).toHaveLength(13);
  });

  it('passes a name it has not through as the file spells it, marked unknown', () => {
    // The decision feature 0.5 left to this one. `read_headers` answers `name.lower()`, so `U16`
    // becomes `u16` and `F8_E4M3FNUZ` becomes `f8_e4m3fnuz` — names that look like dtypes of the
    // language and are none. Nothing is lower-cased here.
    for (const name of ['U16', 'U32', 'U64', 'C64', 'F4', 'E8M0', 'UE8', 'F8_E4M3FNUZ']) {
      expect(dtypeOf(name), name).toEqual({ dtype: name, known: false });
      expect(dtypeOf(name).dtype, name).not.toBe(name.toLowerCase());
    }
  });

  it('marks an unknown dtype on the entry the reader answers', () => {
    const answer = readHeader(file({ w: { dtype: 'F8_E4M3FNUZ', shape: [8] } }), 'x');
    expect(answer.entries['w']).toEqual({
      dtype: 'F8_E4M3FNUZ',
      known: false,
      shape: [8n],
      file: 'x',
    });
  });
});

describe('the pure half of read_headers', () => {
  it('answers the distinct files an index names, sorted', () => {
    const index = toPython(
      parse(`{"metadata": {}, "weight_map": {
        "b.weight": "model-00002-of-00002.safetensors",
        "a.weight": "model-00001-of-00002.safetensors",
        "c.weight": "model-00002-of-00002.safetensors"}}`),
    );
    expect(shardFiles(index)).toEqual([
      'model-00001-of-00002.safetensors',
      'model-00002-of-00002.safetensors',
    ]);
  });

  it('raises where the tools raise: no weight map, and a value that is not a file name', () => {
    expect(() => shardFiles(toPython(parse('{"metadata": {}}')))).toThrowError(PyKeyError);
    expect(() => shardFiles(toPython(parse('{"weight_map": {"a": 3}}')))).toThrowError(PyTypeError);
  });

  it('merges several shards with the later file winning, as `out[name] = …` does', () => {
    const first: CheckpointHeaders = {
      a: { dtype: 'bf16', known: true, shape: [1n], file: '1' },
      b: { dtype: 'bf16', known: true, shape: [2n], file: '1' },
    };
    const second: CheckpointHeaders = {
      b: { dtype: 'f32', known: true, shape: [3n], file: '2' },
    };
    expect(mergeHeaders([first, second])).toEqual({
      a: { dtype: 'bf16', known: true, shape: [1n], file: '1' },
      b: { dtype: 'f32', known: true, shape: [3n], file: '2' },
    });
    expect(mergeHeaders([])).toEqual({});
  });
});
