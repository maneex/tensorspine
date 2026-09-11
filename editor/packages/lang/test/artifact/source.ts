import { existsSync, readFileSync, readdirSync, openSync, readSync, closeSync } from 'node:fs';
import { join } from 'node:path';

import {
  derive,
  headerLength,
  LENGTH_PREFIX_BYTES,
  mergeHeaders,
  parse,
  readHeader,
  shardFiles,
  toPython,
  type CheckpointHeaders,
  type HeaderEntry,
  type PyRecord,
  type PyValue,
} from '../../src/index.js';
import { corpus, library, schemas } from '../describe/source.js';

// What the checkpoint suites read: a corpus document derived by the core, the synthetic headers
// `tests/run_artifact.py` builds from a D3, and — when the weights are on this machine — the real
// `Meta-Llama-3-8B` checkpoint read the way a `CheckpointSource` will read it (§5.2): the index,
// the shards it names, the eight-byte prefix of each and exactly the header it announces.

const derived = new Map<string, PyRecord>();

/** One corpus document, derived by the core; the answer is kept, several suites reading it. */
export function derivedCorpus(name: string): PyRecord {
  const held = derived.get(name);
  if (held !== undefined) return held;
  const answer = derive(parse(corpus(name)), { schemas, library });
  derived.set(name, answer);
  return answer;
}

/** The D3 of one corpus document. */
export function d3Of(name: string): PyValue {
  return derivedCorpus(name)['d3'] as PyValue;
}

/** A derived document carrying a D3 built by hand, which is what `artifact.check` is handed. */
export function documentOf(d3: PyValue): PyRecord {
  return { d3 };
}

/** One header entry, as a checkpoint whose dtypes the reader maps would answer. */
export function entry(dtype: string, shape: readonly bigint[], file = 'x'): HeaderEntry {
  return { dtype, known: true, shape: [...shape], file };
}

/**
 * `tests/run_artifact.py`'s `headers_of`: a checkpoint that holds exactly what a D3 says.
 *
 * It reads `t['location']['tensor']`, so it serves the documents whose every location is a whole
 * tensor — which is llama3-8b, shieldstral-3b and the composite, the three the script uses.
 */
export function headersOf(d3: PyValue): CheckpointHeaders {
  const out: Record<string, HeaderEntry> = {};
  for (const tensor of (d3 as PyRecord)['tensors'] as readonly PyRecord[]) {
    const name = (tensor['location'] as PyRecord)['tensor'] as string;
    const shape = (tensor['shape'] as readonly PyRecord[]).map((axis) => axis['extent'] as PyValue);
    out[name] = { dtype: tensor['dtype'] as string, known: true, shape, file: 'x' };
  }
  return out;
}

/** The same map with one entry's members changed, the way the script's `copy.deepcopy` does. */
export function withEntry(
  headers: CheckpointHeaders,
  name: string,
  changes: Partial<HeaderEntry>,
): CheckpointHeaders {
  const held = headers[name];
  if (held === undefined) throw new Error(`no header named ${name}`);
  return { ...headers, [name]: { ...held, ...changes } };
}

/** The same map without one entry: `h2 = dict(h); del h2[name]`. */
export function withoutEntry(headers: CheckpointHeaders, name: string): CheckpointHeaders {
  const rest: Record<string, HeaderEntry> = { ...headers };
  delete rest[name];
  return rest;
}

/**
 * `weights/` under `TENSORSPINE_MODEL_ARTIFACTS`, which is where `tests/run_artifact.py` looks.
 *
 * "Unset, the checks that need a checkpoint say so instead of looking in somebody's home
 * directory" — the script's own comment, and the rule this suite keeps.
 */
export function checkpointDirectory(name: string): string | null {
  const artifacts = process.env['TENSORSPINE_MODEL_ARTIFACTS'];
  if (artifacts === undefined || artifacts === '') return null;
  const directory = join(artifacts, 'weights', name);
  return existsSync(directory) ? directory : null;
}

/**
 * A checkpoint on disk, read as `read_headers` reads one and as a `CheckpointSource` will.
 *
 * The I/O is the test's — `packages/lang` opens no file — and the format is the core's: the
 * shard list comes from {@link shardFiles} when there is an index, the length from
 * {@link headerLength}, the entries from {@link readHeader}, the merge from {@link mergeHeaders}.
 * Exactly `8 + n` bytes are read per shard, which is what makes this affordable on 16 GB.
 */
export function readCheckpoint(directory: string): { headers: CheckpointHeaders; bytes: number } {
  const index = join(directory, 'model.safetensors.index.json');
  const files = existsSync(index)
    ? shardFiles(toPython(parse(readFileSync(index, 'utf8'))))
    : readdirSync(directory)
        .filter((name) => name.endsWith('.safetensors'))
        .sort();
  if (files.length === 0) throw new Error(`no safetensors file under ${directory}`);
  let bytes = 0;
  const reads = files.map((file) => {
    const handle = openSync(join(directory, file), 'r');
    try {
      const prefix = new Uint8Array(LENGTH_PREFIX_BYTES);
      readSync(handle, prefix, 0, LENGTH_PREFIX_BYTES, 0);
      const length = headerLength(prefix, file);
      const all = new Uint8Array(LENGTH_PREFIX_BYTES + length);
      all.set(prefix, 0);
      readSync(handle, all, LENGTH_PREFIX_BYTES, length, LENGTH_PREFIX_BYTES);
      bytes += all.byteLength;
      return readHeader(all, file).entries;
    } finally {
      closeSync(handle);
    }
  });
  return { headers: mergeHeaders(reads), bytes };
}
