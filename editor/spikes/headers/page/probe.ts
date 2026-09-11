import { CASES, type CaseReport, type EngineReport } from '../report.ts';
import { counting, readHeader, entriesOf, DTYPES, type RawTensor } from './safetensors.ts';
import { synthesise, syntheticBlob } from './synthetic.ts';

/**
 * The spike's page — feature 0.5.
 *
 * It runs the cases of `report.ts` in whatever engine loaded it and answers an `EngineReport`.
 * Two ways in, because the engines available on this box are not all drivable the same way:
 *
 *   - `window.spike.run([...])`, which the Playwright specification calls
 *     (`apps/web/e2e/headers.spec.ts`) — that one can also put a real checkpoint shard on the
 *     file input first;
 *   - `?cases=a,b,c&engine=<name>` in the URL, which makes the page run on load and POST its
 *     report to `/report` — the only way to get an answer out of a browser nobody automates,
 *     which is how Firefox and WebKitGTK are measured here (`run.ts`).
 *
 * Everything it measures, it measures against `page/safetensors.ts`: two slices of a source,
 * whatever the source is. The Hub cases are the exception the feature asks for — there the
 * reading is `@huggingface/hub`'s, and what is measured is the traffic it causes.
 */

/** The Hub repositories the network cases read, pinned to a commit so a run is repeatable. */
const TARGETS = {
  /** One 548 MB `model.safetensors`, no index. */
  single: { repo: 'openai-community/gpt2', revision: '607a30d783dfa663caf39e06633721c8d4cfcd7e' },
  /** Four shards and an index — 16 GB, the checkpoint `llama3-8b` is transcribed from. */
  sharded: { repo: 'NousResearch/Meta-Llama-3-8B', revision: '315b20096dc791d381d514deb5f8bd9c8d6d3061' },
} as const;

/** The payload of the large synthetic file: big enough that reading it would show. */
const LARGE_BYTES = 256 << 20;

/** The payload of the file written to the Origin Private File System. */
const OPFS_BYTES = 64 << 20;

/** A failed expectation inside a case. */
function expect(condition: boolean, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

/** Two values are the same when they print the same, the members in the same order. */
function same(actual: unknown, expected: unknown): boolean {
  return JSON.stringify(actual) === JSON.stringify(expected);
}

/** The tensors of a header `@huggingface/hub` parsed, without its `__metadata__`. */
function tensorsOf(header: Record<string, unknown>): Record<string, RawTensor> {
  const tensors: Record<string, RawTensor> = {};
  for (const [name, value] of Object.entries(header)) {
    if (name !== '__metadata__') tensors[name] = value as RawTensor;
  }
  return tensors;
}

/** One request the Hub cases caused, as the page saw it. */
interface HubRequest {
  readonly url: string;
  readonly range: string | null;
  readonly status: number;
  readonly bytes: number;
}

/**
 * `fetch`, with every call recorded. The Hub client takes a custom `fetch`, so this is how the
 * page learns what a header read costs over the wire — and, when a browser refuses a
 * cross-origin request, which request it was.
 */
function countingFetch(log: HubRequest[]): typeof fetch {
  return async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = input instanceof Request ? input.url : input instanceof URL ? input.href : String(input);
    const headers = new Headers(init?.headers ?? (input instanceof Request ? input.headers : undefined));
    const range = headers.get('range');
    const response = await fetch(input, init);
    const length = Number(response.headers.get('content-length') ?? '');
    log.push({
      url: url.split('?')[0] ?? url,
      range,
      status: response.status,
      bytes: Number.isFinite(length) ? length : 0,
    });
    return response;
  };
}

/** The small synthetic file: every dtype `artifact.py` maps, one it does not, and a metadata. */
function smallFile(): ReturnType<typeof synthesise> {
  const names = Object.keys(DTYPES);
  const tensors = names.map((dtype, index) => ({
    name: `layers.${String(index)}.weight`,
    dtype,
    shape: index % 2 === 0 ? [4, 3] : [8],
  }));
  // `U16` is a safetensors dtype `artifact.py`'s table does not carry: it must come back as the
  // lower-cased name, which is what `read_headers` does with anything it does not know.
  tensors.push({ name: 'unknown.weight', dtype: 'U16', shape: [2, 2] });
  // The reference writer pads the header so the payload starts on an eight-byte boundary; the
  // padding is inside the declared length, so a reader has to accept it.
  return synthesise(tensors, { metadata: { format: 'pt', note: 'feature 0.5' }, padding: 5 });
}

/**
 * The small file, read from a `Blob` slice: the entries are exactly the header's, the metadata
 * is separated as the tools separate it, the two reads are the prefix and the header and
 * nothing else, and a `File` — what a directory picker or a drop hands the editor — reads the
 * same. Three malformed prefixes are refused rather than allocated.
 */
async function syntheticBlobCase(): Promise<CaseReport> {
  const started = performance.now();
  const file = smallFile();
  const blob = syntheticBlob(file);
  const { source, reads } = counting(blob);
  const read = await readHeader(source, 'model.safetensors');

  const expected = entriesOf(tensorsOf(file.header), 'model.safetensors');
  expect(same(read.entries, expected), 'the entries are not the header\'s tensors');
  expect(same(read.metadata, { format: 'pt', note: 'feature 0.5' }), '__metadata__ did not come back');
  expect(read.headerBytes === file.headerBytes, 'the header length is not the prefix');
  expect(read.bytesRead === 8 + file.headerBytes, 'more than the header was read');
  expect(
    same(reads(), [
      { start: 0, bytes: 8 },
      { start: 8, bytes: file.headerBytes },
    ]),
    'the reader asked for something other than the prefix and the header',
  );
  expect(read.entries['unknown.weight']?.dtype === 'u16', 'an unmapped dtype did not fall back');
  expect(read.entries['layers.0.weight']?.dtype === 'bf16', 'BF16 did not map to bf16');

  const asFile = new File([blob], 'shard-00001.safetensors');
  const fromFile = await readHeader(asFile, 'shard-00001.safetensors');
  expect(Object.keys(fromFile.entries).length === Object.keys(expected).length, 'a File read differently');

  const refusals: string[] = [];
  for (const [name, bytes] of [
    ['zero length', new Uint8Array(8)],
    ['a length past the end', (() => {
      const prefix = new Uint8Array(16);
      new DataView(prefix.buffer).setBigUint64(0, 4096n, true);
      return prefix;
    })()],
    ['four bytes', new Uint8Array(4)],
  ] as const) {
    try {
      await readHeader(new Blob([bytes]), 'broken.safetensors');
      throw new Error(`${name} was accepted`);
    } catch (error) {
      expect(error instanceof Error && error.name === 'HeaderError', `${name}: ${String(error)}`);
      refusals.push(error.message);
    }
  }

  return {
    name: CASES.syntheticBlob,
    ok: true,
    ms: Math.round(performance.now() - started),
    bytesRead: read.bytesRead,
    detail: {
      fileBytes: blob.size,
      headerBytes: read.headerBytes,
      tensors: Object.keys(read.entries).length,
      dtypes: Object.values(read.entries).map((entry) => entry.dtype),
      refusals,
    },
  };
}

/** The same reader on a 256 MiB file: the cost is the header's, not the file's. */
async function syntheticLargeCase(): Promise<CaseReport> {
  const file = synthesise([{ name: 'weight', dtype: 'BF16', shape: [LARGE_BYTES / 2] }]);
  const blob = syntheticBlob(file);
  const { source, reads } = counting(blob);
  const started = performance.now();
  const read = await readHeader(source, 'large.safetensors');
  const ms = performance.now() - started;
  // The same read again: it separates what the engine pays once for a freshly built blob (it
  // has to be stored before it can be sliced) from what it pays to read a header.
  const again = performance.now();
  await readHeader(blob, 'large.safetensors');
  const secondMs = performance.now() - again;
  expect(read.bytesRead === 8 + read.headerBytes, 'more than the header was read');
  expect(reads().length === 2, 'the header took more than two reads');
  return {
    name: CASES.syntheticLarge,
    ok: true,
    ms: Math.round(ms * 1000) / 1000,
    bytesRead: read.bytesRead,
    detail: {
      fileBytes: blob.size,
      headerBytes: read.headerBytes,
      tensors: Object.keys(read.entries).length,
      secondMs: Math.round(secondMs * 1000) / 1000,
    },
  };
}

/**
 * A file on disk, not in memory: the Origin Private File System is the one way a page can make
 * one in every engine without a picker. What it proves is that the same two slices work on a
 * `File` the browser reads lazily from storage.
 */
async function opfsFileCase(): Promise<CaseReport> {
  const started = performance.now();
  const storage: StorageManager | undefined = navigator.storage;
  if (typeof storage?.getDirectory !== 'function') {
    return { name: CASES.opfsFile, ok: false, skipped: 'no navigator.storage.getDirectory', ms: 0 };
  }
  const root = await storage.getDirectory();
  const name = 'spike-0.5.safetensors';
  // 64 MiB in the profile, removed however this ends: a failing `expect` used to leave the file
  // behind, and the next run's `create: true` would reopen it rather than write it afresh.
  try {
    const handle = await root.getFileHandle(name, { create: true });
    if (typeof handle.createWritable !== 'function') {
      return { name: CASES.opfsFile, ok: false, skipped: 'no FileSystemFileHandle.createWritable', ms: 0 };
    }
    const file = synthesise([{ name: 'weight', dtype: 'F32', shape: [OPFS_BYTES / 4] }]);
    const writable = await handle.createWritable();
    await writable.write(syntheticBlob(file));
    await writable.close();
    const written = performance.now();

    const onDisk = await handle.getFile();
    const { source, reads } = counting(onDisk);
    const read = await readHeader(source, name);
    const ms = performance.now() - written;
    expect(onDisk.size === file.prefix.byteLength + file.payloadBytes, 'the file on disk is not the file written');
    expect(read.bytesRead === 8 + read.headerBytes, 'more than the header was read');
    expect(reads().length === 2, 'the header took more than two reads');
    return {
      name: CASES.opfsFile,
      ok: true,
      ms: Math.round(ms * 1000) / 1000,
      bytesRead: read.bytesRead,
      detail: {
        fileBytes: onDisk.size,
        headerBytes: read.headerBytes,
        tensors: Object.keys(read.entries).length,
        writeMs: Math.round(written - started),
      },
    };
  } finally {
    // `NotFoundError` where the handle was never created: the removal is the point, not its
    // verdict, and a failure here must not replace the one being reported.
    await root.removeEntry(name).catch(() => undefined);
  }
}

/** The file on the page's input, if one was put there: a real checkpoint shard, on a real disk. */
async function localFileCase(): Promise<CaseReport> {
  const input = document.querySelector('#local');
  const chosen = input instanceof HTMLInputElement ? input.files?.[0] : undefined;
  if (chosen === undefined) {
    return { name: CASES.localFile, ok: false, skipped: 'no file on the input', ms: 0 };
  }
  const { source, reads } = counting(chosen);
  const started = performance.now();
  const read = await readHeader(source, chosen.name);
  const ms = performance.now() - started;
  expect(read.bytesRead === 8 + read.headerBytes, 'more than the header was read');
  expect(reads().length === 2, 'the header took more than two reads');
  const dtypes: Record<string, number> = {};
  for (const entry of Object.values(read.entries)) dtypes[entry.dtype] = (dtypes[entry.dtype] ?? 0) + 1;
  const first = Object.entries(read.entries)[0];
  return {
    name: CASES.localFile,
    ok: true,
    ms: Math.round(ms * 1000) / 1000,
    bytesRead: read.bytesRead,
    detail: {
      file: chosen.name,
      fileBytes: chosen.size,
      headerBytes: read.headerBytes,
      tensors: Object.keys(read.entries).length,
      dtypes,
      first: first === undefined ? null : { name: first[0], ...first[1] },
      metadata: read.metadata ?? null,
    },
  };
}

/**
 * The Hub, through `@huggingface/hub`. The import is dynamic so that the client — and the chunk
 * it drags with it — is loaded only by a page that asks the Hub for something, which is what
 * §5.1's `HubCheckpoint` will want (feature 4.1, and 0.4's lesson about the first bundle).
 *
 * `xet: false` is the second transport: the client sends the file's bytes through the Xet
 * content-addressed store when the Hub offers it, and through plain HTTP range requests on
 * `/resolve/` when it is told not to. `parseSafetensorsMetadata` forwards whatever it is given
 * to `downloadFile`, which reads that flag, but its own parameter type does not name it — hence
 * the cast, and the finding in the note. Both transports are measured because they are not the
 * same request to a browser: one is a range request to two origins, the other is the Xet
 * reconstruction protocol on top of a byte stream.
 */
async function hubCase(
  name: string,
  target: { repo: string; revision: string },
  options: { xet: boolean } = { xet: true },
): Promise<CaseReport> {
  const log: HubRequest[] = [];
  const started = performance.now();
  const { parseSafetensorsMetadata } = await import('@huggingface/hub');
  const loaded = performance.now();
  const parameters = {
    repo: target.repo,
    revision: target.revision,
    fetch: countingFetch(log),
    xet: options.xet,
  } as unknown as Parameters<typeof parseSafetensorsMetadata>[0];

  const detail = (extra: Record<string, unknown>): Record<string, unknown> => ({
    repo: target.repo,
    revision: target.revision,
    xet: options.xet,
    requests: log.map((request) => ({ url: request.url, range: request.range, status: request.status, bytes: request.bytes })),
    clientLoadMs: Math.round(loaded - started),
    ...extra,
  });

  let parsed: Awaited<ReturnType<typeof parseSafetensorsMetadata>>;
  try {
    parsed = await parseSafetensorsMetadata(parameters);
  } catch (error) {
    // The request log is the evidence: it says how far the client got before the engine or the
    // Hub refused, and which request was the last one.
    return {
      name,
      ok: false,
      ms: Math.round(performance.now() - loaded),
      bytesRead: log.reduce((total, request) => total + request.bytes, 0),
      error: error instanceof Error ? `${error.name}: ${error.message}` : String(error),
      detail: detail({}),
    };
  }
  const ms = performance.now() - loaded;

  const entries: Record<string, { dtype: string; shape: number[]; file: string }> = {};
  const files = parsed.sharded ? Object.keys(parsed.headers) : parsed.filepaths;
  if (parsed.sharded) {
    for (const [file, header] of Object.entries(parsed.headers)) {
      Object.assign(entries, entriesOf(tensorsOf(header), file));
    }
  } else {
    Object.assign(entries, entriesOf(tensorsOf(parsed.header), files[0] ?? 'model.safetensors'));
  }
  expect(Object.keys(entries).length > 0, 'the Hub answered no tensor');

  const sample = Object.entries(entries)[0];
  return {
    name,
    ok: true,
    ms: Math.round(ms),
    bytesRead: log.reduce((total, request) => total + request.bytes, 0),
    detail: detail({
      sharded: parsed.sharded,
      files: files.length,
      tensors: Object.keys(entries).length,
      sample: sample === undefined ? null : { name: sample[0], ...sample[1] },
    }),
  };
}

/** What the engine offers of the platform this spike, and feature 0.6, stand on. */
function capabilities(): Record<string, boolean> {
  let byteStreams: boolean;
  try {
    // `@huggingface/hub`'s Xet transport builds one of these; WebKit does not have them.
    new ReadableStream({ type: 'bytes' });
    byteStreams = true;
  } catch {
    byteStreams = false;
  }
  const storage: StorageManager | undefined = navigator.storage;
  return {
    blobSlice: typeof Blob.prototype.slice === 'function' && typeof Blob.prototype.arrayBuffer === 'function',
    readableByteStreams: byteStreams,
    opfs: typeof storage?.getDirectory === 'function',
    // Feature 0.6's writable workspace: a directory the user picks, written in place.
    showDirectoryPicker: 'showDirectoryPicker' in window,
    fileSystemWritable: typeof FileSystemFileHandle !== 'undefined' && 'createWritable' in FileSystemFileHandle.prototype,
    webkitdirectory: 'webkitdirectory' in HTMLInputElement.prototype,
  };
}

/** The cases by name, each one answering a `CaseReport` or throwing. */
const RUNNERS: Record<string, () => Promise<CaseReport>> = {
  [CASES.syntheticBlob]: syntheticBlobCase,
  [CASES.syntheticLarge]: syntheticLargeCase,
  [CASES.opfsFile]: opfsFileCase,
  [CASES.localFile]: localFileCase,
  [CASES.hubSingle]: () => hubCase(CASES.hubSingle, TARGETS.single),
  [CASES.hubSharded]: () => hubCase(CASES.hubSharded, TARGETS.sharded),
  [CASES.hubRange]: () => hubCase(CASES.hubRange, TARGETS.single, { xet: false }),
};

/** Run the named cases, one after another, and answer what each one did. */
async function run(names: readonly string[], engine = 'unnamed'): Promise<EngineReport> {
  const cases: CaseReport[] = [];
  for (const name of names) {
    const runner = RUNNERS[name];
    if (runner === undefined) {
      cases.push({ name, ok: false, ms: 0, error: 'no such case' });
      continue;
    }
    const started = performance.now();
    try {
      cases.push(await runner());
    } catch (error) {
      cases.push({
        name,
        ok: false,
        ms: Math.round(performance.now() - started),
        error: error instanceof Error ? `${error.name}: ${error.message}` : String(error),
      });
    }
  }
  return {
    engine,
    userAgent: navigator.userAgent,
    startedAt: new Date().toISOString(),
    capabilities: capabilities(),
    cases,
  };
}

declare global {
  interface Window {
    spike: {
      readonly dtypes: Readonly<Record<string, string>>;
      readonly targets: typeof TARGETS;
      readonly run: (names: readonly string[], engine?: string) => Promise<EngineReport>;
    };
  }
}

window.spike = { dtypes: DTYPES, targets: TARGETS, run };

/** What the page shows: the report, and a line per case a screenshot can be read from. */
function render(report: EngineReport): void {
  const summary = report.cases
    .map((one) => {
      const verdict = one.skipped !== undefined ? `skipped (${one.skipped})` : one.ok ? 'ok' : `FAILED — ${one.error ?? ''}`;
      const cost = one.bytesRead === undefined ? '' : ` · ${String(one.bytesRead)} bytes · ${String(one.ms)} ms`;
      return `${one.name}: ${verdict}${cost}`;
    })
    .join('\n');
  const out = document.querySelector('#out');
  if (out instanceof HTMLElement) out.textContent = `${report.engine} — ${report.userAgent}\n\n${summary}\n\n${JSON.stringify(report, null, 2)}`;
  document.body.dataset['state'] = report.cases.every((one) => one.ok || one.skipped !== undefined) ? 'ok' : 'failed';
}

// The URL-driven path: run on load, show the result, and post it back to the server that served
// the page, which is how an engine nobody automates reports its answer.
const parameters = new URLSearchParams(window.location.search);
const asked = parameters.get('cases');
if (asked !== null) {
  const engine = parameters.get('engine') ?? 'unnamed';
  void run(asked.split(','), engine).then(async (report) => {
    render(report);
    await fetch('report', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(report) }).catch(
      () => undefined,
    );
  });
} else {
  document.body.dataset['state'] = 'ready';
}
