import { CASES, type CaseReport, type EngineReport } from '../report.ts';
import { DirectoryWorkspace, hasDirectoryPicker } from './directory.ts';
import { forgetWorkspaces, grantPermission, permissionOf, recentWorkspaces, rememberWorkspace } from './recents.ts';
import { SnapshotWorkspace } from './snapshot.ts';
import { WorkspaceError, type WatchEvent } from './workspace.ts';

/**
 * The cases of feature 0.6, run in whatever engine loaded the page.
 *
 * Two ways in, as in feature 0.5's spike, because the engines on this box are not all drivable
 * the same way:
 *
 *   - `window.spike.run([…])`, which `apps/web/e2e/static.spec.ts` calls — that one can put a
 *     real folder on the upload input first, and catch the download the snapshot path produces;
 *   - `?cases=a,b,c&engine=<name>`, which makes the page run on load and POST its report to
 *     `report`, the only channel an unautomated browser has (`../run.ts`).
 *
 * The writable path is exercised **through an Origin Private File System directory handle**:
 * the same `FileSystemDirectoryHandle` interface a picked folder gives, minus the native picker
 * no automated browser can drive. What the picker itself does is measured by the capability
 * probe below and recorded in `../NOTE.md`.
 */

/** Where in the Origin Private File System the cases build their workspaces. */
const SCRATCH = 'spike-0.6';

/** A poll fast enough for a test to wait on, and slow enough to be a poll. */
const POLL_MS = 30;

/** A failed expectation inside a case. */
function expect(condition: boolean, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

/** Two values are the same when they print the same, the members in the same order. */
function same(actual: unknown, expected: unknown): boolean {
  return JSON.stringify(actual) === JSON.stringify(expected);
}

function pause(ms: number): Promise<void> {
  return new Promise((done) => setTimeout(done, ms));
}

/** Run something the workspace must refuse, and answer the refusal it gave. */
async function refused(what: () => Promise<unknown>): Promise<WorkspaceError> {
  try {
    await what();
  } catch (error) {
    if (error instanceof WorkspaceError) return error;
    throw error;
  }
  throw new Error('nothing was refused');
}

/** The Origin Private File System, or nothing where the engine has none (WebKit today). */
async function opfs(): Promise<FileSystemDirectoryHandle | null> {
  const storage: StorageManager | undefined = navigator.storage;
  if (typeof storage?.getDirectory !== 'function') return null;
  return storage.getDirectory();
}

/** A fresh scratch directory: the same handle interface a picked folder gives. */
async function scratch(name: string): Promise<FileSystemDirectoryHandle | null> {
  const root = await opfs();
  if (root === null) return null;
  const parent = await root.getDirectoryHandle(SCRATCH, { create: true });
  await parent.removeEntry(name, { recursive: true }).catch(() => undefined);
  return parent.getDirectoryHandle(name, { create: true });
}

/** Write a file straight through the handle API — a change made *behind* the workspace's back. */
async function writeBehind(root: FileSystemDirectoryHandle, path: string, text: string): Promise<void> {
  const segments = path.split('/');
  const name = segments.pop() ?? '';
  let directory = root;
  for (const segment of segments) directory = await directory.getDirectoryHandle(segment, { create: true });
  const writable = await (await directory.getFileHandle(name, { create: true })).createWritable();
  await writable.write(text);
  await writable.close();
}

/** The two documents the round trip writes: same shape, different lengths. */
const DOCUMENT_A = '{\n  "schema": "tensorspine/2.0",\n  "model": "spike-a"\n}\n';
const DOCUMENT_B = '{\n  "schema": "tensorspine/2.0",\n  "model": "spike-b-longer"\n}\n';

/** The page was served under a base path, and its own module came from under it (D11). */
function basePathCase(): CaseReport {
  const started = performance.now();
  const base = new URL('./', document.baseURI).href;
  const assets = new URL('assets/', base).href;
  expect(import.meta.url.startsWith(assets), `the module ${import.meta.url} does not live under ${assets}`);
  expect(document.baseURI.startsWith(window.location.origin), 'the page is not served from this origin');
  return {
    name: CASES.basePath,
    ok: true,
    ms: Math.round(performance.now() - started),
    detail: {
      base: new URL(base).pathname,
      baseURI: document.baseURI,
      module: import.meta.url.slice(window.location.origin.length),
      pathname: window.location.pathname,
      underRoot: new URL(base).pathname === '/',
    },
  };
}

/** A folder the browser can write: create, write, list, read, resolve, and refuse a stale write. */
async function directoryRoundTripCase(): Promise<CaseReport> {
  const started = performance.now();
  const handle = await scratch('round-trip');
  if (handle === null) return { name: CASES.directoryRoundTrip, ok: false, skipped: 'no navigator.storage.getDirectory', ms: 0 };
  const workspace = DirectoryWorkspace.open(handle, { pollMs: POLL_MS });

  expect(same(workspace.root(), { kind: 'directory', name: 'round-trip', writable: true }), 'the root is not the folder');

  await workspace.mkdir('models');
  await workspace.mkdir('primitive-library/primitives/norm.rms');
  const first = await workspace.write('models/spike.json', DOCUMENT_A);
  await workspace.write('primitive-library/primitive-library.json', '{"schema":"tensorspine-primitive-library-unit/2.0"}\n');

  const top = await workspace.list('');
  expect(
    same(
      top.map((entry) => `${entry.kind} ${entry.path}`),
      ['directory models', 'directory primitive-library'],
    ),
    `the root listed ${JSON.stringify(top)}`,
  );
  const models = await workspace.list('models');
  expect(same(models, [{ name: 'spike.json', path: 'models/spike.json', kind: 'file' }]), `models listed ${JSON.stringify(models)}`);

  const read = await workspace.read('models/spike.json');
  expect(read.text === DOCUMENT_A, 'the text read back is not the text written');
  expect(read.revision === first.revision, `the revision moved without a write: ${read.revision} is not ${first.revision}`);

  // Optimistic concurrency: the second write knows what it is replacing, the third does not.
  const second = await workspace.write('models/spike.json', DOCUMENT_B, first.revision);
  const conflict = await refused(() => workspace.write('models/spike.json', DOCUMENT_A, first.revision));
  expect(conflict.reason === 'conflict', `a stale write was refused as ${conflict.reason}`);
  expect((await workspace.read('models/spike.json')).text === DOCUMENT_B, 'the refused write changed the file');

  // A relative reference is resolved against the directory of the file it is written in.
  expect(
    workspace.resolve('models/spike.json', '../primitive-library/') === 'primitive-library',
    'a library base did not resolve beside the model',
  );

  const notFound = await refused(() => workspace.read('models/absent.json'));
  expect(notFound.reason === 'not-found', `a missing file was reported as ${notFound.reason}`);
  // `write` creates the file, never the directories above it: `mkdir` is what makes a folder.
  const noDirectory = await refused(() => workspace.write('nowhere/spike.json', DOCUMENT_A));
  expect(noDirectory.reason === 'not-found', `a write into a missing directory was ${noDirectory.reason}`);

  return {
    name: CASES.directoryRoundTrip,
    ok: true,
    ms: Math.round(performance.now() - started),
    detail: {
      files: [...(await workspace.revisions('')).keys()].sort((a, b) => a.localeCompare(b)),
      revisions: { first: first.revision, second: second.revision },
      conflict: conflict.message,
    },
  };
}

/** `watch` by polling: an external change arrives, and stops arriving when told to stop. */
async function directoryWatchCase(): Promise<CaseReport> {
  const started = performance.now();
  const handle = await scratch('watch');
  if (handle === null) return { name: CASES.directoryWatch, ok: false, skipped: 'no navigator.storage.getDirectory', ms: 0 };
  const workspace = DirectoryWorkspace.open(handle, { pollMs: POLL_MS });
  await workspace.mkdir('models');
  await workspace.write('models/spike.json', DOCUMENT_A);

  const seen: WatchEvent[] = [];
  const stop = workspace.watch('', (event) => seen.push(event));
  // The first poll is the baseline: it must report nothing at all.
  await pause(POLL_MS * 4);
  expect(seen.length === 0, `the first poll reported ${JSON.stringify(seen)}`);

  const waitFor = async (kind: WatchEvent['kind'], path: string): Promise<number> => {
    const from = performance.now();
    for (let attempt = 0; attempt < 200; attempt += 1) {
      if (seen.some((event) => event.kind === kind && event.path === path)) return Math.round(performance.now() - from);
      await pause(POLL_MS);
    }
    throw new Error(`no ${kind} of ${path} within ${String(Math.round(performance.now() - from))} ms`);
  };

  await writeBehind(handle, 'models/spike.json', DOCUMENT_B);
  const changed = await waitFor('changed', 'models/spike.json');
  await writeBehind(handle, 'models/other.json', DOCUMENT_A);
  const added = await waitFor('added', 'models/other.json');
  await (await handle.getDirectoryHandle('models')).removeEntry('other.json');
  const removed = await waitFor('removed', 'models/other.json');

  stop();
  const after = seen.length;
  await writeBehind(handle, 'models/spike.json', DOCUMENT_A + '\n');
  await pause(POLL_MS * 6);
  expect(seen.length === after, `the watch reported ${String(seen.length - after)} events after it was stopped`);

  // What one poll costs on this tree: every file's modification time, read through the handle.
  // A real workspace is larger, and `../NOTE.md` says what that leaves open for feature 2.6.
  const measured = performance.now();
  const files = (await workspace.revisions('')).size;
  const pollCost = Math.round((performance.now() - measured) * 100) / 100;

  return {
    name: CASES.directoryWatch,
    ok: true,
    ms: Math.round(performance.now() - started),
    detail: {
      pollMs: POLL_MS,
      msToChanged: changed,
      msToAdded: added,
      msToRemoved: removed,
      events: seen.length,
      pollCostMs: pollCost,
      pollFiles: files,
    },
  };
}

/**
 * A revision is a modification time and a size, which is what a poll can read cheaply. Can it
 * tell two writes of the same length apart? The answer is the engine's, and it decides what
 * `write`'s optimistic concurrency actually protects (see `../NOTE.md`).
 */
async function revisionResolutionCase(): Promise<CaseReport> {
  const started = performance.now();
  const handle = await scratch('revisions');
  if (handle === null) return { name: CASES.revisionResolution, ok: false, skipped: 'no navigator.storage.getDirectory', ms: 0 };
  const workspace = DirectoryWorkspace.open(handle, { pollMs: POLL_MS });
  const a = 'a'.repeat(64);
  const b = 'b'.repeat(64);

  const immediate = { first: (await workspace.write('same.json', a)).revision, second: '' };
  immediate.second = (await workspace.write('same.json', b)).revision;

  await pause(20);
  const spaced = { first: (await workspace.write('spaced.json', a)).revision, second: '' };
  await pause(20);
  spaced.second = (await workspace.write('spaced.json', b)).revision;

  return {
    name: CASES.revisionResolution,
    ok: true,
    ms: Math.round(performance.now() - started),
    detail: {
      immediate,
      spaced,
      distinguishableImmediately: immediate.first !== immediate.second,
      distinguishableAfter20ms: spaced.first !== spaced.second,
    },
  };
}

/** A directory handle in IndexedDB: stored, read back, and still the same folder. */
async function handlePersistenceCase(): Promise<CaseReport> {
  const started = performance.now();
  const root = await opfs();
  const handle = await scratch('persisted');
  if (root === null || handle === null) {
    return { name: CASES.handlePersistence, ok: false, skipped: 'no navigator.storage.getDirectory', ms: 0 };
  }
  await forgetWorkspaces();
  const workspace = DirectoryWorkspace.open(handle, { pollMs: POLL_MS });
  await workspace.mkdir('models');
  await workspace.write('models/spike.json', DOCUMENT_A);

  const remembered = await rememberWorkspace(handle);
  const known = await recentWorkspaces();
  expect(known.length === 1, `IndexedDB holds ${String(known.length)} workspaces, not one`);
  const back = known[0];
  expect(back !== undefined && back.id === remembered.id, 'the recent workspace came back with another id');
  expect(await back.handle.isSameEntry(handle), 'the handle read back is not the same folder');

  // Remembering the same folder twice is one entry, since `isSameEntry` is what identity means.
  const another = await (await root.getDirectoryHandle(SCRATCH)).getDirectoryHandle('persisted');
  await rememberWorkspace(another);
  expect((await recentWorkspaces()).length === 1, 'the same folder was remembered twice');

  const reopened = DirectoryWorkspace.open(back.handle, { pollMs: POLL_MS });
  expect((await reopened.read('models/spike.json')).text === DOCUMENT_A, 'the reopened handle does not read the folder');

  const permission = await permissionOf(back.handle);
  const granted = await grantPermission(back.handle);

  return {
    name: CASES.handlePersistence,
    ok: true,
    ms: Math.round(performance.now() - started),
    detail: { name: back.name, permission, granted: granted.granted, asked: granted.asked, state: granted.state },
  };
}

/**
 * The files a snapshot case reads: whatever Playwright put on the upload input, or a synthetic
 * folder built in the page so that an engine nobody automates still measures the path. A
 * synthetic `File` carries its relative path the way an upload does, by an own property that
 * shadows the prototype's accessor — which is a stand-in, and the detail says so.
 */
function snapshotFiles(): { files: File[]; source: 'input' | 'synthetic' } {
  const input = document.getElementById('upload');
  const chosen = input instanceof HTMLInputElement ? Array.from(input.files ?? []) : [];
  if (chosen.length > 0) return { files: chosen, source: 'input' };
  const make = (path: string, text: string): File => {
    const file = new File([text], path.split('/').pop() ?? path, { type: 'application/json' });
    Object.defineProperty(file, 'webkitRelativePath', { value: path });
    return file;
  };
  return {
    source: 'synthetic',
    files: [
      make('data/models/spike.json', DOCUMENT_A),
      make('data/primitive-library/primitive-library.json', '{"schema":"tensorspine-primitive-library-unit/2.0"}\n'),
    ],
  };
}

/** A folder upload, read as the read-only snapshot of D11. */
async function snapshotUploadCase(): Promise<CaseReport> {
  const started = performance.now();
  const { files, source } = snapshotFiles();
  const workspace = SnapshotWorkspace.fromFiles(files, { deliver: false });
  const reference = workspace.root();
  expect(reference.kind === 'snapshot' && !reference.writable, 'an uploaded folder is not a read-only snapshot');

  const paths = workspace.paths();
  const document0 = paths.find((path) => path.startsWith('models/'));
  expect(document0 !== undefined, `the snapshot stripped no folder name: ${JSON.stringify(paths)}`);
  const read = await workspace.read(document0);
  expect(read.text.length > 0, 'the snapshot read an empty file');

  const top = await workspace.list('');
  expect(
    same(
      top.map((entry) => `${entry.kind} ${entry.name}`),
      ['directory models', 'directory primitive-library'],
    ),
    `the snapshot listed ${JSON.stringify(top)}`,
  );

  const refusal = await refused(() => workspace.mkdir('primitive-library/primitives'));
  expect(refusal.reason === 'read-only', `a snapshot refused a new directory as ${refusal.reason}`);

  const written = await workspace.write(document0, 'edited\n');
  expect(written.revision === read.revision, 'saving a snapshot moved the revision, though nothing was written');

  return {
    name: CASES.snapshotUpload,
    ok: true,
    ms: Math.round(performance.now() - started),
    detail: {
      source,
      folder: reference.name,
      files: paths.length,
      paths: paths.slice(0, 8),
      bytes: read.text.length,
      resolved: workspace.resolve(document0, '../primitive-library/'),
      refusal: refusal.message,
    },
  };
}

/** Save, on a snapshot: the browser is handed the document's own bytes as a download. */
async function snapshotSaveCase(deliver: boolean): Promise<CaseReport> {
  const started = performance.now();
  const { files, source } = snapshotFiles();
  const workspace = SnapshotWorkspace.fromFiles(files, { deliver });
  const path = workspace.paths().find((one) => one.startsWith('models/')) ?? workspace.paths()[0];
  expect(path !== undefined, 'the snapshot holds no file to save');
  const text = (await workspace.read(path)).text;

  await workspace.write(path, text);
  const download = workspace.lastDownload;
  expect(download !== null, 'saving produced no download');
  expect(download.name === path.split('/').pop(), `the download is named ${download.name}`);

  // The bytes the browser was handed, read back from the object URL the page gave it.
  const offered = new Uint8Array(await (await fetch(download.url)).arrayBuffer());
  const expected = new TextEncoder().encode(text);
  expect(offered.length === expected.length, `the download is ${String(offered.length)} bytes, the file ${String(expected.length)}`);
  expect(offered.every((byte, index) => byte === expected[index]), 'the download is not the file');

  return {
    name: CASES.snapshotSave,
    ok: true,
    ms: Math.round(performance.now() - started),
    detail: { source, delivered: deliver, name: download.name, path: download.path, bytes: download.bytes },
  };
}

/** What the engine offers of the platform the two workspace paths stand on. */
export function capabilities(): Record<string, boolean> {
  const storage: StorageManager | undefined = navigator.storage;
  const handle = typeof FileSystemHandle === 'undefined' ? undefined : FileSystemHandle.prototype;
  return {
    // The writable path of D11, in one line: a folder the page may write in place.
    showDirectoryPicker: hasDirectoryPicker(),
    showOpenFilePicker: 'showOpenFilePicker' in window,
    showSaveFilePicker: 'showSaveFilePicker' in window,
    fileSystemHandles: typeof FileSystemDirectoryHandle !== 'undefined',
    fileSystemWritable: typeof FileSystemFileHandle !== 'undefined' && 'createWritable' in FileSystemFileHandle.prototype,
    directoryIteration: typeof FileSystemDirectoryHandle !== 'undefined' && 'values' in FileSystemDirectoryHandle.prototype,
    handlePermissions: handle !== undefined && 'queryPermission' in handle && 'requestPermission' in handle,
    opfs: typeof storage?.getDirectory === 'function',
    storagePersist: typeof storage?.persist === 'function',
    // The read-only path: a folder chosen or dropped, and a download to put it back.
    webkitdirectory: 'webkitdirectory' in HTMLInputElement.prototype,
    webkitGetAsEntry: typeof DataTransferItem !== 'undefined' && 'webkitGetAsEntry' in DataTransferItem.prototype,
    dropHandles: typeof DataTransferItem !== 'undefined' && 'getAsFileSystemHandle' in DataTransferItem.prototype,
    downloadAttribute: 'download' in HTMLAnchorElement.prototype,
    objectUrls: typeof URL.createObjectURL === 'function',
    indexedDB: typeof indexedDB !== 'undefined',
  };
}

/** The cases by name, each one answering a `CaseReport` or throwing. */
const RUNNERS: Record<string, (attended: boolean) => Promise<CaseReport> | CaseReport> = {
  [CASES.basePath]: () => basePathCase(),
  [CASES.directoryRoundTrip]: () => directoryRoundTripCase(),
  [CASES.directoryWatch]: () => directoryWatchCase(),
  [CASES.revisionResolution]: () => revisionResolutionCase(),
  [CASES.handlePersistence]: () => handlePersistenceCase(),
  [CASES.snapshotUpload]: () => snapshotUploadCase(),
  // Unattended, the bytes are built and checked but not handed to the browser: a runner that
  // opens three engines must not litter the machine with downloads nobody asked for.
  [CASES.snapshotSave]: (attended) => snapshotSaveCase(attended),
};

/** Run the named cases, one after another, and answer what each one did. */
export async function run(names: readonly string[], engine = 'unnamed', attended = true): Promise<EngineReport> {
  const cases: CaseReport[] = [];
  for (const name of names) {
    const runner = RUNNERS[name];
    if (runner === undefined) {
      cases.push({ name, ok: false, ms: 0, error: 'no such case' });
      continue;
    }
    const started = performance.now();
    try {
      cases.push(await runner(attended));
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
    base: new URL('./', document.baseURI).pathname,
    capabilities: capabilities(),
    cases,
  };
}
