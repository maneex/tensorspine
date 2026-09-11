import {
  ABSENT,
  PlatformError,
  listTree,
  readTree,
  type UploadedFile,
  type WatchEvent,
  type Workspace,
} from '@tensorspine/store/platform';

import {
  BrowserWorkspaces,
  DirectoryWorkspace,
  Recents,
  browserDrafts,
  browserSettings,
  browserShell,
  createBrowserPlatform,
  hasDirectoryPicker,
} from '../../src/platform/index.ts';

/**
 * What the browser layer drives — feature 2.4.
 *
 * The claims of §5.2 and D11 are claims about a *browser*: that a folder can be listed, read and
 * written in place, that a stale revision is refused, that a change made behind the workspace
 * arrives, that a folder upload is a snapshot whose Save downloads, that a handle survives
 * IndexedDB, that `localStorage` can refuse without taking the editor down. None of that can be
 * asked of Node, so this page is where it is asked, and `apps/web/e2e/platform.spec.ts` asks.
 *
 * The writable path runs against an **Origin Private File System** handle: OPFS answers the same
 * `FileSystemDirectoryHandle` interface a picked folder does, and the native picker is the one
 * thing no automated browser can drive (feature 0.6). Everything below the picker is the
 * application's own code, imported from `apps/web/src/platform/`.
 *
 * This page is built only for the browser layer (`vite build --mode check`, see
 * `apps/web/vite.config.ts`), so the deployed application does not carry it.
 */

/** What every probe answers: a verdict the suite reads, and how long it took. */
export interface Outcome {
  readonly ok: boolean;
  readonly detail: Record<string, unknown>;
  readonly ms: number;
}

async function timed(run: () => Promise<Record<string, unknown>>): Promise<Outcome> {
  const start = performance.now();
  try {
    return { ok: true, detail: await run(), ms: performance.now() - start };
  } catch (error) {
    return {
      ok: false,
      detail: { error: error instanceof Error ? `${error.name}: ${error.message}` : String(error) },
      ms: performance.now() - start,
    };
  }
}

/** The refusal a call answered with, as a caller reads it: its reason and its words. */
async function refusal(call: Promise<unknown>): Promise<string> {
  try {
    await call;
  } catch (error) {
    if (error instanceof PlatformError) return `${error.reason}: ${error.message}`;
    return `raised: ${String(error)}`;
  }
  return 'not refused';
}

/** A folder of the Origin Private File System, emptied first so a run does not inherit another. */
async function freshFolder(name: string): Promise<FileSystemDirectoryHandle> {
  const root = await navigator.storage.getDirectory();
  await root.removeEntry(name, { recursive: true }).catch(() => undefined);
  return root.getDirectoryHandle(name, { create: true });
}

/** The corpus document the cases work on, read from the material the build vendored. */
async function corpus(): Promise<string> {
  const response = await fetch(new URL('../../vendor/data/models/llama3-8b.json', document.baseURI));
  if (!response.ok) throw new Error(`the vendored corpus is not there (${String(response.status)})`);
  return response.text();
}

/** Where the vendored material is, from this page (which is two directories below the base). */
function vendorRoot(): string {
  return new URL('../../vendor/', document.baseURI).href;
}

/** A workspace laid out like `data/`: a model, and the base its `primitive_libraries` names. */
async function seeded(name: string, text: string): Promise<DirectoryWorkspace> {
  const workspace = DirectoryWorkspace.open(await freshFolder(name), { pollMs: 30 });
  await workspace.mkdir('models');
  await workspace.mkdir('primitive-library');
  await workspace.write('models/llama3-8b.json', text);
  await workspace.write('primitive-library/primitive-library.json', '{"schema": "tensorspine-primitive-library/2.0"}\n');
  return workspace;
}

/**
 * A file as a folder upload hands it over, made in the page.
 *
 * No cast: a browser `File` satisfies `UploadedFile` as it stands, which is the claim the
 * interface makes by writing the structural minimum instead of naming the DOM.
 */
function file(path: string, text: string): UploadedFile {
  return new File([text], path.split('/').at(-1) ?? path, { type: 'application/json' });
}

/** The same, with the path it had inside the chosen folder — what `webkitdirectory` fills in. */
function inFolder(path: string, text: string): UploadedFile {
  const one = file(path, text);
  Object.defineProperty(one, 'webkitRelativePath', { value: path, configurable: true });
  return one;
}

/** What the last drop opened, for the case that dispatches one. */
let dropped: { workspace: Workspace | null; paths: string[]; error?: string } | null = null;

const probes: Record<string, (argument?: string) => Promise<Outcome>> = {
  /** What this engine has, of the capabilities feature 0.6 measured in three of them. */
  capabilities: () =>
    timed(() =>
      Promise.resolve({
        showDirectoryPicker: hasDirectoryPicker(),
        opfs: typeof navigator.storage?.getDirectory === 'function',
        createWritable: typeof FileSystemFileHandle !== 'undefined' && 'createWritable' in FileSystemFileHandle.prototype,
        indexedDB: typeof indexedDB !== 'undefined',
        localStorage: (() => {
          try {
            return typeof window.localStorage === 'object';
          } catch {
            return false;
          }
        })(),
      }),
    ),

  /**
   * A folder the browser can write: listed, read, written in place, refused on a stale revision,
   * and the relative resolution `primitive_libraries[].base` needs.
   */
  directory: () =>
    timed(async () => {
      const text = await corpus();
      const workspace = await seeded('platform-directory', text);

      const listed = await listTree(workspace, '');
      const entries = await workspace.list('');
      const read = await workspace.read('models/llama3-8b.json');

      // Written in place, with the revision moving, and read back through a fresh handle.
      const edited = text.replace('"model": "llama3-8b"', '"model": "llama3-8b-edited"');
      const written = await workspace.write('models/llama3-8b.json', edited, read.revision);
      const onDisk = await (
        await (
          await (await navigator.storage.getDirectory()).getDirectoryHandle('platform-directory')
        ).getDirectoryHandle('models')
      )
        .getFileHandle('llama3-8b.json')
        .then((handle) => handle.getFile())
        .then((one) => one.text());

      // The optimistic concurrency of §5.2, and the file left exactly as it was.
      const conflict = await refusal(workspace.write('models/llama3-8b.json', 'lost', read.revision));
      const afterConflict = await workspace.read('models/llama3-8b.json');

      // A write must not make the directories above the file: `mkdir` is what makes a folder.
      const typo = await refusal(workspace.write('modles/llama3-8b.json', '{}'));
      await workspace.mkdir('bases/mine/primitives');
      await workspace.write('bases/mine/primitives/norm.rms.json', '{}', ABSENT);
      const twice = await refusal(workspace.write('bases/mine/primitives/norm.rms.json', '{}', ABSENT));

      return {
        root: workspace.root(),
        files: listed,
        entries: entries.map((entry) => `${entry.kind} ${entry.path}`),
        roundTrips: read.text === text,
        revisionMoved: written.revision !== read.revision,
        writtenInPlace: onDisk === edited,
        conflict,
        unchangedAfterConflict: afterConflict.text === edited,
        typo,
        twice,
        made: await listTree(workspace, 'bases'),
        resolved: workspace.resolve('models/llama3-8b.json', '../primitive-library/'),
        resolvedTemplates: workspace.resolve('primitive-library/primitive-library.json', '../models/'),
        notFound: await refusal(workspace.read('models/nothing.json')),
      };
    }),

  /** A change made behind the workspace arrives within a poll, and stops at the unsubscribe. */
  watch: () =>
    timed(async () => {
      const workspace = await seeded('platform-watch', '{"model": "llama3-8b"}\n');
      const seen: WatchEvent[] = [];
      const at = performance.now();
      const times: Record<string, number> = {};
      const stop = workspace.watch('', (event) => {
        seen.push(event);
        times[event.kind] ??= performance.now() - at;
      });

      // Behind the workspace's back: a second handle on the same folder, which is what another
      // editor or a `git checkout` is.
      const behind = DirectoryWorkspace.open(
        await (await navigator.storage.getDirectory()).getDirectoryHandle('platform-watch'),
      );
      const waitFor = async (count: number): Promise<void> => {
        const deadline = performance.now() + 5_000;
        while (seen.length < count && performance.now() < deadline) {
          await new Promise((resolve) => setTimeout(resolve, 10));
        }
      };
      await behind.write('models/llama3-8b.json', '{"model": "changed"}\n');
      await waitFor(1);
      await behind.write('models/added.json', '{}');
      await waitFor(2);
      await (await (await (await navigator.storage.getDirectory()).getDirectoryHandle('platform-watch')).getDirectoryHandle('models')).removeEntry(
        'added.json',
      );
      await waitFor(3);
      stop();

      const after = seen.length;
      await behind.write('models/llama3-8b.json', '{"model": "after the unsubscribe"}\n');
      await new Promise((resolve) => setTimeout(resolve, 200));

      return {
        events: seen.map((event) => `${event.kind} ${event.path}`),
        times,
        quietAfterUnsubscribe: seen.length === after,
      };
    }),

  /** A folder upload is a read-only snapshot, and Save hands the document to the user. */
  snapshot: () =>
    timed(async () => {
      const text = await corpus();
      const shell = browserShell({ deliver: false });
      const workspaces = await BrowserWorkspaces.create({
        deliver: (name, one) => shell.download(name, one),
        vendor: vendorRoot(),
      });
      const workspace = await workspaces.openUpload([
        inFolder('workspace/models/llama3-8b.json', text),
        inFolder('workspace/primitive-library/primitive-library.json', '{}\n'),
      ]);
      const read = await workspace.read('models/llama3-8b.json');
      const written = await workspace.write('models/llama3-8b.json', text, read.revision);
      return {
        root: workspace.root(),
        files: await listTree(workspace, ''),
        roundTrips: read.text === text,
        revisionUnmoved: written.revision === read.revision,
        download: shell.lastDownload,
        // The snapshot still holds what it was handed: nothing was written anywhere.
        untouched: (await workspace.read('models/llama3-8b.json')).text === text,
        resolved: workspace.resolve('models/llama3-8b.json', '../primitive-library/'),
        mkdir: await refusal(workspace.mkdir('bases/mine')),
        current: workspaces.current() === workspace,
      };
    }),

  /** The same, from a real `<input type="file" webkitdirectory>` the suite fills. */
  upload: () =>
    timed(async () => {
      const input = document.querySelector('#upload');
      if (!(input instanceof HTMLInputElement)) throw new Error('the page has no #upload');
      // No cast: a browser `File` satisfies `UploadedFile`, which is why the interface can stay
      // free of the DOM (`packages/store` is compiled without it).
      const files = Array.from(input.files ?? []);
      const shell = browserShell({ deliver: false });
      const workspaces = await BrowserWorkspaces.create({
        deliver: (name, one) => shell.download(name, one),
        vendor: vendorRoot(),
      });
      const workspace = await workspaces.openUpload(files);
      return {
        source: files.length === 0 ? 'nothing was chosen' : 'input',
        root: workspace.root(),
        files: await listTree(workspace, ''),
        bytes: (await workspace.read('models/llama3-8b.json')).text.length,
        resolved: workspace.resolve('models/llama3-8b.json', '../primitive-library/'),
      };
    }),

  /**
   * The same upload, with the download actually handed to the browser.
   *
   * Every other case builds the download without delivering it, so an unattended run does not
   * litter the machine with files; that the browser really takes it is asked once.
   */
  download: () =>
    timed(async () => {
      const input = document.querySelector('#upload');
      if (!(input instanceof HTMLInputElement)) throw new Error('the page has no #upload');
      const platform = await createBrowserPlatform({ vendor: vendorRoot() });
      const workspace = await platform.workspaces.openUpload(Array.from(input.files ?? []));
      const read = await workspace.read('models/llama3-8b.json');
      await workspace.write('models/llama3-8b.json', read.text, read.revision);
      return { download: platform.shell.lastDownload };
    }),

  /** A drop of a single file: what the transfer holds must be read before the first `await`. */
  drop: () =>
    timed(() =>
      Promise.resolve({
        opened: dropped?.workspace?.root() ?? null,
        files: dropped?.paths ?? [],
        refusal: dropped?.error ?? null,
      }),
    ),

  /** The Examples workspace over the material the build vendored (D11, §4.3). */
  examples: () =>
    timed(async () => {
      const shell = browserShell({ deliver: false });
      const workspaces = await BrowserWorkspaces.create({
        deliver: (name, one) => shell.download(name, one),
        vendor: vendorRoot(),
      });
      const workspace = await workspaces.openExamples();
      const read = await workspace.read('models/llama3-8b.json');
      const material = await workspaces.material();
      const base = await readTree(workspace, 'primitive-library', { suffix: '.json' });
      await workspace.write('models/llama3-8b.json', read.text, read.revision);
      return {
        root: workspace.root(),
        models: (await workspace.list('models')).map((entry) => entry.name),
        bytes: read.text.length,
        revision: read.revision,
        // A digest, not a timestamp: vendored material cannot change while the page is open.
        stableRevision: (await workspace.read('models/llama3-8b.json')).revision === read.revision,
        resolved: workspace.resolve('models/llama3-8b.json', '../primitive-library/'),
        baseFiles: Object.keys(base).length,
        manifest: material.manifest.examples,
        schemas: material.manifest.schemas.length,
        download: shell.lastDownload,
      };
    }),

  /**
   * The reference base read out of the Examples workspace, both ways.
   *
   * What feature 2.6 will hand `loadLibrary` is a map of path to text, and there are two ways to
   * build one over vendored material: the workspace's own walk ({@link readTree}, which lists a
   * directory and reads what is in it, and works on every workspace) and the vendor's bulk read
   * (which knows the whole set from the manifest before it asks for anything). They must answer
   * the same map — the prefix arithmetic of one against the listing of the other — and they do
   * not cost the same, which is what the figures are for.
   */
  base: () =>
    timed(async () => {
      const shell = browserShell({ deliver: false });
      const workspaces = await BrowserWorkspaces.create({
        deliver: (name, one) => shell.download(name, one),
        vendor: vendorRoot(),
      });
      const workspace = await workspaces.openExamples();
      const material = await workspaces.material();

      const walkedAt = performance.now();
      const walked = await readTree(workspace, 'primitive-library', { suffix: '.json' });
      const walkedMs = performance.now() - walkedAt;

      const bulkAt = performance.now();
      const bulk = await material.readAll('data/primitive-library');
      const bulkMs = performance.now() - bulkAt;

      const root = `${material.manifest.examples.root}/`;
      const sameKeys =
        Object.keys(bulk)
          .map((path) => path.slice(root.length))
          .sort((a, b) => a.localeCompare(b))
          .join('\n') === Object.keys(walked).sort((a, b) => a.localeCompare(b)).join('\n');
      const sameTexts = Object.entries(walked).every(([path, text]) => bulk[`${root}${path}`] === text);

      return { files: Object.keys(walked).length, sameKeys, sameTexts, walkedMs, bulkMs };
    }),

  /** Settings in `localStorage`: loaded once, written through, and read back after a reload. */
  settings: (argument) =>
    timed(() => {
      const settings = browserSettings();
      const seen: string[] = [];
      const stop = settings.onChange((key) => seen.push(key));
      if (argument === undefined) {
        settings.set('theme', 'dark');
        settings.set('panel.size', 320);
        settings.set('unlocked.bases', ['data/primitive-library']);
        settings.set('theme', 'light');
        settings.remove('panel.size');
      }
      stop();
      let raw: string | null = null;
      try {
        raw = window.localStorage.getItem('tensorspine.editor.theme');
      } catch {
        // A browser that blocks site data throws on the property itself, which is the case the
        // suite drives: the store above must have kept working all the same.
      }
      return Promise.resolve({
        persistent: settings.persistent,
        theme: settings.get('theme', 'unset'),
        bases: settings.get('unlocked.bases', [] as readonly string[]),
        keys: settings.keys(),
        changed: seen,
        raw,
      });
    }),

  /** Drafts in IndexedDB: put, get, list, and what survives a reload. */
  drafts: (argument) =>
    timed(async () => {
      const drafts = await browserDrafts();
      if (argument === undefined) {
        await drafts.put({
          workspace: 'w1',
          path: 'models/llama3-8b.json',
          text: '{"model": "llama3-8b-edited"}\n',
          revision: 'r1',
          savedAt: '2026-09-11T10:00:00.000Z',
        });
        await drafts.put({
          workspace: 'w2',
          path: 'models/other.json',
          text: '{}',
          revision: ABSENT,
          savedAt: '2026-09-11T10:01:00.000Z',
        });
      }
      const found = await drafts.get('w1', 'models/llama3-8b.json');
      const listed = await drafts.list();
      const mine = await drafts.list('w1');
      if (argument === 'clear') await drafts.clear();
      return {
        persistent: drafts.persistent,
        text: found?.text ?? null,
        revision: found?.revision ?? null,
        all: listed.map((one) => `${one.workspace} ${one.path} ${String(one.bytes)}`),
        mine: mine.length,
        cleared: argument === 'clear' ? (await drafts.list()).length : null,
      };
    }),

  /**
   * The Open Folder… command, end to end: the folder is opened, remembered, and comes back.
   *
   * The picker itself is the one thing no automated browser can drive, so the suite replaces that
   * call — and nothing else — with a handle to a folder of the Origin Private File System. What
   * runs below it is the application's own path: adopt, remember by `isSameEntry`, reopen by id
   * with the permission asked at most once, and the last folder reopened without a gesture where
   * the grant is still held (D11).
   */
  openFolder: () =>
    timed(async () => {
      const shell = browserShell({ deliver: false });
      const workspaces = await BrowserWorkspaces.create({
        deliver: (name, one) => shell.download(name, one),
        vendor: vendorRoot(),
      });
      await workspaces.forget();
      const seen: string[] = [];
      const stop = workspaces.onChange((workspace) => seen.push(workspace.root().kind));

      const opened = await workspaces.open();
      if (opened === null) throw new Error('the picker gave nothing');
      await opened.mkdir('models');
      await opened.write('models/llama3-8b.json', await corpus());

      const recent = await workspaces.recent();
      const first = recent[0];
      const again = first === undefined ? null : await workspaces.reopen(first.id);
      const granted = await workspaces.reopenGranted();
      stop();

      const answer = {
        opened: opened.root(),
        files: await listTree(opened, ''),
        recent: recent.map((one) => one.name),
        reopened: again?.root() ?? null,
        // The same folder, under the same identity — which is what a draft is keyed by.
        sameIdentity: again !== null && again.root().id === opened.root().id,
        grantedName: granted?.root().name ?? null,
        changes: seen,
        // And the workspace really is the folder: what was written through the first handle is
        // there through the one that came back out of IndexedDB.
        readBack: again === null ? null : (await again.read('models/llama3-8b.json')).text.length,
        remembers: workspaces.remembers,
      };
      await workspaces.forget();
      return { ...answer, forgotten: (await workspaces.recent()).length };
    }),

  /** A directory handle survives IndexedDB and comes back as the same folder (D11). */
  recents: () =>
    timed(async () => {
      const recents = await Recents.open();
      await recents.forget();
      const handle = await freshFolder('platform-recents');
      const remembered = await recents.remember(handle);
      const again = await recents.remember(await (await navigator.storage.getDirectory()).getDirectoryHandle('platform-recents'));
      const listed = await recents.all();
      const read = await recents.get(remembered.id);
      return {
        persistent: recents.persistent,
        // One entry, because `isSameEntry` is what "the same folder" means.
        sameId: remembered.id === again.id,
        entries: listed.length,
        name: read?.name ?? null,
        sameEntry: read === null ? false : await read.handle.isSameEntry(handle),
        permission: await recents.permission(handle),
      };
    }),

  /** The whole platform, as the application builds it. */
  platform: () =>
    timed(async () => {
      const platform = await createBrowserPlatform({ vendor: vendorRoot(), deliver: false });
      const empty = platform.workspace;
      await platform.workspaces.openExamples();
      const signIn = await refusal(platform.auth.signIn());
      await platform.shell.download('note.txt', 'hello');
      return {
        describes: platform.describe(),
        empty: empty.root(),
        // `platform.workspace` is the open one, not the one a component happened to be holding.
        workspace: platform.workspace.root(),
        writablePicker: platform.workspaces.writablePicker,
        checkpoints: platform.checkpoints.length,
        session: platform.auth.current(),
        signIn,
        settings: platform.settings.persistent,
        drafts: platform.drafts.persistent,
        remembers: platform.workspaces.remembers,
        download: platform.shell.lastDownload,
        recent: (await platform.workspaces.recent()).length,
        // A Save with nothing open is a download, which is the right answer for a New Model.
        savedWithNothingOpen: (await empty.write('untitled.json', '{}')).revision,
      };
    }),
};

declare global {
  interface Window {
    platformProbe: unknown;
  }
}

/**
 * The platform the page holds, as the application holds one: built once, before anything can be
 * dropped on the window.
 *
 * That order is not a convenience. A `DataTransfer` is disabled the moment the synchronous part of
 * a `drop` handler returns, so `openDrop` has to be **called** inside the handler with nothing
 * awaited before it — the platform's own reading is careful (`workspaces.ts`), and it cannot save
 * a caller that reached it one `await` too late. Building the platform first is what makes the
 * call synchronous, and it is what `main.ts` does.
 */
let platform: Awaited<ReturnType<typeof createBrowserPlatform>> | null = null;

window.addEventListener('dragover', (event) => {
  event.preventDefault();
});
window.addEventListener('drop', (event) => {
  event.preventDefault();
  // Called here, synchronously, with the transfer still alive.
  const opening = platform?.workspaces.openDrop(event.dataTransfer) ?? Promise.resolve(null);
  void (async () => {
    try {
      const workspace = await opening;
      dropped = { workspace, paths: workspace === null ? [] : await listTree(workspace, '') };
      document.body.dataset['drop'] = workspace === null ? 'nothing' : workspace.root().kind;
    } catch (error) {
      // A drop that cannot be read is a refusal the chrome shows, never a gesture that vanishes.
      dropped = { workspace: null, paths: [], error: error instanceof Error ? `${error.name}: ${error.message}` : String(error) };
      document.body.dataset['drop'] = 'refused';
    }
  })();
});

window.platformProbe = probes;
void (async () => {
  platform = await createBrowserPlatform({ vendor: vendorRoot(), deliver: false });
  const out = document.querySelector('#out');
  if (out instanceof HTMLElement) {
    out.textContent = `ready — ${platform.describe()}`;
    out.dataset['state'] = 'ready';
  }
})();
