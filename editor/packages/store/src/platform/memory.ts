/**
 * The stub `Platform` — feature 2.4.
 *
 * D11: "CI builds the web app against a stub platform from phase 2 on, so a platform leak is a
 * build failure." This is that platform: a workspace in memory, settings and drafts in memory,
 * `NoAuth`, a shell that records what it was asked to do and shows nothing. It names no browser
 * API, no Node module and no Electron one — this package is compiled without the DOM, so it
 * *cannot* — which is what makes the build it is linked into a proof: an interface that reached
 * around {@link Platform} would not compile there.
 *
 * It is also what the suites use. A memory workspace is a real implementation of the interface,
 * refusal for refusal: a write into a directory that does not exist is refused here as it is on a
 * folder the browser opened (feature 0.6's finding 5, "`write` must not create the directories
 * above the file"), and a stale revision is a conflict here as it is there. A suite that passes
 * against this one is asking the same questions the browser layer asks.
 */
import { byNewest, draftKey, summaryOf } from './autosave.js';
import { byteLength, ReadOnlyWorkspace, snapshotOf, textsOf, type Delivered } from './readonly.js';
import { byName, join, normalise, parentOf, resolveFrom, segmentsOf, type WorkspacePath } from './paths.js';
import {
  ABSENT,
  PlatformError,
  type Draft,
  type DraftStore,
  type Entry,
  type ColourScheme,
  type MenuCommand,
  type Platform,
  type RecentWorkspace,
  type Session,
  type SettingValue,
  type SettingsStore,
  type Shell,
  type Unsubscribe,
  type UploadedFile,
  type WatchEvent,
  type Workspace,
  type WorkspaceRef,
  type Workspaces,
} from './types.js';
import { memoryAuth, noAuth } from './auth.js';

/** A workspace whose files are in memory, written and read back exactly as a folder's are. */
export class MemoryWorkspace implements Workspace {
  /** A workspace holding the given texts, and the directories they imply. */
  static of(texts: Readonly<Record<string, string>> = {}, name = 'memory'): MemoryWorkspace {
    const workspace = new MemoryWorkspace(name);
    for (const [path, text] of Object.entries(texts)) {
      const at = normalise(path);
      for (const directory of ancestorsOf(at)) workspace.directories.add(directory);
      workspace.files.set(at, { text, revision: workspace.next() });
    }
    return workspace;
  }

  private readonly files = new Map<WorkspacePath, { text: string; revision: string }>();
  private readonly directories = new Set<WorkspacePath>(['']);
  private readonly watchers = new Set<{ at: WorkspacePath; callback: (event: WatchEvent) => void }>();
  private clock = 0;

  constructor(private readonly name: string) {}

  root(): WorkspaceRef {
    return { kind: 'memory', id: `memory:${this.name}`, name: this.name, writable: true };
  }

  list(dir: WorkspacePath): Promise<Entry[]> {
    const prefix = normalise(dir);
    if (!this.directories.has(prefix)) {
      return Promise.reject(new PlatformError(`no directory ${prefix} in the workspace`, 'not-found'));
    }
    const seen = new Map<string, Entry>();
    for (const path of this.files.keys()) {
      const head = headUnder(path, prefix);
      if (head === null) continue;
      seen.set(head.name, { name: head.name, path: join(prefix, head.name), kind: head.kind });
    }
    for (const path of this.directories) {
      const head = headUnder(path, prefix);
      if (head === null) continue;
      seen.set(head.name, { name: head.name, path: join(prefix, head.name), kind: 'directory' });
    }
    return Promise.resolve([...seen.values()].sort(byName));
  }

  read(path: WorkspacePath): Promise<{ text: string; revision: string }> {
    const found = this.files.get(normalise(path));
    if (found === undefined) {
      return Promise.reject(new PlatformError(`no file ${normalise(path)} in the workspace`, 'not-found'));
    }
    return Promise.resolve({ ...found });
  }

  write(path: WorkspacePath, text: string, expect?: string): Promise<{ revision: string }> {
    const at = normalise(path);
    if (at === '') return Promise.reject(new PlatformError('the workspace root is not a file', 'bad-path'));
    const current = this.files.get(at);
    const found = current?.revision ?? ABSENT;
    if (expect !== undefined && expect !== found) {
      return Promise.reject(
        new PlatformError(`${at} changed since it was read (${found} is not ${expect})`, 'conflict', {
          found,
          expected: expect,
        }),
      );
    }
    // The directories above the file are never created here: `mkdir` is the operation that makes
    // a folder, so a path with a typo in it is a refusal rather than a new tree (feature 0.6).
    if (!this.directories.has(parentOf(at))) {
      return Promise.reject(
        new PlatformError(`no directory ${parentOf(at)} in the workspace`, 'not-found'),
      );
    }
    const revision = this.next();
    this.files.set(at, { text, revision });
    this.announce(current === undefined ? { kind: 'added', path: at, revision } : { kind: 'changed', path: at, revision });
    return Promise.resolve({ revision });
  }

  mkdir(path: WorkspacePath): Promise<void> {
    const at = normalise(path);
    for (const directory of ancestorsOf(at)) this.directories.add(directory);
    this.directories.add(at);
    return Promise.resolve();
  }

  /** Remove a file — what an external change looks like, and what a suite needs to make one. */
  remove(path: WorkspacePath): Promise<void> {
    const at = normalise(path);
    if (!this.files.delete(at)) {
      return Promise.reject(new PlatformError(`no file ${at} in the workspace`, 'not-found'));
    }
    this.announce({ kind: 'removed', path: at });
    return Promise.resolve();
  }

  /**
   * Watch a subtree.
   *
   * Every change is reported, including the editor's own writes — which is what a poll over a
   * folder reports too, since it cannot tell who wrote. A caller distinguishes them by the
   * revision it holds: an event whose revision is the one the caller just wrote is its own.
   */
  watch(path: WorkspacePath, callback: (event: WatchEvent) => void): Unsubscribe {
    const watcher = { at: normalise(path), callback };
    this.watchers.add(watcher);
    return () => {
      this.watchers.delete(watcher);
    };
  }

  resolve(from: WorkspacePath, relative: string): WorkspacePath {
    return resolveFrom(from, relative);
  }

  private next(): string {
    this.clock += 1;
    return `m${String(this.clock)}`;
  }

  private announce(event: WatchEvent): void {
    for (const watcher of this.watchers) {
      if (watcher.at === '' || event.path === watcher.at || event.path.startsWith(`${watcher.at}/`)) {
        watcher.callback(event);
      }
    }
  }
}

/** Every directory strictly above a path, the root first. */
function ancestorsOf(path: WorkspacePath): WorkspacePath[] {
  const segments = segmentsOf(path);
  const found: WorkspacePath[] = [''];
  for (let depth = 1; depth < segments.length; depth += 1) found.push(segments.slice(0, depth).join('/'));
  return found;
}

/** What `path` contributes to a listing of `prefix`, or `null` when it is not under it. */
function headUnder(
  path: WorkspacePath,
  prefix: WorkspacePath,
): { name: string; kind: 'file' | 'directory' } | null {
  if (prefix !== '' && !path.startsWith(`${prefix}/`)) return null;
  if (path === prefix) return null;
  const rest = prefix === '' ? path : path.slice(prefix.length + 1);
  const head = rest.split('/')[0] ?? '';
  if (head === '') return null;
  return { name: head, kind: rest.includes('/') ? 'directory' : 'file' };
}

/** Settings that live as long as the page does. */
export function memorySettings(initial: Readonly<Record<string, SettingValue>> = {}): SettingsStore {
  const held = new Map<string, SettingValue>(Object.entries(initial));
  const listeners = new Set<(key: string) => void>();
  const announce = (key: string): void => {
    for (const listener of listeners) listener(key);
  };
  return {
    get: <T extends SettingValue>(key: string, fallback: T): T => {
      const found = held.get(key);
      return found === undefined ? fallback : (found as T);
    },
    peek: (key) => held.get(key),
    set: (key, value) => {
      held.set(key, value);
      announce(key);
    },
    remove: (key) => {
      held.delete(key);
      announce(key);
    },
    keys: () => [...held.keys()].sort((a, b) => a.localeCompare(b)),
    onChange: (callback) => {
      listeners.add(callback);
      return () => {
        listeners.delete(callback);
      };
    },
    // Nothing here outlives the page, and the chrome is entitled to say so.
    persistent: false,
  };
}

/** Drafts that live as long as the page does. */
export function memoryDrafts(): DraftStore {
  const held = new Map<string, Draft>();
  return {
    put: (draft) => {
      held.set(draftKey(draft.workspace, draft.path), { ...draft, path: normalise(draft.path) });
      return Promise.resolve();
    },
    get: (workspace, path) => Promise.resolve(held.get(draftKey(workspace, path)) ?? null),
    list: (workspace) =>
      Promise.resolve(
        [...held.values()]
          .filter((draft) => workspace === undefined || draft.workspace === workspace)
          .map(summaryOf)
          .sort(byNewest),
      ),
    remove: (workspace, path) => {
      held.delete(draftKey(workspace, path));
      return Promise.resolve();
    },
    clear: (workspace) => {
      if (workspace === undefined) held.clear();
      else for (const [key, draft] of held) if (draft.workspace === workspace) held.delete(key);
      return Promise.resolve();
    },
    persistent: false,
  };
}

/** What a {@link recordingShell} was asked to do, for the suites and for the stub's own page. */
export interface ShellRecord {
  readonly downloads: Delivered[];
  readonly opened: string[];
  readonly asked: string[];
  menu: readonly MenuCommand[];
  clipboard: string | null;
  /** What {@link Shell.confirm} answers. A stub says no: the destructive branch is never taken. */
  answer: boolean;
  /**
   * What {@link Shell.colourScheme} answers, and what a change announces.
   *
   * A stub answers `light` until a suite says otherwise — the same thing `prefers-color-scheme`
   * answers where no preference is expressed — so a run is not a function of the machine.
   */
  scheme: ColourScheme;
}

/** A shell that shows nothing and records everything — the stub's, and a suite's. */
export function recordingShell(): {
  shell: Shell;
  record: ShellRecord;
  prefer: (scheme: ColourScheme) => void;
} {
  const record: ShellRecord = {
    downloads: [],
    opened: [],
    asked: [],
    menu: [],
    clipboard: null,
    answer: false,
    scheme: 'light',
  };
  const watchers = new Set<(scheme: ColourScheme) => void>();
  const shell: Shell = {
    confirm: (question) => {
      record.asked.push(question);
      return Promise.resolve(record.answer);
    },
    openExternal: (url) => {
      record.opened.push(url);
      return Promise.resolve();
    },
    readClipboard: () => Promise.resolve(record.clipboard),
    writeClipboard: (text) => {
      record.clipboard = text;
      return Promise.resolve();
    },
    download: (name, text) => {
      record.downloads.push({ name, path: name, bytes: byteLength(text) });
      return Promise.resolve();
    },
    setMenu: (commands) => {
      record.menu = commands;
    },
    // `control` and not the machine's: a stub says the same thing on every machine, so an
    // accelerator a suite asserts is the accelerator every run sees (§4.4's "Ctrl elsewhere").
    modifier: 'control',
    colourScheme: () => record.scheme,
    onColourSchemeChange: (callback) => {
      watchers.add(callback);
      return () => watchers.delete(callback);
    },
  };
  return {
    shell,
    record,
    /** Announce a change of the machine's preference — a suite's, and the stub page's. */
    prefer: (scheme: ColourScheme): void => {
      record.scheme = scheme;
      for (const watcher of watchers) watcher(scheme);
    },
  };
}

/** What {@link stubPlatform} is given. */
export interface StubOptions {
  /** The workspace it opens with; a memory workspace holding nothing by default. */
  readonly workspace?: Workspace;
  /** The files the **Examples** workspace holds, if the stub is asked for one. */
  readonly examples?: Readonly<Record<string, string>>;
  readonly settings?: Readonly<Record<string, SettingValue>>;
  /**
   * A session the stub's `AuthProvider` holds.
   *
   * Absent — the default — the stub is `NoAuth`, which is what every deployment of this plan has
   * (Q8 defers the SaaS). Given, a suite can ask the other half of "no avatar without a session".
   */
  readonly session?: Session;
}

/** The stub of D11: every interface of §5.2, implemented in memory and naming no platform. */
export function stubPlatform(options: StubOptions = {}): Platform & {
  readonly shellRecord: ShellRecord;
  /** Announce a change of the machine's colour-scheme preference (§4.21) — a suite's. */
  readonly preferScheme: (scheme: ColourScheme) => void;
} {
  const { shell, record, prefer } = recordingShell();
  const settings = options.settings === undefined ? memorySettings() : memorySettings(options.settings);
  const drafts = memoryDrafts();
  let current: Workspace = options.workspace ?? MemoryWorkspace.of();
  const listeners = new Set<(workspace: Workspace) => void>();
  const adopt = (workspace: Workspace): Workspace => {
    current = workspace;
    for (const listener of listeners) listener(workspace);
    return workspace;
  };
  const deliver = (name: string, text: string): Promise<void> => shell.download(name, text);

  const workspaces: Workspaces = {
    // A stub has no picker, and says so rather than inventing a folder nobody chose.
    writablePicker: false,
    open: () =>
      Promise.reject(
        new PlatformError('this deployment has no directory picker: it holds its files in memory', 'unsupported'),
      ),
    openUpload: (files: readonly UploadedFile[]) =>
      Promise.resolve(adopt(new ReadOnlyWorkspace(snapshotOf(files), deliver))),
    openDrop: () => Promise.resolve(null),
    openExamples: () =>
      Promise.resolve(
        adopt(new ReadOnlyWorkspace(textsOf('examples', 'examples', 'Examples', options.examples ?? {}), deliver)),
      ),
    recent: () => Promise.resolve([] as readonly RecentWorkspace[]),
    reopen: () => Promise.resolve(null),
    forget: () => Promise.resolve(),
    current: () => current,
    onChange: (callback) => {
      listeners.add(callback);
      return () => {
        listeners.delete(callback);
      };
    },
  };

  return {
    get workspace() {
      return current;
    },
    workspaces,
    checkpoints: [],
    auth: options.session === undefined ? noAuth('this deployment has no accounts') : memoryAuth(options.session),
    settings,
    drafts,
    shell,
    shellRecord: record,
    preferScheme: prefer,
    describe: () => 'stub',
  };
}
