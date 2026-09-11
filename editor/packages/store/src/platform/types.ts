/**
 * `Platform` — the interfaces of the implementation plan's §5.2, decided once and implemented
 * per deployment (feature 2.4).
 *
 * > The renderer imports nothing from Node or Electron; everything platform-shaped goes through
 * > `Platform` (D11).
 *
 * Everything the editor cannot compute for itself is here: the files it reads and writes, the
 * settings it remembers, the drafts it autosaves, the checkpoints it reads headers from, who the
 * user is, and the few things only a shell can do (a download, the clipboard, an external link).
 * Three implementations are foreseen — the static web application (feature 2.4,
 * `apps/web/src/platform/`), Electron and the SaaS — and one is in this package: the stub of
 * {@link stubPlatform}, which is what CI builds the application against so that a platform leak
 * is a build failure.
 *
 * **Why the interfaces live in `packages/store` and not in `packages/ui`.** §5.1 gives this
 * package the documents, the sidecars and the drafts — the things a workspace holds — and gives
 * the interface package "no platform import". A type is not an import of a platform, but the
 * dependency has a direction: `packages/ui` depends on this package, and the store's own autosave
 * (feature 2.6) needs {@link DraftStore}. Putting them here points every arrow the right way.
 *
 * **What that buys, mechanically.** This package is compiled with `lib: ES2022` and no DOM (see
 * its `tsconfig.json`), so nothing below *can* name a `File`, a `FileSystemDirectoryHandle` or a
 * `Storage` — an interface that mentioned one would not compile. Where a browser's own object is
 * unavoidable, the declaration takes the structural minimum ({@link UploadedFile}) or admits that
 * the reading is the implementation's ({@link Workspaces.openDrop}), and says so.
 *
 * The language core is **not** behind this interface: it is a library the renderer bundles and
 * runs in a worker, identical on every platform (§5.2), and it is pure — "the UI reads files
 * through `Platform` and hands the core texts and trees" (§5.3).
 */

import type { WorkspacePath } from './paths.js';

export type { WorkspacePath };
export { PlatformError, type PlatformRefusal } from './errors.js';

/** Stop listening. Calling it twice is not an error. */
export type Unsubscribe = () => void;

// ---------------------------------------------------------------------------------------------
// The workspace
// ---------------------------------------------------------------------------------------------

/**
 * What kind of place the open workspace is, for the chrome that has to say so.
 *
 * `directory` is a folder the browser can write in place; `snapshot` is a copy of a folder the
 * page was handed, where Save downloads; `examples` is the corpus and the reference base vendored
 * with the build; `memory` is the stub's; `empty` is the one a deployment holds before the user
 * has opened anything — it lists nothing, finds nothing, and still saves, as a download, because
 * a document made from nothing is still the user's (S17's empty state).
 */
export type WorkspaceKind = 'directory' | 'snapshot' | 'examples' | 'memory' | 'empty';

/** Which workspace is open: what `root()` answers (§5.2, "folder or project id"). */
export interface WorkspaceRef {
  readonly kind: WorkspaceKind;
  /**
   * A stable name for this workspace across reopenings, for whatever is keyed by it — the drafts
   * of §4.3, a per-workspace setting. Stability is the implementation's claim to make: a folder
   * remembered by its handle keeps its identity, a folder uploaded again is recognised by its
   * name alone, and a workspace that cannot promise it says so in its own documentation.
   */
  readonly id: string;
  /** The folder's own name, as the browser or the deployment gave it. */
  readonly name: string;
  /** False where the editor cannot write back: Save downloads instead, and a banner says so. */
  readonly writable: boolean;
}

/** One entry of a directory listing. */
export interface Entry {
  readonly name: string;
  readonly path: WorkspacePath;
  readonly kind: 'file' | 'directory';
}

/**
 * What a watch reports.
 *
 * A poll sees states, so it reports differences: §5.2 asks for "external change → reload prompt",
 * which is what the three kinds are for.
 */
export type WatchEvent =
  | { readonly kind: 'added'; readonly path: WorkspacePath; readonly revision: string }
  | { readonly kind: 'changed'; readonly path: WorkspacePath; readonly revision: string }
  | { readonly kind: 'removed'; readonly path: WorkspacePath };

/**
 * The revision of a file that is not there — what `write` is given to say "it must not exist yet".
 *
 * A revision is otherwise opaque: the browser workspace makes one from a modification time and a
 * size, a REST workspace would use its `ETag`, and no caller may read either.
 */
export const ABSENT = 'absent';

/**
 * The workspace of §5.2: documents, sidecars, library bases, and the schemas a folder carries.
 *
 * Paths are the workspace's own ({@link WorkspacePath}). Nothing here knows what a document is:
 * the editor reads texts through it and hands them to the core.
 */
export interface Workspace {
  /** Which workspace this is, for the chrome and for whatever is keyed by its identity. */
  root(): WorkspaceRef;
  /** The entries of one directory, sorted by name. The root is `''`. */
  list(dir: WorkspacePath): Promise<Entry[]>;
  /** The file's text and the revision it was read at. */
  read(path: WorkspacePath): Promise<{ text: string; revision: string }>;
  /**
   * Write the file and answer its new revision.
   *
   * `expect` is the optimistic concurrency of §5.2: the revision the caller last saw, or
   * {@link ABSENT} for a file that must not exist yet. A write whose expectation does not hold is
   * refused with `conflict` and changes nothing — which is what makes an external change visible
   * instead of silently overwritten.
   *
   * On a workspace that cannot be written the write still happens, as a download: no gesture is
   * refused for the platform's reason (§9 Q5's rule applied to Save), the banner and the button
   * say what it will do, and the revision comes back unmoved because nothing on disk did.
   */
  write(path: WorkspacePath, text: string, expect?: string): Promise<{ revision: string }>;
  /** Create a directory and every directory above it — a new base (§4.22). */
  mkdir(path: WorkspacePath): Promise<void>;
  /** Watch a subtree for changes made behind the editor's back. */
  watch(path: WorkspacePath, callback: (event: WatchEvent) => void): Unsubscribe;
  /**
   * The path a relative reference written in `from` denotes — `primitive_libraries[].base`, a
   * template's file, a sidecar.
   *
   * This is the editor's own navigation. It is **not** the spelling the library loader compares
   * against: the core resolves a base with `os.path`'s rules, where the workspace root is `.`, so
   * a base handed to `loadLibrary` goes through `toPosix` (see `./paths.ts`).
   */
  resolve(from: WorkspacePath, relative: string): WorkspacePath;
}

/**
 * A file the chrome collected from a folder upload or a drop — the read-only snapshot path.
 *
 * The structural minimum of a browser `File`, written out rather than imported so that this
 * package stays free of the DOM: a `File` satisfies it as it stands, and so does anything else a
 * deployment can produce. `webkitRelativePath` is the browser's own name for where the file stood
 * inside the chosen folder, and it is empty for a file that was not chosen as part of one.
 */
export interface UploadedFile {
  readonly name: string;
  readonly webkitRelativePath: string;
  readonly size: number;
  readonly lastModified: number;
  text(): Promise<string>;
}

/** A folder the user opened before, offered by name in the File menu (§4.3). */
export interface RecentWorkspace {
  readonly id: string;
  readonly name: string;
  /** When it was last opened, ISO 8601. */
  readonly openedAt: string;
}

/**
 * Opening a workspace, and remembering which ones were opened — §4.3's File menu.
 *
 * §5.2 gives `Platform` one `workspace`; §4.3 gives the user the commands that replace it. So the
 * open one is {@link Platform.workspace}, a getter, and the commands are here: a component reads
 * the workspace it needs at the moment it needs it and subscribes to {@link onChange} rather than
 * holding one that a reopen would have left behind.
 */
export interface Workspaces {
  /** Whether this deployment can open a folder for **writing** at all (Chromium alone today). */
  readonly writablePicker: boolean;
  /**
   * The Open Folder… command: the gesture a static page needs before it can read anything.
   *
   * Answers the workspace it opened, or `null` when the user dismissed the picker. Refuses with
   * `unsupported` where the deployment has no writable picker — the chrome offers the folder
   * upload there, which is {@link openUpload}.
   */
  open(): Promise<Workspace | null>;
  /** A folder chosen through a folder upload: the read-only snapshot, where Save downloads. */
  openUpload(files: readonly UploadedFile[]): Promise<Workspace>;
  /**
   * A folder dropped on the window.
   *
   * The argument is what a `drop` event carries — a `DataTransfer`. The interface cannot name it
   * without naming the DOM, and naming it would not help: a transfer is disabled the moment the
   * synchronous part of the handler returns, so the reading has to happen inside the
   * implementation, before its first `await`, which is the defect the review repair `910539b`
   * closed in the spike. The caller hands over the event's transfer and nothing else.
   *
   * **Call it in the handler itself, with nothing awaited before it.** That is the other half of
   * the same rule and it is the caller's: an implementation that reads the transfer carefully
   * cannot save a handler that reached it one `await` too late — `items.length` is 0 by then and
   * every entry answers null. So the platform is built before the window can be dropped on (which
   * is what `main.ts` does), and the handler calls this synchronously. Measured here, not
   * reasoned: a handler that awaited the platform's construction first opened an empty snapshot.
   *
   * Where the engine gives a real directory handle for a dropped folder the workspace is
   * *writable* — the picker is not the only way in (feature 0.6) — and otherwise it is a
   * snapshot. `null` when the drop carried nothing this workspace can open.
   */
  openDrop(transfer: unknown): Promise<Workspace | null>;
  /**
   * The **Examples** workspace: the corpus and the reference base vendored with the build,
   * read-only, always available (D11, §4.3).
   */
  openExamples(): Promise<Workspace>;
  /** The folders the user opened before, most recently opened first. */
  recent(): Promise<readonly RecentWorkspace[]>;
  /**
   * Reopen a remembered folder — one permission prompt, never a second picker (D11).
   *
   * `null` when the grant was refused or the folder is no longer reachable.
   */
  reopen(id: string): Promise<Workspace | null>;
  /** Forget one remembered folder, or all of them when no id is given. */
  forget(id?: string): Promise<void>;
  /** The workspace that is open now, whichever command opened it. */
  current(): Workspace;
  /**
   * Called after a command above replaced the open workspace.
   *
   * Whatever a listener started on the workspace it held — a {@link Workspace.watch}, above all —
   * is its own to stop here: two opens can be in flight at once (Open Folder clicked, then a
   * recent clicked while the first listing is still being read), and a poll whose `Unsubscribe`
   * nobody holds any more runs for the life of the page. That is the second defect the review
   * repair `910539b` closed, and it is a caller's defect, not this interface's.
   */
  onChange(callback: (workspace: Workspace) => void): Unsubscribe;
}

// ---------------------------------------------------------------------------------------------
// Authentication
// ---------------------------------------------------------------------------------------------

/** Who the user is, where a deployment knows. */
export interface Session {
  readonly id: string;
  readonly name: string;
  readonly email?: string;
}

/**
 * §5.2's `AuthProvider`: `NoAuth` on the desktop and on the static web application, OIDC or a
 * session cookie when the SaaS is scheduled (Q8 defers it).
 */
export interface AuthProvider {
  current(): Session | null;
  /** Refuses with `unsupported` where the deployment has no accounts. */
  signIn(): Promise<Session>;
  signOut(): Promise<void>;
  onChange(callback: (session: Session | null) => void): Unsubscribe;
}

// ---------------------------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------------------------

/** What a setting may hold: anything that survives `JSON.stringify` unchanged. */
export type SettingValue = string | number | boolean | null | readonly SettingValue[] | { readonly [key: string]: SettingValue };

/**
 * §5.2's `SettingsStore`: "user settings, layouts, recents, unlocked bases".
 *
 * **Read synchronously, written through.** A setting is read while a component renders — which
 * theme, which panel sizes, whether a base is unlocked (Q6) — so the store is loaded once when
 * the platform is built and answers from memory afterwards; a write goes to memory and to the
 * backing store at once. A deployment whose settings live on a server loads them the same way.
 *
 * **Storage can refuse.** `localStorage` is not merely empty in a private window or where site
 * data is blocked: reading the property itself throws. A store that cannot persist still works —
 * settings live for the session — and says so through {@link persistent}, which is what the
 * chrome needs to warn with.
 *
 * Recents are **not** here, though §5.2 lists them: a remembered folder is a
 * `FileSystemDirectoryHandle`, which IndexedDB stores and `localStorage` cannot, so they live
 * with {@link Workspaces} instead.
 */
export interface SettingsStore {
  /**
   * The stored value, or `fallback` when nothing is stored under that key.
   *
   * The fallback's type is the caller's claim about what it stores, and nothing checks it: a value
   * of another shape — written by an older editor, or by hand — comes back as it stands, so a
   * caller that cares looks before it reads. Settings are the one thing here no schema describes.
   */
  get<T extends SettingValue>(key: string, fallback: T): T;
  /** The stored value, or `undefined`. */
  peek(key: string): SettingValue | undefined;
  set(key: string, value: SettingValue): void;
  remove(key: string): void;
  /** Every key the store holds, sorted. */
  keys(): string[];
  /** Called after a key changed — in this page, or in another one where the deployment sees it. */
  onChange(callback: (key: string) => void): Unsubscribe;
  /** False when the backing store refused: the settings are this session's alone. */
  readonly persistent: boolean;
}

// ---------------------------------------------------------------------------------------------
// Drafts
// ---------------------------------------------------------------------------------------------

/**
 * One autosaved document (§4.3: "Autosave of drafts every 30 s and on blur to the platform's
 * draft store; on reopening a document with a newer draft the editor offers to restore it").
 */
export interface Draft {
  /** The workspace the document belongs to — {@link WorkspaceRef.id}. */
  readonly workspace: string;
  readonly path: WorkspacePath;
  /** The text as the editor held it, through the core's serializer or the plain writer (D12). */
  readonly text: string;
  /**
   * The revision the document was read at when the draft was taken, or {@link ABSENT} for a
   * document that was never in the workspace. What "a newer draft" means is compared against it.
   */
  readonly revision: string;
  /** When it was written, ISO 8601. */
  readonly savedAt: string;
}

/** A draft without its text: what a list of drafts shows. */
export type DraftSummary = Omit<Draft, 'text'> & { readonly bytes: number };

/** §5.2's `DraftStore`: the autosave, keyed by the workspace a document belongs to and its path. */
export interface DraftStore {
  put(draft: Draft): Promise<void>;
  get(workspace: string, path: WorkspacePath): Promise<Draft | null>;
  /** Every draft, newest first; of one workspace when it is named. */
  list(workspace?: string): Promise<DraftSummary[]>;
  remove(workspace: string, path: WorkspacePath): Promise<void>;
  /** Drop every draft — of one workspace when it is named. */
  clear(workspace?: string): Promise<void>;
  /** False when the backing store refused: drafts live for the session alone. */
  readonly persistent: boolean;
}

// ---------------------------------------------------------------------------------------------
// The shell
// ---------------------------------------------------------------------------------------------

/** One command of a native menu, as a deployment that has one is told about it. */
export interface MenuCommand {
  readonly id: string;
  readonly label: string;
  readonly accelerator?: string;
  readonly enabled: boolean;
}

/** §5.2's `Shell`: "dialogs, open-external, clipboard, native menu hooks". */
export interface Shell {
  /** A yes/no the platform asks in its own way. */
  confirm(question: string): Promise<boolean>;
  /** Open a link outside the editor — the specification, the guide, a repository. */
  openExternal(url: string): Promise<void>;
  /** Read the clipboard's text; `null` where the platform refused or holds none. */
  readClipboard(): Promise<string | null>;
  writeClipboard(text: string): Promise<void>;
  /**
   * Hand the user a file to keep.
   *
   * The one thing a page can do where it cannot write: Save on a read-only workspace goes through
   * here, and so will "Save As" and the workspace export of §4.3.
   */
  download(name: string, text: string): Promise<void>;
  /**
   * Offer the application's commands to a native menu.
   *
   * A browser has none and ignores them; Electron builds its menu bar from them (§4.4). The
   * in-app menu is the interface's own and is not this.
   */
  setMenu(commands: readonly MenuCommand[]): void;
}

// ---------------------------------------------------------------------------------------------
// Checkpoints
// ---------------------------------------------------------------------------------------------

/** Where a checkpoint came from, for the Weights panel's header (§4.19). */
export interface SourceInfo {
  readonly name: string;
  /** What the user needs to recognise it: a folder, a repository and a revision, a file count. */
  readonly detail?: string;
}

/**
 * §5.2's `CheckpointSource`: `config.json` and the safetensors headers of one checkpoint.
 *
 * Declared here and implemented by feature 4.1 — a local directory, the Hub through range
 * requests, or files the user drops. The bytes a header needs are the file's first eight and the
 * JSON object behind them (D10, feature 0.5): `headerBytes` answers enough of the file for the
 * core's `readHeader`, and no weight is ever read.
 *
 * `config()` answers the text of `config.json`, not a parsed object: the editor parses with the
 * core's lexeme-preserving reader, as it does every other JSON, so that what the config tree of
 * §4.19 shows is what the file says.
 */
export interface CheckpointSource {
  config(): Promise<string>;
  headerBytes(file: string): Promise<Uint8Array>;
  files(): Promise<string[]>;
  describe(): SourceInfo;
}

// ---------------------------------------------------------------------------------------------
// The platform
// ---------------------------------------------------------------------------------------------

/** Everything the editor cannot do for itself, in one place (§5.2). */
export interface Platform {
  /** The workspace that is open now. Replaced by a command of {@link workspaces}. */
  readonly workspace: Workspace;
  /** Opening a workspace, and the folders opened before (§4.3). */
  readonly workspaces: Workspaces;
  /** The checkpoints loaded in this session; feature 4.1 fills it. */
  readonly checkpoints: readonly CheckpointSource[];
  readonly auth: AuthProvider;
  readonly settings: SettingsStore;
  readonly drafts: DraftStore;
  readonly shell: Shell;
  /** What this platform is, for the log and for the chrome: `browser`, `stub`, later `electron`. */
  readonly describe: () => string;
}
