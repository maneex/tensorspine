/**
 * The read-only workspace: a folder the editor can read and cannot write back — feature 2.4.
 *
 * Two of the three engines get it (feature 0.6 measured it: only Chromium has a writable
 * directory picker), and the vendored **Examples** workspace is one too, so this is not a
 * courtesy for old browsers but what most of the editor's users will see on every save:
 *
 *   - a folder chosen through a folder upload or dropped where the engine gives no handle;
 *   - the corpus and the reference base vendored with the static build (D11, §4.3).
 *
 * What "Save" means here is the honest thing and not a refusal: the bytes are handed to the user
 * as a download, the file on disk is left exactly as it was, and the revision comes back unmoved
 * because nothing moved (§5.2). The chrome says so in a banner and the button says what it will
 * do (S18) — the rule of §9 Q5 applied to Save: no gesture is refused for the platform's reason.
 *
 * The one operation with no answer here is {@link ReadOnlyWorkspace.mkdir}: "New Base creates the
 * folder and manifest" (§4.3) needs a folder to create, and a browser that cannot write one
 * cannot pretend. The refusal names the way out, and feature 3.2 decides what New Base offers.
 *
 * Nothing below names the DOM: a source answers paths and texts, and a browser `File` satisfies
 * {@link UploadedFile} as it stands.
 */
import { byName, join, nameOf, normalise, resolveFrom, type WorkspacePath } from './paths.js';
import {
  ABSENT,
  PlatformError,
  type Entry,
  type Unsubscribe,
  type UploadedFile,
  type WatchEvent,
  type Workspace,
  type WorkspaceKind,
  type WorkspaceRef,
} from './types.js';

/** What a read-only workspace reads from: a fixed set of paths, and the text behind each. */
export interface FileSet {
  readonly kind: WorkspaceKind;
  /** {@link WorkspaceRef.id} — how a draft or a setting finds this workspace again. */
  readonly id: string;
  readonly name: string;
  /** Every file, by workspace path. The set does not change while the workspace is open. */
  paths(): readonly WorkspacePath[];
  /** The text of one file and the revision it is at, or `null` when the set does not hold it. */
  read(path: WorkspacePath): Promise<{ text: string; revision: string } | null>;
  /** The revision of one file without reading it, or `null` when the set does not hold it. */
  revision(path: WorkspacePath): string | null;
}

/** How the bytes reach the user when a read-only workspace is asked to save — the shell's. */
export type Deliver = (name: string, text: string) => Promise<void>;

/** What the last save handed over, so the chrome and the suites can read it back. */
export interface Delivered {
  readonly name: string;
  readonly path: WorkspacePath;
  readonly bytes: number;
}

export class ReadOnlyWorkspace implements Workspace {
  /** What the last {@link write} handed to the user. */
  lastDelivered: Delivered | null = null;

  private readonly watchers = new Set<(event: WatchEvent) => void>();

  constructor(
    private readonly files: FileSet,
    private readonly deliver: Deliver,
  ) {}

  root(): WorkspaceRef {
    return { kind: this.files.kind, id: this.files.id, name: this.files.name, writable: false };
  }

  list(dir: WorkspacePath): Promise<Entry[]> {
    const prefix = normalise(dir);
    const seen = new Map<string, Entry>();
    for (const path of this.files.paths()) {
      if (prefix !== '' && !path.startsWith(`${prefix}/`)) continue;
      const rest = prefix === '' ? path : path.slice(prefix.length + 1);
      const head = rest.split('/')[0] ?? '';
      if (head === '') continue;
      seen.set(head, { name: head, path: join(prefix, head), kind: rest.includes('/') ? 'directory' : 'file' });
    }
    if (seen.size === 0 && prefix !== '') {
      return Promise.reject(
        new PlatformError(`no directory ${prefix} in ${this.files.name}`, 'not-found'),
      );
    }
    return Promise.resolve([...seen.values()].sort(byName));
  }

  async read(path: WorkspacePath): Promise<{ text: string; revision: string }> {
    const found = await this.files.read(normalise(path));
    if (found === null) {
      throw new PlatformError(`no file ${normalise(path)} in ${this.files.name}`, 'not-found');
    }
    return found;
  }

  /**
   * Save: the bytes go to the user as a download and the folder is left alone.
   *
   * A path the workspace never held can be saved all the same — it is a new file the user will
   * put somewhere — and `expect` is still honoured, so a caller that reads, edits and saves gets
   * the same conflict story it would get on a folder the editor can write.
   */
  async write(path: WorkspacePath, text: string, expect?: string): Promise<{ revision: string }> {
    const at = normalise(path);
    const revision = this.files.revision(at) ?? ABSENT;
    if (expect !== undefined && expect !== revision) {
      throw new PlatformError(
        `${at} is not at ${expect} in ${this.files.name} (it is at ${revision})`,
        'conflict',
        { found: revision, expected: expect },
      );
    }
    const name = nameOf(at);
    await this.deliver(name, text);
    this.lastDelivered = { name, path: at, bytes: byteLength(text) };
    return { revision };
  }

  /** There is no folder to create in: this workspace is a copy, or a build's own material. */
  mkdir(path: WorkspacePath): Promise<void> {
    return Promise.reject(
      new PlatformError(
        `${this.files.name} is read-only: ${normalise(path)} cannot be created. Open the folder in ` +
          'a browser with the writable directory picker, or create it on disk.',
        'read-only',
      ),
    );
  }

  /**
   * A watcher is registered and never called: nothing can change under a copy the page was
   * handed, or under the files a build vendored. The editor above does not have to know which
   * workspace it holds — it watches, and this one has nothing to report.
   */
  watch(path: WorkspacePath, callback: (event: WatchEvent) => void): Unsubscribe {
    normalise(path);
    this.watchers.add(callback);
    return () => {
      this.watchers.delete(callback);
    };
  }

  resolve(from: WorkspacePath, relative: string): WorkspacePath {
    return resolveFrom(from, relative);
  }
}

/** The bytes a text takes as UTF-8 — what a download's size is, and not its length in characters. */
export function byteLength(text: string): number {
  return new TextEncoder().encode(text).length;
}

/**
 * The folder's own name is the first segment of every `webkitRelativePath`, and the workspace's
 * paths are what is left. When the files do not share one — several folders dropped at once, or
 * a single file — nothing is stripped and the workspace is named for the drop.
 */
export function stripCommonRoot(
  paths: readonly string[],
  fallback: string,
): { name: string; depth: number } {
  const roots = new Set(paths.map((path) => path.split('/')[0] ?? ''));
  const only = [...roots][0];
  if (roots.size === 1 && only !== undefined && only !== '' && paths.every((path) => path.includes('/'))) {
    return { name: only, depth: 1 };
  }
  return { name: fallback, depth: 0 };
}

/** What a snapshot names itself when the files it was given share no folder. */
export const DROPPED_FOLDER = 'dropped folder';

/**
 * A {@link FileSet} over the files a folder upload or a drop handed to the page.
 *
 * The texts are **not** read here: an {@link UploadedFile} is a lazy handle (a browser `File` is
 * one), so a folder of any size costs a listing and nothing more until a document is opened.
 *
 * A revision is the file's modification time and size, as it is on the writable side. Feature 0.6
 * measured what that can tell apart; on a copy nothing moves, so it only ever answers "unchanged".
 */
export function snapshotOf(files: readonly UploadedFile[], fallback = DROPPED_FOLDER): FileSet {
  const relative = files.map((file) => (file.webkitRelativePath === '' ? file.name : file.webkitRelativePath));
  const { name, depth } = stripCommonRoot(relative, fallback);
  const held = new Map<WorkspacePath, UploadedFile>();
  for (const [index, file] of files.entries()) {
    const path = normalise((relative[index] ?? file.name).split('/').slice(depth).join('/'));
    if (path !== '') held.set(path, file);
  }
  const revisionOf = (file: UploadedFile): string => `${String(file.lastModified)}:${String(file.size)}`;
  return {
    kind: 'snapshot',
    // A folder uploaded again is the same workspace as far as a draft is concerned: the page is
    // handed a copy and has nothing else to recognise it by. Stated rather than promised.
    id: `upload:${name}`,
    name,
    paths: () => [...held.keys()].sort((a, b) => a.localeCompare(b)),
    read: async (path) => {
      const file = held.get(path);
      if (file === undefined) return null;
      return { text: await file.text(), revision: revisionOf(file) };
    },
    revision: (path) => {
      const file = held.get(path);
      return file === undefined ? null : revisionOf(file);
    },
  };
}

/**
 * A {@link FileSet} over texts already in hand — what the stub platform's Examples workspace is,
 * and what a suite hands a read-only workspace.
 *
 * The revision is the text's own length and a counter of the set, which never moves: a set built
 * once cannot change, so two reads of one file answer the same revision and a `write` against it
 * is never in conflict with itself.
 */
export function textsOf(
  kind: WorkspaceKind,
  id: string,
  name: string,
  texts: Readonly<Record<string, string>>,
): FileSet {
  const held = new Map<WorkspacePath, string>();
  for (const [path, text] of Object.entries(texts)) {
    const at = normalise(path);
    if (at !== '') held.set(at, text);
  }
  const revisionOf = (text: string): string => `0:${String(byteLength(text))}`;
  return {
    kind,
    id,
    name,
    paths: () => [...held.keys()].sort((a, b) => a.localeCompare(b)),
    read: (path) => {
      const text = held.get(path);
      return Promise.resolve(text === undefined ? null : { text, revision: revisionOf(text) });
    },
    revision: (path) => {
      const text = held.get(path);
      return text === undefined ? null : revisionOf(text);
    },
  };
}
