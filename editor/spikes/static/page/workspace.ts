/**
 * The `Workspace` of the implementation plan's §5.2, as the static application needs it —
 * feature 0.6's spike.
 *
 * This is spike code, not `packages/store` and not `packages/ui`: feature 2.4 owns the real
 * `Platform`, and it will inherit this shape and the findings of `../NOTE.md`. Two
 * implementations sit behind it, exactly as D11 describes:
 *
 *   - `directory.ts` — a folder the browser opened through the File System Access API, read and
 *     written in place;
 *   - `snapshot.ts` — a folder dropped on the window or chosen through a folder upload, read
 *     only, where `write` downloads the file for the user to put back.
 *
 * Nothing here knows what a document is. Paths are the workspace's own: relative to its root,
 * `/`-separated, no leading slash, no `.` and no `..` left in them.
 */

/** A path inside the workspace: relative to its root, `/`-separated. `''` is the root itself. */
export type Path = string;

/** What the workspace is, for the chrome that has to say so. */
export interface WorkspaceRef {
  /** `directory` when a handle backs it, `snapshot` when a copy of the files does. */
  readonly kind: 'directory' | 'snapshot';
  /** The folder's own name, as the browser gave it. */
  readonly name: string;
  /** False when the browser cannot write back: Save downloads instead, and a banner says so. */
  readonly writable: boolean;
}

/** One entry of a directory listing. */
export interface Entry {
  readonly name: string;
  readonly path: Path;
  readonly kind: 'file' | 'directory';
}

/** What `watch` reports, by polling — the poll sees states, so it reports differences. */
export type WatchEvent =
  | { readonly kind: 'added'; readonly path: Path; readonly revision: string }
  | { readonly kind: 'changed'; readonly path: Path; readonly revision: string }
  | { readonly kind: 'removed'; readonly path: Path };

/** Stop watching. Calling it twice is not an error. */
export type Unsubscribe = () => void;

/** The revision of a file that is not there — what `write` expects when it must create. */
export const ABSENT = 'absent';

/** The platform interface of plan §5.2, with the types it leaves to the implementation. */
export interface Workspace {
  root(): WorkspaceRef;
  list(dir: Path): Promise<Entry[]>;
  read(path: Path): Promise<{ text: string; revision: string }>;
  /** `expect` is the revision the caller last saw; {@link ABSENT} says "it must not exist yet". */
  write(path: Path, text: string, expect?: string): Promise<{ revision: string }>;
  mkdir(path: Path): Promise<void>;
  watch(path: Path, callback: (event: WatchEvent) => void): Unsubscribe;
  /**
   * The path a relative reference written *in* `from` denotes — `primitive_libraries[].base`,
   * a template's file, a sidecar. `from` is the file the reference is written in, so the
   * reference resolves against the directory that holds it.
   */
  resolve(from: Path, relative: string): Path;
}

/** A refusal of the workspace itself, as opposed to one of the language. */
export class WorkspaceError extends Error {
  constructor(
    message: string,
    /** `not-found`, `read-only`, `conflict`, `bad-path`. */
    readonly reason: 'not-found' | 'read-only' | 'conflict' | 'bad-path',
    /** The revisions a conflict compared: what the file holds, and what the caller expected. */
    readonly revisions?: { readonly found: string; readonly expected: string },
  ) {
    super(message);
    this.name = 'WorkspaceError';
  }
}

/**
 * A path with `.` and empty segments dropped and `..` applied, refusing anything that would
 * leave the workspace. A workspace path never begins with `/` and never ends with one, so
 * `resolve` and the listings agree on one spelling per file.
 */
export function normalise(path: Path): Path {
  const segments: string[] = [];
  for (const segment of path.split('/')) {
    if (segment === '' || segment === '.') continue;
    if (segment !== '..') {
      segments.push(segment);
      continue;
    }
    if (segments.length === 0) {
      throw new WorkspaceError(`the path ${JSON.stringify(path)} leaves the workspace`, 'bad-path');
    }
    segments.pop();
  }
  return segments.join('/');
}

/** The segments of a path, the root being none. */
export function segmentsOf(path: Path): string[] {
  const normalised = normalise(path);
  return normalised === '' ? [] : normalised.split('/');
}

/** The directory a path is in — `''` for a file at the root. */
export function parentOf(path: Path): Path {
  const segments = segmentsOf(path);
  segments.pop();
  return segments.join('/');
}

/** The last segment of a path: the file's own name. */
export function nameOf(path: Path): string {
  return segmentsOf(path).at(-1) ?? '';
}

/** Join a directory and a name, either of which may be empty. */
export function join(directory: Path, name: string): Path {
  return normalise(directory === '' ? name : `${directory}/${name}`);
}

/** {@link Workspace.resolve}, shared by both implementations: `relative` is read from `from`. */
export function resolveFrom(from: Path, relative: string): Path {
  if (relative.startsWith('/')) return normalise(relative);
  return normalise(`${parentOf(from)}/${relative}`);
}

/** Order a listing the way a tree shows it: directories and files by name, as the site sorts. */
export function byName(a: Entry, b: Entry): number {
  return a.name.localeCompare(b.name);
}
