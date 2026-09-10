import {
  ABSENT,
  byName,
  join,
  normalise,
  parentOf,
  resolveFrom,
  segmentsOf,
  WorkspaceError,
  type Entry,
  type Path,
  type Unsubscribe,
  type WatchEvent,
  type Workspace,
  type WorkspaceRef,
} from './workspace.ts';

/**
 * The writable workspace of D11: a folder the browser opened, read and written **in place**
 * through the File System Access API.
 *
 * Everything below goes through `FileSystemDirectoryHandle` and `FileSystemFileHandle`, and
 * through nothing else — which is why the tests can exercise it against an Origin Private File
 * System handle: OPFS answers the same interface, and the native picker (which no automated
 * browser can drive) is only where the handle *comes from*. `open` therefore takes a handle;
 * `pickDirectory` is the one line that asks for one.
 */

/** A revision is what a poll can see cheaply: the modification time and the size. */
function revisionOf(file: File): string {
  return `${String(file.lastModified)}:${String(file.size)}`;
}

/** `showDirectoryPicker`, which no `lib.dom` declares — the writable picker of D11. */
interface DirectoryPicker {
  showDirectoryPicker?: (options?: { mode?: 'read' | 'readwrite'; id?: string }) => Promise<FileSystemDirectoryHandle>;
}

/** True where a folder can be opened for writing at all (Chromium today; see `../NOTE.md`). */
export function hasDirectoryPicker(): boolean {
  return typeof (window as unknown as DirectoryPicker).showDirectoryPicker === 'function';
}

/**
 * Ask the browser for a folder to write in. The user's gesture is the Open Folder command; the
 * page cannot reach a folder without one, and this call is the whole of that path.
 */
export async function pickDirectory(): Promise<FileSystemDirectoryHandle | null> {
  const picker = (window as unknown as DirectoryPicker).showDirectoryPicker;
  if (picker === undefined) return null;
  return picker({ mode: 'readwrite', id: 'tensorspine-workspace' });
}

/** A `DOMException` the File System Access API throws for a name that is not there. */
function notFound(error: unknown): boolean {
  return error instanceof DOMException && (error.name === 'NotFoundError' || error.name === 'TypeMismatchError');
}

/** How often {@link DirectoryWorkspace.watch} looks, in milliseconds. */
export const DEFAULT_POLL_MS = 1_000;

export class DirectoryWorkspace implements Workspace {
  /** Open a workspace on a handle — from the picker, from a drop, or from IndexedDB. */
  static open(handle: FileSystemDirectoryHandle, options: { pollMs?: number } = {}): DirectoryWorkspace {
    return new DirectoryWorkspace(handle, options.pollMs ?? DEFAULT_POLL_MS);
  }

  private constructor(
    readonly handle: FileSystemDirectoryHandle,
    private readonly pollMs: number,
  ) {}

  root(): WorkspaceRef {
    return { kind: 'directory', name: this.handle.name, writable: true };
  }

  async list(dir: Path): Promise<Entry[]> {
    const directory = await this.directory(dir, false);
    const entries: Entry[] = [];
    for await (const child of directory.values()) {
      entries.push({ name: child.name, path: join(dir, child.name), kind: child.kind === 'directory' ? 'directory' : 'file' });
    }
    return entries.sort(byName);
  }

  async read(path: Path): Promise<{ text: string; revision: string }> {
    const file = await (await this.file(path, false)).getFile();
    return { text: await file.text(), revision: revisionOf(file) };
  }

  /**
   * Write the file, in place. With `expect`, the write is refused when the file has moved on
   * since the caller read it — the optimistic concurrency of §5.2, which is what makes an
   * external change (the `watch` above, another editor, a `git checkout`) visible instead of
   * silently overwritten.
   */
  async write(path: Path, text: string, expect?: string): Promise<{ revision: string }> {
    if (expect !== undefined) {
      const found = await this.revision(path);
      if (found !== expect) {
        throw new WorkspaceError(
          `${path} changed since it was read (${found} is not ${expect})`,
          'conflict',
          { found, expected: expect },
        );
      }
    }
    const handle = await this.file(path, true);
    const writable = await handle.createWritable();
    await writable.write(text);
    await writable.close();
    return { revision: revisionOf(await handle.getFile()) };
  }

  /** Create a directory and every directory above it that is missing. */
  async mkdir(path: Path): Promise<void> {
    await this.directory(path, true);
  }

  /**
   * Watch by polling, as §5.2 says: every `pollMs`, the subtree under `path` is listed and its
   * revisions compared with the last listing. What the caller gets is the difference.
   */
  watch(path: Path, callback: (event: WatchEvent) => void): Unsubscribe {
    let previous: Map<Path, string> | null = null;
    let stopped = false;
    let polling = false;
    const tick = async (): Promise<void> => {
      // One poll at a time: a listing of a large tree can outlast the interval, and two polls
      // diffing against the same previous state would report every change twice.
      if (polling || stopped) return;
      polling = true;
      // A folder that has gone (removed, or the grant withdrawn) must not throw into a timer.
      const current = await this.revisions(path).catch(() => null);
      polling = false;
      if (stopped || current === null) return;
      if (previous !== null) {
        for (const [file, revision] of current) {
          const was = previous.get(file);
          if (was === undefined) callback({ kind: 'added', path: file, revision });
          else if (was !== revision) callback({ kind: 'changed', path: file, revision });
        }
        for (const file of previous.keys()) if (!current.has(file)) callback({ kind: 'removed', path: file });
      }
      previous = current;
    };
    void tick();
    const timer = setInterval(() => void tick(), this.pollMs);
    return () => {
      stopped = true;
      clearInterval(timer);
    };
  }

  resolve(from: Path, relative: string): Path {
    return resolveFrom(from, relative);
  }

  /** Every file under `path`, with its revision — one poll of {@link watch}. */
  async revisions(path: Path): Promise<Map<Path, string>> {
    const found = new Map<Path, string>();
    const walk = async (directory: FileSystemDirectoryHandle, prefix: Path): Promise<void> => {
      for await (const child of directory.values()) {
        const childPath = join(prefix, child.name);
        if (child.kind === 'directory') await walk(child, childPath);
        else found.set(childPath, revisionOf(await child.getFile()));
      }
    };
    await walk(await this.directory(path, false), normalise(path));
    return found;
  }

  /** The revision of one file, or {@link ABSENT} when it is not there. */
  async revision(path: Path): Promise<string> {
    try {
      return revisionOf(await (await this.file(path, false)).getFile());
    } catch (error) {
      if (notFound(error) || (error instanceof WorkspaceError && error.reason === 'not-found')) return ABSENT;
      throw error;
    }
  }

  private async directory(path: Path, create: boolean): Promise<FileSystemDirectoryHandle> {
    let handle = this.handle;
    for (const segment of segmentsOf(path)) {
      try {
        handle = await handle.getDirectoryHandle(segment, { create });
      } catch (error) {
        if (notFound(error)) throw new WorkspaceError(`no directory ${normalise(path)} in the workspace`, 'not-found');
        throw error;
      }
    }
    return handle;
  }

  private async file(path: Path, create: boolean): Promise<FileSystemFileHandle> {
    const name = segmentsOf(path).at(-1);
    if (name === undefined) throw new WorkspaceError('the workspace root is not a file', 'bad-path');
    // The directories above the file are never created here: `mkdir` is the operation that
    // makes a folder, so a path with a typo in it is a refusal rather than a new tree on disk.
    const directory = await this.directory(parentOf(path), false);
    try {
      return await directory.getFileHandle(name, { create });
    } catch (error) {
      if (notFound(error)) throw new WorkspaceError(`no file ${normalise(path)} in the workspace`, 'not-found');
      throw error;
    }
  }
}
