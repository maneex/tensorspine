/**
 * The writable workspace of D11: a folder the browser opened, read and written **in place**
 * through the File System Access API — feature 2.4, promoted from feature 0.6's spike.
 *
 * Everything below goes through `FileSystemDirectoryHandle` and `FileSystemFileHandle` and through
 * nothing else, which is why the browser layer can exercise it against an **Origin Private File
 * System** handle: OPFS answers the same interface, and the native picker — which no automated
 * browser can drive — is only where the handle *comes from*. {@link DirectoryWorkspace.open}
 * therefore takes a handle; `workspaces.ts` holds the one line that asks for one.
 *
 * Where the picker exists, measured in three engines (feature 0.6): Chromium alone. Firefox has
 * every other piece — handles, `createWritable`, directory iteration, OPFS, persistence in
 * IndexedDB — and no way to *obtain* a folder; WebKit has none of it. So this is the minority
 * path, and the read-only snapshot beside it is what most users will see.
 */
import {
  ABSENT,
  join,
  normalise,
  parentOf,
  PlatformError,
  resolveFrom,
  segmentsOf,
  byName,
  type Entry,
  type Unsubscribe,
  type WatchEvent,
  type Workspace,
  type WorkspacePath,
  type WorkspaceRef,
} from '@tensorspine/store/platform';

/**
 * A revision is what a poll can see cheaply: the modification time and the size.
 *
 * Feature 0.6 measured what that can and cannot tell apart — two writes of the *same length*
 * inside one millisecond produce the same revision, and that was observed — so the optimistic
 * concurrency of §5.2 is **best-effort**: it catches every change a human or another process
 * makes, and cannot catch a race. Feature 2.6 decides whether the reload prompt needs a content
 * hash, which would cost a read of every file per poll instead of a stat.
 */
function revisionOf(file: File): string {
  return `${String(file.lastModified)}:${String(file.size)}`;
}

/** A `DOMException` the File System Access API throws for a name that is not there. */
function notFound(error: unknown): boolean {
  return error instanceof DOMException && (error.name === 'NotFoundError' || error.name === 'TypeMismatchError');
}

/**
 * How often {@link DirectoryWorkspace.watch} looks.
 *
 * One poll of a `data/`-sized tree (146 files) cost 22–25 ms in Chromium over OPFS (feature 0.6),
 * so a second is a few per cent of a core and a tenth of one would not be.
 */
export const DEFAULT_POLL_MS = 1_000;

/** What {@link DirectoryWorkspace.open} is given besides the handle. */
export interface DirectoryOptions {
  /** {@link WorkspaceRef.id} — the remembered folder's, so a draft finds it again. */
  readonly id?: string;
  readonly pollMs?: number;
}

export class DirectoryWorkspace implements Workspace {
  /** Open a workspace on a handle — from the picker, from a drop, or from IndexedDB. */
  static open(handle: FileSystemDirectoryHandle, options: DirectoryOptions = {}): DirectoryWorkspace {
    return new DirectoryWorkspace(handle, options.id ?? `directory:${handle.name}`, options.pollMs ?? DEFAULT_POLL_MS);
  }

  private constructor(
    readonly handle: FileSystemDirectoryHandle,
    private readonly id: string,
    private readonly pollMs: number,
  ) {}

  root(): WorkspaceRef {
    return { kind: 'directory', id: this.id, name: this.handle.name, writable: true };
  }

  async list(dir: WorkspacePath): Promise<Entry[]> {
    const directory = await this.directory(dir, false);
    const entries: Entry[] = [];
    for await (const child of directory.values()) {
      entries.push({
        name: child.name,
        path: join(dir, child.name),
        kind: child.kind === 'directory' ? 'directory' : 'file',
      });
    }
    return entries.sort(byName);
  }

  async read(path: WorkspacePath): Promise<{ text: string; revision: string }> {
    const file = await (await this.file(path, false)).getFile();
    return { text: await file.text(), revision: revisionOf(file) };
  }

  /**
   * Write the file, in place, refusing when it has moved on since the caller read it.
   *
   * `expect` is the optimistic concurrency of §5.2 — what makes an external change (the watch
   * below, another editor, a `git checkout`) visible instead of silently overwritten.
   */
  async write(path: WorkspacePath, text: string, expect?: string): Promise<{ revision: string }> {
    if (expect !== undefined) {
      const found = await this.revision(path);
      if (found !== expect) {
        throw new PlatformError(`${normalise(path)} changed since it was read (${found} is not ${expect})`, 'conflict', {
          found,
          expected: expect,
        });
      }
    }
    const handle = await this.file(path, true);
    const writable = await handle.createWritable();
    await writable.write(text);
    await writable.close();
    return { revision: revisionOf(await handle.getFile()) };
  }

  /** Create a directory and every directory above it that is missing — a new base (§4.22). */
  async mkdir(path: WorkspacePath): Promise<void> {
    await this.directory(path, true);
  }

  /**
   * Watch by polling, as §5.2 says: every `pollMs` the subtree under `path` is listed and its
   * revisions compared with the last listing. What the caller gets is the difference.
   */
  watch(path: WorkspacePath, callback: (event: WatchEvent) => void): Unsubscribe {
    let previous: Map<WorkspacePath, string> | null = null;
    let stopped = false;
    let polling = false;
    const tick = async (): Promise<void> => {
      // One poll at a time: a listing of a large tree can outlast the interval, and two polls
      // diffing against the same previous state would report every change twice.
      if (polling || stopped) return;
      polling = true;
      // A folder that has gone — removed, or the grant withdrawn — must not throw into a timer.
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

  resolve(from: WorkspacePath, relative: string): WorkspacePath {
    return resolveFrom(from, relative);
  }

  /** Every file under `path`, with its revision — one poll of {@link watch}. */
  async revisions(path: WorkspacePath): Promise<Map<WorkspacePath, string>> {
    const found = new Map<WorkspacePath, string>();
    const walk = async (directory: FileSystemDirectoryHandle, prefix: WorkspacePath): Promise<void> => {
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
  async revision(path: WorkspacePath): Promise<string> {
    try {
      return revisionOf(await (await this.file(path, false)).getFile());
    } catch (error) {
      if (notFound(error) || (error instanceof PlatformError && error.reason === 'not-found')) return ABSENT;
      throw error;
    }
  }

  private async directory(path: WorkspacePath, create: boolean): Promise<FileSystemDirectoryHandle> {
    let handle = this.handle;
    for (const segment of segmentsOf(path)) {
      try {
        handle = await handle.getDirectoryHandle(segment, { create });
      } catch (error) {
        if (notFound(error)) throw new PlatformError(`no directory ${normalise(path)} in the workspace`, 'not-found');
        throw error;
      }
    }
    return handle;
  }

  private async file(path: WorkspacePath, create: boolean): Promise<FileSystemFileHandle> {
    const name = segmentsOf(path).at(-1);
    if (name === undefined) throw new PlatformError('the workspace root is not a file', 'bad-path');
    // The directories above the file are never created here: `mkdir` is the operation that makes
    // a folder, so a path with a typo in it is a refusal rather than a new tree on disk. The File
    // System Access API makes the wrong behaviour a one-character difference (feature 0.6).
    const directory = await this.directory(parentOf(path), false);
    try {
      return await directory.getFileHandle(name, { create });
    } catch (error) {
      if (notFound(error)) throw new PlatformError(`no file ${normalise(path)} in the workspace`, 'not-found');
      throw error;
    }
  }
}
