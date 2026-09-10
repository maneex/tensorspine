import {
  ABSENT,
  byName,
  join,
  nameOf,
  normalise,
  parentOf,
  resolveFrom,
  WorkspaceError,
  type Entry,
  type Path,
  type Unsubscribe,
  type WatchEvent,
  type Workspace,
  type WorkspaceRef,
} from './workspace.ts';

/**
 * The read-only workspace of D11: a folder **dropped on the window or chosen through a folder
 * upload**, in a browser with no writable picker (Firefox and Safari today — `../NOTE.md` has
 * the measurements).
 *
 * What the browser gives is a list of `File`s with the path each had inside the chosen folder,
 * and nothing that can be written back. So this workspace reads from that copy, and `write`
 * does the only honest thing left: it **downloads** the file for the user to put back, and
 * reports the revision unchanged, because nothing on disk has moved. The banner in the chrome
 * says so, and Save says what it will actually do (S18).
 */

/** One file of the snapshot: where it was in the chosen folder, and what the browser handed over. */
interface SnapshotFile {
  readonly path: Path;
  readonly file: File;
}

/** What the last {@link SnapshotWorkspace.write} offered, so a caller can check what it wrote. */
export interface Download {
  readonly name: string;
  readonly path: Path;
  readonly url: string;
  readonly bytes: number;
}

/** A revision is a modification time and a size, as it is for a directory handle. */
function revisionOf(file: File): string {
  return `${String(file.lastModified)}:${String(file.size)}`;
}

/**
 * The folder's own name is the first segment of every `webkitRelativePath`; the workspace's
 * paths are what is left. When the files do not share one — several folders dropped at once —
 * nothing is stripped and the workspace is named for the drop.
 */
function stripCommonRoot(paths: readonly string[]): { name: string; depth: number } {
  const roots = new Set(paths.map((path) => path.split('/')[0] ?? ''));
  const only = [...roots][0];
  if (roots.size === 1 && only !== undefined && only !== '' && paths.every((path) => path.includes('/'))) {
    return { name: only, depth: 1 };
  }
  return { name: 'dropped folder', depth: 0 };
}

/** Walk a dropped directory entry, which is the only reading a drop offers without handles. */
async function walkEntry(entry: FileSystemEntry, into: SnapshotFile[]): Promise<void> {
  if (entry.isFile) {
    const file = await new Promise<File>((resolve, reject) => {
      (entry as FileSystemFileEntry).file(resolve, reject);
    });
    into.push({ path: normalise(entry.fullPath), file });
    return;
  }
  const reader = (entry as FileSystemDirectoryEntry).createReader();
  for (;;) {
    const batch = await new Promise<FileSystemEntry[]>((resolve, reject) => {
      reader.readEntries(resolve, reject);
    });
    if (batch.length === 0) return;
    for (const child of batch) await walkEntry(child, into);
  }
}

/**
 * `deliver: false` builds the download without handing it to the browser. The one caller is the
 * cross-engine runner of `../run.ts`, which opens three browsers nobody is watching and must
 * not litter the machine with files; every other path delivers, which is the point of Save.
 */
export interface SnapshotOptions {
  readonly deliver?: boolean;
}

export class SnapshotWorkspace implements Workspace {
  /** A folder chosen through `<input type="file" webkitdirectory>`. */
  static fromFiles(files: readonly File[], options: SnapshotOptions = {}): SnapshotWorkspace {
    const relative = files.map((file) => (file.webkitRelativePath === '' ? file.name : file.webkitRelativePath));
    const { name, depth } = stripCommonRoot(relative);
    const entries = files.map((file, index) => ({
      path: normalise((relative[index] ?? file.name).split('/').slice(depth).join('/')),
      file,
    }));
    return new SnapshotWorkspace(name, entries, options.deliver ?? true);
  }

  /** A folder dropped on the window, in an engine with no writable handle to give. */
  static async fromDataTransfer(items: DataTransferItemList, options: SnapshotOptions = {}): Promise<SnapshotWorkspace> {
    const roots: FileSystemEntry[] = [];
    for (let index = 0; index < items.length; index += 1) {
      const entry = items[index]?.webkitGetAsEntry() ?? null;
      if (entry !== null) roots.push(entry);
    }
    const found: SnapshotFile[] = [];
    for (const entry of roots) await walkEntry(entry, found);
    const { name, depth } = stripCommonRoot(found.map((one) => one.path));
    return new SnapshotWorkspace(
      name,
      found.map((one) => ({ path: normalise(one.path.split('/').slice(depth).join('/')), file: one.file })),
      options.deliver ?? true,
    );
  }

  private readonly files: Map<Path, File>;
  private readonly watchers = new Set<(event: WatchEvent) => void>();
  /** What the last `write` handed to the browser — the note's evidence, and the page's. */
  lastDownload: Download | null = null;

  private constructor(
    private readonly folder: string,
    entries: readonly SnapshotFile[],
    private readonly deliver: boolean,
  ) {
    this.files = new Map(entries.map((one) => [one.path, one.file]));
  }

  root(): WorkspaceRef {
    return { kind: 'snapshot', name: this.folder, writable: false };
  }

  list(dir: Path): Promise<Entry[]> {
    const prefix = normalise(dir);
    const seen = new Map<string, Entry>();
    for (const path of this.files.keys()) {
      if (prefix !== '' && !path.startsWith(`${prefix}/`)) continue;
      const rest = prefix === '' ? path : path.slice(prefix.length + 1);
      const head = rest.split('/')[0] ?? '';
      if (head === '') continue;
      seen.set(head, { name: head, path: join(prefix, head), kind: rest.includes('/') ? 'directory' : 'file' });
    }
    if (seen.size === 0 && prefix !== '' && !this.files.has(prefix)) {
      return Promise.reject(new WorkspaceError(`no directory ${prefix} in the snapshot`, 'not-found'));
    }
    return Promise.resolve([...seen.values()].sort(byName));
  }

  async read(path: Path): Promise<{ text: string; revision: string }> {
    const file = this.fileAt(path);
    return { text: await file.text(), revision: revisionOf(file) };
  }

  /**
   * Save, on a snapshot: the browser is handed the bytes as a download and the file on disk is
   * left alone, so the revision is the snapshot's own — unchanged, since nothing moved. A path
   * the snapshot never held can still be saved: it is a new file the user will put somewhere.
   */
  write(path: Path, text: string, expect?: string): Promise<{ revision: string }> {
    const current = this.files.get(normalise(path));
    const revision = current === undefined ? ABSENT : revisionOf(current);
    if (expect !== undefined && expect !== revision) {
      return Promise.reject(
        new WorkspaceError(`${path} is not at ${expect} in this snapshot`, 'conflict', { found: revision, expected: expect }),
      );
    }
    const blob = new Blob([text], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = nameOf(path);
    anchor.rel = 'noopener';
    anchor.style.display = 'none';
    if (this.deliver) {
      document.body.append(anchor);
      anchor.click();
      anchor.remove();
    }
    // The object URL has to outlive the browser's own fetch of it; a minute is generous, and
    // the page keeps the last one so a caller can read back exactly what was offered.
    setTimeout(() => URL.revokeObjectURL(url), 60_000);
    this.lastDownload = { name: anchor.download, path: normalise(path), url, bytes: blob.size };
    return Promise.resolve({ revision });
  }

  /** There is no folder to create in: a snapshot is a copy the page was handed. */
  mkdir(path: Path): Promise<void> {
    return Promise.reject(
      new WorkspaceError(
        `this workspace is a read-only snapshot: ${normalise(path)} cannot be created. Open the folder in a browser with the writable picker, or create it on disk.`,
        'read-only',
      ),
    );
  }

  /**
   * A watcher is registered and never called: nothing can change under a snapshot, because it
   * is a copy the page was handed. The editor above does not have to know which workspace it
   * holds — it watches, and this one has nothing to report.
   */
  watch(path: Path, callback: (event: WatchEvent) => void): Unsubscribe {
    normalise(path);
    this.watchers.add(callback);
    return () => {
      this.watchers.delete(callback);
    };
  }

  resolve(from: Path, relative: string): Path {
    return resolveFrom(from, relative);
  }

  /** Every file of the snapshot, deepest last — what the explorer walks. */
  paths(): Path[] {
    return [...this.files.keys()].sort((a, b) => a.localeCompare(b));
  }

  private fileAt(path: Path): File {
    const file = this.files.get(normalise(path));
    if (file === undefined) {
      const where = parentOf(path);
      throw new WorkspaceError(`no file ${normalise(path)} in the snapshot of ${this.folder}/${where}`, 'not-found');
    }
    return file;
  }
}
