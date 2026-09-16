/**
 * Opening a workspace in a browser — feature 2.4, §4.3 and D11.
 *
 * Four ways in, and the order between them is the measured one (feature 0.6):
 *
 *   1. **Open Folder…** — `showDirectoryPicker`, which only Chromium has today. The folder is read
 *      and written in place and its handle is remembered, so it reopens with one prompt.
 *   2. **A folder dropped on the window** — where the engine gives a real handle for it
 *      (`getAsFileSystemHandle`, Chromium), the drop is the *same* writable workspace the picker
 *      gives; §4.3 describes a drop as a read-only snapshot, which is right for Firefox and Safari
 *      and unnecessarily poor for Chromium. Prefer the handle, fall back to the copy.
 *   3. **A folder upload** (`<input type="file" webkitdirectory>`) — the one File System capability
 *      all three engines have. A read-only snapshot, where Save downloads.
 *   4. **Examples** — the corpus and the reference base vendored with the build, always available.
 *
 * What a `drop` costs to read is the trap this class exists to contain: a `DataTransfer` is
 * disabled the moment the synchronous part of the handler returns, after which `items.length` is 0
 * and `webkitGetAsEntry()` answers null (the review repair `910539b`). So {@link
 * BrowserWorkspaces.openDrop} reads **everything** off the transfer before its first `await`, and
 * reads both the entries and the files, because an item gives one or the other: a dropped
 * *directory* is an entry and only an entry, and an engine without `webkitGetAsEntry` gives a file
 * and only a file.
 */
import {
  PlatformError,
  ReadOnlyWorkspace,
  normalise,
  publishedWorkspace,
  snapshotOf,
  textsOf,
  type Deliver,
  type PublishedSet,
  type RecentWorkspace,
  type Unsubscribe,
  type UploadedFile,
  type Workspace,
  type Workspaces,
} from '@tensorspine/store/platform';

import { DirectoryWorkspace, type DirectoryOptions } from './directory.js';
import { Vendor, VENDOR } from './examples.js';
import { Recents } from './recents.js';

/** `showDirectoryPicker`, which no `lib.dom` declares — the writable picker of D11. */
interface DirectoryPicker {
  showDirectoryPicker?: (options?: { mode?: 'readwrite'; id?: string }) => Promise<FileSystemDirectoryHandle>;
}

/** `DataTransferItem.getAsFileSystemHandle`, which Chromium alone has (feature 0.6). */
type HandleItem = DataTransferItem & { getAsFileSystemHandle?: () => Promise<FileSystemHandle | null> };

/** True where a folder can be opened for writing at all. */
export function hasDirectoryPicker(): boolean {
  return typeof (window as unknown as DirectoryPicker).showDirectoryPicker === 'function';
}

/** What {@link BrowserWorkspaces.create} is given. */
export interface BrowserWorkspacesOptions {
  /** How a read-only Save reaches the user — the shell's download. */
  readonly deliver: Deliver;
  /** Where the vendored material is served from; the application's base path by default. */
  readonly vendor?: string;
  /** How often a writable workspace polls for changes made behind it. */
  readonly pollMs?: number;
}

/** A file of a dropped directory, named by where it stood in the drop. */
function droppedFile(path: string, file: File): UploadedFile {
  return {
    name: file.name,
    webkitRelativePath: path,
    size: file.size,
    lastModified: file.lastModified,
    text: () => file.text(),
  };
}

/** Walk a dropped directory entry, which is the only reading a drop offers without handles. */
async function walkEntry(entry: FileSystemEntry, into: UploadedFile[]): Promise<void> {
  if (entry.isFile) {
    const file = await new Promise<File>((resolve, reject) => {
      (entry as FileSystemFileEntry).file(resolve, reject);
    });
    into.push(droppedFile(normalise(entry.fullPath), file));
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

export class BrowserWorkspaces implements Workspaces {
  static async create(options: BrowserWorkspacesOptions): Promise<BrowserWorkspaces> {
    return new BrowserWorkspaces(await Recents.open(), options);
  }

  private readonly listeners = new Set<(workspace: Workspace) => void>();
  private vendor: Vendor | null = null;
  private held: Workspace;

  private constructor(
    private readonly recents: Recents,
    private readonly options: BrowserWorkspacesOptions,
  ) {
    // Before anything is opened the editor holds a workspace all the same: it lists nothing and
    // finds nothing, and a Save through it is a download — which is the right answer for a
    // document made from nothing (§4.3's New Model, S17's empty state).
    this.held = new ReadOnlyWorkspace(textsOf('empty', 'unopened', 'no workspace', {}), options.deliver);
  }

  get writablePicker(): boolean {
    return hasDirectoryPicker();
  }

  /** Whether a remembered folder will still be there after a reload (IndexedDB was reachable). */
  get remembers(): boolean {
    return this.recents.persistent;
  }

  current(): Workspace {
    return this.held;
  }

  onChange(callback: (workspace: Workspace) => void): Unsubscribe {
    this.listeners.add(callback);
    return () => {
      this.listeners.delete(callback);
    };
  }

  /** The Open Folder… command. `null` when the user dismissed the picker. */
  async open(): Promise<Workspace | null> {
    const picker = (window as unknown as DirectoryPicker).showDirectoryPicker;
    if (picker === undefined) {
      throw new PlatformError(
        'this browser has no writable directory picker: open the folder as a read-only snapshot, ' +
          'or use a Chromium browser to edit it in place',
        'unsupported',
      );
    }
    let handle: FileSystemDirectoryHandle;
    try {
      handle = await picker({ mode: 'readwrite', id: 'tensorspine-workspace' });
    } catch (error) {
      // Dismissing the picker is an `AbortError`, and is not a failure of anything.
      if (error instanceof DOMException && error.name === 'AbortError') return null;
      throw new PlatformError(`the folder was not opened (${describe(error)})`, 'denied');
    }
    return this.adoptDirectory(handle);
  }

  /** A folder chosen through a folder upload: the read-only snapshot (§4.3). */
  openUpload(files: readonly UploadedFile[]): Promise<Workspace> {
    return Promise.resolve(this.adopt(new ReadOnlyWorkspace(snapshotOf(files), this.options.deliver)));
  }

  /**
   * A folder dropped on the window.
   *
   * Everything the transfer holds is read before the first `await`, including the call that asks
   * for a directory handle: the promise it answers outlives the transfer, the transfer itself does
   * not.
   */
  async openDrop(transfer: unknown): Promise<Workspace | null> {
    const items = (transfer as DataTransfer | null | undefined)?.items;
    if (items === undefined || items.length === 0) return null;
    const entries: FileSystemEntry[] = [];
    const files: UploadedFile[] = [];
    const asking: Promise<FileSystemHandle | null>[] = [];
    for (let index = 0; index < items.length; index += 1) {
      const item = items[index];
      if (item === undefined) continue;
      const entry = item.webkitGetAsEntry();
      if (entry !== null) entries.push(entry);
      else {
        const file = item.getAsFile();
        if (file !== null) files.push(droppedFile(file.name, file));
      }
      // Asked of every item, not only the first: a drop of a file and a folder should open the
      // folder. Called here, with the rest, because the promise outlives the transfer and the
      // call does not.
      const handled = item as HandleItem;
      if (handled.getAsFileSystemHandle !== undefined) {
        asking.push(handled.getAsFileSystemHandle().catch(() => null));
      }
    }

    // A workspace is one folder, so the first item that carries a directory handle is the one
    // opened and the others are left alone. `undefined`, not `null`, is what Chromium answers for
    // an item with no handle behind it — a synthesised transfer, a plain file — so the test is a
    // nullish one and not `!== null`.
    for (const handle of await Promise.all(asking)) {
      if (handle?.kind === 'directory') return this.adoptDirectory(handle as FileSystemDirectoryHandle);
    }
    for (const entry of entries) await walkEntry(entry, files);
    if (files.length === 0) return null;
    return this.adopt(new ReadOnlyWorkspace(snapshotOf(files), this.options.deliver));
  }

  /** The Examples workspace: the corpus and the reference base vendored with the build (D11). */
  async openExamples(): Promise<Workspace> {
    return this.adopt((await this.material()).workspace(this.options.deliver));
  }

  /**
   * A published file set opened as the workspace — feature 2.20.
   *
   * The fetching, the integrity check and the cache are `Platform.remote`'s; this is the adoption
   * alone, and the workspace it adopts is `ReadOnlyWorkspace` over the set's own `FileSet`, which
   * is the shape the snapshot and the Examples workspace already read.
   */
  openPublished(set: PublishedSet): Promise<Workspace> {
    return Promise.resolve(this.adopt(publishedWorkspace(set, this.options.deliver)));
  }

  /** The vendored material, read once per page: the examples, and the schemas beside them. */
  async material(): Promise<Vendor> {
    this.vendor ??= await Vendor.open(this.options.vendor ?? VENDOR);
    return this.vendor;
  }

  async recent(): Promise<readonly RecentWorkspace[]> {
    return (await this.recents.all()).map(({ id, name, openedAt }) => ({ id, name, openedAt }));
  }

  /** Reopen a remembered folder: one permission prompt, never a second picker (D11). */
  async reopen(id: string): Promise<Workspace | null> {
    const remembered = await this.recents.get(id);
    if (remembered === null) return null;
    const permission = await this.recents.grant(remembered.handle);
    if (!permission.granted) return null;
    return this.adoptDirectory(remembered.handle);
  }

  /**
   * Reopen the folder last used, if the browser still holds the grant.
   *
   * The prompt needs the user's gesture, so this is what happens without one: the folder comes
   * back when the grant is still held, and is offered by name — never re-picked — when it is not.
   */
  async reopenGranted(): Promise<Workspace | null> {
    const last = (await this.recents.all())[0];
    if (last === undefined) return null;
    const state = await this.recents.permission(last.handle);
    if (state !== 'granted' && state !== 'unsupported') return null;
    return this.adoptDirectory(last.handle);
  }

  async forget(id?: string): Promise<void> {
    await this.recents.forget(id);
  }

  /** Remember the folder, then open it: the id is the entry's, so a draft finds it again. */
  private async adoptDirectory(handle: FileSystemDirectoryHandle): Promise<Workspace> {
    const remembered = await this.recents.remember(handle);
    const options: DirectoryOptions = {
      id: remembered.id,
      ...(this.options.pollMs === undefined ? {} : { pollMs: this.options.pollMs }),
    };
    return this.adopt(DirectoryWorkspace.open(handle, options));
  }

  private adopt(workspace: Workspace): Workspace {
    this.held = workspace;
    for (const listener of this.listeners) listener(workspace);
    return workspace;
  }
}

/** A refusal in the words whatever raised it gave it. */
function describe(error: unknown): string {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}
