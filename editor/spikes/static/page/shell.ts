import { DirectoryWorkspace, hasDirectoryPicker, pickDirectory } from './directory.ts';
import { grantPermission, permissionOf, recentWorkspaces, rememberWorkspace, type RecentWorkspace } from './recents.ts';
import { SnapshotWorkspace, type DroppedItem } from './snapshot.ts';
import { nameOf, WorkspaceError, type Path, type Unsubscribe, type Workspace } from './workspace.ts';

/**
 * The shell of the static application — feature 0.6's spike, drawn from artboards S17 and S18.
 *
 * It is deliberately the smallest thing that exercises both workspace paths end to end: open a
 * folder, list what is in it, read a file, edit it, save it, and see an external change arrive.
 * No canvas, no core, no schema: this spike proves the *deployment* and the *workspace*, and
 * features 2.4 to 2.6 build the real chrome over the same interface.
 *
 * What it must say out loud, because D11 and S18 require it:
 *
 *   - which path is open — a folder written in place, or a read-only snapshot;
 *   - on the snapshot path, that Save downloads the file for the user to put back (a banner,
 *     never silent, and a button that says what it will actually do);
 *   - which folders are remembered, as handles, reopened with one permission prompt.
 */

/** The elements the page carries; the shell fails loudly rather than half-wiring itself. */
function element<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (found === null) throw new Error(`the page has no #${id}`);
  return found as T;
}

interface Open {
  readonly workspace: Workspace;
  path: Path | null;
  revision: string | null;
}

export class Shell {
  private open: Open | null = null;
  private unwatch: Unsubscribe | null = null;
  private readonly lines: string[] = [];

  private readonly parts = {
    body: document.body,
    empty: element('empty'),
    workspaceView: element('workspace'),
    banner: element('banner'),
    files: element('files'),
    editor: element<HTMLTextAreaElement>('editor'),
    save: element<HTMLButtonElement>('save'),
    openFolder: element<HTMLButtonElement>('open-folder'),
    upload: element<HTMLInputElement>('upload'),
    recents: element('recents'),
    status: element('status'),
    log: element('log'),
  };

  /** Wire the page up and show what there is to show before any folder is open. */
  async start(): Promise<void> {
    this.parts.openFolder.disabled = !hasDirectoryPicker();
    this.parts.openFolder.title = hasDirectoryPicker()
      ? 'Read and write a folder in place'
      : 'This browser has no writable directory picker — use the folder upload, which is a read-only snapshot';
    this.parts.openFolder.addEventListener('click', () => void this.openFolder());
    this.parts.upload.addEventListener('change', () => {
      void this.openUpload();
    });
    this.parts.save.addEventListener('click', () => void this.save());
    window.addEventListener('dragover', (event) => {
      event.preventDefault();
    });
    window.addEventListener('drop', (event) => {
      event.preventDefault();
      void this.openDrop(event);
    });
    await this.showRecents();
    await this.reopenGranted();
    this.render();
  }

  /**
   * D11: "its handle kept in IndexedDB so the workspace reopens after a reload with one
   * permission prompt". The prompt needs the user's gesture, so what happens without one is
   * this: the most recent folder is reopened when the browser still holds the grant, and is
   * offered by name — never re-picked — when it does not.
   */
  private async reopenGranted(): Promise<void> {
    const known = await recentWorkspaces().catch(() => [] as RecentWorkspace[]);
    const last = known[0];
    if (last === undefined) return;
    const state = await permissionOf(last.handle).catch(() => 'denied' as const);
    if (state !== 'granted' && state !== 'unsupported') return;
    this.say(`Reopened ${last.name} without a prompt — the grant is still held`);
    await this.adopt(DirectoryWorkspace.open(last.handle, { pollMs: 500 }));
  }

  /** The Open Folder command: the gesture a static page needs before it can read anything. */
  async openFolder(): Promise<void> {
    const handle = await pickDirectory().catch((error: unknown) => {
      // The user dismissing the picker is an `AbortError`, and not a failure of anything.
      this.say(`Open Folder: ${describe(error)}`);
      return null;
    });
    if (handle === null) return;
    await rememberWorkspace(handle);
    await this.adopt(DirectoryWorkspace.open(handle, { pollMs: 500 }));
    await this.showRecents();
  }

  /** Reopen a remembered folder: one permission prompt, never a second picker (S18). */
  async reopen(recent: RecentWorkspace): Promise<void> {
    const permission = await grantPermission(recent.handle);
    this.say(
      `Reopen ${recent.name}: permission ${permission.state}${permission.asked ? ' (asked once)' : ' (already granted)'}`,
    );
    if (!permission.granted) return;
    await rememberWorkspace(recent.handle);
    await this.adopt(DirectoryWorkspace.open(recent.handle, { pollMs: 500 }));
    await this.showRecents();
  }

  /** A folder upload: a read-only snapshot, where Save downloads (D11). */
  private async openUpload(): Promise<void> {
    const files = Array.from(this.parts.upload.files ?? []);
    if (files.length === 0) return;
    await this.adopt(SnapshotWorkspace.fromFiles(files));
  }

  /**
   * A folder dropped on the window. Where the engine gives a real handle for it the folder is
   * writable — the picker is not the only way in — and otherwise it is a snapshot.
   */
  private async openDrop(event: DragEvent): Promise<void> {
    const items = event.dataTransfer?.items;
    if (items === undefined || items.length === 0) return;
    // Everything the transfer holds is read **before** the first `await`. A `DataTransfer` is
    // disabled as soon as the synchronous part of the `drop` handler returns: after the await
    // below `items.length` is 0 and `webkitGetAsEntry()` answers null, so a Chromium drop of
    // anything that is not a directory handle — a single file, most obviously — adopted an empty
    // snapshot called "dropped folder". The entries themselves outlive the transfer, which is what
    // makes reading them first enough.
    const dropped: DroppedItem[] = [];
    for (let index = 0; index < items.length; index += 1) {
      const item = items[index];
      if (item === undefined) continue;
      dropped.push({ entry: item.webkitGetAsEntry(), file: item.getAsFile() });
    }
    const first = items[0] as (DataTransferItem & { getAsFileSystemHandle?: () => Promise<FileSystemHandle | null> }) | undefined;
    const handle = first?.getAsFileSystemHandle === undefined ? null : await first.getAsFileSystemHandle();
    if (handle !== null && handle !== undefined && handle.kind === 'directory') {
      const directory = handle as FileSystemDirectoryHandle;
      await rememberWorkspace(directory);
      await this.adopt(DirectoryWorkspace.open(directory, { pollMs: 500 }));
      await this.showRecents();
      return;
    }
    await this.adopt(await SnapshotWorkspace.fromDrop(dropped));
  }

  /**
   * Take a workspace, whichever kind it is: everything below this line is the same code.
   *
   * Two `adopt`s can be in flight at once — Open Folder clicked, then a recent clicked while the
   * first listing is still being read — so nothing here may outlive its own call. The watch is
   * therefore started *before* the first `await` and its handle stored with it, or the first
   * workspace's poll would still be running with nothing left holding its `unwatch`; and every
   * resumption asks whether this call is still the one that owns the page before it writes to it.
   */
  async adopt(workspace: Workspace): Promise<void> {
    this.unwatch?.();
    this.unwatch = null;
    const open: Open = { workspace, path: null, revision: null };
    this.open = open;
    const reference = workspace.root();
    this.say(`Opened ${reference.name} — ${reference.kind}, ${reference.writable ? 'read-write' : 'read-only'}`);
    if (reference.writable) {
      this.unwatch = workspace.watch('', (event) => {
        if (this.open !== open) return;
        this.say(`watch: ${event.kind} ${event.path}`);
        if (event.kind === 'removed') return;
        if (open.path === event.path && open.revision !== event.revision) {
          this.parts.body.dataset['external'] = 'changed';
          this.say(`${event.path} changed on disk since it was read — reload to see it`);
        }
      });
    }
    await this.showFiles();
    if (this.open !== open) return;
    this.render();
  }

  /** Open one file of the workspace in the editor. */
  async openFile(path: Path): Promise<void> {
    if (this.open === null) return;
    const { text, revision } = await this.open.workspace.read(path);
    this.open.path = path;
    this.open.revision = revision;
    this.parts.editor.value = text;
    delete this.parts.body.dataset['external'];
    this.render();
  }

  /**
   * Save. On a folder the browser can write, this writes the file in place, refusing when the
   * file has moved on since it was read; on a snapshot it downloads it. The button says which.
   */
  async save(): Promise<void> {
    if (this.open === null || this.open.path === null) return;
    const { workspace, path, revision } = this.open;
    try {
      const written = await workspace.write(path, this.parts.editor.value, revision ?? undefined);
      this.open.revision = written.revision;
      delete this.parts.body.dataset['external'];
      this.say(
        workspace.root().writable
          ? `Saved ${path} in place — revision ${written.revision}`
          : `Downloaded ${nameOf(path)} — the folder itself is untouched`,
      );
    } catch (error) {
      if (error instanceof WorkspaceError && error.reason === 'conflict') {
        this.parts.body.dataset['external'] = 'conflict';
        this.say(`Refused: ${error.message}`);
      } else {
        this.say(`Save: ${describe(error)}`);
      }
    }
    this.render();
  }

  /** The files of the open workspace, deepest last — the explorer, in one flat list. */
  private async showFiles(): Promise<void> {
    this.parts.files.replaceChildren();
    if (this.open === null) return;
    const workspace = this.open.workspace;
    const paths: Path[] = [];
    const walk = async (directory: Path): Promise<void> => {
      for (const entry of await workspace.list(directory)) {
        if (entry.kind === 'directory') await walk(entry.path);
        else paths.push(entry.path);
      }
    };
    await walk('');
    for (const path of paths) {
      const row = document.createElement('button');
      row.type = 'button';
      row.className = 'row';
      row.dataset['testid'] = 'file';
      row.dataset['path'] = path;
      row.textContent = path;
      row.addEventListener('click', () => void this.openFile(path));
      this.parts.files.append(row);
    }
  }

  /** The remembered folders, as handles (S18: never a path the page would have to re-pick). */
  private async showRecents(): Promise<void> {
    this.parts.recents.replaceChildren();
    const known = await recentWorkspaces().catch(() => [] as RecentWorkspace[]);
    for (const recent of known) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'recent';
      button.dataset['testid'] = 'recent';
      button.dataset['name'] = recent.name;
      button.textContent = recent.name;
      button.addEventListener('click', () => void this.reopen(recent));
      this.parts.recents.append(button);
    }
    this.parts.recents.dataset['count'] = String(known.length);
  }

  /** One line of the log — every refusal, every external change, in the page itself. */
  private say(line: string): void {
    this.lines.push(line);
    this.parts.log.textContent = this.lines.join('\n');
  }

  private render(): void {
    const reference = this.open?.workspace.root() ?? null;
    const path = this.open?.path ?? null;
    const body = this.parts.body.dataset;
    body['state'] = reference === null ? 'no-workspace' : 'ready';
    body['workspace'] = reference?.kind ?? 'none';
    body['writable'] = reference === null ? '' : String(reference.writable);
    this.parts.empty.hidden = reference !== null;
    this.parts.workspaceView.hidden = reference === null;
    this.parts.banner.hidden = reference === null || reference.writable;
    this.parts.save.disabled = path === null;
    this.parts.save.textContent =
      reference !== null && !reference.writable && path !== null ? `Save — download ${nameOf(path)}` : 'Save';
    this.parts.status.textContent =
      reference === null
        ? 'no workspace · schemas: vendored tensorspine/2.0'
        : [
            reference.name,
            reference.writable ? 'read-write' : 'read-only snapshot',
            path ?? 'no file open',
            this.open?.revision ?? '',
          ]
            .filter((part) => part !== '')
            .join(' · ');
  }
}

/** A refusal in the words whatever raised it gave it. */
function describe(error: unknown): string {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}
