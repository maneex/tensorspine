/**
 * Recent workspaces — feature 2.4, promoted from feature 0.6's spike.
 *
 * D11: "a directory picked by the user is read and written in place, its handle kept in IndexedDB
 * so the workspace reopens after a reload with one permission prompt". A recent workspace is
 * therefore a **handle** and not a path: a page cannot turn a path back into a folder, and the
 * browser will not hand one over without a gesture. What is stored is the handle itself —
 * IndexedDB serialises it — and what is asked on the way back is the permission, once.
 *
 * `queryPermission` and `requestPermission` are the File System Access API's own and no `lib.dom`
 * declares them. **Firefox has neither** (feature 0.6 measured it) while carrying handles,
 * `createWritable` and persistence, so a missing permission query means *usable* here and never
 * *denied*: the day Firefox ships a directory picker, the editor gains a writable workspace there
 * with no change beyond the line that asks for a folder.
 */
import type { RecentWorkspace } from '@tensorspine/store/platform';

import { memoryStore, WORKSPACES, openStore, type Keyed, type Store } from './idb.js';

/** A remembered folder, as the store holds it: the handle, and enough to name it in the menu. */
export interface RememberedWorkspace extends Keyed, RecentWorkspace {
  readonly handle: FileSystemDirectoryHandle;
}

/** The two calls of the File System Access API's permission model, where the engine has them. */
interface HandlePermissions {
  queryPermission?: (descriptor?: { mode?: 'readwrite' }) => Promise<PermissionState>;
  requestPermission?: (descriptor?: { mode?: 'readwrite' }) => Promise<PermissionState>;
}

/** What a permission query answered, or that the engine has no such query. */
export type Permission = PermissionState | 'unsupported';

/** What the editor asks a folder for: it opens one to edit it. */
const MODE = { mode: 'readwrite' } as const;

/** The folders the user opened before, and the permission dance on the way back. */
export class Recents {
  /** Open the store; the recents fall back to this session's memory when IndexedDB refused. */
  static async open(): Promise<Recents> {
    const store = await openStore<RememberedWorkspace>(WORKSPACES);
    return new Recents(store ?? memoryStore<RememberedWorkspace>(), store !== null);
  }

  private constructor(
    private readonly store: Store<RememberedWorkspace>,
    /** False when IndexedDB refused: a remembered folder will not survive the reload. */
    readonly persistent: boolean,
  ) {}

  /** The remembered folders, most recently opened first. */
  async all(): Promise<RememberedWorkspace[]> {
    const stored = await this.store.all().catch(() => [] as RememberedWorkspace[]);
    return stored
      .filter((one) => typeof one.id === 'string' && typeof one.name === 'string')
      .sort((a, b) => b.openedAt.localeCompare(a.openedAt));
  }

  async get(id: string): Promise<RememberedWorkspace | null> {
    return this.store.get(id).catch(() => null);
  }

  /**
   * Remember a folder, and answer the entry it is held under.
   *
   * A folder already in the list keeps its id — `isSameEntry` is what says "the same folder",
   * since two handles to one directory are not the same object — which is what makes the id
   * something a draft or a per-workspace setting can be keyed by.
   */
  async remember(handle: FileSystemDirectoryHandle): Promise<RememberedWorkspace> {
    const known = await this.all();
    let id = `w${String(Date.now())}${String(Math.floor(Math.random() * 1e6))}`;
    for (const one of known) {
      if (await one.handle.isSameEntry(handle).catch(() => false)) {
        id = one.id;
        break;
      }
    }
    const entry: RememberedWorkspace = { id, name: handle.name, openedAt: new Date().toISOString(), handle };
    await this.store.put(entry).catch(() => undefined);
    return entry;
  }

  /** Forget one remembered folder, or all of them. */
  async forget(id?: string): Promise<void> {
    const done = id === undefined ? this.store.clear() : this.store.remove(id);
    await done.catch(() => undefined);
  }

  /** What the browser says about writing in this folder, without asking the user anything. */
  async permission(handle: FileSystemDirectoryHandle): Promise<Permission> {
    const query = (handle as unknown as HandlePermissions).queryPermission;
    if (query === undefined) return 'unsupported';
    return query.call(handle, MODE).catch(() => 'denied' as const);
  }

  /**
   * Reopen a remembered folder: query first, and ask **once** if the answer is not already yes.
   *
   * The request needs the user's gesture, which is the click on the recent entry — the one prompt
   * D11 allows, and never a second picker.
   */
  async grant(handle: FileSystemDirectoryHandle): Promise<{ granted: boolean; asked: boolean; state: Permission }> {
    const state = await this.permission(handle);
    if (state === 'granted' || state === 'unsupported') return { granted: true, asked: false, state };
    const ask = (handle as unknown as HandlePermissions).requestPermission;
    if (ask === undefined) return { granted: false, asked: false, state };
    const answer = await ask.call(handle, MODE).catch(() => 'denied' as const);
    return { granted: answer === 'granted', asked: true, state: answer };
  }
}
