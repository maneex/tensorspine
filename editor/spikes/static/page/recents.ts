/**
 * Recent workspaces — feature 0.6.
 *
 * D11: "a directory picked by the user is read and written in place, its handle kept in
 * IndexedDB so the workspace reopens after a reload with one permission prompt". So a recent
 * workspace is a **handle**, not a path: the page cannot turn a path back into a folder, and
 * the browser will not hand one over without a gesture. What is stored is the handle itself —
 * IndexedDB serialises it — and what is asked on the way back is the permission, once.
 *
 * `queryPermission` and `requestPermission` are the File System Access API's own, and no
 * `lib.dom` declares them; where they are absent (an engine that has handles but not the
 * permission model — the Origin Private File System is the case the tests run in) a handle is
 * usable as it is, and this module says so rather than pretending a grant was asked for.
 */

/** The database and store the static application keeps its handles in. */
const DATABASE = 'tensorspine-editor';
const STORE = 'workspaces';

/** A workspace the user opened before: the handle, and enough to name it in the File menu. */
export interface RecentWorkspace {
  readonly id: string;
  readonly name: string;
  /** When it was last opened, ISO 8601. */
  readonly openedAt: string;
  readonly handle: FileSystemDirectoryHandle;
}

/** The two calls of the File System Access API's permission model, where the engine has them. */
interface HandlePermissions {
  queryPermission?: (descriptor?: { mode?: 'read' | 'readwrite' }) => Promise<PermissionState>;
  requestPermission?: (descriptor?: { mode?: 'read' | 'readwrite' }) => Promise<PermissionState>;
}

/** What a permission query answered, or that the engine has no such query. */
export type Permission = PermissionState | 'unsupported';

/** A promise over one IndexedDB request. */
function request<T>(operation: IDBRequest<T>): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    operation.onsuccess = () => {
      resolve(operation.result);
    };
    operation.onerror = () => {
      reject(operation.error ?? new Error('the request failed'));
    };
  });
}

function openDatabase(): Promise<IDBDatabase> {
  return new Promise<IDBDatabase>((resolve, reject) => {
    const opening = indexedDB.open(DATABASE, 1);
    opening.onupgradeneeded = () => {
      if (!opening.result.objectStoreNames.contains(STORE)) opening.result.createObjectStore(STORE, { keyPath: 'id' });
    };
    opening.onsuccess = () => {
      resolve(opening.result);
    };
    opening.onerror = () => {
      reject(opening.error ?? new Error('IndexedDB refused to open'));
    };
  });
}

async function withStore<T>(mode: IDBTransactionMode, body: (store: IDBObjectStore) => Promise<T>): Promise<T> {
  const database = await openDatabase();
  try {
    const transaction = database.transaction(STORE, mode);
    const done = new Promise<void>((resolve, reject) => {
      transaction.oncomplete = () => {
        resolve();
      };
      transaction.onerror = () => {
        reject(transaction.error ?? new Error('the transaction failed'));
      };
      transaction.onabort = () => {
        reject(transaction.error ?? new Error('the transaction was aborted'));
      };
    });
    const answer = await body(transaction.objectStore(STORE));
    if (mode !== 'readonly') await done;
    return answer;
  } finally {
    database.close();
  }
}

/** The recent workspaces, most recently opened first. */
export async function recentWorkspaces(): Promise<RecentWorkspace[]> {
  const stored = await withStore('readonly', (store) => request<RecentWorkspace[]>(store.getAll() as IDBRequest<RecentWorkspace[]>));
  return stored
    .filter((one) => typeof one.id === 'string' && typeof one.name === 'string')
    .sort((a, b) => b.openedAt.localeCompare(a.openedAt));
}

/**
 * Remember a folder. A folder already in the list keeps its id — `isSameEntry` is what says
 * "the same folder", since two handles to one directory are not the same object.
 */
export async function rememberWorkspace(handle: FileSystemDirectoryHandle): Promise<RecentWorkspace> {
  const known = await recentWorkspaces();
  let id = `w${String(Date.now())}${String(Math.floor(Math.random() * 1e6))}`;
  for (const one of known) {
    if (await one.handle.isSameEntry(handle)) {
      id = one.id;
      break;
    }
  }
  const entry: RecentWorkspace = { id, name: handle.name, openedAt: new Date().toISOString(), handle };
  await withStore('readwrite', async (store) => {
    await request(store.put(entry));
  });
  return entry;
}

/** Forget every recent workspace — the settings command, and what a test starts from. */
export async function forgetWorkspaces(): Promise<void> {
  await withStore('readwrite', async (store) => {
    await request(store.clear());
  });
}

/** What the browser says about writing in this folder, without asking the user anything. */
export async function permissionOf(handle: FileSystemDirectoryHandle, mode: 'read' | 'readwrite' = 'readwrite'): Promise<Permission> {
  const query = (handle as unknown as HandlePermissions).queryPermission;
  if (query === undefined) return 'unsupported';
  return query.call(handle, { mode });
}

/**
 * Reopen a remembered folder: query first, and ask **once** if the answer is not already yes.
 * The request needs the user's gesture, which is the click on the recent entry — the one
 * prompt D11 allows, and never a second picker.
 */
export async function grantPermission(
  handle: FileSystemDirectoryHandle,
  mode: 'read' | 'readwrite' = 'readwrite',
): Promise<{ granted: boolean; asked: boolean; state: Permission }> {
  const state = await permissionOf(handle, mode);
  if (state === 'granted' || state === 'unsupported') return { granted: true, asked: false, state };
  const ask = (handle as unknown as HandlePermissions).requestPermission;
  if (ask === undefined) return { granted: false, asked: false, state };
  const answer = await ask.call(handle, { mode });
  return { granted: answer === 'granted', asked: true, state: answer };
}
