/**
 * The editor's IndexedDB, and what to do when there is none — feature 2.4.
 *
 * Two things the static application must remember between reloads, and both of them need
 * IndexedDB rather than `localStorage`:
 *
 *   - the **handle** of a folder the user opened (D11: "its handle kept in IndexedDB so the
 *     workspace reopens after a reload with one permission prompt"), which is a
 *     `FileSystemDirectoryHandle` — structured-cloneable and not stringifiable;
 *   - the **drafts** of §4.3, which are whole documents and have no business in a store meant for
 *     a few kilobytes of settings.
 *
 * **Storage can refuse, and not merely come back empty.** In a private window, with site data
 * cleared, or in a browser set to block it, `indexedDB` can be absent, `open` can fail, and a read
 * can throw. Every call here answers instead of raising, and {@link openStore} says whether the
 * database was reachable at all so the stores above can fall back to memory and the chrome can say
 * that nothing will be remembered.
 *
 * One open per operation, which is what a store touched a few times a minute can afford: a
 * session autosaving every thirty seconds pays one `open` per save, and the alternative — a
 * connection held for the life of the page — blocks the version change of the next release.
 *
 * A trap feature 0.6 measured and every later feature inherits: reading a
 * `FileSystemDirectoryHandle` back out of IndexedDB **kills the renderer** in Playwright's
 * Chromium when the browser context is the default ephemeral one — not an exception, a dead page.
 * A persistent profile is fine, which is what the browser layer uses. Nothing here can defend
 * against it; it is written down so that the next reader knows what they are looking at.
 */

/** The database the static application keeps its own material in. */
export const DATABASE = 'tensorspine-editor';

/** The store of remembered folders, keyed by the id the File menu offers them under. */
export const WORKSPACES = 'workspaces';

/** The store of autosaved documents, keyed by workspace and path (`draftKey`). */
export const DRAFTS = 'drafts';

const STORES = [WORKSPACES, DRAFTS] as const;

/** A promise over one IndexedDB request, refusing rather than raising. */
export function request<T>(operation: IDBRequest<T>): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    operation.onsuccess = () => {
      resolve(operation.result);
    };
    operation.onerror = () => {
      reject(operation.error ?? new Error('the request failed'));
    };
  });
}

/** Open the database, creating whatever store is missing. */
function openDatabase(): Promise<IDBDatabase> {
  return new Promise<IDBDatabase>((resolve, reject) => {
    // `indexedDB` is absent in some contexts and throws on access in others; either is a refusal.
    let opening: IDBOpenDBRequest;
    try {
      opening = indexedDB.open(DATABASE, 1);
    } catch (error) {
      reject(error instanceof Error ? error : new Error(String(error)));
      return;
    }
    opening.onupgradeneeded = () => {
      for (const store of STORES) {
        if (!opening.result.objectStoreNames.contains(store)) {
          opening.result.createObjectStore(store, { keyPath: 'id' });
        }
      }
    };
    opening.onsuccess = () => {
      resolve(opening.result);
    };
    opening.onerror = () => {
      reject(opening.error ?? new Error('IndexedDB refused to open'));
    };
    opening.onblocked = () => {
      reject(new Error('IndexedDB is blocked by another tab of the editor'));
    };
  });
}

/** One transaction over one store, opened and closed around the body. */
async function withStore<T>(
  store: string,
  mode: IDBTransactionMode,
  body: (store: IDBObjectStore) => Promise<T>,
): Promise<T> {
  const database = await openDatabase();
  try {
    const transaction = database.transaction(store, mode);
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
    const answer = await body(transaction.objectStore(store));
    if (mode !== 'readonly') await done;
    return answer;
  } finally {
    database.close();
  }
}

/** One record of a store: whatever it holds, under the id it is keyed by. */
export interface Keyed {
  readonly id: string;
}

/**
 * A store of records, or `null` where IndexedDB is unreachable.
 *
 * The probe is a real transaction rather than a feature test: a browser can carry the API and
 * refuse to use it, which is exactly the private-window case.
 */
export interface Store<T extends Keyed> {
  all(): Promise<T[]>;
  get(id: string): Promise<T | null>;
  put(record: T): Promise<void>;
  remove(id: string): Promise<void>;
  clear(): Promise<void>;
}

/** Open one store of the database, or answer `null` when it cannot be reached at all. */
export async function openStore<T extends Keyed>(name: string): Promise<Store<T> | null> {
  try {
    await withStore(name, 'readonly', (store) => request<number>(store.count()));
  } catch {
    return null;
  }
  return {
    all: () => withStore(name, 'readonly', (store) => request<T[]>(store.getAll() as IDBRequest<T[]>)),
    get: (id) =>
      withStore(name, 'readonly', async (store) => {
        const found = await request<T | undefined>(store.get(id) as IDBRequest<T | undefined>);
        return found ?? null;
      }),
    put: (record) =>
      withStore(name, 'readwrite', async (store) => {
        await request(store.put(record));
      }),
    remove: (id) =>
      withStore(name, 'readwrite', async (store) => {
        await request(store.delete(id));
      }),
    clear: () =>
      withStore(name, 'readwrite', async (store) => {
        await request(store.clear());
      }),
  };
}

/** A store held entirely in memory — what the editor falls back to where IndexedDB refused. */
export function memoryStore<T extends Keyed>(): Store<T> {
  const held = new Map<string, T>();
  return {
    all: () => Promise.resolve([...held.values()]),
    get: (id) => Promise.resolve(held.get(id) ?? null),
    put: (record) => {
      held.set(record.id, record);
      return Promise.resolve();
    },
    remove: (id) => {
      held.delete(id);
      return Promise.resolve();
    },
    clear: () => {
      held.clear();
      return Promise.resolve();
    },
  };
}
