/**
 * Drafts in IndexedDB — feature 2.4, §5.2 and §4.3.
 *
 * "Autosave of drafts every 30 s and on blur to the platform's draft store; on reopening a
 * document with a newer draft the editor offers to restore it." A draft is a whole document, so
 * it belongs in IndexedDB and not in `localStorage`; and what it carries beside its text is the
 * revision the document was read at, which is what "newer" is decided against (feature 2.6).
 *
 * Where IndexedDB refused — a private window, site data blocked — the store keeps working with
 * nothing behind it: drafts live for the session and {@link DraftStore.persistent} is false.
 */
import {
  byNewest,
  draftKey,
  summaryOf,
  type Draft,
  type DraftStore,
  type DraftSummary,
  type WorkspacePath,
} from '@tensorspine/store/platform';

import { DRAFTS, memoryStore, openStore, type Keyed, type Store } from './idb.js';

/** A draft as the store holds it: the draft, under the key it is found by. */
interface StoredDraft extends Keyed, Draft {}

/** The draft store of the static application. */
export async function browserDrafts(): Promise<DraftStore> {
  const store = await openStore<StoredDraft>(DRAFTS);
  return draftsOver(store ?? memoryStore<StoredDraft>(), store !== null);
}

/** The store's behaviour, over whichever record store it was given. */
export function draftsOver(store: Store<StoredDraft>, persistent: boolean): DraftStore {
  const all = (): Promise<StoredDraft[]> => store.all().catch(() => [] as StoredDraft[]);
  return {
    put: async (draft: Draft) => {
      await store.put({ ...draft, id: draftKey(draft.workspace, draft.path) }).catch(() => undefined);
    },
    get: async (workspace: string, path: WorkspacePath) => store.get(draftKey(workspace, path)).catch(() => null),
    list: async (workspace?: string): Promise<DraftSummary[]> =>
      (await all())
        .filter((draft) => workspace === undefined || draft.workspace === workspace)
        .map(summaryOf)
        .sort(byNewest),
    remove: async (workspace: string, path: WorkspacePath) => {
      await store.remove(draftKey(workspace, path)).catch(() => undefined);
    },
    clear: async (workspace?: string) => {
      if (workspace === undefined) {
        await store.clear().catch(() => undefined);
        return;
      }
      for (const draft of await all()) {
        if (draft.workspace === workspace) await store.remove(draft.id).catch(() => undefined);
      }
    },
    persistent,
  };
}
