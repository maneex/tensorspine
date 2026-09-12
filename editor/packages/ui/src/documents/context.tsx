/**
 * How a component reaches the open documents.
 *
 * A third context beside the shell's two, and one a page may not have: the shell renders without
 * it, which is what feature 2.5 left it doing — the bar's pills and eight of the status bar's ten
 * fields are a document's, and with no document they are not shown at all rather than shown empty.
 *
 * **The absence is a store and not a `null` check in every reader.** A hook is not conditional, so
 * a component that subscribed only where a provider exists would be a component React refuses to
 * re-render honestly. {@link NO_DOCUMENTS} is a real store holding the state of an editor with
 * nothing open, whose gestures do nothing; a reader written once then reads in both cases.
 */
import { createContext, useContext, type JSX, type ReactNode } from 'react';
import { createStore } from 'zustand';
import { useStore } from 'zustand';

import type { Documents, DocumentsStore } from './store.js';

/** The state of an editor with no workspace and no document — S17's, as data. */
function nothingOpen(): Documents {
  const nothing = (): void => undefined;
  const later = (): Promise<void> => Promise.resolve();
  return {
    workspace: { kind: 'empty', id: '', name: '', writable: false },
    open: [],
    current: null,
    library: {
      loading: false,
      handle: null,
      problems: [],
      files: 0,
      ms: 0,
      templates: new Set(),
      versions: new Map(),
      arguments: new Map(),
    },
    banner: null,
    toast: null,
    dialog: null,
    documents: [],
    notices: [],
    filter: '',
    registry: null,
    start: later,
    adopt: later,
    openFolder: later,
    openExamples: later,
    openRecent: later,
    openUpload: later,
    dropped: later,
    forget: later,
    openDocument: later,
    newModel: later,
    select: nothing,
    selectPlace: nothing,
    selectPlaces: nothing,
    drillInto: nothing,
    setScrub: nothing,
    revealPlace: nothing,
    lint: nothing,
    loadArgumentSchema: nothing,
    togglePlace: nothing,
    toggleGroup: nothing,
    moveBox: nothing,
    resetLayout: nothing,
    setViewport: nothing,
    setFilter: nothing,
    checkEdge: () => Promise.resolve(null),
    checkMember: () => Promise.resolve(null),
    partnersOf: () => Promise.resolve(null),
    identityFacts: () => Promise.resolve(null),
    edit: () => null,
    undo: nothing,
    redo: nothing,
    addDeclaration: () => null,
    expose: () => null,
    offer: nothing,
    confirmed: nothing,
    showSource: nothing,
    mayClose: () => Promise.resolve(true),
    closeDocument: nothing,
    save: later,
    saveAs: later,
    saveAll: later,
    revert: later,
    downloadZip: later,
    autosave: later,
    restoreDraft: nothing,
    discardDraft: later,
    validateNow: nothing,
    setDialog: nothing,
    setToast: nothing,
    note: nothing,
    stop: nothing,
  };
}

/** The store a page with no documents reads: the state above, and nothing ever changes it. */
export const NO_DOCUMENTS: DocumentsStore = createStore<Documents>()(() => nothingOpen());

const DocumentsContext = createContext<DocumentsStore | null>(null);

/** Put the documents in reach of everything below — the application does it around the shell. */
export function DocumentsProvider({
  store,
  children,
}: {
  store: DocumentsStore;
  children: ReactNode;
}): JSX.Element {
  return <DocumentsContext.Provider value={store}>{children}</DocumentsContext.Provider>;
}

/** The store itself, for a gesture that reads the state at the moment it fires. */
export function useDocumentsStore(): DocumentsStore {
  return useContext(DocumentsContext) ?? NO_DOCUMENTS;
}

/** One value of the documents' state, subscribed to on its own. */
export function useDocuments<T>(selector: (state: Documents) => T): T {
  return useStore(useDocumentsStore(), selector);
}
