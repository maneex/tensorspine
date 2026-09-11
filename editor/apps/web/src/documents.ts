/**
 * The File menu, wired — feature 2.6's half of the application.
 *
 * The shell (2.5) registers §4.4's ninety commands and performs the ones it can; this binds the
 * fifteen of the File menu that need a workspace, and gives the shell the view a document's tab
 * draws with. Nothing of the language is here: the store of `@tensorspine/ui/documents` holds the
 * documents and the core answers about them.
 */
import type { Platform } from '@tensorspine/store/platform';
import type { Lang } from '@tensorspine/lang/api';
import {
  createDocuments,
  type DocumentsStore,
  type TabSink,
} from '@tensorspine/ui/documents';
import { selectionHandlers } from '@tensorspine/ui/explorer';
import type { ShellStore } from '@tensorspine/ui/shell';

/** What {@link wireDocuments} is given. */
export interface DocumentsWiring {
  readonly platform: Platform;
  readonly lang: Lang;
  readonly shell: ShellStore;
  /** The schema files the build vendored — read by whoever knows where they are served from. */
  readonly vendoredSchemas: () => Promise<Readonly<Record<string, string>>>;
  /** A suite's: how long the pipeline waits after an edit, and how often a draft is written. */
  readonly debounceMs?: number;
  readonly autosaveMs?: number;
}

/** The shell's tab strip, as the documents drive it. */
function tabsOf(shell: ShellStore): TabSink {
  return {
    open: (tab) => {
      shell.getState().openTab(tab);
    },
    update: (id, patch) => {
      shell.getState().updateTab(id, patch);
    },
    close: (id) => {
      shell.getState().closeTab(id);
    },
    select: (id) => {
      shell.getState().selectTab(id);
    },
    current: () => shell.getState().activeTab,
  };
}

/** Build the documents over a platform and a core, and bind the commands they answer. */
export function wireDocuments(wiring: DocumentsWiring): {
  store: DocumentsStore;
  dispose: () => void;
} {
  const { platform, lang, shell } = wiring;
  const built = createDocuments({
    platform,
    lang,
    vendoredSchemas: wiring.vendoredSchemas,
    log: (line) => {
      shell.getState().note(line);
    },
    tabs: tabsOf(shell),
    ...(wiring.debounceMs === undefined ? {} : { debounceMs: wiring.debounceMs }),
    ...(wiring.autosaveMs === undefined ? {} : { autosaveMs: wiring.autosaveMs }),
  });
  const documents = built.store;
  const now = () => documents.getState();

  /** The document a command with no argument is about — the tab the strip has current. */
  const current = (): string | null => shell.getState().activeTab;

  /** §4.4's `Rename` and `Delete`, which act on the selection the explorer and the canvas share. */
  const edit = selectionHandlers(documents, () => {
    shell.getState().revealPanel('panel.properties');
  });

  shell.getState().bind({
    'file.new-model': () => now().newModel(),
    'file.new-template': () => now().newModel(true),
    'file.open-folder': () => now().openFolder(),
    // §4.4 writes "Open Model… (a single file)"; what a static page can offer is the documents
    // of the workspace it has open, which is where a single file of one *is*. A file from
    // anywhere else reaches the editor as a drop, which the window already takes (§4.3, D11).
    // The Model explorer of §4.5 (feature 2.7) is what replaces this list.
    'file.open-model': () => {
      now().setDialog({ kind: 'documents' });
    },
    'file.open-recent': async () => {
      now().setDialog({ kind: 'workspaces', recent: await platform.workspaces.recent() });
    },
    'file.save': () => now().save(),
    'file.save-as': () => {
      const id = current();
      const one = now().open.find((open) => open.id === id);
      if (one === undefined) return;
      now().setDialog({ kind: 'save-as', id: one.id, path: one.path });
    },
    'file.save-all': () => now().saveAll(),
    'file.download-zip': () => now().downloadZip(),
    'file.revert': () => now().revert(),
    'file.close-tab': () => {
      const id = current();
      if (id !== null) shell.getState().closeTab(id);
    },
    'file.close-all': () => {
      shell.getState().closeAllTabs();
    },
    // The pipeline of §5.4 is one run: the semantic stage, then the derivation that follows a
    // passing one. `Validate Now` and `Derive Now` both ask for it without waiting for the
    // debounce; what the Derived panel makes of the products is feature 2.15's.
    'model.validate-now': () => {
      now().validateNow();
    },
    'model.derive-now': () => {
      now().validateNow();
    },
    // §4.4's Edit menu, on whatever is selected in the document (§4.5). The tree answers the
    // same two keys while it has the focus; these are what the menu and the palette reach.
    'edit.rename': edit.rename,
    'edit.delete': edit.remove,
  });

  // §4.3's "Close with unsaved changes asks": the strip is the shell's, the answer is the
  // document's, and this is the one seam between them.
  shell.getState().guardClose((tab) => now().mayClose(tab.id));

  // The strip is the shell's and the documents behind it are not, so the two are kept in step
  // here and in one direction only: a tab closed in the strip — by the ×, the middle click,
  // Ctrl+W or Close All — takes its document with it, and the tab the strip made current is the
  // document a command with no argument is about.
  const following = shell.subscribe((state, before) => {
    if (state.tabs !== before.tabs) {
      const held = new Set(state.tabs.map((tab) => tab.id));
      for (const one of now().open) if (!held.has(one.id)) now().closeDocument(one.id);
    }
    if (state.activeTab !== before.activeTab && state.activeTab !== null) {
      now().select(state.activeTab);
    }
  });

  return {
    store: documents,
    dispose: () => {
      following();
      built.dispose();
    },
  };
}
