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
  documentTab,
  drillOf,
  type DocumentsStore,
  type TabSink,
} from '@tensorspine/ui/documents';
import { duplicateAt } from '@tensorspine/ui/canvas';
import { selectionHandlers } from '@tensorspine/ui/explorer';
import { presentation } from '@tensorspine/ui';
import type { ShellStore } from '@tensorspine/ui/shell';

/** What {@link wireDocuments} is given. */
export interface DocumentsWiring {
  readonly platform: Platform;
  readonly lang: Lang;
  readonly shell: ShellStore;
  /** The schema files the build vendored — read by whoever knows where they are served from. */
  readonly vendoredSchemas: () => Promise<Readonly<Record<string, string>>>;
  /** One primitive's generated argument schema, by identity — the same vendor (F5, §4.12). */
  readonly vendoredArguments?: (id: string) => Promise<string | null>;
  /** A suite's: how long the pipeline waits after an edit, and how often a draft is written. */
  readonly debounceMs?: number;
  readonly autosaveMs?: number;
}

/**
 * §4.4's four `Add …` commands, each adding to the map that declares the word its identity carries.
 *
 * `Add Quantity`, `Add Constant`, `Add Input` and `Add Output` differ in one word, and that word
 * is `presentation.json`'s `declares` — the same one a rename and a delete read (feature 2.2). So
 * the four handlers are one handler over the four identities, and the wiring names no member of
 * the grammar: which map holds the quantities is the schemas' and the bindings' answer, not this
 * file's.
 */
function adding(documents: DocumentsStore, shell: ShellStore): Record<string, () => void> {
  const handlers: Record<string, () => void> = {};
  for (const id of ADD_COMMANDS) {
    handlers[id] = (): void => {
      const declares = id.slice(id.lastIndexOf('-') + 1);
      const added = documents.getState().addDeclaration({ declares });
      if (added === null) return;
      shell.getState().revealPanel('panel.properties');
      documents.getState().note(`added ${declares} ${added}`);
    };
  }
  return handlers;
}

/** The commands of §4.4 that add a declaration to a map of them. */
const ADD_COMMANDS: readonly string[] = [
  'model.add-quantity',
  'model.add-constant',
  'model.add-input',
  'model.add-output',
];

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
    ...(wiring.vendoredArguments === undefined
      ? {}
      : { vendoredArguments: wiring.vendoredArguments }),
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
    // Feature 2.20's two: a published file set opened as the workspace, and a published base
    // stood at a path of the open one so that a document can pin it. Both ask for the address of
    // a **manifest** — plain HTTP has no directory listing — and both refuse with the address in
    // the message rather than spinning.
    'file.open-workspace-url': () => {
      now().setDialog({ kind: 'remote', target: 'workspace' });
    },
    'file.open-base-url': () => {
      now().setDialog({ kind: 'remote', target: 'library' });
    },
    'file.save': () => now().save(),
    'file.save-as': () => {
      // §4.4's own rule, the one feature 2.9 wrote down: "a command with no argument acts on the
      // document behind it (`documentTab` strips the suffix)". A tab that is a *view* of a
      // document — its JSON source, a drill-in, the expanded graph — carries a suffix, and the
      // command did nothing at all while one was current. Feature 2.19 met it on a build that
      // wires a composition and then saves.
      const id = current();
      const one = id === null ? undefined : now().open.find((open) => open.id === documentTab(id));
      if (one === undefined) {
        now().note('Save As…: there is no document open to save');
        return;
      }
      now().setDialog({ kind: 'save-as', id: one.id, path: one.path });
    },
    'file.save-all': () => now().saveAll(),
    'file.download-zip': () => now().downloadZip(),
    // §4.4's `Export Derived Document… (<model>.derived.json, validated against the derived
    // schema)`, which feature 2.6 registered and left to the feature that has the products.
    'file.export-derived': () => now().exportDerived(),
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
    // `Model ▸ Lint`: the advisories, over the workspace's own model documents. It is a command
    // and not part of the debounced run because `--lint`'s answer is a function of the set
    // (feature 1.10) and the set is the workspace — fourteen documents cost 781 ms, one `analyse`
    // each, against the 300 ms a keystroke's validation is given (§5.6). The rows land in
    // Problems, which is where the reader will be looking.
    'model.lint': () => {
      shell.getState().revealPanel('panel.problems');
      now().lint();
    },
    // §4.4's `Undo (Ctrl+Z, named)` and `Redo`, over the command log of D13. Feature 2.1 built
    // the log and feature 2.5 registered the commands; this is where the two meet, and what an
    // undo takes back is the *command* — a rename that rewrote ten references is one of them.
    'edit.undo': () => {
      now().undo();
    },
    'edit.redo': () => {
      now().redo();
    },
    // §4.4's Edit menu, on whatever is selected in the document (§4.5). The tree answers the
    // same two keys while it has the focus; these are what the menu and the palette reach.
    'edit.rename': edit.rename,
    'edit.delete': edit.remove,
    // §4.7's context menu is mirrored in the menu bar, so Duplicate acts on the same selection
    // the canvas and the explorer share: the declaration copied under a name nothing else has.
    'edit.duplicate': () => {
      const state = now();
      const one = state.open.find((open) => open.id === state.current);
      if (one === undefined || one.selection === undefined) {
        state.note('Duplicate: nothing is selected');
        return;
      }
      const context = {
        shapes: one.session.store.shapes,
        bindings: presentation(),
        role: one.session.store.role,
      };
      const path = one.selection;
      state.edit((made) => duplicateAt(context, made, path), one.id);
    },
    // §4.4's "JSON Source (Ctrl+Shift+J, opens beside)": §4.2's own tab, with the bytes a Save
    // would write. Feature 2.17 gives it Monaco and the schema.
    'view.json-source': () => {
      now().showSource();
    },
    // §4.4's `View ▸ Expanded Graph` (§4.9): the read-only tab over D1, which feature 2.16 draws.
    // `Layer Preview` is the same view restricted to one index value and is shown *in place* of a
    // drill-in's canvas, so its command is the drill-in's own (`Canvas.tsx` binds it while one is
    // open) and there is nothing for it here: a document with no drill-in open has no layer to
    // preview, and the menu entry says so in the Log rather than opening something else.
    'view.expanded-graph': () => {
      now().openExpanded();
    },
    'view.layer-preview': () => {
      const tab = current();
      const composition = tab === null ? null : drillOf(tab);
      if (tab === null || composition === null) {
        now().note('Layer Preview: open a composition’s drill-in first — §4.9 shows it there, in place of its canvas');
        return;
      }
      const id = documentTab(tab);
      const one = now().open.find((open) => open.id === id);
      now().setEmittedView({ preview: one?.emitted.preview === composition ? null : composition }, id);
    },
    // §4.4's `Model ▸ Add Instance… (opens the library picker at the cursor)` — bound at last
    // (feature 2.21). The picker is the catalog the core gathered, and where the instance lands is
    // §4.7's own answer: the root canvas, or the composition a drill-in has open. The **cursor**
    // belongs to a canvas, so a mounted one rebinds this with its own point and puts this back
    // when it goes; from the bar with nothing mounted the automatic layout places it (D6).
    'model.add-instance': () => {
      now().offerPrimitives();
    },
    // §4.4's `Document Properties` — "(the model id, `primitive_libraries`, `version`)", which is
    // §4.11's Document sheet: the sheet of the document's own place, which is the sheet shown
    // when nothing else is selected. The command is the way to it from the menu.
    'model.document-properties': () => {
      now().selectPlace([]);
      shell.getState().revealPanel('panel.properties');
    },
    ...adding(documents, shell),
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
