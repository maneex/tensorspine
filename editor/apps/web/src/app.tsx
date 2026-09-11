import { flushSync } from 'react-dom';
import { createRoot, type Root } from 'react-dom/client';

import type { Platform } from '@tensorspine/store/platform';
import type { Lang } from '@tensorspine/lang/api';
import {
  DocumentDialogs,
  DocumentToast,
  DOCUMENT_VIEWS,
  DocumentsProvider,
  type DocumentsStore,
} from '@tensorspine/ui/documents';
import { createShell, documentationBase, Shell, type ShellStore } from '@tensorspine/ui/shell';

import '@tensorspine/ui/style.css';

import { wireDocuments } from './documents.js';

/**
 * The application, whichever platform it was built against.
 *
 * D11's seam — **one renderer, three deployments** — and it is still two lines of substance: the
 * page builds a `Platform`, this builds the shell over it. Nothing here names a browser storage
 * API, a file handle or a picker; everything platform-shaped arrives as {@link Platform}, so the
 * same code runs against the browser's (`main.ts`) and against the stub CI builds it with
 * (`stub.ts`), and a leak would be a build failure in the second.
 *
 * **The store is built here and not in the component.** It subscribes to the platform's session,
 * workspace and colour scheme; its life is the page's. It is also what feature 2.6 onwards binds
 * command handlers through, and what a suite reads to see what the shell did.
 *
 * **The core and the documents are built here too, and for the same reason.** The `Lang` is a
 * worker (§5.3) and the documents subscribe to it and to the platform; both live as long as the
 * page. What the page supplies is the two things only it knows: how to reach the core
 * ({@link Application.lang}) and where the vendored schemas are served from — the browser's page
 * reads them off the vendor manifest, and the stub's has none, which is what makes the leak build
 * a build of the renderer and nothing else.
 *
 * **Where the documentation is.** The Help menu's links are resolved against the directory above
 * this page: the editor is deployed *beside* the documentation site, not inside it (D11), so
 * `…/tensorspine/editor/` finds it at `…/tensorspine/`. Feature 2.18 settles the published base
 * path; until then the page reads its own address rather than carrying a host.
 */
export interface Application {
  readonly store: ShellStore;
  readonly documents: DocumentsStore;
  /** Take the page down — a suite's, and whatever replaces this page later. */
  readonly stop: () => void;
}

/** What {@link start} is given beside the platform. */
export interface StartOptions {
  /** The language core, reached however this page reaches it — a worker, or this thread. */
  readonly lang: Lang;
  /** The schema files the build vendored, read once when the first document needs them. */
  readonly vendoredSchemas: () => Promise<Readonly<Record<string, string>>>;
  /** A suite's: how long the pipeline waits after an edit, and how often a draft is written. */
  readonly debounceMs?: number;
  readonly autosaveMs?: number;
}

export function start(platform: Platform, options: StartOptions): Application {
  const root = document.querySelector('#root');
  if (!(root instanceof HTMLElement)) {
    throw new Error('the page has no #root to render into');
  }
  const { store, dispose } = createShell({
    platform,
    docsBase: documentationBase(window.location.href),
  });
  const documents = wireDocuments({
    platform,
    lang: options.lang,
    shell: store,
    vendoredSchemas: options.vendoredSchemas,
    ...(options.debounceMs === undefined ? {} : { debounceMs: options.debounceMs }),
    ...(options.autosaveMs === undefined ? {} : { autosaveMs: options.autosaveMs }),
  });
  const react: Root = createRoot(root);
  // Flushed, so that the attributes below are true when they are written: `render` commits when
  // React chooses to, and a page that said `ready` before it was would be a page the browser
  // layer waits on for the wrong thing.
  flushSync(() => {
    react.render(
      <DocumentsProvider store={documents.store}>
        <Shell store={store} platform={platform} views={DOCUMENT_VIEWS}>
          <DocumentDialogs />
          <DocumentToast />
        </Shell>
      </DocumentsProvider>,
    );
  });

  const describe = (): void => {
    root.dataset['workspace'] = store.getState().workspace.kind;
  };
  const watching = store.subscribe(describe);
  describe();
  root.dataset['platform'] = platform.describe();
  root.dataset['state'] = 'ready';

  // Whatever was open when the page was last here, and nothing else (D11, §4.3).
  void documents.store.getState().start().finally(() => {
    root.dataset['documents'] = 'ready';
  });

  return {
    store,
    documents: documents.store,
    stop: () => {
      watching();
      react.unmount();
      documents.dispose();
      dispose();
    },
  };
}
