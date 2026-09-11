import { flushSync } from 'react-dom';
import { createRoot, type Root } from 'react-dom/client';

import type { Platform } from '@tensorspine/store/platform';
import { createShell, documentationBase, Shell, type ShellStore } from '@tensorspine/ui/shell';

import '@tensorspine/ui/style.css';

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
 * **Where the documentation is.** The Help menu's links are resolved against the directory above
 * this page: the editor is deployed *beside* the documentation site, not inside it (D11), so
 * `…/tensorspine/editor/` finds the site at `…/tensorspine/`. Feature 2.18 settles the published
 * base path; until then the page reads its own address rather than carrying a host.
 */
export interface Application {
  readonly store: ShellStore;
  /** Take the page down — a suite's, and whatever replaces this page later. */
  readonly stop: () => void;
}

export function start(platform: Platform): Application {
  const root = document.querySelector('#root');
  if (!(root instanceof HTMLElement)) {
    throw new Error('the page has no #root to render into');
  }
  const { store, dispose } = createShell({
    platform,
    docsBase: documentationBase(window.location.href),
  });
  const react: Root = createRoot(root);
  // Flushed, so that the attributes below are true when they are written: `render` commits when
  // React chooses to, and a page that said `ready` before it was would be a page the browser
  // layer waits on for the wrong thing.
  flushSync(() => {
    react.render(<Shell store={store} platform={platform} />);
  });

  const describe = (): void => {
    root.dataset['workspace'] = store.getState().workspace.kind;
  };
  const watching = store.subscribe(describe);
  describe();
  root.dataset['platform'] = platform.describe();
  root.dataset['state'] = 'ready';

  return {
    store,
    stop: () => {
      watching();
      react.unmount();
      dispose();
    },
  };
}
