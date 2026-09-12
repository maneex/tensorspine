import { createBrowserPlatform } from './platform/index.js';
import { startLang } from './lang/connect.js';
import { start } from './app.js';

/**
 * The application's entry point, on the browser's platform (D11's first deployment).
 *
 * Three things of its own, which is the point: the page builds a `Platform`, starts the core in
 * its worker, and hands both to the renderer. `stub.ts` beside it does the same with the stub, and
 * that build is what proves the renderer reached for nothing else.
 *
 * **The drop handler is here and nowhere else, and it is synchronous.** A `DataTransfer` is
 * disabled the moment the synchronous part of a `drop` handler returns — so `openDrop` is called
 * *in the handler*, with nothing awaited before it, and the platform is built before the window
 * can be dropped on at all. Feature 2.4 measured both halves: a handler that awaited the
 * platform's construction first opened an empty snapshot.
 */
void (async () => {
  const platform = await createBrowserPlatform();
  const application = start(platform, {
    lang: startLang(),
    vendoredSchemas: async () => (await platform.workspaces.material()).readAll('schemas'),
    vendoredArguments: async (id) => (await platform.workspaces.material()).primitiveSchema(id),
  });

  // A drop is a way into a workspace, so the window takes one (§4.3, D11).
  window.addEventListener('dragover', (event) => {
    event.preventDefault();
  });
  window.addEventListener('drop', (event) => {
    event.preventDefault();
    // Read before the first await: the transfer is handed over as it stands.
    const opening = platform.workspaces.openDrop(event.dataTransfer);
    void application.documents.getState().dropped(opening);
  });

  // §4.3's "every 30 s **and on blur**": what the interval does not catch, a page losing the
  // focus does — and `pagehide` is the one event a browser guarantees before it takes the page.
  const autosave = (): void => {
    void application.documents.getState().autosave();
  };
  window.addEventListener('blur', autosave);
  window.addEventListener('pagehide', autosave);
})();
