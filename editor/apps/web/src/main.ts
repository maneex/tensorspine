import { createBrowserPlatform } from './platform/index.js';
import { start } from './app.js';

/**
 * The application's entry point, on the browser's platform (D11's first deployment).
 *
 * Two lines of its own, which is the point: the page builds a `Platform` and hands it to the
 * renderer. `stub.ts` beside it does the same with the stub, and that build is what proves the
 * renderer reached for nothing else.
 */
void (async () => {
  start(await createBrowserPlatform());
})();
