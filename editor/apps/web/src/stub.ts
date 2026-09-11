import { stubPlatform, type Session } from '@tensorspine/store/platform';

import { start } from './app.js';

/**
 * The application against the stub `Platform` — D11's "CI builds the web app against a stub
 * platform from phase 2 on, so a platform leak is a build failure".
 *
 * The stub holds its files in memory and names no browser API at all (it is compiled in a package
 * without the DOM). So this page is the renderer with **no** platform implementation linked into
 * it: a component that reached around `Platform` for `localStorage`, a file handle or a picker
 * would either fail to compile or appear in this page's chunk, and the browser layer asserts that
 * it does not.
 *
 * **It reads two query parameters, and only this page does.** `stub.html` is emitted by
 * `vite build --mode check` and by nothing else — it is never deployed — so it is the right place
 * to put the two facts a suite cannot otherwise arrange:
 *
 *   - `?session=Perceval Lambert` gives the stub's `AuthProvider` a session, which is the other
 *     half of the inventory's "no avatar without a session": without it nothing shows, with it
 *     the bar ends in an avatar. `NoAuth` can only be asked the first half.
 *   - `?scheme=dark` makes the machine prefer dark, which is what `theme: system` resolves
 *     against (§4.21). A headless browser's own preference is the machine's, and a suite that
 *     asserted a theme would otherwise be asserting something about the machine it ran on.
 */
function ask(name: string): string | null {
  return new URL(window.location.href).searchParams.get(name);
}

const named = ask('session');
const session: Session | undefined =
  named === null || named === '' ? undefined : { id: named, name: named };
const scheme = ask('scheme');

const platform = stubPlatform(session === undefined ? {} : { session });
if (scheme === 'dark' || scheme === 'light') platform.preferScheme(scheme);

const application = start(platform);

/**
 * What the browser layer reads off this page.
 *
 * `stub.html` is emitted by `vite build --mode check` and by nothing else, so this global exists
 * only where a suite is looking at it. It is how two claims can be asked that have no mark on the
 * page: that §4.4's commands were offered to the platform's native-menu hook (a browser has none
 * and ignores them, and Electron will build its menu from exactly this), and what the shell's
 * store holds.
 */
declare global {
  interface Window {
    tensorspineStub: { platform: typeof platform; store: typeof application.store };
  }
}

window.tensorspineStub = { platform, store: application.store };
