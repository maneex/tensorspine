import { stubPlatform } from '@tensorspine/store/platform';

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
 */
start(stubPlatform());
