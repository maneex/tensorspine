import { defineConfig, devices } from '@playwright/test';

const inCI = process.env['CI'] !== undefined && process.env['CI'] !== '';

// The preview server is bound to the loopback address by name, not to `localhost`, which
// resolves to IPv6 alone on some machines and to both elsewhere.
const host = '127.0.0.1';
const port = 4173;

// Feature 2.4's two extra pages of the application's own build (`vite build --mode check`): the
// renderer against the stub `Platform` — D11's leak build — and the driver the browser layer
// runs the browser platform's claims through. `pnpm build` emits neither.
export const stubUrl = `http://${host}:${String(port)}/stub.html`;
export const platformUrl = `http://${host}:${String(port)}/e2e/page/platform.html`;

// The spike of feature 0.5 is a static build of its own, served beside the application by
// `editor/spikes/headers/serve.ts` — the same page the cross-engine runner opens in Firefox and
// WebKitGTK, so what the suite holds to account here is what the note measured there.
const spikePort = 4174;
export const spikeUrl = `http://${host}:${String(spikePort)}/`;

// Feature 0.6's spike is the same idea one step further: a static build served **under a base
// path**, as GitHub Pages serves the documentation site's pages. Its server answers nothing
// outside that base, so a build that ignored it fails here instead of passing by accident.
const staticPort = 4175;
export const staticBase = '/editor/';
export const staticOrigin = `http://${host}:${String(staticPort)}`;
export const staticUrl = `${staticOrigin}${staticBase}`;

// Feature 1.11's page: the `Lang` API over a **real** `Worker`, which is the only place three of
// its claims can be asked — that a derivation does not block the page's thread, that a cancel
// message overtakes the work it cancels, and that the core reaches the worker's chunk and not the
// page's. The page imports the application's own `startLang`, so what Vite emits there is the
// worker the static build emits.
const langPort = 4176;
export const langUrl = `http://${host}:${String(langPort)}/`;

// The browser layer of the implementation plan's §0.3: headless Chromium against the built
// application, served by Vite's preview server as a static page.
export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: inCI,
  retries: inCI ? 1 : 0,
  reporter: 'list',
  use: {
    baseURL: `http://${host}:${String(port)}`,
    trace: 'on-first-retry',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: [
    {
      command: `vite preview --host ${host} --port ${String(port)} --strictPort`,
      url: `http://${host}:${String(port)}/`,
      reuseExistingServer: !inCI,
      stdout: 'ignore',
    },
    {
      command: `node --experimental-strip-types ../../spikes/headers/serve.ts --port ${String(spikePort)}`,
      url: spikeUrl,
      reuseExistingServer: !inCI,
      stdout: 'ignore',
      // The spike's page is built by Vite when the server starts.
      timeout: 120_000,
    },
    {
      command: `node --experimental-strip-types ../../spikes/static/serve.ts --port ${String(staticPort)} --base ${staticBase}`,
      url: staticUrl,
      reuseExistingServer: !inCI,
      stdout: 'ignore',
      timeout: 120_000,
    },
    {
      command: `node --experimental-strip-types ../../spikes/lang/serve.ts --port ${String(langPort)}`,
      url: langUrl,
      reuseExistingServer: !inCI,
      stdout: 'ignore',
      timeout: 120_000,
    },
  ],
});
