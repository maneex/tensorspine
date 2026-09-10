import { defineConfig, devices } from '@playwright/test';

const inCI = process.env['CI'] !== undefined && process.env['CI'] !== '';

// The preview server is bound to the loopback address by name, not to `localhost`, which
// resolves to IPv6 alone on some machines and to both elsewhere.
const host = '127.0.0.1';
const port = 4173;

// The spike of feature 0.5 is a static build of its own, served beside the application by
// `editor/spikes/headers/serve.ts` — the same page the cross-engine runner opens in Firefox and
// WebKitGTK, so what the suite holds to account here is what the note measured there.
const spikePort = 4174;
export const spikeUrl = `http://${host}:${String(spikePort)}/`;

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
  ],
});
