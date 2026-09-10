import { defineConfig, devices } from '@playwright/test';

const inCI = process.env['CI'] !== undefined && process.env['CI'] !== '';

// The preview server is bound to the loopback address by name, not to `localhost`, which
// resolves to IPv6 alone on some machines and to both elsewhere.
const host = '127.0.0.1';
const port = 4173;

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
  webServer: {
    command: `vite preview --host ${host} --port ${String(port)} --strictPort`,
    url: `http://${host}:${String(port)}/`,
    reuseExistingServer: !inCI,
    stdout: 'ignore',
  },
});
