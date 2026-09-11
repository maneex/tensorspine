import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { defineConfig, type UserConfig } from 'vite';

// The application is a static build (the plan's D11): no server, no runtime beyond the page.
// The base path under which it is published is decided with the deployment (feature 2.18).

const here = dirname(fileURLToPath(import.meta.url));

/**
 * The mode that builds the pages the browser layer needs beside the application.
 *
 * Two of them, and neither is part of what is deployed:
 *
 *   - `stub.html` — the application against the stub `Platform` (D11: "CI builds the web app
 *     against a stub platform from phase 2 on, so a platform leak is a build failure"). It is a
 *     build of the renderer with **no** platform implementation linked into it.
 *   - `e2e/page/platform.html` — the driver for the browser platform's own claims (feature 2.4),
 *     which are claims about a browser and can be asked nowhere else.
 *
 * `pnpm build` builds the application alone; `pnpm test:e2e` builds all three. A test holds both
 * lists, so a page cannot reach a deployment by accident.
 */
export const CHECK_MODE = 'check';

/** The pages a build emits, by the mode it was asked for. */
export function inputsFor(mode: string): Record<string, string> {
  const application = { main: resolve(here, 'index.html') };
  if (mode !== CHECK_MODE) return application;
  return {
    ...application,
    stub: resolve(here, 'stub.html'),
    platform: resolve(here, 'e2e/page/platform.html'),
  };
}

export default defineConfig(({ mode }): UserConfig => {
  return {
    build: {
      outDir: 'dist',
      emptyOutDir: true,
      rollupOptions: { input: inputsFor(mode) },
    },
  };
});
