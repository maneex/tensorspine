import { defineConfig } from 'vite';

// The application is a static build (the plan's D11): no server, no runtime beyond the page.
// The base path under which it is published is decided with the deployment (feature 2.18).
export default defineConfig({
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
});
