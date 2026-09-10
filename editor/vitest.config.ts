import { defineConfig } from 'vitest/config';

// The test layers of the implementation plan §0.3, one Vitest project each, so that
// `pnpm check` can run them separately and name the one that failed:
//
//   unit      packages/*/test/**          `pnpm test:unit`
//   parity    packages/lang/test/parity/  `pnpm test:parity`   (reads the oracle of §0.5)
//   snapshot  packages/ui/test/snapshots/ `pnpm test:snapshot`
//   audit     tests/audit/**              `pnpm audit:rules`
//
// The browser layer is Playwright, configured in apps/web (`pnpm test:e2e`).
export default defineConfig({
  test: {
    projects: [
      {
        test: {
          name: 'lang',
          root: './packages/lang',
          environment: 'node',
          include: ['test/**/*.test.ts'],
          exclude: ['test/parity/**'],
        },
      },
      {
        test: {
          name: 'store',
          root: './packages/store',
          environment: 'node',
          include: ['test/**/*.test.ts'],
        },
      },
      {
        test: {
          name: 'ui',
          root: './packages/ui',
          environment: 'node',
          include: ['test/**/*.test.ts'],
          exclude: ['test/snapshots/**'],
        },
      },
      {
        test: {
          name: 'parity',
          root: './packages/lang',
          environment: 'node',
          include: ['test/parity/**/*.test.ts'],
        },
      },
      {
        test: {
          name: 'snapshot',
          root: './packages/ui',
          environment: 'node',
          include: ['test/snapshots/**/*.test.ts'],
        },
      },
      {
        test: {
          name: 'audit',
          root: './tests/audit',
          environment: 'node',
          include: ['**/*.test.ts'],
        },
      },
    ],
  },
});
