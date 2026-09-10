import js from '@eslint/js';
import tseslint from 'typescript-eslint';

// Flat configuration for the whole workspace. Type-aware rules everywhere TypeScript is
// typechecked; the configuration files that are plain JavaScript are linted without types.
export default tseslint.config(
  {
    ignores: [
      '**/node_modules/**',
      '**/dist/**',
      // Feature 0.6's runner builds the page once per base path (`spikes/static/run.ts`).
      '**/dist-bases/**',
      '**/coverage/**',
      'tests/oracle/out/**',
      '**/test-results/**',
      '**/playwright-report/**',
    ],
  },
  js.configs.recommended,
  ...tseslint.configs.recommendedTypeChecked,
  {
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
  },
  {
    files: ['**/*.js'],
    ...tseslint.configs.disableTypeChecked,
  },
);
