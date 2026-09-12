import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

// Feature 0.4's lesson, in its fourth instance and as a test rather than a habit.
//
//   > **ELK must not reach the shell's first bundle**: re-exporting the layout from
//   > `packages/ui/src/index.ts` put 1.6 MB of compiled ELK into the static application's chunk
//   > though nothing imported it.
//
// Monaco is the largest third-party surface the editor carries — measured: 2 654 kB for the
// editor API and 1 208 kB for the JSON language, in chunks of their own — and it is paid for by
// whoever opens a JSON source view and by nobody else. Two rules keep it there, and both are read
// off the sources:
//
//   * **four modules name the package.** `src/source/monaco.ts`, its feature list and the two
//     worker entry points beside them; an `import type` counts, because a type import is a
//     specifier a later edit can make real without anybody noticing.
//   * **that module is reached with `await import(…)`.** A static `from './monaco.js'` anywhere
//     would put the editor in the chunk that draws the shell.

const here = dirname(fileURLToPath(import.meta.url));
const sourceDirectory = resolve(here, '..', '..', 'src', 'source');
const packages = resolve(here, '..', '..', '..');

/** The modules that may name `monaco-editor`: the loader, its feature list, and the two workers. */
const NAMING = ['monaco.ts', 'features.ts', 'json.worker.ts', 'editor.worker.ts'];

/** Every import specifier of a source, static or dynamic, with the comments left out. */
function specifiersOf(text: string): { specifier: string; dynamic: boolean }[] {
  const stripped = text.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\*.*$/gm, '').replace(/^\s*\/\/.*$/gm, '');
  const found: { specifier: string; dynamic: boolean }[] = [];
  for (const match of stripped.matchAll(/(from|import)\s*(\()?\s*['"]([^'"]+)['"]/g)) {
    found.push({ specifier: match[3] ?? '', dynamic: match[2] === '(' });
  }
  return found;
}

/** Every `.ts`/`.tsx` under a directory of a package, by its name. */
function sourcesIn(directory: string): { name: string; text: string }[] {
  return readdirSync(directory)
    .filter((name) => name.endsWith('.ts') || name.endsWith('.tsx'))
    .map((name) => ({ name, text: readFileSync(join(directory, name), 'utf8') }));
}

/** Every `.ts`/`.tsx` of every package's `src/`, as `<package>/src/<path>`. */
function everySource(): { path: string; text: string }[] {
  const found: { path: string; text: string }[] = [];
  const walk = (at: string, prefix: string): void => {
    for (const entry of readdirSync(at, { withFileTypes: true })) {
      const path = join(at, entry.name);
      if (entry.isDirectory()) walk(path, `${prefix}${entry.name}/`);
      else if (/\.tsx?$/.test(entry.name)) {
        found.push({ path: `${prefix}${entry.name}`, text: readFileSync(path, 'utf8') });
      }
    }
  };
  for (const name of readdirSync(packages, { withFileTypes: true })) {
    if (!name.isDirectory()) continue;
    const root = join(packages, name.name, 'src');
    try {
      walk(root, `${name.name}/src/`);
    } catch {
      // A package with no `src/` is no part of this rule.
    }
  }
  return found;
}

describe('the text editor stays out of the shell’s first chunk', () => {
  it('is named by four modules of the source view and by nothing else in the workspace', () => {
    const offenders = everySource()
      .flatMap((file) =>
        specifiersOf(file.text)
          .filter((one) => one.specifier.startsWith('monaco-editor'))
          .map(() => file.path),
      )
      .filter((path) => !NAMING.some((name) => path === `ui/src/source/${name}`));
    expect(offenders).toEqual([]);
  });

  it('names it in all four of them, so the rule is about something', () => {
    const naming = sourcesIn(sourceDirectory)
      .filter((file) => specifiersOf(file.text).some((one) => one.specifier.startsWith('monaco-editor')))
      .map((file) => file.name)
      .sort();
    expect(naming).toEqual([...NAMING].sort());
  });

  it('is reached only with `await import(…)`, never with a static import', () => {
    const statics = sourcesIn(sourceDirectory)
      .filter((file) => file.name !== 'monaco.ts' && file.name !== 'features.ts')
      .flatMap((file) =>
        specifiersOf(file.text)
          .filter((one) => /(^|\/)monaco\.js$/.test(one.specifier) && !one.dynamic)
          .map((one) => `${file.name}: ${one.specifier}`),
      );
    // `Source.tsx` imports the *types* from it, which `verbatimModuleSyntax` erases; that line is
    // `import type`, which the scan below tells apart.
    const typesOnly = readFileSync(join(sourceDirectory, 'Source.tsx'), 'utf8')
      .split('\n')
      .filter((line) => line.includes("from './monaco.js'"))
      .every((line) => line.trimStart().startsWith('import type'));
    expect(typesOnly).toBe(true);
    expect(statics.filter((one) => !one.startsWith('Source.tsx'))).toEqual([]);
  });

  it('is not reached from the shell, the canvas, the expanded view or the documents', () => {
    // The view is composed by the application (`apps/web/src/app.tsx`) from the package's own
    // `/source` entry point, so no other directory of the interface names it.
    const reaching = everySource()
      .filter((file) => file.path.startsWith('ui/src/') && !file.path.startsWith('ui/src/source/'))
      .flatMap((file) =>
        specifiersOf(file.text)
          .filter((one) => /source\/(monaco|Source)\.js$/.test(one.specifier))
          .map((one) => `${file.path}: ${one.specifier}`),
      );
    expect(reaching).toEqual([]);
  });
});
