import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';

import type { LibrarySource } from '../../src/library/index.js';
import { repositoryRoot } from '../json/repository.js';

/**
 * A {@link LibrarySource} over the real filesystem, for the tests alone.
 *
 * `packages/lang` never imports `node:fs` — it runs in a Web Worker (§5.3) — so the loader reads
 * through the interface and this is what the tests put behind it. It is also the one reading that
 * makes the parity suites comparable with `tools/`: the tools read `data/primitive-library/` and
 * `tests/rejections/primitive-library/` from disk, and so does this.
 */
export function nodeSource(root?: string): LibrarySource {
  // A rooted source lets a test name its bases the way the tools' own runner names them —
  // `data/primitive-library`, `tests/rejections/primitive-library/unknown-axis` — so that every
  // refusal comes out with the repository-relative paths the oracle recorded.
  const at = (path: string): string => (root === undefined ? path : resolve(root, path));
  return {
    isDirectory: (path) => existsSync(at(path)) && statSync(at(path)).isDirectory(),
    isFile: (path) => existsSync(at(path)) && statSync(at(path)).isFile(),
    exists: (path) => existsSync(at(path)),
    find(directory) {
      // `glob.glob('<dir>/**/*.json', recursive=True)`: every depth, and no name beginning with a
      // dot, which is what `**` and `*` skip by default.
      const found: string[] = [];
      const walk = (current: string): void => {
        for (const entry of readdirSync(current, { withFileTypes: true })) {
          if (entry.name.startsWith('.')) continue;
          const path = join(current, entry.name);
          if (entry.isDirectory()) walk(path);
          else if (entry.name.endsWith('.json')) found.push(path);
        }
      };
      walk(at(directory));
      return found.map((path) => (root === undefined ? path : relative(root, path)));
    },
    read: (path) => readFileSync(at(path), 'utf8'),
  };
}

/** A path of the repository, as the tools' own runners name one: relative to the root. */
export function repositoryPath(...parts: readonly string[]): string {
  return resolve(repositoryRoot, ...parts);
}

/** The reference base of `data/primitive-library/`. */
export const REFERENCE_BASE = repositoryPath('data', 'primitive-library');

/** One rejection base of `tests/rejections/primitive-library/`. */
export function rejectionBase(name: string): string {
  return repositoryPath('tests', 'rejections', name);
}

/**
 * Several sources read as one: the first that holds a path answers for it, and `find` is the
 * union over those that hold the directory.
 *
 * What it is for: a base built in memory beside the repository's own reference base, so that a
 * rule the reference base cannot exercise — a second version of one primitive, a conflicting
 * identity — is tested against the real axes and precision roles.
 */
export function overlay(...sources: readonly LibrarySource[]): LibrarySource {
  return {
    isDirectory: (path) => sources.some((one) => one.isDirectory(path)),
    isFile: (path) => sources.some((one) => one.isFile(path)),
    exists: (path) => sources.some((one) => one.exists(path)),
    find: (directory) =>
      sources.filter((one) => one.isDirectory(directory)).flatMap((one) => one.find(directory)),
    read: (path) => (sources.find((one) => one.isFile(path)) ?? sources[0]!).read(path),
  };
}
