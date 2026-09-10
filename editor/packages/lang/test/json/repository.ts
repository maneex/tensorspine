import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * The repository the editor is part of (plan D14): the corpus, the reference base, the schemas
 * and the rejection fixtures are read from it at the source, never copied into the editor.
 */
export const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', '..', '..');

/** Every file under a directory of the repository, deepest last, as repository-relative paths. */
export function filesUnder(directory: string): string[] {
  const found: string[] = [];
  const walk = (current: string): void => {
    for (const entry of readdirSync(current, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
      const path = join(current, entry.name);
      if (entry.isDirectory()) walk(path);
      else found.push(relative(repositoryRoot, path));
    }
  };
  walk(resolve(repositoryRoot, directory));
  return found;
}

/**
 * Every file the editor loads and stores as an interchange document: the corpus of `data/models/`
 * — the fourteen models and the template — and the reference base of `data/primitive-library/`.
 */
export function interchangeFiles(): string[] {
  return [...filesUnder('data/models'), ...filesUnder('data/primitive-library')];
}

/** The bytes of a repository file, decoded as UTF-8, as `open(encoding='utf-8')` reads them. */
export function readRepositoryFile(path: string): string {
  return readFileSync(join(repositoryRoot, path), 'utf8');
}
