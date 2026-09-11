import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

import { createLang } from '@tensorspine/lang/api/engine';
import type { Lang, LibraryBaseFiles, LibraryHandle, SchemasHandle } from '@tensorspine/lang/api';

import { readRepositoryFile, repositoryRoot, schemaFiles } from '../presentation/source.js';

export { readRepositoryFile, registry, repositoryRoot, schemaFiles } from '../presentation/source.js';

/**
 * What the documents' suites read: the repository's own material, handed over as **files**.
 *
 * The core opens nothing (§5.3), so these helpers do what `Platform` will do and nothing the core
 * is supposed to do itself — the same shape `packages/lang/test/api/source.ts` takes one package
 * along.
 */

/** Every `.json` of a repository directory, by its repository path. */
export function filesUnder(directory: string): Record<string, string> {
  const found: Record<string, string> = {};
  const walk = (at: string): void => {
    for (const entry of readdirSync(join(repositoryRoot, at), { withFileTypes: true })) {
      const path = `${at}/${entry.name}`;
      if (entry.isDirectory()) walk(path);
      else if (entry.name.endsWith('.json')) found[path] = readFileSync(join(repositoryRoot, path), 'utf8');
    }
  };
  walk(directory);
  return found;
}

/** The schemas as a map of path to text, which is what the API takes. */
export function schemaTexts(): Record<string, string> {
  const found: Record<string, string> = {};
  for (const file of schemaFiles()) found[file.path] = file.text;
  return found;
}

/** The reference base with the template documents it pins, as the workspace would gather it. */
export function referenceBase(): LibraryBaseFiles {
  return {
    base: 'data/primitive-library',
    files: { ...filesUnder('data/primitive-library'), ...filesUnder('data/models/decoder-causal-yarn') },
  };
}

/** The text of one corpus document, as the file holds it. */
export function corpusText(name: string): string {
  return readRepositoryFile(`data/models/${name}.json`);
}

/** A `Lang` with the schemas and the reference base loaded, in this thread. */
export async function core(): Promise<{
  lang: Lang;
  schemas: SchemasHandle;
  library: LibraryHandle;
  stop: () => void;
}> {
  const lang = createLang();
  const schemas = await lang.loadSchemas(schemaTexts(), { origin: 'schemas' });
  const library = await lang.loadLibrary([referenceBase()], schemas.handle);
  return {
    lang,
    schemas: schemas.handle,
    library: library.handle,
    stop: () => {
      lang.close();
    },
  };
}
