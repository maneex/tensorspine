import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { loadSchemas, type SchemaRegistry } from '@tensorspine/lang';

/**
 * What the presentation suites read.
 *
 * The schemas are the repository's own and are read at the source (plan D14, §0.1): the static
 * build vendors that directory byte for byte, and `tests/audit/vendor.test.ts` is what proves it,
 * so auditing against `schemas/` is auditing against what the application ships.
 */
export const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', '..', '..');

/** The editor's own directory, which holds the presentation and layout schemas (plan §5.1). */
export const editorRoot = join(repositoryRoot, 'editor');

let once: SchemaRegistry | null = null;

/** The repository's schemas, loaded once for the whole suite (the registry costs ~200 ms). */
export function registry(): SchemaRegistry {
  once ??= loadSchemas(schemaFiles(), { origin: 'schemas' });
  return once;
}

/** Every file of the repository's `schemas/`, as the registry is handed them. */
export function schemaFiles(): { path: string; text: string }[] {
  const directory = join(repositoryRoot, 'schemas');
  return readdirSync(directory)
    .filter((name) => name.endsWith('.json'))
    .map((name) => ({ path: `schemas/${name}`, text: readFileSync(join(directory, name), 'utf8') }));
}

/** The text of a repository file, decoded as UTF-8. */
export function readRepositoryFile(path: string): string {
  return readFileSync(join(repositoryRoot, path), 'utf8');
}
