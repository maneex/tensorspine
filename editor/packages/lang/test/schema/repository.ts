import { readFileSync } from 'node:fs';

import { loadSchemas, type SchemaRegistry } from '../../src/schema/index.js';
import { filesUnder, readRepositoryFile, repositoryRoot } from '../json/repository.js';

/**
 * The registry the schema suites read: the repository's own `schemas/`, loaded as the static
 * build vendors them (plan §1 — "the static build vendors `schemas/` … and reads them at startup
 * indexed by `$id`").
 */
export function repositorySchemas(): SchemaRegistry {
  const files = filesUnder('schemas')
    .filter((path) => path.endsWith('.json'))
    .map((path) => ({ path, text: readRepositoryFile(path) }));
  return loadSchemas(files, { origin: 'schemas' });
}

/** The rejection manifests of `tests/rejections/`, as the tools' own runner reads them. */
export interface RejectionCase {
  document?: string;
  base?: string;
  expect?: string;
  match: string;
  note?: string;
  assign?: unknown;
}

/** One rejection manifest, read from the repository at the source. */
export function rejectionCases(name: 'models' | 'primitive-library'): RejectionCase[] {
  const manifest = JSON.parse(readRepositoryFile(`tests/rejections/${name}.json`)) as {
    cases: RejectionCase[];
  };
  return manifest.cases;
}

/** Every JSON file of a rejection base, deepest last, as repository-relative paths. */
export function baseFiles(base: string): string[] {
  return filesUnder(`tests/rejections/${base}`).filter((path) => path.endsWith('.json'));
}

/** The text of a file named relative to the repository root. */
export function read(path: string): string {
  return readFileSync(`${repositoryRoot}/${path}`, 'utf8');
}
