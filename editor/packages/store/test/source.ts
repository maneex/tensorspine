import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { isJsonObject, loadSchemas, parse, type JsonObject, type SchemaRegistry } from '@tensorspine/lang';

import { SchemaShapes } from '../src/shape.js';

/**
 * What the store's suites read.
 *
 * The corpus, the reference base and the schemas are the repository's own and are read at the
 * source (plan D14, §0.1) — never copied into the editor. The store takes trees and a schema
 * registry, so these helpers do what a `Workspace` will do (plan §5.2) and nothing the store is
 * supposed to do itself.
 */
export const repositoryRoot = resolve(
  dirname(fileURLToPath(import.meta.url)),
  '..',
  '..',
  '..',
  '..',
);

/** The editor's own directory, which holds the sidecar schemas (plan §5.1). */
export const editorRoot = join(repositoryRoot, 'editor');

/** The text of a repository file, decoded as UTF-8. */
export function readRepositoryFile(path: string): string {
  return readFileSync(join(repositoryRoot, path), 'utf8');
}

/**
 * The fourteen models and the template, by the path each is read at.
 *
 * The template is `data/models/decoder-causal-yarn/1.0.0.json` — a directory of its own, because
 * the reference base pins it by file (`"templates": "../models/"`) — so the corpus is walked
 * rather than listed.
 */
export function corpusNames(): string[] {
  const root = join(repositoryRoot, 'data', 'models');
  const found: string[] = [];
  const walk = (directory: string, prefix: string): void => {
    for (const entry of readdirSync(directory, { withFileTypes: true }).sort((left, right) =>
      left.name < right.name ? -1 : 1,
    )) {
      if (entry.isDirectory()) walk(join(directory, entry.name), `${prefix}${entry.name}/`);
      else if (entry.name.endsWith('.json')) found.push(`${prefix}${entry.name.slice(0, -'.json'.length)}`);
    }
  };
  walk(root, '');
  return found;
}

/** The text of one corpus document, as the file holds it. */
export function corpusText(name: string): string {
  return readRepositoryFile(`data/models/${name}.json`);
}

/** One corpus document as the tree the store holds (feature 0.3's parser). */
export function corpusTree(name: string): JsonObject {
  const tree = parse(corpusText(name));
  if (!isJsonObject(tree)) throw new TypeError(`${name} is not an object`);
  return tree;
}

let registryOnce: SchemaRegistry | null = null;

/** The repository's schemas, loaded once for the whole suite (the registry costs ~200 ms). */
export function registry(): SchemaRegistry {
  if (registryOnce === null) {
    const directory = join(repositoryRoot, 'schemas');
    const files = readdirSync(directory)
      .filter((name) => name.endsWith('.json'))
      .map((name) => ({ path: `schemas/${name}`, text: readFileSync(join(directory, name), 'utf8') }));
    registryOnce = loadSchemas(files, { origin: 'schemas' });
  }
  return registryOnce;
}

let shapesOnce: SchemaShapes | null = null;

/** The schema reading the store walks a document with, shared by the suites. */
export function shapes(): SchemaShapes {
  shapesOnce ??= new SchemaShapes(registry());
  return shapesOnce;
}

/** The role of a model document, as the registry indexes it. */
export const MODEL = 'model';
