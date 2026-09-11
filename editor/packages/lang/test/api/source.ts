import { readFileSync } from 'node:fs';

import { connectLang, type Lang } from '../../src/api/index.js';
import { createLang, serveLang } from '../../src/api/engine.js';
import type { LibraryBaseFiles, LibraryHandle, SchemasHandle } from '../../src/api/index.js';
import { filesUnder, readRepositoryFile, repositoryRoot } from '../json/repository.js';

/**
 * What the API suites read, and how the two implementations are put side by side.
 *
 * The inputs are the repository's own — `schemas/`, `data/primitive-library/`, `data/models/` —
 * handed over as **files**, which is what the API takes: "the core is pure (no I/O): the UI reads
 * files through `Platform` and hands the core texts and trees" (§5.3). So these helpers do what
 * `BrowserWorkspace` will do (§5.2) and nothing the core is supposed to do itself.
 */

/** Every schema of the repository, as the static build vendors the directory (plan §1). */
export function schemaFiles(): Record<string, string> {
  const files: Record<string, string> = {};
  for (const path of filesUnder('schemas')) {
    if (path.endsWith('.json')) files[path] = readRepositoryFile(path);
  }
  return files;
}

/**
 * The reference base as one {@link LibraryBaseFiles}.
 *
 * The template documents the manifest pins (`"templates": "../models/"`) travel with it: they are
 * outside the base's own directory and the loader opens them to check a template primitive's
 * file, name, version and id, so a base handed over without them would be refused for a reason
 * the workspace, not the base, is responsible for.
 */
export function referenceBase(): LibraryBaseFiles {
  const files: Record<string, string> = {};
  for (const path of filesUnder('data/primitive-library')) {
    if (path.endsWith('.json')) files[path] = readRepositoryFile(path);
  }
  for (const path of filesUnder('data/models/decoder-causal-yarn')) {
    if (path.endsWith('.json')) files[path] = readRepositoryFile(path);
  }
  return { base: 'data/primitive-library', files };
}

/** The text of one corpus document, as the file holds it. */
export function corpus(name: string): string {
  return readFileSync(`${repositoryRoot}/data/models/${name}.json`, 'utf8');
}

/** The path a corpus document is opened under, which is what its bases resolve against. */
export function corpusPath(name: string): string {
  return `data/models/${name}.json`;
}

/** The fourteen models and the template, by name. */
export function corpusNames(): string[] {
  return filesUnder('data/models')
    .filter((path) => path.endsWith('.json'))
    .map((path) => path.slice('data/models/'.length, -'.json'.length));
}

/** A `Lang` with the schemas and the reference base already loaded. */
export interface Loaded {
  readonly lang: Lang;
  readonly schemas: SchemasHandle;
  readonly library: LibraryHandle;
  /** Stop the proxy and, for the worker case, the host behind it. */
  readonly stop: () => void;
}

/** The two ways the same interface is reached, so that every case runs against both. */
export type Deployment = 'in-process' | 'worker';

/**
 * Open a `Lang` of either deployment.
 *
 * The worker case is a `MessageChannel` with the host on one end and the client on the other. It
 * is not a thread — `packages/lang` names its own imports with the `.js` specifiers a browser
 * bundler resolves and plain Node does not, so a Node worker thread cannot load the core at all —
 * but it *is* a true structured-clone boundary: a `postMessage` between two ports of one channel
 * serializes exactly as a `postMessage` between threads does, and refuses a symbol with the same
 * `DataCloneError`. The thread itself is exercised in the browser layer (`apps/web/e2e`), where a
 * real `Worker` runs the built bundle.
 */
export function open(deployment: Deployment): Lang {
  if (deployment === 'in-process') return createLang();
  const channel = new MessageChannel();
  const host = serveLang(channel.port2);
  return connectLang(channel.port1, {
    onClose: () => {
      host.stop();
      channel.port1.close();
      channel.port2.close();
    },
  });
}

/** The schemas and the reference base loaded into a fresh `Lang` of that deployment. */
export async function loaded(deployment: Deployment): Promise<Loaded> {
  const lang = open(deployment);
  const schemas = await lang.loadSchemas(schemaFiles(), { origin: 'schemas' });
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
