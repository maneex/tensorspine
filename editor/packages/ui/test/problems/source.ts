import { createLang } from '@tensorspine/lang/api/engine';
import type { Lang, LibraryHandle, Problem, SchemasHandle, Verdict } from '@tensorspine/lang/api';
import { parse, type JsonObject } from '@tensorspine/lang';
import { referenceIndex, SchemaShapes, type ReferenceIndex } from '@tensorspine/store';

import { outlineOf, type OutlineRow } from '../../src/explorer/outline.js';
import { presentation } from '../../src/presentation/index.js';
import { filesUnder, schemaTexts } from '../documents/source.js';
import { readRepositoryFile, registry } from '../presentation/source.js';

export { readRepositoryFile, registry, repositoryRoot } from '../presentation/source.js';

/**
 * What the Problems suites read: the repository's own corpus, judged by the core itself.
 *
 * The rows the panel arranges are the core's, so a suite that made them up would be testing the
 * arrangement against a fiction. Everything here runs the real `Lang` over the real documents and
 * the real reference base, exactly as feature 2.6's own suites do one directory along.
 */

let shapesOnce: SchemaShapes | null = null;

/** The schema shapes, built once. */
export function shapes(): SchemaShapes {
  shapesOnce ??= new SchemaShapes(registry());
  return shapesOnce;
}

/** One corpus document, parsed as the store parses it. */
export function corpus(name: string): JsonObject {
  return parse(readRepositoryFile(`data/models/${name}.json`)) as JsonObject;
}

/** The path a corpus document is named by, as a workspace holds it. */
export function corpusPath(name: string): string {
  return `data/models/${name}.json`;
}

/** Every row of a document's outline — what grouping, navigation and the notices read. */
export function outline(tree: JsonObject): OutlineRow[] {
  return outlineOf({ tree, shapes: shapes(), bindings: presentation(), openAll: true });
}

/** The reference index of a document, as the store builds one. */
export function index(tree: JsonObject): ReferenceIndex {
  return referenceIndex(tree, shapes(), 'model');
}

/** The reference base as a workspace hands it over: its own files and the templates it pins. */
export function referenceFiles(): Record<string, string> {
  return {
    ...filesUnder('data/primitive-library'),
    ...filesUnder('data/models/decoder-causal-yarn'),
  };
}

/** A `Lang` with the repository's schemas and reference base loaded, in this thread. */
export async function core(): Promise<{
  lang: Lang;
  schemas: SchemasHandle;
  library: LibraryHandle;
  stop: () => void;
}> {
  const lang = createLang();
  const loaded = await lang.loadSchemas(schemaTexts(), { origin: 'schemas' });
  const gathered = await lang.loadLibrary(
    [{ base: 'data/primitive-library', files: referenceFiles() }],
    loaded.handle,
  );
  return {
    lang,
    schemas: loaded.handle,
    library: gathered.handle,
    stop: () => {
      lang.close();
    },
  };
}

/** The verdict on a document, and its rows as the panel is handed them. */
export async function verdictOf(
  lang: Lang,
  library: LibraryHandle,
  name: string,
  edit: (text: string) => string = (text) => text,
): Promise<{ verdict: Verdict; tree: JsonObject; rows: { problem: Problem; stale: boolean }[] }> {
  const text = edit(readRepositoryFile(`data/models/${name}.json`));
  const tree = parse(text) as JsonObject;
  const verdict = await lang.validate(tree, corpusPath(name), { library });
  return { verdict, tree, rows: verdict.problems.map((problem) => ({ problem, stale: false })) };
}
