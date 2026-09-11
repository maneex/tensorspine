import { parse, type JsonObject } from '@tensorspine/lang';
import { SchemaShapes } from '@tensorspine/store';

import { presentation, type Presentation } from '../../src/presentation/index.js';
import { outlineOf, type OutlineRow } from '../../src/explorer/outline.js';
import { readRepositoryFile, registry } from '../presentation/source.js';

export { readRepositoryFile, registry, repositoryRoot } from '../presentation/source.js';

/**
 * What the explorer's suites read: the repository's own corpus, against the repository's own
 * schemas and the interface's one data file.
 */

let shapesOnce: SchemaShapes | null = null;

/** The schema shapes, built once: the registry costs ~200 ms and the shapes memoise every descent. */
export function shapes(): SchemaShapes {
  shapesOnce ??= new SchemaShapes(registry());
  return shapesOnce;
}

/** The presentation bindings, as the application reads them. */
export function bindings(): Presentation {
  return presentation();
}

/** One corpus document, parsed as the store parses it (lexemes kept, member order kept). */
export function corpus(name: string): JsonObject {
  return parse(readRepositoryFile(`data/models/${name}.json`)) as JsonObject;
}

/** The outline of a corpus document, with whatever a case wants to say about it. */
export function outline(
  name: string,
  request: Partial<Parameters<typeof outlineOf>[0]> = {},
): OutlineRow[] {
  return outlineOf({ tree: corpus(name), shapes: shapes(), bindings: bindings(), ...request });
}

/** A row as a line, the way a reader of the tree sees it: indent, name, count, tail, marks. */
export function lineOf(row: OutlineRow): string {
  const chevron = row.kind === 'note' ? ' ' : row.container ? (row.open ? '▾' : '▸') : ' ';
  const parts = [
    '  '.repeat(row.depth) + chevron,
    row.label,
    row.count === undefined ? '' : `(${String(row.count)})`,
    row.figure ?? '',
    row.marks.join(' '),
    row.tail === undefined ? '' : `· ${row.tail}`,
  ];
  return parts.filter((one) => one !== '').join(' ');
}

/** The whole outline as lines — what a failing case prints. */
export function linesOf(rows: readonly OutlineRow[]): string[] {
  return rows.map(lineOf);
}
