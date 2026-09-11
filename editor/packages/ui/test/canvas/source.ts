import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';

import {
  basesOf,
  derive,
  describe as describeDocument,
  foldedGraph,
  loadLibrary,
  parse,
  toPython,
  whereOfSite,
  type FoldedGraph,
  type JsonObject,
  type Library,
  type PyValue,
  type SchemaRegistry,
} from '@tensorspine/lang';
import type { Facts, Problem } from '@tensorspine/lang/api';
import { SchemaShapes } from '@tensorspine/store';

import { presentation } from '../../src/presentation/index.js';
import { canvasModel, type CanvasModel, type CanvasView } from '../../src/canvas/model.js';
import { readRepositoryFile, registry as repositorySchemas, repositoryRoot } from '../presentation/source.js';

/**
 * What the canvas suites read: a corpus document, described and derived by the core.
 *
 * The whole point of the feature is that the canvas computes nothing, so the suites feed it the
 * core's real answers over the repository's real documents — never a fixture made for them.
 */

export const registry: SchemaRegistry = repositorySchemas();
export const shapes = new SchemaShapes(registry);
export const bindings = presentation();

/**
 * The library source the loader reads through: the repository's own files, rooted at its root.
 *
 * `packages/lang` never imports `node:fs` (§5.3), so the loader takes an interface and a suite is
 * what puts a filesystem behind it — the same shape `packages/lang/test/library/source.ts` uses,
 * so that a base is named the way the tools name it.
 */
const source = {
  isDirectory: (path: string) => existsSync(at(path)) && statSync(at(path)).isDirectory(),
  isFile: (path: string) => existsSync(at(path)) && statSync(at(path)).isFile(),
  exists: (path: string) => existsSync(at(path)),
  find(directory: string): string[] {
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
    return found.map((path) => relative(repositoryRoot, path));
  },
  read: (path: string) => readFileSync(at(path), 'utf8'),
};

function at(path: string): string {
  return resolve(repositoryRoot, path);
}

/** The reference base, gathered once for the whole suite. */
let gathered: Library | null = null;

export function library(): Library {
  gathered ??= loadLibrary(['data/primitive-library'], { schemas: registry, source });
  return gathered;
}

/** One corpus document's tree. */
export function corpusTree(name: string): JsonObject {
  return parse(readRepositoryFile(`data/models/${name}.json`)) as JsonObject;
}

/** What the core says about one corpus document: the folded reading, the facts and the products. */
export interface Reading {
  readonly tree: JsonObject;
  readonly folded: FoldedGraph;
  readonly facts: Facts;
  readonly derived: PyValue;
  readonly problems: readonly Problem[];
}

const readings = new Map<string, Reading>();

/** One corpus document read as the editor reads it — the answer is kept, the suites read several. */
export function reading(name: string): Reading {
  const held = readings.get(name);
  if (held !== undefined) return held;
  const tree = corpusTree(name);
  const path = `data/models/${name}.json`;
  const { bases, problem } = basesOf(path, toPython(tree));
  if (problem !== null) throw new Error(`${name}: ${problem.message}`);
  const gatheredFor = loadLibrary(bases, { schemas: registry, source });
  const description = describeDocument(tree, { schemas: registry, library: gatheredFor, folded: true });
  const sites = new Map(
    [...description.sites.values()].map((site) => [whereOfSite(site.key), site] as const),
  );
  const answer: Reading = {
    tree,
    folded: foldedGraph(tree),
    facts: { conforms: true, structural: [], sites },
    derived: derive(tree, { schemas: registry, library: gatheredFor }),
    problems: (description.analysis?.problems ?? []).map((one) => ({
      code: one.code,
      message: one.message,
      path: one.path,
      severity: 'error' as const,
      source: 'semantic' as const,
    })),
  };
  readings.set(name, answer);
  return answer;
}

/** The canvas model of one corpus document, with whatever the suite wants shown. */
export function modelOf(
  name: string,
  options: {
    readonly view?: Partial<CanvasView>;
    readonly collapsed?: ReadonlySet<string>;
    readonly derived?: boolean;
    readonly problems?: readonly Problem[];
  } = {},
): CanvasModel {
  const read = reading(name);
  const shown: CanvasView = {
    families: true,
    derivedFigures: true,
    edgeTypes: false,
    identities: false,
    ...options.view,
  };
  return canvasModel({
    folded: read.folded,
    facts: read.facts,
    derived: options.derived === false ? null : read.derived,
    problems: options.problems ?? read.problems,
    collapsed: options.collapsed ?? everyComposition(read.folded),
    view: shown,
    registry,
    shapes,
    bindings,
  });
}

/** §4.7's default: every composition is drawn collapsed until the reader opens one. */
export function everyComposition(folded: FoldedGraph): Set<string> {
  const shut = new Set<string>();
  for (const node of folded.nodes) if (node.children.length > 0) shut.add(node.pointer);
  return shut;
}
