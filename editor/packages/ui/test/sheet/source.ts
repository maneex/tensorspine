import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';

import {
  basesOf,
  derive,
  describe as describeDocument,
  loadLibrary,
  parse,
  pyStr,
  siteDerived,
  toPython,
  whereOfSite,
  type JsonObject,
  type Analysis,
  type Library,
  type SiteDerived,
  type SiteDescription,
} from '@tensorspine/lang';
import type { Facts } from '@tensorspine/lang/api';

import { formContext, type FormContext } from '../../src/forms/index.js';
import { presentation } from '../../src/presentation/index.js';
import { argumentSheet, type ArgumentSheet } from '../../src/sheet/arguments.js';
import { ArgumentSchema } from '../../src/sheet/artifact.js';
import { instanceSheet, type InstanceSheet } from '../../src/sheet/instance.js';
import { library, registry, reading as canvasReading, shapes } from '../canvas/source.js';
import { repositoryRoot } from '../presentation/source.js';

/**
 * What the sheet suites read: a corpus document described and derived by the core, the tools' own
 * generated argument schema beside it, and the sheet those three make.
 *
 * Nothing is written for the suites. The feature's whole claim is that the sheet shows what the
 * core and the artifacts answer, so a fixture with an invented fact in it would prove the opposite
 * of what is wanted (the canvas suites took the same course, and this reads through theirs).
 */

export { library, registry, shapes };

/** The form context the sheets are generated in — one per registry, as the interface builds it. */
export const context: FormContext = formContext(registry, presentation());

/** One corpus document as the editor holds it. */
export function tree(name: string): JsonObject {
  return canvasReading(name).tree;
}

/** The core's facts for a corpus document, folded as §5.4 asks for them. */
export function facts(name: string): Facts {
  return canvasReading(name).facts;
}

/** One described site of a corpus document, by the identifier D1 and every refusal name it. */
export function siteOf(name: string, where: string): SiteDescription {
  const site = facts(name).sites.get(where);
  if (site === undefined) throw new Error(`${name}: no site ${where}`);
  return site;
}

const artifacts = new Map<string, ArgumentSchema | null>();

/**
 * One primitive's generated argument schema, read from the vendored directory the build serves.
 *
 * `--document primitive-schema`'s own output, consumed as built (F5): the suite reads the very
 * file the static application fetches, so what the rows are held to is the artifact and not a copy.
 */
export function artifactOf(name: string, version: string): ArgumentSchema | null {
  const id = `${name}@${version}`;
  const held = artifacts.get(id);
  if (held !== undefined) return held;
  const directory = resolve(repositoryRoot, 'editor/apps/web/public/vendor/generated/primitive-schema');
  // The directory is the vendor's and the vendor is generated (`pnpm vendor`, which `pnpm check`
  // runs before this layer). Its *absence* is not F5's case below — it is a checkout that has not
  // vendored, and the answer to that is the command, not ten rows quietly read off the grammar:
  // that was five days of red CI at the unit stage (12–17 Sep 2026) that every local run passed.
  if (!existsSync(directory)) {
    throw new Error(
      `no vendored argument schemas under ${directory}: run \`pnpm vendor\` (\`pnpm check\` does)`,
    );
  }
  const path = resolve(directory, `${name}_${version}.json`);
  let found: ArgumentSchema | null;
  try {
    found = new ArgumentSchema(JSON.parse(readFileSync(path, 'utf8')));
  } catch {
    // A primitive the build generated no schema for: F5's own case, and an answer rather than a
    // failure — the rows are then what the *grammar* says a literal is.
    found = null;
  }
  artifacts.set(id, found);
  return found;
}

/** The argument sheet of one site of one corpus document (§4.12). */
export function sheetOf(
  name: string,
  where: string,
  options: { readonly showInapplicable?: boolean; readonly problemsFirst?: boolean } = {},
): ArgumentSheet {
  const site = siteOf(name, where);
  return argumentSheet({
    facts: site.arguments.facts,
    invariants: site.arguments.invariants,
    tree: tree(name),
    role: MODEL,
    context,
    shapes: context.shapes,
    artifact: artifactOf(pyStr(site.primitive), pyStr(site.version)) ?? undefined,
    ...options,
  });
}

/** The instance sheet of one site, with the products the core derived for the document. */
export function instanceOf(name: string, where: string): InstanceSheet {
  const site = siteOf(name, where);
  return instanceSheet(site, derivedOf(name, where));
}

/** What the products say about one site. */
export function derivedOf(name: string, where: string): SiteDerived {
  return siteDerived(canvasReading(name).derived, where);
}

/** The role a `tensorspine/2.0` document is read under. */
export const MODEL = 'model';

const analyses = new Map<string, Analysis>();

/**
 * The semantic analysis of one corpus document — what an identity's facts are answered from.
 *
 * `describe` keeps it (§5.4's one reading per revision) and the worker answers `identityFacts`
 * off it; a suite that wants the same answer reads it the same way, so what is tested is the
 * call the sheet makes and not a second path into the core.
 */
export function analysisOf(name: string): Analysis {
  const held = analyses.get(name);
  if (held !== undefined) return held;
  const path = `data/models/${name}.json`;
  const held_ = tree(name);
  const { bases, problem } = basesOf(path, toPython(held_));
  if (problem !== null) throw new Error(`${name}: ${problem.message}`);
  const description = describeDocument(held_, {
    schemas: registry,
    library: loadLibrary(bases, { schemas: registry, source: fileSource() }),
    folded: true,
  });
  if (description.analysis === null) throw new Error(`${name}: no analysis`);
  analyses.set(name, description.analysis);
  return description.analysis;
}

/** A fresh tree of a corpus document, for a suite that edits one. */
export function freshTree(name: string): JsonObject {
  return parse(readFileSync(resolve(repositoryRoot, `data/models/${name}.json`), 'utf8')) as JsonObject;
}

/** A document held in memory, described afresh: what an edit in the sheet produces. */
export function described(name: string, held: JsonObject): Facts {
  const path = `data/models/${name}.json`;
  const { bases, problem } = basesOf(path, toPython(held));
  if (problem !== null) throw new Error(`${name}: ${problem.message}`);
  const gathered: Library = loadLibrary(bases, { schemas: registry, source: fileSource() });
  const description = describeDocument(held, { schemas: registry, library: gathered, folded: true });
  const sites = new Map(
    [...description.sites.values()].map((site) => [whereOfSite(site.key), site] as const),
  );
  return { conforms: true, structural: [], sites };
}

/** The derived document of a held tree, for a sheet that shows its products. */
export function derivedFor(name: string, held: JsonObject): ReturnType<typeof derive> {
  const { bases } = basesOf(`data/models/${name}.json`, toPython(held));
  return derive(held, {
    schemas: registry,
    library: loadLibrary(bases, { schemas: registry, source: fileSource() }),
  });
}

/** The sheet of a site of a held tree — what the interface builds after every keystroke. */
export function sheetFor(name: string, held: JsonObject, where: string): ArgumentSheet {
  const site = described(name, held).sites.get(where);
  if (site === undefined) throw new Error(`${name}: no site ${where}`);
  return argumentSheet({
    facts: site.arguments.facts,
    invariants: site.arguments.invariants,
    tree: held,
    role: MODEL,
    context,
    shapes: context.shapes,
    artifact: artifactOf(pyStr(site.primitive), pyStr(site.version)) ?? undefined,
  });
}

/**
 * The loader's own file interface over the repository.
 *
 * `packages/lang` imports no `node:fs` (§5.3), so a suite is what puts a filesystem behind it —
 * the same shape `packages/lang/test/library/source.ts` and the canvas suites use, so a base is
 * named the way the tools name it.
 */
export function fileSource(): { isDirectory: (path: string) => boolean; isFile: (path: string) => boolean; exists: (path: string) => boolean; find: (directory: string) => string[]; read: (path: string) => string } {
  const at = (path: string): string => resolve(repositoryRoot, path);
  return {
    isDirectory: (path) => existsSync(at(path)) && statSync(at(path)).isDirectory(),
    isFile: (path) => existsSync(at(path)) && statSync(at(path)).isFile(),
    exists: (path) => existsSync(at(path)),
    find(directory) {
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
    read: (path) => readFileSync(at(path), 'utf8'),
  };
}
