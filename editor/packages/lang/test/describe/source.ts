import { readFileSync } from 'node:fs';

import { parse } from '../../src/json/index.js';
import {
  basesOf,
  describe as describeDocument,
  loadLibrary,
  toPython,
  whereOfSite,
  type Description,
  type Library,
  type SchemaRegistry,
  type SiteDescription,
} from '../../src/index.js';
import { repositoryRoot } from '../json/repository.js';
import { nodeSource } from '../library/source.js';
import { repositorySchemas } from '../schema/repository.js';

/**
 * The corpus, described once per document, for the `describe` and `check` suites.
 *
 * The documents are the repository's own and the base is the reference base, read through the
 * loader's own interface (`packages/lang` never imports `node:fs`): what these suites pin is the
 * answer the editor gets when it opens a corpus model, so the input is that file and nothing
 * built for the test.
 */
export const schemas: SchemaRegistry = repositorySchemas();

const source = nodeSource(repositoryRoot);

/** The reference base of `data/primitive-library`, gathered once. */
export const library: Library = loadLibrary(['data/primitive-library'], { schemas, source });

/** The text of one corpus document. */
export function corpus(name: string): string {
  return readFileSync(`${repositoryRoot}/data/models/${name}.json`, 'utf8');
}

const described = new Map<string, Description>();

/** One corpus document, described; the answer is kept, since every suite reads several sites. */
export function describedCorpus(name: string): Description {
  const held = described.get(name);
  if (held !== undefined) return held;
  const path = `data/models/${name}.json`;
  const tree = parse(corpus(name));
  const { bases, problem } = basesOf(path, toPython(tree));
  if (problem !== null) throw new Error(`${name}: ${problem.message}`);
  const answer = describeDocument(tree, { schemas, library: loadLibrary(bases, { schemas, source }) });
  described.set(name, answer);
  return answer;
}

const indexed = new Map<string, Map<string, SiteDescription>>();

/** The sites of one corpus document, by the identifier a refusal and D1 name a node with. */
export function sites(name: string): Map<string, SiteDescription> {
  const held = indexed.get(name);
  if (held !== undefined) return held;
  const answer = new Map(
    [...describedCorpus(name).sites.values()].map((site) => [whereOfSite(site.key), site]),
  );
  indexed.set(name, answer);
  return answer;
}

/** One site of one corpus document, by that identifier. */
export function siteOf(name: string, where: string): SiteDescription {
  const site = sites(name).get(where);
  if (site === undefined) throw new Error(`${name}: no site ${where}`);
  return site;
}
