import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { expect } from 'vitest';

import {
  basesOf,
  derivationGraph,
  loadLibrary,
  parse,
  serialize,
  toJsonValue,
  toPython,
  type Derivation,
  type Library,
  type PyRecord,
  type PyValue,
} from '../../src/index.js';
import { nodeSource } from '../library/source.js';
import { repositorySchemas } from '../schema/repository.js';
import { oracleGenerated, oracleOut, readOracleManifest, repositoryRoot } from './oracle.js';

// The fixture every product's parity suite reads: the corpus derived once, beside the document
// `tools/tensorspine --derive` wrote for it.
//
// The expectation is the repository's own derived document — `out/derive/<name>.derived.json`,
// what `--derive` wrote for the corpus and for the template under its documented assignment — and
// a product's suite compares its own member of it, twice: the readings deep-equal, then the two
// written with the core's serializer and compared as text. The reading decides the structure *and*
// the integer/float distinction, because `toPython` keeps it (a `bf16` payload's `bytes` is an
// integer and a `fp4` one's a float, and `1024n` is not `1024`); the text names the line that
// moved.
//
// The assignment is read from the derived document itself, which records it (§7), so the whole
// expectation is one file and the external quantities keep their lexemes: `3072` a whole number,
// `1e-05` a real. The derivation is built once per document and shared by every product's suite in
// the same process, since `derivationGraph` is the gate all of them stand behind.

export const inCI = process.env['CI'] !== undefined && process.env['CI'] !== '';
export const generated = oracleGenerated();

const schemas = repositorySchemas();
const source = nodeSource(repositoryRoot);
const libraries = new Map<string, Library>();

function libraryFor(bases: readonly string[]): Library {
  const key = bases.join('|');
  const held = libraries.get(key);
  if (held !== undefined) return held;
  const library = loadLibrary(bases, { schemas, source });
  libraries.set(key, library);
  return library;
}

/** The bases a corpus document declares, as `load_for` resolves them. */
function declaredBases(path: string, document: PyValue): string[] {
  return [...basesOf(path, document).bases];
}

/** A member of a record read from a document, or a failure naming it. */
export function at(value: PyValue, name: string): PyValue {
  const held = (value as PyRecord)[name];
  expect(held, `the recorded document has no '${name}'`).toBeDefined();
  return held as PyValue;
}

/** One corpus document, derived by the core, beside the document the tools derived. */
export interface Case {
  name: string;
  path: string;
  /** `--derive`'s own output, read as a document is read: `1e-05` a float, `4096` a whole number. */
  expected: PyValue;
  derivation: Derivation;
}

const cases: Case[] = [];

/** The fifteen documents `--derive` was run over, derived by the core. */
export function derivedCorpus(): readonly Case[] {
  if (cases.length > 0) return cases;
  for (const one of readOracleManifest().documents) {
    const expected = toPython(parse(readFileSync(join(oracleOut, one.derived), 'utf8')));
    const assignment = at(expected, 'assignment') as PyRecord;
    const text = readFileSync(join(repositoryRoot, one.path), 'utf8');
    const tree = parse(text);
    const library = libraryFor(declaredBases(one.path, toPython(tree)));
    cases.push({
      name: one.name,
      path: one.path,
      expected,
      derivation: derivationGraph(tree, { schemas, library, assignment }),
    });
  }
  return cases;
}

/** The library the document was derived under, by name. */
export function libraryOf(one: Case): Library {
  return libraryFor(declaredBases(one.path, one.derivation.document));
}

/** The two comparisons, in the order that makes a failure readable. */
export function agrees(answer: PyValue, expected: PyValue, where: string): void {
  expect(answer, where).toEqual(expected);
  expect(serialize(toJsonValue(answer)), where).toBe(serialize(toJsonValue(expected)));
}
