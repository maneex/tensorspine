import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { parse } from '../../src/json/index.js';
import {
  describeArguments,
  identityKey,
  loadLibrary,
  loadModel,
  member,
  primitiveOf,
  resolveQuantities,
  staticArgument,
  templateInterfaces,
  toPython,
  type ArgumentFact,
  type Env,
  type InvariantVerdict,
  type Library,
  type PyRecord,
  type PyValue,
  type SemanticProblem,
} from '../../src/index.js';
import { repositorySchemas } from '../schema/repository.js';
import { nodeSource } from '../library/source.js';
import { decodeRecord, encode } from './encoding.js';
import { raisedAs, type Raised } from './raised.js';
import { oracleGenerated, oracleOut, readOracleManifest, repositoryRoot } from './oracle.js';

// Parity of the arguments (feature 1.6a): `resolve_arguments`, `_resolve_record`,
// `_check_argument_domain` and the V8 block of `tools/validate.py`, over every site of every
// model document of the repository — the corpus, the template under its documented assignment,
// the 73 of `tests/rejections/models/` — and of every template a document instantiates, expanded
// at the call site as `analyse` expands it.
//
// The oracle records *cases*, not a walk (feature 1.2's idiom): each carries the document and the
// pointer of the instance, the assignment and the index environment it was resolved in, and what
// the tools answered — the resolved values, the refusals, the invariant verdicts and the facts
// the argument sheet reads (§4.12). The inputs are read from the repository's own bytes on both
// sides: only the declaration is looked up in the primitive library, which feature 1.3's loader
// gathers here as the tools gather it.
//
// The environments of a composition's grid are walked in full by the generator and recorded as
// one case per *distinct outcome*, plus the last of the grid: `gemma3n-kvshare`'s `aux_select`,
// whose argument is the index itself, keeps its thirty, and a site whose arguments do not read
// the index keeps one.
//
// Beside them, the synthetic declarations of `tests/oracle/argument_cases.py`, because the
// reference base declares thirteen `present_when`s, no `seconds` argument, no set domain and no
// default that fails to resolve: the corpus alone leaves half of `_resolve_record` unvisited.

const inCI = process.env['CI'] !== undefined && process.env['CI'] !== '';
const generated = oracleGenerated();

/** One fact as the oracle records one: the row of §4.12, and what V3 said about it. */
interface RecordedFact {
  path: string;
  applicable: boolean;
  source: string;
  domain: string;
  value?: PyValue;
  written?: PyValue;
  problems: [string, string][];
}

/** One invariant verdict as the oracle records one. */
interface RecordedInvariant {
  description: string;
  verdict: string;
  reads: string[];
  shown: string;
  message?: string;
}

/** What the tools answered about one instance's arguments. */
interface Answer {
  values: PyRecord;
  problems: [string, string][];
  facts: RecordedFact[];
  invariants: RecordedInvariant[];
}

/** One site of one document: where it is, what it was read under, and the answer. */
type SiteCase = Answer & {
  document: string;
  /** The primitive library bases the tools gathered for it, repository-relative. */
  bases: string[];
  at: string;
  where: string;
  within: string;
  assignment: PyRecord | null;
  primitive: { name: string; version: string };
  template: boolean;
  env: PyRecord;
  environments: bigint;
};

/** One synthetic declaration, resolved against arguments written beside it. */
type SyntheticCase = Partial<Answer> & {
  name: string;
  definition: PyValue;
  given: PyValue;
  quantities: PyRecord;
  env: PyRecord;
  error: Raised | null;
};

/** One document walked: what it is, and the lines the walk produced for it. */
interface DocumentCase {
  name: string;
  path: string;
  assignment: PyRecord | null;
  sites: bigint;
  /** `[V2] required argument missing 'mask'`: the refusals without `analyse`'s own prefix. */
  messages?: string[];
  analysed?: boolean;
  error: Raised | null;
}

interface Recorded {
  documents: DocumentCase[];
  cases: SiteCase[];
  synthetic: SyntheticCase[];
}

interface RejectionCase {
  document: string;
  expect: string;
  match: string;
}

/** The fixture, read as a document is read: `1e-05` a float, `4096` a whole number. */
function recorded(): Recorded {
  const text = readFileSync(join(oracleOut, 'arguments', 'index.json'), 'utf8');
  return toPython(parse(text)) as unknown as Recorded;
}

function rejections(): RejectionCase[] {
  const file = join(oracleOut, 'rejections', 'models.json');
  return (JSON.parse(readFileSync(file, 'utf8')) as { cases: RejectionCase[] }).cases;
}

const schemas = repositorySchemas();
const source = nodeSource(repositoryRoot);

/** Every document is read once, and every set of bases gathered once. */
const documents = new Map<string, PyRecord>();
const libraries = new Map<string, Library>();

function documentAt(path: string): PyRecord {
  const held = documents.get(path);
  if (held !== undefined) return held;
  const read = loadModel(readFileSync(join(repositoryRoot, path), 'utf8'));
  documents.set(path, read);
  return read;
}

function libraryFor(bases: readonly string[]): Library {
  const key = bases.join('|');
  const held = libraries.get(key);
  if (held !== undefined) return held;
  const library = loadLibrary(bases, { schemas, source });
  libraries.set(key, library);
  return library;
}

/** The value at a document pointer, `instances/embed` or `compositions/decoder/instances/attn`. */
function at(model: PyRecord, pointer: string): PyValue {
  let cursor: PyValue = model;
  for (const step of pointer.split('/')) {
    cursor = member(cursor as PyRecord, step) as PyValue;
    expect(cursor, pointer).toBeDefined();
  }
  return cursor;
}

/** An encoded map as the fixture writes one: `generate.py`'s `encode_map`. */
function encodeRecord(values: PyRecord): Record<string, unknown> {
  return Object.fromEntries(Object.entries(values).map(([name, one]) => [name, encode(one)]));
}

/** A recorded map read back as the values it stands for. */
function decodedRecord(encoded: PyRecord | undefined): PyRecord {
  return encoded === undefined ? {} : decodeRecord(encoded);
}

/** A recorded map read back as an index environment or a quantity map. */
function decodedMap(encoded: PyRecord | undefined): Map<string, PyValue> {
  return new Map(Object.entries(decodedRecord(encoded)));
}

/** One fact as the fixture writes one. */
function encodeFact(fact: ArgumentFact): Record<string, unknown> {
  const out: Record<string, unknown> = {
    path: fact.path,
    applicable: fact.applicable,
    source: fact.source,
    domain: fact.domain,
    problems: fact.problems.map((one) => [one.code, one.message]),
  };
  if (fact.written !== undefined) out['written'] = encode(fact.written);
  if (fact.value !== undefined) out['value'] = encode(fact.value);
  return out;
}

/** One invariant verdict as the fixture writes one. */
function encodeInvariant(verdict: InvariantVerdict): Record<string, unknown> {
  const out: Record<string, unknown> = {
    description: verdict.description,
    verdict: verdict.verdict,
    reads: [...verdict.reads],
    shown: verdict.shown,
  };
  if (verdict.problem !== undefined) out['message'] = verdict.problem.message;
  return out;
}

/** A recorded answer, as the comparison reads it. */
function expected(one: Partial<Answer>): unknown {
  return {
    values: one.values,
    problems: (one.problems ?? []).map((pair) => [...pair]),
    facts: (one.facts ?? []).map((fact) => {
      const out: Record<string, unknown> = {
        path: fact.path,
        applicable: fact.applicable,
        source: fact.source,
        domain: fact.domain,
        problems: (fact.problems ?? []).map((pair) => [...pair]),
      };
      if (fact.written !== undefined) out['written'] = fact.written;
      if (fact.value !== undefined) out['value'] = fact.value;
      return out;
    }),
    invariants: (one.invariants ?? []).map((verdict) => {
      const out: Record<string, unknown> = {
        description: verdict.description,
        verdict: verdict.verdict,
        reads: [...verdict.reads],
        shown: verdict.shown,
      };
      if (verdict.message !== undefined) out['message'] = verdict.message;
      return out;
    }),
  };
}

/** What the port answers, in the same shape. */
function answered(
  definition: PyValue,
  given: PyValue,
  quantities: ReadonlyMap<string, PyValue>,
  env: Env,
): {
  answer: unknown;
  problems: readonly SemanticProblem[];
  facts: readonly ArgumentFact[];
  invariants: readonly InvariantVerdict[];
  keys: string[];
} {
  const description = describeArguments(definition, given, (one) =>
    staticArgument(one, quantities, env),
  );
  return {
    answer: {
      values: encodeRecord(description.values),
      problems: description.problems.map((one) => [one.code, one.message]),
      facts: description.facts.map(encodeFact),
      invariants: description.invariants.map(encodeInvariant),
    },
    problems: description.problems,
    facts: description.facts,
    invariants: description.invariants,
    keys: Object.keys(description.values),
  };
}

/**
 * The refusals of a description that no row and no invariant carries.
 *
 * The sheet's rows are the *declarations* (§4.12), so an argument the primitive does not declare
 * has no row to put its refusal on: the only unattributed problem is that V2, at the top level of
 * the map — an unknown *field* lands on its record's row, where the reader will look for it.
 */
function unattributed(mine: {
  problems: readonly SemanticProblem[];
  facts: readonly ArgumentFact[];
  invariants: readonly InvariantVerdict[];
}): string[] {
  const claimed = new Set<SemanticProblem>([
    ...mine.facts.flatMap((fact) => [...fact.problems]),
    ...mine.invariants.flatMap((one) => (one.problem === undefined ? [] : [one.problem])),
  ]);
  return mine.problems
    .filter((problem) => !claimed.has(problem))
    .filter(
      (problem) =>
        !(problem.code === 'V2' && /^unknown argument '[^.']+'$/.test(problem.message)),
    )
    .map((problem) => problem.message);
}

/** The declaration one site names: the primitive's, or the interface a template presents. */
function declarationOf(one: SiteCase, library: Library): PyValue {
  const reference = { name: one.primitive.name, version: one.primitive.version };
  const definition = primitiveOf(library, reference);
  expect(definition, `${one.document}: ${reference.name}`).toBeDefined();
  if (!one.template) return definition as PyValue;
  // `analyse` computes the interface at the call site; the loader computed it at load, from the
  // document it pinned — the same `template_interface` over the same two objects (feature 1.3).
  const interfaces = templateInterfaces(library);
  const found = interfaces.get(identityKey(reference.name, reference.version));
  expect(found, `${one.document}: interface of ${reference.name}`).toBeDefined();
  return found as PyValue;
}

describe('arguments, types, domains and invariants against the tools', () => {
  it.runIf(inCI)('has the oracle to compare with', () => {
    expect(generated).toBe(true);
  });

  it.skipIf(!generated)('records every model document of the repository, and the cases', () => {
    const material = recorded();
    const manifest = readOracleManifest() as unknown as {
      arguments?: { documents: number; sites: number; synthetic: number };
    };
    expect(manifest.arguments?.documents).toBe(material.documents.length);
    expect(manifest.arguments?.sites).toBe(material.cases.length);
    expect(manifest.arguments?.synthetic).toBe(material.synthetic.length);
    // The corpus and the template, and the 73 rejection documents.
    expect(material.documents).toHaveLength(15 + 73);
    expect(material.cases.length).toBeGreaterThan(500);
    expect(material.synthetic.length).toBeGreaterThan(60);
    expect(new Set(material.synthetic.map((one) => one.name)).size).toBe(
      material.synthetic.length,
    );
    // Every site of a composition stands for the environments that resolve to it.
    expect(material.cases.some((one) => one.environments > 1n)).toBe(true);
    // The one template the corpus instantiates is expanded at its call site, and its own sites
    // are recorded under the assignment the arguments make (§4.6).
    const expanded = material.cases.filter((one) => one.within !== '');
    expect(expanded.length).toBeGreaterThan(0);
    expect(new Set(expanded.map((one) => one.document))).toEqual(
      new Set(['data/models/decoder-causal-yarn/1.0.0.json']),
    );
    // Every branch the walk can take is reached by some case.
    const sources = new Set(material.cases.flatMap((one) => one.facts.map((f) => f.source)));
    expect([...sources].sort()).toEqual(['absent', 'default', 'given']);
    const verdicts = new Set([
      ...material.cases.flatMap((one) => one.invariants.map((each) => each.verdict)),
      ...material.synthetic.flatMap((one) => (one.invariants ?? []).map((each) => each.verdict)),
    ]);
    expect([...verdicts].sort()).toEqual(['fails', 'holds', 'skipped']);
    const domains = new Set(material.cases.flatMap((one) => one.facts.map((f) => f.domain)));
    expect([...domains].sort()).toEqual(['ok', 'refused', 'unchecked', 'undeclared']);
  });

  it.skipIf(!generated)('resolves every site of every document as the tools resolve it', () => {
    const material = recorded();
    for (const one of material.cases) {
      const model = documentAt(one.document);
      const definition = declarationOf(one, libraryFor(one.bases));
      const given = member(at(model, one.at) as PyRecord, 'arguments') as PyValue;
      const quantities = resolveQuantities(
        model,
        one.assignment === null ? undefined : decodedRecord(one.assignment),
      );
      const where = `${one.document} @${one.where}`;
      const mine = answered(definition, given, quantities, decodedMap(one.env));
      expect(mine.answer, where).toEqual(expected(one));
      // Python's dictionary order is the order the map is written in: the given names first,
      // then the defaults as they resolved.
      expect(mine.keys, where).toEqual(Object.keys(one.values));
      expect(unattributed(mine), where).toEqual([]);
    }
  }, 120_000);

  it.skipIf(!generated)('answers every synthetic declaration as the tools answer it', () => {
    for (const one of recorded().synthetic) {
      const quantities = decodedMap(one.quantities);
      const env = decodedMap(one.env);
      let mine: { answer: unknown } | null = null;
      let raised: Raised | null = null;
      try {
        mine = answered(one.definition, one.given, quantities, env);
      } catch (error) {
        raised = raisedAs(error);
      }
      expect(raised, one.name).toEqual(one.error);
      if (one.error === null) expect(mine?.answer, one.name).toEqual(expected(one));
    }
  }, 120_000);

  it.skipIf(!generated)('decides the V2, V3 and V8 cases of the rejection suite', () => {
    const material = recorded();
    const suite = rejections();
    // What this feature decides: the argument cases. The two V3 cases about a *quantity* are
    // feature 1.5's, and every other code belongs to the rules of 1.6b and 1.6c.
    const mine = suite.filter(
      (one) =>
        ['V2', 'V3', 'V8'].includes(one.expect) && !one.match.startsWith("quantity '"),
    );
    expect(mine.map((one) => one.document.split('/').pop()).sort()).toEqual([
      'v2-chunked-without-chunk.json',
      'v2-missing-required-field.json',
      'v2-missing-required.json',
      'v2-template-missing-argument.json',
      'v2-unknown-argument.json',
      'v2-yarn-without-attention-factor.json',
      'v3-boolean-string.json',
      'v3-cardinality-boolean.json',
      'v3-cardinality-negative.json',
      'v3-cardinality-string.json',
      'v3-chunk-with-causal.json',
      'v3-enum-out-of-set.json',
      'v3-eps-zero.json',
      'v3-inapplicable-field.json',
      'v3-kv-heads-zero.json',
      'v3-nested-enum.json',
      'v3-real-string.json',
      'v3-record-scalar.json',
      'v3-record-unknown-field.json',
      'v3-span-fractional.json',
      'v3-span-zero.json',
      'v3-streams-below-two.json',
      'v3-template-domain-interval.json',
      'v3-template-domain-set.json',
      'v3-window-with-chunked.json',
      'v3-window-with-none.json',
      'v8-kv-heads-non-divisor.json',
      'v8-streaming-bidirectional.json',
      'v8-top-k-above-experts.json',
    ]);
    for (const rejection of mine) {
      const path = join('tests', 'rejections', rejection.document);
      const cases = material.cases.filter((one) => one.document === path);
      expect(cases.length, path).toBeGreaterThan(0);
      const messages: string[] = [];
      for (const one of cases) {
        const model = documentAt(one.document);
        const definition = declarationOf(one, libraryFor(one.bases));
        const given = member(at(model, one.at) as PyRecord, 'arguments') as PyValue;
        const quantities = resolveQuantities(
          model,
          one.assignment === null ? undefined : decodedRecord(one.assignment),
        );
        for (const problem of answered(definition, given, quantities, decodedMap(one.env))
          .problems) {
          // `analyse` prefixes the instance and its site (feature 1.6b's); what this feature
          // answers is the refusal itself, and the fixture records the tools' own without the
          // prefix, since one recorded case stands for every environment resolving to it.
          messages.push(`[${problem.code}] ${problem.message}`);
        }
      }
      const document = material.documents.find((one) => one.path === path);
      expect(document, path).toBeDefined();
      expect([...new Set(messages)].sort(), path).toEqual(
        (document as DocumentCase).messages ?? [],
      );
      // The runner's own reading (`tests/run_rejections.py`): a line with the expected code
      // carrying the expected words.
      const decided = messages.filter((line) => line.startsWith(`[${rejection.expect}] `));
      expect(decided.length, path).toBeGreaterThan(0);
      expect(
        decided.some((line) => line.includes(rejection.match)),
        `${path}: ${rejection.match}`,
      ).toBe(true);
    }
  }, 120_000);

  it.skipIf(!generated)('refuses nothing on the corpus, and defaults what the corpus omits', () => {
    const material = recorded();
    const corpus = material.cases.filter((one) => one.document.startsWith('data/models/'));
    expect(corpus.length).toBeGreaterThan(300);
    for (const one of corpus) {
      expect(one.problems, `${one.document} @${one.where}`).toEqual([]);
      expect(
        one.invariants.filter((each) => each.verdict !== 'holds'),
        `${one.document} @${one.where}`,
      ).toEqual([]);
    }
    // A default the corpus never writes is the sheet's `kv_heads = 8 (default: heads → 32)`.
    expect(corpus.some((one) => one.facts.some((fact) => fact.source === 'default'))).toBe(true);
  });
});
