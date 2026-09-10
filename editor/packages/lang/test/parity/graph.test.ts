import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { parse } from '../../src/json/index.js';
import {
  analyseGraphText,
  formatSemanticProblems,
  keyOf,
  loadLibrary,
  rootSite,
  toPython,
  whereOfSite,
  type DeclaredSite,
  type GraphAnalysis,
  type InterfacePort,
  type InterfacePorts,
  type Library,
  type PyRecord,
  type PyValue,
  type ResolvedSite,
  type ShapeIdentity,
} from '../../src/index.js';
import { repositorySchemas } from '../schema/repository.js';
import { nodeSource } from '../library/source.js';
import { decodeRecord, encode } from './encoding.js';
import { raisedAs, type Raised } from './raised.js';
import { oracleGenerated, oracleOut, readOracleManifest, repositoryRoot } from './oracle.js';

// Parity of the graph (feature 1.6b): `analyse` from the quantities to V19 — the sites its guards
// keep, the primitives and arguments they resolve to, the value edges, the public interfaces, the
// indexing domains, the acyclicity, and the template instances expanded at their call sites.
//
// The oracle builds the expectation from the tools' *own source*: `validate.analyse` cut at the
// comment that opens the parameter bindings (feature 1.6c's half) and closed with the two blocks
// that end it — the interface ports a caller reads back, and the merge of an expanded template's
// counters. Nothing else is transcribed, and the truncation is held to the whole function on every
// document: its lines are a subsequence of `analyse`'s, and the counters and the interface ports
// it computes are `analyse`'s own. A `validate.py` that moves those comments makes the oracle die
// rather than record a half-truth.
//
// Recorded per document, and compared here: the refusals in the tools' words and order, the five
// counters, the interface ports with their evaluated shapes, and the graph itself — the resolved
// sites in walk order, the sites a guard removed, the edges, the topological order, the indexing
// domain of every port, what feeds and consumes each one, the evaluated `weights_location_prefix`
// of every template instance, and each expansion's own counters and ports.
//
// The 73 documents of `tests/rejections/models/` are read under `data/primitive-library`, which is
// what `tests/run_rejections.py` hands `validate.semantic`: 21 of them declare
// `../primitive-library/`, which from `tests/rejections/models/` names the directory of rejection
// *bases* and gathers nothing (feature 1.6a's finding, and the reason this suite reads them the
// runner's way rather than the loader's).

const inCI = process.env['CI'] !== undefined && process.env['CI'] !== '';
const generated = oracleGenerated();

/** A shape as the fixture writes one: the axis identity, and the extent tagged. */
type RecordedShape = [string, PyValue][];

/** One public interface port, as the fixture writes one. */
interface RecordedPort {
  kind: PyValue;
  stream: PyValue;
  shape: RecordedShape | null;
}

/** The interface ports of a document or of one expansion. */
interface RecordedPorts {
  inputs: Record<string, RecordedPort>;
  outputs: Record<string, RecordedPort>;
}

/** One document analysed as far as V19. */
interface DocumentCase {
  name: string;
  path: string;
  /** The primitive library bases the tools gathered for it, repository-relative. */
  bases: string[];
  assignment: PyRecord | null;
  /** The exception `analyse` raised, when it raised one instead of answering. */
  error: Raised | null;
  /** Whether `model.load` could read the document at all. */
  read?: boolean;
  /** Whether the whole `analyse` answered too, so that the truncation could be held to it. */
  whole?: boolean;
  /**
   * Whether an instance of this document took an *arbitrary* own domain.
   *
   * `mine = next(iter(agree))` reads a Python `set`, so an instance whose inputs disagree — a V5
   * refusal in itself — takes a hash-seeded one of them, and what follows the choice differs
   * between two runs of the tools themselves. The fixture records no `own`, `domains` or `ports`
   * for such a document; its refusals it does, the V5 line sorting what it lists.
   */
  arbitrary_own?: boolean;
  errors?: string[];
  stats?: Record<string, PyValue>;
  ports?: RecordedPorts;
  advisories?: string[];
  sites?: string[];
  absent?: string[];
  edges?: [string, string, string, string, string][];
  order?: string[];
  domains?: [string, string, string, string][];
  own?: [string, [string, string] | null][];
  seeds?: [string, string, string, string][];
  fragmented?: string[];
  producers?: [string, string, string][];
  consumed?: [string, string][];
  weights_prefixes?: [string, string][];
  expansions?: [string, Record<string, PyValue>, RecordedPorts][];
}

interface RejectionCase {
  document: string;
  expect: string;
  match: string;
}

/** The fixture, read as a document is read: `1e-05` a float, `4096` a whole number. */
function recorded(): DocumentCase[] {
  const text = readFileSync(join(oracleOut, 'graph', 'index.json'), 'utf8');
  return (toPython(parse(text)) as unknown as { documents: DocumentCase[] }).documents;
}

function rejections(): RejectionCase[] {
  const file = join(oracleOut, 'rejections', 'models.json');
  return (JSON.parse(readFileSync(file, 'utf8')) as { cases: RejectionCase[] }).cases;
}

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

/** What the port answers for one recorded document. */
function analysed(one: DocumentCase): GraphAnalysis {
  const text = readFileSync(join(repositoryRoot, one.path), 'utf8');
  return analyseGraphText(text, libraryFor(one.bases), {
    ...(one.assignment === null ? {} : { assignment: decodeRecord(one.assignment) }),
  });
}

/** A shape as the fixture writes one, from the port's own. */
function shapeOf(shape: ShapeIdentity | null): unknown {
  return shape === null ? null : shape.map((axis) => [axis[0], encode(axis[1])]);
}

/** The interface ports as the fixture writes them. */
function portsOf(ports: InterfacePorts): unknown {
  const side = (entries: ReadonlyMap<string, InterfacePort>): unknown =>
    Object.fromEntries(
      [...entries].map(([name, entry]) => [
        name,
        { kind: encode(entry.kind), stream: encode(entry.stream), shape: shapeOf(entry.shape) },
      ]),
    );
  return { inputs: side(ports.inputs), outputs: side(ports.outputs) };
}

/**
 * A domain kind, a stream or a port name, which the fixture writes as text rather than tagged.
 *
 * Every one of them is a string of the grammar — an enum value, an identifier — so the fixture
 * writes it bare; a value that is not one is a defect this reading names rather than passes on.
 */
function label(value: PyValue): string {
  if (typeof value !== 'string') {
    throw new TypeError(`a domain kind, stream or port name is a string, not ${typeof value}`);
  }
  return value;
}

/** A counter map as the fixture writes one. */
function statsOf(stats: ReadonlyMap<string, PyValue>): unknown {
  return Object.fromEntries([...stats].map(([name, one]) => [name, encode(one)]));
}

/**
 * The whole answer in the fixture's shape.
 *
 * The kinds and streams of a domain are the schema's own strings, which the fixture writes as
 * text rather than tagged; the reading here says so, and a value that is not one would show up
 * as a mismatch rather than pass unnoticed.
 */
function answered(answer: GraphAnalysis, arbitrary = false): Record<string, unknown> {
  const settled = <T>(value: T): T | undefined => (arbitrary ? undefined : value);
  return {
    errors: formatSemanticProblems([...answer.problems]),
    stats: statsOf(answer.stats),
    ports: settled(portsOf(answer.ports)),
    advisories: [...answer.advisories],
    sites: [...answer.resolved.values()].map((site) => whereOfSite(site.key)),
    absent: [...answer.absent.values()].map((key) => whereOfSite(key)).sort(),
    edges: answer.edges.map((edge) => [
      whereOfSite(edge.from),
      label(edge.fromPort),
      whereOfSite(edge.to),
      label(edge.toPort),
      edge.binding,
    ]),
    order: answer.order.map((key) => whereOfSite(key)),
    domains: settled(
      [...answer.domains.values()].map((entry) => [
        whereOfSite(entry.site),
        label(entry.port),
        label(entry.domain[0]),
        label(entry.domain[1]),
      ]),
    ),
    own: settled(
      [...answer.own].map(([key, mine]) => [
        whereOfSite((answer.resolved.get(key) as ResolvedSite).key),
        mine === null ? null : [label(mine[0]), label(mine[1])],
      ]),
    ),
    seeds: [...answer.seeds.values()].map((entry) => [
      whereOfSite(entry.site),
      label(entry.port),
      label(entry.domain[0]),
      label(entry.domain[1]),
    ]),
    fragmented: [...answer.fragmented].sort(),
    producers: [...answer.producers.values()].map((entry) => [
      whereOfSite(entry.site),
      label(entry.port),
      entry.by,
    ]),
    consumed: [...answer.consumed.values()]
      .map((entry) => [whereOfSite(entry.site), label(entry.port)])
      .sort(compareRows),
    weights_prefixes: [...answer.weightsPrefixes].map(([key, prefix]) => [
      whereOfSite((answer.sites.get(key) as DeclaredSite).key),
      prefix,
    ]),
    expansions: [...answer.subResults].map(([key, sub]) => [
      whereOfSite((answer.sites.get(key) as DeclaredSite).key),
      statsOf(sub.stats),
      portsOf(sub.ports),
    ]),
  };
}

/** The same reading of a recorded document, so that the two are compared field by field. */
function expected(one: DocumentCase): Record<string, unknown> {
  const arbitrary = one.arbitrary_own === true;
  const settled = <T>(value: T): T | undefined => (arbitrary ? undefined : value);
  return {
    errors: one.errors ?? [],
    stats: one.stats ?? {},
    ports: settled(one.ports ?? { inputs: {}, outputs: {} }),
    advisories: one.advisories ?? [],
    sites: one.sites ?? [],
    absent: one.absent ?? [],
    edges: (one.edges ?? []).map((row) => [...row]),
    order: one.order ?? [],
    domains: settled((one.domains ?? []).map((row) => [...row])),
    own: settled(
      (one.own ?? []).map(([where, mine]) => [where, mine === null ? null : [...mine]]),
    ),
    seeds: (one.seeds ?? []).map((row) => [...row]),
    fragmented: one.fragmented ?? [],
    producers: (one.producers ?? []).map((row) => [...row]),
    consumed: (one.consumed ?? []).map((row) => [...row]),
    weights_prefixes: (one.weights_prefixes ?? []).map((row) => [...row]),
    expansions: (one.expansions ?? []).map((row) => [...row]),
  };
}

/** `sorted([...])` over rows of two strings: Python compares the first, then the second. */
function compareRows(one: readonly string[], other: readonly string[]): number {
  for (let index = 0; index < one.length; index += 1) {
    const mine = one[index] as string;
    const theirs = other[index] as string;
    if (mine !== theirs) return mine < theirs ? -1 : 1;
  }
  return 0;
}

describe('the graph, its edges and its interfaces against the tools', () => {
  it.runIf(inCI)('has the oracle to compare with', () => {
    expect(generated).toBe(true);
  });

  it.skipIf(!generated)('records every model document of the repository', () => {
    const documents = recorded();
    const manifest = readOracleManifest() as unknown as { graph?: { documents: number } };
    expect(manifest.graph?.documents).toBe(documents.length);
    // The corpus and the template, and the 73 rejection documents.
    expect(documents).toHaveLength(15 + 73);
    // The whole `analyse` answered for every one the truncation answered for, so every recorded
    // expectation was held to it.
    expect(documents.filter((one) => one.error === null && one.whole !== true)).toEqual([]);
    // Every branch this feature decides is reached by some document.
    expect(documents.some((one) => (one.absent ?? []).length > 0)).toBe(true);
    expect(documents.some((one) => (one.expansions ?? []).length > 0)).toBe(true);
    expect(documents.some((one) => (one.weights_prefixes ?? []).length > 0)).toBe(true);
    expect(documents.some((one) => (one.fragmented ?? []).length > 0)).toBe(true);
    expect(documents.some((one) => one.read === false)).toBe(true);
    expect(documents.some((one) => one.error !== null)).toBe(true);
    // One document of the repository, and one only, reaches the place where the tools choose an
    // instance's own domain out of a Python set — a coin flip no implementation can be at parity
    // with. A second one appearing is a finding, not a silent exemption.
    expect(documents.filter((one) => one.arbitrary_own === true).map((one) => one.path)).toEqual([
      'tests/rejections/models/v5-fusion-without-join.json',
    ]);
    const codes = new Set(
      documents.flatMap((one) => (one.errors ?? []).map((line) => line.slice(0, line.indexOf(']') + 1))),
    );
    for (const code of ['[V1]', '[V4]', '[V5]', '[V7]', '[V10]', '[V13]', '[V19]']) {
      expect(codes, code).toContain(code);
    }
  });

  it.skipIf(!generated)('answers every document as the tools answer it', () => {
    for (const one of recorded()) {
      let answer: GraphAnalysis | null = null;
      let raised: Raised | null = null;
      try {
        answer = analysed(one);
      } catch (error) {
        raised = raisedAs(error);
      }
      expect(raised, one.path).toEqual(one.error);
      if (one.error !== null) continue;
      expect(answered(answer as GraphAnalysis, one.arbitrary_own === true), one.path).toEqual(
        expected(one),
      );
    }
  }, 300_000);

  it.skipIf(!generated)('refuses nothing on the corpus, and counts what the tools count', () => {
    const corpus = recorded().filter((one) => one.path.startsWith('data/models/'));
    expect(corpus).toHaveLength(15);
    for (const one of corpus) {
      expect(one.errors, one.path).toEqual([]);
      const answer = analysed(one);
      expect(formatSemanticProblems([...answer.problems]), one.path).toEqual([]);
      // The tools' `stats`, key for key — this feature's five of them.
      expect(statsOf(answer.stats), one.path).toEqual(one.stats);
      expect([...answer.stats.keys()].sort(), one.path).toEqual([
        'composite_primitives',
        'dag',
        'edges',
        'instances',
        'resolved_domains',
      ]);
      expect(answer.stats.get('dag'), one.path).toBe(true);
      // Every site of the graph is in the topological order, and every edge joins two of them.
      expect(answer.order).toHaveLength(answer.resolved.size);
      for (const edge of answer.edges) {
        expect(answer.resolved.has(keyOf(edge.from)), one.path).toBe(true);
        expect(answer.resolved.has(keyOf(edge.to)), one.path).toBe(true);
      }
    }
  }, 300_000);

  it.skipIf(!generated)('chooses an own domain in document order where the tools choose one at all', () => {
    // The one place the tools are not reproducible: an instance whose inputs disagree takes
    // `next(iter(agree))` out of a Python set. The port takes the first in *document order* —
    // the order the primitive declares its input ports — which is deterministic, and it is the
    // reading a set that kept insertion order would give.
    const one = recorded().find(
      (each) => each.path === 'tests/rejections/models/v5-fusion-without-join.json',
    ) as DocumentCase;
    expect(one, 'v5-fusion-without-join').toBeDefined();
    const first = analysed(one);
    const again = analysed(one);
    const domainOf = (answer: GraphAnalysis, where: string): readonly PyValue[] | null => {
      for (const [key, mine] of answer.own) {
        if (whereOfSite((answer.resolved.get(key) as ResolvedSite).key) === where) return mine;
      }
      return null;
    };
    expect(domainOf(first, 'fuse')).toEqual(domainOf(again, 'fuse'));
    // `residual.add` declares `a` before `b`, and `a` carries the token stream `tokens`.
    expect(domainOf(first, 'fuse')).toEqual(['token', 'tokens']);
    // The refusal itself is stable, because the V5 line sorts the domains it lists.
    expect(
      formatSemanticProblems([...first.problems]).filter((line) =>
        line.includes('inputs in different domains'),
      ),
    ).toEqual([
      "[V5] residual.add@fuse: inputs in different domains [('token', 'audio'), " +
        "('token', 'tokens')], and no domain_transform declares it",
    ]);
  });

  it.skipIf(!generated)('validates the template under its documented assignment', () => {
    const documents = recorded();
    const template = documents.find((one) => one.path.includes('decoder-causal-yarn'));
    expect(template, 'the corpus template').toBeDefined();
    const one = template as DocumentCase;
    // "A template denotes one graph per admissible assignment (§4.6)": the assignment is the one
    // `tests/signature.py` records, which the oracle reads rather than copying (plan §0.5).
    expect(one.assignment).not.toBeNull();
    expect(one.errors).toEqual([]);
    const answer = analysed(one);
    expect(formatSemanticProblems([...answer.problems])).toEqual([]);
    expect(statsOf(answer.stats)).toEqual(one.stats);
    // Its public interface is what a call site reads back as the instance's ports.
    expect(portsOf(answer.ports)).toEqual(one.ports);
    expect([...answer.ports.inputs.keys()]).toEqual(['hidden']);
    expect([...answer.ports.outputs.keys()]).toEqual(['hidden_out']);
  });

  it.skipIf(!generated)(
    "expands the composite's template into what the flat document writes",
    () => {
      const documents = recorded();
      const flat = documents.find((one) => one.name === 'shieldstral-3b') as DocumentCase;
      const composite = documents.find(
        (one) => one.name === 'shieldstral-3b-composite',
      ) as DocumentCase;
      expect(flat, 'shieldstral-3b').toBeDefined();
      expect(composite, 'shieldstral-3b-composite').toBeDefined();
      const written = analysed(flat);
      const through = analysed(composite);
      expect(formatSemanticProblems([...written.problems])).toEqual([]);
      expect(formatSemanticProblems([...through.problems])).toEqual([]);

      // `tests/run_templates.py`'s claim at this stage: what the template expands to at the call
      // site is what the flat document writes out by hand. The counters this feature computes say
      // it of the *sites* — the composite's `decoder` is the template's, and the flat document's
      // sites of that composition are the same count — and the difference in the totals is the
      // one site the expansion stands for.
      const expansions = [...through.subResults.values()];
      expect(expansions).toHaveLength(1);
      const expansion = expansions[0] as GraphAnalysis;
      const decoder = [...written.resolved.values()].filter(
        (site) => site.key.kind === 'gen' && site.key.composition === 'decoder',
      );
      expect(expansion.stats.get('instances')).toBe(BigInt(decoder.length));
      expect(through.stats.get('instances')).toBe(
        (written.stats.get('instances') as bigint) + 1n,
      );
      expect(through.stats.get('composite_primitives')).toBe(1n);
      expect(written.stats.get('composite_primitives')).toBe(0n);

      // The counters are merged, not the caller's alone: the sites of the expansion are counted
      // with the caller's, and the caller declares fewer sites than it reports.
      expect(BigInt(through.resolved.size)).toBeLessThan(
        through.stats.get('instances') as bigint,
      );
      expect(BigInt(through.resolved.size) + (expansion.stats.get('instances') as bigint)).toBe(
        through.stats.get('instances'),
      );

      // The instance's ports are the expansion's interfaces (§4.6), so V4 and V5 crossed the
      // boundary: the template's `hidden` and `hidden_out` carry the residual width.
      expect(through.resolved.has(keyOf(rootSite('text'))), 'the instance `text`').toBe(true);
      expect(portsOf(expansion.ports)).toEqual(composite.expansions?.[0]?.[2]);
      // And its weights are located through the instance's prefix (§3.4).
      expect([...through.weightsPrefixes.values()]).toEqual(['language_model.model.']);
    },
    300_000,
  );

  it.skipIf(!generated)('decides the V1, V4, V5, V6, V7, V10, V13 and V19 cases', () => {
    const documents = recorded();
    const suite = rejections();
    // What this feature decides, case by case. V6 has no fixture in the suite at all — the one
    // rule of the eight the corpus and `tests/rejections` never exercise (a finding); the unit
    // suite carries a cycle of its own.
    const mine = [
      'v1-unknown-primitive.json',
      'v1-unknown-stream.json',
      'v1-unpinned-version.json',
      'v10-binding-when-unresolved.json',
      'v10-site-when-unresolved.json',
      'v13-dangling-output.json',
      'v19-join-at-a-kind-the-stream-lacks.json',
      'v4-input-feeds-unequal-ports.json',
      'v5-fusion-without-join.json',
      'v5-interface-kind-mismatch.json',
      'v7-binding-when-excludes-edge.json',
      'v7-input-fed-twice.json',
    ];
    for (const name of mine) {
      const rejection = suite.find((one) => one.document === `models/${name}`);
      expect(rejection, name).toBeDefined();
      const path = join('tests', 'rejections', (rejection as RejectionCase).document);
      const one = documents.find((each) => each.path === path);
      expect(one, path).toBeDefined();
      const answer = analysed(one as DocumentCase);
      const lines = formatSemanticProblems([...answer.problems]);
      expect(lines, path).toEqual((one as DocumentCase).errors);
      // The runner's own reading (`tests/run_rejections.py`): a line with the expected code
      // carrying the expected words.
      const decided = lines.filter((line) =>
        line.startsWith(`[${(rejection as RejectionCase).expect}] `),
      );
      expect(decided.length, path).toBeGreaterThan(0);
      expect(
        decided.some((line) => line.includes((rejection as RejectionCase).match)),
        `${path}: ${(rejection as RejectionCase).match}`,
      ).toBe(true);
    }
    // Every model case of the suite the earlier features do not decide is one of these, or is
    // decided by a rule feature 1.6c owns: the codes are the partition, and it is stated here so
    // that a case moving between features shows up.
    const undecided = suite.filter(
      (one) =>
        !mine.includes(one.document.split('/').pop() as string) &&
        ['V1', 'V4', 'V5', 'V6', 'V13', 'V19'].includes(one.expect),
    );
    expect(undecided.map((one) => one.document)).toEqual([
      // `model.load` refuses it before any rule of this feature (feature 1.4).
      'models/v1-scoped-unknown-site.json',
      // `check_quantities` refuses it (feature 1.5).
      'models/v1-derivation-undeclared-quantity.json',
    ]);
  }, 300_000);
});
