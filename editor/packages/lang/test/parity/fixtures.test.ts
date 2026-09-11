import { readFileSync, readdirSync } from 'node:fs';
import { basename, join } from 'node:path';

import { describe, expect, it } from 'vitest';

import {
  analyseText,
  basesOf,
  checkCheckpoint,
  d3,
  derivationGraph,
  derive,
  expand,
  loadLibrary,
  parse,
  pyStr,
  serialize,
  toJsonValue,
  toPython,
  type CheckpointHeaders,
  type CheckpointProblem,
  type HeaderEntry,
  type Library,
  type PyRecord,
  type PyValue,
} from '../../src/index.js';
import { nodeSource } from '../library/source.js';
import { repositorySchemas } from '../schema/repository.js';
import { raisedAs } from './raised.js';
import { oracleGenerated, oracleOut, readOracleManifest, repositoryRoot } from './oracle.js';

// Parity over the acceptance fixtures (feature 1.13, finding F8).
//
// Every other parity suite is held to the repository's own files. These are not in the repository
// — they are in `editor/tests/fixtures/`, written for this suite — because the grammar they use is
// written nowhere in `data/models/`, `data/primitive-library/` or `tests/rejections/`, measured
// document by document by the features before this one:
//
//   * a composition of **several indices** (feature 1.4: none of the corpus's 23 index ranges is
//     one of a pair), and with it the fact that a layer graph split is emitted for a composition of
//     one index only — `len(comp[1]) == 1` in `derive._structural_graph_splits`;
//   * a **top-level `for_each`** as an author writes it: the corpus writes none, and the 706 its
//     normalised reading carries are `model.normalise` hoisting its 706 scoped rules, every one of
//     them over a composition's own indices — this one is over an index range of its own, whose
//     `start` is not 0 and whose `step` is not 1, where the corpus has 23 ranges of `0 … stop
//     step 1` and nothing else;
//   * **`constants`**: a document's `constants` map, a `bindings.constants` rule, its scoped form,
//     and the one unit of the repository declaring a **constant slot** (features 1.4, 1.6c);
//   * a **root-level `when`**, on a root instance and on a top-level binding (feature 1.7), with a
//     guard that fires, one that does not, and a binding naming an instance a guard removed —
//     written in all five forms of the model condition language, where the corpus's 107 model
//     conditions are `compare` and `all` alone;
//   * a **`concat`** location, written nowhere (features 1.6c, 1.8a, 1.9) and, with it, the
//     `artifact.check` branch that sums a concat's parts;
//   * a `stack` along an axis that is **not the first** — every stack of the corpus is at
//     dimension 0 (feature 1.9);
//   * the five operators the corpus and the reference base never write — `divide`, `min`, `max`,
//     `negate`, `absolute` (feature 1.8d) — a **boolean** and a **physical** quantity and an
//     **upper-inclusive** domain bound, of which the corpus's 215 quantities carry none (the one
//     upper bound it writes, the template's `eps < 1`, is exclusive; feature 1.5), and a cost
//     entry **per invocation**, which none of the reference base's eight declares (1.8d);
//   * a caller whose **public interfaces name a template instance**, which `inputs_at` and
//     `outputs_at` resolve through (features 1.8a and 1.8c), in two readings so that the dtype a
//     selector inside the instance takes is told apart from the template's own and the role's;
//   * a **port shape that does not resolve**, on the input side and on the output side, where the
//     tools guard one figure and not the other (features 1.8c, 1.8e, 1.11).
//
// The last three sets had no oracle at all until this feature: their answers were read off
// `tools/derive.py` once and written into the unit suites, because no repository file produced
// them (feature 1.12's register of expectations that are not the oracle's). They are documents
// now, the oracle runs the tools over them like any other, and this suite is where the comparison
// happens — which is what makes them expectations nobody has to refresh.
//
// The comparison is every other suite's: `--validate`'s printed line, D1 compared as **bytes**
// (feature 1.7: `--d1` writes what the core's serializer writes), the whole derived document
// deep-equal and then serialized and compared as text (feature 1.8b), and `artifact.check` over a
// checkpoint the oracle synthesised from each fixture's own D3 (feature 1.9). One document does
// not derive at all and its answer is the exception the tools raised, compared as feature 1.7
// compares `d1.emit`'s refusals.

const inCI = process.env['CI'] !== undefined && process.env['CI'] !== '';
const generated = oracleGenerated();

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

/** One fixture document as the oracle recorded it. */
interface FixtureCase {
  /** The file's stem: `grid-composition`. */
  name: string;
  /** Its path, relative to the repository root. */
  path: string;
  /** The bases `bases_of` resolved for it, repository-relative. */
  bases: string[];
  /** What `derive.products` raised, when it raised instead of answering. */
  raised?: { type: string; message: string };
  /** D1 and the derived document, relative to the oracle's output directory. */
  d1?: string;
  derived?: string;
  /** The checkpoint synthesised from its own D3, when it locates anything. */
  headers?: string;
  tensors?: number;
  located?: number;
  physical?: number;
}

interface FixtureIndex {
  documents: FixtureCase[];
  /** Every unit of the fixtures' own bases, relative to `editor/tests/fixtures/`. */
  units: string[];
  cases: {
    name: string;
    document: string;
    headers_of: string;
    edits: { op: string; tensor: string; entry?: unknown }[];
    errors: string[];
    advisories: string[];
    stats: { located: number; physical: number; unnamed: number };
  }[];
  validate: string;
}

let held: FixtureIndex | undefined;

function recorded(): FixtureIndex {
  held ??= JSON.parse(
    readFileSync(join(oracleOut, 'fixtures', 'index.json'), 'utf8'),
  ) as FixtureIndex;
  return held;
}

/** A file the oracle wrote, read as a document is read: `1e-05` a float, `8` a whole number. */
function read(relative: string): PyValue {
  return toPython(parse(readFileSync(join(oracleOut, relative), 'utf8')));
}

/** The text of one fixture document. */
function text(one: FixtureCase): string {
  return readFileSync(join(repositoryRoot, one.path), 'utf8');
}

/** The library one fixture document resolves from, gathered once per set of bases. */
function libraryOf(one: FixtureCase): Library {
  return libraryFor([...basesOf(one.path, toPython(parse(text(one)))).bases]);
}

/** `{dtype, shape, file}` as the oracle records it, as the core's `HeaderEntry`. */
function headersOf(value: PyValue): CheckpointHeaders {
  const out: Record<string, HeaderEntry> = {};
  for (const [name, entry] of Object.entries(value as PyRecord)) {
    const record = entry as PyRecord;
    out[name] = {
      dtype: record['dtype'] as string,
      known: true,
      shape: record['shape'] as readonly PyValue[],
      file: record['file'] as string,
    };
  }
  return out;
}

/** The case's edits, applied as `_edited` applies them. */
function edited(
  headers: CheckpointHeaders,
  edits: readonly { op: string; tensor: string; entry?: unknown }[],
): CheckpointHeaders {
  const out: Record<string, HeaderEntry> = { ...headers };
  for (const edit of edits) {
    if (edit.op === 'delete') {
      delete out[edit.tensor];
    } else {
      const entry = toPython(parse(JSON.stringify(edit.entry)));
      out[edit.tensor] = headersOf({ [edit.tensor]: entry })[edit.tensor] as HeaderEntry;
    }
  }
  return out;
}

function messages(problems: readonly CheckpointProblem[]): string[] {
  return problems.map((problem) => problem.message);
}

describe('the acceptance fixtures against the tools', () => {
  it.runIf(inCI)('has the oracle to compare with', () => {
    expect(generated).toBe(true);
  });

  it.skipIf(!generated)('records every document of `editor/tests/fixtures/models`', () => {
    const manifest = readOracleManifest() as unknown as {
      fixtures?: { index: string; documents: number; cases: number; validate: string };
    };
    expect(manifest.fixtures, 'the manifest names no fixtures product').toBeDefined();
    const written = recorded();
    // The set is the directory's, so a fixture added without an expectation fails here rather
    // than passing unread — which is what "the parity job covers them" asks of this suite.
    const onDisk = readdirSync(join(repositoryRoot, 'editor', 'tests', 'fixtures', 'models'))
      .filter((name) => name.endsWith('.json'))
      .map((name) => name.slice(0, -'.json'.length))
      .sort();
    expect(written.documents.map((one) => one.name).sort()).toEqual(onDisk);
    expect(onDisk.length).toBeGreaterThanOrEqual(8);
    // Exactly one of them has no products: `scratch-blank`, whose input port's extent does not
    // resolve. A second would be a finding, not a silent exemption.
    expect(written.documents.filter((one) => one.raised !== undefined).map((one) => one.name))
      .toEqual(['scratch-blank']);
    // Two bases of their own, five units between them, and no third.
    expect(written.units).toEqual([
      'base/primitive-library.json',
      'base/primitives/fixture/position_bias/1.0.0.json',
      'scratch/primitive-library.json',
      'scratch/primitives/scratch/blank/1.0.0.json',
      'scratch/primitives/scratch/fed/1.0.0.json',
    ]);
  });

  it.skipIf(!generated)('gathers each fixture’s bases where the tools gathered them', () => {
    for (const one of recorded().documents) {
      const bases = [...basesOf(one.path, toPython(parse(text(one)))).bases];
      expect(bases, one.name).toEqual(one.bases);
      // The loader takes the fixtures' own bases beside the reference base with no refusal: a unit
      // of a base the repository does not ship is read exactly as one it does.
      expect(libraryOf(one).problems.map((problem) => problem.message), one.name).toEqual([]);
    }
  });

  it.skipIf(!generated)('validates every one of them, with the tools’ own counters', () => {
    const lines = readFileSync(join(oracleOut, recorded().validate), 'utf8').split('\n');
    const count = recorded().documents.length;
    expect(lines[0]).toBe(`validate  ${String(count)} document(s)`);
    expect(lines.filter((line) => line.trim().length > 0).at(-1)).toBe(
      `  ${String(count)}/${String(count)} pass`,
    );
    for (const one of recorded().documents) {
      const analysis = analyseText(text(one), libraryOf(one));
      expect(analysis.problems, one.name).toEqual([]);
      const summary = [...analysis.stats]
        .map(([name, value]) => `${name}=${pyStr(value)}`)
        .join(' | ');
      const name = basename(one.path);
      const theirs = lines.find((line) => line.startsWith(`  ${name} `));
      expect(theirs, `no --validate line for ${name}`).toBeDefined();
      expect(`  ${name.padEnd(34)} ok   ${summary}`, one.name).toBe(theirs);
    }
  });

  it.skipIf(!generated)('expands each of them to the bytes `--d1` wrote', () => {
    for (const one of recorded().documents) {
      if (one.d1 === undefined) continue;
      const tree = parse(text(one));
      const answer = expand(toPython(tree), libraryOf(one));
      const expected = read(one.d1);
      expect(answer, one.name).toEqual(expected);
      expect(serialize(toJsonValue(answer)), one.name).toBe(serialize(toJsonValue(expected)));
    }
  });

  it.skipIf(!generated)('derives each of them to the document `--derive` wrote', () => {
    for (const one of recorded().documents) {
      if (one.derived === undefined) continue;
      const answer = derive(parse(text(one)), { schemas, library: libraryOf(one) });
      const expected = read(one.derived);
      expect(answer, one.name).toEqual(expected);
      expect(serialize(toJsonValue(answer)), one.name).toBe(serialize(toJsonValue(expected)));
    }
  });

  it.skipIf(!generated)('refuses the one they refuse, in the words they raised it with', () => {
    // The asymmetry the tools carry: a *produced* value's byte size is guarded and a public
    // input's is not, so a port shape that does not resolve raises out of the whole derivation
    // (feature 1.8c) — while the sibling document, whose *output* port carries the same extent,
    // derives with the blanks the derived schema admits. Both readings are here, and the refusal
    // is the tools' own text.
    for (const one of recorded().documents) {
      if (one.raised === undefined) continue;
      let raised: { type: string; message: string } | null = null;
      try {
        derive(parse(text(one)), { schemas, library: libraryOf(one) });
      } catch (error) {
        raised = raisedAs(error);
      }
      expect(raised, one.name).toEqual(one.raised);
    }
  });

  it.skipIf(!generated)('checks each of them against a checkpoint built from its own D3', () => {
    const checkpoints = new Map<string, CheckpointHeaders>();
    const inventories = new Map<string, PyRecord>();
    for (const one of recorded().documents) {
      if (one.headers === undefined) continue;
      checkpoints.set(one.name, headersOf(read(one.headers)));
      const graph = derivationGraph(parse(text(one)), { schemas, library: libraryOf(one) }).graph;
      inventories.set(one.name, d3(graph, libraryOf(one)));
    }
    const kinds = new Set<string>();
    for (const one of recorded().cases) {
      const headers = edited(checkpoints.get(one.headers_of) as CheckpointHeaders, one.edits);
      const answer = checkCheckpoint({ d3: inventories.get(one.document) as PyValue }, headers);
      expect(messages(answer.errors), one.name).toEqual(one.errors);
      expect(
        messages(answer.warnings.filter((problem) => problem.kind === 'unnamed')),
        one.name,
      ).toEqual(one.advisories);
      expect(answer.stats, one.name).toEqual(one.stats);
      for (const problem of [...answer.errors, ...answer.warnings]) kinds.add(problem.kind);
    }
    expect(recorded().cases.length).toBeGreaterThan(20);
    // The kinds the fixtures reach. `stack` is not among them and cannot be: the stack branch
    // refuses a count that is not the axis's extent, and a count is a property of the **D3**,
    // which builds exactly `extent` parts — no edit of a checkpoint can produce it, and the
    // oracle's hand-built `stack-count-wrong` form is where it lives. `slice` is not among them
    // because no fixture writes a slice: the corpus writes 486 of them, so it is not a gap.
    expect([...kinds].sort()).toEqual(['absent', 'concat', 'dtype', 'shape', 'unnamed']);
    const concat = [...(inventories.get('grid-composition') as PyRecord)['tensors'] as PyValue[]]
      .map((tensor) => (tensor as PyRecord)['location'])
      .filter((location) => location !== undefined && 'concat' in (location as PyRecord));
    expect(concat, 'no fixture locates a concat').toHaveLength(1);
    const stacked = [...(inventories.get('declared-constant') as PyRecord)['tensors'] as PyValue[]]
      .map((tensor) => (tensor as PyRecord)['location'] as PyRecord | undefined)
      .filter((location) => location !== undefined && 'stack' in location)
      .map((location) => ((location as PyRecord)['stack'] as PyRecord)['dim']);
    expect(stacked, 'no fixture stacks along an axis that is not the first').toContain(1n);
  });
});
