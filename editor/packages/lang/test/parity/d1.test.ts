import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import {
  basesOf,
  expand,
  expandText,
  loadLibrary,
  parse,
  serialize,
  toJsonValue,
  toPython,
  type Library,
  type PyRecord,
  type PyValue,
} from '../../src/index.js';
import { nodeSource } from '../library/source.js';
import { repositorySchemas } from '../schema/repository.js';
import { applyEdits, type Edit } from './edits.js';
import { decode, decodeRecord } from './encoding.js';
import { raisedAs, type Raised } from './raised.js';
import { oracleGenerated, oracleOut, readOracleManifest, repositoryRoot } from './oracle.js';

// Parity of D1 (feature 1.7): the expanded graph of §5.2 as `tools/d1.py` emits it.
//
// The comparison is over **bytes**, not over a reading. `--d1` writes
// `json.dumps(document, indent=2, ensure_ascii=False)` and the core's serializer reproduces
// exactly that (D12, feature 0.3), so a byte comparison decides the structure, the listing order,
// the member order *and* the integer/float distinction in one — `1` against `1.0`, `1e-05` against
// `0.00001` — which a deep-equal over a JSON reading cannot. A deep-equal over the two readings is
// asserted beside it, because it is what the plan's parity criterion names and because it names
// the node that moved when the bytes differ.
//
// Three sets are compared:
//
//   * the corpus and the template, against `out/d1/<name>.d1.json`, the tools' own output;
//   * every model document of the repository — the 73 of `tests/rejections/models/` included —
//     against `out/expansion/emitted/<slug>.d1.json`, or against the exception the tools raised.
//     `d1.emit` does not validate: it emits a graph for a document V2 or V8 would refuse, and six
//     kinds of refusal leave it with no graph at all — an undecidable site guard, an undecidable
//     binding guard, a primitive the base does not hold, an edge whose destination is not a node,
//     a scoped-binding refusal and a duplicate member name — beside the `TypeError` an interface
//     written in the old form raises out of Python's own subscript (feature 1.6b's note). The set
//     is the one features 1.6b and 1.6c read, under the base `tests/run_rejections.py` hands the
//     validator;
//   * the edited cases of `tests/oracle/d1_cases.py`, for the branches no document of the
//     repository reaches — a root instance's guard kept, removed and undecidable, a composition
//     over two indices, a value cycle, an undecidable `across_positions`, a record argument that
//     is not a record, a template instance a guard removes, a prefix whose index does not resolve,
//     and a template read with no assignment at all.

const inCI = process.env['CI'] !== undefined && process.env['CI'] !== '';
const generated = oracleGenerated();

/** One model document of the repository, expanded (or refused) by the tools. */
interface DocumentCase {
  name: string;
  path: string;
  /** The primitive library bases the tools gathered for it, repository-relative. */
  bases: string[];
  assignment: PyRecord | null;
  /** The exception `d1.emit` raised, when it raised one instead of answering. */
  error: Raised | null;
  /** The emitted document, relative to the oracle's output directory. */
  emitted?: string;
  /** The figures, as the fixture's reading gives them: whole numbers, so `bigint`. */
  nodes?: bigint;
  edges?: bigint;
  /** The template instances D1 listed, in order; absent when there is none. */
  instances?: string[];
}

/** One edited document, in the idiom features 1.3–1.6 established. */
interface EditedCase {
  name: string;
  source: string;
  edits: Edit[];
  error: Raised | null;
  emitted?: string;
  nodes?: bigint;
  edges?: bigint;
  instances?: string[];
}

interface Recorded {
  documents: DocumentCase[];
  cases: EditedCase[];
}

function recorded(): Recorded {
  const text = readFileSync(join(oracleOut, 'expansion', 'index.json'), 'utf8');
  return toPython(parse(text)) as unknown as Recorded;
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

/** The bases a corpus document declares, as `load_for` resolves them. */
function declaredBases(path: string): string[] {
  const document = toPython(parse(readFileSync(join(repositoryRoot, path), 'utf8')));
  return [...basesOf(path, document).bases];
}

/** The emitted document as the tools write it: `json.dumps(…, indent=2, ensure_ascii=False)`. */
function written(document: PyRecord): string {
  return serialize(toJsonValue(document));
}

/** The text the oracle recorded for one case, and the reading of it. */
function expectedText(file: string): string {
  return readFileSync(join(oracleOut, file), 'utf8');
}

/** The two comparisons, in the order that makes a failure readable. */
function agrees(answer: PyRecord, file: string, where: string): void {
  const text = expectedText(file);
  // The reading first: a structural difference names the node, where a byte difference names a
  // line. `toPython` keeps the integer/float distinction, so the deep-equal decides it too.
  expect(answer, where).toEqual(toPython(parse(text)));
  expect(written(answer), where).toBe(text);
}

describe('D1 against the tools', () => {
  it.runIf(inCI)('has the oracle to compare with', () => {
    expect(generated).toBe(true);
  });

  it.skipIf(!generated)('emits the corpus and the template byte for byte', () => {
    // The manifest names what `--d1` wrote for each of them; the assignment is read from the
    // `expansion` fixture beside it, where it carries the tagged encoding that keeps `3072` an
    // integer and `1e-05` a float (feature 1.2). The bases are the document's own, resolved as
    // `load_for` resolves them — the path `--d1` itself took.
    const emitted = new Map(readOracleManifest().documents.map((one) => [one.name, one.d1]));
    const corpus = recorded().documents.filter((one) => one.path.startsWith('data/models/'));
    expect(corpus).toHaveLength(15);
    expect([...emitted.keys()].sort()).toEqual(corpus.map((one) => one.name).sort());
    for (const one of corpus) {
      const text = readFileSync(join(repositoryRoot, one.path), 'utf8');
      const answer = expandText(text, libraryFor(declaredBases(one.path)), {
        ...(one.assignment === null ? {} : { assignment: decodeRecord(one.assignment) }),
      });
      agrees(answer, emitted.get(one.name) as string, one.path);
    }
  }, 300_000);

  it.skipIf(!generated)('records every model document of the repository', () => {
    const { documents, cases } = recorded();
    const manifest = readOracleManifest() as unknown as {
      expansion?: { documents: number; cases: number };
    };
    expect(manifest.expansion?.documents).toBe(documents.length);
    expect(manifest.expansion?.cases).toBe(cases.length);
    // The corpus and the template, and the 73 rejection documents.
    expect(documents).toHaveLength(15 + 73);
    // Every branch of the emitter that a repository document reaches, reached by one.
    expect(documents.some((one) => (one.instances ?? []).length > 0)).toBe(true);
    expect(documents.filter((one) => one.error !== null)).not.toHaveLength(0);
    const raised = new Set(
      documents.flatMap((one) => (one.error === null ? [] : [one.error.type])),
    );
    // The refusals the repository's own documents reach. `d1.run` catches `ValueError`,
    // `KeyError`, `OSError` and `ModelError` and prints `failed: …`; the `TypeError` of
    // `structural-interface-old-form` escapes it, as it escapes `--validate` (feature 1.6b).
    expect([...raised].sort()).toEqual(['KeyError', 'ModelError', 'TypeError', 'ValueError']);
    // `Unassigned` — a template read without one — is reached by an edited case alone: `--d1`
    // skips a document `missing_assignment` reports on, and the editor does not (§4.6).
    expect(cases.filter((one) => one.error?.type === 'Unassigned')).toHaveLength(1);
  });

  it.skipIf(!generated)('expands every model document as the tools expand it', () => {
    for (const one of recorded().documents) {
      let answer: PyRecord | null = null;
      let error: Raised | null = null;
      try {
        const text = readFileSync(join(repositoryRoot, one.path), 'utf8');
        answer = expandText(text, libraryFor(one.bases), {
          ...(one.assignment === null ? {} : { assignment: decodeRecord(one.assignment) }),
        });
      } catch (raised) {
        error = raisedAs(raised);
      }
      expect(error, one.path).toEqual(one.error);
      if (one.error !== null) continue;
      agrees(answer as PyRecord, one.emitted as string, one.path);
    }
  }, 600_000);

  it.skipIf(!generated)('expands every edited case as the tools expand it', () => {
    for (const one of recorded().cases) {
      const document = toPython(parse(readFileSync(join(repositoryRoot, one.source), 'utf8')));
      applyEdits(document, one.edits, (value) => decode(value as PyValue), one.name);
      let answer: PyRecord | null = null;
      let error: Raised | null = null;
      try {
        answer = expand(document, libraryFor(['data/primitive-library']));
      } catch (raised) {
        error = raisedAs(raised);
      }
      expect(error, one.name).toEqual(one.error);
      if (one.error !== null) continue;
      agrees(answer as PyRecord, one.emitted as string, one.name);
    }
  }, 300_000);

  it.skipIf(!generated)('counts what the tools counted, document by document', () => {
    const { documents, cases } = recorded();
    for (const one of [...documents, ...cases] as (DocumentCase & EditedCase)[]) {
      if (one.error !== null) continue;
      const graph = expectedGraph(one.emitted as string);
      expect(BigInt(graph.nodes), one.name).toBe(one.nodes);
      expect(BigInt(graph.edges), one.name).toBe(one.edges);
      expect(graph.instances, one.name).toEqual(one.instances ?? []);
    }
  }, 300_000);
});

/** The figures the fixture records beside the document, read back out of the document itself. */
function expectedGraph(file: string): { nodes: number; edges: number; instances: string[] } {
  const document = toPython(parse(expectedText(file))) as PyRecord;
  const graph = (document as Record<string, PyValue>)['d1'] as PyRecord;
  return {
    nodes: Object.keys(graph['nodes'] as PyRecord).length,
    edges: (graph['edges'] as PyValue[]).length,
    instances: Object.keys((graph['instances'] ?? {}) as PyRecord),
  };
}
