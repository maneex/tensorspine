import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import {
  assignmentNeeded,
  checkAssignment,
  checkQuantities,
  comparePythonStrings,
  formatSemanticProblems,
  loadModel,
  ModelError,
  parse,
  resolveQuantities,
  toPython,
  variableQuantities,
  type PyRecord,
  type PyValue,
  type SemanticProblem,
} from '../../src/index.js';
import { decode, decodeRecord, encodeMap } from './encoding.js';
import { raisedAs, type Raised } from './raised.js';
import { applyEdits, type Edit } from './edits.js';
import { oracleGenerated, oracleOut, readOracleManifest, repositoryRoot } from './oracle.js';

// Parity of quantities and assignments (feature 1.5): `check_quantities`, `check_assignment`,
// `variable_quantities` and the assignment-needed report of `tools/validate.py`.
//
// The material is the oracle's (§0.5), in two kinds, as features 1.3 and 1.4 record theirs:
//
//   * every model document of the repository — the fourteen of the corpus, the template under its
//     documented assignment, and the 73 of `tests/rejections/models/` under the assignment each
//     case names — with everything the four functions answer about it;
//   * the edited documents and assignments of `tests/oracle/quantity_cases.py`, each reaching a
//     branch the corpus does not take. It takes very few: its 215 quantities are 186 cardinalities,
//     15 reals and 14 enums — no boolean, no physical quantity — with eight domains between
//     fifteen documents and not one variable quantity outside the template.
//
// Values are compared as *encodings*, never as values (feature 1.2): `1` and `1.0` are one double
// in JavaScript, and only the encoding tells them apart — which is exactly what V3 reads.

const inCI = process.env['CI'] !== undefined && process.env['CI'] !== '';
const generated = oracleGenerated();

/** Everything feature 1.5 answers about one document, as the oracle records it. */
interface Facts {
  external: string[];
  required: string[];
  missing: string[];
  unassigned_missing: string[];
  report: string | null;
  variable: string[];
  resolved: Record<string, unknown>;
  problems: [string, string][];
  assignment_errors: string[];
}

/** One model document of the repository. */
type DocumentCase = Partial<Facts> & {
  path: string;
  assignment: PyRecord | null;
  /** `str(ModelError)` when `model.load` refuses the document, which stops the tools here. */
  refused: string | null;
  error: Raised | null;
};

/** One edited document, or one assignment against an unedited one. */
type EditedCase = Partial<Facts> & {
  name: string;
  document: string;
  edits: Edit[];
  assignment: PyRecord | null;
  error: Raised | null;
};

interface Recorded {
  source: string;
  template: string;
  documents: DocumentCase[];
  cases: EditedCase[];
}

interface RejectionCase {
  document: string;
  expect: string;
  match: string;
  assign?: Record<string, unknown>;
}

function recorded(): Recorded {
  return JSON.parse(readFileSync(join(oracleOut, 'quantities', 'index.json'), 'utf8')) as Recorded;
}

function rejections(): RejectionCase[] {
  const file = join(oracleOut, 'rejections', 'models.json');
  return (JSON.parse(readFileSync(file, 'utf8')) as { cases: RejectionCase[] }).cases;
}

/** A repository file's bytes, as `open(encoding='utf-8')` reads them. */
function repositoryText(relative: string): string {
  return readFileSync(join(repositoryRoot, relative), 'utf8');
}

/**
 * The four functions, in the order the fixture computes them.
 *
 * A document off the grammar raises somewhere, and the fixture records the *first* refusal, so
 * this side has to reach it at the same point. `checkQuantities` is handed the normalised reading
 * and the rest the document as read, which is how the tools' own callers hand them one.
 */
function facts(model: PyRecord, raw: PyRecord, assignment: PyRecord | undefined): Facts {
  const needed = assignmentNeeded(raw, assignment);
  const bare = assignmentNeeded(raw);
  const variable = [...variableQuantities(model)].sort(comparePythonStrings);
  const resolved = resolveQuantities(model, assignment);
  const problems: SemanticProblem[] = checkQuantities(model, resolved);
  const errors = checkAssignment(raw, assignment);
  return {
    external: [...needed.external].sort(comparePythonStrings),
    required: [...needed.required].sort(comparePythonStrings),
    missing: [...needed.unset],
    unassigned_missing: [...bare.unset],
    report: needed.report,
    variable,
    resolved: encodeMap(resolved),
    problems: problems.map((one) => [one.code, one.message]),
    assignment_errors: formatSemanticProblems(errors),
  };
}

/** The recorded facts of a case, as the comparison reads them. */
function expected(one: Partial<Facts>): Facts {
  return {
    external: one.external as string[],
    required: one.required as string[],
    missing: one.missing as string[],
    unassigned_missing: one.unassigned_missing as string[],
    report: one.report ?? null,
    variable: one.variable as string[],
    resolved: one.resolved as Record<string, unknown>,
    problems: one.problems as [string, string][],
    assignment_errors: one.assignment_errors as string[],
  };
}

/** What the four functions answered, or the refusal they raised. */
function outcome(run: () => Facts): { facts: Facts | null; error: Raised | null } {
  try {
    return { facts: run(), error: null };
  } catch (error) {
    return { facts: null, error: raisedAs(error) };
  }
}

/** The assignment a case is read under. */
function assignmentOf(one: { assignment: PyRecord | null }): PyRecord | undefined {
  return one.assignment === null ? undefined : decodeRecord(one.assignment);
}

/** The reading of an edited case: the document as it stands, with the case's edits applied. */
function edited(one: EditedCase): PyRecord {
  const read = toPython(parse(repositoryText(one.document)));
  applyEdits(read, one.edits, (value) => decode(value as PyValue), one.name);
  return read as PyRecord;
}

/** The lines `check_quantities` would have produced for one recorded document. */
function linesOf(one: DocumentCase): string[] {
  return (one.problems ?? []).map(([code, message]) => `[${code}] ${message}`);
}

describe('quantities and assignments against the tools', () => {
  it.runIf(inCI)('has the oracle to compare with', () => {
    expect(generated).toBe(true);
  });

  it.skipIf(!generated)('records every model document of the repository, and the cases', () => {
    const material = recorded();
    const manifest = readOracleManifest() as unknown as {
      quantities?: { documents: number; cases: number };
    };
    expect(manifest.quantities?.documents).toBe(material.documents.length);
    expect(manifest.quantities?.cases).toBe(material.cases.length);
    // The corpus and the template, and the 73 rejection documents.
    expect(material.documents).toHaveLength(15 + 73);
    // The three `model.load` refuses stop before the quantities are read (feature 1.4).
    expect(material.documents.filter((one) => one.refused !== null)).toHaveLength(3);
    expect(material.cases.length).toBeGreaterThanOrEqual(60);
    expect(new Set(material.cases.map((one) => one.name)).size).toBe(material.cases.length);
    // Every branch that raises is reached by at least one case, and every rule by one line.
    expect(material.cases.filter((one) => one.error !== null).length).toBeGreaterThanOrEqual(8);
    const codes = new Set(
      material.cases.flatMap((one) => (one.problems ?? []).map(([code]) => code)),
    );
    expect([...codes].sort()).toEqual(['V1', 'V10', 'V11', 'V3']);
    expect(material.source).toBe('data/models/llama3-8b.json');
    expect(material.template).toBe('data/models/decoder-causal-yarn/1.0.0.json');
  });

  it.skipIf(!generated)('answers every document of the repository as the tools answer it', () => {
    for (const one of recorded().documents) {
      const text = repositoryText(one.path);
      if (one.refused !== null) {
        expect(() => loadModel(text), one.path).toThrow(ModelError);
        continue;
      }
      const model = loadModel(text);
      const raw = toPython(parse(text)) as PyRecord;
      const answer = outcome(() => facts(model, raw, assignmentOf(one)));
      expect(answer.error, one.path).toEqual(one.error);
      if (one.error === null) expect(answer.facts, one.path).toEqual(expected(one));
    }
  }, 120_000);

  it.skipIf(!generated)('answers every edited case as the tools answer it', () => {
    for (const one of recorded().cases) {
      const document = edited(one);
      const answer = outcome(() => facts(document, document, assignmentOf(one)));
      expect(answer.error, one.name).toEqual(one.error);
      if (one.error === null) expect(answer.facts, one.name).toEqual(expected(one));
    }
  }, 120_000);

  it.skipIf(!generated)('reads the normalised document and the document as read alike', () => {
    // `analyse` hands `check_quantities` the normalised reading and `run` hands
    // `check_assignment` the document as read; normalisation moves only `bindings` and
    // `compositions` (feature 1.4), so the two readings must answer the same thing here — which
    // is what lets the edited cases be read without normalising them.
    for (const one of recorded().documents) {
      if (one.refused !== null) continue;
      const text = repositoryText(one.path);
      const model = loadModel(text);
      const raw = toPython(parse(text)) as PyRecord;
      const assignment = assignmentOf(one);
      const asRead = outcome(() => facts(raw, raw, assignment));
      const normalised = outcome(() => facts(model, model, assignment));
      expect(asRead, one.path).toEqual(normalised);
    }
  }, 120_000);

  it.skipIf(!generated)('decides five cases of the rejection suite, with their codes and words', () => {
    // The runner's own reading (`tests/run_rejections.py`): the lines whose code is the expected
    // one must carry the `match` text. Restricted to what this feature decides — the rest of the
    // suite belongs to the rules features 1.6a–1.6c port.
    const material = recorded();
    const decided = material.documents.filter(
      (one) => one.path.includes('tests/rejections/') && (one.problems ?? []).length > 0,
    );
    expect(decided.map((one) => one.path.split('/').pop())).toEqual([
      'v1-derivation-undeclared-quantity.json',
      'v10-derivation-cycle.json',
      'v11-literal-disagrees-with-derivation.json',
      'v3-derived-cardinality-not-integer.json',
      'v3-derived-variable-without-domain.json',
    ]);
    const suite = rejections();
    for (const one of decided) {
      const name = one.path.split('/').pop() as string;
      const rejection = suite.find((each) => each.document.endsWith(name));
      expect(rejection, name).toBeDefined();
      const expects = `[${(rejection as RejectionCase).expect}]`;
      const model = loadModel(repositoryText(one.path));
      const answered = formatSemanticProblems(
        checkQuantities(model, resolveQuantities(model, assignmentOf(one))),
      );
      const lines = answered.filter((line) => line.startsWith(expects));
      expect(lines, name).toEqual(linesOf(one).filter((line) => line.startsWith(expects)));
      expect(lines.length, name).toBeGreaterThan(0);
      expect(
        lines.some((line) => line.includes((rejection as RejectionCase).match)),
        name,
      ).toBe(true);
    }
  }, 120_000);

  it.skipIf(!generated)('passes the template under its documented assignment', () => {
    // The README's `--assign` example, which is `tests/signature.py`'s ASSIGNMENTS entry and
    // `tests/run_templates.py`'s ASSIGNMENT: "admissible assignment accepted" and "template alone
    // under that assignment passes", as far as this feature decides them.
    const material = recorded();
    const one = material.documents.find((each) => each.path === material.template);
    expect(one, material.template).toBeDefined();
    const document = one as DocumentCase;
    const assignment = assignmentOf(document);
    expect(assignment, 'the template is recorded under an assignment').toBeDefined();
    const text = repositoryText(document.path);
    const model = loadModel(text);
    const raw = toPython(parse(text)) as PyRecord;
    expect(checkAssignment(raw, assignment)).toEqual([]);
    expect(checkQuantities(model, resolveQuantities(model, assignment))).toEqual([]);
    expect(assignmentNeeded(raw, assignment).unset).toEqual([]);
    // Read alone it is a family of graphs, not a defect: eight names, `eps` included.
    expect(assignmentNeeded(raw).unset).toEqual([
      'eps',
      'head_dim',
      'heads',
      'inner',
      'kv_heads',
      'layers',
      'precision',
      'width',
    ]);
    expect(assignmentNeeded(raw).report).toBe(
      "needs --assign for ['eps', 'head_dim', 'heads', 'inner', 'kv_heads', 'layers', " +
        "'precision', 'width']",
    );
  });

  it.skipIf(!generated)('refuses an inadmissible assignment with the tools’ wording', () => {
    // `tests/run_templates.py`'s own check, run against the port: "inadmissible assignment
    // refused (layers=0, precision=fp4)" — two errors, the first about a domain bound and the
    // second about a set.
    const material = recorded();
    const one = material.cases.find((each) => each.name === 'template-inadmissible');
    expect(one, 'template-inadmissible').toBeDefined();
    const bad = one as EditedCase;
    const raw = toPython(parse(repositoryText(bad.document))) as PyRecord;
    const errors = formatSemanticProblems(checkAssignment(raw, assignmentOf(bad)));
    expect(errors).toHaveLength(2);
    expect(errors[0]).toContain('below the domain bound');
    expect(errors[1]).toContain('not among');
    expect(errors).toEqual(bad.assignment_errors);
  });
});
