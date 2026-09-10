import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { toJsonValue, toPython, type PyRecord, type PyValue } from '../../src/expr/index.js';
import { parse, serialize } from '../../src/json/index.js';
import { ModelError, loadModel, normalise } from '../../src/model/index.js';
import { applyEdits, put, type Edit } from './edits.js';
import { oracleGenerated, oracleOut, readOracleManifest, repositoryRoot } from './oracle.js';

// Parity of model normalisation (feature 1.4): what `tools/model.py` reads a document as.
//
// The material is the oracle's (§0.5), and it is of two kinds:
//
//   * every model document of the repository — the fourteen of the corpus, the template, and the
//     73 of `tests/rejections/models/` — read through `model.load`, recorded as the *whole text*
//     `json.dumps(…, indent=2, ensure_ascii=False)` writes, or as the `ModelError` it raised;
//   * 36 edited documents, each one llama3-8b with a few values changed at JSON pointers, reaching
//     the branches of `model.py` no document of the repository takes: a scoped `constants` rule, a
//     declared `tensor` or `identity`, a composition with two indices, an endpoint written as an
//     explicit selector, and the third refusal, which `tests/rejections` has no case for.
//
// The comparison is on **bytes**, which is stronger than a deep-equal and is available here for a
// reason feature 0.3 measured: every file of the corpus is exactly what Python's `json.dumps`
// writes, and `toJsonValue(toPython(tree))` round-trips all 146 of them byte for byte (1.2). So a
// normalised document written by the core's serializer is comparable, character for character,
// with the one the tools would have dumped — and a failure names the line that moved.
//
// The edits are pointers into a repository file, as the library step's mutations are: both
// implementations read the same bytes and apply the same change, and neither side owns the input.

const inCI = process.env['CI'] !== undefined && process.env['CI'] !== '';
const generated = oracleGenerated();

/** The refusal the tools raised, as `str(ModelError)`. */
interface Refusal {
  type: string;
  message: string;
}

/** One model document of the repository, normalised or refused. */
interface DocumentCase {
  /** Repository-relative. */
  path: string;
  error: Refusal | null;
  /** The normalised text's file, relative to the oracle's output directory. */
  normalised: string | null;
}

/** One edited document: the source, the edits, and what `model.normalise` answered. */
interface EditedCase {
  name: string;
  source: string;
  edits: Edit[];
  error: Refusal | null;
  normalised: string | null;
}

interface Recorded {
  source: string;
  documents: DocumentCase[];
  cases: EditedCase[];
}

function recorded(): Recorded {
  return JSON.parse(readFileSync(join(oracleOut, 'model', 'index.json'), 'utf8')) as Recorded;
}

/** A repository file's bytes, as `open(encoding='utf-8')` reads them. */
function repositoryText(relative: string): string {
  return readFileSync(join(repositoryRoot, relative), 'utf8');
}

/** A normalised document as the oracle wrote it. */
function oracleText(relative: string): string {
  return readFileSync(join(oracleOut, relative), 'utf8');
}

/** A normalised document as the core writes it (D12). */
function written(model: PyRecord): string {
  return serialize(toJsonValue(model));
}

/**
 * A fixture value as the reading holds one.
 *
 * The fixture's numbers are whole — the generator refuses a float in an edit — so every one of
 * them is Python's `int`, which is a `bigint` here. Guessing from `Number.isInteger` is exactly
 * what the two types exist to avoid, so anything else fails the test rather than being read.
 */
function fixtureValue(value: unknown, where: string): PyValue {
  if (typeof value === 'number') {
    expect(Number.isInteger(value), `${where}: a fixture value must be a whole number`).toBe(true);
    return BigInt(value);
  }
  if (Array.isArray(value)) return value.map((one) => fixtureValue(one, where));
  if (typeof value === 'object' && value !== null) {
    const record: Record<string, PyValue> = {};
    for (const [name, one] of Object.entries(value)) put(record, name, fixtureValue(one, where));
    return record;
  }
  return value as PyValue;
}

/** The reading of an edited case, before normalisation. */
function edited(one: EditedCase): PyValue {
  const read = toPython(parse(repositoryText(one.source)));
  applyEdits(read, one.edits, (value) => fixtureValue(value, one.name), one.name);
  return read;
}

/** What normalising answered, or the refusal it raised. */
function outcome(run: () => PyRecord): { text: string | null; error: ModelError | null } {
  try {
    return { text: written(run()), error: null };
  } catch (error) {
    if (error instanceof ModelError) return { text: null, error };
    throw error;
  }
}

describe('model normalisation against the tools', () => {
  it.runIf(inCI)('has the oracle to compare with', () => {
    expect(generated).toBe(true);
  });

  it.skipIf(!generated)('records every model document of the repository, and the edited cases', () => {
    const material = recorded();
    const manifest = readOracleManifest() as unknown as {
      model?: { documents: number; cases: number };
    };
    expect(manifest.model?.documents).toBe(material.documents.length);
    expect(manifest.model?.cases).toBe(material.cases.length);
    // The corpus and the template, the 73 rejection documents, and one case per branch.
    expect(material.documents).toHaveLength(15 + 73);
    expect(material.documents.filter((one) => one.error !== null)).toHaveLength(3);
    expect(material.cases.length).toBeGreaterThanOrEqual(30);
    expect(material.cases.filter((one) => one.error !== null).length).toBeGreaterThanOrEqual(10);
    expect(new Set(material.cases.map((one) => one.name)).size).toBe(material.cases.length);
    expect(material.source).toBe('data/models/llama3-8b.json');
  });

  it.skipIf(!generated)('normalises every model document to the tools’ own bytes', () => {
    for (const one of recorded().documents) {
      const answer = outcome(() => loadModel(repositoryText(one.path)));
      if (one.error === null) {
        expect(answer.error, one.path).toBeNull();
        expect(answer.text, one.path).toBe(oracleText(one.normalised as string));
      } else {
        expect(answer.error, one.path).not.toBeNull();
        expect(answer.error?.message, one.path).toBe(one.error.message);
      }
    }
  }, 120_000);

  it.skipIf(!generated)('answers the edited cases as the tools answer them', () => {
    for (const one of recorded().cases) {
      const answer = outcome(() => normalise(edited(one)));
      if (one.error === null) {
        expect(answer.error, one.name).toBeNull();
        expect(answer.text, one.name).toBe(oracleText(one.normalised as string));
      } else {
        expect(answer.error, one.name).not.toBeNull();
        expect(answer.error?.message, one.name).toBe(one.error.message);
      }
    }
  }, 120_000);

  it.skipIf(!generated)('is idempotent on every one of them', () => {
    // `tests/run_expressions.py` requires it of the tools ("normalisation is idempotent"); the
    // normalised document is what D1 and the derivation read, and reading it twice must not move
    // a rule or a member.
    const material = recorded();
    const again = (relative: string): void => {
      const text = oracleText(relative);
      expect(written(normalise(toPython(parse(text)))), relative).toBe(text);
    };
    for (const one of material.documents) if (one.normalised !== null) again(one.normalised);
    for (const one of material.cases) if (one.normalised !== null) again(one.normalised);
  }, 120_000);

  it.skipIf(!generated)('hoists one top-level rule per scoped rule, and empties the compositions', () => {
    // The count is the claim of §5.2 rule 7 — "each such rule denotes exactly the top-level rule
    // `C.R`" — read off the corpus rather than off the port: the document before normalisation is
    // the tools' input, the document after it is the tools' output, and both are on disk.
    let scoped = 0;
    for (const one of recorded().documents) {
      if (one.normalised === null) continue;
      const before = toPython(parse(repositoryText(one.path))) as PyRecord;
      const after = toPython(parse(oracleText(one.normalised))) as PyRecord;
      let hoisted = 0;
      for (const composition of Object.values(before['compositions'] as PyRecord)) {
        const rules = (composition as PyRecord)['bindings'] as PyRecord | undefined;
        for (const kind of Object.values(rules ?? {})) {
          hoisted += Object.keys(kind as PyRecord).length;
        }
      }
      for (const composition of Object.values(after['compositions'] as PyRecord)) {
        expect(Object.keys(composition as PyRecord), one.path).not.toContain('bindings');
      }
      const count = (model: PyRecord): number => {
        let total = 0;
        for (const map of Object.values(model['bindings'] as PyRecord)) {
          total += Object.keys(map as PyRecord).length;
        }
        return total;
      };
      expect(count(after) - count(before), one.path).toBe(hoisted);
      scoped += hoisted;
    }
    expect(scoped).toBeGreaterThan(1500);
  }, 120_000);
});
