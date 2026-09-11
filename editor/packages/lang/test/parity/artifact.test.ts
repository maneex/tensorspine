import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { expect, it, suite } from 'vitest';

import {
  checkCheckpoint,
  d3,
  isRecord,
  parse,
  toPython,
  type CheckpointHeaders,
  type CheckpointProblem,
  type HeaderEntry,
  type PyRecord,
  type PyValue,
} from '../../src/index.js';
import { derivedCorpus, generated, inCI, libraryOf, type Case } from './derived.js';
import { oracleOut, readOracleManifest } from './oracle.js';

// Parity of the checkpoint check (feature 1.9): `artifact.check` over the corpus, over mutations
// of its checkpoints, and over every branch of `_check_part`.
//
// The corpus has no checkpoint in the repository — the weights are gigabytes and live outside it —
// so the oracle **synthesises** one per document out of the document's own D3 and records it
// literally (`out/artifact/headers/<name>.json`), together with what `artifact.check` answered.
// Both implementations therefore read the same bytes, and the synthesis, which is test material
// and not a rule of the language, is written once, in Python, and never ported.
//
// What the comparison is made of: the core's **own** D3 over the core's derivation, against the
// answer the tools gave over theirs. Feature 1.8a already proves the two D3s equal on every corpus
// document, so a failure here is the check's and not the inventory's — and the pairing is the one
// a reader of §4.19 sees, a derived document and a header map.
//
// The tools' `advisories` are the editor's `unnamed` warnings, in the same order and the same
// words (§4.19 renames them, it does not rewrite them); the `unlocated` warnings beside them are
// the plan's own and have no counterpart to compare with — the unit suite owns them.

let recorded: PyRecord | undefined;

/** The oracle's own index, read once and as a document is read: a whole number is an integer. */
function index(): PyRecord {
  recorded ??= toPython(
    parse(readFileSync(join(oracleOut, 'artifact', 'index.json'), 'utf8')),
  ) as PyRecord;
  return recorded;
}

/** One recorded checkpoint, read the same way and folded into the map the core takes. */
function checkpointOf(file: string): CheckpointHeaders {
  const read = toPython(parse(readFileSync(join(oracleOut, file), 'utf8')));
  return headersOf(read);
}

/** `{dtype, shape, file}` as the oracle records it, as the core's `HeaderEntry`. */
function headersOf(read: PyValue): CheckpointHeaders {
  const out: Record<string, HeaderEntry> = {};
  for (const [name, entry] of Object.entries(read as PyRecord)) {
    const record = entry as PyRecord;
    out[name] = {
      dtype: record['dtype'] as string,
      // Every dtype the oracle writes comes from a D3, so it is a dtype of the language: what the
      // reader's table would have answered for the safetensors name that denotes it.
      known: true,
      shape: record['shape'] as readonly PyValue[],
      file: record['file'] as string,
    };
  }
  return out;
}

/** The case's edits, applied as `_edited` applies them. */
function edited(headers: CheckpointHeaders, edits: readonly PyValue[]): CheckpointHeaders {
  const out: Record<string, HeaderEntry> = { ...headers };
  for (const edit of edits) {
    const record = edit as PyRecord;
    const tensor = record['tensor'] as string;
    if (record['op'] === 'delete') {
      delete out[tensor];
    } else {
      out[tensor] = headersOf({ [tensor]: record['entry'] as PyValue })[tensor] as HeaderEntry;
    }
  }
  return out;
}

/** The recorded strings of a list the oracle wrote. */
function strings(value: PyValue | undefined): string[] {
  return (value as readonly PyValue[]).map((one) => one as string);
}

/** The tools' `errors`: the core's messages, without the prefix its printer adds. */
function messages(problems: readonly CheckpointProblem[]): string[] {
  return problems.map((problem) => problem.message);
}

/** The warnings that stand for the tools' advisories. */
function advisories(problems: readonly CheckpointProblem[]): CheckpointProblem[] {
  return problems.filter((problem) => problem.kind === 'unnamed');
}

suite('the checkpoint check against the tools', () => {
  it.runIf(inCI)('has the oracle to compare with', () => {
    expect(generated).toBe(true);
  });

  it.skipIf(!generated)('records a checkpoint per document, the cases and the forms', () => {
    const manifest = readOracleManifest() as unknown as {
      artifact?: { index: string; documents: number; cases: number; forms: number };
    };
    expect(manifest.artifact, 'the manifest names no artifact product').toBeDefined();
    const written = index();
    const documents = written['documents'] as readonly PyRecord[];
    // The fifteen `--derive` was run over: fourteen models and the template, twelve of which
    // locate their weights (feature 1.8a measured 9 254 located identity instances).
    expect(documents).toHaveLength(15);
    expect(documents.filter((one) => Number(one['located']) > 0)).toHaveLength(13);
    expect((written['cases'] as readonly PyValue[]).length).toBeGreaterThan(80);
    expect(written['forms']).toHaveLength(18);
  });

  it.skipIf(!generated)(
    'answers every case as the tools answered it, word for word',
    { timeout: 300_000 },
    () => {
      const corpus = new Map(derivedCorpus().map((one) => [one.name, one]));
      const inventories = new Map<string, PyRecord>();
      const inventoryOf = (name: string): PyRecord => {
        const held = inventories.get(name);
        if (held !== undefined) return held;
        const one = corpus.get(name);
        expect(one, `the oracle names a document the derivation suite has not: ${name}`)
          .toBeDefined();
        const answer = d3((one as Case).derivation.graph, libraryOf(one as Case));
        inventories.set(name, answer);
        return answer;
      };
      const checkpoints = new Map<string, CheckpointHeaders>();
      for (const record of index()['documents'] as readonly PyRecord[]) {
        checkpoints.set(record['name'] as string, checkpointOf(record['headers'] as string));
      }

      const kinds = new Set<string>();
      let compared = 0;
      for (const one of index()['cases'] as readonly PyRecord[]) {
        const where = one['name'] as string;
        const headers = edited(
          checkpoints.get(one['headers_of'] as string) as CheckpointHeaders,
          one['edits'] as readonly PyValue[],
        );
        const answer = checkCheckpoint(
          { d3: inventoryOf(one['document'] as string) },
          headers,
        );
        expect(messages(answer.errors), where).toEqual(strings(one['errors']));
        expect(messages(advisories(answer.warnings)), where).toEqual(strings(one['advisories']));
        expect(answer.stats, where).toEqual({
          located: Number((one['stats'] as PyRecord)['located']),
          physical: Number((one['stats'] as PyRecord)['physical']),
          unnamed: Number((one['stats'] as PyRecord)['unnamed']),
        });
        // A valid document locates all its identities or none, so no corpus case can produce the
        // warning §4.19 asks for beside the advisories — the unit suite is where it is proved.
        expect(
          answer.warnings.filter((problem) => problem.kind === 'unlocated'),
          where,
        ).toEqual([]);
        for (const problem of [...answer.errors, ...answer.warnings]) kinds.add(problem.kind);
        compared += 1;
      }

      for (const one of index()['forms'] as readonly PyRecord[]) {
        const where = one['name'] as string;
        const answer = checkCheckpoint(
          { d3: one['d3'] as PyValue },
          headersOf(one['headers'] as PyValue),
        );
        expect(messages(answer.errors), where).toEqual(strings(one['errors']));
        expect(messages(advisories(answer.warnings)), where).toEqual(strings(one['advisories']));
        expect(answer.stats.located, where).toBe(Number((one['stats'] as PyRecord)['located']));
        expect(answer.stats.physical, where).toBe(Number((one['stats'] as PyRecord)['physical']));
        expect(answer.stats.unnamed, where).toBe(Number((one['stats'] as PyRecord)['unnamed']));
        for (const problem of [...answer.errors, ...answer.warnings]) kinds.add(problem.kind);
        compared += 1;
      }

      expect(compared).toBeGreaterThan(100);
      // Every branch of `_check_part` that produces a line, reached: a corpus or a fixture that
      // stopped exercising one fails here rather than going quiet.
      expect([...kinds].sort()).toEqual([
        'absent',
        'concat',
        'dtype',
        'shape',
        'slice',
        'stack',
        'unnamed',
      ]);
    },
  );

  it.skipIf(!generated)('reads a recorded checkpoint the way the core reads a header', () => {
    // The shapes the oracle recorded are Python integers, and the core compares them against D3's
    // extents, which are too (feature 1.2). A fixture read with `JSON.parse` would compare two
    // `number`s and pass on a document where the two kinds differ.
    const checkpoint = checkpointOf('artifact/headers/llama3-8b.json');
    const entry = checkpoint['model.norm.weight'];
    expect(entry).toBeDefined();
    expect(entry?.shape.every((extent) => typeof extent === 'bigint')).toBe(true);
    expect(isRecord(toPython(parse('{"a": 1}')))).toBe(true);
  });
});
