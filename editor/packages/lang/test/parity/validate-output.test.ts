import { readFileSync } from 'node:fs';
import { basename, join } from 'node:path';

import { describe, expect, it } from 'vitest';

import {
  analyseText,
  basesOf,
  loadLibrary,
  parse,
  pyStr,
  toPython,
  type Library,
  type PyRecord,
  type PyValue,
} from '../../src/index.js';
import { nodeSource } from '../library/source.js';
import { repositorySchemas } from '../schema/repository.js';
import { oracleGenerated, oracleOut, readOracleManifest, repositoryRoot } from './oracle.js';

// `--validate`'s own printed answer, line for line (feature 1.12).
//
// This is the product of §0.5 that no other suite reads, and it is not a repetition of one that
// does. Feature 1.6c compares the seventeen counters against the `bindings` step, which is the
// oracle's **in-process transcription** of `validate.analyse`; here they are compared against the
// text the *command line* wrote — the answer a user sees. Feature 1.10 drew the same distinction
// for `--lint` ("the two repository sets are recomputed in process and required to equal the files
// the command line wrote"), and it is worth drawing twice: a transcription that drifted would
// pass its own suite and fail this one.
//
// What the line is contract for, beyond the figures: the counters' **names** and their **order**,
// which `validate.run` prints as `" | ".join(f"{k}={v}")` over the dictionary `analyse` returns;
// the rendering of each value, which is Python's `str` — so `dag=True`, not `dag=true`, and an
// integer, never a float; and the verdict word, `ok`, in the column `f"  {name:34s} ok   "` puts
// it in. The core does not write that line — the driver does, and `tools/tensorspine` is not
// ported (§5.3, feature 1.10's rule) — so the *test* assembles it from the core's own counters,
// which is exactly what makes a moved counter name a failure here.
//
// Two lines of the file are the driver's alone and are checked as such: its header
// (`validate  N document(s)`) and its footer (`N/N pass`), which say what set was read — the same
// set-dependence feature 1.10 found in `--lint`, in its milder form.

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

/** The counters as `validate.run` prints them: `" | ".join(f"{k}={v}")`, Python's `str`. */
function summaryOf(stats: ReadonlyMap<string, PyValue>): string {
  return [...stats].map(([name, value]) => `${name}=${pyStr(value)}`).join(' | ');
}

/** The line `validate.run` writes for a document that passed: `f"  {name:34s} ok   {summary}"`. */
function lineOf(name: string, summary: string): string {
  return `  ${name.padEnd(34)} ok   ${summary}`;
}

interface Recorded {
  /** The `--validate` output a set of documents was read in. */
  readonly file: string;
  /** The documents of that set, in the order the manifest names them. */
  readonly documents: { name: string; path: string; assignment: Record<string, unknown> | null }[];
}

/** The recorded invocations, one per set — `--validate` is read for a set, as `--lint` is. */
function invocations(): Recorded[] {
  const sets = new Map<string, Recorded>();
  for (const one of readOracleManifest().documents) {
    const held = sets.get(one.validate) ?? { file: one.validate, documents: [] };
    held.documents.push({ name: one.name, path: one.path, assignment: one.assignment });
    sets.set(one.validate, held);
  }
  return [...sets.values()];
}

/** What the core answers for one document, as the command line would print it. */
function answered(path: string, assignment: Record<string, unknown> | null): string {
  const text = readFileSync(join(repositoryRoot, path), 'utf8');
  const bases = [...basesOf(path, toPython(parse(text))).bases];
  const analysis = analyseText(
    text,
    libraryFor(bases),
    assignment === null ? {} : { assignment: toPython(parse(JSON.stringify(assignment))) as PyRecord },
  );
  expect(analysis.problems, path).toEqual([]);
  return lineOf(basename(path), summaryOf(analysis.stats));
}

describe('`--validate`’s own output', () => {
  it.runIf(inCI)('has the oracle to compare with', () => {
    expect(generated).toBe(true);
  });

  it.skipIf(!generated)('prints, for every document of every set, the line the tools printed', () => {
    const sets = invocations();
    expect(sets).toHaveLength(2);
    let lines = 0;
    for (const one of sets) {
      const recorded = readFileSync(join(oracleOut, one.file), 'utf8').split('\n');
      for (const document of one.documents) {
        const line = answered(document.path, document.assignment);
        const name = basename(document.path);
        const theirs = recorded.find((each) => each.startsWith(`  ${name} `));
        expect(theirs, `${one.file}: no line for ${name}`).toBeDefined();
        expect(line, document.path).toBe(theirs);
        lines += 1;
      }
    }
    // The fourteen models of `data/models/` and the one template, under its assignment.
    expect(lines).toBe(15);
  });

  it.skipIf(!generated)('is read for a set, and the set is what the header and footer name', () => {
    for (const one of invocations()) {
      const recorded = readFileSync(join(oracleOut, one.file), 'utf8').split('\n');
      const count = one.documents.length;
      expect(recorded[0], one.file).toBe(`validate  ${String(count)} document(s)`);
      expect(recorded.filter((line) => line.trim().length > 0).at(-1), one.file).toBe(
        `  ${String(count)}/${String(count)} pass`,
      );
    }
  });

  it.skipIf(!generated)('names the seventeen counters, in the order the tools print them', () => {
    const [first] = invocations();
    expect(first).toBeDefined();
    const recorded = readFileSync(join(oracleOut, first?.file ?? ''), 'utf8');
    const shapes = new Set<string>();
    for (const line of recorded.split('\n')) {
      const summary = line.split(' ok   ')[1];
      if (summary === undefined) continue;
      shapes.add(summary.split(' | ').map((pair) => pair.split('=')[0] ?? '').join(','));
    }
    expect([...shapes]).toEqual([
      [
        'composite_primitives',
        'instances',
        'edges',
        'dag',
        'resolved_domains',
        'parameter_slots',
        'tensors',
        'shared',
        'located',
        'parameter_elements',
        'ops_per_element',
        'ops_per_cached_position',
        'ops_per_sequence',
        'ops_per_invocation',
        'precisions_checked',
        'state_slots',
        'state_identities',
      ].join(','),
    ]);
  });
});
