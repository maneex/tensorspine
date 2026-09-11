import { readFileSync } from 'node:fs';
import { join, relative } from 'node:path';

import { describe, expect, it } from 'vitest';

import {
  basesOf,
  lint,
  lintReport,
  loadLibrary,
  parse,
  serialize,
  toJsonValue,
  toPython,
  type Library,
  type LintDocument,
  type PyValue,
} from '../../src/index.js';
import { nodeSource } from '../library/source.js';
import { repositorySchemas } from '../schema/repository.js';
import { applyEdits, type Edit } from './edits.js';
import { decode } from './encoding.js';
import { raisedAs, type Raised } from './raised.js';
import { oracleGenerated, oracleOut, readOracleManifest, repositoryRoot } from './oracle.js';

// Parity of `--lint` (feature 1.10): the hygiene command, whose findings are advice rather than
// refusals and which therefore always exits 0.
//
// What is compared is a **set**, not a document. "called by none of the N model(s) linted" names
// the size of the set, and what the set calls is what decides which primitives are reported, so
// two comparisons are made here:
//
//   - the two files the oracle's command-line invocations wrote (`lint/corpus.txt` and the
//     template's, which the manifest names per document) — the feature's own test, line for line,
//     with the first line checked as the *driver's* (`lint  N document(s)`), which `lint.py` does
//     not print and the core therefore does not write;
//   - every set of `lint/index.json`, which `tests/oracle/lint_cases.py` built to reach what the
//     corpus cannot: the advisory, the deduplication over `(scope, message)`, an axis and a
//     precision role no primitive cites, a primitive reported at a version other than `1.0.0`,
//     the two places the command raises, and a document reported as off the grammar.
//
// The edited documents are the idiom features 1.3 to 1.6c established — a pointer into a
// repository file with a few values changed — so both implementations read the same bytes and
// apply the same change; the one hand-written text (a file that is not JSON) is recorded in the
// fixture and read identically by both.

const inCI = process.env['CI'] !== undefined && process.env['CI'] !== '';
const generated = oracleGenerated();

/** One document of a set, as the fixture writes it. */
interface RecordedDocument {
  /** `repository` for a file of the repository, `edit` for an edited one, `raw` for a text. */
  kind: 'repository' | 'edit' | 'raw';
  /** The path the tools were given, whose base name is the scope of a document's findings. */
  path: string;
  /** `edit`: the repository file the edits apply to. */
  source?: string;
  edits?: Edit[];
  /** `raw`: the text itself. */
  text?: string;
}

/** One whole `lint.run` invocation. */
interface RecordedSet {
  name: string;
  documents: RecordedDocument[];
  /** The bases, or `null` for "the first document's own", which is the command line's default. */
  bases: string[] | null;
  /** Every line `lint.run` printed. */
  output: string[];
  error: Raised | null;
}

function recorded(): { sets: RecordedSet[]; base: { directory: string; files: string[] } } {
  const text = readFileSync(join(oracleOut, 'lint', 'index.json'), 'utf8');
  return JSON.parse(text) as { sets: RecordedSet[]; base: { directory: string; files: string[] } };
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

/**
 * The text of one recorded document.
 *
 * An edited one is re-derived here rather than read back from the oracle's own output: the source
 * is the repository's bytes and the edits are the fixture's, so neither implementation owns the
 * input. Only the document's *value* decides anything `--lint` says — the lines are read from the
 * analysis, never from the source text — so the serializer's layout is free.
 */
function textOf(one: RecordedDocument): string {
  if (one.kind === 'repository') return readFileSync(join(repositoryRoot, one.path), 'utf8');
  if (one.kind === 'raw') return one.text as string;
  const document = toPython(parse(readFileSync(join(repositoryRoot, one.source as string), 'utf8')));
  applyEdits(document, one.edits ?? [], (value) => decode(value as PyValue), one.path);
  return serialize(toJsonValue(document));
}

/** The documents of a set, as the core takes them: the path the tools used, and the bytes. */
function documentsOf(one: RecordedSet): LintDocument[] {
  return one.documents.map((document) => ({ path: document.path, text: textOf(document) }));
}

/** Where the base the `lab` sets add lives, named relative to the repository as every base is. */
const LAB = relative(repositoryRoot, join(oracleOut, 'lint', 'bases', 'lab'));

/**
 * The bases a set resolves from: the ones it names, or the first document's own.
 *
 * `lint.run` reads the first document and calls `bases_of` on it when the command line named
 * none, which is exactly what `basesOf` answers here — resolved against the document's own
 * directory, so the corpus's `../primitive-library/` lands on the reference base.
 */
function basesFor(one: RecordedSet): string[] {
  if (one.bases !== null) return one.bases.map((base) => (base === 'lab' ? LAB : base));
  const first = one.documents[0] as RecordedDocument;
  const resolved = basesOf(first.path, toPython(parse(textOf(first))));
  expect(resolved.problem, one.name).toBeNull();
  return [...resolved.bases];
}

/** The lines the core prints for one set. */
function reportOf(one: RecordedSet): string[] {
  const documents = documentsOf(one);
  return lintReport(lint(documents, { schemas, library: libraryFor(basesFor(one)) }));
}

describe('lint against the tools', () => {
  it.runIf(inCI)('has the oracle to compare with', () => {
    expect(generated).toBe(true);
  });

  it.skipIf(!generated)('records the sets, and every branch of lint.py is reached', () => {
    const { sets, base } = recorded();
    const manifest = readOracleManifest() as unknown as { lint?: { sets: number } };
    expect(manifest.lint?.sets).toBe(sets.length);
    const byName = new Map(sets.map((one) => [one.name, one]));
    const lines = sets.flatMap((one) => one.output);
    const reaches = (fragment: string): boolean => lines.some((line) => line.includes(fragment));

    // The corpus lints clean: every primitive of the reference base is called, every axis outside
    // the `storage` space and every precision role is cited, and no document of the repository
    // produces an advisory (feature 1.6c's finding). Both summary lines are therefore reached.
    expect(byName.get('corpus')?.output).toEqual(['  nothing to report']);
    expect(reaches('advisory finding(s) — nothing blocking')).toBe(true);

    for (const fragment of [
      'is in the primitive library, called by none of the',
      'is cited by no primitive',
      'off the schema, not analysed (--validate refuses it):',
      // The JSON layer, which only a document with a member written twice reaches: the two rules
      // read such a document differently, and the difference is the tools' own — the walk that
      // collects the called primitives uses a plain `json.load`, which takes the last value,
      // while the schema stage refuses the text as V12.
      "off the schema, not analysed (--validate refuses it): [V12] duplicate member name 'instances' (V12)",
      'a self-indexed state on the fragmented stream',
    ]) {
      expect(reaches(fragment), fragment).toBe(true);
    }

    // The `storage` skip, the highest-version reading of the vocabulary, and a version other
    // than 1.0.0 in an uncalled line — the three things the reference base alone cannot show,
    // which is what the base the `lab` sets add is for. `zzz.stored` is a `storage` axis no
    // primitive cites and must *not* be reported; `zzz.only` is cited by version 1.0.0 of
    // `zzz.demo` alone, and `unreferenced_vocabulary` reads the highest version.
    const lab = byName.get('lab')?.output ?? [];
    expect(lab).toEqual([
      "  W  primitive_library: primitive 'zzz.demo' 1.1.0 is in the primitive library, " +
        'called by none of the 14 model(s) linted',
      "  W  primitive_library: axis 'zzz.only' is cited by no primitive",
      "  W  primitive_library: axis 'zzz.unused' is cited by no primitive",
      "  W  primitive_library: precision role 'zzz.unused' is cited by no primitive",
      '  4 advisory finding(s) — nothing blocking',
    ]);
    expect(base.files).toContain('axes/zzz/stored.json');
    expect(lab.some((line) => line.includes('zzz.stored'))).toBe(false);

    // The template recursion, which the corpus does not make observable: measured, the one
    // corpus document that instantiates a template calls every primitive the template calls
    // itself, so a port that never followed a template would answer the same on every document
    // of the repository. `template-instance-only` names the template and nothing else, so the
    // four primitives the template calls are exactly the lines that are *absent*.
    const only = byName.get('template-instance-only')?.output ?? [];
    const uncalled = only
      .filter((line) => line.includes('is in the primitive library'))
      .map((line) => line.split("'")[1] as string);
    for (const name of ['decoder.causal_yarn', 'attention.dense', 'ffn.gated', 'norm.rms',
                        'residual.add']) {
      expect(uncalled, name).not.toContain(name);
    }
    expect(uncalled).toContain('ffn.dense');

    // The deduplication `run` does over `(scope, message)`: the same document linted twice
    // produces the advisory twice and prints it once, so the two sets print the same count.
    const once = byName.get('advisory')?.output ?? [];
    const twice = byName.get('advisory-twice')?.output ?? [];
    expect(twice.filter((line) => line.includes('self-indexed state'))).toHaveLength(2);
    expect(once[once.length - 1]).toBe(twice[twice.length - 1]);
    // And the count still names the size of the set, which is what makes the answer
    // set-dependent: the same finding, two different lines.
    expect(once.some((line) => line.includes('none of the 1 model(s)'))).toBe(true);
    expect(twice.some((line) => line.includes('none of the 2 model(s)'))).toBe(true);

    // The two places `--lint` raises rather than reporting: `uncalled_primitives` indexes
    // `instances` and `compositions` before anything checks the grammar, and
    // `_called_primitives` catches `OSError` alone.
    expect(byName.get('no-compositions')?.error).toEqual({
      type: 'KeyError',
      message: "'compositions'",
    });
    expect(byName.get('not-json')?.error?.type).toBe('JSONDecodeError');
    // Every other set answers rather than raising.
    expect(
      sets.filter((one) => one.error !== null).map((one) => one.name).sort(),
    ).toEqual(['no-compositions', 'not-json']);
  });

  it.skipIf(!generated)('prints every set as the tools print it, line for line', () => {
    for (const one of recorded().sets) {
      let lines: string[] | null = null;
      let raised: Raised | null = null;
      try {
        lines = reportOf(one);
      } catch (error) {
        raised = raisedAs(error);
      }
      expect(raised, one.name).toEqual(one.error);
      if (one.error !== null) continue;
      expect(lines, one.name).toEqual(one.output);
    }
    // Sixteen whole `--lint` invocations, each loading a primitive library and analysing every
    // document it is given: four seconds alone on the development box, and more whenever another
    // suite of the parity project is running beside it. The budget is the structural suite's
    // (feature 1.3's flake); nothing of the comparison changes.
  }, 120_000);

  it.skipIf(!generated)('reproduces the --lint output of the corpus and of the template', () => {
    // The feature's own test, taken from the files the oracle's *command line* wrote rather than
    // from the in-process reading above: every document of the manifest names the file that holds
    // its findings, and the two files together are the corpus set and the template set.
    const manifest = readOracleManifest();
    const sets = new Map(recorded().sets.map((one) => [one.name, one]));
    const files = new Map<string, string[]>();
    for (const document of manifest.documents) {
      const names = files.get(document.lint) ?? [];
      names.push(document.name);
      files.set(document.lint, names);
    }
    expect([...files.keys()].sort()).toEqual([
      'lint/corpus.txt',
      'lint/decoder-causal-yarn@1.0.0.txt',
    ]);
    for (const [file, documents] of files) {
      const name = file === 'lint/corpus.txt' ? 'corpus' : 'template';
      const recordedSet = sets.get(name) as RecordedSet;
      expect(recordedSet.documents.map((one) => one.path.replace(/^.*\//, '')).length).toBe(
        documents.length,
      );
      const printed = readFileSync(join(oracleOut, file), 'utf8').split('\n');
      expect(printed[printed.length - 1]).toBe('');
      const [head, ...rest] = printed.slice(0, -1);
      // The first line is the command line's, not `lint.py`'s: the core does not write it.
      expect(head).toBe(`lint  ${String(documents.length)} document(s)`);
      expect(reportOf(recordedSet), file).toEqual(rest);
    }
  }, 120_000);
});
