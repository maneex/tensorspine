import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { parse } from '@tensorspine/lang';
import type { Lang, LibraryHandle, Problem } from '@tensorspine/lang/api';
import { PROBLEM_SOURCE, schemaProblem } from '@tensorspine/lang/api';

import { core, corpusPath, readRepositoryFile, registry } from './source.js';

// §5.4's first branch: **Ajv, synchronously, on the interface's own thread**.
//
// > gesture ──► command (patches) ──► document tree ──┬──► Ajv (sync) ──► Problems (schema)
//
// The pipeline runs it per revision on the page's own registry, so a member the grammar refuses
// is a row before the keystroke has finished rather than three hundred milliseconds later with
// everything else; and it drops the verdict's schema rows where it did, so nothing is shown twice.
// That is only sound if the two answer the *same* rows — one implementation of the stage, run on
// two threads — and this is what says so, on the documents a reader actually breaks.

let lang: Lang;
let library: LibraryHandle;
let stop: () => void;

beforeAll(async () => {
  const session = await core();
  lang = session.lang;
  library = session.library;
  stop = session.stop;
}, 120_000);

afterAll(() => {
  stop();
});

/** A row as the panel reads it: source, code, place, message. */
function line(one: Problem): string {
  return `${one.source} ${one.code} ${one.path} ${one.message}`;
}

describe('the page’s own Ajv', () => {
  for (const [what, edit] of [
    ['a member the grammar does not admit', (text: string) => text.replace('"quantities"', '"kernels"')],
    ['a value of the wrong type', (text: string) => text.replace('"version": "1.0.0"', '"version": 1')],
    ['a required member removed', (text: string) => text.replace('"interfaces"', '"interfaced"')],
  ] as const) {
    it(`answers what the worker answers for ${what}`, async () => {
      const path = corpusPath('llama3-8b');
      const text = edit(readRepositoryFile('data/models/llama3-8b.json'));
      let tree;
      try {
        tree = parse(text);
      } catch {
        // A text that is not JSON at all is the JSON layer's refusal, not the grammar's, and the
        // pipeline never holds one: every gesture produces an on-schema tree (D5).
        throw new Error(`${what}: the edit must leave the text parseable`);
      }
      const here = registry().conforms(tree, 'model')
        ? []
        : registry().structural(tree, 'model').map((one) => schemaProblem(one, path));
      const verdict = await lang.validate(tree, path, { library });
      const there = verdict.problems.filter((one) => one.source === PROBLEM_SOURCE.schema);

      expect(here.length).toBeGreaterThan(0);
      expect(here.map(line)).toEqual(there.map(line));
      // And nothing after the grammar ran, which is what makes dropping the verdict's copy safe:
      // the two lists are the whole of what the stage answered.
      expect(verdict.stagesRun).toEqual(['schema']);
    });
  }

  it('says nothing at all about a document the grammar admits', () => {
    const tree = parse(readRepositoryFile('data/models/llama3-8b.json'));
    expect(registry().conforms(tree, 'model')).toBe(true);
    expect(registry().structural(tree, 'model')).toEqual([]);
  });
});
