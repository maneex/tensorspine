import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import type { Lang, LibraryHandle, Problem } from '@tensorspine/lang/api';
import { PROBLEM_SEVERITY, PROBLEM_SOURCE } from '@tensorspine/lang/api';

import { aboutOf, FLAT, problemsView } from '../../src/problems/rows.js';
import { core, corpusPath, outline, verdictOf } from './source.js';

// The panel's row model — §4.17. Every row it arranges is a row the core answered, so every case
// below runs the real validator over a real document: what is under test is the *arrangement* —
// the fold, the groups, the filter, the place a row navigates to — and a fixture of made-up rows
// would test it against something the editor never sees.

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

describe('a corpus document', () => {
  it('shows no row at all: the panel is empty because the core said nothing', async () => {
    const { verdict, tree } = await verdictOf(lang, library, 'llama3-8b');
    expect(verdict.problems).toEqual([]);
    const view = problemsView({
      rows: [],
      about: aboutOf(outline(tree), corpusPath('llama3-8b'), 'llama3-8b'),
    });
    expect(view.total).toBe(0);
    expect(view.groups).toEqual([]);
    expect(view.counts).toEqual({ error: 0, warning: 0, notice: 0 });
  });
});

describe('the fold', () => {
  it('shows one row per line the core prints, with the count it stood for', async () => {
    // A scoped parameter rule naming a slot the primitive has not: `analyse` answers the same line
    // once per iteration of `decoder`, thirty-two times, with no index anywhere in it.
    const { rows, tree } = await verdictOf(lang, library, 'llama3-8b', (text) =>
      text.replaceAll('"parameter": "gate"', '"parameter": "gatez"'),
    );
    const about = aboutOf(outline(tree), corpusPath('llama3-8b'), 'llama3-8b');
    const view = problemsView({ rows, about, group: FLAT });
    const folded = view.groups[0]?.rows ?? [];
    const v7 = folded.filter((row) => row.problem.code === 'V7' && row.problem.message.includes('gatez'));
    expect(v7).toHaveLength(1);
    expect(v7[0]?.count).toBe(32);
    // Nothing is lost: the counts add up to what the core answered.
    expect(folded.reduce((sum, row) => sum + row.count, 0)).toBe(rows.length);
    expect(view.total).toBe(rows.length);
  });

  it('never folds two rows that differ in anything a reader can see', async () => {
    const { rows } = await verdictOf(lang, library, 'llama3-8b', (text) =>
      text.replaceAll('"parameter": "gate"', '"parameter": "gatez"'),
    );
    const view = problemsView({ rows, group: FLAT });
    const seen = new Set<string>();
    for (const row of view.groups[0]?.rows ?? []) {
      const key = [row.problem.source, row.problem.code, row.problem.message, row.problem.path].join('|');
      expect(seen.has(key)).toBe(false);
      seen.add(key);
    }
  });
});

describe('grouping', () => {
  it('puts a row under the deepest place the document declares at or above its pointer', async () => {
    const { rows, tree } = await verdictOf(lang, library, 'llama3-8b', (text) =>
      text.replaceAll('"parameter": "gate"', '"parameter": "gatez"'),
    );
    const about = aboutOf(outline(tree), corpusPath('llama3-8b'), 'llama3-8b');
    const view = problemsView({ rows, about, group: 'node' });
    const keys = view.groups.map((group) => group.key);
    // The scoped binding rule the refusal names — the *written* place, which is feature 2.7's
    // finding answered in the core (§5.2 rule 7 read backwards) — and the site whose slot the
    // rule stopped binding.
    expect(keys).toContain('place:/compositions/decoder/bindings/parameters/ffn.gate');
    expect(keys).toContain('place:/compositions/decoder/instances/ffn');
    const binding = view.groups.find(
      (group) => group.key === 'place:/compositions/decoder/bindings/parameters/ffn.gate',
    );
    expect(binding?.label).toBe('ffn.gate');
    expect(binding?.detail).toBe('/compositions/decoder/bindings/parameters/ffn.gate');
  });

  it('puts a row that names no place under the document', () => {
    const rows = [
      {
        problem: {
          code: '',
          message: 'something about the whole document',
          path: '',
          severity: PROBLEM_SEVERITY.warning,
          source: PROBLEM_SOURCE.lint,
        } satisfies Problem,
        stale: false,
      },
    ];
    const about = aboutOf([], 'models/x.json', 'x');
    const view = problemsView({ rows, about, group: 'node' });
    expect(view.groups.map((group) => group.key)).toEqual(['document']);
    expect(view.groups[0]?.label).toBe('x');
  });

  it('groups by unit when asked, which is what a row about another file needs', () => {
    const rows = [
      row({ file: 'bases/acme/primitives/a/1.0.0.json', message: 'axis missing' }),
      row({ file: 'bases/acme/primitives/b/1.0.0.json', message: 'role missing' }),
      row({ file: 'bases/acme/primitives/a/1.0.0.json', message: 'port missing' }),
    ];
    const view = problemsView({ rows, group: 'unit' });
    expect(view.groups.map((group) => group.label)).toEqual(['1.0.0.json', '1.0.0.json']);
    expect(view.groups.map((group) => group.detail)).toEqual([
      'bases/acme/primitives/a/1.0.0.json',
      'bases/acme/primitives/b/1.0.0.json',
    ]);
    expect(view.groups[0]?.count).toBe(2);
  });

  it('groups nothing when asked for a flat list, and keeps the core’s own order', async () => {
    const { rows } = await verdictOf(lang, library, 'llama3-8b', (text) =>
      text.replaceAll('"parameter": "gate"', '"parameter": "gatez"'),
    );
    const view = problemsView({ rows, group: FLAT });
    expect(view.groups).toHaveLength(1);
    expect(view.groups[0]?.key).toBe('');
    const messages = (view.groups[0]?.rows ?? []).map((one) => one.problem.message);
    const first = rows.map((one) => one.problem.message);
    // Every row appears where its first occurrence did.
    expect(messages).toEqual([...new Set(first)]);
  });
});

describe('the filter', () => {
  it('keeps a source, a severity and a text, and says how many it held back', () => {
    const rows = [
      row({ message: 'an error of the grammar', source: PROBLEM_SOURCE.schema }),
      row({ message: 'an advisory', source: PROBLEM_SOURCE.lint, severity: PROBLEM_SEVERITY.warning }),
      row({ message: 'a notice of the editor', source: PROBLEM_SOURCE.editor, severity: PROBLEM_SEVERITY.notice }),
    ];
    expect(problemsView({ rows, group: FLAT }).shown).toBe(3);
    expect(problemsView({ rows, group: FLAT, filter: { sources: [PROBLEM_SOURCE.lint], severities: [], text: '' } }).shown).toBe(1);
    expect(
      problemsView({
        rows,
        group: FLAT,
        filter: { sources: [], severities: [PROBLEM_SEVERITY.notice], text: '' },
      }).shown,
    ).toBe(1);
    expect(problemsView({ rows, group: FLAT, filter: { sources: [], severities: [], text: 'ADVISORY' } }).shown).toBe(1);
    const none = problemsView({ rows, group: FLAT, filter: { sources: [], severities: [], text: 'zzz' } });
    expect(none.shown).toBe(0);
    expect(none.total).toBe(3);
  });

  it('counts every row by severity before the filter, which is what the segment shows', () => {
    const rows = [
      row({ message: 'a' }),
      row({ message: 'b', severity: PROBLEM_SEVERITY.warning }),
      row({ message: 'c', severity: PROBLEM_SEVERITY.notice }),
      row({ message: 'd', severity: PROBLEM_SEVERITY.notice }),
    ];
    const view = problemsView({ rows, group: FLAT, filter: { sources: [], severities: [PROBLEM_SEVERITY.error], text: '' } });
    expect(view.counts).toEqual({ error: 1, warning: 1, notice: 2 });
    expect(view.shown).toBe(1);
  });

  it('offers every source that is present, and no other', () => {
    const rows = [row({ message: 'a' }), row({ message: 'b', source: PROBLEM_SOURCE.lint })];
    expect(problemsView({ rows }).sources).toEqual([PROBLEM_SOURCE.schema, PROBLEM_SOURCE.lint]);
  });
});

/** A row, with the fields a case names and the panel's own defaults for the rest. */
function row(one: Partial<Problem> & { message: string }): { problem: Problem; stale: boolean } {
  return {
    problem: {
      code: '',
      path: '',
      severity: PROBLEM_SEVERITY.error,
      source: PROBLEM_SOURCE.schema,
      ...one,
    },
    stale: false,
  };
}
