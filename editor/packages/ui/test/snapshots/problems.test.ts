import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import type { Lang, LibraryHandle } from '@tensorspine/lang/api';

import { bannerOf } from '../../src/problems/banner.js';
import { aboutOf, problemsView } from '../../src/problems/rows.js';
import { core, corpusPath, outline, verdictOf } from '../problems/source.js';

// The panel as lines — plan §0.3's snapshot layer, "rendered strings".
//
// What it pins is the arrangement: which groups the core's rows fall into for a document, the
// place each names, how many rows each holds, and the tools' wording of the first of them. A
// change to any of those is a reviewed diff rather than a surprise in a feature written later —
// and, because every row is the core's, a change to the core's own wording shows up here too,
// beside the parity job that owns it.
//
// **The first row of each group and not every row**, and the reason is a measurement: a primitive
// the base does not carry costs `llama3-8b` four hundred and fifty-five rows, because `analyse`
// answers one per *iteration* and the index is inside the message (`{'layer': 7}`), so the fold —
// which merges rows identical in every field a reader can see — leaves them apart. That is what
// `--validate` prints and what the panel shows; a snapshot of four hundred and fifty-five lines
// would be a diff nobody reads, and the group's own count is what says how many there are.

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

/** The panel for a corpus document with one text substitution, as lines a reader would see. */
async function panelOf(name: string, from?: string, to?: string): Promise<string[]> {
  const { verdict, tree, rows } = await verdictOf(lang, library, name, (text) =>
    from === undefined || to === undefined ? text : text.replaceAll(from, to),
  );
  const view = problemsView({
    rows,
    about: aboutOf(outline(tree), corpusPath(name), name),
    group: 'node',
  });
  const banner = bannerOf(verdict);
  const lines: string[] = [`${String(view.total)} row(s) in ${String(view.groups.length)} group(s)`];
  if (banner !== null) lines.push(`banner ${banner.kind}: ${banner.head}`);
  for (const group of view.groups) {
    lines.push(`${group.label} (${String(group.count)}) @ ${group.detail ?? '—'}`);
    const row = group.rows[0];
    if (row === undefined) continue;
    lines.push(
      [
        ' ',
        row.problem.severity,
        row.problem.source,
        row.problem.code === '' ? '—' : row.problem.code,
        row.problem.message,
        row.count > 1 ? `× ${String(row.count)}` : '',
        group.rows.length > 1 ? `(+${String(group.rows.length - 1)} more)` : '',
        `@ ${row.problem.path === '' ? '—' : row.problem.path}`,
      ]
        .filter((part) => part !== '')
        .join(' '),
    );
  }
  return lines;
}

describe('the panel', () => {
  it('has nothing to say about a corpus document', async () => {
    expect(await panelOf('llama3-8b')).toMatchSnapshot();
  });

  it('says what a slot the primitive has not costs, at the place it was written', async () => {
    expect(await panelOf('llama3-8b', '"parameter": "gate"', '"parameter": "gatez"')).toMatchSnapshot();
  });

  it('says what a primitive the base does not carry costs', async () => {
    expect(await panelOf('llama3-8b', '"norm.rms"', '"norm.rmz"')).toMatchSnapshot();
  });

  it('says what an argument that resolves to nothing costs', async () => {
    expect(await panelOf('llama3-8b', '"quantity": "eps"', '"quantity": "epsz"')).toMatchSnapshot();
  });
});
