import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { hoistingOf, toPython } from '@tensorspine/lang';
import type { Lang, LibraryHandle } from '@tensorspine/lang/api';

import { unusedQuantities } from '../../src/problems/notices.js';
import { aboutOf, problemsView } from '../../src/problems/rows.js';
import { core, corpus, corpusPath, index, outline, registry, verdictOf } from './source.js';

// What the panel costs, on the corpus's largest documents.
//
// Three of the four are on a **keystroke's** path and are what §5.6's sixteen milliseconds are
// about: Ajv, which §5.4 runs synchronously on the interface's own thread ("< 5 ms on a corpus
// document"); the arrangement, which is redrawn whenever a reading is published; and the map of
// §5.2 rule 7, which `validate` asks for once per run. The fourth — the editor's own notices — is
// computed where the verdict is and not per keystroke.
//
// The bound is far above what is measured, for the reason features 2.5, 2.6 and 2.7 recorded of
// their own guards: the unit layer runs eighty-odd files at once and a guard tight enough to be
// interesting on an idle machine fails on a loaded one. What this catches is an algorithmic
// regression, and the printed line beside it carries the measurement.
//
// One such regression was caught here rather than shipped: the unused-quantity notice asked
// `ReferenceIndex.select` per quantity, which answers the occurrences in *document order* and so
// walks the whole index each time — 27.5 ms over `deepseek-v4-pro`'s 1 245 occurrences and 22
// quantities. What the rule asks is whether *anything* names the quantity, which `of` answers
// against the name alone: 0.14 ms.

/** Far above the measurement and far below a regression: the shape of the answer, not its speed. */
const BOUND_MS = 200;

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

/** The median of twenty runs, and the line that records it. */
function median(body: () => void): number {
  const times: number[] = [];
  for (let run = 0; run < 20; run += 1) {
    const at = performance.now();
    body();
    times.push(performance.now() - at);
  }
  times.sort((left, right) => left - right);
  return times[Math.floor(times.length / 2)] ?? 0;
}

describe('the cost of the panel', () => {
  for (const name of ['llama3-8b', 'gemma3n-kvshare', 'deepseek-v4-pro'] as const) {
    it(`stays inside the budget on ${name}`, async () => {
      const tree = corpus(name);
      const rowsOfOutline = outline(tree);
      const references = index(tree);
      // A document with as many rows as any gesture of the editor can produce: every instance of
      // `norm.rms` pinned to a primitive the base does not carry.
      const { rows } = await verdictOf(lang, library, name, (text) =>
        text.replaceAll('"norm.rms"', '"norm.rmz"'),
      );
      const about = aboutOf(rowsOfOutline, corpusPath(name), name);

      const ajv = median(() => {
        registry().conforms(tree, 'model');
      });
      const hoisting = median(() => {
        hoistingOf(toPython(tree));
      });
      const notices = median(() => {
        unusedQuantities({ rows: rowsOfOutline, index: references });
      });
      const view = median(() => {
        problemsView({ rows, about, group: 'node' });
      });

      console.log(
        `${name}: Ajv ${ajv.toFixed(2)} ms · hoisting ${hoisting.toFixed(2)} ms · ` +
          `notices ${notices.toFixed(2)} ms · arrangement ${view.toFixed(2)} ms ` +
          `over ${String(rows.length)} rows`,
      );
      for (const [what, measured] of [
        ['Ajv', ajv],
        ['hoisting', hoisting],
        ['notices', notices],
        ['arrangement', view],
      ] as const) {
        expect(measured, `${name}: ${what}`).toBeLessThan(BOUND_MS);
      }
    }, 300_000);
  }
});
