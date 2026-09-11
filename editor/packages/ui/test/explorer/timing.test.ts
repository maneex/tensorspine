import { describe, expect, it } from 'vitest';

import { outlineOf } from '../../src/explorer/outline.js';
import { bindings, corpus, shapes } from './source.js';

// What the outline costs, on the corpus's largest documents — §5.6's "sheet keystroke to render
// < 16 ms", which is the budget the filter box has to live inside: a keystroke there rebuilds the
// whole outline, and under a filter *every* row is open.
//
// The bound below is far above what is measured, for the reason features 2.5 and 2.6 recorded of
// their own timing guards: the unit layer runs eighty-two files at once, and a guard tight enough
// to be interesting on an idle machine fails on a loaded one. What this catches is an algorithmic
// regression — a walk that became quadratic in the document — and it is the printed line beside
// it that carries the measurement.

/** Far above the measurement and far below a regression: the shape of the answer, not its speed. */
const BOUND_MS = 200;

describe('the cost of the outline', () => {
  for (const [name, expected] of [
    ['llama3-8b', { closed: 30, open: 76 }],
    ['gemma3n-kvshare', { closed: 48, open: 156 }],
    ['deepseek-v4-pro', { closed: 77, open: 334 }],
  ] as const) {
    it(`builds ${name} inside the keystroke budget, opened and closed`, () => {
      const tree = corpus(name);
      for (const openAll of [false, true]) {
        const request = { tree, shapes: shapes(), bindings: bindings(), openAll };
        let rows = 0;
        const times: number[] = [];
        for (let run = 0; run < 20; run += 1) {
          const at = performance.now();
          rows = outlineOf(request).length;
          times.push(performance.now() - at);
        }
        times.sort((left, right) => left - right);
        const median = times[Math.floor(times.length / 2)] ?? 0;
        console.log(
          `${name} ${openAll ? 'every row' : 'as it opens'}: ${String(rows)} rows, ` +
            `median ${median.toFixed(2)} ms, worst ${(times.at(-1) ?? 0).toFixed(2)} ms`,
        );
        expect(rows, name).toBe(openAll ? expected.open : expected.closed);
        expect(median, name).toBeLessThan(BOUND_MS);
      }
    });
  }
});
