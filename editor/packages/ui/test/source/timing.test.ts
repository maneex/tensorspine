import { describe, expect, it } from 'vitest';

import { parse, serialize, spansOf } from '@tensorspine/lang';
import { schemaProblem } from '@tensorspine/lang/api';

import { minimalEdit } from '../../src/source/edits.js';
import { markersFor } from '../../src/source/markers.js';
import { corpusText, registry } from '../documents/source.js';

/**
 * What the source view costs — §5.6's 16 ms from keystroke to render.
 *
 * Three figures, because three things happen at three different moments:
 *
 *  - **the span map** runs whenever the markers are recomputed or a place is revealed: the whole
 *    text, read by the core's own parser with a `Map.set` per value;
 *  - **the minimal edit** runs when the document moves for a reason that is not the pane's, which
 *    is every canvas gesture and every sheet row;
 *  - **the markers** run when the core's rows change, over the spans.
 *
 * None of the three is on the *typing* path — Monaco owns the text and the store hears from it
 * once per settled edit (§4.10's own debounce) — so what these figures bound is the cost of the
 * pane following the document, which is where a reader notices a stall.
 *
 * The bound is far above what is measured, for the reason every feature since 2.5 has recorded of
 * its own: the unit layer runs a hundred files at once and a guard tight enough to be interesting
 * on an idle machine fails on a loaded one. What it catches is an algorithmic regression; the line
 * printed beside each figure is the measurement.
 */

/** Far above what is measured and far below a regression. */
const BOUND_MS = 400;

const MODELS = ['llama3-8b', 'gemma3n-kvshare', 'deepseek-v4-pro'] as const;

function median(times: number[]): number {
  times.sort((left, right) => left - right);
  return times[Math.floor(times.length / 2)] ?? 0;
}

describe('the cost of the source view', () => {
  for (const name of MODELS) {
    it(`maps ${name}’s places inside the budget for a keystroke`, () => {
      const text = corpusText(name);
      const times: number[] = [];
      let places = 0;
      for (let run = 0; run < 10; run += 1) {
        const at = performance.now();
        const spans = spansOf(text);
        times.push(performance.now() - at);
        places = spans.size;
      }
      const cost = median(times);
      console.log(
        `${name}: ${String(places)} places of ${String(text.length)} characters in ${cost.toFixed(2)} ms`,
      );
      expect(cost).toBeLessThan(BOUND_MS);
    });

    it(`answers the edit that follows a gesture on ${name} inside the budget`, () => {
      const before = corpusText(name);
      const after = serialize(parse(before.replace('"kind": "literal"', '"kind": "literal" ')));
      const times: number[] = [];
      for (let run = 0; run < 10; run += 1) {
        const at = performance.now();
        minimalEdit(before, after);
        times.push(performance.now() - at);
      }
      const cost = median(times);
      console.log(`${name}: the minimal edit of ${String(before.length)} characters in ${cost.toFixed(2)} ms`);
      expect(cost).toBeLessThan(BOUND_MS);
    });

    it(`places ${name}’s rows inside the budget`, () => {
      // A document the grammar refuses — an unknown member of the root — and the rows the stage
      // answers for it, placed over the whole span map of the largest documents there are.
      const text = corpusText(name).replace('"quantities": {', '"kernels": {},\n  "quantities": {');
      const rows = registry()
        .structural(parse(text), 'model')
        .map((row) => schemaProblem(row, `${name}.json`));
      const spans = spansOf(text);
      const times: number[] = [];
      let markers = 0;
      for (let run = 0; run < 10; run += 1) {
        const at = performance.now();
        markers = markersFor(rows, spans, text).length;
        times.push(performance.now() - at);
      }
      const cost = median(times);
      console.log(
        `${name}: ${String(markers)} marker(s) from ${String(rows.length)} row(s) in ${cost.toFixed(2)} ms`,
      );
      expect(markers).toBeGreaterThan(0);
      expect(cost).toBeLessThan(BOUND_MS);
    });
  }
});
