import { describe, expect, it } from 'vitest';

import { keptBy } from '../../src/derived/naming.js';
import {
  derivedReading,
  ROW_LIMIT,
  sectionsOf,
  TABLE,
  tableRows,
} from '../../src/derived/products.js';
import { context, derivedOf } from './source.js';

/**
 * What the Derived panel costs — §5.6's 16 ms from keystroke to render.
 *
 * The panel redraws whenever a derivation lands and whenever the selection moves, so three figures
 * matter and they are kept apart on purpose:
 *
 *  - **finding the products** walks the derived document's root and nothing else, so it is the
 *    cheapest of the three and runs on every derivation;
 *  - **one product's sections** is the expensive one — it walks every row of the tab that is
 *    *showing*, which is why `derivedReading` answers the products and `sectionsOf` builds one
 *    ({@link sectionsOf}'s own note);
 *  - **filtering and rendering** runs on every selection change over rows that are already built.
 *
 * The bound is far above what is measured, for the reason every feature since 2.5 has recorded of
 * its own: the unit layer runs a hundred files at once and a guard tight enough to be interesting
 * on an idle machine fails on a loaded one. What it catches is an algorithmic regression; the
 * line printed beside each figure is the measurement.
 */

/** Far above what is measured and far below a regression. */
const BOUND_MS = 900;

const MODELS = ['llama3-8b', 'gemma3n-kvshare', 'deepseek-v4-pro'] as const;

function median(times: number[]): number {
  times.sort((left, right) => left - right);
  return times[Math.floor(times.length / 2)] ?? 0;
}

describe('the cost of reading a derived document', () => {
  for (const name of MODELS) {
    it(`finds ${name}’s six products inside the budget`, () => {
      const derived = derivedOf(name);
      const times: number[] = [];
      for (let run = 0; run < 10; run += 1) {
        const at = performance.now();
        const reading = derivedReading(derived, context);
        times.push(performance.now() - at);
        expect(reading.products).toHaveLength(6);
      }
      const cost = median(times);
      console.log(`${name}: the six products found in ${cost.toFixed(2)} ms`);
      expect(cost).toBeLessThan(BOUND_MS);
    });

    it(`builds ${name}’s largest tab and filters it inside the budget`, () => {
      const derived = derivedOf(name);
      const products = derivedReading(derived, context).products;
      let worst = { member: '', cost: 0, rows: 0 };
      for (const product of products) {
        const at = performance.now();
        const sections = sectionsOf(product, context);
        const cost = performance.now() - at;
        const rows = sections.reduce(
          (total, section) => total + (section.kind === TABLE ? section.entries.length : 0),
          0,
        );
        if (cost > worst.cost) worst = { member: product.member, cost, rows };
      }
      console.log(
        `${name}: ${worst.member} is the costliest tab — ${String(worst.rows)} rows in ${worst.cost.toFixed(2)} ms`,
      );
      expect(worst.cost).toBeLessThan(BOUND_MS);
    });

    it(`draws ${name}’s rows under a subject inside the budget`, () => {
      const derived = derivedOf(name);
      const products = derivedReading(derived, context).products;
      const subject = { kind: 'node', name: 'nothing-is-called-this', label: 'x' };
      const times: number[] = [];
      for (const product of products) {
        const sections = sectionsOf(product, context);
        const at = performance.now();
        for (const section of sections) {
          if (section.kind === TABLE) tableRows(section, context, () => true, ROW_LIMIT);
          if (section.kind === TABLE) {
            tableRows(section, context, (entry) => keptBy(entry.named, subject), ROW_LIMIT);
          }
        }
        times.push(performance.now() - at);
      }
      const cost = median(times);
      console.log(`${name}: a tab's rows drawn and filtered in ${cost.toFixed(2)} ms`);
      expect(cost).toBeLessThan(BOUND_MS);
    });
  }
});
