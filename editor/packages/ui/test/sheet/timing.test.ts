import { describe, expect, it } from 'vitest';

import { analyse, describeAnalysis, parse, pyStr, toPython } from '@tensorspine/lang';

import { argumentSheet } from '../../src/sheet/arguments.js';
import { instanceSheet } from '../../src/sheet/instance.js';
import { library } from '../canvas/source.js';
import { readRepositoryFile } from '../presentation/source.js';
import { MODEL, artifactOf, context, derivedOf, facts, siteOf, tree } from './source.js';

/**
 * What the sheet costs — §5.6's "sheet keystroke to render < 16 ms", measured where the work is.
 *
 * Two figures, and they are the two this feature turns on:
 *
 *  - **the sheet of one site**, which is rebuilt on every keystroke: the rows of §4.12, the
 *    sections of §4.11 and the products' rows for that site. Feature 2.3 measured the form model
 *    underneath at 0.24 ms for eleven rows and warned that `attention.dense` as *one* form is 66 ms
 *    for 9 271 rows — which is why the argument rows are the core's flat facts and not a walk of
 *    the declaration.
 *  - **the compatibility knob** of feature 1.11: what `describe` costs with the partner lists and
 *    without them, on the document the finding named.
 *
 * The bounds are far above the measurements, as every timing suite of this package states: the
 * unit layer runs a hundred files at once and a guard tight enough to be interesting on an idle
 * machine fails on a loaded one. The printed line beside each is the measurement.
 */

const BOUND_MS = 400;

function median(times: number[]): number {
  times.sort((left, right) => left - right);
  return times[Math.floor(times.length / 2)] ?? 0;
}

describe('the cost of one site’s sheet', () => {
  for (const [name, where] of [
    ['llama3-8b', 'decoder/attn[layer=0]'],
    ['deepseek-v4-pro', 'main/attn_hca[layer=0]'],
  ] as const) {
    it(`builds ${where} of ${name} inside the keystroke budget`, () => {
      const site = siteOf(name, where);
      const artifact = artifactOf(pyStr(site.primitive), pyStr(site.version)) ?? undefined;
      const held = tree(name);
      const derived = derivedOf(name, where);
      const times: number[] = [];
      let rows = 0;
      for (let run = 0; run < 10; run += 1) {
        const at = performance.now();
        const sheet = argumentSheet({
          facts: site.arguments.facts,
          invariants: site.arguments.invariants,
          tree: held,
          role: MODEL,
          context,
          shapes: context.shapes,
          artifact,
          showInapplicable: true,
        });
        instanceSheet(site, derived);
        times.push(performance.now() - at);
        rows = sheet.rows.length;
      }
      const ms = median(times);
      console.log(`sheet ${name} ${where}: ${ms.toFixed(2)} ms / ${String(rows)} rows`);
      expect(ms).toBeLessThan(BOUND_MS);
    });
  }
});

describe('the compatibility knob', () => {
  it('is what a document-wide walk costs, measured on the document feature 1.11 named', () => {
    const name = 'deepseek-v4-pro';
    const document = toPython(parse(readRepositoryFile(`data/models/${name}.json`)));
    const analysis = analyse(document, library());
    const timed = (compatibility: boolean): number => {
      const times: number[] = [];
      for (let run = 0; run < 3; run += 1) {
        const at = performance.now();
        describeAnalysis(analysis, { folded: true, compatibility });
        times.push(performance.now() - at);
      }
      return median(times);
    };
    const without = timed(false);
    const with_ = timed(true);
    console.log(
      `describe(folded) on ${name}: ${with_.toFixed(1)} ms with the lists, ${without.toFixed(1)} ms without`,
    );
    expect(without).toBeLessThan(BOUND_MS);
    expect(with_).toBeLessThan(BOUND_MS * 3);
  });
});

describe('the facts the sheet reads', () => {
  it('are the core’s — one folded site per declared instance (§5.4)', () => {
    expect(facts('llama3-8b').sites.size).toBe(9);
  });
});
