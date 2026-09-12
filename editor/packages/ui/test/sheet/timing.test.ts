import { describe, expect, it } from 'vitest';

import {
  analyse,
  describeAnalysis,
  identityFacts,
  foldedGraph,
  identityReadings,
  parse,
  pyStr,
  quantityReadings,
  serialize,
  toPython,
  type PyRecord,
} from '@tensorspine/lang';
import { DocumentStore, type Path } from '@tensorspine/store';

import { outlineOf } from '../../src/explorer/outline.js';
import { presentation } from '../../src/presentation/index.js';
import { argumentSheet } from '../../src/sheet/arguments.js';
import { instanceSheet } from '../../src/sheet/instance.js';
import { placeSheet } from '../../src/sheet/places.js';
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

/** The key a description holds a site under, which is the analysis's own and not the caller's. */
function keyOfSite(described: ReturnType<typeof describeAnalysis>, where: string): string {
  for (const [key, site] of described.sites) if (site.where === where) return key;
  return '';
}

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

describe('what a menu of §4.11 costs, and what §4.14’s sheet costs (feature 2.13)', () => {
  // The site is `embed` on both, because it is where there is anything to offer: 9 of the 116
  // parameter slots the reference base declares are `shareable` (feature 1.6d), and an attention
  // projection is not one of them — a list of nothing would measure the walk and not the answer.
  for (const [name, where] of [
    ['llama3-8b', 'embed'],
    ['deepseek-v4-pro', 'embed'],
  ] as const) {
    it(`answers ${name}'s partner lists for one site, which is what "Tie to…" asks for`, () => {
      const document = toPython(parse(readRepositoryFile(`data/models/${name}.json`)));
      const analysis = analyse(document, library());
      const times: number[] = [];
      let partners = 0;
      for (let run = 0; run < 3; run += 1) {
        const at = performance.now();
        const described = describeAnalysis(analysis, { only: [where], compatibility: true });
        times.push(performance.now() - at);
        const site = described.sites.get(keyOfSite(described, where));
        partners = (site?.parameters ?? []).reduce((sum, slot) => sum + slot.tiesWith.length, 0);
      }
      const ms = median(times);
      console.log(
        `partners of ${where} (${name}): ${ms.toFixed(1)} ms for ${String(partners)} offered`,
      );
      expect(ms).toBeLessThan(BOUND_MS * 3);
    });
  }

  for (const [name, rule] of [
    ['llama3-8b', 'decoder.attn.q'],
    ['deepseek-v4-pro', 'main.attn_hca.q_a'],
  ] as const) {
    it(`answers ${name}'s ${rule} — the members, the dtypes, the axes and the evaluated names`, () => {
      const document = toPython(parse(readRepositoryFile(`data/models/${name}.json`)));
      const analysis = analyse(document, library());
      const times: number[] = [];
      let instances = 0;
      for (let run = 0; run < 5; run += 1) {
        const at = performance.now();
        const answer = identityFacts(analysis, library(), { rule, state: false });
        times.push(performance.now() - at);
        instances = answer?.instances.length ?? 0;
      }
      const ms = median(times);
      console.log(
        `identityFacts ${rule} (${name}): ${ms.toFixed(2)} ms for ${String(instances)} instances`,
      );
      expect(instances).toBeGreaterThan(0);
      expect(ms).toBeLessThan(BOUND_MS);
    });
  }
});

describe('the facts the sheet reads', () => {
  it('are the core’s — one folded site per declared instance (§5.4)', () => {
    expect(facts('llama3-8b').sites.size).toBe(9);
  });
});

describe('the cost of the other sheets of §4.11 (feature 2.12)', () => {
  for (const name of ['llama3-8b', 'deepseek-v4-pro'] as const) {
    it(`builds ${name}’s Document sheet and its Quantities table inside the budget`, () => {
      const document = tree(name);
      const python = toPython(document);
      const shapes = context.shapes;
      const outline = outlineOf({
        tree: document,
        shapes,
        bindings: presentation(),
        role: MODEL,
        openAll: true,
      });
      const rows = new Map(outline.map((row) => [row.pointer, row]));
      const index = DocumentStore.open(serialize(document), shapes).context.index;
      // What the interface gathers once per revision: the readings every sheet of the document
      // reads (feature 2.12's own memo), and then the sheet itself, per keystroke.
      const gathering: number[] = [];
      const sheets: number[] = [];
      let cells = 0;
      for (let run = 0; run < 5; run += 1) {
        const at = performance.now();
        const readings = {
          quantities: quantityReadings(python as PyRecord),
          identities: identityReadings(python),
          folded: foldedGraph(document),
        };
        gathering.push(performance.now() - at);
        const then = performance.now();
        const common = {
          tree: document,
          role: MODEL,
          context,
          rows,
          index,
          derived: null,
          values: [],
          streams: [],
          problems: [],
          ...readings,
        };
        placeSheet({ ...common, path: [], row: rows.get('') ?? null });
        const table = placeSheet({
          ...common,
          path: ['quantities'] as Path,
          row: rows.get('/quantities') ?? null,
        });
        sheets.push(performance.now() - then);
        cells = (table.table?.rows.length ?? 0) * (table.table?.columns.length ?? 0);
      }
      const readingsMs = median(gathering);
      const sheetMs = median(sheets);
      console.log(
        `places ${name}: ${readingsMs.toFixed(2)} ms of readings, ` +
          `${sheetMs.toFixed(2)} ms for the document sheet and a table of ${String(cells)} cells`,
      );
      expect(readingsMs).toBeLessThan(BOUND_MS);
      expect(sheetMs).toBeLessThan(BOUND_MS);
    });
  }
});
