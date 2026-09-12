import { describe, expect, it } from 'vitest';

import { readdirSync, statSync } from 'node:fs';

import type { Lang } from '@tensorspine/lang/api';
import { createLang } from '@tensorspine/lang/api/engine';
import {
  basesOf,
  drillGraph,
  drillPresence,
  foldedGraph,
  parse,
  toPython,
  type DrillGraph,
  type PortRef,
} from '@tensorspine/lang';

import { drillModel, type DrillModel } from '../../src/canvas/drill.js';
import { canvasModel, DEFAULT_VIEW } from '../../src/canvas/model.js';
import { place, routes } from '../../src/canvas/layout.js';
import { bindings, everyComposition, reading, registry, shapes } from './source.js';
import { readRepositoryFile, repositoryRoot } from '../presentation/source.js';

/**
 * What the canvas costs — §5.6's table, measured where the feature's own work is.
 *
 * Three figures, and they are the three the feature turns on:
 *
 *  - the **folded reading and the model**, which run on the interface's own thread on every
 *    keystroke (§5.6: "sheet keystroke to render < 16 ms");
 *  - the **layout**, which runs when the shape of the drawing changes and never while a pointer is
 *    down (feature 0.4: the folded canvas is 8–42 boxes and 8–35 ms);
 *  - the **drag's round trip**, `describe` then `check`, which §5.6 gives 20 ms and feature 1.6d
 *    made possible by letting `check` read the analysis the session already holds.
 *
 * The bounds are far above the measurements, for the reason features 2.5, 2.6 and 2.7 recorded of
 * their own: the unit layer runs ninety files at once and a guard tight enough to be interesting
 * on an idle machine fails on a loaded one. What they catch is an algorithmic regression; the
 * printed line beside each is the measurement.
 */

/** Far above what is measured and far below a regression. */
const BOUND_MS = 400;

const MODELS = ['llama3-8b', 'gemma3n-kvshare', 'deepseek-v4-pro'] as const;

function median(times: number[]): number {
  times.sort((left, right) => left - right);
  return times[Math.floor(times.length / 2)] ?? 0;
}

describe('the cost of the folded reading and the model', () => {
  for (const name of MODELS) {
    it(`reads and models ${name} inside the keystroke budget`, () => {
      const read = reading(name);
      const times: number[] = [];
      let boxes = 0;
      for (let run = 0; run < 10; run += 1) {
        const at = performance.now();
        const folded = foldedGraph(read.tree);
        boxes = canvasModel({
          folded,
          facts: read.facts,
          derived: read.derived,
          problems: read.problems,
          collapsed: everyComposition(folded),
          view: DEFAULT_VIEW,
          registry,
          shapes,
          bindings,
        }).boxes.length;
        times.push(performance.now() - at);
      }
      const best = median(times);
      console.log(
        `${name}: ${String(boxes)} boxes, folded + model median ${best.toFixed(2)} ms ` +
          `(§5.6 gives a keystroke 16)`,
      );
      expect(best, name).toBeLessThan(BOUND_MS);
    });
  }
});

describe('the cost of the layout', () => {
  for (const name of MODELS) {
    it(`places ${name}'s folded canvas`, async () => {
      const read = reading(name);
      const model = canvasModel({
        folded: read.folded,
        facts: read.facts,
        derived: read.derived,
        problems: read.problems,
        collapsed: everyComposition(read.folded),
        view: DEFAULT_VIEW,
        registry,
        shapes,
        bindings,
      });
      let best = Number.POSITIVE_INFINITY;
      let drawn = 0;
      for (let run = 0; run < 3; run += 1) {
        const placement = await place({ model });
        best = Math.min(best, placement.milliseconds);
        drawn = routes(model, placement).length;
      }
      console.log(
        `${name}: ${String(model.boxes.length)} boxes and ${String(drawn)} wires, ` +
          `layout best ${best.toFixed(0)} ms (feature 0.4 measured 8–35)`,
      );
      // Feature 1.11's rule for a budget a suite runs beside ninety files: a regression guard.
      expect(best, name).toBeLessThan(1000);
    }, 30_000);
  }
});

describe('the drag’s round trip (§5.6: 20 ms)', () => {
  it('describes the folded canvas and answers a candidate through the API', async () => {
    const lang: Lang = createLang();
    try {
      const files = schemaFiles();
      const loaded = await lang.loadSchemas(files);
      for (const name of MODELS) {
        const text = readRepositoryFile(`data/models/${name}.json`);
        const tree = parse(text);
        const path = `data/models/${name}.json`;
        const { bases } = basesOf(path, toPython(tree));
        const gathered = await lang.loadLibrary(
          bases.map((base) => ({ base, files: baseFiles(base) })),
          loaded.handle,
        );
        const described: number[] = [];
        for (let run = 0; run < 3; run += 1) {
          const at = performance.now();
          await lang.describe(tree, path, {
            library: gathered.handle,
            revision: run,
            folded: true,
          });
          described.push(performance.now() - at);
        }
        const folded = foldedGraph(tree);
        const candidate = twoPorts(folded);
        const checks: number[] = [];
        if (candidate !== null) {
          for (let run = 0; run < 20; run += 1) {
            const at = performance.now();
            await lang.check(path, { edge: candidate });
            checks.push(performance.now() - at);
          }
        }
        console.log(
          `${name}: describe(folded) median ${median(described).toFixed(1)} ms, ` +
            `check median ${median(checks).toFixed(2)} ms (§5.6 gives each 20)`,
        );
        expect(median(checks), name).toBeLessThan(BOUND_MS);
      }
    } finally {
      lang.close();
    }
  }, 120_000);
});

/** Two ports of one folded graph, for a candidate to be asked about. */
function twoPorts(
  folded: ReturnType<typeof foldedGraph>,
): { from: PortRef; to: PortRef } | null {
  const sites = [...folded.byPointer.values()].filter((node) => node.site !== null);
  const from = sites[0];
  const to = sites[1];
  if (from?.site == null || to?.site == null) return null;
  return { from: { site: from.site, port: 'output' }, to: { site: to.site, port: 'input' } };
}

/** The repository's schemas, as the API is handed them: by path. */
function schemaFiles(): Record<string, string> {
  const found: Record<string, string> = {};
  for (const name of readdirSync(`${repositoryRoot}/schemas`)) {
    if (name.endsWith('.json')) found[`schemas/${name}`] = readRepositoryFile(`schemas/${name}`);
  }
  return found;
}

/** Every file of a base, as `loadLibrary` takes them. */
function baseFiles(base: string): Record<string, string> {
  const found: Record<string, string> = {};
  const walk = (at: string): void => {
    for (const entry of readdirSync(`${repositoryRoot}/${at}`)) {
      const path = `${at}/${entry}`;
      if (statSync(`${repositoryRoot}/${path}`).isDirectory()) walk(path);
      else if (entry.endsWith('.json')) found[path] = readRepositoryFile(path);
    }
  };
  walk(base.replace(/\/$/, ''));
  return found;
}

describe('the cost of the drill-in (§4.8)', () => {
  // The drill-in is built on the same branch a keystroke runs (§5.4's `describe` at once), so its
  // reading is held to the same budget as the folded canvas's: the model, the presence over D1 and
  // the placement of the ghost columns, over the three documents the table above measures.
  for (const name of MODELS) {
    it(`reads and models ${name}'s first composition inside the keystroke budget`, () => {
      const read = reading(name);
      const composition = read.folded.nodes.find((node) => node.children.length > 0);
      expect(composition, name).toBeDefined();
      const times: number[] = [];
      let model: DrillModel | null = null;
      for (let run = 0; run < 10; run += 1) {
        const at = performance.now();
        model = drillModel({
          tree: read.tree,
          composition: composition?.name ?? '',
          folded: foldedGraph(read.tree),
          facts: read.facts,
          derived: read.derived,
          problems: read.problems,
          registry,
          shapes,
          bindings,
        });
        times.push(performance.now() - at);
      }
      const best = median(times);
      console.log(
        `${name} › ${composition?.name ?? ''}: ${String(model?.canvas.boxes.length ?? 0)} boxes, ` +
          `${String(model?.canvas.wires.length ?? 0)} wires, ${String(model?.ghosts.length ?? 0)} ghosts, ` +
          `${String(model?.points.length ?? 0)} iterations, ${String(model?.rows.length ?? 0)} strip rows, ` +
          `median ${best.toFixed(2)} ms (§5.6 gives a keystroke 16)`,
      );
      expect(best, name).toBeLessThan(BOUND_MS);
    });
  }

  it('costs the presence over D1 what a reading of the expanded graph costs', () => {
    for (const name of MODELS) {
      const read = reading(name);
      const composition = read.folded.nodes.find((node) => node.children.length > 0);
      const drill = drillGraph(read.tree, { composition: composition?.name ?? '', folded: read.folded });
      expect(drill, name).not.toBeNull();
      const times: number[] = [];
      for (let run = 0; run < 10; run += 1) {
        const at = performance.now();
        drillPresence(drill as DrillGraph, read.derived);
        times.push(performance.now() - at);
      }
      const best = median(times);
      console.log(
        `${name} › ${composition?.name ?? ''}: presence over ${String(drill?.points.length ?? 0)} ` +
          `iterations and ${String(drill?.sites.length ?? 0)} sites, median ${best.toFixed(2)} ms`,
      );
      expect(best, name).toBeLessThan(BOUND_MS);
    }
  });
});
