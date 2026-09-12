import { describe, expect, it } from 'vitest';

import { emittedGraph, emittedSplit, nodeProducts } from '@tensorspine/lang';

import { expandedReading, NO_FILTER, ROW_HEIGHT, windowOf } from '../../src/expanded/graph.js';
import { derivedOf } from '../derived/source.js';

/**
 * What the expanded tab costs — §5.6's 16 ms from keystroke to render, and 0.4's two-second
 * budget it is built to stay away from.
 *
 * Three figures, because three things happen at three different moments:
 *
 *  - **reading D1** runs when a derivation lands, and it is the whole graph: every node's
 *    identifier taken apart, its D3 and D4 figures attributed, its edges counted;
 *  - **the rows** run on every filter keystroke, over a graph already read;
 *  - **a node's product counts** run on a click.
 *
 * The comparison that matters is the one *not* made: feature 0.4 measured ELK on these same
 * graphs at 1 504 ms (`gemma3n-kvshare`) and 176 ms (`deepseek-v4-pro`), against §5.6's two
 * seconds and with five flakes under load. Nothing below is within two orders of magnitude of
 * that, because §4.9's reading is D1's own order and no layout runs.
 *
 * The bound is far above what is measured, for the reason every feature since 2.5 has recorded of
 * its own: the unit layer runs a hundred files at once and a guard tight enough to be interesting
 * on an idle machine fails on a loaded one. What it catches is an algorithmic regression; the line
 * printed beside each figure is the measurement.
 */

/** Far above what is measured and far below a regression. */
const BOUND_MS = 900;

const MODELS = ['llama3-8b', 'gemma3n-kvshare', 'deepseek-v4-pro'] as const;

function median(times: number[]): number {
  times.sort((left, right) => left - right);
  return times[Math.floor(times.length / 2)] ?? 0;
}

describe('the cost of the expanded graph', () => {
  for (const name of MODELS) {
    it(`reads ${name}’s D1 inside the budget`, () => {
      const derived = derivedOf(name);
      const times: number[] = [];
      let nodes = 0;
      let edges = 0;
      for (let run = 0; run < 10; run += 1) {
        const at = performance.now();
        const graph = emittedGraph(derived);
        times.push(performance.now() - at);
        nodes = graph.nodes.length;
        edges = graph.edges.length;
      }
      const cost = median(times);
      console.log(
        `${name}: D1 read as ${String(nodes)} nodes and ${String(edges)} edges in ${cost.toFixed(2)} ms` +
          ' — no layout runs (feature 0.4 measured ELK on the same graphs at 176–1 504 ms)',
      );
      expect(cost).toBeLessThan(BOUND_MS);
    });

    it(`filters ${name}’s rows inside the budget for a keystroke`, () => {
      const graph = emittedGraph(derivedOf(name));
      const split = emittedSplit(derivedOf(name), 'decoder[layer<=0]');
      const times: number[] = [];
      let kept = 0;
      for (let run = 0; run < 10; run += 1) {
        const at = performance.now();
        const reading = expandedReading(
          graph,
          { ...NO_FILTER, indices: { layer: { from: 0, to: 3 } } },
          split,
        );
        times.push(performance.now() - at);
        kept = reading.nodes;
      }
      const cost = median(times);
      console.log(
        `${name}: ${String(kept)} of ${String(graph.nodes.length)} rows kept in ${cost.toFixed(2)} ms`,
      );
      expect(cost).toBeLessThan(BOUND_MS);
    });

    it(`counts the products of one of ${name}’s nodes inside the budget`, () => {
      const derived = derivedOf(name);
      const graph = emittedGraph(derived);
      const id = graph.nodes[Math.floor(graph.nodes.length / 2)]?.id ?? '';
      const times: number[] = [];
      for (let run = 0; run < 10; run += 1) {
        const at = performance.now();
        nodeProducts(derived, graph, id);
        times.push(performance.now() - at);
      }
      const cost = median(times);
      console.log(`${name}: the products naming ${id} counted in ${cost.toFixed(2)} ms`);
      expect(cost).toBeLessThan(BOUND_MS);
    });
  }

  it('draws a screenful whatever the graph, which is what keeps the drawing off the budget', () => {
    for (const name of MODELS) {
      const total = emittedGraph(derivedOf(name)).nodes.length;
      const shown = windowOf(total, 0, 24 * ROW_HEIGHT);
      console.log(`${name}: ${String(shown.count)} of ${String(total)} rows rendered at a time`);
      expect(shown.count).toBeLessThanOrEqual(36);
    }
  });
});
