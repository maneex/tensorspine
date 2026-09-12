import { readdirSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

import { emittedGraph, emittedSplit, type EmittedGraph } from '@tensorspine/lang';

import {
  expandedReading,
  filtering,
  keeps,
  NO_FILTER,
  OVERSCAN,
  ROW_HEIGHT,
  windowOf,
  type ExpandedFilter,
} from '../../src/expanded/graph.js';
import { derivedOf } from '../derived/source.js';
import { repositoryRoot } from '../presentation/source.js';

/**
 * What the expanded tab shows — §4.9's four filters, its counts, its shading and its window.
 *
 * The facts are the core's (`packages/lang/test/describe/emitted.test.ts` holds those to D1); what
 * is asserted here is the *view*: which rows the reader asked for, what the strip says about them,
 * and that a graph of 455 nodes is drawn a screenful at a time.
 *
 * One of these is not about a filter at all. **ELK must not be reachable from this directory**:
 * feature 0.4 measured a whole expanded layout at 1 504 ms on `gemma3n-kvshare` and 1 588 ms on
 * `whisper-large-v3` against §5.6's two seconds, and §4.9's view is D1's own topological order,
 * which needs no layout. The last case reads the sources and says so, so that a later change that
 * quietly imported the layout would be a failing test and not a two-second tab.
 */

const LLAMA = 'llama3-8b';
const QWEN = 'qwen3.5-4b-text';

const graphs = new Map<string, EmittedGraph>();

function graphOf(name: string): EmittedGraph {
  const held = graphs.get(name);
  if (held !== undefined) return held;
  const graph = emittedGraph(derivedOf(name));
  graphs.set(name, graph);
  return graph;
}

/** A filter with one thing changed. */
function filter(patch: Partial<ExpandedFilter>): ExpandedFilter {
  return { ...NO_FILTER, ...patch };
}

describe('the rows the tab draws', () => {
  it('is every node of D1, in D1’s order, when nothing is filtered', () => {
    const graph = graphOf(QWEN);
    const reading = expandedReading(graph, NO_FILTER, null);
    expect(reading.rows).toHaveLength(195);
    expect(reading.nodes).toBe(195);
    expect(reading.total).toBe(195);
    expect(reading.edges).toBe(258);
    expect(reading.totalEdges).toBe(258);
    expect(reading.rows.map((row) => row.node.id)).toEqual(graph.nodes.map((node) => node.id));
    expect(filtering(NO_FILTER)).toBe(false);
  });

  it('draws the chain’s wire only where D1 carries an edge between two consecutive rows', () => {
    // `llama3-8b`'s topological order is a path: 194 of 194 consecutive pairs are edges, so the
    // first row has no wire and every other one does.
    const llama = expandedReading(graphOf(LLAMA), NO_FILTER, null);
    expect(llama.rows.filter((row) => row.linked)).toHaveLength(194);
    // `gemma3n-kvshare`'s is not: the AltUp path leaves ninety pairs unjoined, and a chain that
    // drew a wire there would be asserting an edge D1 has not got.
    const gemma = expandedReading(graphOf('gemma3n-kvshare'), NO_FILTER, null);
    expect(gemma.rows).toHaveLength(455);
    expect(gemma.rows.filter((row) => row.linked)).toHaveLength(364);
    // And every wire it does draw is an edge of D1.
    const edges = new Set(graphOf('gemma3n-kvshare').edges.map((edge) => `${edge.from} ${edge.to}`));
    gemma.rows.forEach((row, at) => {
      const before = gemma.rows[at - 1];
      const joined = before !== undefined && edges.has(`${before.node.id} ${row.node.id}`);
      expect(row.linked, row.node.id).toBe(joined);
    });
  });

  it('writes a byte figure with the one rendering of a byte count there is', () => {
    const rows = expandedReading(graphOf(LLAMA), NO_FILTER, null).rows;
    const attn = rows.find((row) => row.node.id === 'decoder/attn[layer=1]');
    // `sizeText`, which is `view.py`'s own `fmt_bytes`: S12's generator writes `80.0 MiB` here and
    // `0.0 MiB` for `norm.rms`, where the repository's convention writes `16.0 KiB` — the board's
    // own `.1f MiB` at every scale, and the ninth byte spelling found not to be the repository's.
    expect(attn?.figure).toBe('80.0 MiB');
    const norm = rows.find((row) => row.node.id === 'decoder/attn_n[layer=1]');
    expect(norm?.figure).toBe('8.0 KiB');
    const residual = rows.find((row) => row.node.id === 'decoder/attn_r[layer=1]');
    expect(residual?.figure).toBe('');
  });
});

describe('§4.9’s filters', () => {
  it('keeps the nodes D1 names in an index range — `layer 0–3`', () => {
    const graph = graphOf(QWEN);
    const reading = expandedReading(graph, filter({ indices: { layer: { from: 0, to: 3 } } }), null);
    // Every node whose `layer` is in range, and every node that carries no `layer` at all: a root
    // instance is not outside a composition's range, it is outside the composition.
    const expected = graph.nodes.filter((node) => {
      const bound = node.indices.find((one) => one.name === 'layer');
      return bound === undefined || (bound.value >= 0n && bound.value <= 3n);
    });
    expect(reading.nodes).toBe(expected.length);
    expect(reading.rows.map((row) => row.node.id)).toEqual(expected.map((node) => node.id));
    // Four iterations of six sites, plus `embed`, `final_n` and `lm_head`.
    expect(reading.nodes).toBe(27);
    expect(reading.total).toBe(195);
    // An edge is counted where both of its ends pass, which the strip's own title says.
    const ids = new Set(reading.rows.map((row) => row.node.id));
    expect(reading.edges).toBe(
      graph.edges.filter((edge) => ids.has(edge.from) && ids.has(edge.to)).length,
    );
    expect(reading.edges).toBeLessThan(reading.totalEdges);
  });

  it('keeps a family, a primitive and an identifier search, and reports each as filtering', () => {
    const graph = graphOf(LLAMA);
    const family = expandedReading(graph, filter({ families: ['sequence_operator'] }), null);
    expect(family.nodes).toBe(
      graph.nodes.filter((node) => node.families.includes('sequence_operator')).length,
    );
    expect(family.nodes).toBe(32);

    const primitive = expandedReading(graph, filter({ primitives: ['norm.rms'] }), null);
    expect(primitive.rows.every((row) => row.node.primitive === 'norm.rms')).toBe(true);
    expect(primitive.nodes).toBe(65);

    const search = expandedReading(graph, filter({ search: 'ffn_r[layer=1]' }), null);
    expect(search.rows.map((row) => row.node.id)).toEqual(['decoder/ffn_r[layer=1]']);
    // The search is on the identifier and is not case-sensitive, which a reader typing one expects.
    expect(expandedReading(graph, filter({ search: 'FFN_R[LAYER=1]' }), null).nodes).toBe(1);

    for (const one of [
      { families: ['sequence_operator'] },
      { primitives: ['norm.rms'] },
      { search: 'x' },
      { indices: { layer: { to: 3 } } },
      { within: 'decoder' },
    ]) {
      expect(filtering(filter(one)), JSON.stringify(one)).toBe(true);
    }
  });

  it('narrows to one composition for the layer preview, which is §4.9’s "in place"', () => {
    // `whisper-large-v3` has two compositions, both indexed `layer`: a preview of the decoder at
    // layer 4 must not draw the encoder's fourth layer as well.
    const graph = graphOf('whisper-large-v3');
    const both = expandedReading(graph, filter({ indices: { layer: { from: 4, to: 4 } } }), null);
    const decoder = expandedReading(
      graph,
      filter({ within: 'decoder', indices: { layer: { from: 4, to: 4 } } }),
      null,
    );
    expect(both.nodes).toBeGreaterThan(decoder.nodes);
    expect(decoder.rows.every((row) => row.node.within === 'decoder')).toBe(true);
    expect(decoder.rows.every((row) => row.node.id.includes('[layer=4]'))).toBe(true);
  });

  it('combines the filters, each narrowing what the last left', () => {
    const graph = graphOf(LLAMA);
    const one = expandedReading(
      graph,
      filter({ indices: { layer: { from: 0, to: 3 } }, primitives: ['attention.dense'] }),
      null,
    );
    expect(one.rows.map((row) => row.node.id)).toEqual([
      'decoder/attn[layer=0]',
      'decoder/attn[layer=1]',
      'decoder/attn[layer=2]',
      'decoder/attn[layer=3]',
    ]);
    // And nothing is held back by a filter that holds nothing back.
    const embed = graph.nodes[0];
    expect(embed).toBeDefined();
    if (embed !== undefined) expect(keeps(embed, NO_FILTER)).toBe(true);
  });

  it('keeps nothing where nothing matches, and says so rather than drawing an empty chain', () => {
    const reading = expandedReading(graphOf(LLAMA), filter({ search: 'no such node' }), null);
    expect(reading.rows).toHaveLength(0);
    expect(reading.nodes).toBe(0);
    expect(reading.total).toBe(195);
    expect(reading.edges).toBe(0);
  });
});

describe('§4.18’s "Show split on canvas", arrived at', () => {
  it('shades exactly the block D6 lists, and counts what is on each side', () => {
    for (const model of [LLAMA, QWEN]) {
      const graph = graphOf(model);
      const split = emittedSplit(derivedOf(model), 'decoder[layer<=0]');
      expect(split, model).not.toBeNull();
      const reading = expandedReading(graph, NO_FILTER, split);
      const shaded = reading.rows.filter((row) => row.inBlock).map((row) => row.node.id);
      expect(shaded, model).toEqual(split?.block);
      expect(reading.inBlock, model).toBe(7);
      expect(reading.split?.sizes, model).toEqual([7, 188]);
      expect(reading.split?.crossingValues, model).toBe(1);
      // Nothing outside the block is marked, which is what "shades the block" means: the rows
      // beyond it are drawn as they always are (the stylesheet says why the board's opacity is not).
      expect(reading.rows.filter((row) => !row.inBlock)).toHaveLength(188);
    }
  });

  it('shades only the block’s nodes that also pass the filter', () => {
    const split = emittedSplit(derivedOf(LLAMA), 'decoder[layer<=0]');
    const reading = expandedReading(
      graphOf(LLAMA),
      filter({ primitives: ['attention.dense'] }),
      split,
    );
    expect(reading.inBlock).toBe(1);
    expect(reading.rows.filter((row) => row.inBlock).map((row) => row.node.id)).toEqual([
      'decoder/attn[layer=0]',
    ]);
  });

  it('shades nothing where no split is chosen', () => {
    const reading = expandedReading(graphOf(LLAMA), NO_FILTER, null);
    expect(reading.rows.some((row) => row.inBlock)).toBe(false);
    expect(reading.inBlock).toBe(0);
    expect(reading.split).toBeNull();
  });
});

describe('the virtual window', () => {
  it('draws a screenful with the overscan around it, wherever the scroll is', () => {
    const total = 455;
    const height = 20 * ROW_HEIGHT;
    const top = windowOf(total, 0, height);
    expect(top.first).toBe(0);
    expect(top.count).toBe(20 + OVERSCAN * 2);
    expect(top.before).toBe(0);
    expect(top.after).toBe((total - top.count) * ROW_HEIGHT);
    expect(top.visible).toBe(20);

    const middle = windowOf(total, 100 * ROW_HEIGHT, height);
    expect(middle.first).toBe(100 - OVERSCAN);
    expect(middle.before).toBe((100 - OVERSCAN) * ROW_HEIGHT);
    expect(middle.before + middle.count * ROW_HEIGHT + middle.after).toBe(total * ROW_HEIGHT);

    // The end: the window stops at the last row rather than running past it.
    const end = windowOf(total, total * ROW_HEIGHT, height);
    expect(end.first + end.count).toBe(total);
    expect(end.after).toBe(0);
  });

  it('draws a screenful for a viewport nothing has measured — jsdom’s, and a tab’s first paint', () => {
    const shown = windowOf(455, 0, 0);
    expect(shown.count).toBeGreaterThan(0);
    expect(shown.first).toBe(0);
    expect(shown.before).toBe(0);
  });

  it('never asks for more rows than there are, however tall the viewport', () => {
    const shown = windowOf(3, 0, 4000);
    expect(shown.first).toBe(0);
    expect(shown.count).toBe(3);
    expect(shown.after).toBe(0);
    expect(windowOf(0, 0, 400).count).toBe(0);
  });

  it('is the window every corpus document is drawn through, and it is a screenful', () => {
    // §5.6's "expanded view: 500 nodes interactive, 2 000 with virtualisation": what is drawn is
    // a screenful whatever the graph, which is the whole point of the window.
    for (const model of [LLAMA, QWEN, 'gemma3n-kvshare', 'deepseek-v4-pro']) {
      const total = graphOf(model).nodes.length;
      const shown = windowOf(total, 0, 24 * ROW_HEIGHT);
      expect(shown.count, model).toBeLessThanOrEqual(24 + OVERSCAN * 2);
    }
  });
});

describe('the layout ELK would run', () => {
  it('is not reachable from this view — §4.9 draws D1’s order and loads no engine', () => {
    // Feature 0.4: `gemma3n-kvshare` expanded is 1 504 ms and `whisper-large-v3` 1 588 ms against
    // §5.6's two seconds, and the figure has flaked under load. So the rule is not "lay it out
    // lazily", it is "do not lay it out": the chain is `topological_order`, which the product
    // already computed. A source that imported the layout — statically or dynamically — would put
    // 1.5 s and 1.4 MB back on the path this view is opened by, so the sources are read.
    const here = resolve(repositoryRoot, 'editor', 'packages', 'ui', 'src', 'expanded');
    const files = readdirSync(here).filter((name) => name.endsWith('.ts') || name.endsWith('.tsx'));
    expect(files.length).toBeGreaterThan(0);
    const offenders: string[] = [];
    for (const name of files) {
      const source = readFileSync(join(here, name), 'utf8');
      // Every import specifier of the file, static or dynamic, with the comments left out.
      const text = source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\*.*$/gm, '');
      for (const match of text.matchAll(/(?:from|import)\s*\(?\s*['"]([^'"]+)['"]/g)) {
        const specifier = match[1] ?? '';
        if (/layout|elk/i.test(specifier)) offenders.push(`${name}: ${specifier}`);
      }
    }
    expect(offenders).toEqual([]);
  });
});
