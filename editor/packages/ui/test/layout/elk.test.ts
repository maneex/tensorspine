import { existsSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import {
  LayoutError,
  layout,
  type Layout,
  type PlacedNode,
  type RoutedEdge,
} from '../../src/layout/elk.js';
import {
  diagramsFor,
  expandedDiagram,
  expandedPath,
  foldedDiagram,
  readExpanded,
  readModel,
  spikeModels,
  type Diagram,
} from '../../../../spikes/layout/graphs.js';

/**
 * Feature 0.4 — the layout spike. What ELK must give the canvases of §4.7 and §4.9, held to the
 * real graphs of the corpus rather than to a fixture: `llama3-8b`, `gemma3n-kvshare` and
 * `deepseek-v4-pro`, folded (compositions as compound nodes, and collapsed as artboard S1 draws
 * them) and expanded (D1, as artboard S12 draws it).
 *
 * Four properties, the ones the feature names:
 *
 *   1. no two node boxes overlap — for boxes where neither contains the other, since a compound
 *      node contains its children by construction;
 *   2. every edge's source is above its target, which is what "top-to-bottom" means when the
 *      graph is acyclic (and the folded graph is: a carry edge is a badge, never an edge, §4.8);
 *   3. a compound node's box contains its children, with room for the chrome it declared;
 *   4. the largest expanded graph is laid out within two seconds (§5.6's budget for a product
 *      that must never block; the layout runs in a worker).
 *
 * The graphs come from `editor/spikes/layout/graphs.ts` — the spike's projection of a document
 * and of D1 — because the language's vocabulary may not appear in `packages/ui/src` (§1) and the
 * core that will answer it is feature 2.9. The expanded graphs are the oracle's `--d1` output
 * (§0.5): they need `pnpm oracle`, which CI runs before every suite.
 */

const inCI = process.env['CI'] !== undefined && process.env['CI'] !== '';
const oracleGenerated = spikeModels.every((model) => existsSync(expandedPath(model)));

/** One layout per diagram, shared by the properties: laying gemma3n's D1 out costs a second. */
const laidOut = new Map<string, Promise<Layout>>();
function placedOnce(diagram: Diagram): Promise<Layout> {
  const known = laidOut.get(diagram.id);
  if (known !== undefined) return known;
  const started = layout(diagram.graph);
  laidOut.set(diagram.id, started);
  return started;
}

/** Every ancestor of a node, innermost first, from the containment the layout returns. */
function ancestors(placed: Layout, node: PlacedNode): string[] {
  const chain: string[] = [];
  let parent = node.parent;
  while (parent !== null) {
    chain.push(parent);
    parent = placed.byId.get(parent)?.parent ?? null;
  }
  return chain;
}

/** Two boxes share ink: a strict overlap, touching borders excluded. */
function overlap(a: PlacedNode, b: PlacedNode): boolean {
  return (
    a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height
  );
}

/** Every pair of boxes that overlap without one holding the other. */
function overlappingPairs(placed: Layout): string[] {
  const found: string[] = [];
  for (let i = 0; i < placed.nodes.length; i += 1) {
    const a = placed.nodes[i];
    if (a === undefined) continue;
    const above = new Set(ancestors(placed, a));
    for (let j = i + 1; j < placed.nodes.length; j += 1) {
      const b = placed.nodes[j];
      if (b === undefined) continue;
      if (above.has(b.id) || ancestors(placed, b).includes(a.id)) continue;
      if (overlap(a, b)) found.push(`${a.id} × ${b.id}`);
    }
  }
  return found;
}

/** Every edge whose source box is not wholly above its target box. */
function notFlowingDown(placed: Layout): string[] {
  const found: string[] = [];
  for (const edge of placed.edges) {
    const from = placed.byId.get(edge.source);
    const to = placed.byId.get(edge.target);
    if (from === undefined || to === undefined) continue;
    if (from.y + from.height > to.y) found.push(`${edge.source} → ${edge.target}`);
  }
  return found;
}

/** Every child that leaves the box of the node holding it. */
function escapingChildren(placed: Layout): string[] {
  const found: string[] = [];
  for (const node of placed.nodes) {
    if (node.parent === null) continue;
    const parent = placed.byId.get(node.parent);
    if (parent === undefined) {
      found.push(`${node.id} names ${node.parent}, which was not placed`);
      continue;
    }
    const inside =
      node.x >= parent.x &&
      node.y >= parent.y &&
      node.x + node.width <= parent.x + parent.width &&
      node.y + node.height <= parent.y + parent.height;
    if (!inside) found.push(`${node.id} escapes ${parent.id}`);
  }
  return found;
}

/** A topological order, or the cycle's absence of one: the precondition property 2 rests on. */
function acyclic(diagram: Diagram): boolean {
  const incoming = new Map<string, number>();
  const outgoing = new Map<string, string[]>();
  const walk = (nodes: Diagram['graph']['nodes']): void => {
    for (const node of nodes) {
      incoming.set(node.id, incoming.get(node.id) ?? 0);
      if (node.children !== undefined) walk(node.children);
    }
  };
  walk(diagram.graph.nodes);
  for (const edge of diagram.graph.edges) {
    incoming.set(edge.target, (incoming.get(edge.target) ?? 0) + 1);
    outgoing.set(edge.source, [...(outgoing.get(edge.source) ?? []), edge.target]);
  }
  const ready = [...incoming].filter(([, count]) => count === 0).map(([id]) => id);
  let seen = 0;
  while (ready.length > 0) {
    const id = ready.pop();
    if (id === undefined) break;
    seen += 1;
    for (const next of outgoing.get(id) ?? []) {
      const left = (incoming.get(next) ?? 0) - 1;
      incoming.set(next, left);
      if (left === 0) ready.push(next);
    }
  }
  return seen === incoming.size;
}

describe('the folded canvas is laid out top to bottom (§4.7)', () => {
  for (const model of spikeModels) {
    const document = readModel(model);
    for (const reading of ['folded', 'collapsed'] as const) {
      const diagram = foldedDiagram(model, document, reading);

      it(`${model} · ${reading}: the graph the canvas draws is acyclic`, () => {
        // A carry edge (`ffn_r[layer−1] → attn_n`) is a badge on the receiving card, not an edge:
        // drawn inside one representative iteration it would look like a cycle without being one
        // (§4.8, D8). Property 2 below is only meaningful because of it.
        expect(acyclic(diagram)).toBe(true);
      });

      it(`${model} · ${reading}: no two boxes overlap`, async () => {
        expect(overlappingPairs(await placedOnce(diagram))).toEqual([]);
      });

      it(`${model} · ${reading}: every edge's source is above its target`, async () => {
        expect(notFlowingDown(await placedOnce(diagram))).toEqual([]);
      });

      it(`${model} · ${reading}: every compound node contains its children`, async () => {
        const placed = await placedOnce(diagram);
        expect(escapingChildren(placed)).toEqual([]);
        const declared = diagram.graph.nodes.filter((node) => node.children !== undefined).length;
        const compound = placed.nodes.filter((node) =>
          placed.nodes.some((child) => child.parent === node.id),
        );
        expect(compound.length).toBe(reading === 'folded' ? declared : 0);
      });
    }
  }

  it('draws every composition of the three documents as a compound node', () => {
    // The property above is vacuous if nothing is compound: llama3-8b has one composition,
    // gemma3n-kvshare one, deepseek-v4-pro three, and the folded reading opens all of them.
    const compound = spikeModels.map(
      (model) =>
        foldedDiagram(model, readModel(model), 'folded').graph.nodes.filter(
          (node) => node.children !== undefined,
        ).length,
    );
    expect(compound).toEqual([1, 1, 3]);
    expect(compound.every((count) => count > 0)).toBe(true);
  });
});

describe.skipIf(!oracleGenerated)('the expanded graph is laid out over D1 (§4.9)', () => {
  for (const model of spikeModels) {
    const diagram = expandedDiagram(model, readExpanded(model));

    it(`${model}: no two boxes overlap`, async () => {
      expect(overlappingPairs(await placedOnce(diagram))).toEqual([]);
    });

    it(`${model}: every edge's source is above its target`, async () => {
      expect(notFlowingDown(await placedOnce(diagram))).toEqual([]);
    });

    it(`${model}: every node D1 emits is placed, in D1's order`, () => {
      const d1 = readExpanded(model).d1;
      expect(diagram.graph.nodes.map((node) => node.id)).toEqual(d1.topological_order);
      expect(diagram.graph.edges.length).toBe(d1.edges.length);
    });
  }

  it(
    'lays the largest expanded graph out in under two seconds',
    async () => {
      // The budget of §5.6, on the largest of the three: gemma3n-kvshare, 455 nodes and 634
      // edges. The figure is the best of three runs in a warm process, which is what a session
      // pays after its first layout; the measurements of `editor/spikes/layout/NOTE.md` record
      // the first one too.
      const largest = spikeModels
        .map((model) => expandedDiagram(model, readExpanded(model)))
        .sort((a, b) => b.graph.nodes.length - a.graph.nodes.length)[0];
      expect(largest).toBeDefined();
      if (largest === undefined) return;
      expect(largest.model).toBe('gemma3n-kvshare');
      expect(largest.graph.nodes.length).toBeGreaterThanOrEqual(455);

      let best = Number.POSITIVE_INFINITY;
      for (let run = 0; run < 3; run += 1) {
        const placed = await layout(largest.graph);
        best = Math.min(best, placed.milliseconds);
      }
      expect(best).toBeLessThan(2000);
    },
    30_000,
  );

  it('draws the three models the spike renders', () => {
    const drawn = spikeModels.flatMap((model) => diagramsFor(model).map((diagram) => diagram.id));
    expect(drawn).toEqual([
      'llama3-8b.folded',
      'llama3-8b.collapsed',
      'llama3-8b.expanded',
      'gemma3n-kvshare.folded',
      'gemma3n-kvshare.collapsed',
      'gemma3n-kvshare.expanded',
      'deepseek-v4-pro.folded',
      'deepseek-v4-pro.collapsed',
      'deepseek-v4-pro.expanded',
    ]);
  });
});

describe('the oracle the expanded graphs are read from', () => {
  it.runIf(inCI)('has been generated before the suites, as the CI job runs it', () => {
    expect(oracleGenerated).toBe(true);
  });
});

describe('the layout module', () => {
  it('returns absolute coordinates, whatever the containment depth', async () => {
    const placed = await layout({
      nodes: [
        { id: 'a', width: 100, height: 40 },
        {
          id: 'group',
          width: 0,
          height: 0,
          padding: { top: 30, right: 12, bottom: 12, left: 12 },
          children: [
            { id: 'inner', width: 80, height: 30 },
            { id: 'inner2', width: 80, height: 30 },
          ],
        },
      ],
      edges: [
        { id: 'a→inner', source: 'a', target: 'inner' },
        { id: 'inner→inner2', source: 'inner', target: 'inner2' },
      ],
    });
    const group = placed.byId.get('group');
    const inner = placed.byId.get('inner');
    expect(group).toBeDefined();
    expect(inner).toBeDefined();
    if (group === undefined || inner === undefined) return;
    // The declared padding is room the children may not take.
    expect(inner.x - group.x).toBeGreaterThanOrEqual(12);
    expect(inner.y - group.y).toBeGreaterThanOrEqual(30);
    expect(escapingChildren(placed)).toEqual([]);
    // An edge inside the compound node is routed in the same frame as one outside it.
    for (const edge of placed.edges) {
      expect(edge.points.length).toBeGreaterThanOrEqual(2);
      for (const point of edge.points) {
        expect(point.x).toBeGreaterThanOrEqual(0);
        expect(point.y).toBeGreaterThanOrEqual(0);
        expect(point.y).toBeLessThanOrEqual(placed.height);
      }
    }
  });

  it('reserves room for an edge label and places it', async () => {
    const placed = await layout({
      nodes: [
        { id: 'a', width: 100, height: 40 },
        { id: 'b', width: 100, height: 40 },
      ],
      edges: [
        { id: 'e', source: 'a', target: 'b', label: { text: 'decoder.entry', width: 74, height: 12 } },
      ],
    });
    const edge: RoutedEdge | undefined = placed.edges[0];
    expect(edge?.label?.text).toBe('decoder.entry');
    expect(edge?.label?.x).toBeGreaterThan(0);
    expect(placed.width).toBeGreaterThan(100);
  });

  it('refuses a graph it cannot lay out, rather than draw something else', async () => {
    await expect(
      layout({ nodes: [{ id: 'a', width: 1, height: 1 }], edges: [{ id: 'e', source: 'a', target: 'b' }] }),
    ).rejects.toBeInstanceOf(LayoutError);
    await expect(
      layout({
        nodes: [
          { id: 'a', width: 1, height: 1 },
          { id: 'a', width: 1, height: 1 },
        ],
        edges: [],
      }),
    ).rejects.toBeInstanceOf(LayoutError);
    await expect(
      layout({
        nodes: [
          { id: 'a', width: 1, height: 1 },
          { id: 'b', width: 1, height: 1 },
        ],
        edges: [
          { id: 'e', source: 'a', target: 'b' },
          { id: 'e', source: 'b', target: 'a' },
        ],
      }),
    ).rejects.toBeInstanceOf(LayoutError);
  });

  it('lays a graph out through an engine the caller supplies', async () => {
    // The static application hands the module an ELK backed by a Web Worker (§5.6); the module
    // speaks only the interface, which is what makes that substitution possible.
    let asked = 0;
    const placed = await layout(
      { nodes: [{ id: 'a', width: 10, height: 10 }], edges: [] },
      {
        engine: {
          layout: (graph) => {
            asked += 1;
            return Promise.resolve({
              ...graph,
              x: 0,
              y: 0,
              width: 10,
              height: 10,
              children: [{ id: 'a', x: 3, y: 4, width: 10, height: 10 }],
            });
          },
        },
      },
    );
    expect(asked).toBe(1);
    expect(placed.byId.get('a')).toMatchObject({ x: 3, y: 4 });
  });
});
