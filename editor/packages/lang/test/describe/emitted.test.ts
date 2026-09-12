import { beforeAll, describe, expect, it } from 'vitest';

import { parse } from '../../src/json/index.js';
import {
  basesOf,
  declaredSite,
  derive,
  emittedGraph,
  emittedSplit,
  loadLibrary,
  nodeIndices,
  nodeProducts,
  nodeWithin,
  pointLabel,
  toPython,
  type EmittedGraph,
  type PyValue,
} from '../../src/index.js';
import { nodeSource } from '../library/source.js';
import { repositoryRoot } from '../json/repository.js';
import { repositorySchemas } from '../schema/repository.js';
import { corpus } from './source.js';

/**
 * D1 read as §4.9's expanded view reads it, held to D1 itself.
 *
 * Four claims, and each is one a component must not decide for itself:
 *
 *  - **the identifier is taken apart the way `whereOfSite` puts it together** — over every node of
 *    every corpus document, `<declared site>[<indices>]` rebuilt from what this module answers is
 *    the identifier D1 wrote. That is the inverse the inventory's §7 forbids a component to
 *    improvise, and it is what the index-range filter stands on;
 *  - **the nodes and the edges are D1's, in D1's order** — the counts and the sequence are the
 *    product's own, never recomputed;
 *  - **a node's figure is D3's and D4's numbers, added once per identity** — the same rule
 *    `derivedFacts` applies one level up, checked against the row a card already shows;
 *  - **a split's block is D6's** — read, never derived, which is what "shades the block D6 lists"
 *    means.
 */

const source = nodeSource(repositoryRoot);

/** The corpus documents these claims are made over — the three of the layout spike and four more. */
const MODELS = [
  'llama3-8b',
  'qwen3.5-4b-text',
  'gemma3n-kvshare',
  'deepseek-v4-pro',
  'whisper-large-v3',
  'shieldstral-3b-composite',
  'colbert-v2',
] as const;

const derivations = new Map<string, PyValue>();

/** One corpus document derived, once per suite. */
function derivedOf(name: string): PyValue {
  const held = derivations.get(name);
  if (held !== undefined) return held;
  const path = `data/models/${name}.json`;
  const tree = parse(corpus(name));
  const { bases, problem } = basesOf(path, toPython(tree));
  if (problem !== null) throw new Error(`${name}: ${problem.message}`);
  const library = loadLibrary(bases, { schemas: repositorySchemas(), source });
  const derived: PyValue = derive(tree, { schemas: repositorySchemas(), library });
  derivations.set(name, derived);
  return derived;
}

const graphs = new Map<string, EmittedGraph>();

function graphOf(name: string): EmittedGraph {
  const held = graphs.get(name);
  if (held !== undefined) return held;
  const graph = emittedGraph(derivedOf(name));
  graphs.set(name, graph);
  return graph;
}

// Seven corpus documents derived once for the whole file, so that no case is timed on the others'
// derivations: the first of them would otherwise pay for all seven, which on a loaded runner is
// the five-second default and not anything about this reading (feature 2.14's own lesson, one
// suite along).
beforeAll(() => {
  for (const model of MODELS) graphOf(model);
}, 300_000);

/** D1 as the derived document holds it, for the assertions that compare with the product itself. */
function d1Of(name: string): Record<string, PyValue> {
  const derived = derivedOf(name) as Record<string, PyValue>;
  return derived['d1'] as unknown as Record<string, PyValue>;
}

describe('a node identifier, taken apart', () => {
  it('rebuilds every node of every corpus document exactly as §5.2 rule 2 writes it', () => {
    for (const model of MODELS) {
      const graph = graphOf(model);
      expect(graph.nodes.length, model).toBeGreaterThan(0);
      for (const node of graph.nodes) {
        const rebuilt =
          node.indices.length === 0 ? node.site : `${node.site}[${pointLabel(node.indices)}]`;
        expect(rebuilt, `${model}: ${node.id}`).toBe(node.id);
      }
    }
  });

  it('reads the bindings of every segment, so a template inside a composition is read too', () => {
    // §5.2 rule 2's three forms, one identifier each.
    expect(nodeIndices('embed')).toEqual([]);
    expect(nodeIndices('decoder/attn[layer=3]')).toEqual([{ name: 'layer', value: 3n }]);
    expect(nodeIndices('grid/cell[col=2,row=1]')).toEqual([
      { name: 'col', value: 2n },
      { name: 'row', value: 1n },
    ]);
    expect(nodeIndices('text/decoder/attn[layer=0]')).toEqual([{ name: 'layer', value: 0n }]);
    expect(nodeIndices('outer/inner[a=1]/leaf[b=2]')).toEqual([
      { name: 'a', value: 1n },
      { name: 'b', value: 2n },
    ]);
  });

  it('contributes nothing rather than a guess where the bracket holds something else', () => {
    // `node_identifier` admits any text between the brackets; a reading that invented a value
    // there would be deciding what D1 meant.
    expect(nodeIndices('a[layer]')).toEqual([]);
    expect(nodeIndices('a[layer=x]')).toEqual([]);
    expect(nodeIndices('a[=1]')).toEqual([]);
    expect(nodeIndices('a[layer=1.5]')).toEqual([]);
    expect(nodeIndices('a[layer=-1]')).toEqual([{ name: 'layer', value: -1n }]);
  });

  it('names the composition a site sits in, and nothing for a root instance', () => {
    expect(nodeWithin('embed')).toBeNull();
    expect(nodeWithin('decoder/attn[layer=3]')).toBe('decoder');
    expect(nodeWithin('text/decoder/attn[layer=0]')).toBe('text/decoder');
    expect(declaredSite('text/decoder/attn[layer=0]')).toBe('text/decoder/attn');
  });
});

describe('the graph the view draws', () => {
  it('is D1’s nodes and edges, in D1’s own topological order', () => {
    for (const model of MODELS) {
      const graph = graphOf(model);
      const d1 = d1Of(model);
      const nodes = d1['nodes'] as Record<string, PyValue>;
      const order = d1['topological_order'] as unknown as string[];
      expect(graph.nodes.length, model).toBe(Object.keys(nodes).length);
      expect((d1['edges'] as unknown as unknown[]).length, model).toBe(graph.edges.length);
      expect(graph.nodes.map((one) => one.id), model).toEqual(order);
      graph.nodes.forEach((one, at) => {
        expect(one.position, `${model}: ${one.id}`).toBe(at);
      });
    }
  });

  it('counts qwen3.5-4b-text’s 195 nodes and 258 edges — what the e2e reads on the page', () => {
    const graph = graphOf('qwen3.5-4b-text');
    expect(graph.nodes).toHaveLength(195);
    expect(graph.edges).toHaveLength(258);
    // Its one composition index, with every value D1 writes for it.
    expect(graph.indices.map((one) => one.name)).toEqual(['layer']);
    expect(graph.indices[0]?.values).toHaveLength(32);
    expect(graph.indices[0]?.values[0]).toBe(0n);
    expect(graph.indices[0]?.values[31]).toBe(31n);
  });

  it('answers the families and the primitives the filters offer, from D1 and nowhere else', () => {
    const graph = graphOf('llama3-8b');
    const d1 = d1Of('llama3-8b');
    const nodes = Object.values(d1['nodes'] as Record<string, PyValue>) as Record<string, PyValue>[];
    const families = new Set<string>();
    const primitives = new Set<string>();
    for (const node of nodes) {
      for (const family of node['families'] as unknown as string[]) families.add(family);
      primitives.add((node['primitive'] as Record<string, PyValue>)['name'] as unknown as string);
    }
    expect(graph.families).toEqual([...families].sort());
    expect(graph.primitives).toEqual([...primitives].sort());
  });

  it('carries each node’s D1 arguments with the defaults applied, as §4.9 asks', () => {
    const graph = graphOf('llama3-8b');
    const attn = graph.byId.get('decoder/attn[layer=1]');
    expect(attn?.primitive).toBe('attention.dense');
    expect(attn?.version).toBe('1.0.0');
    expect(attn?.families).toEqual(['decoder', 'sequence_operator']);
    expect(attn?.acrossPositions).toBe(true);
    // S12's own reading: `kv_source` is not written in the document and D1 writes the default.
    const written = attn?.arguments as Record<string, PyValue>;
    expect(written['kv_source']).toBe('own');
    expect(written['heads']).toBe(32n);
  });

  it('counts the edges of each node, which is how many D1 draws into and out of it', () => {
    const graph = graphOf('llama3-8b');
    const d1 = d1Of('llama3-8b');
    const edges = d1['edges'] as unknown as { from: { node: string }; to: { node: string } }[];
    for (const node of graph.nodes) {
      expect(node.incoming, node.id).toBe(edges.filter((one) => one.to.node === node.id).length);
      expect(node.outgoing, node.id).toBe(edges.filter((one) => one.from.node === node.id).length);
    }
  });

  it('answers nothing at all for a value that carries no D1', () => {
    expect(emittedGraph(null).nodes).toEqual([]);
    expect(emittedGraph({}).edges).toEqual([]);
    expect(emittedGraph({ d1: {} }).nodes).toEqual([]);
  });
});

describe('a node’s figure', () => {
  it('is D3’s and D4’s own numbers, added once per identity', () => {
    const graph = graphOf('llama3-8b');
    // The same figure S2 draws on the card of one iteration (feature 2.9's own assertion), here
    // attributed to the D1 node rather than to the folded box.
    const attn = graph.byId.get('decoder/attn[layer=1]');
    expect(attn?.figures.bytes).toBe(80n * 1024n * 1024n);
    expect(attn?.figures.bytesPerCachedPosition).toBe(4096n);
    expect(attn?.figures.tensors).toBe(4);
    expect(attn?.figures.states).toBe(1);
  });

  it('counts a tied identity once on each node that carries a member', () => {
    // `qwen3.5-4b-text` ties `embed.weight` and `lm_head.weight`: one identity, 1.18 GiB, and a
    // node says the tensors that instance uses — so each of the two says the same figure.
    const graph = graphOf('qwen3.5-4b-text');
    const derived = derivedOf('qwen3.5-4b-text') as Record<string, PyValue>;
    const tensors = (derived['d3'] as Record<string, PyValue>)['tensors'] as unknown as {
      identity: string;
      members: string[];
      bytes: number | bigint;
      tied?: boolean;
    }[];
    const tied = tensors.find((one) => one.members.length > 1);
    expect(tied, 'the corpus document ties one identity').toBeDefined();
    const bytes = BigInt(tied?.bytes ?? 0);
    for (const member of tied?.members ?? []) {
      const node = graph.byId.get(member.slice(0, member.lastIndexOf('.')));
      expect(node?.figures.bytes, member).toBe(bytes);
      expect(node?.figures.tensors, member).toBe(1);
    }
  });

  it('sums to D3’s own total over the identities, not over the nodes', () => {
    const graph = graphOf('llama3-8b');
    const derived = derivedOf('llama3-8b') as Record<string, PyValue>;
    const d3 = derived['d3'] as Record<string, PyValue>;
    const totals = d3['totals'] as Record<string, PyValue>;
    let overNodes = 0n;
    for (const node of graph.nodes) overNodes += node.figures.bytes ?? 0n;
    // No tie in `llama3-8b`, so the two agree; the tied document above is where they differ.
    expect(overNodes).toBe(BigInt(totals['bytes'] as unknown as number));
  });

  it('counts the rows of each product that name a node — the facts panel’s own line', () => {
    const derived = derivedOf('llama3-8b');
    const graph = graphOf('llama3-8b');
    const facts = nodeProducts(derived, graph, 'decoder/attn[layer=1]');
    expect(facts.tensors).toBe(4);
    expect(facts.states).toBe(1);
    // Erratum E7: `attention.dense` has five partition options, not the board's earlier reading.
    expect(facts.options).toBe(5);
    expect(facts.consumes).toBe(1);
    expect(facts.produces).toBe(1);
  });
});

describe('a graph split', () => {
  it('is read from D6 and never computed — `decoder[layer<=0]` holds the seven nodes it lists', () => {
    for (const model of ['llama3-8b', 'qwen3.5-4b-text'] as const) {
      const derived = derivedOf(model) as Record<string, PyValue>;
      const listed = (derived['d6'] as Record<string, PyValue>)['graph_splits'] as unknown as {
        graph_split: string;
        block: string[];
        sizes: number[];
        crossing_values: number;
      }[];
      const first = listed[0];
      const split = emittedSplit(derived, first?.graph_split ?? '');
      expect(split?.id, model).toBe('decoder[layer<=0]');
      expect(split?.kind, model).toBe('layer');
      expect(split?.block, model).toEqual(first?.block);
      expect(split?.block, model).toHaveLength(7);
      expect(split?.sizes, model).toEqual([7, 188]);
      expect(split?.crossingValues, model).toBe(1);
      // Every node of the block is a node of D1 — which is what makes shading it possible.
      const graph = graphOf(model);
      for (const id of split?.block ?? []) expect(graph.byId.has(id), `${model}: ${id}`).toBe(true);
    }
  });

  it('answers nothing for an identifier D6 does not list', () => {
    expect(emittedSplit(derivedOf('llama3-8b'), 'decoder[layer<=999]')).toBeNull();
    expect(emittedSplit(null, 'anything')).toBeNull();
  });
});
