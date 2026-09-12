/**
 * D1 read as a graph a reader can look at — plan §4.9's expanded view, artboard S12.
 *
 * The types are `Emitted…` and not `Expanded…` because the derivation already has an
 * `ExpandedGraph`: `derive/expand.ts`'s reading of the *analysis*, which is what D2–D6 are
 * computed from. This is a reading of the **document** D1 writes, for a reader — and §4.9's own
 * sentence ("every emitted node") is where the name comes from.
 *
 * > View ▸ Expanded Graph opens a read-only tab over D1: every emitted node
 * > (`decoder/attn[layer=3]`), every edge, in D1's order […]. Filters: index range (`layer 0–3`),
 * > family (`sequence_operator`), primitive; a search by identifier.
 *
 * Everything the view filters on is a fact of a node, and every one of those facts is read here
 * rather than in a component — the component inventory's §7, which forbids a component to take a
 * node identifier apart for itself and puts every figure in the products. So this module answers:
 * what D1's nodes and edges are, in D1's own order; which indices each node's identifier carries
 * (§5.2 rule 2, the inverse of `whereOfSite`); what D3 and D4 attribute to each node; and what a
 * D6 graph split's block holds.
 *
 * **The indices are read off the identifier, and that is the only place they are written.**
 * `whereOfSite` composes `<composition>/<site>[<index>=<value>,…]`; {@link nodeIndices} takes one
 * apart again, over every segment of the path, so an instance of a template inside a composition
 * is read the same way a root composition's site is. `test/describe/expanded.test.ts` holds the
 * two to each other over every node of every corpus D1: the identifier rebuilt from its declared
 * site and its indices is the identifier D1 wrote.
 *
 * **What is counted, and what is not.** A node's byte figure is D3's own numbers added over the
 * identities that name it, once per identity — the same rule `derivedFacts` applies one level up,
 * where a folded box stands for every iteration of a site. Nothing is converted or rounded: the
 * sum is a `bigint` and the interface renders it with the one rendering of a byte count there is.
 */
import type { PyValue } from '../expr/value.js';
import { entries, get, listOf } from '../library/access.js';
import { pyStr } from '../library/repr.js';

import { declaredSite, splitMember } from './figures.js';

// The members of a derived document this reading walks: the derived schema's own, read in the
// core as every other product is (plan §1 keeps them out of the interface's source).
const D1 = 'd1';
const D3 = 'd3';
const D4 = 'd4';
const D6 = 'd6';
const NODES = 'nodes';
const EDGES = 'edges';
const TOPOLOGICAL_ORDER = 'topological_order';
const PRIMITIVE = 'primitive';
const NAME = 'name';
const VERSION = 'version';
const ARGUMENTS = 'arguments';
const FAMILIES = 'families';
const ACROSS_POSITIONS = 'across_positions';
const RULE = 'rule';
const FROM = 'from';
const TO = 'to';
const NODE = 'node';
const PORT = 'port';
const TENSORS = 'tensors';
const STATES = 'states';
const MEMBERS = 'members';
const BYTES = 'bytes';
const PER_POSITION = 'bytes_per_cached_position';
const GRAPH_SPLITS = 'graph_splits';
const GRAPH_SPLIT = 'graph_split';
const PARTITION_OPTIONS = 'partition_options';
const BLOCK = 'block';
const SIZES = 'sizes';
const CROSSING_VALUES = 'crossing_values';
const KIND = 'kind';

/** One index binding an identifier carries: `layer=3`. */
export interface NodeIndex {
  readonly name: string;
  /** The value as §5.2 rule 2 writes it; a whole number in every document the grammar admits. */
  readonly value: bigint;
}

/** What D3 and D4 attribute to one node — the figure a row of the view carries. */
export interface NodeFigures {
  /** D3 bytes over the identities whose members name this node, each identity once. */
  readonly bytes: bigint | null;
  /** D4 bytes per cached position over its state ports, likewise. */
  readonly bytesPerCachedPosition: bigint | null;
  readonly tensors: number;
  readonly states: number;
}

/** Nothing at all: what a node of a document nothing has been derived for carries. */
export const NO_FIGURES: NodeFigures = {
  bytes: null,
  bytesPerCachedPosition: null,
  tensors: 0,
  states: 0,
};

/** One node of D1, as §4.9 shows one. */
export interface EmittedNode {
  /** D1's own identifier, which is the key of its `nodes` map. */
  readonly id: string;
  /** The declared site behind it — `decoder/attn` for `decoder/attn[layer=3]`. */
  readonly site: string;
  /** The composition path the site sits in, or `null` for a root instance. */
  readonly within: string | null;
  readonly primitive: string;
  readonly version: string;
  readonly families: readonly string[];
  readonly indices: readonly NodeIndex[];
  readonly acrossPositions: boolean;
  /** Its place in D1's `topological_order`, from zero; `-1` where the order omits it. */
  readonly position: number;
  /** Its resolved arguments, as D1 writes them — §4.9's "defaults applied". */
  readonly arguments: PyValue;
  readonly figures: NodeFigures;
  /** How many edges of D1 enter it, and how many leave it. */
  readonly incoming: number;
  readonly outgoing: number;
}

/** One edge of D1. */
export interface EmittedEdge {
  /** The rule that emitted it, as D1 names it. */
  readonly rule: string;
  readonly from: string;
  readonly fromPort: string;
  readonly to: string;
  readonly toPort: string;
}

/** One index name D1's identifiers carry, with the values they write for it. */
export interface EmittedIndex {
  readonly name: string;
  /** Every value, ascending and without repetition. */
  readonly values: readonly bigint[];
}

/** D1, read as the view reads it. */
export interface EmittedGraph {
  /** Every node, in D1's `topological_order`; a node the order omits comes after, in D1's order. */
  readonly nodes: readonly EmittedNode[];
  readonly byId: ReadonlyMap<string, EmittedNode>;
  readonly edges: readonly EmittedEdge[];
  /** The index names any identifier carries, in the order they are first written. */
  readonly indices: readonly EmittedIndex[];
  /** Every family any node declares, sorted. */
  readonly families: readonly string[];
  /** Every primitive any node pins, by name, sorted. */
  readonly primitives: readonly string[];
}

/** A graph of nothing at all. */
export const NO_EMITTED_GRAPH: EmittedGraph = {
  nodes: [],
  byId: new Map(),
  edges: [],
  indices: [],
  families: [],
  primitives: [],
};

/**
 * The index bindings an identifier carries, in the order it writes them.
 *
 * The inverse of `whereOfSite` (`validate/graph.ts`), read over **every** segment of the path:
 * §5.2 rule 2 composes a generated instance as `<composition>/<site>[<index>=<value>,…]` and an
 * instance of a template as `<instance>/<identifier in the template>`, so a template instance
 * that is itself a site of a composition carries bindings on two segments. A segment whose
 * bracket holds something other than `<name>=<whole number>` contributes nothing rather than a
 * guess: the derived schema's `node_identifier` admits any text between the brackets, and a
 * reading that invented a value there would be the interface deciding what D1 meant.
 */
export function nodeIndices(identifier: string): NodeIndex[] {
  const found: NodeIndex[] = [];
  for (const segment of identifier.split('/')) {
    const open = segment.indexOf('[');
    if (open < 0 || !segment.endsWith(']')) continue;
    for (const binding of segment.slice(open + 1, -1).split(',')) {
      const at = binding.indexOf('=');
      if (at < 0) continue;
      const name = binding.slice(0, at);
      const written = binding.slice(at + 1);
      if (name === '' || !/^-?\d+$/.test(written)) continue;
      found.push({ name, value: BigInt(written) });
    }
  }
  return found;
}

/**
 * The composition an identifier's site sits in, or `null` for a root instance.
 *
 * The declared site without its last segment: `decoder/attn` sits in `decoder`, and
 * `text/decoder/attn` in `text/decoder` — which is the prefix §4.9's layer preview restricts
 * itself to, and what the drill-in of that composition is over.
 */
export function nodeWithin(identifier: string): string | null {
  const site = declaredSite(identifier);
  const at = site.lastIndexOf('/');
  return at < 0 ? null : site.slice(0, at);
}

/**
 * D1 read as a graph, with what D3 and D4 attribute to each node.
 *
 * The value is a derived document or a `--d1` document — the two carry `d1` the same way, which
 * is what lets the drill-in's presence reading take either (feature 2.14). The figures are only
 * there when the value is a whole derived document; a `--d1` document has no D3 to read.
 */
export function emittedGraph(derived: PyValue): EmittedGraph {
  const graph = get(derived, D1);
  const nodes = get(graph, NODES);
  if (nodes === null || typeof nodes !== 'object' || Array.isArray(nodes)) return NO_EMITTED_GRAPH;

  const edges = edgesOf(graph);
  const incoming = new Map<string, number>();
  const outgoing = new Map<string, number>();
  for (const edge of edges) {
    incoming.set(edge.to, (incoming.get(edge.to) ?? 0) + 1);
    outgoing.set(edge.from, (outgoing.get(edge.from) ?? 0) + 1);
  }

  const figures = nodeFigures(derived);
  const order = new Map<string, number>();
  listOf(get(graph, TOPOLOGICAL_ORDER) ?? []).forEach((one, at) => {
    order.set(pyStr(one), at);
  });

  const indexNames: string[] = [];
  const indexValues = new Map<string, Set<bigint>>();
  const families = new Set<string>();
  const primitives = new Set<string>();

  const read: EmittedNode[] = [];
  for (const [id, node] of entries(nodes)) {
    const indices = nodeIndices(id);
    for (const index of indices) {
      let values = indexValues.get(index.name);
      if (values === undefined) {
        values = new Set<bigint>();
        indexValues.set(index.name, values);
        indexNames.push(index.name);
      }
      values.add(index.value);
    }
    const primitive = get(node, PRIMITIVE);
    const name = pyStr(get(primitive, NAME) ?? '');
    primitives.add(name);
    const listed = listOf(get(node, FAMILIES) ?? []).map((one) => pyStr(one));
    for (const family of listed) families.add(family);
    read.push({
      id,
      site: declaredSite(id),
      within: nodeWithin(id),
      primitive: name,
      version: pyStr(get(primitive, VERSION) ?? ''),
      families: listed,
      indices,
      acrossPositions: get(node, ACROSS_POSITIONS) === true,
      position: order.get(id) ?? -1,
      arguments: get(node, ARGUMENTS),
      figures: figures.get(id) ?? NO_FIGURES,
      incoming: incoming.get(id) ?? 0,
      outgoing: outgoing.get(id) ?? 0,
    });
  }

  // D1's own order, which §4.9 asks for: `topological_order` first, and a node the order does not
  // carry after it in the order `nodes` writes it. Every corpus document orders every node, so
  // the tail is empty there; a derived document that did not would still draw every node it has.
  read.sort((one, other) => {
    if (one.position >= 0 && other.position >= 0) return one.position - other.position;
    if (one.position >= 0) return -1;
    if (other.position >= 0) return 1;
    return 0;
  });

  return {
    nodes: read,
    byId: new Map(read.map((one) => [one.id, one])),
    edges,
    indices: indexNames.map((name) => ({
      name,
      values: [...(indexValues.get(name) ?? [])].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0)),
    })),
    families: [...families].sort(),
    primitives: [...primitives].sort(),
  };
}

/** D1's edges, in the order the product lists them. */
function edgesOf(graph: PyValue): EmittedEdge[] {
  return listOf(get(graph, EDGES) ?? []).map((edge) => {
    const from = get(edge, FROM);
    const to = get(edge, TO);
    return {
      rule: pyStr(get(edge, RULE) ?? ''),
      from: pyStr(get(from, NODE) ?? ''),
      fromPort: pyStr(get(from, PORT) ?? ''),
      to: pyStr(get(to, NODE) ?? ''),
      toPort: pyStr(get(to, PORT) ?? ''),
    };
  });
}

/**
 * What D3 and D4 attribute to each node, in one walk.
 *
 * The rule is `derivedFacts`', one level down: an identity contributes to a node once, however
 * many of its members that node carries, because a tied tensor is one tensor. Asking `siteDerived`
 * per node would be the same answer at the cost of one walk of D3 per node — 382 × 1 771 rows on
 * `deepseek-v4-pro` — and the view reads every node at once.
 */
export function nodeFigures(derived: PyValue): Map<string, NodeFigures> {
  const running = new Map<string, { bytes: bigint | null; per: bigint | null; tensors: number; states: number }>();
  const at = (node: string): { bytes: bigint | null; per: bigint | null; tensors: number; states: number } => {
    let found = running.get(node);
    if (found === undefined) {
      found = { bytes: null, per: null, tensors: 0, states: 0 };
      running.set(node, found);
    }
    return found;
  };

  const walk = (rows: readonly PyValue[], figure: string, state: boolean): void => {
    for (const row of rows) {
      const amount = integer(get(row, figure) ?? null);
      const nodes = new Set<string>();
      for (const member of listOf(get(row, MEMBERS) ?? [])) {
        const split = splitMember(pyStr(member));
        if (split !== null) nodes.add(split.site);
      }
      for (const node of nodes) {
        const one = at(node);
        if (state) {
          one.states += 1;
          if (amount !== null) one.per = (one.per ?? 0n) + amount;
        } else {
          one.tensors += 1;
          if (amount !== null) one.bytes = (one.bytes ?? 0n) + amount;
        }
      }
    }
  };

  walk(listOf(get(get(derived, D3) ?? null, TENSORS) ?? []), BYTES, false);
  walk(listOf(get(get(derived, D4) ?? null, STATES) ?? []), PER_POSITION, true);

  const figures = new Map<string, NodeFigures>();
  for (const [node, one] of running) {
    figures.set(node, {
      bytes: one.bytes,
      bytesPerCachedPosition: one.per,
      tensors: one.tensors,
      states: one.states,
    });
  }
  return figures;
}

/** One D6 graph split, as §4.18's "Show split on canvas" shades it. */
export interface EmittedSplit {
  /** The split's own identifier — `decoder[layer<=0]`. */
  readonly id: string;
  readonly kind: string;
  /** The first block's nodes, in D1 order, as D6 lists them. */
  readonly block: readonly string[];
  /** How many nodes each side holds, as D6 counts them. */
  readonly sizes: readonly number[];
  /** How many values cross it, as D6 counts them. */
  readonly crossingValues: number | null;
}

/**
 * One graph split of D6, by the identifier the reader chose, or `null` where D6 has no such one.
 *
 * The block is read and never computed: D6 states it ("the ancestor closure of the split's layer
 * prefix or family", the schema's own description), the derivation is what knows how, and the
 * view shades what it is given.
 */
export function emittedSplit(derived: PyValue, id: string): EmittedSplit | null {
  for (const row of listOf(get(get(derived, D6) ?? null, GRAPH_SPLITS) ?? [])) {
    if (pyStr(get(row, GRAPH_SPLIT) ?? '') !== id) continue;
    const crossing = integer(get(row, CROSSING_VALUES) ?? null);
    return {
      id,
      kind: pyStr(get(row, KIND) ?? ''),
      block: listOf(get(row, BLOCK) ?? []).map((one) => pyStr(one)),
      sizes: listOf(get(row, SIZES) ?? []).map((one) => Number(one)),
      crossingValues: crossing === null ? null : Number(crossing),
    };
  }
  return null;
}

/** How many rows of each product name one node — the counts §4.9's node facts carry. */
export interface NodeProducts {
  /** D2 values this node produces, and values it consumes. */
  readonly produces: number;
  readonly consumes: number;
  readonly tensors: number;
  readonly states: number;
  /** D6 partition options recorded for it. */
  readonly options: number;
}

/**
 * How many rows of D2, D3, D4 and D6 name one node.
 *
 * Counts and not rows: §4.9 sends the *rows* to the Derived panel ("every product's rows for it in
 * the Derived panel"), which feature 2.15 built and holds to the node's declared site. What the
 * view's own facts panel says is how many there are, so a reader knows what is waiting there.
 */
export function nodeProducts(derived: PyValue, graph: EmittedGraph, id: string): NodeProducts {
  const figures = graph.byId.get(id)?.figures ?? NO_FIGURES;
  let produces = 0;
  let consumes = 0;
  for (const edge of graph.edges) {
    if (edge.from === id) produces += 1;
    if (edge.to === id) consumes += 1;
  }
  let options = 0;
  for (const row of listOf(get(get(derived, D6) ?? null, PARTITION_OPTIONS) ?? [])) {
    if (pyStr(get(row, NODE) ?? '') === id) options += 1;
  }
  return { produces, consumes, tensors: figures.tensors, states: figures.states, options };
}

/** A figure read off a product: an integer, or `null` where the product left it open. */
function integer(value: PyValue): bigint | null {
  if (typeof value === 'bigint') return value;
  if (typeof value === 'number' && Number.isInteger(value)) return BigInt(value);
  return null;
}
