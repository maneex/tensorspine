/**
 * D2, the Derived Value Shapes and Lifetimes: the port of `derive.d2` (§7, derived guide §4).
 *
 * "The value and shape inventory; the payload of every valid graph split — the values live at it,
 * sized per invocation; the peak of live values along one order of the graph, the activation peak
 * of an invocation; and the fragment alignment of every fragmented stream (§5.3)."
 *
 * Four things are computed here, and the first is a walk of its own:
 *
 * **The counts** ({@link counts}). An element count is never a number — "how many tokens a request
 * has is deployment intent (§10.3)" — so a value's `count` is a *combination of the public inputs'
 * counts*: `{"tokens": 1.0}` is one element per element of the `tokens` input, `{"pixels": 0.0625}`
 * one per sixteen pixels after a `merge` by 16, and `{"tokens": 1.0, "pixels": 0.25}` a token
 * stream a `splice` inserted the merged patches into. They are seeded by the public inputs that
 * introduce a stream and propagated along the validator's order: a `merge` divides by its factor,
 * an `insert` adds the inserted stream's to the instance's own, an `align` leaves the instance's
 * own untouched.
 *
 * **The joining input** (§5.3, V19). A public input that joins an existing stream introduces none,
 * so it has no count to seed: it "carries the count the stream has at that kind — the one count of
 * the values in that domain that do not descend from the input". That is a second pass, after the
 * introducing inputs' counts have reached the values the join meets, and a stream carrying no
 * count or several at that kind **refuses the join** rather than guessing one — an exception out
 * of the derivation and not a problem row, because V19 already admits only a kind the stream
 * carries independently of the input, and the document was valid before any product was computed.
 *
 * **The structural graph splits** ({@link structuralGraphSplits}). "A graph split is valid by
 * construction: the ancestor closure of a set of nodes is downward closed, so every crossing edge
 * points out of it." This emitter chooses the ancestor closure of every layer prefix of a
 * single-index composition and of every family — "the specification defines legality, not an
 * exhaustive graph split family" — and a payload is the *distinct* values crossing, "counted once
 * per graph split however many consumers it has across it".
 *
 * **The peak of live values** ({@link peakLive}). At each node of the order, the values produced or
 * delivered and not yet consumed by every consumer — its own outputs included, its inputs still
 * held while it runs, a value a public output exposes live to the end. "The peak is a property of
 * this one order (another valid order may peak lower), stated as such."
 *
 * The arithmetic is Python's throughout (feature 1.2): a count is a `float`, being born as `1.0`
 * and divided from there; a byte size is an `int` unless a sub-byte dtype or a count made it a
 * `float`; and `sum` starts at the integer `0`, so the order of the terms is part of the answer
 * and every total below is taken in the tools' own order.
 */
import { pyAdd, pyDivide, pyEqual, pyMultiply, pyOrder } from '../expr/arithmetic.js';
import { PyKeyError, PyValueError } from '../expr/errors.js';
import { primitiveValue } from '../expr/primitive.js';
import { truthy, type PyRecord, type PyValue } from '../expr/value.js';
import { formatNumber } from '../json/number.js';
import { put } from '../json/tree.js';
import { demand, entries, has, listOf, optional } from '../library/access.js';
import type { Library } from '../library/load.js';
import { pyRepr, pyStr } from '../library/repr.js';
import { comparePythonStrings } from '../schema/repr.js';
import { fragmentedStreams, present, type Indexing } from '../validate/index.js';
import {
  defaultDtype,
  elementsOf,
  numberOf,
  orZero,
  productShape,
  pySum,
  widthOf,
  type Figure,
} from './figures.js';
import {
  expandedKeyOf,
  expandedPortKeyOf,
  identOf,
  type ExpandedEdge,
  type ExpandedGraph,
  type ExpandedMeta,
  type ExpandedNode,
  type ExpandedPort,
  type ExpandedSite,
} from './expand.js';

/**
 * The count of one value: how many of it there are per element of each public input.
 *
 * A record and not a map, because it is written into the product as it stands and its member order
 * is the order the propagation built it in — which is part of the derived document's bytes.
 */
export type Count = PyRecord;

/** One entry of the count map: the `(site, port)` the tools key it by, and the count it carries. */
interface PortCount {
  readonly site: ExpandedSite;
  readonly port: PyValue;
  count: Count;
}

/** `_value_id(key, port)`: `node.port` as D1 names it, under the prefixes above it (§5.2 rule 2). */
export function valueId(site: ExpandedSite, port: PyValue): string {
  return `${identOf(site)}.${pyStr(port)}`;
}

/** `resolved[node]`, and the `KeyError` a node the expansion has not raises. */
function nodeOf(graph: ExpandedGraph, site: ExpandedSite): ExpandedNode {
  const node = graph.resolved.get(expandedKeyOf(site));
  if (node === undefined) throw new PyKeyError(pyRepr(identOf(site)));
  return node;
}

/** `meta[key]`, which `_expand` fills for every node it resolves. */
function metaOf(graph: ExpandedGraph, at: string, site: ExpandedSite): ExpandedMeta {
  const meta = graph.meta.get(at);
  if (meta === undefined) throw new PyKeyError(pyRepr(identOf(site)));
  return meta;
}

/** `graph['inputs_at'][name]`: the instance ports a public input feeds, or Python's `KeyError`. */
function inputsAt(graph: ExpandedGraph, name: string): readonly ExpandedPort[] {
  const at = graph.inputsAt.get(name);
  if (at === undefined) throw new PyKeyError(pyRepr(name));
  return at;
}

/** `domains.get((key, port))`: the indexing domain of one port of the expanded graph. */
function domainAt(graph: ExpandedGraph, site: ExpandedSite, port: PyValue): Indexing | null {
  return graph.domains.get(expandedPortKeyOf(site, port))?.domain ?? null;
}

/** `graph['model']['interfaces'][side]`, the *calling* document's at every level. */
function interfacesOf(graph: ExpandedGraph, side: 'inputs' | 'outputs'): [string, PyValue][] {
  return entries(demand(demand(graph.model, 'interfaces'), side));
}

/** `definition['ports'][side][name]`: one declared port, or Python's `KeyError`. */
function portOf(node: ExpandedNode, side: 'inputs' | 'outputs', name: PyValue): PyValue {
  return demand(demand(demand(node.definition, 'ports'), side), pyStr(name));
}

/**
 * `_counts(graph)`: the element count of every port, as a combination of the inputs' counts.
 *
 * "Seeded by the public inputs, merges divide by their factor, inserts add." The map is keyed by
 * `(site, port)` as the tools key it, and both halves of the key travel with the value because the
 * joining pass and the stream listing read the site back out of it — feature 1.6b's rule for every
 * dictionary the tools key by a tuple.
 */
export function counts(graph: ExpandedGraph): Map<string, PortCount> {
  const incoming = new Map<string, ExpandedEdge[]>();
  const downstream = new Map<string, Set<string>>();
  for (const edge of graph.edges) {
    const at = expandedKeyOf(edge.to);
    const held = incoming.get(at);
    if (held === undefined) incoming.set(at, [edge]);
    else held.push(edge);
    const from = expandedKeyOf(edge.from);
    const seen = downstream.get(from);
    if (seen === undefined) downstream.set(from, new Set([at]));
    else seen.add(at);
  }

  const found = new Map<string, PortCount>();
  const record = (site: ExpandedSite, port: PyValue, count: Count): void => {
    const key = expandedPortKeyOf(site, port);
    const held = found.get(key);
    // A Python dictionary keeps a re-assigned key where it was, and a `Map` does the same — which
    // is what leaves a joining input's seed at the place its endpoint's count already had.
    if (held === undefined) found.set(key, { site, port, count });
    else held.count = count;
  };
  const countAt = (site: ExpandedSite, port: PyValue): Count | undefined =>
    found.get(expandedPortKeyOf(site, port))?.count;

  const propagate = (nodes: ReadonlySet<string>): void => {
    for (const site of graph.order) {
      const at = expandedKeyOf(site);
      if (!nodes.has(at)) continue;
      const node = nodeOf(graph, site);
      const transforms = listOf(optional(node.definition, 'domain_transforms', []));
      const sources = new Set(transforms.map((transform) => pyStr(demand(transform, 'from_port'))));
      for (const edge of incoming.get(at) ?? []) {
        const delivered = countAt(edge.from, edge.fromPort);
        if (delivered !== undefined) record(site, edge.toPort, delivered);
      }
      // The instance's own count: the first input port no transform takes its elements from and
      // that something has delivered to — declaration order, as the tools break the loop.
      let own: Count | undefined;
      for (const [name] of entries(demand(demand(node.definition, 'ports'), 'inputs'))) {
        if (sources.has(name)) continue;
        const delivered = countAt(site, name);
        if (delivered !== undefined) {
          own = delivered;
          break;
        }
      }
      for (const [name] of entries(demand(demand(node.definition, 'ports'), 'outputs'))) {
        let count = own;
        for (const transform of transforms) {
          if (!pyEqual(demand(transform, 'to_port'), name)) continue;
          const source = countAt(site, demand(transform, 'from_port'));
          const relation = demand(transform, 'relation');
          if (relation === 'merge' && source !== undefined) {
            // `f = _num(primitive_value(t['factor'], args)) or 1`: a factor that does not resolve
            // to a number, or resolves to zero, divides by one rather than refusing.
            const resolved = numberOf(primitiveValue(demand(transform, 'factor'), node.args));
            const factor = resolved === null || !truthy(resolved) ? 1n : resolved;
            const divided: Count = {};
            for (const [stream, value] of entries(source)) put(divided, stream, pyDivide(value, factor));
            count = divided;
          } else if (relation === 'insert' && source !== undefined) {
            // "The input port's elements enter the instance's stream, which the output keeps":
            // the instance's own count, plus the inserted stream's, stream by stream.
            const inserted: Count = {};
            for (const [stream, value] of entries(own ?? {})) put(inserted, stream, value);
            for (const [stream, value] of entries(source)) {
              const held = has(inserted, stream) ? (inserted[stream] as PyValue) : 0n;
              put(inserted, stream, pyAdd(held, value));
            }
            count = inserted;
          }
        }
        if (count !== undefined) record(site, name, count);
      }
    }
  };

  const joining: [string, PyValue][] = [];
  for (const [name, declared] of interfacesOf(graph, 'inputs')) {
    if (has(declared, 'stream')) {
      joining.push([name, declared]);
      continue;
    }
    for (const endpoint of inputsAt(graph, name)) {
      // A dictionary of its own per endpoint, as the tools write a fresh literal each time.
      const seed: Count = {};
      put(seed, name, 1);
      record(endpoint.site, endpoint.port, seed);
    }
  }
  propagate(new Set(graph.order.map((site) => expandedKeyOf(site))));

  for (const [name, declared] of joining) {
    const endpoints = inputsAt(graph, name);
    const descends = new Set<string>();
    const queue = endpoints.map((endpoint) => expandedKeyOf(endpoint.site));
    while (queue.length > 0) {
      const at = queue.pop() as string;
      if (descends.has(at)) continue;
      descends.add(at);
      queue.push(...(downstream.get(at) ?? []));
    }
    const others = new Set<string>();
    for (const [other] of interfacesOf(graph, 'inputs')) {
      if (other === name) continue;
      for (const endpoint of inputsAt(graph, other)) {
        others.add(expandedPortKeyOf(endpoint.site, endpoint.port));
      }
    }
    const kind = demand(declared, 'kind');
    const stream = demand(declared, 'stream');
    // "Another input's delivery, or a value the join does not reach", in the joined domain — kept
    // by the text `json.dumps(c, sort_keys=True)` writes, so that two equal counts are one reading.
    const carried = new Map<string, Count>();
    for (const [, entry] of found) {
      const reached = descends.has(expandedKeyOf(entry.site));
      const delivered = others.has(expandedPortKeyOf(entry.site, entry.port));
      if (reached && !delivered) continue;
      const domain = domainAt(graph, entry.site, entry.port);
      if (domain === null) continue;
      if (!pyEqual(domain[0], kind) || !pyEqual(domain[1], stream)) continue;
      const written = sortedJson(entry.count);
      if (!carried.has(written)) carried.set(written, entry.count);
    }
    if (carried.size !== 1) {
      const held =
        carried.size === 0 ? 'no count' : `${carried.size} counts ${pyRepr([...carried.values()])}`;
      throw new PyValueError(
        `input ${name}: joins stream '${pyStr(stream)}' at kind ${pyStr(kind)}, where the stream ` +
          `carries ${held} independently of it; a joining input takes the stream's one count at ` +
          `its kind (§5.3)`,
      );
    }
    const seed = [...carried.values()][0] as Count;
    for (const endpoint of endpoints) {
      const copy: Count = {};
      for (const [stream_, value] of entries(seed)) put(copy, stream_, value);
      record(endpoint.site, endpoint.port, copy);
    }
    propagate(descends);
  }
  return found;
}

/**
 * `json.dumps(c, sort_keys=True)`: the text two counts are one reading by.
 *
 * Only its injectivity is used — the tools key a dictionary of the counts they *found* by it — so
 * the one thing that matters is that it separates exactly what Python separates: the names sorted
 * by code point, and each value written as Python writes a number, `1.0` apart from `1`. A stream
 * name is an identifier, so `ensure_ascii`'s escaping cannot arise.
 */
function sortedJson(count: Count): string {
  const members = Object.keys(count)
    .sort(comparePythonStrings)
    .map((name) => {
      const value = count[name] as PyValue;
      const written =
        typeof value === 'bigint' ? value.toString() : formatNumber(value as number, true);
      return `${JSON.stringify(name)}: ${written}`;
    });
  return `{${members.join(', ')}}`;
}

/** `_ancestors(nodes, edges)`: those nodes, and everything that reaches them. */
export function ancestors(nodes: ReadonlySet<string>, edges: readonly ExpandedEdge[]): Set<string> {
  return closureOf(reverseOf(edges), nodes);
}

/** `up`: the sources of every destination, built once where the tools build it per closure. */
function reverseOf(edges: readonly ExpandedEdge[]): Map<string, Set<string>> {
  const up = new Map<string, Set<string>>();
  for (const edge of edges) {
    const at = expandedKeyOf(edge.to);
    const from = expandedKeyOf(edge.from);
    const seen = up.get(at);
    if (seen === undefined) up.set(at, new Set([from]));
    else seen.add(from);
  }
  return up;
}

/** The closure itself: a breadth-first walk up, which is the tools' `deque` walk. */
function closureOf(up: ReadonlyMap<string, Set<string>>, nodes: ReadonlySet<string>): Set<string> {
  const seen = new Set(nodes);
  const queue = [...nodes];
  for (let read = 0; read < queue.length; read += 1) {
    for (const source of up.get(queue[read] as string) ?? []) {
      if (seen.has(source)) continue;
      seen.add(source);
      queue.push(source);
    }
  }
  return seen;
}

/** One structural graph split: its name, its kind, and the block on the near side. */
export interface StructuralSplit {
  readonly name: string;
  readonly kind: string;
  readonly block: ReadonlySet<string>;
}

/**
 * `_structural_graph_splits(graph)`: "valid graph_splits by construction".
 *
 * Two families of them, in the order the tools emit them: the ancestor closure of every layer
 * prefix of every single-index composition, in the expansion's composition order, then the
 * ancestor closure of every family, by name — the latter only where it leaves something on the
 * other side, which is what stops a family covering the whole graph from being a split of it.
 */
export function structuralGraphSplits(graph: ExpandedGraph): StructuralSplit[] {
  const splits: StructuralSplit[] = [];
  // `_ancestors` rebuilds this map per graph split; it is a function of the edges alone, so it is
  // built once here — the same closures, and one walk of the edges instead of one per split.
  const up = reverseOf(graph.edges);
  // composition -> node -> its indices, for the sites of a composition with exactly one index.
  const layers = new Map<string, Map<string, ReadonlyMap<string, PyValue>>>();
  for (const [at, node] of graph.resolved) {
    const composition = metaOf(graph, at, node.site).composition;
    if (composition === null || composition.indices.size !== 1) continue;
    const held = layers.get(composition.name);
    if (held === undefined) layers.set(composition.name, new Map([[at, composition.indices]]));
    else held.set(at, composition.indices);
  }
  for (const name of graph.compositions) {
    const sites = layers.get(name);
    if (sites === undefined) continue;
    // `next(iter(next(iter(layers[comp].values()))))`: the index of the first site listed, which
    // is that composition's only one.
    const first = [...sites.values()][0] as ReadonlyMap<string, PyValue>;
    const index = [...first.keys()][0] as string;
    const values = sortedValues([...sites.values()].map((indices) => indices.get(index) as PyValue));
    for (const value of values.slice(0, -1)) {
      const block = new Set<string>();
      for (const [at, indices] of sites) {
        if (comparePyValues(indices.get(index) as PyValue, value) <= 0) block.add(at);
      }
      splits.push({
        name: `${name}[${index}<=${pyStr(value)}]`,
        kind: 'layer',
        block: closureOf(up, block),
      });
    }
  }
  const families = new Map<string, Set<string>>();
  for (const [at, node] of graph.resolved) {
    for (const family of metaOf(graph, at, node.site).families) {
      const held = families.get(family);
      if (held === undefined) families.set(family, new Set([at]));
      else held.add(at);
    }
  }
  for (const family of [...families.keys()].sort(comparePythonStrings)) {
    const block = closureOf(up, families.get(family) as Set<string>);
    if (block.size < graph.resolved.size) {
      splits.push({ name: `family:${family}`, kind: 'family', block });
    }
  }
  return splits;
}

/** `sorted({…})`: the distinct index values of a composition, in Python's order. */
function sortedValues(values: readonly PyValue[]): PyValue[] {
  const distinct: PyValue[] = [];
  for (const value of values) {
    if (!distinct.some((held) => pyEqual(held, value))) distinct.push(value);
  }
  return distinct.sort(comparePyValues);
}

/** Python's own order over two index values: `sorted` reports `<`, and raises where Python does. */
function comparePyValues(left: PyValue, right: PyValue): number {
  return pyOrder(left, right) ?? 0;
}

/** `d2(graph, cat)`: the values, streams, graph splits and peak of one expanded graph. */
export function d2(graph: ExpandedGraph, library: Library): PyRecord {
  const found = counts(graph);
  const values = new Map<string, PyRecord>();

  for (const edge of graph.edges) {
    const id = valueId(edge.from, edge.fromPort);
    if (!values.has(id)) {
      const node = nodeOf(graph, edge.from);
      const port = portOf(node, 'outputs', edge.fromPort);
      values.set(id, produced(graph, library, id, node, port, edge.from, edge.fromPort, found));
    }
    ((values.get(id) as PyRecord)['to'] as PyValue[]).push(valueId(edge.to, edge.toPort));
  }

  const streams: PyRecord = {};
  for (const [, entry] of found) {
    const domain = domainAt(graph, entry.site, entry.port);
    if (domain === null) continue;
    const name = pyStr(domain[1]);
    if (has(streams, name)) continue;
    const stream: PyRecord = {};
    put(stream, 'kind', domain[0]);
    put(stream, 'count', entry.count);
    put(streams, name, stream);
  }

  // "A public input is an edge like any other (§5.3): the value it delivers, named by the input,
  // with the shape of the port it feeds (V4 makes every fed port agree)."
  for (const [name, declared] of interfacesOf(graph, 'inputs')) {
    const endpoints = inputsAt(graph, name);
    const first = endpoints[0];
    if (first === undefined) continue;
    const node = nodeOf(graph, first.site);
    const port = portOf(node, 'inputs', first.port);
    // `_elements(port['shape'], args) if 'shape' in port else 1` — one, where a produced value
    // takes zero: a port without a shape still delivers one element of the stream.
    const count = has(port, 'shape') ? elementsOf(demand(port, 'shape'), node.args, null) : 1n;
    const role = demand(port, 'role');
    const dtype = defaultDtype(library, role);
    const stream = optional(declared, 'stream', name);
    const entry: PyRecord = {};
    put(entry, 'value', name);
    put(entry, 'input', name);
    put(entry, 'to', endpoints.map((endpoint) => valueId(endpoint.site, endpoint.port)));
    put(entry, 'shape', has(port, 'shape') ? productShape(demand(port, 'shape'), node.args) : []);
    put(entry, 'role', role);
    put(entry, 'dtype', dtype);
    put(entry, 'elements', count);
    // Unguarded, as the tools write it: a port shape that does not resolve raises here rather than
    // writing the blank a produced value's conditional writes. Stated in `test/derive/d2.test.ts`.
    put(entry, 'bytes_per_element', pyMultiply(count, widthOf(dtype)));
    const domain: PyRecord = {};
    put(domain, 'kind', demand(declared, 'kind'));
    put(domain, 'stream', stream);
    put(entry, 'domain', domain);
    const delivered = found.get(expandedPortKeyOf(first.site, first.port))?.count;
    if (delivered !== undefined) put(entry, 'count', delivered);
    else {
      // `counts.get((key, pname), {stream: 1.0})`: a joining input's own, at its stream's name.
      const own: Count = {};
      put(own, pyStr(stream), 1);
      put(entry, 'count', own);
    }
    values.set(name, entry);
  }

  // "Required (§7): an input is required for an output when the output is not evaluated without
  // it — evaluated meaning every input port fed, an insert transform's source excepted."
  const outputs = [...graph.outputsAt];
  const inputs = interfacesOf(graph, 'inputs').map(([name]) => name);
  for (const [id, entry] of [...values]) {
    if (!has(entry, 'input')) continue;
    const without = evaluated(graph, new Set(inputs.filter((name) => name !== id)));
    const needed = outputs
      .filter(([, at]) => !without.has(expandedKeyOf(at.site)))
      .map(([name]) => name);
    put(entry, 'required_for', needed);
    put(entry, 'required', needed.length > 0);
  }

  // "A public output exposes a value whether or not an edge consumes it."
  for (const [name, at] of outputs) {
    const id = valueId(at.site, at.port);
    if (!values.has(id) && graph.resolved.has(expandedKeyOf(at.site))) {
      const node = nodeOf(graph, at.site);
      const port = portOf(node, 'outputs', at.port);
      values.set(id, produced(graph, library, id, node, port, at.site, at.port, found));
    }
    const entry = values.get(id);
    if (entry === undefined) continue;
    if (!has(entry, 'exposed')) put(entry, 'exposed', []);
    (entry['exposed'] as PyValue[]).push(name);
  }

  // "The fragment alignment of a fragmented stream (§5.3): every fragment delivers a multiple of
  // the cumulative merge factors of the values on it, so that every merge sees whole groups."
  for (const name of fragmentedStreams(graph.model)) {
    if (!has(streams, name)) continue;
    let alignment = 1n;
    for (const [, entry] of values) {
      const domain = optional(entry, 'domain', null);
      const count = optional(entry, 'count', null);
      if (domain === null || count === null) continue;
      if (!pyEqual(demand(domain, 'stream'), name)) continue;
      const on = has(count, name) ? demand(count, name) : null;
      if (on === null || !truthy(on)) continue;
      const groups = pyRound(1 / Number(on));
      alignment = lcm(alignment, groups > 1n ? groups : 1n);
    }
    put(streams[name] as PyRecord, 'fragment_alignment', alignment);
  }

  // The crossing walk asks the same two questions of every edge once per graph split, so the keys
  // and the value identifier each edge carries are taken here rather than inside the loop.
  const crossings = graph.edges.map((edge) => ({
    from: expandedKeyOf(edge.from),
    to: expandedKeyOf(edge.to),
    id: valueId(edge.from, edge.fromPort),
  }));
  const splits: PyValue[] = [];
  for (const split of structuralGraphSplits(graph)) {
    const crossing = new Map<string, PyRecord>();
    for (const edge of crossings) {
      if (!split.block.has(edge.from) || split.block.has(edge.to)) continue;
      if (!crossing.has(edge.id)) crossing.set(edge.id, values.get(edge.id) as PyRecord);
    }
    const perInvocation: PyRecord = {};
    for (const [, value] of crossing) weigh(perInvocation, value);
    const payload = [...crossing.keys()].sort(comparePythonStrings).map((id) => {
      const value = crossing.get(id) as PyRecord;
      const entry: PyRecord = {};
      put(entry, 'value', id);
      put(entry, 'bytes_per_element', value['bytes_per_element'] as PyValue);
      put(entry, 'count', optional(value, 'count', null));
      return entry;
    });
    const written: PyRecord = {};
    put(written, 'graph_split', split.name);
    put(written, 'kind', split.kind);
    put(written, 'sizes', [BigInt(split.block.size), BigInt(graph.resolved.size - split.block.size)]);
    put(written, 'payload', payload);
    put(
      written,
      'bytes_per_element',
      pySum([...crossing.values()].map((value) => orZero(value['bytes_per_element'] as Figure))),
    );
    put(written, 'bytes_per_invocation', perInvocation);
    splits.push(written);
  }

  const product: PyRecord = {};
  put(product, 'streams', streams);
  put(product, 'values', [...values.values()]);
  put(product, 'graph_splits', splits);
  put(product, 'peak_live', peakLive(graph, values));
  return product;
}

/**
 * `per_invocation[inp] += (v['bytes_per_element'] or 0) * mult`: one value's weight, per input.
 *
 * A `Counter` starts a key at the integer zero and keeps it where it was first seen, so the member
 * order of `bytes_per_invocation` is the order the values were walked in and its arithmetic is
 * Python's — a blank byte size weighs the integer zero, and a count that is a float makes the
 * total one from that term on.
 */
function weigh(per: PyRecord, value: PyRecord): void {
  const bytes = orZero(value['bytes_per_element'] as Figure);
  for (const [name, multiplier] of entries(optional(value, 'count', null) ?? {})) {
    put(per, name, pyAdd(has(per, name) ? (per[name] as PyValue) : 0n, pyMultiply(bytes, multiplier)));
  }
}

/** One value an output port produces: everything but its consumers, which the edges append. */
function produced(
  graph: ExpandedGraph,
  library: Library,
  id: string,
  node: ExpandedNode,
  port: PyValue,
  site: ExpandedSite,
  name: PyValue,
  found: ReadonlyMap<string, PortCount>,
): PyRecord {
  // `_elements(port['shape'], args) if 'shape' in port else 0`: a port without a shape produces a
  // value of no elements, where a public input's delivers one.
  const count = has(port, 'shape') ? elementsOf(demand(port, 'shape'), node.args, null) : 0n;
  const role = demand(port, 'role');
  const dtype = defaultDtype(library, role);
  const entry: PyRecord = {};
  put(entry, 'value', id);
  put(entry, 'to', []);
  put(entry, 'shape', has(port, 'shape') ? productShape(demand(port, 'shape'), node.args) : []);
  put(entry, 'role', role);
  put(entry, 'dtype', dtype);
  put(entry, 'elements', count);
  put(entry, 'bytes_per_element', count === null ? null : pyMultiply(count, widthOf(dtype)));
  put(entry, 'domain', domainValue(domainAt(graph, site, name)));
  put(entry, 'count', found.get(expandedPortKeyOf(site, name))?.count ?? null);
  return entry;
}

/** `{"kind": dom[0], "stream": dom[1]} if dom else None`. */
function domainValue(domain: Indexing | null): PyValue {
  if (domain === null) return null;
  const written: PyRecord = {};
  put(written, 'kind', domain[0]);
  put(written, 'stream', domain[1]);
  return written;
}

/**
 * `evaluated(delivered)`: the nodes evaluated when only those public inputs deliver.
 *
 * "Evaluated meaning every input port fed, an insert transform's source excepted" (§7) — and a
 * port absent under its `present_when` is no obligation either. One forward pass along the order:
 * a node whose ports are all fed feeds its own consumers.
 */
function evaluated(graph: ExpandedGraph, delivered: ReadonlySet<string>): Set<string> {
  const fed = new Set<string>();
  for (const [name] of interfacesOf(graph, 'inputs')) {
    if (!delivered.has(name)) continue;
    for (const endpoint of inputsAt(graph, name)) {
      fed.add(expandedPortKeyOf(endpoint.site, endpoint.port));
    }
  }
  // The tools walk every edge of the graph inside the loop; the edges are indexed by their source
  // here, which makes the same set of assignments and nothing else.
  const outgoing = new Map<string, ExpandedEdge[]>();
  for (const edge of graph.edges) {
    const at = expandedKeyOf(edge.from);
    const held = outgoing.get(at);
    if (held === undefined) outgoing.set(at, [edge]);
    else held.push(edge);
  }
  const done = new Set<string>();
  for (const site of graph.order) {
    const at = expandedKeyOf(site);
    const node = nodeOf(graph, site);
    const inserts = new Set(
      listOf(optional(node.definition, 'domain_transforms', []))
        .filter((transform) => optional(transform, 'relation', null) === 'insert')
        .map((transform) => pyStr(demand(transform, 'from_port'))),
    );
    let ok = true;
    for (const [name, port] of entries(demand(demand(node.definition, 'ports'), 'inputs'))) {
      if (present(port, node.args) && !fed.has(expandedPortKeyOf(site, name)) && !inserts.has(name)) {
        ok = false;
        break;
      }
    }
    if (!ok) continue;
    done.add(at);
    for (const edge of outgoing.get(at) ?? []) fed.add(expandedPortKeyOf(edge.to, edge.toPort));
  }
  return done;
}

/**
 * `_peak_live(graph, values)`: the peak of live values along D1's order.
 *
 * "At each node, the values produced so far and not yet consumed by every consumer — its own
 * outputs included, its inputs still held while it runs — sized per element and per invocation
 * like a graph split's payload. A value a public output exposes is live to the end."
 *
 * The live set is sorted at every node, so the sums are taken in one order whatever the set's own
 * is, and the **first** node reaching the maximum keeps it: the comparison is `>`, not `>=`.
 */
export function peakLive(graph: ExpandedGraph, values: ReadonlyMap<string, PyRecord>): PyRecord {
  const remaining = new Map<string, bigint>();
  const owe = (id: string, more: bigint): void => {
    remaining.set(id, (remaining.get(id) ?? 0n) + more);
  };
  for (const edge of graph.edges) owe(valueId(edge.from, edge.fromPort), 1n);
  // "Exposed: consumed at the end."
  for (const [, at] of graph.outputsAt) owe(valueId(at.site, at.port), 1n);

  const consumes = new Map<string, string[]>();
  for (const edge of graph.edges) {
    const at = expandedKeyOf(edge.to);
    const id = valueId(edge.from, edge.fromPort);
    const held = consumes.get(at);
    if (held === undefined) consumes.set(at, [id]);
    else held.push(id);
  }
  const produces = new Map<string, string[]>();
  for (const [id, value] of values) {
    if (has(value, 'input')) continue;
    // `node, port = vid.rsplit('.', 1)`: the identifier of the node that produced it.
    const node = id.slice(0, id.lastIndexOf('.'));
    const held = produces.get(node);
    if (held === undefined) produces.set(node, [id]);
    else held.push(id);
  }
  // "A public input's value is live from the start."
  const live = new Set<string>();
  for (const [name] of interfacesOf(graph, 'inputs')) {
    const value = values.get(name);
    owe(name, value === undefined ? 0n : BigInt((value['to'] as readonly PyValue[]).length));
  }
  for (const [name] of interfacesOf(graph, 'inputs')) {
    if ((remaining.get(name) ?? 0n) !== 0n) live.add(name);
  }
  const consumersOfInput = new Map<string, string[]>();
  for (const [name] of interfacesOf(graph, 'inputs')) {
    for (const endpoint of inputsAt(graph, name)) {
      const at = expandedKeyOf(endpoint.site);
      const held = consumersOfInput.get(at);
      if (held === undefined) consumersOfInput.set(at, [name]);
      else held.push(name);
    }
  }

  const size = (ids: readonly string[]): { element: bigint | number; invocation: PyRecord } => {
    const element = pySum(
      ids.map((id) => orZero((values.get(id) as PyRecord)['bytes_per_element'] as Figure)),
    );
    const invocation: PyRecord = {};
    for (const id of ids) weigh(invocation, values.get(id) as PyRecord);
    return { element, invocation };
  };

  let peak: {
    total: bigint | number;
    node: string;
    ids: string[];
    element: bigint | number;
    invocation: PyRecord;
  } | null = null;
  for (const site of graph.order) {
    const node = identOf(site);
    for (const id of produces.get(node) ?? []) live.add(id);
    const ids = [...live].sort(comparePythonStrings);
    const { element, invocation } = size(ids);
    const total = pySum(Object.values(invocation) as (bigint | number)[]);
    if (peak === null || (pyOrder(total, peak.total, '>') ?? -1) > 0) {
      peak = { total, node, ids, element, invocation };
    }
    const at = expandedKeyOf(site);
    for (const id of [...(consumes.get(at) ?? []), ...(consumersOfInput.get(at) ?? [])]) {
      const left = (remaining.get(id) ?? 0n) - 1n;
      remaining.set(id, left);
      if (left <= 0n) live.delete(id);
    }
  }
  const product: PyRecord = {};
  put(product, 'node', peak === null ? null : peak.node);
  put(product, 'values', peak === null ? [] : peak.ids);
  put(product, 'bytes_per_element', peak === null ? 0n : peak.element);
  put(product, 'bytes_per_invocation', peak === null ? {} : peak.invocation);
  return product;
}

/** `round(x)`: Python's own, which breaks a tie towards the even integer. */
export function pyRound(value: number): bigint {
  const down = Math.floor(value);
  const rest = value - down;
  if (rest > 0.5) return BigInt(down) + 1n;
  if (rest < 0.5) return BigInt(down);
  const whole = BigInt(down);
  return whole % 2n === 0n ? whole : whole + 1n;
}

/** `math.lcm(a, b)` over two integers. */
function lcm(left: bigint, right: bigint): bigint {
  if (left === 0n || right === 0n) return 0n;
  let a = left < 0n ? -left : left;
  let b = right < 0n ? -right : right;
  const product = a * b;
  while (b !== 0n) [a, b] = [b, a % b];
  return product / a;
}
