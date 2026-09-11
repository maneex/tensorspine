/**
 * D6, the Derived Decomposition Options: the port of `derive.d6` (§7, derived guide §8).
 *
 * "D6 answers one question of a model a consumer has never seen: *along which lines can it be
 * taken apart, and what is at each line* — without choosing a line." Two decompositions, and D6
 * carries both without deciding between them:
 *
 * - the **sequential** one, the graph splits. D2 already computed them — "the ancestor closure of
 *   a layer prefix or of a family is downward closed, so every crossing edge points out of it" —
 *   and D6 restates each with the three facts a reader of a *line* wants that a reader of a
 *   *payload* does not: the `block` itself, written in D1's published topological order, how many
 *   values cross (the length of D2's payload, the payload staying in D2 and D5), and the state
 *   identities the line **separates**.
 * - the **parallel** one, the `partition_options`: per node, every partition its primitive
 *   declares whose condition holds, with the communications it admits — "always a list" — and the
 *   granularity a shard keeps whole, evaluated on that node's arguments.
 *
 * Beside them, `information_loss`: O5.10's rule that a flattened axis without declared factors is
 * *unknown* partitionability and never *absent* partitionability. "Placement derivation must use
 * the non-partitionable case and report information loss, never present non-partitionability as a
 * known fact" (§4.3) — so the row names the slot and the axis and says why, and the consumer
 * decides what to do about it.
 *
 * **A separated state is the one figure D6 computes of its own.** A state identity whose members
 * fall on both sides of a line has its storage written on one side and read on the other; the
 * entry says which side the writer is on, who is on each, and — for a `window` identity alone —
 * which far-side members may need positions the ring no longer holds. "A `window` state keeps
 * `span` positions; a reader served through it alone cannot see older ones … The list says whom
 * the precaution concerns; it decides nothing" (derived guide §8, the reference plan's finding
 * 26). An `append` or `fixed` identity needs no such list: nothing is evicted.
 *
 * Every name in the product is D1's (§5.2 rule 2), a node inside a template instance under its
 * prefix — which is why the `block` is filtered out of D1's own order rather than sorted here.
 */
import { pyEqual } from '../expr/arithmetic.js';
import { PyKeyError } from '../expr/errors.js';
import { primitiveCondition, primitiveValue } from '../expr/primitive.js';
import { truthy, type PyRecord, type PyValue } from '../expr/value.js';
import { put } from '../json/tree.js';
import { demand, entries, get, has, listOf, optional } from '../library/access.js';
import { pyRepr, pyStr } from '../library/repr.js';
import { comparePythonStrings } from '../schema/repr.js';
import { structuralGraphSplits } from './d2.js';
import {
  expandedKeyOf,
  identOf,
  type ExpandedGraph,
  type ExpandedNode,
  type ExpandedSite,
} from './expand.js';
import { numberOf } from './figures.js';

/** O5.10's line, as the tools write it: two literals joined, one sentence. */
const FLATTENED =
  'flattened axis without declared factors (O5.10): ' +
  'partitionability along its factors is unknown';

/**
 * `d6(graph, cat, products2, products4, order)`: the decomposition options of one expanded graph.
 *
 * The tools' `cat` is dead — the body never reads it, D6 writing no dtype and no width — so it is
 * not a parameter here, as `d5`'s was not (feature 1.8d), `_visits`' first was not (feature 1.8b)
 * and `_check_type`'s two were not (feature 1.5). The fourth instance.
 *
 * `order` is D1's `topological_order`: the emitted document's own list, not the expansion's
 * `order`, because a `block` is published in the order a reader of D1 already has (feature 1.8a
 * measured that the two orders differ on `gemma3n-kvshare`).
 */
export function d6(
  graph: ExpandedGraph,
  products2: PyValue,
  products4: PyValue,
  order: readonly PyValue[],
): PyRecord {
  const partitions: PyRecord[] = [];
  const loss: PyRecord[] = [];
  for (const [, node] of graph.resolved) {
    partitions.push(...partitionsOf(node));
    loss.push(...lossOf(node));
  }

  const product: PyRecord = {};
  put(product, 'graph_splits', splitsOf(graph, products2, products4, order));
  put(product, 'partition_options', partitions);
  put(product, 'information_loss', loss);
  return product;
}

/**
 * "For every node — template instances expanded — every partition its primitive declares whose
 * condition holds" (derived guide §8).
 *
 * Three readings of the declaration, each the tools' own. The **condition** is the partition's
 * `when`, absent meaning it always applies. The **communication** is carried as declared and
 * wrapped in a list when it is a single value — "always a list", so that a consumer reads one
 * shape whether the primitive admits one pattern or several (`embed` admits two). The
 * **granularity** is evaluated on the node's arguments through `_num`, so a granularity the
 * arguments leave undetermined is a blank rather than a refusal, and an undeclared one is the
 * *integer* `1` — the number the schema's nullable `granularity` is written for.
 *
 * The `target` is carried whole and read not at all: an argument axis, an instance-key axis, a
 * state payload axis, `any_axis` or `none` mean something to the consumer that places the model
 * and nothing to the derivation (the audit of §1 (d) holds D6 to that).
 */
function partitionsOf(node: ExpandedNode): PyRecord[] {
  const rows: PyRecord[] = [];
  for (const declared of listOf(optional(node.definition, 'partition_options', []))) {
    if (has(declared, 'when') && !truthy(primitiveCondition(demand(declared, 'when'), node.args))) {
      continue;
    }
    const communication = demand(declared, 'communication');
    const row: PyRecord = {};
    put(row, 'node', identOf(node.site));
    put(row, 'primitive', node.primitive);
    put(row, 'target', demand(declared, 'target'));
    put(row, 'communication', Array.isArray(communication) ? communication : [communication]);
    put(
      row,
      'granularity',
      has(declared, 'granularity')
        ? numberOf(primitiveValue(demand(declared, 'granularity'), node.args))
        : 1n,
    );
    rows.push(row);
  }
  return rows;
}

/**
 * O5.10: "a flattened shape declares its decomposition; without one, placement derivation reports
 * information loss rather than asserting non-partitionability".
 *
 * A flattened axis is one whose extent is written as an n-ary `multiply` — the product of the
 * axes it stands for — and the loss is the absence of the `factors` that name them. Every
 * parameter slot the primitive declares is walked, a slot whose `present_when` fails on this
 * node's arguments excepted: the slot is not there to be partitioned.
 *
 * The reading of `multiply` is the one place D6 names a member of the operator vocabulary, and it
 * is the reason the audit of §1 (d) asks this module which operators flatten.
 */
function lossOf(node: ExpandedNode): PyRecord[] {
  const rows: PyRecord[] = [];
  for (const [name, slot] of entries(demand(node.definition, 'parameters'))) {
    if (has(slot, 'present_when') && !truthy(primitiveCondition(demand(slot, 'present_when'), node.args))) {
      continue;
    }
    for (const axis of listOf(demand(demand(slot, 'shape'), 'axes'))) {
      // `isinstance(a['extent'], dict) and a['extent'].get('op') == 'multiply'`: a `.get` that
      // tolerates an extent which is not a record at all, since the membership test guards it.
      const flattened = pyEqual(get(demand(axis, 'extent'), 'op'), 'multiply');
      if (!flattened || has(axis, 'factors')) continue;
      const row: PyRecord = {};
      put(row, 'node', identOf(node.site));
      put(row, 'slot', name);
      put(row, 'axis', demand(axis, 'axis'));
      put(row, 'reason', FLATTENED);
      rows.push(row);
    }
  }
  return rows;
}

/**
 * D2's splits, each with its block and the state identities it separates.
 *
 * The blocks are recomputed rather than carried: D2 emits a split's *payload* and keeps the block
 * to itself, so `_structural_graph_splits` is run again here and its answers are keyed by the
 * split's name — which is what makes the two products agree entry for entry, D2's list being
 * walked in its own order and each entry looked up by name.
 */
function splitsOf(
  graph: ExpandedGraph,
  products2: PyValue,
  products4: PyValue,
  order: readonly PyValue[],
): PyValue[] {
  const sites = siteIndex(graph);
  const blocks = new Map<string, ReadonlySet<string>>();
  for (const split of structuralGraphSplits(graph)) blocks.set(split.name, split.block);
  const states = listOf(demand(products4, 'states'));

  const written: PyValue[] = [];
  for (const split of listOf(demand(products2, 'graph_splits'))) {
    const name = pyStr(demand(split, 'graph_split'));
    const block = blocks.get(name);
    // `blocks[c['graph_split']]`, subscripted: a D2 split the structural walk does not name is
    // Python's own `KeyError`, never a split without a block.
    if (block === undefined) throw new PyKeyError(pyRepr(name));
    const ids = new Set<string>();
    for (const key of block) ids.add(identOf(siteAt(sites, key)));

    const separated: PyRecord[] = [];
    for (const state of states) {
      const entry = separatedOf(state, ids);
      if (entry !== null) separated.push(entry);
    }
    separated.sort((left, right) =>
      comparePythonStrings(pyStr(left['identity'] as PyValue), pyStr(right['identity'] as PyValue)),
    );

    const row: PyRecord = {};
    put(row, 'graph_split', demand(split, 'graph_split'));
    put(row, 'kind', demand(split, 'kind'));
    put(row, 'sizes', demand(split, 'sizes'));
    put(row, 'crossing_values', BigInt(listOf(demand(split, 'payload')).length));
    put(row, 'block', order.filter((node) => ids.has(pyStr(node))));
    put(row, 'separated_states', separated);
    written.push(row);
  }
  return written;
}

/**
 * One D4 identity against one block, or `null` when the line does not separate it.
 *
 * "Every D4 state identity whose members are not all on one side of the split: its storage is
 * written on one side and read on the other." The side of a member is the side of its *node*, and
 * the member is `node.state`, so the node is everything up to the member's **last** dot — which no
 * identifier of the repository makes different from its first, a node being a name or
 * `<composition>/<site>[<index>=<value>]`.
 *
 * `history_needed_by` is the far side minus the writer, and only for a `window` identity: an
 * `append` state holds every position it was given and a `fixed` one holds one, so no reader of
 * either can ask for something the storage has dropped. The subtraction of the writer removes
 * nothing: `far` is by construction the side *away* from the writer, so the writer is never among
 * its members, and the filter the tools write is dead in both directions. It is reproduced as
 * written, and the unit suite states the equality so that changing either reading has to say which
 * one it meant.
 */
function separatedOf(state: PyValue, ids: ReadonlySet<string>): PyRecord | null {
  const members = listOf(demand(state, 'members'));
  const sides = new Map<string, boolean>();
  for (const member of members) sides.set(pyStr(member), ids.has(nodeOf(pyStr(member))));
  const first = members.filter((member) => sideOf(sides, pyStr(member)));
  const second = members.filter((member) => !sideOf(sides, pyStr(member)));
  if (first.length === 0 || second.length === 0) return null;

  const writer = demand(state, 'writer');
  // `sides[s['writer']] if s['writer'] else True`: a subscript, so a writer that is not one of the
  // members raises Python's own `KeyError` — which is what a writer stripped of the prefix its
  // members carry would do (feature 1.8a's finding on `shieldstral-3b-composite`, unreachable
  // because an identity inside a template instance has one member and is never separated).
  const writerFirst = truthy(writer) ? sideOf(sides, pyStr(writer)) : true;
  const far = writerFirst ? second : first;
  const history = pyEqual(demand(state, 'evolution'), 'window')
    ? far.filter((member) => !pyEqual(member, writer))
    : [];

  const entry: PyRecord = {};
  put(entry, 'identity', demand(state, 'identity'));
  put(entry, 'evolution', demand(state, 'evolution'));
  put(entry, 'span', demand(state, 'span'));
  put(entry, 'bytes_per_cached_position', demand(state, 'bytes_per_cached_position'));
  put(entry, 'sharing', demand(state, 'sharing'));
  put(entry, 'writer', writer);
  put(entry, 'writer_side', writerFirst ? 'first' : 'second');
  put(entry, 'first', first);
  put(entry, 'second', second);
  put(entry, 'history_needed_by', history);
  return entry;
}

/** `sides[name]`, subscripted: a name the map has not is Python's own `KeyError`. */
function sideOf(sides: ReadonlyMap<string, boolean>, name: string): boolean {
  const held = sides.get(name);
  if (held === undefined) throw new PyKeyError(pyRepr(name));
  return held;
}

/** `m.rsplit('.', 1)[0]`: the node a `node.state` member names. */
function nodeOf(member: string): string {
  const dot = member.lastIndexOf('.');
  return dot === -1 ? member : member.slice(0, dot);
}

/**
 * The site behind every key a block can hold, so that `ident(k)` has a key to read.
 *
 * A block is a set of the expansion's own keys — its seeds are resolved nodes and the closure adds
 * the source of every edge — and the port keys a `Map` by a string where the tools key a `dict`
 * by a tuple, so the site has to be found again to be named. Both endpoints of every edge are
 * indexed beside the resolved nodes: the tools' `ident` reads the key itself and never asks
 * whether the node was resolved.
 */
function siteIndex(graph: ExpandedGraph): ReadonlyMap<string, ExpandedSite> {
  const sites = new Map<string, ExpandedSite>();
  for (const [key, node] of graph.resolved) sites.set(key, node.site);
  for (const edge of graph.edges) {
    for (const site of [edge.from, edge.to]) {
      const key = expandedKeyOf(site);
      if (!sites.has(key)) sites.set(key, site);
    }
  }
  return sites;
}

/** The site of one key, or the defect of a caller that built a block of its own. */
function siteAt(sites: ReadonlyMap<string, ExpandedSite>, key: string): ExpandedSite {
  const held = sites.get(key);
  if (held === undefined) throw new PyKeyError(pyRepr(key));
  return held;
}
