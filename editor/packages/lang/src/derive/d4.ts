/**
 * D4, the Derived State Inventory and Behavior: the port of `derive.d4` (§7, derived guide §6).
 *
 * "Descriptors, instances, keys, the writer of each identity instance, state liveness, visits per
 * phase, and permitted operations." One entry per **state identity instance**, as D3 writes one
 * per parameter identity instance: the identity is the allocation, its members are the state ports
 * that name it, and — §4.4 — "sharing is declared by listing several members under one identity,
 * never by a relation over keys; exactly one member writes an identity instance … and the others
 * read it (V20)". Cross-layer key/value sharing is that and nothing else.
 *
 * Everything an entry says about a port is read from its **first** member's definition, as D3 does
 * — V9 makes the others agree on the key axes, the payload, the rule and the stream, which is
 * exactly the list of facts below — and an identity instance no member of which resolved has no
 * entry at all.
 *
 * **The rule is the first one whose condition holds** (§4.3: "rules are ordered; the first
 * matching rule wins"), and with it come the evolution, the access geometry, the sharing
 * granularity, the indexing source and the two modulators. A present state port that no rule
 * matches is a V9 rejection, so `rule = None` cannot come out of a valid document; the tools guard
 * every field with it all the same, and so does this.
 *
 * **Three derived facts the graph supplies, not the primitive.** The `stream` the state grows
 * along is the instance's own indexing domain, or the domain of the port the rule indexes it by
 * (§4.3, §5.3). `carried_across_fragments` is the primitive's carrying condition **or** a state
 * indexed by a port whose stream is a fragmented input, "which carries it by definition" (§5.3,
 * V16). The `instance_key` is the identity's own indices followed by the port's `key_axes` (O5.5):
 * one allocation per distinct key.
 *
 * **The figures.** A payload is declared per position of the stream the state grows along — "one
 * cached position for `append` and `window` … the whole state for `fixed`" (§4.3) — so
 * `bytes_per_cached_position` is the payload's bytes at that reading, and `bytes_bounded` is
 * `span × bytes_per_cached_position` for a `window`, "the bytes of a full ring whose shape is
 * `[span] + shape`". `_sound` guards the element count, the byte size and the span against a
 * negative or non-finite value (R11).
 */
import { pyMultiply } from '../expr/arithmetic.js';
import { primitiveCondition, primitiveValue } from '../expr/primitive.js';
import { truthy, type PyRecord, type PyValue } from '../expr/value.js';
import { put } from '../json/tree.js';
import { demand, entries, has, listOf } from '../library/access.js';
import type { Library } from '../library/load.js';
import { pyStr } from '../library/repr.js';
import { sortedEffects } from '../describe/site.js';
import {
  applicableRule,
  fragmentedStreams,
  whereOfSite,
  type Indexing,
} from '../validate/index.js';
import {
  elementsOf,
  numberOf,
  orZero,
  productShape,
  pySum,
  selectedDtype,
  sound,
  widthOf,
  type Figure,
} from './figures.js';
import {
  expandedKeyOf,
  expandedPortKeyOf,
  identOf,
  nodeAt,
  type ExpandedGraph,
  type ExpandedSite,
  type ExpandedStateInstance,
} from './expand.js';

/** `d4(graph, cat)`: the state inventory of one expanded graph. */
export function d4(graph: ExpandedGraph, library: Library): PyRecord {
  // `_fragmented_streams(graph)` is a pure function of the document the graph was analysed over —
  // the *calling* document at every level, which `_expand` keeps — and the tools call it once per
  // carried state; it is read once here, for the same answer.
  const fragmented = fragmentedStreams(graph.model);
  const states: PyRecord[] = [];
  for (const instance of graph.stateInstances) {
    const entry = stateOf(graph, library, instance, fragmented);
    if (entry !== null) states.push(entry);
  }
  const product: PyRecord = {};
  put(product, 'states', states);
  put(product, 'totals', totalsOf(states));
  return product;
}

/** One entry, or `null` for an identity instance no member of which resolved. */
function stateOf(
  graph: ExpandedGraph,
  library: Library,
  instance: ExpandedStateInstance,
  fragmented: ReadonlySet<string>,
): PyRecord | null {
  const first = instance.members[0];
  if (first === undefined) return null;
  const node = nodeAt(graph, first);
  const port = demand(demand(node.definition, 'state_ports'), pyStr(first.name));
  const args = node.args;

  // "Rules are ordered; the first matching rule wins." The tools filter the whole list and take
  // its head; `applicableRule` stops at the first match, which answers the same rule because
  // `analyse` has already evaluated every rule's condition of every present state port of every
  // site — a condition that raised would have raised there, on a document no derivation reaches.
  const rule = applicableRule(port, args);
  // `rule['indexed_by']` is read before any truthiness test, as the tools read it: a rule that
  // declares none raises here rather than being read as no rule at all.
  const indexedBy = rule === null ? null : demand(rule, 'indexed_by');
  const sourceIndexed = indexedBy !== null && has(indexedBy, 'port');
  const stream = rule === null ? null : streamOf(graph, first.site, sourceIndexed, indexedBy);

  // "The primitive's carrying condition holds and the stream is a fragmented input (§5.3, V16), or
  // the state is indexed by a port whose stream is a fragmented input, which carries it by
  // definition." The first half reads the primitive alone — V16 is what ties it to the stream, and
  // D4 states the primitive's answer, not V16's.
  const carrying = has(port, 'carried_across');
  const carried =
    (carrying && truthy(primitiveCondition(demand(demand(port, 'carried_across'), 'when'), args))) ||
    (sourceIndexed && stream !== null && fragmented.has(pyStr(stream[1])));

  const payload = payloadOf(graph, library, instance, port, args);
  const perPosition = pySum(payload.map((component) => orZero(component['bytes'] as Figure)));
  const span = sound(modulator(rule, 'span', args), 'state span', instance.identity);

  const entry: PyRecord = {};
  put(entry, 'identity', instance.identity);
  put(
    entry,
    'members',
    instance.members.map((member) => `${identOf(member.site)}.${pyStr(member.name)}`),
  );
  // The writer keeps the sub-graph's own key: `instances(kind)` prefixes `identity` and `members`
  // and nothing else, so a state inside a template instance names a node the expanded graph has
  // not — `decoder/attn[layer=0].kv` beside the member `text/decoder/attn[layer=0].kv`.
  // Reproduced, not corrected: feature 1.8a measured it on `shieldstral-3b-composite`, and what
  // the tools write is what this writes.
  put(
    entry,
    'writer',
    instance.writer === null
      ? null
      : `${whereOfSite(instance.writer.site)}.${pyStr(instance.writer.name)}`,
  );
  put(entry, 'primitive', node.primitive);
  put(entry, 'state', first.name);
  put(entry, 'evolution', rule === null ? null : demand(rule, 'evolution'));
  put(entry, 'access', rule === null ? null : demand(rule, 'access'));
  put(entry, 'sharing', rule === null ? null : demand(rule, 'sharing'));
  put(entry, 'stream', streamValue(stream));
  put(entry, 'indexed_by_source', sourceIndexed);
  // `rule['indexed_by'].get('port')`: the port for a source-indexed state, `None` for `self`.
  put(entry, 'indexed_by_port', sourceIndexed ? demand(indexedBy as PyValue, 'port') : null);
  put(entry, 'instance_key', [...instance.indices, ...listOf(demand(port, 'key_axes'))]);
  put(entry, 'carried_across_fragments', carried);
  put(entry, 'span', span);
  put(entry, 'stride', modulator(rule, 'stride', args));
  put(entry, 'payload', payload);
  put(entry, 'bytes_per_cached_position', perPosition);
  // `(per_position * span) if span else None`: a span of zero bounds nothing, as a span that did
  // not resolve bounds nothing — Python's truth of the figure, not `is not None`.
  put(entry, 'bytes_bounded', span !== null && truthy(span) ? pyMultiply(perPosition, span) : null);
  put(entry, 'operations', sortedEffects(demand(port, 'operations')));
  put(entry, 'visits', visitsOf(sourceIndexed, rule === null ? null : demand(rule, 'evolution')));
  return entry;
}

/**
 * The payload, component by component in declaration order (§4.3, O5.1).
 *
 * "One state may contain several components of different types": each carries its own role, hence
 * its own selected dtype, its own evaluated shape and its own figures. The dtype selector is the
 * *identity's* — one binding selects the precision of the whole allocation — and the role's
 * default stands where it selects nothing (V14).
 */
function payloadOf(
  graph: ExpandedGraph,
  library: Library,
  instance: ExpandedStateInstance,
  port: PyValue,
  args: PyRecord,
): PyRecord[] {
  const payload: PyRecord[] = [];
  for (const [component, declared] of entries(demand(port, 'payload'))) {
    const role = demand(declared, 'role');
    const dtype = selectedDtype(graph.quantities, library, instance.dtype, role);
    const shape = demand(declared, 'shape');
    const count = elementsOf(shape, args, null);
    const entry: PyRecord = {};
    put(entry, 'component', component);
    put(entry, 'role', role);
    put(entry, 'dtype', dtype);
    put(entry, 'shape', productShape(shape, args));
    put(entry, 'elements', sound(count, 'state element count', instance.identity));
    // The width is looked up only where the count resolved, as the tools' conditional expression
    // looks it up: a dtype the table has not is a `KeyError` on a countable payload alone.
    put(
      entry,
      'bytes',
      sound(
        count === null ? null : (pyMultiply(count, widthOf(dtype)) as Figure),
        'state byte size',
        instance.identity,
      ),
    );
    payload.push(entry);
  }
  return payload;
}

/**
 * The stream the state grows along: "the instance's own, or the stream of the port it is indexed
 * by" (derived guide §6).
 *
 * The tools branch on `source_indexed` rather than on `'self' in indexed_by`, which is the same
 * reading under the grammar — `indexing_source` is a `oneOf` of the two — and the branch is kept
 * as `d4` writes it. Both maps are the *expanded* graph's, keyed by the site under the prefixes of
 * the instances above it, since a state inside a template instance grows along the caller's
 * stream, renamed by the expansion (§5.1).
 */
function streamOf(
  graph: ExpandedGraph,
  site: ExpandedSite,
  sourceIndexed: boolean,
  indexedBy: PyValue,
): Indexing | null {
  if (!sourceIndexed) return graph.own.get(expandedKeyOf(site)) ?? null;
  return graph.domains.get(expandedPortKeyOf(site, demand(indexedBy, 'port')))?.domain ?? null;
}

/** `{"kind": stream[0], "stream": stream[1]} if stream else None`. */
function streamValue(stream: Indexing | null): PyValue {
  if (stream === null) return null;
  const written: PyRecord = {};
  put(written, 'kind', stream[0]);
  put(written, 'stream', stream[1]);
  return written;
}

/** `_num(primitive_value(rule[name], args)) if rule and name in rule else None` (O5.8). */
function modulator(rule: PyValue | null, name: string, args: PyRecord): Figure {
  if (rule === null || !truthy(rule) || !has(rule, name)) return null;
  return numberOf(primitiveValue(demand(rule, name), args));
}

/**
 * `_visits(rule, source_indexed, evolution)`: "the rule of §7 in words".
 *
 * §7 states the three: "an `append` or `window` state indexed by its own stream is written once
 * per new element and read once per element produced; a state indexed by a source stream is
 * written once per source element and frozen when the source is complete; a `fixed` state is read
 * and written once per element". The source-indexed reading comes **first**, so a `fixed` state so
 * indexed takes it; every other evolution — and the absence of a rule — takes the third. The
 * tools' first parameter, the rule itself, is never read by the body, and is not a parameter here.
 */
function visitsOf(sourceIndexed: boolean, evolution: PyValue | null): PyRecord {
  if (sourceIndexed) {
    return visits(
      'once per element of the source stream, until the source is complete',
      'once per element produced',
    );
  }
  if (evolution === 'fixed') return visits('once per element', 'once per element');
  return visits('once per new element of its stream', 'once per element produced');
}

/** `{"write": …, "read": …}`, in the order the product writes them. */
function visits(write: string, read: string): PyRecord {
  const written: PyRecord = {};
  put(written, 'write', write);
  put(written, 'read', read);
  return written;
}

/**
 * `totals`: the six the guide names — "`identities`, `by_evolution`,
 * `append_bytes_per_cached_position` (the 'cache bytes per token' of a decoder), `bounded_bytes`,
 * `fixed_bytes`, `carried`".
 *
 * Each evolution feeds one byte total and no other: `append` and `fixed` contribute their payload
 * per cached position, `window` its bounded size with `or 0` for a span that did not resolve. The
 * sums are Python's, term by term in the order the states were emitted, so a real in one payload
 * promotes the total from that term on and not before.
 */
function totalsOf(states: readonly PyRecord[]): PyRecord {
  const summed = (evolution: string, name: string, blank = false): bigint | number =>
    pySum(
      states
        .filter((entry) => entry['evolution'] === evolution)
        .map((entry) =>
          blank ? orZero(entry[name] as Figure | undefined) : (entry[name] as bigint | number),
        ),
    );
  const totals: PyRecord = {};
  put(totals, 'identities', BigInt(states.length));
  put(totals, 'by_evolution', byEvolution(states));
  put(totals, 'append_bytes_per_cached_position', summed('append', 'bytes_per_cached_position'));
  put(totals, 'bounded_bytes', summed('window', 'bytes_bounded', true));
  put(totals, 'fixed_bytes', summed('fixed', 'bytes_per_cached_position'));
  put(
    totals,
    'carried',
    states
      .filter((entry) => truthy(entry['carried_across_fragments'] as PyValue))
      .map((entry) => entry['identity'] as PyValue),
  );
  return totals;
}

/**
 * `dict(Counter(s['evolution'] for s in states))`: how many identities each evolution has.
 *
 * A `Counter` keeps the order in which a key was first seen, which is the order of the states, and
 * that order is part of the derived document's bytes. The one key a Python dictionary can hold
 * here and a JSON object cannot is `None` — an identity no rule applies to, which V9 refuses
 * before any derivation; `json.dumps` writes such a key as `"null"`, and so does this, rather than
 * inventing a spelling of its own for a member no valid document produces. Python would keep a
 * literal evolution named `null` apart from it and write the member twice; the enumeration has
 * three values and none of them is that, so the two cannot meet.
 */
function byEvolution(states: readonly PyRecord[]): PyRecord {
  const counted: PyRecord = {};
  for (const entry of states) {
    const evolution = entry['evolution'];
    const name = evolution === null || evolution === undefined ? 'null' : pyStr(evolution);
    put(counted, name, ((counted[name] as bigint | undefined) ?? 0n) + 1n);
  }
  return counted;
}
