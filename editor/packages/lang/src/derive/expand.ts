/**
 * `derive._expand`: the analysis graph with every template instance expanded in place (§5.1).
 *
 * "Template instances are expanded before anything is derived: every product is computed once,
 * over the expanded graph, and an instance, tensor, state, value or graph_split inside an instance
 * carries the instance's prefix (§5.2 rule 2) — as D1 names it." This is that expansion, and it is
 * *not* D1's: `d1.emit` builds the emitted document, this rebuilds the **validator's** graph, so
 * that D2–D6 read one graph whose nodes are D1's nodes and whose arguments are the ones V1–V20
 * were decided on. `products` runs both and requires them to agree (`consistent`, in
 * `products.ts`).
 *
 * **A site of the expanded graph is a prefix and a key.** The tools wrap a key as
 * `('sub', prefix, key)` and read it back with `ident`; here an {@link ExpandedSite} carries the
 * two beside each other, {@link identOf} is `ident` — `whereOfSite` under the instance's prefix,
 * which is §5.2 rule 2's identifier and the validator's own `where` — and {@link expandedKeyOf}
 * is the string a `Map` keys by, injective because no identifier of the grammar carries a `NUL`.
 *
 * **What the expansion rewires.** An edge into a template instance fans out to the template's own
 * destinations and an edge out of one starts at the template's source; the template's streams take
 * the names of the caller's streams that feed its inputs; its tensor and state identities join the
 * caller's under the prefix, and a located tensor's physical names go under the instance's
 * `weights_location_prefix` — without one, in a document that locates nothing, the tensor is
 * unlocated like every other (§3.4).
 *
 * **What it does not rewire, reproduced.** A state identity's `writer` keeps the sub-graph's own
 * key: `instances(kind)` replaces `identity` and `members` and nothing else, at every level, so a
 * state inside a template instance names its writer *without* the prefix its members carry
 * (`decoder/attn[layer=0].kv` beside the member `text/decoder/attn[layer=0].kv`). Measured on
 * `shieldstral-3b-composite`, where D4 writes exactly that. The writer is therefore kept as the
 * unwrapped {@link IdentityMember} it is, and feature 1.8b writes what the tools write.
 */
import type { Env, Quantities } from '../expr/model.js';
import type { PyRecord, PyValue } from '../expr/value.js';
import { PyKeyError } from '../expr/errors.js';
import { put } from '../json/tree.js';
import { demand, entries, listOf, optional } from '../library/access.js';
import { pyRepr, pyStr } from '../library/repr.js';
import {
  keyOf,
  portKeyOf,
  selectSite,
  valueToken,
  whereOfSite,
  type GraphAnalysis,
  type Indexing,
  type ResolvedSite,
  type SiteKey,
} from '../validate/graph.js';
import {
  EMPTY_BINDINGS,
  type BindingsAnalysis,
  type EvaluatedLocation,
  type EvaluatedParts,
  type IdentityMember,
  type StateInstance,
  type TensorInstance,
} from '../validate/bindings/index.js';

/** A site of the expanded graph: the validator's key, under the prefixes of the instances above it. */
export interface ExpandedSite {
  /** `''` at the top level, `text/` inside one template instance, `a/b/` inside two. */
  readonly prefix: string;
  /** The site as the document that declares it names it. */
  readonly key: SiteKey;
}

/** `ident(key)`: the identifier of §5.2 rule 2 — what D1 keys its nodes by and D2–D6 refer to. */
export function identOf(site: ExpandedSite): string {
  return site.prefix + whereOfSite(site.key);
}

/**
 * The site as one string, so that a `Map` keys what Python keys by a tuple.
 *
 * A prefix is a run of identifiers and index brackets and carries no `NUL`, so the first `NUL`
 * separates it from {@link keyOf}'s own encoding and the reading is injective. The top level's
 * empty prefix is `('root', name)` unwrapped in the tools, and one key space here.
 */
export function expandedKeyOf(site: ExpandedSite): string {
  return `${site.prefix}\u0000${keyOf(site.key)}`;
}

/** `(site, port)` of the expanded graph: what an interface resolves to and what an edge names. */
export interface ExpandedPort {
  readonly site: ExpandedSite;
  readonly port: PyValue;
}

/** One node: the site, and `resolved[key]`'s `(name, definition, args)`. */
export interface ExpandedNode {
  readonly site: ExpandedSite;
  /** The primitive's name, as the instance writes it. */
  readonly primitive: PyValue;
  /** The definition — a template's interface replaced by its expansion, as `analyse` replaced it. */
  readonly definition: PyValue;
  /** `resolve_arguments`' answer: the typed argument map V1–V20 were decided on. */
  readonly args: PyRecord;
}

/** One value edge of the expanded graph, its endpoints resolved through every template instance. */
export interface ExpandedEdge {
  readonly from: ExpandedSite;
  readonly fromPort: PyValue;
  readonly to: ExpandedSite;
  readonly toPort: PyValue;
  /** The binding that emitted it, under the prefixes of the instances above it. */
  readonly binding: string;
}

/** One entry of `domains`: the `(site, port)` the tools key it by, and the domain it carries. */
export interface ExpandedPortDomain {
  readonly site: ExpandedSite;
  readonly port: PyValue;
  readonly domain: Indexing;
}

/** The composition a generated site belongs to: its prefixed name, and the site's index values. */
export interface ExpandedComposition {
  readonly name: string;
  /** `dict(key[3])`: the indices in name order, which is the order the key sorts them in. */
  readonly indices: ReadonlyMap<string, PyValue>;
}

/** `meta[key]`: what the expansion remembers about a node its `resolved` entry does not say. */
export interface ExpandedMeta {
  /** The site's families and its composition's, as one set (D1 lists them sorted). */
  readonly families: ReadonlySet<string>;
  /** The composition and the point of its grid, or `null` for a root instance. */
  readonly composition: ExpandedComposition | null;
}

/** One parameter identity instance of the expanded graph (D3). */
export interface ExpandedTensorInstance {
  /** The identity instance, under the prefix of the instance it lives in. */
  readonly identity: string;
  readonly rule: string;
  readonly env: Env;
  readonly members: readonly ExpandedMember[];
  readonly dtype: PyValue | null;
  /** The evaluated location, its physical names under the instance's weights prefix (§3.4). */
  readonly location?: EvaluatedLocation;
}

/** One state identity instance of the expanded graph (D4). */
export interface ExpandedStateInstance {
  readonly identity: string;
  readonly rule: string;
  readonly members: readonly ExpandedMember[];
  readonly dtype: PyValue | null;
  readonly indices: readonly string[];
  /**
   * The member that writes it (V20), **unprefixed**: `instances(kind)` replaces `identity` and
   * `members` alone, so the writer of a state inside a template instance is the sub-graph's own
   * key. Reproduced, not corrected — see the module's note.
   */
  readonly writer: IdentityMember | null;
}

/** One member of an identity instance, its site resolved through the instances above it. */
export interface ExpandedMember {
  readonly site: ExpandedSite;
  /** The parameter slot or the state port. */
  readonly name: PyValue;
}

/** The expanded graph: `_expand`'s answer, under names rather than dictionary keys. */
export interface ExpandedGraph {
  /** The document normalised — the *calling* document at every level, as `dict(graph, …)` keeps it. */
  readonly model: PyRecord;
  /** The quantities of that document, under the assignment in force. */
  readonly quantities: Quantities;
  /** Every node, by {@link expandedKeyOf}. */
  readonly resolved: ReadonlyMap<string, ExpandedNode>;
  /** Every value edge, the caller's first and each expansion's after them. */
  readonly edges: readonly ExpandedEdge[];
  /** The indexing domain of every `(site, port)`, by {@link expandedPortKeyOf}. */
  readonly domains: ReadonlyMap<string, ExpandedPortDomain>;
  /** Each node's own indexing domain, or `null` where it has none. */
  readonly own: ReadonlyMap<string, Indexing | null>;
  /** The validator's topological order, each template instance replaced by its expansion in place. */
  readonly order: readonly ExpandedSite[];
  /** Families and composition per node, by {@link expandedKeyOf}. */
  readonly meta: ReadonlyMap<string, ExpandedMeta>;
  /** Every composition name, prefixed, in declaration order, the expansions' after the caller's. */
  readonly compositions: readonly string[];
  /** The public inputs of this level, resolved to the instance ports they feed. */
  readonly inputsAt: ReadonlyMap<string, readonly ExpandedPort[]>;
  /** The public outputs of this level, resolved to the instance port each starts at. */
  readonly outputsAt: ReadonlyMap<string, ExpandedPort>;
  readonly tensorInstances: readonly ExpandedTensorInstance[];
  readonly stateInstances: readonly ExpandedStateInstance[];
}

/**
 * `resolved[key]` for an identity instance's member: the node a product reads its facts from.
 *
 * D3 and D4 each read everything an entry says about a slot or a state port from the identity
 * instance's **first** member — "of the first member; V15 makes the others compatible" — so this
 * is the one lookup they share. `members` holds only the members whose site the *analysis*
 * resolved, and the expansion then drops every template instance from `resolved`, so the one key
 * that can miss here is a slot or a port bound on a template instance — which V7 refuses long
 * before a derivation, a template primitive declaring no parameter and no state of its own. The
 * tools raise `KeyError` on the site's tuple; the port names the identifier, which is the same
 * site said the way §5.2 rule 2 says it.
 */
export function nodeAt(graph: ExpandedGraph, member: ExpandedMember): ExpandedNode {
  const node = graph.resolved.get(expandedKeyOf(member.site));
  if (node === undefined) throw new PyKeyError(pyRepr(identOf(member.site)));
  return node;
}

/** A `(site, port)` pair as one key, keyed as {@link expandedKeyOf} keys a site. */
export function expandedPortKeyOf(site: ExpandedSite, port: PyValue): string {
  return `${expandedKeyOf(site)}\u0000\u0000${valueToken(port)}`;
}

/** `_wrap(prefix, key)`: the key of a node of this level, seen from the document above it. */
function wrap(prefix: string, key: SiteKey): ExpandedSite {
  return { prefix, key };
}

/**
 * `_prefixed(location, prefix)`: an evaluated location with every physical name under the
 * instance's weights prefix (§3.4) — "the tensor, the parts of a stack or a concat, the tensor of
 * a slice".
 */
export function prefixedLocation(location: EvaluatedLocation, prefix: string): EvaluatedLocation {
  if ('tensor' in location) return { tensor: prefix + location.tensor };
  if ('stack' in location) {
    const stack = location.stack;
    return { stack: { ...stack, parts: stack.parts.map((part) => prefixedLocation(part, prefix)) } };
  }
  if ('concat' in location) {
    const concat = location.concat;
    return {
      concat: { ...concat, parts: concat.parts.map((part) => prefixedLocation(part, prefix)) },
    };
  }
  return { slice: { ...location.slice, tensor: prefix + location.slice.tensor } };
}

/**
 * The evaluated location as a product writes it: `evaluate_location`'s own dictionary.
 *
 * The validator builds a typed structure (`validate/bindings/analysis.ts`); the product writes the
 * four forms with the members in the order the tools write them, which the derived schema's
 * `location` fixes — `{tensor}`, `{stack: {axis, dim, parts}}`, `{concat: {axis, dim, parts}}`,
 * `{slice: {tensor, axis, dim, offset, extent}}`. `dim` is "the position of the named axis in the
 * tensor's shape", the storage axis being position 0 where a slot declares a multiplicity.
 */
export function locatedValue(location: EvaluatedLocation): PyValue {
  if ('tensor' in location) {
    const written: PyRecord = {};
    put(written, 'tensor', location.tensor);
    return written;
  }
  if ('stack' in location) return partsValue('stack', location.stack);
  if ('concat' in location) return partsValue('concat', location.concat);
  const slice: PyRecord = {};
  put(slice, 'tensor', location.slice.tensor);
  put(slice, 'axis', location.slice.axis);
  put(slice, 'dim', location.slice.dim);
  put(slice, 'offset', location.slice.offset);
  put(slice, 'extent', location.slice.extent);
  const written: PyRecord = {};
  put(written, 'slice', slice);
  return written;
}

/** `{"<form>": {"axis", "dim", "parts"}}`: a stack over its coordinates, or a concatenation. */
function partsValue(form: 'stack' | 'concat', parts: EvaluatedParts): PyValue {
  const inner: PyRecord = {};
  put(inner, 'axis', parts.axis);
  put(inner, 'dim', parts.dim);
  put(inner, 'parts', parts.parts.map((part) => locatedValue(part)));
  const written: PyRecord = {};
  put(written, form, inner);
  return written;
}

/** What one level of the expansion answers; the caller reads only these of a nested one. */
interface Expansion {
  resolved: Map<string, ExpandedNode>;
  edges: ExpandedEdge[];
  domains: Map<string, ExpandedPortDomain>;
  own: Map<string, Indexing | null>;
  order: ExpandedSite[];
  meta: Map<string, ExpandedMeta>;
  compositions: string[];
  inputsAt: Map<string, ExpandedPort[]>;
  outputsAt: Map<string, ExpandedPort>;
  tensorInstances: ExpandedTensorInstance[];
  stateInstances: ExpandedStateInstance[];
}

/**
 * `_expand(graph)`: the analysis of a valid document, with every template instance expanded.
 *
 * The analysis is `validate.analyse`'s — the whole of it, bindings included, since the identity
 * instances D3 and D4 read are the bindings stage's answer.
 */
export function expandAnalysis(analysis: GraphAnalysis): ExpandedGraph {
  const expansion = expansionOf(analysis, '');
  return { model: modelOf(analysis), quantities: analysis.quantities, ...expansion };
}

/** The document a level was analysed over; `null` only where the reading refused it. */
function modelOf(analysis: GraphAnalysis): PyRecord {
  if (analysis.model === null) {
    throw new TypeError('a graph with no document cannot be expanded: it was never analysed');
  }
  return analysis.model;
}

/** The bindings stage's answer, which every analysis of a valid document carries. */
function bindingsOf(analysis: GraphAnalysis): BindingsAnalysis {
  return analysis.bindings ?? EMPTY_BINDINGS;
}

/** The site behind a key of `resolved`, which every template instance of `subResults` also has. */
function siteOf(analysis: GraphAnalysis, id: string): ResolvedSite {
  const found = analysis.resolved.get(id);
  if (found === undefined) throw new PyKeyError(pyRepr(id));
  return found;
}

function expansionOf(analysis: GraphAnalysis, prefix: string): Expansion {
  const model = modelOf(analysis);
  const bindings = bindingsOf(analysis);
  const subs = analysis.subResults;

  // The template instances, expanded under their own prefix, in the order the walk resolved them.
  const inner = new Map<string, Expansion>();
  for (const [id, sub] of subs) {
    inner.set(id, expansionOf(sub, `${prefix}${whereOfSite(siteOf(analysis, id).key)}/`));
  }

  // "The template's streams are the caller's streams that feed its inputs."
  for (const [id, sub] of subs) {
    const site = siteOf(analysis, id).key;
    const rename = new Map<string, PyValue>();
    for (const [name, declared] of entries(
      demand(demand(modelOf(sub), 'interfaces'), 'inputs'),
    )) {
      const fed = analysis.domains.get(portKeyOf(site, name));
      if (fed !== undefined) rename.set(pyStr(optional(declared, 'stream', name)), fed.domain[1]);
    }
    const nested = inner.get(id) as Expansion;
    nested.domains = new Map(
      [...nested.domains].map(([key, entry]) => [
        key,
        { ...entry, domain: [entry.domain[0], renamed(rename, entry.domain[1])] as Indexing },
      ]),
    );
    nested.own = new Map(
      [...nested.own].map(([key, domain]) => [
        key,
        domain === null ? null : ([domain[0], renamed(rename, domain[1])] as Indexing),
      ]),
    );
  }

  /** `targets(key, port)`: the instance ports an edge into this site delivers to. */
  const targets = (key: SiteKey, port: PyValue): readonly ExpandedPort[] => {
    const nested = inner.get(keyOf(key));
    if (nested === undefined) return [{ site: wrap(prefix, key), port }];
    const found = nested.inputsAt.get(pyStr(port));
    if (found === undefined) throw new PyKeyError(pyRepr(port));
    return found;
  };

  /** `source(key, port)`: the instance port an edge out of this site starts at. */
  const source = (key: SiteKey, port: PyValue): ExpandedPort => {
    const nested = inner.get(keyOf(key));
    if (nested === undefined) return { site: wrap(prefix, key), port };
    const found = nested.outputsAt.get(pyStr(port));
    if (found === undefined) throw new PyKeyError(pyRepr(port));
    return found;
  };

  const select = (selector: PyValue): SiteKey =>
    selectSite(selector, analysis.quantities, new Map());

  const interfaces = demand(model, 'interfaces');
  const inputsAt = new Map<string, ExpandedPort[]>();
  for (const [name, declared] of entries(demand(interfaces, 'inputs'))) {
    const at: ExpandedPort[] = [];
    for (const endpoint of listOf(demand(declared, 'to'))) {
      at.push(
        ...targets(select(demand(endpoint, 'instance')), demand(endpoint, 'port')),
      );
    }
    inputsAt.set(name, at);
  }
  const outputsAt = new Map<string, ExpandedPort>();
  for (const [name, declared] of entries(demand(interfaces, 'outputs'))) {
    const from = demand(declared, 'from');
    outputsAt.set(name, source(select(demand(from, 'instance')), demand(from, 'port')));
  }

  const resolved = new Map<string, ExpandedNode>();
  const own = new Map<string, Indexing | null>();
  const meta = new Map<string, ExpandedMeta>();
  for (const [id, entry] of analysis.resolved) {
    if (inner.has(id)) continue;
    const site = wrap(prefix, entry.key);
    const at = expandedKeyOf(site);
    resolved.set(at, {
      site,
      primitive: entry.primitive,
      definition: entry.definition,
      args: entry.args,
    });
    own.set(at, analysis.own.get(id) ?? null);
    meta.set(at, metaOf(model, prefix, entry.key));
  }

  const domains = new Map<string, ExpandedPortDomain>();
  for (const [, entry] of analysis.domains) {
    if (inner.has(keyOf(entry.site))) continue;
    const site = wrap(prefix, entry.site);
    domains.set(expandedPortKeyOf(site, entry.port), {
      site,
      port: entry.port,
      domain: entry.domain,
    });
  }

  const order: ExpandedSite[] = [];
  for (const key of analysis.order) {
    const nested = inner.get(keyOf(key));
    if (nested !== undefined) {
      order.push(...nested.order);
      continue;
    }
    const site = wrap(prefix, key);
    if (resolved.has(expandedKeyOf(site))) order.push(site);
  }

  const edges: ExpandedEdge[] = [];
  for (const edge of analysis.edges) {
    const from = source(edge.from, edge.fromPort);
    for (const to of targets(edge.to, edge.toPort)) {
      edges.push({
        from: from.site,
        fromPort: from.port,
        to: to.site,
        toPort: to.port,
        binding: prefix + edge.binding,
      });
    }
  }

  const compositions = listOf(demand(model, 'compositions')).map((name) => prefix + pyStr(name));

  const tensorInstances = bindings.tensorInstances.map((instance) =>
    ownTensor(instance, prefix),
  );
  const stateInstances = bindings.stateInstances.map((instance) => ownState(instance, prefix));

  for (const [id, nested] of inner) {
    for (const [key, node] of nested.resolved) resolved.set(key, node);
    for (const [key, entry] of nested.domains) domains.set(key, entry);
    for (const [key, domain] of nested.own) own.set(key, domain);
    for (const [key, entry] of nested.meta) meta.set(key, entry);
    edges.push(...nested.edges);
    compositions.push(...nested.compositions);
    const weights = analysis.weightsPrefixes.get(id);
    for (const instance of nested.tensorInstances) {
      tensorInstances.push(located(instance, weights));
    }
    stateInstances.push(...nested.stateInstances);
  }

  return {
    resolved,
    edges,
    domains,
    own,
    order,
    meta,
    compositions,
    inputsAt,
    outputsAt,
    tensorInstances,
    stateInstances,
  };
}

/** `rename.get(stream, stream)`: the caller's name for a stream the template introduced. */
function renamed(rename: ReadonlyMap<string, PyValue>, stream: PyValue): PyValue {
  if (typeof stream !== 'string') return stream;
  const found = rename.get(stream);
  return found === undefined ? stream : found;
}

/** `meta[w]`: the site's families and, for a generated one, its composition and its indices. */
function metaOf(model: PyRecord, prefix: string, key: SiteKey): ExpandedMeta {
  if (key.kind === 'root') {
    const declared = demand(demand(model, 'instances'), key.name);
    return { families: familiesOf(declared), composition: null };
  }
  const composition = demand(demand(model, 'compositions'), key.composition);
  const site = demand(demand(composition, 'instances'), key.name);
  const families = new Set([...familiesOf(site), ...namesOf(demand(composition, 'families'))]);
  return {
    families,
    composition: {
      name: prefix + key.composition,
      indices: new Map(key.indices.map((index) => [index.name, index.value])),
    },
  };
}

/** `set(declaration['families'])`. */
function familiesOf(declared: PyValue): ReadonlySet<string> {
  return new Set(namesOf(demand(declared, 'families')));
}

/** `set(names)`: a declared list of names, as `for name in names` reads it. */
function namesOf(value: PyValue): string[] {
  return listOf(value).map((name) => pyStr(name));
}

/** One of this level's own tensor identity instances, seen from the document above it. */
function ownTensor(instance: TensorInstance, prefix: string): ExpandedTensorInstance {
  const expanded: ExpandedTensorInstance = {
    identity: prefix + instance.identity,
    rule: instance.rule,
    env: instance.env,
    members: instance.members.map((member) => wrapMember(member, prefix)),
    dtype: instance.dtype,
  };
  return instance.location === undefined
    ? expanded
    : { ...expanded, location: instance.location };
}

/** The same for a state identity instance; its `writer` is left where the tools leave it. */
function ownState(instance: StateInstance, prefix: string): ExpandedStateInstance {
  return {
    identity: prefix + instance.identity,
    rule: instance.rule,
    members: instance.members.map((member) => wrapMember(member, prefix)),
    dtype: instance.dtype,
    indices: instance.indices,
    writer: instance.writer,
  };
}

function wrapMember(member: IdentityMember, prefix: string): ExpandedMember {
  return { site: wrap(prefix, member.site), name: member.name };
}

/**
 * A tensor of an expanded template instance, located under the instance's weights prefix.
 *
 * "An instance's tensor is located at the instance's evaluated prefix followed by the template's
 * evaluated name (§3.4); without a prefix the template's names alone locate nothing, and the
 * tensor is unlocated like every other of the document."
 */
function located(
  instance: ExpandedTensorInstance,
  prefix: string | undefined,
): ExpandedTensorInstance {
  if (instance.location === undefined) return instance;
  if (prefix === undefined) {
    // `{k: v for k, v in inst.items() if k != 'location'}`: the same instance, unlocated.
    return {
      identity: instance.identity,
      rule: instance.rule,
      env: instance.env,
      members: instance.members,
      dtype: instance.dtype,
    };
  }
  return { ...instance, location: prefixedLocation(instance.location, prefix) };
}
