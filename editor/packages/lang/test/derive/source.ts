import { parse } from '../../src/json/index.js';
import {
  expandedKeyOf,
  expandedPortKeyOf,
  rootSite,
  toPython,
  type ExpandedGraph,
  type ExpandedNode,
  type ExpandedStateInstance,
  type ExpandedTensorInstance,
  type Indexing,
  type PyRecord,
  type PyValue,
} from '../../src/index.js';

/**
 * A hand-built expanded graph, for the branches of D3 no corpus document reaches.
 *
 * D3 is a pure function of `_expand`'s answer and the library, so the honest unit test hands it
 * that answer directly: one root node per declaration, the identity instances the case needs, and
 * nothing else. Everything a *document* would have to go through to produce such a graph — a
 * primitive declaring a negative extent, a sparsity unit on an axis its slot has not — is
 * refused by the loader or the validator long before the derivation, which is exactly why those
 * branches have no document.
 */
export interface SyntheticNode {
  /** The root instance's name, which is also its identifier (§5.2 rule 2). */
  readonly name: string;
  /** The primitive's name, as an instance writes it. */
  readonly primitive?: string;
  readonly definition: PyValue;
  readonly args?: PyRecord;
}

/** A value as a document writes it, with its float-ness: `2` a whole number, `2.0` a real. */
export function value(text: string): PyValue {
  return toPython(parse(text));
}

/** A record as a document writes it. */
export function record(text: string): PyRecord {
  return value(text) as PyRecord;
}

/** `{prefix: '', key: ('root', name)}`: a top-level site of an expanded graph. */
export function site(name: string): { prefix: string; key: ReturnType<typeof rootSite> } {
  return { prefix: '', key: rootSite(name) };
}

/**
 * What a case needs of an expanded graph beside its nodes and its parameter identity instances.
 *
 * D3 reads the library and the nodes alone; D4 also reads the state identity instances, the
 * indexing domains a stream is read from, and the document's public inputs — "the streams the
 * fragmented public inputs introduce or join" is the whole of what `model` is for here.
 */
export interface SyntheticExtra {
  readonly quantities?: ReadonlyMap<string, PyValue>;
  readonly states?: readonly ExpandedStateInstance[];
  /**
   * The document the graph was analysed over: only its `interfaces.inputs` are ever read.
   *
   * The default is a document with no public input at all, since the grammar requires the two
   * members and D4 reads them unguarded — a graph without them raises there as it raises in the
   * tools, which is a defect of the caller and not a fact about a document.
   */
  readonly model?: PyRecord;
  /** Each node's own indexing domain, by node name; a node absent from it has none. */
  readonly own?: Readonly<Record<string, Indexing | null>>;
  /** The indexing domain of a node's input port, keyed `<node>.<port>`. */
  readonly domains?: Readonly<Record<string, Indexing>>;
}

/** An expanded graph holding those nodes and those identity instances, and nothing else. */
export function syntheticGraph(
  nodes: readonly SyntheticNode[],
  tensors: readonly ExpandedTensorInstance[],
  extra: SyntheticExtra = {},
): ExpandedGraph {
  const resolved = new Map<string, ExpandedNode>();
  for (const node of nodes) {
    const at = site(node.name);
    resolved.set(expandedKeyOf(at), {
      site: at,
      primitive: node.primitive ?? node.name,
      definition: node.definition,
      args: node.args ?? {},
    });
  }
  const own = new Map<string, Indexing | null>();
  for (const [name, domain] of Object.entries(extra.own ?? {})) {
    own.set(expandedKeyOf(site(name)), domain);
  }
  const domains = new Map<string, { site: ReturnType<typeof site>; port: PyValue; domain: Indexing }>();
  for (const [at, domain] of Object.entries(extra.domains ?? {})) {
    const dot = at.lastIndexOf('.');
    const node = site(at.slice(0, dot));
    const port = at.slice(dot + 1);
    domains.set(expandedPortKeyOf(node, port), { site: node, port, domain });
  }
  return {
    model: extra.model ?? record('{"interfaces": {"inputs": {}, "outputs": {}}}'),
    quantities: extra.quantities ?? new Map(),
    resolved,
    edges: [],
    domains,
    own,
    order: [...resolved.values()].map((node) => node.site),
    meta: new Map(),
    compositions: [],
    inputsAt: new Map(),
    outputsAt: new Map(),
    tensorInstances: tensors,
    stateInstances: extra.states ?? [],
  };
}

/** One parameter identity instance over one node's slot, with whatever the case needs beside. */
export function tensorInstance(
  identity: string,
  members: readonly (readonly [string, string])[],
  extra: Partial<ExpandedTensorInstance> = {},
): ExpandedTensorInstance {
  return {
    identity,
    rule: identity,
    env: new Map(),
    members: members.map(([node, slot]) => ({ site: site(node), name: slot })),
    dtype: null,
    ...extra,
  };
}

/** One state identity instance over one node's port, with whatever the case needs beside. */
export function stateInstance(
  identity: string,
  members: readonly (readonly [string, string])[],
  extra: Partial<ExpandedStateInstance> = {},
): ExpandedStateInstance {
  const first = members[0];
  return {
    identity,
    rule: identity,
    members: members.map(([node, port]) => ({ site: site(node), name: port })),
    dtype: null,
    indices: [],
    // "Exactly one member writes an identity instance" (V20): the first, unless a case says
    // otherwise. It is an unprefixed member, as the expansion leaves it.
    writer: first === undefined ? null : { site: rootSite(first[0]), name: first[1] },
    ...extra,
  };
}
