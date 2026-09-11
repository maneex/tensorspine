import { parse } from '../../src/json/index.js';
import {
  expandedKeyOf,
  rootSite,
  toPython,
  type ExpandedGraph,
  type ExpandedNode,
  type ExpandedTensorInstance,
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

/** An expanded graph holding those nodes and those parameter identity instances, and nothing else. */
export function syntheticGraph(
  nodes: readonly SyntheticNode[],
  tensors: readonly ExpandedTensorInstance[],
  quantities: ReadonlyMap<string, PyValue> = new Map(),
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
  return {
    model: {},
    quantities,
    resolved,
    edges: [],
    domains: new Map(),
    own: new Map(),
    order: [...resolved.values()].map((node) => node.site),
    meta: new Map(),
    compositions: [],
    inputsAt: new Map(),
    outputsAt: new Map(),
    tensorInstances: tensors,
    stateInstances: [],
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
