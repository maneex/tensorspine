import { readFileSync, readdirSync } from 'node:fs';

import { parse } from '../../src/json/index.js';
import {
  basesOf,
  derivationGraph,
  expandedKeyOf,
  expandedPortKeyOf,
  loadLibrary,
  rootSite,
  toPython,
  type ExpandedEdge,
  type ExpandedGraph,
  type ExpandedMeta,
  type ExpandedNode,
  type ExpandedPort,
  type ExpandedStateInstance,
  type ExpandedTensorInstance,
  type Indexing,
  type Library,
  type PyRecord,
  type PyValue,
} from '../../src/index.js';
import { schemas } from '../describe/source.js';
import { repositoryRoot } from '../json/repository.js';
import { nodeSource } from '../library/source.js';

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
   * The value edges, as `['<node>.<port>', '<node>.<port>']` pairs in the order the graph holds
   * them — which D2 reads as the order its values, its crossings and its counts are built in.
   */
  readonly edges?: readonly (readonly [string, string])[];
  /** The ports each public input feeds, `graph['inputs_at']` of the expanded graph. */
  readonly inputsAt?: Readonly<Record<string, readonly string[]>>;
  /** The port each public output starts at, `graph['outputs_at']`. */
  readonly outputsAt?: Readonly<Record<string, string>>;
  /** A node's families, by node name; a node absent from it belongs to none. */
  readonly families?: Readonly<Record<string, readonly string[]>>;
  /**
   * The composition a node belongs to and the point of its grid, by node name.
   *
   * The site stays a *root* site — D2 reads a node's composition off `meta` alone, never off the
   * key — so a case declares `attn0` with `['decoder', {layer: 0n}]` and gets a layer split named
   * `decoder[layer<=0]` over it.
   */
  readonly generated?: Readonly<Record<string, readonly [string, Readonly<Record<string, PyValue>>]>>;
  /** The composition names, in the order `_expand` lists them. */
  readonly compositions?: readonly string[];
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
  // Every node the expansion resolves has a `meta` entry, so every node here does too: D2 reads
  // `meta[key]` unguarded, as the tools do, and a graph without one is a defect of the caller.
  const meta = new Map<string, ExpandedMeta>();
  for (const node of nodes) {
    const generated = extra.generated?.[node.name];
    meta.set(expandedKeyOf(site(node.name)), {
      families: new Set(extra.families?.[node.name] ?? []),
      composition:
        generated === undefined
          ? null
          : { name: generated[0], indices: new Map(Object.entries(generated[1])) },
    });
  }
  return {
    model: extra.model ?? record('{"interfaces": {"inputs": {}, "outputs": {}}}'),
    quantities: extra.quantities ?? new Map(),
    resolved,
    edges: (extra.edges ?? []).map(([from, to]) => edge(from, to)),
    domains,
    own,
    order: [...resolved.values()].map((node) => node.site),
    meta,
    compositions: extra.compositions ?? [],
    inputsAt: new Map(
      Object.entries(extra.inputsAt ?? {}).map(([name, at]) => [name, at.map((one) => port(one))]),
    ),
    outputsAt: new Map(
      Object.entries(extra.outputsAt ?? {}).map(([name, at]) => [name, port(at)]),
    ),
    tensorInstances: tensors,
    stateInstances: extra.states ?? [],
  };
}

/** `<node>.<port>` as an expanded graph names one endpoint. */
export function port(at: string): ExpandedPort {
  const dot = at.lastIndexOf('.');
  return { site: site(at.slice(0, dot)), port: at.slice(dot + 1) };
}

/** One value edge between two `<node>.<port>` endpoints, under no binding name. */
function edge(from: string, to: string): ExpandedEdge {
  const source = port(from);
  const target = port(to);
  return {
    from: source.site,
    fromPort: source.port,
    to: target.site,
    toPort: target.port,
    binding: `${from}->${to}`,
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

// --- the acceptance fixtures of `editor/tests/fixtures/` (feature 1.13, finding F8) ------------

/**
 * A model whose **interfaces** name a template instance.
 *
 * No document of the repository has one — `shieldstral-3b-composite`'s public input feeds `embed`
 * and its output comes from `lm_head`, so `inputs_at` and `outputs_at` reach a template instance
 * through no corpus document, and only the edges do. `fixture-one-template-instance` is the
 * smallest caller that reaches them: one instance of `decoder.causal_yarn` at two layers, the
 * template's own input as the public input and its own output as the public output.
 *
 * It was a string in this file until feature 1.13, and its answers had been read off
 * `tools/derive.py` by hand. It is a fixture document now (`editor/tests/fixtures/models/`), the
 * oracle runs the tools over it like any corpus document, and `test/parity/fixtures.test.ts` is
 * where its D1, its derived document and its `--validate` line are compared. What is asserted in
 * the unit layer is behaviour, not a recorded figure.
 */
export function oneInstance(): ExpandedGraph {
  return fixtureGraph('one-template-instance');
}

/**
 * The same caller with the three readings of a dtype selector told apart: the caller declares
 * `precision = f32`, the instance's argument supplies `f16`, and the role's default is `bf16`.
 * `_dtype` looks the selector's quantity up in the **caller's** quantities at every level, so the
 * tools answer `f32` for all eighteen tensors.
 */
export function oneInstanceDtype(): ExpandedGraph {
  return fixtureGraph('one-template-instance-dtype');
}

/** The base of the two primitives whose port shapes may not resolve, as a path the loader takes. */
export const SCRATCH_BASE = 'editor/tests/fixtures/scratch';

/**
 * The same base as files, for the callers that hand a library over rather than name a directory.
 *
 * The worker API takes texts, never paths (§5.3: "the core is pure — the UI reads files through
 * `Platform` and hands the core texts and trees"), so feature 1.11's suite needs the base as a
 * map; it is read off the fixture directory so that both readings are the same bytes.
 */
export const SCRATCH: Record<string, string> = Object.fromEntries(
  scratchFiles().map((path) => [path, readFileSync(`${repositoryRoot}/${path}`, 'utf8')]),
);

/** Every unit of the scratch base, by its repository-relative path. */
function scratchFiles(): string[] {
  const out: string[] = [];
  const walk = (relative: string): void => {
    for (const entry of readdirSync(`${repositoryRoot}/${relative}`, { withFileTypes: true })) {
      const path = `${relative}/${entry.name}`;
      if (entry.isDirectory()) walk(path);
      else if (entry.name.endsWith('.json')) out.push(path);
    }
  };
  walk(SCRATCH_BASE);
  return out.sort();
}

/** The path a fixture document is opened under, which is what its bases resolve against. */
export function fixturePath(name: string): string {
  return `editor/tests/fixtures/models/${name}.json`;
}

/** The text of one fixture document, as the file holds it. */
export function fixture(name: string): string {
  return readFileSync(`${repositoryRoot}/${fixturePath(name)}`, 'utf8');
}

/** One fixture document, expanded under the bases it declares. */
function fixtureGraph(name: string): ExpandedGraph {
  return derivationGraph(parse(fixture(name)), { schemas, library: fixtureLibrary(name) }).graph;
}

const fixtureLibraries = new Map<string, Library>();

/** The library one fixture document resolves from, gathered once per set of bases. */
function fixtureLibrary(name: string): Library {
  const path = fixturePath(name);
  const bases = [...basesOf(path, toPython(parse(fixture(name)))).bases];
  const key = bases.join('|');
  const held = fixtureLibraries.get(key);
  if (held !== undefined) return held;
  const gathered = loadLibrary(bases, { schemas, source: nodeSource(repositoryRoot) });
  fixtureLibraries.set(key, gathered);
  return gathered;
}

/**
 * The one-instance document whose **input** port's extent cites an argument nobody supplies: the
 * public input that feeds it has no element count, and the byte size the tools take of it is
 * unguarded, so the whole derivation raises (features 1.8c, 1.8e, 1.11).
 */
export const BLANK_MODEL = fixture('scratch-blank');

/** Its sibling, whose **output** port's extent does: there the tools write the blank. */
export const FED_MODEL = fixture('scratch-fed');
