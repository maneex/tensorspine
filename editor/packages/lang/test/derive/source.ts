import { parse } from '../../src/json/index.js';
import {
  derivationGraph,
  expandedKeyOf,
  expandedPortKeyOf,
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
  type PyRecord,
  type PyValue,
} from '../../src/index.js';
import { library, schemas } from '../describe/source.js';

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

// A model whose *interfaces* name a template instance. No document of the repository has one —
// `shieldstral-3b-composite`'s public input feeds `embed` and its output comes from `lm_head`, so
// the expansion's `inputs_at` and `outputs_at` reach a template instance through no corpus
// document, and only the edges do. This is the smallest caller that reaches them: one instance of
// `decoder.causal_yarn` at two layers, the template's own input as the public input and its own
// output as the public output.
//
// Its answers are taken from `tools/derive.py` itself, run on this very document (feature 1.8a's
// finding, and D2 reads the same two paths). It belongs in `editor/tests/fixtures/` when feature
// 1.13 lands (F8), where the oracle would own the expectation instead of a hand-recorded one.
const ONE_INSTANCE = `{
  "schema": "tensorspine/2.0",
  "model": "one_template_instance",
  "primitive_libraries": [{"base": "../primitive-library/"}],
  "quantities": {
    "d": {"type": {"kind": "cardinality"}, "source": {"kind": "literal", "value": 64}},
    "layers": {"type": {"kind": "cardinality"}, "source": {"kind": "literal", "value": 2}},
    "heads": {"type": {"kind": "cardinality"}, "source": {"kind": "literal", "value": 4}},
    "kv_heads": {"type": {"kind": "cardinality"}, "source": {"kind": "literal", "value": 2}},
    "hd": {"type": {"kind": "cardinality"}, "source": {"kind": "literal", "value": 16}},
    "ffn": {"type": {"kind": "cardinality"}, "source": {"kind": "literal", "value": 128}},
    "eps": {"type": {"kind": "real"}, "source": {"kind": "literal", "value": 1e-05}},
    "precision": {"type": {"kind": "enum", "values": ["bf16", "f16", "f32"]},
                  "source": {"kind": "literal", "value": PRECISION}}
  },
  "constants": {},
  "instances": {
    "text": {
      "primitive": {"name": "decoder.causal_yarn", "version": "1.0.0"},
      "arguments": {
        "width": {"quantity": "d"}, "layers": {"quantity": "layers"},
        "heads": {"quantity": "heads"}, "kv_heads": {"quantity": "kv_heads"},
        "head_dim": {"quantity": "hd"}, "inner": {"quantity": "ffn"},
        "eps": {"quantity": "eps"}, "precision": ARGUMENT
      },
      "families": ["decoder"]
    }
  },
  "compositions": {},
  "bindings": {"values": {}, "parameters": {}, "constants": {}, "states": {}},
  "interfaces": {
    "inputs": {"hidden": {"to": [{"instance": {"kind": "root", "instance": "text"},
                                  "port": "hidden"}], "kind": "token"}},
    "outputs": {"hidden_out": {"from": {"instance": {"kind": "root", "instance": "text"},
                                        "port": "hidden_out"}, "generative": false}}
  }
}`;

/** The caller above, with the precision its quantity declares and the one its argument supplies. */
export function oneInstance(precision = '"bf16"', argument = '{"quantity": "precision"}'): ExpandedGraph {
  const text = ONE_INSTANCE.replace('PRECISION', precision).replace('ARGUMENT', argument);
  return derivationGraph(parse(text), { schemas, library }).graph;
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

// --- a scratch base, for the derivation branches the reference base cannot reach ----------------

/**
 * Two primitives, each leaving one port's extent to an argument nobody supplies, and the
 * one-instance documents that instantiate them.
 *
 * Feature 1.8c found the asymmetry and feature 1.8e reached it end to end: a *produced* value's
 * byte size is guarded (`n * BYTES[dtype] if n is not None else None`) and a *public input's* is
 * not, so a port shape that does not resolve raises `TypeError: unsupported operand type(s) for
 * *: 'NoneType' and 'int'` out of the whole derivation. No port shape of the reference base cites
 * an argument that may be absent, so no document of the repository reaches it — and a primitive
 * declared in the editor (D15, feature 3.3) can.
 *
 * It lives here rather than in one suite because two of them read it: the derivation's own, which
 * pins the raise and the blank beside it, and feature 1.11's worker suite, which is where the
 * decision that raise forced is proved — `derive` neither catches it nor blanks it, and the
 * worker answers it as a refusal instead of dying on it. Feature 1.13 (F8) is where the two
 * documents become editor-side fixtures the oracle owns.
 */
/** A port shape over `model.width`, with whatever extent the case wants. */
function shapeOf(extent: unknown): unknown {
  return {
    axes: [{ name: 'feature', axis: 'model.width', nature: 'feature', extent }],
  };
}

/** One primitive whose two ports carry those extents, with no parameter and no state. */
function portsOnly(input: unknown, output: unknown): unknown {
  const port = (extent: unknown) => ({
    shape: shapeOf(extent),
    domain: { kind: 'inherit', from: { self: true } },
    role: 'activation.hidden',
  });
  return {
    version: '1.0.0',
    arguments: {
      width: { type: { kind: 'cardinality' }, required: true, structural: true },
      hidden: { type: { kind: 'cardinality' }, required: false, structural: true },
    },
    ports: { inputs: { input: port(input) }, outputs: { output: port(output) } },
    parameters: {},
    constants: {},
    state_ports: {},
    effects: { reads: ['input'], writes: ['output'] },
    partition_options: [{ target: { any_axis: true }, communication: 'none' }],
  };
}

/** A base of two primitives, each leaving one port's extent to an argument nobody supplies. */
export const SCRATCH: Record<string, string> = {
  'scratch/base/primitive-library.json': JSON.stringify({
    schema: 'tensorspine-primitive-library-unit/2.0',
    kind: 'base',
    name: 'tensorspine.scratch.derive',
    definition: {
      primitive_library: 'tensorspine/scratch-derive',
      title: 'A base for the derivation tests',
    },
  }),
  // Its *input* port's extent is the optional argument: the public input that feeds it has no
  // element count, and the byte size the tools take of it is unguarded.
  'scratch/base/primitives/scratch/blank/1.0.0.json': JSON.stringify({
    schema: 'tensorspine-primitive-library-unit/2.0',
    kind: 'primitive',
    name: 'scratch.blank',
    definition: portsOnly({ argument: 'hidden' }, { argument: 'hidden' }),
  }),
  // Its *output* port's is: the value it produces has no element count either, and there the tools
  // write the blank the schema admits.
  'scratch/base/primitives/scratch/fed/1.0.0.json': JSON.stringify({
    schema: 'tensorspine-primitive-library-unit/2.0',
    kind: 'primitive',
    name: 'scratch.fed',
    definition: portsOnly({ argument: 'width' }, { argument: 'hidden' }),
  }),
};

/** A model of one instance, its input public and its output exposed. */
export function scratchModel(primitive: string): string {
  const bound = { kind: 'root', instance: 'blank' };
  return JSON.stringify({
    schema: 'tensorspine/2.0',
    model: 'scratch_blank',
    primitive_libraries: [{ base: '../primitive-library/' }, { base: '../../scratch/base/' }],
    quantities: {
      d: { type: { kind: 'cardinality' }, source: { kind: 'literal', value: 8 } },
    },
    constants: {},
    instances: {
      blank: {
        primitive: { name: primitive, version: '1.0.0' },
        arguments: { width: { quantity: 'd' } },
        families: ['scratch'],
      },
    },
    compositions: {},
    bindings: { values: {}, parameters: {}, constants: {}, states: {} },
    interfaces: {
      inputs: { in: { to: [{ instance: bound, port: 'input' }], kind: 'token' } },
      outputs: { out: { from: { instance: bound, port: 'output' }, generative: false } },
    },
  });
}

export const BLANK_MODEL = scratchModel('scratch.blank');
export const FED_MODEL = scratchModel('scratch.fed');
