import { describe, expect, it } from 'vitest';

import {
  analyse,
  d3,
  derivationGraph,
  expandAnalysis,
  identOf,
  locatedValue,
  parse,
  pyStr,
  prefixedLocation,
  toPython,
  type EvaluatedLocation,
  type ExpandedGraph,
  type PyRecord,
} from '../../src/index.js';
import { corpus, library, schemas } from '../describe/source.js';

// The expansion of §5.1 (feature 1.8a): `derive._expand`, the analysis graph with every template
// instance expanded in place.
//
// The parity suite holds its nodes, edges and interfaces to D1's own on all fifteen corpus
// documents. What is here is what D1 cannot witness: the composition list, the stream renaming,
// the prefixed locations, and the one branch no corpus document takes — a template instance with
// no `weights_location_prefix` in a document that locates none of its identities, whose tensors
// are unlocated "like every other identity of that document, the template's names alone locating
// nothing" (§3.4).

const COMPOSITE = 'shieldstral-3b-composite';

/** The composite as the corpus writes it, expanded. */
function composite(): ExpandedGraph {
  return derivationGraph(parse(corpus(COMPOSITE)), { schemas, library }).graph;
}

/** The composite with every location and the instance's weights prefix removed. */
function unlocated(): ExpandedGraph {
  const document = JSON.parse(corpus(COMPOSITE)) as {
    instances: Record<string, Record<string, unknown>>;
    bindings: { parameters: Record<string, Record<string, unknown>> };
    compositions: Record<string, { bindings?: { parameters?: Record<string, Record<string, unknown>> } }>;
  };
  for (const binding of Object.values(document.bindings.parameters)) delete binding['location'];
  for (const composition of Object.values(document.compositions)) {
    for (const binding of Object.values(composition.bindings?.parameters ?? {})) {
      delete binding['location'];
    }
  }
  delete document.instances['text']?.['weights_location_prefix'];
  const answer = analyse(toPython(parse(JSON.stringify(document))), library);
  expect(answer.problems.map((problem) => problem.message)).toEqual([]);
  return expandAnalysis(answer);
}

/** The identifiers of the expanded graph's nodes. */
function nodes(graph: ExpandedGraph): string[] {
  return [...graph.resolved.values()].map((node) => identOf(node.site));
}

describe('a template instance, expanded in place', () => {
  it('lists the caller’s compositions and then the instance’s, prefixed', () => {
    // §5.2 rule 2: "an instance inside a template instance is prefixed by the instance".
    expect(composite().compositions).toEqual(['vision', 'text/decoder']);
  });

  it('puts the instance’s prefix on every node it stands for', () => {
    const inside = nodes(composite()).filter((name) => name.startsWith('text/'));
    expect(inside).toContain('text/decoder/attn[layer=0]');
    expect(inside.length).toBe(156);
    // The caller's own nodes keep their names, and the instance itself is no longer a node.
    expect(nodes(composite())).toContain('embed');
    expect(nodes(composite())).not.toContain('text');
  });

  it('gives the instance’s streams the names of the caller’s streams that feed it', () => {
    // The template declares one public input, `hidden`, introducing a stream of that name; the
    // composite feeds it from `tokens`, and the expansion renames it throughout the instance.
    const graph = composite();
    const inside = [...graph.domains.values()].filter((entry) =>
      identOf(entry.site).startsWith('text/'),
    );
    expect(inside.length).toBeGreaterThan(0);
    expect(new Set(inside.map((entry) => pyStr(entry.domain[1])))).toEqual(new Set(['tokens']));
    const own = [...graph.own.entries()].filter(([key]) => key.includes('text/'));
    expect(own.length).toBeGreaterThan(0);
    for (const [, domain] of own) expect(domain?.[1]).toBe('tokens');
  });

  it('locates the instance’s tensors under its weights prefix (§3.4)', () => {
    const rows = d3(composite(), library)['tensors'] as readonly PyRecord[];
    const first = rows.find((row) => row['identity'] === 'text/decoder.attn_n.weight[layer=0]');
    expect(first?.['location']).toEqual({
      tensor: 'language_model.model.layers.0.input_layernorm.weight',
    });
  });

  it('leaves them unlocated where the instance supplies no prefix and the document locates none', () => {
    const rows = d3(unlocated(), library)['tensors'] as readonly PyRecord[];
    expect(rows).toHaveLength(458);
    expect(rows.filter((row) => row['location'] !== undefined)).toHaveLength(0);
    const inside = rows.filter((row) => pyStr(row['identity'] ?? null).startsWith('text/'));
    expect(inside).toHaveLength(234);
  });
});

// A model whose *interfaces* name a template instance. No document of the repository has one —
// `shieldstral-3b-composite`'s public input feeds `embed` and its output comes from `lm_head`, so
// the expansion's `inputs_at` and `outputs_at` reach a template instance through no corpus
// document, and only the edges do. This is the smallest caller that reaches them: one instance,
// the template's own input as the public input and its own output as the public output. Its
// answers below were taken from `tools/derive.py` itself on the same document (feature 1.8a's
// finding), and a fixture for it belongs to the acceptance suite the plan postpones (F8).
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
function oneInstance(precision = '"bf16"', argument = '{"quantity": "precision"}'): ExpandedGraph {
  const text = ONE_INSTANCE.replace('PRECISION', precision).replace('ARGUMENT', argument);
  return derivationGraph(parse(text), { schemas, library }).graph;
}

describe('a model whose interfaces name a template instance', () => {
  it('resolves its public input into the instance and its output out of it', () => {
    const graph = oneInstance();
    expect(nodes(graph)).toHaveLength(12);
    expect(
      (graph.inputsAt.get('hidden') ?? []).map((port) => `${identOf(port.site)}.${pyStr(port.port)}`),
    ).toEqual(['text/decoder/attn_n[layer=0].input', 'text/decoder/attn_r[layer=0].a']);
    const output = graph.outputsAt.get('hidden_out');
    expect(output === undefined ? null : `${identOf(output.site)}.${pyStr(output.port)}`).toBe(
      'text/decoder/ffn_r[layer=1].output',
    );
    expect(d3(graph, library)['totals']).toEqual({
      tensors: 18n,
      elements: 73984n,
      bytes: 147968n,
      tied: 0n,
    });
  });

  it('selects a dtype inside the instance from the *caller’s* quantities', () => {
    // `_dtype` reads `graph['quantities']`, and `_expand` answers `dict(graph, …)` — the calling
    // document's quantities at every level. The template's scoped bindings select
    // `{"quantity": "precision"}`; here the caller's `precision` is `f32`, the template's own
    // resolves to `f16`, and the role's default is `bf16`, so the three readings are told apart:
    // the tools answer `f32` for all eighteen tensors, and so does this.
    const graph = oneInstance('"f32"', '{"literal": "f16"}');
    const rows = d3(graph, library)['tensors'] as readonly PyRecord[];
    expect(new Set(rows.map((row) => row['dtype']))).toEqual(new Set(['f32']));
    expect(rows).toHaveLength(18);
  });
});

describe('an evaluated location under a prefix', () => {
  const prefix = 'language_model.';

  it('prefixes the name of every form it can carry', () => {
    expect(prefixedLocation({ tensor: 'w' }, prefix)).toEqual({ tensor: 'language_model.w' });
    const stack: EvaluatedLocation = {
      stack: { axis: 'a.b', dim: 1n, parts: [{ tensor: 'w.0' }, { tensor: 'w.1' }] },
    };
    expect(prefixedLocation(stack, prefix)).toEqual({
      stack: {
        axis: 'a.b',
        dim: 1n,
        parts: [{ tensor: 'language_model.w.0' }, { tensor: 'language_model.w.1' }],
      },
    });
    const concat: EvaluatedLocation = {
      concat: { axis: 'a.b', dim: 0n, parts: [{ tensor: 'l' }, { tensor: 'r' }] },
    };
    expect(prefixedLocation(concat, prefix)).toEqual({
      concat: {
        axis: 'a.b',
        dim: 0n,
        parts: [{ tensor: 'language_model.l' }, { tensor: 'language_model.r' }],
      },
    });
    const slice: EvaluatedLocation = {
      slice: { tensor: 'packed', axis: 'a.b', dim: 0n, offset: 4n, extent: 2n },
    };
    expect(prefixedLocation(slice, prefix)).toEqual({
      slice: { tensor: 'language_model.packed', axis: 'a.b', dim: 0n, offset: 4n, extent: 2n },
    });
  });

  it('writes the members in the order the derived schema fixes', () => {
    const value = locatedValue({
      stack: { axis: 'a.b', dim: 2n, parts: [{ tensor: 'w' }] },
    }) as PyRecord;
    expect(Object.keys(value)).toEqual(['stack']);
    expect(Object.keys(value['stack'] as PyRecord)).toEqual(['axis', 'dim', 'parts']);
    const slice = locatedValue({
      slice: { tensor: 'p', axis: 'a.b', dim: 0n, offset: 1n, extent: 2n },
    }) as PyRecord;
    expect(Object.keys(slice['slice'] as PyRecord)).toEqual([
      'tensor',
      'axis',
      'dim',
      'offset',
      'extent',
    ]);
  });
});

describe('a document with no template instance', () => {
  it('is expanded into itself, node for node', () => {
    const graph = derivationGraph(parse(corpus('llama3-8b')), { schemas, library }).graph;
    expect(nodes(graph)).toHaveLength(195);
    expect(nodes(graph).every((name) => !name.includes('/'.repeat(2)))).toBe(true);
    expect(graph.compositions).toEqual(['decoder']);
    const output = graph.outputsAt.get('logits');
    expect(output === undefined ? null : identOf(output.site)).toBe('lm_head');
    const inputs = graph.inputsAt.get('tokens') ?? [];
    expect(inputs.map((port) => `${identOf(port.site)}.${pyStr(port.port)}`)).toEqual([
      'embed.tokens',
    ]);
  });
});
