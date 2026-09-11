import { describe, expect, it } from 'vitest';

import {
  d2,
  identOf,
  pyRound,
  pyStr,
  streamAxis,
  valueGeometry,
  type PyRecord,
  type PyValue,
} from '../../src/index.js';
import { PyTypeError, PyValueError } from '../../src/expr/errors.js';
import { library } from '../describe/source.js';
import { oneInstance, record, syntheticGraph, type SyntheticExtra } from './source.js';

// D2 (feature 1.8c), on the branches no corpus document reaches.
//
// The fifteen corpus documents and their 5 189 values are the parity suite's: the merge, the
// insert, the joining input and its fragment alignment, both kinds of graph split, the exposed
// values and the peak. What they do not carry is here — a joining input the stream answers with
// no count or with two, a merge factor that does not resolve, a count above one, a value no public
// input reaches, a port shape that does not resolve on each side of the guard the tools wrote on
// one side only, a composition of two indices, a family that covers the whole graph, and an empty
// graph's peak. Each is built as `_expand`'s own answer, since D2 is a function of that, of the
// library and of nothing else.
//
// Beside them, the two interface paths that reach *through* a template instance: `inputs_at` and
// `outputs_at` (feature 1.8a's finding), read here from the same one-instance caller, whose
// answers were taken from `tools/derive.py` itself.

/** A port of the given shape, in the role whose default dtype is `bf16`. */
function port(extent: string | null, axis = 'model.width'): string {
  const shape =
    extent === null
      ? ''
      : `, "shape": {"axes": [{"name": "f", "axis": "${axis}", "nature": "feature",
           "extent": ${extent}}]}`;
  return `{"role": "activation.hidden"${shape}}`;
}

/** A primitive with one input and one output, and whatever else a case declares. */
function through(extra = '', extent: string | null = '{"literal": 4}'): PyValue {
  return record(`{"ports": {"inputs": {"input": ${port(extent)}},
    "outputs": {"output": ${port(extent)}}}${extra}}`);
}

/** The `d2` of a graph of `through`-like nodes. */
function inventory(
  nodes: readonly { name: string; definition?: PyValue; args?: PyRecord }[],
  extra: SyntheticExtra,
): PyRecord {
  return d2(
    syntheticGraph(
      nodes.map((node) => ({
        name: node.name,
        primitive: 'norm.rms',
        definition: node.definition ?? through(),
        ...(node.args === undefined ? {} : { args: node.args }),
      })),
      [],
      extra,
    ),
    library,
  );
}

/** The values of a product, by the identifier they are keyed under. */
function values(product: PyRecord): Map<string, PyRecord> {
  return new Map(
    (product['values'] as readonly PyValue[]).map((value) => [
      pyStr((value as PyRecord)['value'] as PyValue),
      value as PyRecord,
    ]),
  );
}

// A document with one public input `a` of kind `token`, introducing the stream of its own name.
const ONE_INPUT = `{"interfaces": {"inputs": {"a": {"kind": "token"}}, "outputs": {}}}`;

describe('a model whose interfaces name a template instance', () => {
  // The answers below are `tools/derive.py`'s own on that document, recorded by hand: no
  // repository document resolves a public interface *through* a template instance, so there is no
  // oracle for them until the acceptance fixtures of feature 1.13 (F8).
  const product = (): PyRecord => d2(oneInstance(), library);

  it('names the instance’s stream after the caller’s and counts one per element', () => {
    expect(product()['streams']).toEqual({ hidden: { kind: 'token', count: { hidden: 1 } } });
  });

  it('delivers the public input to the ports inside the instance', () => {
    const value = values(product()).get('hidden') as PyRecord;
    expect(value['to']).toEqual([
      'text/decoder/attn_n[layer=0].input',
      'text/decoder/attn_r[layer=0].a',
    ]);
    expect(value['elements']).toBe(64n);
    expect(value['bytes_per_element']).toBe(128n);
    expect(value['count']).toEqual({ hidden: 1 });
    expect(value['required_for']).toEqual(['hidden_out']);
    expect(value['required']).toBe(true);
  });

  it('exposes the value the instance’s own output port produces', () => {
    // The last layer's residual output feeds nothing and is listed all the same, because a public
    // output exposes a value whether or not an edge consumes it.
    const value = values(product()).get('text/decoder/ffn_r[layer=1].output') as PyRecord;
    expect(value['to']).toEqual([]);
    expect(value['exposed']).toEqual(['hidden_out']);
  });

  it('splits the instance’s composition under the instance’s prefix', () => {
    const splits = product()['graph_splits'] as readonly PyValue[];
    expect(
      splits.map((split) => [
        (split as PyRecord)['graph_split'],
        (split as PyRecord)['kind'],
        (split as PyRecord)['sizes'],
      ]),
    ).toEqual([
      ['text/decoder[layer<=0]', 'layer', [6n, 6n]],
      ['family:feed_forward', 'family', [11n, 1n]],
      ['family:norm', 'family', [10n, 2n]],
      ['family:sequence_operator', 'family', [8n, 4n]],
    ]);
    // The payload is per element as an integer and per invocation as a float, the count being one.
    expect(splits[0]).toEqual({
      graph_split: 'text/decoder[layer<=0]',
      kind: 'layer',
      sizes: [6n, 6n],
      payload: [
        {
          value: 'text/decoder/ffn_r[layer=0].output',
          bytes_per_element: 128n,
          count: { hidden: 1 },
        },
      ],
      bytes_per_element: 128n,
      bytes_per_invocation: { hidden: 128 },
    });
  });

  it('peaks at a node inside the instance', () => {
    expect(product()['peak_live']).toEqual({
      node: 'text/decoder/attn[layer=0]',
      values: [
        'hidden',
        'text/decoder/attn[layer=0].output',
        'text/decoder/attn_n[layer=0].output',
      ],
      bytes_per_element: 384n,
      bytes_per_invocation: { hidden: 384 },
    });
  });
});

describe('a joining input the stream cannot answer', () => {
  // V19 admits only a kind the stream carries independently of the input, so a valid document
  // never reaches either refusal; the tools raise them all the same, and so does this.
  const MERGED = `{"ports": {"inputs": {"input": ${port('{"literal": 4}')}},
    "outputs": {"output": ${port('{"literal": 4}')}}},
    "domain_transforms": [{"from_port": "input", "to_port": "output", "relation": "merge",
      "kind": "token", "factor": {"literal": 2}}]}`;

  /** `a` introduces a stream two values carry at different counts; `b` joins it at `kind`. */
  function joining(kind: string): SyntheticExtra {
    return {
      model: record(`{"interfaces": {"inputs": {
        "a": {"kind": "position"},
        "b": {"kind": "${kind}", "stream": "a"}}, "outputs": {}}}`),
      inputsAt: { a: ['p.input', 'q.input'], b: ['r.input'] },
      domains: { 'p.output': ['token', 'a'], 'q.output': ['token', 'a'] },
    };
  }

  const nodes = [{ name: 'p', definition: record(MERGED) }, { name: 'q' }, { name: 'r' }];

  it('refuses a kind the stream carries at two counts', () => {
    expect(() => inventory(nodes, joining('token'))).toThrow(PyValueError);
    expect(() => inventory(nodes, joining('token'))).toThrow(
      "input b: joins stream 'a' at kind token, where the stream carries 2 counts " +
        "[{'a': 0.5}, {'a': 1.0}] independently of it; a joining input takes the stream's one " +
        'count at its kind (§5.3)',
    );
  });

  it('takes a count another input delivers inside its own descendants', () => {
    // "(key not in descends **or** (key, port) in others)": a port another public input feeds
    // carries the stream's count whether or not the join reaches it. `b` joins at `q`, `s` is
    // downstream of `q`, and `s.input` is `a`'s delivery — the one count the stream carries
    // independently of `b`. Without the exemption the stream would carry none and refuse.
    const two = `{"ports": {"inputs": {"input": ${port('{"literal": 4}')},
      "other": ${port('{"literal": 4}')}}, "outputs": {"output": ${port('{"literal": 4}')}}}}`;
    const product = inventory([{ name: 'q' }, { name: 's', definition: record(two) }], {
      model: record(`{"interfaces": {"inputs": {
        "a": {"kind": "token"}, "b": {"kind": "token", "stream": "a"}}, "outputs": {}}}`),
      inputsAt: { a: ['s.input'], b: ['q.input'] },
      edges: [['q.output', 's.other']],
      domains: { 's.input': ['token', 'a'] },
    });
    expect(values(product).get('b')?.['count']).toEqual({ a: 1 });
  });

  it('refuses a kind the stream carries at none', () => {
    expect(() => inventory(nodes, joining('patch'))).toThrow(
      "input b: joins stream 'a' at kind patch, where the stream carries no count independently " +
        "of it; a joining input takes the stream's one count at its kind (§5.3)",
    );
  });
});

describe('a transform’s arithmetic', () => {
  /** One node with one transform, fed by the public input `a`. */
  function transformed(relation: string, factor = '{"literal": 2}', from = 'input'): PyRecord {
    const inputs = `{"input": ${port('{"literal": 4}')}, "other": ${port('{"literal": 4}')}}`;
    const definition = record(`{"ports": {"inputs": ${inputs},
      "outputs": {"output": ${port('{"literal": 4}')}}},
      "domain_transforms": [{"from_port": "${from}", "to_port": "output",
        "relation": "${relation}", "kind": "token", "factor": ${factor}}]}`);
    return inventory([{ name: 'n', definition }], {
      model: record(ONE_INPUT),
      inputsAt: { a: ['n.input', 'n.other'] },
      outputsAt: { out: 'n.output' },
    });
  }

  it('divides a merged stream by its factor', () => {
    expect(values(transformed('merge')).get('n.output')?.['count']).toEqual({ a: 0.5 });
  });

  it('divides by one where the factor does not resolve, and where it is zero', () => {
    // `f = _num(primitive_value(t['factor'], args)) or 1`: neither an unresolved factor nor a zero
    // one refuses, and the zero never reaches the division.
    expect(values(transformed('merge', '{"argument": "absent"}')).get('n.output')?.['count']).toEqual(
      { a: 1 },
    );
    expect(values(transformed('merge', '{"literal": 0}')).get('n.output')?.['count']).toEqual({
      a: 1,
    });
  });

  it('adds the inserted stream’s count to the instance’s own, above one where they meet', () => {
    // "An insert adds": the instance's own count is `other`'s, the inserted one is `input`'s, and
    // both are the same stream here — which is the only way a count above one arises at all.
    expect(values(transformed('insert')).get('n.output')?.['count']).toEqual({ a: 2 });
  });

  it('leaves the instance’s own count alone under an align', () => {
    expect(values(transformed('align')).get('n.output')?.['count']).toEqual({ a: 1 });
  });

  it('takes the instance’s own count from a port no transform reads', () => {
    // "The first input port that no transform takes its elements from": `input` is declared first
    // and is the transform's source, so the own count is `other`'s — a different stream here, so
    // a port that read the own count off the transformed port answers `a` where this answers `b`.
    const inputs = `{"input": ${port('{"literal": 4}')}, "other": ${port('{"literal": 4}')}}`;
    const definition = record(`{"ports": {"inputs": ${inputs},
      "outputs": {"output": ${port('{"literal": 4}')}}},
      "domain_transforms": [{"from_port": "input", "to_port": "output",
        "relation": "align", "kind": "token", "factor": {"literal": 1}}]}`);
    const product = inventory([{ name: 'n', definition }], {
      model: record(`{"interfaces": {"inputs": {"a": {"kind": "token"}, "b": {"kind": "token"}},
        "outputs": {}}}`),
      inputsAt: { a: ['n.input'], b: ['n.other'] },
      outputsAt: { out: 'n.output' },
    });
    expect(values(product).get('n.output')?.['count']).toEqual({ b: 1 });
  });
});

describe('a figure the graph leaves undetermined', () => {
  it('writes a blank for a produced value whose shape does not resolve', () => {
    const product = inventory([{ name: 'n', definition: through('', '{"argument": "absent"}') }], {
      model: record('{"interfaces": {"inputs": {}, "outputs": {}}}'),
      outputsAt: { out: 'n.output' },
    });
    const value = values(product).get('n.output') as PyRecord;
    expect(value['elements']).toBeNull();
    expect(value['bytes_per_element']).toBeNull();
    expect(value['shape']).toEqual([{ axis: 'model.width', extent: null }]);
    // A blank weighs the integer zero in every total, and the value still crosses a split.
    expect((product['peak_live'] as PyRecord)['bytes_per_element']).toBe(0n);
  });

  it('raises for a public input whose shape does not resolve', () => {
    // The tools guard the produced value's byte size with `if n is not None` and write the public
    // input's unguarded: `n * BYTES[dtype]` on a blank is a `TypeError` in Python's own words. No
    // port shape of the reference base cites an argument that may be absent, so the corpus cannot
    // reach it; a primitive declared in the editor can. Reproduced, not corrected — a finding.
    expect(() =>
      inventory([{ name: 'n', definition: through('', '{"argument": "absent"}') }], {
        model: record(ONE_INPUT),
        inputsAt: { a: ['n.input'] },
      }),
    ).toThrow(new PyTypeError("unsupported operand type(s) for *: 'NoneType' and 'int'"));
  });

  it('counts a port with no shape as no elements, and a public input’s as one', () => {
    const definition = record(`{"ports": {"inputs": {"input": ${port(null)}},
      "outputs": {"output": ${port(null)}}}}`);
    const product = inventory([{ name: 'n', definition }], {
      model: record(ONE_INPUT),
      inputsAt: { a: ['n.input'] },
      outputsAt: { out: 'n.output' },
    });
    expect(values(product).get('n.output')).toMatchObject({
      elements: 0n,
      bytes_per_element: 0n,
      shape: [],
    });
    expect(values(product).get('a')).toMatchObject({
      elements: 1n,
      bytes_per_element: 2n,
      shape: [],
    });
  });

  it('leaves a value no public input reaches without a count', () => {
    const product = inventory([{ name: 'n' }], { outputsAt: { out: 'n.output' } });
    const value = values(product).get('n.output') as PyRecord;
    expect(value['count']).toBeNull();
    expect(value['domain']).toBeNull();
    // A value with no count weighs nothing per invocation, though it has a size per element.
    expect(product['peak_live']).toEqual({
      node: 'n',
      values: ['n.output'],
      bytes_per_element: 8n,
      bytes_per_invocation: {},
    });
  });
});

describe('the structural graph splits', () => {
  const chain: SyntheticExtra = {
    model: record(ONE_INPUT),
    inputsAt: { a: ['x.input'] },
    edges: [
      ['x.output', 'y.input'],
      ['y.output', 'z.input'],
    ],
    compositions: ['decoder'],
  };
  const nodes = [{ name: 'x' }, { name: 'y' }, { name: 'z' }];

  it('splits a single-index composition after every value but the last', () => {
    const product = inventory(nodes, {
      ...chain,
      generated: {
        x: ['decoder', { layer: 0n }],
        y: ['decoder', { layer: 1n }],
        z: ['decoder', { layer: 2n }],
      },
    });
    const splits = product['graph_splits'] as readonly PyValue[];
    expect(splits.map((split) => (split as PyRecord)['graph_split'])).toEqual([
      'decoder[layer<=0]',
      'decoder[layer<=1]',
    ]);
    expect((splits[0] as PyRecord)['sizes']).toEqual([1n, 2n]);
    expect((splits[0] as PyRecord)['payload']).toEqual([
      { value: 'x.output', bytes_per_element: 8n, count: { a: 1 } },
    ]);
  });

  it('splits no composition of two indices', () => {
    // `len(comp[1]) == 1`: a grid has no layer prefix to close under. No corpus document has one
    // (feature 1.7's finding, F8), so this is the only place the branch is taken.
    const product = inventory(nodes, {
      ...chain,
      generated: {
        x: ['decoder', { layer: 0n, expert: 0n }],
        y: ['decoder', { layer: 0n, expert: 1n }],
        z: ['decoder', { layer: 1n, expert: 0n }],
      },
    });
    expect(product['graph_splits']).toEqual([]);
  });

  it('drops a family whose ancestor closure is the whole graph', () => {
    const product = inventory(nodes, {
      ...chain,
      compositions: [],
      families: { x: ['all'], y: ['all', 'tail'], z: ['all', 'tail'] },
    });
    // `family:all` closes over every node and is no split; `family:tail` closes over `y` and `z`
    // and their ancestor `x`, which is also every node. Neither is emitted.
    expect(product['graph_splits']).toEqual([]);
  });

  it('counts a value crossing to two consumers once', () => {
    const product = inventory([...nodes, { name: 'w' }], {
      model: record(ONE_INPUT),
      inputsAt: { a: ['x.input'] },
      edges: [
        ['x.output', 'y.input'],
        ['x.output', 'z.input'],
        ['y.output', 'w.input'],
      ],
      compositions: [],
      families: { x: ['first'] },
    });
    const split = (product['graph_splits'] as readonly PyValue[])[0] as PyRecord;
    expect(split['graph_split']).toBe('family:first');
    expect(split['payload']).toEqual([
      { value: 'x.output', bytes_per_element: 8n, count: { a: 1 } },
    ]);
    expect(split['bytes_per_element']).toBe(8n);
    expect(split['bytes_per_invocation']).toEqual({ a: 8 });
  });
});

describe('the peak of live values', () => {
  it('is empty for a graph with no node', () => {
    expect(inventory([], {})['peak_live']).toEqual({
      node: null,
      values: [],
      bytes_per_element: 0n,
      bytes_per_invocation: {},
    });
  });

  it('keeps the first node that reaches the maximum', () => {
    // `total > peak[0]`, not `>=`. At `x` the input's value and `x.output` are live, 16 bytes per
    // invocation; at `y` the input is spent and `y.output` has taken its place, 16 again — and the
    // peak stays where it was first reached. A port written `>=` would answer `z`.
    const product = inventory([{ name: 'x' }, { name: 'y' }, { name: 'z' }], {
      model: record(ONE_INPUT),
      inputsAt: { a: ['x.input'] },
      edges: [
        ['x.output', 'y.input'],
        ['x.output', 'z.input'],
      ],
    });
    const peak = product['peak_live'] as PyRecord;
    expect(peak['node']).toBe('x');
    expect(peak['values']).toEqual(['a', 'x.output']);
    expect(peak['bytes_per_invocation']).toEqual({ a: 16 });
  });

  it('holds a value a public output exposes to the end', () => {
    const product = inventory([{ name: 'x' }, { name: 'y' }, { name: 'z' }], {
      model: record(ONE_INPUT),
      inputsAt: { a: ['x.input'] },
      edges: [
        ['x.output', 'y.input'],
        ['y.output', 'z.input'],
      ],
      outputsAt: { first: 'x.output', second: 'x.output', third: 'z.output' },
    });
    // Two outputs expose one value, which is listed once carrying both their names.
    expect(values(product).get('x.output')?.['exposed']).toEqual(['first', 'second']);
    // `x.output` owes one consumption per exposure beside the edge's, so it is still live at the
    // last node — where the peak is, three values deep.
    expect((product['peak_live'] as PyRecord)['node']).toBe('z');
    expect((product['peak_live'] as PyRecord)['values']).toEqual([
      'x.output',
      'y.output',
      'z.output',
    ]);
  });
});

describe('the fragment alignment of a fragmented stream', () => {
  /** `a` is fragmented, and `n` merges it by `factor` — the value on the stream is 1/factor. */
  function aligned(factor: string): PyRecord {
    const definition = record(`{"ports": {"inputs": {"input": ${port('{"literal": 4}')}},
      "outputs": {"output": ${port('{"literal": 4}')}}},
      "domain_transforms": [{"from_port": "input", "to_port": "output", "relation": "merge",
        "kind": "token", "factor": ${factor}}]}`);
    return inventory([{ name: 'n' }, { name: 'm', definition }], {
      model: record(`{"interfaces": {"inputs": {"a": {"kind": "position", "fragmented": true}},
        "outputs": {}}}`),
      inputsAt: { a: ['n.input', 'm.input'] },
      outputsAt: { merged: 'm.output' },
      domains: { 'n.input': ['position', 'a'], 'm.output': ['token', 'a'] },
    });
  }

  it('is the least common multiple of the merge factors on the stream', () => {
    // One value at `{a: 1.0}` and one at `{a: 1/6}` give lcm(1, 6) = 6.
    expect((aligned('{"literal": 6}')['streams'] as PyRecord)['a']).toEqual({
      kind: 'position',
      count: { a: 1 },
      fragment_alignment: 6n,
    });
  });

  it('is the least common multiple of two factors, not the last of them', () => {
    // Voxtral's stream carries a stride of 2 and then 4 frames per token, whose lcm is the last
    // of them; two factors that do not divide each other are what tells the two readings apart.
    const merging = (factor: string): PyValue =>
      record(`{"ports": {"inputs": {"input": ${port('{"literal": 4}')}},
        "outputs": {"output": ${port('{"literal": 4}')}}},
        "domain_transforms": [{"from_port": "input", "to_port": "output", "relation": "merge",
          "kind": "token", "factor": {"literal": ${factor}}}]}`);
    const product = inventory(
      [{ name: 'm', definition: merging('2') }, { name: 'k', definition: merging('3') }],
      {
        model: record(`{"interfaces": {"inputs": {"a": {"kind": "position", "fragmented": true}},
          "outputs": {}}}`),
        inputsAt: { a: ['m.input', 'k.input'] },
        outputsAt: { halves: 'm.output', thirds: 'k.output' },
        domains: { 'm.output': ['token', 'a'], 'k.output': ['token', 'a'] },
      },
    );
    expect((product['streams'] as PyRecord)['a']).toMatchObject({ fragment_alignment: 6n });
  });

  it('is one where nothing on the stream is merged', () => {
    expect((aligned('{"literal": 1}')['streams'] as PyRecord)['a']).toMatchObject({
      fragment_alignment: 1n,
    });
  });

  it('rounds the reciprocal as Python rounds it', () => {
    // `max(1, round(1 / c))`: `round` is Python's, half to even, and a count above one aligns at
    // one rather than at zero.
    expect([pyRound(0.5), pyRound(1.5), pyRound(2.5), pyRound(-0.5), pyRound(8)]).toEqual([
      0n,
      2n,
      2n,
      0n,
      8n,
    ]);
    expect((aligned('{"literal": 3}')['streams'] as PyRecord)['a']).toMatchObject({
      fragment_alignment: 3n,
    });
  });

  it('is absent from a stream no public input fragments', () => {
    const product = inventory([{ name: 'n' }], {
      model: record(ONE_INPUT),
      inputsAt: { a: ['n.input'] },
      domains: { 'n.input': ['token', 'a'] },
    });
    expect((product['streams'] as PyRecord)['a']).toEqual({ kind: 'token', count: { a: 1 } });
  });
});

describe('the value-type label of a diagram', () => {
  // `view.py`'s conventions, restated in the core (F6). The fixture of `labels.test.ts` holds the
  // labels `--view` itself printed; what is here is the two branches no corpus count reaches.
  it('writes a count that is not a reciprocal as a multiplier', () => {
    expect(streamAxis(record('{"pixels": 0.3}'))).toBe('pixels×0.3');
    expect(streamAxis(record('{"tokens": 1.0, "pixels": 0.25}'))).toBe('tokens + pixels/4');
    expect(streamAxis(null)).toBe('');
  });

  it('writes an extent the graph left undetermined as Python writes it', () => {
    expect(
      valueGeometry(
        record(`{"dtype": "bf16", "count": {"a": 1.0},
          "shape": [{"axis": "model.width", "extent": null}]}`),
      ),
    ).toBe('bf16[a, model.width=None]');
  });

  it('writes a value with no count and no shape as its dtype alone', () => {
    expect(valueGeometry(record('{"dtype": "i32", "count": null, "shape": []}'))).toBe('i32[]');
  });
});

describe('a node of the graph the product cannot read', () => {
  it('raises where the tools raise, naming the node', () => {
    // `resolved[node]` on a key the expansion has not: a `KeyError` in the tools, named here by
    // the identifier of §5.2 rule 2, as feature 1.8a's `nodeAt` names it.
    const graph = syntheticGraph([{ name: 'n', definition: through() }], [], {
      model: record(ONE_INPUT),
      inputsAt: { a: ['absent.input'] },
    });
    expect(() => d2(graph, library)).toThrow(/'absent'/);
    expect(identOf({ prefix: '', key: graph.order[0]?.key as never })).toBe('n');
  });
});
