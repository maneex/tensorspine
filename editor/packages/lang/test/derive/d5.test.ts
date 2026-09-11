import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import {
  d2,
  d3,
  d4,
  d5,
  derivationGraph,
  OPERATION_COUNTERS,
  parse,
  type PyRecord,
  type PyValue,
} from '../../src/index.js';
import { PyKeyError } from '../../src/expr/errors.js';
import { corpus, library, schemas } from '../describe/source.js';
import { repositoryRoot } from '../json/repository.js';
import { record, syntheticGraph, type SyntheticNode } from './source.js';

// D5 (feature 1.8d), on the branches no corpus document reaches, and on the claims
// `tests/run_costs.py` makes of the inventory rule.
//
// The fifteen corpus documents, their 889 applying corrections and their 446 sparsity units are
// the parity suite's: three of the four `per` values, two of the four statuses, a sparsity unit
// with an extent and one without, and the one document whose `operations` total is not exact.
// What they do not carry is here — a `per: invocation` correction, an `upper_bound` and a
// `lower_bound` correction and the two together, a correction whose expression does not resolve
// and one that resolves to a boolean, a unit whose axis extent is zero, a unit naming two slots
// and a slot naming the axis twice, a unit naming a slot the primitive has not, and a counter the
// analysis did not keep. Each is built as `_expand`'s own answer beside hand-built `totals`, since
// D5 is a function of the resolved nodes, of the three other products' totals and of the
// validator's counters, and of nothing else.
//
// The propagation table of §2.2 itself is `qualified.test.ts`'s: it is the specification's
// contract and is proved there against the specification, cell by cell. What is proved here is
// that D5 consults it where §7 says it does — once per `per`, over `exact` and the statuses of
// that unit's corrections.

/** A primitive declaring whatever cost entries and sparsity units a case needs. */
function primitive(declared: string): PyValue {
  return record(`{"parameters": {}, ${declared}}`);
}

/** The three products' `totals` and splits, as D5 reads them: figures and nothing else. */
const P3 = record('{"totals": {"tensors": 2, "elements": 96, "bytes": 192}}');
const P4 = record(
  '{"totals": {"identities": 1, "append_bytes_per_cached_position": 8, "bounded_bytes": 0,' +
    ' "fixed_bytes": 4}}',
);
const P2 = record(
  '{"graph_splits": [{"graph_split": "decoder[layer<=0]", "kind": "layer", "sizes": [1, 2],' +
    ' "payload": [{"value": "a.out", "bytes_per_element": 8, "count": {"tokens": 1.0}}],' +
    ' "bytes_per_element": 8, "bytes_per_invocation": {"tokens": 8.0}}]}',
);

/** The four counters the validator's first derivation keeps, all zero unless a case says so. */
function counters(supplied: Readonly<Record<string, bigint>> = {}): Map<string, PyValue> {
  const stats = new Map<string, PyValue>();
  for (const [, counter] of OPERATION_COUNTERS) stats.set(counter, supplied[counter] ?? 0n);
  stats.set('parameter_elements', 96n);
  return stats;
}

/** D5 over one or more declarations, with the totals above. */
function costs(
  nodes: readonly SyntheticNode[],
  stats: Map<string, PyValue> = counters(),
): PyRecord {
  return d5(syntheticGraph(nodes, []), P3, P4, P2, stats);
}

/** D5 over one declaration called `n`. */
function one(declared: string, args: PyRecord = {}, stats?: Map<string, PyValue>): PyRecord {
  return costs([{ name: 'n', primitive: 'p', definition: primitive(declared), args }], stats);
}

/** The rows of one member of the product. */
function rows(product: PyRecord, name: string): PyRecord[] {
  return (product[name] as readonly PyValue[]).map((row) => row as PyRecord);
}

/** The status of one `operations` figure. */
function status(product: PyRecord, per: string): PyValue {
  return ((product['operations'] as PyRecord)[per] as PyRecord)['status'] as PyValue;
}

describe('the shape of the product', () => {
  it('writes the six members in the tools’ own order', () => {
    expect(Object.keys(one('"logical_cost": []'))).toEqual([
      'parameters',
      'operations',
      'corrections',
      'sparsity',
      'state',
      'graph_splits',
    ]);
  });

  it('carries D3’s and D4’s totals with the status the inventory gives them', () => {
    const product = one('"logical_cost": []');
    expect(product['parameters']).toEqual({ elements: 96n, bytes: 192n, status: 'exact' });
    expect(product['state']).toEqual({
      append_bytes_per_cached_position: 8n,
      bounded_bytes: 0n,
      fixed_bytes: 4n,
      status: 'exact',
    });
    // "Everything derived from the inventory is exact" (derived guide §2): the two blocks say so
    // unconditionally, and no correction of any node can move them.
    expect(Object.keys(product['parameters'] as PyRecord)).toEqual([
      'elements',
      'bytes',
      'status',
    ]);
  });

  it('restates a split’s two figures and drops its payload', () => {
    const split = rows(one('"logical_cost": []'), 'graph_splits')[0] as PyRecord;
    expect(Object.keys(split)).toEqual([
      'graph_split',
      'bytes_per_element',
      'bytes_per_invocation',
    ]);
    expect(split['graph_split']).toBe('decoder[layer<=0]');
    expect(split['bytes_per_element']).toBe(8n);
    expect(split['bytes_per_invocation']).toEqual({ tokens: 8 });
  });

  it('writes the four `per` figures in the tools’ own order, whatever the corrections say', () => {
    const product = one(
      '"logical_cost": [{"expression": {"literal": 3}, "status": "exact", "per": "sequence"}]',
    );
    expect(Object.keys(product['operations'] as PyRecord)).toEqual([
      'element',
      'cached_position',
      'sequence',
      'invocation',
    ]);
    expect(Object.keys((product['operations'] as PyRecord)['element'] as PyRecord)).toEqual([
      'value',
      'status',
    ]);
  });

  it('reads a counter the analysis kept, and raises for one it did not', () => {
    const stats = counters({ ops_per_element: 512n, ops_per_cached_position: 16n });
    const operations = one('"logical_cost": []', {}, stats)['operations'] as PyRecord;
    expect((operations['element'] as PyRecord)['value']).toBe(512n);
    expect((operations['cached_position'] as PyRecord)['value']).toBe(16n);
    const short = counters();
    short.delete('ops_per_sequence');
    expect(() => one('"logical_cost": []', {}, short)).toThrowError(
      new PyKeyError("'ops_per_sequence'"),
    );
  });
});

describe('the corrections a primitive declares (§4.1)', () => {
  const THREE =
    '"logical_cost": [' +
    '{"when": {"present": "index"}, "expression": {"literal": 1}, "status": "exact",' +
    ' "per": "element"},' +
    '{"when": {"not": {"present": "index"}}, "expression": {"literal": 2}, "status": "exact",' +
    ' "per": "cached_position"},' +
    '{"expression": {"literal": 3}, "status": "exact", "per": "sequence"}]';

  it('keeps the declaration’s index, whatever a guard removed before it', () => {
    // "An ordered list of entries, each guarded by a condition over the arguments … every entry
    // whose condition holds contributes." The index is the entry's place in the primitive's list
    // and not a running count of the rows, so a reader lands on the declaration that produced it.
    const kept = rows(one(THREE, {}), 'corrections');
    expect(kept.map((row) => [row['entry'], row['per'], row['value']])).toEqual([
      [1n, 'cached_position', 2n],
      [2n, 'sequence', 3n],
    ]);
    const withIndex = rows(one(THREE, { index: { topk: 4n } }), 'corrections');
    expect(withIndex.map((row) => [row['entry'], row['per']])).toEqual([
      [0n, 'element'],
      [2n, 'sequence'],
    ]);
  });

  it('names the node, the primitive and the entry, in the tools’ own order', () => {
    const row = rows(one(THREE, {}), 'corrections')[0] as PyRecord;
    expect(Object.keys(row)).toEqual(['node', 'primitive', 'entry', 'value', 'status', 'per']);
    expect(row['node']).toBe('n');
    expect(row['primitive']).toBe('p');
  });

  it('walks the resolved nodes in their own order, each primitive’s entries in declaration order', () => {
    const product = costs([
      { name: 'a', primitive: 'p', definition: primitive(THREE), args: {} },
      { name: 'b', primitive: 'q', definition: primitive(THREE), args: { index: { topk: 1n } } },
    ]);
    expect(rows(product, 'corrections').map((row) => [row['node'], row['entry']])).toEqual([
      ['a', 1n],
      ['a', 2n],
      ['b', 0n],
      ['b', 2n],
    ]);
  });

  it('writes a blank where the expression does not resolve, and where it is a boolean', () => {
    // `_num` answers `None` for the sentinel and for a `bool`, which Python's `int` otherwise
    // swallows: the validator's counter adds a boolean (`ops[per] += True` is `+= 1`) where this
    // writes a blank. Two readings of one fact, each kept where its tool writes it.
    const blank = rows(
      one('"logical_cost": [{"expression": {"argument": "missing"}, "status": "exact",' +
        ' "per": "element"}]'),
      'corrections',
    )[0] as PyRecord;
    expect(blank['value']).toBeNull();
    const boolean = rows(
      one('"logical_cost": [{"expression": {"literal": true}, "status": "exact",' +
        ' "per": "invocation"}]'),
      'corrections',
    )[0] as PyRecord;
    expect(boolean['value']).toBeNull();
  });

  it('keeps a real value a real, as the derived document records it', () => {
    const row = rows(
      one('"logical_cost": [{"expression": {"literal": 2.5}, "status": "exact",' +
        ' "per": "element"}]'),
      'corrections',
    )[0] as PyRecord;
    expect(row['value']).toBe(2.5);
  });
});

/** One sparsity unit over a slot of `rows` rows: the lookup of §4.5, one unit per element. */
const SPARSE =
  '"sparsity": [{"unit": {"parameters": ["table"], "axis": "model.vocabulary"},' +
  ' "policy": {"element": true}, "activated_per_element": {"literal": 1},' +
  ' "union_per_invocation": {"expression": {"argument": "rows"}, "status": "upper_bound",' +
  ' "per": "invocation"}}]';

/** A slot laid out along `model.vocabulary`, with whatever extents a case declares. */
function slot(extents: readonly string[], axis = 'model.vocabulary'): string {
  const axes = extents
    .map(
      (extent) =>
        `{"name": "v", "axis": "${axis}", "nature": "vocabulary", "extent": ${extent}}`,
    )
    .join(', ');
  return `{"shape": {"axes": [${axes}]}, "role": "weight.embedding"}`;
}

describe('the status §2.2 gives each `operations` total', () => {
  /** One correction of the given status and `per`. */
  const correction = (per: string, declared: string): string =>
    `{"expression": {"literal": 1}, "status": "${declared}", "per": "${per}"}`;

  it('is `exact` for a unit no correction touches — the inventory’s own status', () => {
    const product = one(`"logical_cost": [${correction('element', 'upper_bound')}]`);
    expect(status(product, 'element')).toBe('upper_bound');
    for (const per of ['cached_position', 'sequence', 'invocation']) {
      expect(status(product, per), per).toBe('exact');
    }
  });

  it('carries a one-sided bound through, and cancels two opposite ones into an estimate', () => {
    // §2.2's sum row over `exact` and the unit's corrections: "estimate absorbs, opposite bounds
    // cancel into an estimate, one-sided bounds survive". The corpus reaches `exact` and
    // `estimate`; the two bounds and their cancellation are here.
    expect(status(one(`"logical_cost": [${correction('invocation', 'lower_bound')}]`), 'invocation'))
      .toBe('lower_bound');
    const both = one(
      `"logical_cost": [${correction('element', 'upper_bound')},` +
        `${correction('element', 'lower_bound')}]`,
    );
    expect(status(both, 'element')).toBe('estimate');
    const absorbed = one(
      `"logical_cost": [${correction('element', 'upper_bound')},` +
        `${correction('element', 'estimate')}]`,
    );
    expect(status(absorbed, 'element')).toBe('estimate');
  });

  it('combines the corrections of every node, not of one', () => {
    const product = costs([
      {
        name: 'a',
        primitive: 'p',
        definition: primitive(`"logical_cost": [${correction('sequence', 'upper_bound')}]`),
      },
      {
        name: 'b',
        primitive: 'q',
        definition: primitive(`"logical_cost": [${correction('sequence', 'lower_bound')}]`),
      },
    ]);
    expect(status(product, 'sequence')).toBe('estimate');
  });

  it('reads the status of an applying correction alone', () => {
    // A guard that removed an entry removed its status with it: the total is the sum over the rows
    // D5 wrote, which is what `by_per` is built from.
    const product = one(
      '"logical_cost": [{"when": {"present": "index"}, "expression": {"literal": 1},' +
        ' "status": "estimate", "per": "element"}]',
    );
    expect(rows(product, 'corrections')).toHaveLength(0);
    expect(status(product, 'element')).toBe('exact');
  });

  it('is not moved by a sparsity bound, which stands on its own row', () => {
    // §4.5's three quantities stay apart: "exact resident cost, upper-bounded worst-case transfer,
    // and estimated expected transfer". The union bound is an upper bound and it does not reach
    // the per-element operation count, which is exact at the activated fraction.
    const product = costs([
      {
        name: 'n',
        primitive: 'p',
        definition: record(`{"parameters": {"table": ${slot(['{"argument": "rows"}'])}}, ${SPARSE}}`),
        args: { rows: 8n },
      },
    ]);
    expect(rows(product, 'sparsity')).toHaveLength(1);
    for (const [per] of OPERATION_COUNTERS) expect(status(product, per), per).toBe('exact');
  });
});

describe('the sparsity units a primitive declares (§4.5)', () => {
  /** A primitive with one sparsity unit and the slots a case declares. */
  function sparse(parameters: string, unit = SPARSE): PyValue {
    return record(`{"parameters": ${parameters}, ${unit}}`);
  }

  /** The one sparsity row of a one-unit declaration. */
  function unit(parameters: string, args: PyRecord, declared = SPARSE): PyRecord {
    const product = costs([
      { name: 'n', primitive: 'p', definition: sparse(parameters, declared), args },
    ]);
    return rows(product, 'sparsity')[0] as PyRecord;
  }

  it('names the node, the primitive and the unit, in the tools’ own order', () => {
    const row = unit(`{"table": ${slot(['{"argument": "rows"}'])}}`, { rows: 8n });
    expect(Object.keys(row)).toEqual([
      'node',
      'primitive',
      'unit',
      'activated_per_element',
      'units',
      'activated_fraction',
      'union_per_invocation',
    ]);
    expect(row['unit']).toBe(0n);
    expect(row['activated_per_element']).toBe(1n);
    expect(row['units']).toBe(8n);
    expect(row['activated_fraction']).toBe(0.125);
    expect(row['union_per_invocation']).toEqual({ value: 8n, status: 'upper_bound' });
    expect(Object.keys(row['union_per_invocation'] as PyRecord)).toEqual(['value', 'status']);
  });

  it('takes the last matching axis of the last matching slot', () => {
    // Neither loop breaks. The loader requires every named slot to carry the unit's axis, so the
    // reference base's units find one extent twice; a slot that declares the axis twice, or two
    // slots of different extents, are decided by the last one either way.
    const two = unit(
      `{"table": ${slot(['{"literal": 4}'])}, "other": ${slot(['{"literal": 9}'])}}`,
      {},
      SPARSE.replace('["table"]', '["table", "other"]'),
    );
    expect(two['units']).toBe(9n);
    const twice = unit(`{"table": ${slot(['{"literal": 4}', '{"literal": 6}'])}}`, {});
    expect(twice['units']).toBe(6n);
  });

  it('keeps an earlier slot’s extent where a later one does not carry the axis', () => {
    // `extent` is declared once for the whole unit, above the loop over its slots, so a slot that
    // does not carry the axis leaves what an earlier one found. A reset per slot would answer a
    // blank here — the mutation the case is written for — and the loader refuses the
    // configuration, so nothing in the repository tells the two apart.
    const carried = unit(
      `{"table": ${slot(['{"literal": 4}'])}, "other": ${slot(['{"literal": 9}'], 'model.width')}}`,
      {},
      SPARSE.replace('["table"]', '["table", "other"]'),
    );
    expect(carried['units']).toBe(4n);
  });

  it('leaves the extent blank where no axis of the slot is the unit’s', () => {
    // `extent` starts at `None` and only a matching axis moves it. The loader refuses such a unit
    // ("unit axis '…' is not an axis of slot '…'"), so no gathered primitive reaches this; a
    // shape whose extent does not resolve reaches the same blank through `_num`.
    const elsewhere = unit(`{"table": ${slot(['{"literal": 4}'], 'model.width')}}`, {});
    expect(elsewhere['units']).toBeNull();
    expect(elsewhere['activated_fraction']).toBeNull();
    const unresolved = unit(`{"table": ${slot(['{"argument": "missing"}'])}}`, {});
    expect(unresolved['units']).toBeNull();
    expect(unresolved['activated_fraction']).toBeNull();
  });

  it('has no fraction where the extent is zero, rather than a division by zero', () => {
    // `(activated / extent) if activated is not None and extent else None` reads the extent's
    // *truth*, so a unit laid out along an axis of extent zero has no fraction at all.
    const zero = unit(`{"table": ${slot(['{"literal": 0}'])}}`, {});
    expect(zero['units']).toBe(0n);
    expect(zero['activated_fraction']).toBeNull();
    const real = unit(`{"table": ${slot(['{"literal": 0.0}'])}}`, {});
    expect(real['units']).toBe(0);
    expect(real['activated_fraction']).toBeNull();
  });

  it('has no fraction where the count per element does not resolve, extent or no extent', () => {
    const blank = unit(
      `{"table": ${slot(['{"literal": 4}'])}}`,
      {},
      SPARSE.replace('"activated_per_element": {"literal": 1}', '"activated_per_element": {"argument": "missing"}'),
    );
    expect(blank['activated_per_element']).toBeNull();
    expect(blank['units']).toBe(4n);
    expect(blank['activated_fraction']).toBeNull();
  });

  it('writes a blank bound where the bound’s expression does not resolve', () => {
    const row = unit(`{"table": ${slot(['{"literal": 4}'])}}`, {});
    expect(row['union_per_invocation']).toEqual({ value: null, status: 'upper_bound' });
  });

  it('raises for a unit naming a slot the primitive has not', () => {
    // `definition['parameters'][pname]` is subscripted here and read with `.get(pname)` in the
    // validator's own counter, which skips it. The loader's "unit parameter '…' is not a slot of
    // this primitive" refuses the difference away for every gathered primitive; a base read
    // without the loader would meet the two readings.
    expect(() => unit('{}', { rows: 8n })).toThrowError(new PyKeyError("'table'"));
  });

  it('keeps a unit whose arguments make it bite at nothing', () => {
    // A unit is a property of the primitive, not of the instance: it is written whether or not
    // the fraction resolves, which is what makes 210 of the corpus's 446 rows blank.
    const row = unit(`{"table": ${slot(['{"argument": "missing"}'])}}`, {});
    expect(row['activated_per_element']).toBe(1n);
    expect(row['units']).toBeNull();
  });
});

describe('a graph with nothing to correct', () => {
  it('answers empty lists and four exact totals', () => {
    const product = costs([{ name: 'n', primitive: 'p', definition: record('{"parameters": {}}') }]);
    expect(product['corrections']).toEqual([]);
    expect(product['sparsity']).toEqual([]);
    for (const [per] of OPERATION_COUNTERS) expect(status(product, per), per).toBe('exact');
  });

  it('answers the same over a graph with no node at all', () => {
    const product = d5(syntheticGraph([], []), P3, P4, P2, counters());
    expect(product['corrections']).toEqual([]);
    expect(product['sparsity']).toEqual([]);
    expect(rows(product, 'graph_splits')).toHaveLength(1);
  });
});

// `tests/run_costs.py`, reproduced: "D5, first derivation (§4.1, §4.5): operations per token
// follow from the parameter inventory — two per weight element consumed — scaled by the
// activated fraction of a sparse unit, plus the corrections a primitive declares."
//
// The tools' script reads `validate.semantic`'s counters; this reads D5's `operations`, which is
// where those counters land, so the claims are made of the product a consumer sees. The oracles
// are the script's own: the closed formulas it computes from the document's own literals, not a
// number written down here.

/** D5 of one corpus document, over the products the tools compute before it. */
function corpusCosts(name: string): { product: PyRecord; stats: ReadonlyMap<string, PyValue> } {
  const derivation = derivationGraph(parse(corpus(name)), { schemas, library });
  const p3 = d3(derivation.graph, library);
  const p4 = d4(derivation.graph, library);
  const p2 = d2(derivation.graph, library);
  return {
    product: d5(derivation.graph, p3, p4, p2, derivation.analysis.stats),
    stats: derivation.analysis.stats,
  };
}

/** One `operations` figure of a corpus document, as an integer. */
function operations(product: PyRecord, per: string): bigint {
  return ((product['operations'] as PyRecord)[per] as PyRecord)['value'] as bigint;
}

/** A literal quantity of a corpus document. */
function literal(name: string, quantity: string): bigint {
  const document = JSON.parse(corpus(name)) as {
    quantities: Record<string, { source: { value: number } }>;
  };
  return BigInt(document.quantities[quantity]?.source.value as number);
}

describe('the claims of tests/run_costs.py', () => {
  it('llama3-8b: ops per element = 2 × (elements − embedding table) + one embedding row', () => {
    // "A lookup being the limiting case of a sparsity unit": the embedding table's weights are
    // counted at one row per element, not at two operations per element of the whole table.
    const { product, stats } = corpusCosts('llama3-8b');
    const table = literal('llama3-8b', 'vocab') * literal('llama3-8b', 'd');
    const resident = stats.get('parameter_elements') as bigint;
    expect(operations(product, 'element')).toBe(
      2n * (resident - table) + 2n * literal('llama3-8b', 'd'),
    );
    expect(product['parameters']).toEqual({
      elements: resident,
      bytes: resident * 2n,
      status: 'exact',
    });
  });

  it('llama3-8b: the per-cached-position term is 32 layers × 4·32·128', () => {
    // `attention.dense`'s one correction, per cached position, at every one of the 32 layers.
    const { product } = corpusCosts('llama3-8b');
    expect(operations(product, 'cached_position')).toBe(32n * 4n * 32n * 128n);
    const corrections = (product['corrections'] as readonly PyValue[]).filter(
      (row) => (row as PyRecord)['per'] === 'cached_position',
    );
    expect(corrections).toHaveLength(32);
    expect((corrections[0] as PyRecord)['value']).toBe(4n * 32n * 128n);
  });

  it('llama4-scout: the routed experts count at top_k / experts', () => {
    // The script's independent oracle is "the closed formula the primitive library used to
    // declare": the dense part at two operations per weight element, plus the routed experts at
    // `top_k` of `experts` — 6·width·inner per expert per layer.
    const { product, stats } = corpusCosts('llama4-scout');
    const document = JSON.parse(corpus('llama4-scout')) as {
      quantities: Record<string, { source: { kind: string; value: number } }>;
      instances: Record<string, { arguments: Record<string, PyValue> }>;
      compositions: Record<
        string,
        { instances: Record<string, { arguments: Record<string, PyValue> }> }
      >;
    };
    const value = (written: PyValue): bigint => {
      const node = written as { literal?: number; quantity?: string };
      if (node.literal !== undefined) return BigInt(node.literal);
      return BigInt(document.quantities[node.quantity as string]?.source.value as number);
    };
    const moe = document.compositions['decoder']?.instances['moe']?.arguments as Record<
      string,
      PyValue
    >;
    const experts = value(moe['experts'] as PyValue);
    const topK = value(moe['top_k'] as PyValue);
    const width = value(moe['width'] as PyValue);
    const inner = value(moe['inner'] as PyValue);
    const layers = 48n;
    const routed = layers * experts * 3n * width * inner;
    const embed = document.instances['embed']?.arguments as Record<string, PyValue>;
    const table = value(embed['vocabulary'] as PyValue) * value(embed['width'] as PyValue);
    const resident = stats.get('parameter_elements') as bigint;
    const dense = 2n * (resident - routed - table) + 2n * value(embed['width'] as PyValue);
    const expected = dense + layers * 6n * width * inner * topK;
    const derived = operations(product, 'element');
    // The script's own tolerance: the counter is `int()` of a sum of floats, the fraction
    // `top_k / experts` being a real.
    const difference = derived > expected ? derived - expected : expected - derived;
    expect(difference, `${derived} against ${expected}`).toBeLessThanOrEqual(1n);
    expect(topK < experts).toBe(true);
  });

  it('llama4-scout: sparse operations are fewer than dense ones', () => {
    const { product, stats } = corpusCosts('llama4-scout');
    expect(operations(product, 'element')).toBeLessThan(
      2n * (stats.get('parameter_elements') as bigint),
    );
    // The routed experts' unit is what makes the difference, and D5 states it: one `moe` row per
    // routed layer, its fraction `top_k / experts`.
    const routed = (product['sparsity'] as readonly PyValue[]).filter(
      (row) => (row as PyRecord)['primitive'] === 'moe',
    );
    expect(routed.length).toBeGreaterThan(0);
    expect((routed[0] as PyRecord)['activated_fraction']).toBe(1 / 16);
  });

  it('shieldstral-3b and its composite derive the same figures', () => {
    const flat = corpusCosts('shieldstral-3b');
    const composite = corpusCosts('shieldstral-3b-composite');
    for (const counter of ['parameter_elements', 'ops_per_element', 'ops_per_cached_position']) {
      expect(composite.stats.get(counter), counter).toBe(flat.stats.get(counter));
    }
    for (const per of ['element', 'cached_position', 'sequence', 'invocation']) {
      expect(operations(composite.product, per), per).toBe(operations(flat.product, per));
    }
    expect(composite.product['parameters']).toEqual(flat.product['parameters']);
    expect(composite.product['state']).toEqual(flat.product['state']);
  });
});

// What D5 can leave blank, and where the derived schema admits a blank.
//
// Feature 1.8b found the tools writing `writer: None` where `value_reference` requires a string,
// and feature 1.8c the inverse asymmetry in D2 — a public input's value taking `n * BYTES[dtype]`
// unguarded, which raises instead of writing a blank. The brief asks whether D5 carries either
// shape. It carries neither, and the claim is checked rather than asserted: every place D5 can
// leave undetermined is read off the product built over a graph where nothing resolves, and every
// place the schema admits `null` is read off `schemas/tensorspine-derived.schema.json` — the two
// sets are compared, in both directions.
//
// The four figures D5 hands on from another product (`parameters`, `state`, a split's
// `bytes_per_element` and `bytes_per_invocation`) are the places the schema does *not* make
// nullable, and none of them can be blank: each is a `sum(… or 0 …)` in the product it comes from,
// which is a number whatever the graph says. That is D3's, D4's and D2's claim and their suites
// carry it; what is stated here is that D5 adds no blank of its own beside them.

/** Every place of the derived schema's `d5` that admits `null`, as `a/*\/b` shape paths. */
function nullablePlaces(): string[] {
  const schema = JSON.parse(
    readFileSync(`${repositoryRoot}/schemas/tensorspine-derived.schema.json`, 'utf8'),
  ) as { $defs: Record<string, Record<string, unknown>> };
  const found: string[] = [];
  const walk = (node: Record<string, unknown>, at: string, seen: readonly string[]): void => {
    const ref = node['$ref'];
    if (typeof ref === 'string') {
      const local = /^#\/\$defs\/(?<name>[^/]+)$/u.exec(ref)?.groups?.['name'];
      // A `$ref` out of the file names another schema's definition: those are the identifiers and
      // the enumerations, none of which admits a blank.
      if (local === undefined || seen.includes(local)) return;
      walk(schema.$defs[local] as Record<string, unknown>, at, [...seen, local]);
      return;
    }
    const type = node['type'];
    if (Array.isArray(type) && type.includes('null')) found.push(at);
    const properties = node['properties'] as Record<string, Record<string, unknown>> | undefined;
    for (const [name, child] of Object.entries(properties ?? {})) {
      walk(child, at === '' ? name : `${at}/${name}`, seen);
    }
    const items = node['items'] as Record<string, unknown> | undefined;
    if (items !== undefined) walk(items, `${at}/*`, seen);
  };
  walk(schema.$defs['d5'] as Record<string, unknown>, '', ['d5']);
  return found.sort();
}

/** Every place of a product whose value is `null`, with array indices written `*`. */
function blankPlaces(value: PyValue, at = ''): string[] {
  if (value === null) return [at];
  if (Array.isArray(value)) {
    return (value as readonly PyValue[]).flatMap((one) => blankPlaces(one, `${at}/*`));
  }
  if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
    return Object.entries(value as PyRecord).flatMap(([name, child]) =>
      blankPlaces(child, at === '' ? name : `${at}/${name}`),
    );
  }
  return [];
}

describe('the blanks the derived schema admits (for feature 1.8e)', () => {
  /** D5 over a graph where every expression the product evaluates is undetermined. */
  function undetermined(): PyRecord {
    const definition = record(
      `{"parameters": {"table": ${slot(['{"argument": "missing"}'])}},` +
        ' "logical_cost": [{"expression": {"argument": "missing"}, "status": "estimate",' +
        ' "per": "invocation"}],' +
        ` ${SPARSE.replace('"activated_per_element": {"literal": 1}', '"activated_per_element": {"argument": "missing"}')}}`,
    );
    return costs([{ name: 'n', primitive: 'p', definition, args: {} }]);
  }

  it('leaves a blank only where the schema admits one', () => {
    const admitted = new Set(nullablePlaces());
    expect([...admitted].sort()).toEqual([
      'corrections/*/value',
      'operations/cached_position/value',
      'operations/element/value',
      'operations/invocation/value',
      'operations/sequence/value',
      'sparsity/*/activated_fraction',
      'sparsity/*/activated_per_element',
      'sparsity/*/union_per_invocation/value',
      'sparsity/*/units',
    ]);
    const blanks = blankPlaces(undetermined());
    expect(blanks.length).toBeGreaterThan(0);
    for (const place of blanks) expect(admitted.has(place), place).toBe(true);
  });

  it('reaches every blank the schema admits but the four figures it hands on', () => {
    // The other direction: a place the schema declares nullable that D5 could never leave blank
    // would be a field of the D4 shape in reverse — a nullable the emitter does not use. The four
    // `operations` values are those: they are `--validate`'s own counters, which are integers
    // whatever the graph says, and the schema is wider than the emitter there.
    const reached = new Set(blankPlaces(undetermined()));
    const unreached = nullablePlaces().filter((place) => !reached.has(place));
    expect(unreached).toEqual([
      'operations/cached_position/value',
      'operations/element/value',
      'operations/invocation/value',
      'operations/sequence/value',
    ]);
  });

  it('never writes a blank where a string is required, and never raises on an unguarded figure', () => {
    // D4's `writer: None` has no counterpart here: every string D5 writes is an identifier it
    // builds or a value the declaration carries. And no arithmetic of D5 is unguarded — the one
    // division it does is behind the extent's own truth — so a shape that does not resolve is a
    // blank and not a `TypeError`.
    const product = undetermined();
    for (const row of rows(product, 'corrections')) {
      expect(typeof row['node']).toBe('string');
      expect(typeof row['primitive']).toBe('string');
      expect(typeof row['status']).toBe('string');
      expect(typeof row['per']).toBe('string');
    }
    for (const row of rows(product, 'sparsity')) {
      expect(typeof row['node']).toBe('string');
      expect(typeof (row['union_per_invocation'] as PyRecord)['status']).toBe('string');
    }
    expect(product['parameters']).toEqual({ elements: 96n, bytes: 192n, status: 'exact' });
  });
});
