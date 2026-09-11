import { describe, expect, it } from 'vitest';

import { d3, serialize, toJsonValue, type PyRecord, type PyValue } from '../../src/index.js';
import { PyValueError } from '../../src/expr/errors.js';
import { library } from '../describe/source.js';
import { record, syntheticGraph, tensorInstance } from './source.js';

// D3 (feature 1.8a), on the branches no corpus document reaches.
//
// The fifteen corpus documents and their 11 838 identity instances are the parity suite's; what
// they do not carry is here — an extent that does not resolve, one that is negative (R11), a
// sparsity unit whose axis is not an axis of its slot, a slot two units name, a `concat` location
// ("a `concat` that holds is written nowhere in the repository", feature 1.6c), and an identity
// instance no member of which resolved. Each is built as `_expand`'s own answer, since D3 is a
// function of that and of the library and of nothing else.

/** A parameter slot of `norm.scale`, whose default is `bf16` and whose sensitivity is full. */
function slot(extent: string, extra = ''): string {
  return `{"role": "norm.scale", "shape": {"axes": [
    {"name": "f", "axis": "model.width", "nature": "feature", "extent": ${extent}}]}${extra}}`;
}

/** The one row of a one-tensor graph. */
function only(answer: PyRecord): PyRecord {
  const tensors = answer['tensors'] as readonly PyValue[];
  expect(tensors).toHaveLength(1);
  return tensors[0] as PyRecord;
}

/** D3 over one declaration, one slot, one identity instance. */
function inventory(definition: PyValue, args: PyRecord = {}, dtype: PyValue | null = null): PyRecord {
  const graph = syntheticGraph(
    [{ name: 'n', primitive: 'norm.rms', definition, args }],
    [tensorInstance('w', [['n', 'weight']], { dtype })],
  );
  return d3(graph, library);
}

function definitionOf(parameters: string, extra = ''): PyValue {
  return record(`{"parameters": ${parameters}${extra}}`);
}

describe('one entry per identity instance', () => {
  it('writes the slot’s facts, its stored shape, its figures and its totals', () => {
    const answer = inventory(
      definitionOf(`{"weight": ${slot('{"literal": 4096}')}}`),
    );
    expect(only(answer)).toEqual({
      identity: 'w',
      members: ['n.weight'],
      primitive: 'norm.rms',
      slot: 'weight',
      role: 'norm.scale',
      sensitivity: 'full_precision',
      dtype: 'bf16',
      shape: [{ axis: 'model.width', extent: 4096n }],
      multiplicity: 1n,
      elements: 4096n,
      bytes: 8192n,
      tied: false,
    });
    expect(answer['totals']).toEqual({ tensors: 1n, elements: 4096n, bytes: 8192n, tied: 0n });
  });

  it('writes the members of a tied tensor once, and counts the tie', () => {
    const definition = definitionOf(`{"weight": ${slot('{"literal": 8}')}}`);
    const graph = syntheticGraph(
      [
        { name: 'embed', definition },
        { name: 'lm_head', definition },
      ],
      [tensorInstance('tied_embeddings', [
        ['embed', 'weight'],
        ['lm_head', 'weight'],
      ])],
    );
    const row = only(d3(graph, library));
    expect(row['members']).toEqual(['embed.weight', 'lm_head.weight']);
    expect(row['tied']).toBe(true);
    // "Of the first member; V15 makes the others compatible."
    expect(row['primitive']).toBe('embed');
    expect((d3(graph, library)['totals'] as PyRecord)['tied']).toBe(1n);
  });

  it('skips an identity instance no member of which resolved', () => {
    const graph = syntheticGraph([], [tensorInstance('w', [])]);
    const answer = d3(graph, library);
    expect(answer['tensors']).toEqual([]);
    expect(answer['totals']).toEqual({ tensors: 0n, elements: 0n, bytes: 0n, tied: 0n });
  });
});

describe('the figures', () => {
  it('takes a sub-byte dtype’s size as a real, and a whole width’s as an integer', () => {
    const definition = definitionOf(`{"weight": ${slot('{"literal": 6}')}}`);
    expect(only(inventory(definition, {}, 'fp4'))['bytes']).toBe(3);
    expect(only(inventory(definition, {}, 'i64'))['bytes']).toBe(48n);
  });

  it('leaves the count and the size blank where an extent does not resolve', () => {
    const row = only(inventory(definitionOf(`{"weight": ${slot('{"argument": "absent"}')}}`)));
    expect(row['shape']).toEqual([{ axis: 'model.width', extent: null }]);
    expect(row['elements']).toBeNull();
    expect(row['bytes']).toBeNull();
  });

  it('sums a blank as nothing, and promotes the total at the first real term', () => {
    const open = definitionOf(`{"weight": ${slot('{"argument": "absent"}')}}`);
    const whole = definitionOf(`{"weight": ${slot('{"literal": 6}')}}`);
    const graph = syntheticGraph(
      [
        { name: 'a', definition: open },
        { name: 'b', definition: whole },
        { name: 'c', definition: whole },
      ],
      [
        tensorInstance('wa', [['a', 'weight']]),
        tensorInstance('wb', [['b', 'weight']], { dtype: 'fp4' }),
        tensorInstance('wc', [['c', 'weight']], { dtype: 'bf16' }),
      ],
    );
    // 0 + None + 3.0 + 12 — `or 0` gives the integer zero, and the float arrives with `wb`.
    expect(d3(graph, library)['totals']).toEqual({
      tensors: 3n,
      elements: 12n,
      bytes: 15,
      tied: 0n,
    });
  });

  it('refuses a negative figure by name, as R11 requires', () => {
    const definition = definitionOf(`{"weight": ${slot('{"literal": -4}')}}`);
    expect(() => inventory(definition)).toThrowError(PyValueError);
    expect(() => inventory(definition)).toThrowError(
      "w: derived element count is -4 — negative or non-finite, which the validator's domains " +
        'should have refused (admitted upstream, a domain is missing)',
    );
  });

  it('restates a declared multiplicity and puts its storage axis first (§3.4)', () => {
    const declared = slot('{"literal": 5}', ', "multiplicity": {"argument": "copies"}');
    const row = only(
      inventory(definitionOf(`{"weight": ${declared}}`), record('{"copies": 3}')),
    );
    expect(row['shape']).toEqual([
      { axis: 'storage.multiplicity', extent: 3n },
      { axis: 'model.width', extent: 5n },
    ]);
    // "`elements` is the product of the shape's extents — the declared count is in it once, never
    // a further multiplier; `multiplicity` restates that count, descriptive."
    expect(row['multiplicity']).toBe(3n);
    expect(row['elements']).toBe(15n);
  });
});

describe('the sparsity unit (§4.5)', () => {
  function withUnits(units: string, extent = '{"literal": 4}'): PyRecord {
    const declared = slot(extent);
    return only(
      inventory(
        definitionOf(`{"weight": ${declared}}`, `, "sparsity": ${units}`),
        record('{"absent_is_not_here": 0}'),
      ),
    );
  }

  it('names the unit, its axis, the count and the fraction', () => {
    const row = withUnits(
      '[{"unit": {"parameters": ["weight"], "axis": "model.width"}, "activated_per_element": {"literal": 1}}]',
    );
    expect(row['sparsity']).toEqual({
      unit: 0n,
      axis: 'model.width',
      activated_per_element: 1n,
      units: 4n,
      // "A lookup table is `1 / vocabulary`": Python's true division, a real.
      activated_fraction: 0.25,
    });
  });

  it('leaves the extent and the fraction blank where the unit’s axis is not the slot’s', () => {
    const row = withUnits(
      '[{"unit": {"parameters": ["weight"], "axis": "moe.expert"}, "activated_per_element": {"literal": 2}}]',
    );
    expect(row['sparsity']).toEqual({
      unit: 0n,
      axis: 'moe.expert',
      activated_per_element: 2n,
      units: null,
      activated_fraction: null,
    });
  });

  it('leaves the fraction blank where the count does not resolve or the extent is zero', () => {
    const open = withUnits(
      '[{"unit": {"parameters": ["weight"], "axis": "model.width"}, "activated_per_element": {"argument": "absent"}}]',
    );
    expect((open['sparsity'] as PyRecord)['activated_per_element']).toBeNull();
    expect((open['sparsity'] as PyRecord)['activated_fraction']).toBeNull();
    const empty = withUnits(
      '[{"unit": {"parameters": ["weight"], "axis": "model.width"}, "activated_per_element": {"literal": 1}}]',
      '{"literal": 0}',
    );
    expect((empty['sparsity'] as PyRecord)['units']).toBe(0n);
    expect((empty['sparsity'] as PyRecord)['activated_fraction']).toBeNull();
  });

  it('lets the last unit naming the slot win, the loop not breaking', () => {
    const row = withUnits(
      '[{"unit": {"parameters": ["weight"], "axis": "model.width"}, "activated_per_element": {"literal": 1}},' +
        ' {"unit": {"parameters": ["other"], "axis": "moe.expert"}, "activated_per_element": {"literal": 9}},' +
        ' {"unit": {"parameters": ["weight"], "axis": "model.width"}, "activated_per_element": {"literal": 2}}]',
    );
    expect((row['sparsity'] as PyRecord)['unit']).toBe(2n);
    expect((row['sparsity'] as PyRecord)['activated_per_element']).toBe(2n);
  });

  it('says nothing where no unit names the slot', () => {
    const row = withUnits(
      '[{"unit": {"parameters": ["other"], "axis": "model.width"}, "activated_per_element": {"literal": 1}}]',
    );
    expect(row['sparsity']).toBeUndefined();
  });
});

describe('the evaluated location', () => {
  it('writes each of the four forms as the derived schema fixes it', () => {
    const declared = slot('{"literal": 2}');
    const definition = definitionOf(`{"weight": ${declared}}`);
    const graph = syntheticGraph(
      [
        { name: 'a', definition },
        { name: 'b', definition },
        { name: 'c', definition },
        { name: 'd', definition },
      ],
      [
        tensorInstance('one', [['a', 'weight']], { location: { tensor: 'w.bin' } }),
        tensorInstance('two', [['b', 'weight']], {
          location: {
            stack: {
              axis: 'model.width',
              dim: 0n,
              parts: [{ tensor: 'w.0' }, { tensor: 'w.1' }],
            },
          },
        }),
        tensorInstance('three', [['c', 'weight']], {
          location: {
            // "A `concat` that holds is written nowhere in the repository" (feature 1.6c).
            concat: {
              axis: 'model.width',
              dim: 0n,
              parts: [{ tensor: 'left' }, { tensor: 'right' }],
            },
          },
        }),
        tensorInstance('four', [['d', 'weight']], {
          location: {
            slice: { tensor: 'packed', axis: 'model.width', dim: 0n, offset: 8n, extent: 2n },
          },
        }),
      ],
    );
    const rows = d3(graph, library)['tensors'] as readonly PyRecord[];
    expect(rows.map((row) => row['location'])).toEqual([
      { tensor: 'w.bin' },
      { stack: { axis: 'model.width', dim: 0n, parts: [{ tensor: 'w.0' }, { tensor: 'w.1' }] } },
      { concat: { axis: 'model.width', dim: 0n, parts: [{ tensor: 'left' }, { tensor: 'right' }] } },
      { slice: { tensor: 'packed', axis: 'model.width', dim: 0n, offset: 8n, extent: 2n } },
    ]);
    // The member order is the derived schema's, and a JSON reading is what a consumer sees.
    expect(serialize(toJsonValue(rows[3] as PyValue))).toContain(
      '"slice": {\n      "tensor": "packed",\n      "axis": "model.width",\n      "dim": 0,\n' +
        '      "offset": 8,\n      "extent": 2\n    }',
    );
  });
});
