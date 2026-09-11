import { describe, expect, it } from 'vitest';

import {
  BYTES,
  defaultDtype,
  elementsOf,
  numberOf,
  productShape,
  selectedDtype,
  sensitivityOf,
  sound,
  UNRESOLVED,
  widthOf,
  type PyRecord,
  type PyValue,
} from '../../src/index.js';
import { PyKeyError, PyOverflowError, PyValueError } from '../../src/expr/errors.js';
import { parse, storageShape, toPython } from '../../src/index.js';
import { library } from '../describe/source.js';

// The figures every product writes (feature 1.8a): the dtype widths, `_num`, `_sound`, `_shape`,
// `_elements` and `_dtype` of `tools/derive.py`.
//
// What the corpus reaches is the parity suite's business; what it does not is here. Six of the
// sixteen dtypes are written by a corpus document, so the ten others are exercised through the
// table; no corpus tensor has an unresolved extent or a negative one, so `None` and `_sound`'s
// refusal are exercised here; and the quantity form of a dtype selector is reached on both of its
// branches — the quantity that resolves to a dtype, and the one that does not.

/** A value as a document writes it, with its float-ness: `2` a whole number, `2.0` a real. */
function value(text: string): PyValue {
  return toPython(parse(text));
}

/** A record as a document writes it. */
function record(text: string): PyRecord {
  return value(text) as PyRecord;
}

describe('the dtype width table', () => {
  it('gives every dtype of the language a width, the sub-byte kinds as reals', () => {
    expect(Object.keys(BYTES)).toHaveLength(16);
    expect(widthOf('bool')).toBe(1n);
    expect(widthOf('i64')).toBe(8n);
    expect(widthOf('bf16')).toBe(2n);
    expect(widthOf('f32')).toBe(4n);
    expect(widthOf('f64')).toBe(8n);
    // "A `fp4`, `u4` or `i4` tensor's byte size is a float because the width is 0.5."
    expect(widthOf('u4')).toBe(0.5);
    expect(widthOf('i4')).toBe(0.5);
    expect(widthOf('fp4')).toBe(0.5);
  });

  it('raises the KeyError the tools raise for a name it has not', () => {
    // `BYTES[dtype]` — reachable only from a document whose V14 the validator did not refuse.
    expect(() => widthOf('f8e3m4')).toThrowError(PyKeyError);
    expect(() => widthOf('f8e3m4')).toThrowError("'f8e3m4'");
  });
});

describe('`_num`', () => {
  it('answers a number and nothing else', () => {
    expect(numberOf(4096n)).toBe(4096n);
    expect(numberOf(0.5)).toBe(0.5);
    expect(numberOf(-3n)).toBe(-3n);
    // `not isinstance(v, bool)`: Python's `bool` is an `int` and the tools exclude it all the same.
    expect(numberOf(true)).toBeNull();
    expect(numberOf(false)).toBeNull();
    expect(numberOf(UNRESOLVED)).toBeNull();
    expect(numberOf(null)).toBeNull();
    expect(numberOf('4096')).toBeNull();
    expect(numberOf([1n])).toBeNull();
    expect(numberOf({ a: 1n })).toBeNull();
  });
});

describe('`_sound`', () => {
  it('passes an absent figure and a non-negative one through', () => {
    expect(sound(null, 'element count', 'wq')).toBeNull();
    expect(sound(0n, 'element count', 'wq')).toBe(0n);
    expect(sound(4096n, 'element count', 'wq')).toBe(4096n);
    expect(sound(0.5, 'byte size', 'wq')).toBe(0.5);
  });

  it('names the identity and the figure when R11 is broken', () => {
    expect(() => sound(-1n, 'element count', 'wq[layer=3]')).toThrowError(PyValueError);
    expect(() => sound(-1n, 'element count', 'wq[layer=3]')).toThrowError(
      "wq[layer=3]: derived element count is -1 — negative or non-finite, which the " +
        "validator's domains should have refused (admitted upstream, a domain is missing)",
    );
    expect(() => sound(Number.NaN, 'byte size', 'wq')).toThrowError(
      'wq: derived byte size is nan — negative or non-finite,',
    );
    expect(() => sound(Number.POSITIVE_INFINITY, 'byte size', 'wq')).toThrowError(
      'wq: derived byte size is inf — negative or non-finite,',
    );
  });

  it('raises Python’s own OverflowError for an integer past the double range', () => {
    // `math.isfinite` converts to a float first, so the sign is never reached.
    expect(() => sound(10n ** 400n, 'element count', 'wq')).toThrowError(PyOverflowError);
    expect(() => sound(10n ** 400n, 'element count', 'wq')).toThrowError(
      'int too large to convert to float',
    );
    expect(() => sound(-(10n ** 400n), 'element count', 'wq')).toThrowError(PyOverflowError);
  });
});

describe('`_shape`', () => {
  const shape = record(`{"axes": [
    {"name": "out", "axis": "model.width", "nature": "feature",
     "extent": {"argument": "width"}},
    {"name": "in", "axis": "attention.projection", "nature": "feature",
     "extent": {"argument": "heads"},
     "factors": [{"name": "h", "axis": "attention.heads", "nature": "structural",
                  "extent": {"argument": "heads"}},
                 {"name": "d", "axis": "attention.head", "nature": "feature",
                  "extent": {"argument": "head_dim"}}]}
  ]}`);
  const args = record('{"width": 4096, "heads": 32, "head_dim": 128, "copies": 3}');

  it('writes the axis identity and the evaluated extent, with the declared factors', () => {
    expect(productShape(shape, args)).toEqual([
      { axis: 'model.width', extent: 4096n },
      {
        axis: 'attention.projection',
        extent: 32n,
        factors: [
          { axis: 'attention.heads', extent: 32n },
          { axis: 'attention.head', extent: 128n },
        ],
      },
    ]);
  });

  it('leads with the storage axis of a declared multiplicity (§3.4)', () => {
    const slot = { role: 'ffn.projection', shape, multiplicity: { argument: 'copies' } };
    expect(productShape(storageShape(slot as PyValue), args)).toEqual([
      { axis: 'storage.multiplicity', extent: 3n },
      { axis: 'model.width', extent: 4096n },
      expect.objectContaining({ axis: 'attention.projection' }),
    ]);
  });

  it('writes a blank where an extent does not resolve to a number', () => {
    expect(productShape(record('{"axes": [{"name": "x", "axis": "a.b", "nature": "feature", "extent": {"argument": "absent"}}]}'), {})).toEqual([
      { axis: 'a.b', extent: null },
    ]);
  });
});

describe('`_elements`', () => {
  const shape = record(`{"axes": [
    {"name": "a", "axis": "x.a", "nature": "feature", "extent": {"literal": 4}},
    {"name": "b", "axis": "x.b", "nature": "feature", "extent": {"literal": 8}}]}`);

  it('multiplies the extents, keeping Python’s integers', () => {
    expect(elementsOf(shape, {}, null)).toBe(32n);
  });

  it('applies the declared count once, after the shape’s own axes', () => {
    expect(elementsOf(shape, record('{"copies": 3}'), { argument: 'copies' })).toBe(96n);
  });

  it('answers nothing as soon as one extent does not resolve', () => {
    const open = record(
      '{"axes": [{"name": "a", "axis": "x.a", "nature": "feature", "extent": {"argument": "absent"}}]}',
    );
    expect(elementsOf(open, {}, null)).toBeNull();
    expect(elementsOf(shape, {}, { argument: 'absent' })).toBeNull();
  });

  it('takes a real extent as a real, as Python’s arithmetic does', () => {
    const half = record(
      '{"axes": [{"name": "a", "axis": "x.a", "nature": "feature", "extent": {"literal": 2.5}}]}',
    );
    expect(elementsOf(half, {}, null)).toBe(2.5);
  });
});

describe('`_dtype`', () => {
  const role = 'activation.hidden';

  it('answers the role’s default where the binding selects none (V14)', () => {
    expect(defaultDtype(library, role)).toBe('bf16');
    expect(sensitivityOf(library, role)).toBe('reduced');
    expect(selectedDtype(new Map(), library, null, role)).toBe('bf16');
  });

  it('answers a literal selector as it stands', () => {
    expect(selectedDtype(new Map(), library, 'f8e4m3', role)).toBe('f8e4m3');
  });

  it('answers a quantity that resolved to a dtype, and the default otherwise', () => {
    const quantities = new Map<string, PyValue>([
      ['precision', 'f16'],
      ['layers', 32n],
    ]);
    expect(selectedDtype(quantities, library, { quantity: 'precision' }, role)).toBe('f16');
    // "`v if isinstance(v, str) else cat['precision'][role]['default']`": a quantity that is not a
    // string, and a quantity the document does not declare at all, both fall back.
    expect(selectedDtype(quantities, library, { quantity: 'layers' }, role)).toBe('bf16');
    expect(selectedDtype(quantities, library, { quantity: 'absent' }, role)).toBe('bf16');
  });

  it('raises the KeyError an undeclared precision role raises', () => {
    expect(() => defaultDtype(library, 'no.such.role')).toThrowError(PyKeyError);
    expect(() => sensitivityOf(library, 'no.such.role')).toThrowError("'no.such.role'");
  });
});
