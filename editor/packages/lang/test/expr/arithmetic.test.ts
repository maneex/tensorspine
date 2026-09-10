import { describe, expect, it } from 'vitest';

import {
  apply,
  compare,
  COMPARISONS,
  OPERATORS,
  PyIndexError,
  PyKeyError,
  PyTypeError,
  PyValueError,
  UNRESOLVED,
} from '../../src/expr/index.js';

// The algebra of §2.2 as `tools/expr.py` computes it. Every value asserted here was read from
// the tools themselves (`_apply` and `_COMPARISONS` over the same operands); the parity fixture
// re-reads them on every run of the oracle, and these cases are the ones that must never move
// silently — the four the feature's block names, and the rules of Python's numeric tower that
// JavaScript does not share.

describe('the operator table', () => {
  it('holds one implementation per operator of the algebra', () => {
    expect(Object.keys(OPERATORS).sort()).toEqual([
      'absolute',
      'add',
      'ceil_divide',
      'divide',
      'floor_divide',
      'max',
      'min',
      'modulo',
      'multiply',
      'negate',
      'subtract',
    ]);
  });

  it('answers UNRESOLVED for a name it does not carry, as `_apply_raw` falls through', () => {
    expect(apply('logarithm', [2n, 8n])).toBe(UNRESOLVED);
    expect(apply(null, [2n, 8n])).toBe(UNRESOLVED);
  });
});

describe('integers stay integers and divisions choose their rounding', () => {
  it('adds, multiplies and subtracts within the integers', () => {
    expect(apply('add', [1n, 2n])).toBe(3n);
    expect(apply('multiply', [3n, 4n])).toBe(12n);
    expect(apply('subtract', [1n, 2n])).toBe(-1n);
  });

  it('answers a float from `divide`, `4 / 2` included', () => {
    expect(apply('divide', [4n, 2n])).toBe(2);
    expect(apply('divide', [-7n, 2n])).toBe(-3.5);
  });

  it('keeps a float operand a float', () => {
    expect(apply('add', [1n, 2])).toBe(3);
    expect(apply('subtract', [1, 2n])).toBe(-1);
    expect(apply('multiply', [2, 3n])).toBe(6);
    expect(apply('floor_divide', [7.5, 2n])).toBe(3);
    expect(apply('modulo', [7.5, 2n])).toBe(1.5);
  });

  it('floors `floor_divide` towards minus infinity, on both sides of zero', () => {
    expect(apply('floor_divide', [7n, 2n])).toBe(3n);
    expect(apply('floor_divide', [-7n, 2n])).toBe(-4n);
    expect(apply('floor_divide', [7n, -2n])).toBe(-4n);
    expect(apply('floor_divide', [-7n, -2n])).toBe(3n);
    expect(apply('floor_divide', [-7.5, 2n])).toBe(-4);
  });

  it('ceils `ceil_divide` on negatives as `-((-a) // b)` computes it', () => {
    expect(apply('ceil_divide', [7n, 2n])).toBe(4n);
    expect(apply('ceil_divide', [-7n, 2n])).toBe(-3n);
    expect(apply('ceil_divide', [7n, -2n])).toBe(-3n);
    expect(apply('ceil_divide', [-7n, -2n])).toBe(4n);
    expect(apply('ceil_divide', [0n, 5n])).toBe(0n);
    expect(apply('ceil_divide', [7.5, 2n])).toBe(4);
  });

  it('gives `modulo` the sign of its divisor', () => {
    expect(apply('modulo', [7n, 2n])).toBe(1n);
    expect(apply('modulo', [-7n, 2n])).toBe(1n);
    expect(apply('modulo', [7n, -2n])).toBe(-1n);
    expect(apply('modulo', [-7n, -2n])).toBe(-1n);
    expect(apply('modulo', [-7.5, 2n])).toBe(0.5);
  });

  it('answers UNRESOLVED for every division by zero, integer or float', () => {
    for (const operator of ['divide', 'floor_divide', 'ceil_divide', 'modulo']) {
      expect(apply(operator, [1n, 0n])).toBe(UNRESOLVED);
      expect(apply(operator, [1, 0])).toBe(UNRESOLVED);
      expect(apply(operator, [1n, 0])).toBe(UNRESOLVED);
    }
  });

  it('keeps the operand `min` and `max` chose, so the first of two equals wins', () => {
    expect(apply('min', [3n, 1n, 2n])).toBe(1n);
    expect(apply('max', [3n, 1n, 2n])).toBe(3n);
    expect(apply('min', [1n, 1])).toBe(1n);
    expect(apply('min', [1, 1n])).toBe(1);
    expect(apply('max', [1n, 1])).toBe(1n);
    expect(apply('min', ['b', 'a'])).toBe('a');
  });
});

describe('a boolean is an integer, as it is in Python', () => {
  it('counts as one in arithmetic and answers an integer', () => {
    expect(apply('add', [true, true])).toBe(2n);
    expect(apply('multiply', [true, true])).toBe(1n);
    expect(apply('negate', [true])).toBe(-1n);
    expect(apply('absolute', [true])).toBe(1n);
  });

  it('is returned as it stands by `min`, which chooses an operand', () => {
    expect(apply('min', [true, 2n])).toBe(true);
  });

  it('compares equal to one', () => {
    expect(compare('equal', true, 1n)).toBe(true);
    expect(compare('less', true, 2n)).toBe(true);
  });
});

describe('an operand of the wrong kind is UNRESOLVED, never a guess', () => {
  it('refuses to add a string to a number', () => {
    expect(apply('add', ['a', 'b'])).toBe(UNRESOLVED);
    expect(apply('add', [1n, 'a'])).toBe(UNRESOLVED);
    expect(apply('subtract', ['a', 'b'])).toBe(UNRESOLVED);
    expect(apply('negate', ['a'])).toBe(UNRESOLVED);
    expect(apply('absolute', ['a'])).toBe(UNRESOLVED);
    expect(apply('min', [1n, 'a'])).toBe(UNRESOLVED);
  });

  it('repeats a string multiplied by an integer, which is Python’s operator', () => {
    expect(apply('multiply', ['ab', 3n])).toBe('ababab');
    expect(apply('multiply', [3n, 'ab'])).toBe('ababab');
    expect(apply('multiply', ['ab', -1n])).toBe('');
    expect(apply('multiply', [1.5, 'ab'])).toBe(UNRESOLVED);
  });

  it('refuses a record or a list where a number belongs', () => {
    expect(apply('add', [{ a: 1n }, 1n])).toBe(UNRESOLVED);
    expect(apply('multiply', [[1n], 2n])).toBe(UNRESOLVED);
  });
});

describe('the refusals that are not values', () => {
  it('raises where the tools raise: too few operands, an empty `min`', () => {
    expect(() => apply('subtract', [1n])).toThrow(PyIndexError);
    expect(() => apply('min', [])).toThrow(PyValueError);
    expect(() => compare('is_odd', 1n, 1n)).toThrow(PyKeyError);
  });

  it('sums and multiplies an empty list to the identity, as `sum` and the loop do', () => {
    expect(apply('add', [])).toBe(0n);
    expect(apply('multiply', [])).toBe(1n);
  });
});

describe('the comparison table', () => {
  it('holds one implementation per comparison operator', () => {
    expect(Object.keys(COMPARISONS).sort()).toEqual([
      'equal',
      'greater',
      'greater_or_equal',
      'less',
      'less_or_equal',
      'not_equal',
    ]);
  });

  it('compares an integer with a float exactly, beyond the doubles', () => {
    expect(compare('equal', 1n, 1)).toBe(true);
    expect(compare('less', 10n ** 17n + 1n, 1e17)).toBe(false);
    expect(compare('greater', 10n ** 17n + 1n, 1e17)).toBe(true);
    expect(compare('less', 3n, 3.5)).toBe(true);
    expect(compare('greater_or_equal', 3n, 3.5)).toBe(false);
  });

  it('answers false to every ordering of a NaN, and false to its equality', () => {
    for (const operator of ['less', 'greater', 'less_or_equal', 'greater_or_equal', 'equal']) {
      expect(compare(operator, Number.NaN, Number.NaN)).toBe(false);
    }
    expect(compare('not_equal', Number.NaN, Number.NaN)).toBe(true);
  });

  it('orders strings by code point', () => {
    expect(compare('less', 'a', 'b')).toBe(true);
    expect(compare('less', 'Z', 'a')).toBe(true);
    // U+FF3A comes after U+1D400 in UTF-16 code units and before it in code points.
    expect(compare('less', 'Ｚ', '\u{1D400}')).toBe(true);
  });

  it('holds two records equal member by member, across the numeric kinds', () => {
    expect(compare('equal', { a: 1n }, { a: 1 })).toBe(true);
    expect(compare('equal', { a: 1n }, { a: 2n })).toBe(false);
    expect(compare('equal', { a: 1n }, { a: 1n, b: 1n })).toBe(false);
    expect(compare('not_equal', { a: 1n }, { b: 1n })).toBe(true);
  });

  it('answers false rather than raising when the kinds differ, for equality only', () => {
    expect(compare('equal', 1n, '1')).toBe(false);
    expect(compare('not_equal', 1n, '1')).toBe(true);
    expect(() => compare('less', 1n, '1')).toThrow(PyTypeError);
    expect(() => compare('less', { a: 1n }, { a: 1n })).toThrow(PyTypeError);
  });
});
