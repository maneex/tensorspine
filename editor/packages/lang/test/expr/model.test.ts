import { describe, expect, it } from 'vitest';

import {
  conditionReferences,
  modelCondition,
  modelValue,
  quantityReferences,
  UNRESOLVED,
  type PyValue,
} from '../../src/expr/index.js';

// The model side of `tools/expr.py`, held to the cases the repository's own suite for this
// module asserts (`tests/run_expressions.py`) and to the rules its docstrings state: an
// undecidable guard is UNRESOLVED and never false (I7, V10), a comparison across types is
// undecidable rather than an exception, and a conditional is undecidable when its test is.

const quantities = (entries: Record<string, PyValue>): Map<string, PyValue> =>
  new Map(Object.entries(entries));

const layers = quantities({ layers: 30n });

/** `i mod 5 = 4`, the periodic pattern a composition guard is written with. */
const periodic = {
  compare: {
    operator: 'equal',
    left: { op: 'modulo', args: [{ index: 'i' }, { literal: 5n }] },
    right: { literal: 4n },
  },
};

describe('conditions', () => {
  it('decides a periodic guard at a bound index', () => {
    expect(modelCondition(periodic, layers, new Map([['i', 9n]]))).toBe(true);
    expect(modelCondition(periodic, layers, new Map([['i', 8n]]))).toBe(false);
  });

  it('answers UNRESOLVED over an unbound index, not false', () => {
    expect(modelCondition(periodic, layers)).toBe(UNRESOLVED);
    expect(modelCondition(periodic, layers, new Map())).toBe(UNRESOLVED);
  });

  it('keeps `not` of an undecidable condition undecidable', () => {
    expect(modelCondition({ not: periodic }, layers)).toBe(UNRESOLVED);
  });

  it('keeps `all` with one undecidable part undecidable', () => {
    expect(modelCondition({ all: [{ boolean: true }, periodic] }, layers)).toBe(UNRESOLVED);
  });

  it('is undecidable even when another part already decides the answer', () => {
    // The model side evaluates every part before looking for an unresolved one: `all` with a
    // false part is still undecidable, where the primitive side would stop at the false.
    expect(modelCondition({ all: [{ boolean: false }, periodic] }, layers)).toBe(UNRESOLVED);
    expect(modelCondition({ any: [{ boolean: true }, periodic] }, layers)).toBe(UNRESOLVED);
  });

  it('decides `all` and `any` when every part decides', () => {
    expect(modelCondition({ all: [{ boolean: true }, { boolean: true }] }, layers)).toBe(true);
    expect(modelCondition({ all: [{ boolean: true }, { boolean: false }] }, layers)).toBe(false);
    expect(modelCondition({ any: [{ boolean: false }, { boolean: true }] }, layers)).toBe(true);
    expect(modelCondition({ any: [{ boolean: false }, { boolean: false }] }, layers)).toBe(false);
  });

  it('answers UNRESOLVED to a comparison across types, not an exception', () => {
    const across = {
      compare: { operator: 'less', left: { literal: 'a' }, right: { literal: 1n } },
    };
    expect(modelCondition(across, layers)).toBe(UNRESOLVED);
  });

  it('answers UNRESOLVED when a quantity the comparison reads is absent', () => {
    const unknown = {
      compare: { operator: 'equal', left: { quantity: 'width' }, right: { literal: 1n } },
    };
    expect(modelCondition(unknown, layers)).toBe(UNRESOLVED);
  });
});

describe('expressions', () => {
  const conditional = {
    if: { compare: { operator: 'less', left: { index: 'i' }, right: { literal: 10n } } },
    then: { literal: 0.95 },
    else: { literal: 0n },
  };

  it('takes the branch the test chooses', () => {
    expect(modelValue(conditional, layers, new Map([['i', 3n]]))).toBe(0.95);
    expect(modelValue(conditional, layers, new Map([['i', 12n]]))).toBe(0n);
  });

  it('is UNRESOLVED when the test is', () => {
    expect(modelValue(conditional, layers)).toBe(UNRESOLVED);
  });

  it('reads a quantity, an index and a literal', () => {
    expect(modelValue({ quantity: 'layers' }, layers)).toBe(30n);
    expect(modelValue({ index: 'i' }, layers, new Map([['i', 4n]]))).toBe(4n);
    expect(modelValue({ literal: 'bf16' }, layers)).toBe('bf16');
    expect(modelValue({ quantity: 'width' }, layers)).toBe(UNRESOLVED);
    expect(modelValue({ index: 'i' }, layers)).toBe(UNRESOLVED);
  });

  it('propagates UNRESOLVED through an operator rather than computing with a guess', () => {
    const half = { op: 'floor_divide', args: [{ quantity: 'width' }, { literal: 2n }] };
    expect(modelValue(half, layers)).toBe(UNRESOLVED);
  });

  it('computes the corpus’s own derivation shape', () => {
    // `head_dim` of `llama3-8b`: `floor_divide(d, heads)`, with its literal value beside it.
    const document = quantities({ d: 4096n, heads: 32n });
    const derivation = { op: 'floor_divide', args: [{ quantity: 'd' }, { quantity: 'heads' }] };
    expect(modelValue(derivation, document)).toBe(128n);
  });

  it('answers UNRESOLVED for a node the grammar does not admit', () => {
    expect(modelValue({ unknown: 1n }, layers)).toBe(UNRESOLVED);
  });
});

describe('the reference walkers', () => {
  it('names the quantities an expression reads, through args and both branches', () => {
    const expression = {
      if: { compare: { operator: 'greater', left: { quantity: 'a' }, right: { literal: 0n } } },
      then: { quantity: 'b' },
      else: { op: 'add', args: [{ quantity: 'c' }, { literal: 1n }] },
    };
    expect([...quantityReferences(expression)].sort()).toEqual(['a', 'b', 'c']);
  });

  it('names the quantities a condition reads, through not, all and any', () => {
    const condition = {
      any: [
        { not: { compare: { operator: 'equal', left: { quantity: 'a' }, right: { literal: 1n } } } },
        { compare: { operator: 'less', left: { index: 'i' }, right: { quantity: 'b' } } },
      ],
    };
    expect([...conditionReferences(condition)].sort()).toEqual(['a', 'b']);
  });

  it('reads nothing from a boolean condition or a literal', () => {
    expect(conditionReferences({ boolean: true }).size).toBe(0);
    expect(quantityReferences({ literal: 1n }).size).toBe(0);
    expect(quantityReferences('not an expression').size).toBe(0);
  });
});
