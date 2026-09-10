import { describe, expect, it } from 'vitest';

import {
  argumentAt,
  argumentPresent,
  argumentReferences,
  expressionReferences,
  primitiveCondition,
  primitiveValue,
  PyTypeError,
  UNRESOLVED,
  type PyRecord,
} from '../../src/expr/index.js';

// The primitive side of `tools/expr.py`: a declaration's expressions against an instance's
// resolved arguments. The three answers that are not the model side's are asserted here as they
// stand — an absent argument is `None`, an undecidable comparison is false, and a comparison
// across types raises — together with the rule that makes the false safe: §4.3's `present` test,
// which the loader requires of every argument that may be absent.

/** An instance of `attention.dense` as the corpus writes one, records included. */
const attention: PyRecord = {
  heads: 32n,
  kv_heads: 8n,
  head_dim: 128n,
  mask: 'causal',
  rope: { theta: 500000.0, scaling: { kind: 'yarn', factor: 8.0 } },
  scale: UNRESOLVED,
};

describe('argument paths', () => {
  it('reads a field of a record through the dots', () => {
    expect(argumentAt(['rope', 'scaling', 'kind'], attention)).toBe('yarn');
    expect(argumentAt(['heads'], attention)).toBe(32n);
  });

  it('answers null for a path that resolves to nothing, at any depth', () => {
    expect(argumentAt(['window'], attention)).toBe(null);
    expect(argumentAt(['rope', 'scaling', 'beta_fast'], attention)).toBe(null);
    expect(argumentAt(['heads', 'inner'], attention)).toBe(null);
  });

  it('answers the sentinel an argument that was refused holds', () => {
    expect(argumentAt(['scale'], attention)).toBe(UNRESOLVED);
  });

  it('tells presence from value: `present` is a member test, not a value test', () => {
    expect(argumentPresent(['rope', 'scaling'], attention)).toBe(true);
    expect(argumentPresent(['window'], attention)).toBe(false);
    expect(argumentPresent(['scale'], attention)).toBe(true);
    expect(argumentPresent(['sink'], { sink: null })).toBe(true);
    expect(argumentAt(['sink'], { sink: null })).toBe(null);
  });
});

describe('expressions over resolved arguments', () => {
  it('computes a shape extent as a declaration writes one', () => {
    // `attention.dense`'s q slot: `heads * head_dim`.
    const extent = {
      op: 'multiply',
      args: [{ argument: 'heads' }, { argument: 'head_dim' }],
    };
    expect(primitiveValue(extent, attention)).toBe(4096n);
  });

  it('answers UNRESOLVED when an operand is absent, not zero', () => {
    const extent = { op: 'multiply', args: [{ argument: 'window' }, { literal: 2n }] };
    expect(primitiveValue(extent, attention)).toBe(UNRESOLVED);
  });

  it('answers UNRESOLVED when an operand was refused upstream', () => {
    const extent = { op: 'multiply', args: [{ argument: 'scale' }, { literal: 2n }] };
    expect(primitiveValue(extent, attention)).toBe(UNRESOLVED);
  });

  it('takes the branch of a conditional default', () => {
    // `kv_heads` defaults to `heads`; a guarded default takes a branch on another argument.
    const declared = {
      if: { compare: { operator: 'equal', left: { argument: 'mask' }, right: { literal: 'causal' } } },
      then: { argument: 'heads' },
      else: { literal: 1n },
    };
    expect(primitiveValue(declared, attention)).toBe(32n);
    expect(primitiveValue(declared, { ...attention, mask: 'none' })).toBe(1n);
  });
});

describe('conditions over resolved arguments', () => {
  const guarded = {
    all: [
      { present: 'window' },
      { compare: { operator: 'greater', left: { argument: 'window' }, right: { literal: 0n } } },
    ],
  };

  it('does not fire a guard whose argument is absent, and does not raise', () => {
    expect(primitiveCondition(guarded, attention)).toBe(false);
  });

  it('fires it when the argument is there', () => {
    expect(primitiveCondition(guarded, { ...attention, window: 4096n })).toBe(true);
    expect(primitiveCondition(guarded, { ...attention, window: 0n })).toBe(false);
  });

  it('stops at the first false part, so a later undecidable part is never reached', () => {
    // The primitive side short-circuits where the model side does not: `all` answers false here
    // even though its second part would raise.
    const across = {
      compare: { operator: 'less', left: { literal: 'a' }, right: { literal: 1n } },
    };
    expect(primitiveCondition({ all: [{ boolean: false }, across] }, attention)).toBe(false);
    expect(() => primitiveCondition({ all: [{ boolean: true }, across] }, attention)).toThrow(
      PyTypeError,
    );
  });

  it('answers false, not UNRESOLVED, to a comparison whose operand does not resolve', () => {
    const unresolved = {
      compare: { operator: 'equal', left: { argument: 'window' }, right: { literal: 1n } },
    };
    expect(primitiveCondition(unresolved, attention)).toBe(false);
    expect(primitiveCondition({ not: unresolved }, attention)).toBe(true);
  });

  it('decides a comparison on a record field, which is where `present_when` lives', () => {
    const yarn = {
      compare: {
        operator: 'equal',
        left: { argument: 'rope.scaling.kind' },
        right: { literal: 'yarn' },
      },
    };
    expect(primitiveCondition(yarn, attention)).toBe(true);
    expect(primitiveCondition(yarn, { ...attention, rope: { scaling: { kind: 'linear' } } })).toBe(
      false,
    );
  });
});

describe('argumentReferences', () => {
  it('names the paths a compare reads through not and any, and every present test', () => {
    expect(
      [
        ...argumentReferences({
          any: [
            {
              not: {
                compare: {
                  operator: 'equal',
                  left: { argument: 'rope.scaling.kind' },
                  right: { literal: 'yarn' },
                },
              },
            },
            { present: 'bias' },
          ],
        }),
      ].sort(),
    ).toEqual(['bias', 'rope.scaling.kind']);
  });

  it('reads through the expressions compared — a conditional’s test and both branches', () => {
    expect(
      [
        ...argumentReferences({
          compare: {
            operator: 'greater',
            left: {
              if: { present: 'a' },
              then: { argument: 'b' },
              else: { op: 'add', args: [{ argument: 'c' }, { literal: 1n }] },
            },
            right: { literal: 1n },
          },
        }),
      ].sort(),
    ).toEqual(['a', 'b', 'c']);
  });

  it('reads nothing from a boolean condition', () => {
    expect(argumentReferences({ boolean: true }).size).toBe(0);
    expect(expressionReferences({ literal: 1n }).size).toBe(0);
    expect(expressionReferences(1n).size).toBe(0);
  });
});
