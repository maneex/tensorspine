import { describe, expect, it } from 'vitest';

import {
  conditionalStatus,
  flipped,
  propagate,
  propagationOf,
  PROPAGATIONS,
  roundQualified,
  STATUSES,
  sumStatus,
  type Status,
} from '../../src/index.js';
import { PyOverflowError, PyValueError } from '../../src/expr/errors.js';

// The propagation table of the specification's §2.2, cell by cell.
//
// This is the feature's first test and it is written against the *specification*, not against the
// tools: `derive._status` implements one row of the table — the `add`, `multiply`, `min`, `max`
// row, which is D5's only combination — and the corpus reaches two of its cells (`E,E` and one
// `estimate` correction in `voxtral-realtime`). Every other cell of every row is here because
// §2.2 is the contract, and a contract is proved where it is written and not where a document
// happens to land.
//
// The table is transcribed below as data, straight from §2.2, before any implementation is
// consulted: the four statuses as `E`, `U`, `L`, `S`, the six columns as the ordered pairs they
// stand for, and the row-independent rule "any operation with an `S` operand yields `S`" as its
// own seven pairs. `propagate` derives the two binary rows from the n-ary one by flipping the
// second operand — that is the mechanism, and this suite checks the mechanism against all sixteen
// pairs of each row rather than against itself.

/** §2.2's shorthand: `E` is exact, `U` an upper bound, `L` a lower bound, `S` an estimate. */
const E: Status = 'exact';
const U: Status = 'upper_bound';
const L: Status = 'lower_bound';
const S: Status = 'estimate';

/** Every ordered pair of statuses, as `<a>,<b>`. */
const PAIRS: readonly (readonly [Status, Status])[] = STATUSES.flatMap((left) =>
  STATUSES.map((right) => [left, right] as const),
);

/** A pair as a key: `E,U`. */
const short: Readonly<Record<Status, string>> = {
  exact: 'E',
  upper_bound: 'U',
  lower_bound: 'L',
  estimate: 'S',
};
const key = ([a, b]: readonly [Status, Status]): string => `${short[a]},${short[b]}`;

// | operation | E,E | E,U / U,E | E,L / L,E | U,U | L,L | U,L / L,U |
// | `add`, `multiply`, `min`, `max` | E | U | L | U | L | S |
//
// plus "any operation with an `S` operand yields `S`".
const SUM_ROW: Readonly<Record<string, Status>> = {
  'E,E': E,
  'E,U': U,
  'U,E': U,
  'E,L': L,
  'L,E': L,
  'U,U': U,
  'L,L': L,
  'U,L': S,
  'L,U': S,
  'E,S': S,
  'S,E': S,
  'U,S': S,
  'S,U': S,
  'L,S': S,
  'S,L': S,
  'S,S': S,
};

// | `subtract` a−b | E | a−U: L · U−b: U | a−L: U · L−b: L | S | S | U−L: U · L−U: L |
const SUBTRACT_ROW: Readonly<Record<string, Status>> = {
  'E,E': E,
  'E,U': L,
  'U,E': U,
  'E,L': U,
  'L,E': L,
  'U,U': S,
  'L,L': S,
  'U,L': U,
  'L,U': L,
  'E,S': S,
  'S,E': S,
  'U,S': S,
  'S,U': S,
  'L,S': S,
  'S,L': S,
  'S,S': S,
};

// | `divide` a/b, `floor_divide`, `ceil_divide` (b > 0) | E | a/U: L · U/b: U | a/L: U · L/b: L |
// | S | S | U/L: U · L/U: L |
//
// The same six cells as `subtract`, written again from §2.2's own row rather than shared with it:
// two rows of a table that agree are two claims, and a suite that shared one object would prove
// one of them twice.
const DIVIDE_ROW: Readonly<Record<string, Status>> = {
  'E,E': E,
  'E,U': L,
  'U,E': U,
  'E,L': U,
  'L,E': L,
  'U,U': S,
  'L,L': S,
  'U,L': U,
  'L,U': L,
  'E,S': S,
  'S,E': S,
  'U,S': S,
  'S,U': S,
  'L,S': S,
  'S,L': S,
  'S,S': S,
};

describe('§2.2, the propagation table', () => {
  it('names exactly the four statuses of O0.5', () => {
    expect([...STATUSES]).toEqual(['exact', 'upper_bound', 'lower_bound', 'estimate']);
  });

  for (const operation of ['add', 'multiply', 'min', 'max']) {
    it(`propagates \`${operation}\` through every cell of its row`, () => {
      for (const pair of PAIRS) {
        expect(propagate(operation, pair), `${operation} ${key(pair)}`).toBe(
          SUM_ROW[key(pair)] as Status,
        );
      }
    });
  }

  it('propagates `subtract` through every cell of its row', () => {
    for (const pair of PAIRS) {
      expect(propagate('subtract', pair), `subtract ${key(pair)}`).toBe(
        SUBTRACT_ROW[key(pair)] as Status,
      );
    }
  });

  for (const operation of ['divide', 'floor_divide', 'ceil_divide']) {
    it(`propagates \`${operation}\` through every cell of its row`, () => {
      for (const pair of PAIRS) {
        expect(propagate(operation, pair), `${operation} ${key(pair)}`).toBe(
          DIVIDE_ROW[key(pair)] as Status,
        );
      }
    });
  }

  it('refuses `negate` and `absolute`, whatever the operand’s status', () => {
    // "not applicable to qualified values: a rejection" — one cell spanning every column, and
    // O0.5 calls a value carrying any status, `exact` among them, a qualified value. The strict
    // reading is the conservative one and it is what the module takes.
    for (const operation of ['negate', 'absolute']) {
      for (const status of STATUSES) {
        expect(() => propagate(operation, [status]), `${operation} ${status}`).toThrowError(
          new PyValueError(
            `\`${operation}\` is not applicable to a qualified value: a rejection (§2.2)`,
          ),
        );
      }
      expect(() => propagate(operation, [])).toThrowError(PyValueError);
    }
  });

  it('takes the chosen branch’s status for an `if`, over an exact condition alone', () => {
    // "the condition ranges over exact values only; the result has the chosen branch's status."
    for (const chosen of STATUSES) {
      expect(conditionalStatus([E, E], chosen), chosen).toBe(chosen);
    }
    expect(conditionalStatus([], S)).toBe(S);
    for (const status of [U, L, S]) {
      expect(() => conditionalStatus([E, status], E), status).toThrowError(
        new PyValueError(
          `the condition of an \`if\` ranges over exact values only: a \`${status}\` operand is ` +
            `a rejection (§2.2)`,
        ),
      );
    }
  });

  it('keeps the two binary rows apart from the n-ary one where they differ', () => {
    // The rows are not the same table read twice: eight of the sixteen pairs answer differently
    // under `add` and under `subtract`, and a port that reused one row for the other would pass
    // neither of the two suites above. Stated here so the difference is a claim of its own.
    const differing = PAIRS.filter(
      (pair) => (SUM_ROW[key(pair)] as Status) !== (SUBTRACT_ROW[key(pair)] as Status),
    );
    expect(differing.map(key)).toEqual(['E,U', 'E,L', 'U,U', 'U,L', 'L,U', 'L,L']);
  });

  it('turns no estimate into a bound, in any row', () => {
    // "No propagation turns an estimate into a bound." An estimate anywhere among the operands is
    // an estimate out, for every operator of the algebra that answers at all.
    for (const operation of Object.keys(PROPAGATIONS)) {
      if (propagationOf(operation) === 'rejection') continue;
      for (const pair of PAIRS) {
        if (!pair.includes(S)) continue;
        expect(propagate(operation, pair), `${operation} ${key(pair)}`).toBe(S);
      }
    }
  });
});

describe('§2.2, what the table does not state', () => {
  it('gives `modulo` no bound, having no row: exact operands alone keep an exact result', () => {
    // O0.1 puts modulo in the operator set; §2.2's table has four rows and none of them is its.
    // The conservative reading, recorded as a finding: exact in, exact out; anything else is an
    // estimate, since a bound would be a guarantee the specification does not grant.
    expect(propagationOf('modulo')).toBe('opaque');
    for (const pair of PAIRS) {
      const expected = pair[0] === E && pair[1] === E ? E : S;
      expect(propagate('modulo', pair), `modulo ${key(pair)}`).toBe(expected);
    }
  });

  it('rounds an estimate by the operator’s own choice, the direction being undefined for it', () => {
    // "A qualified value is rounded in the direction its status requires — an upper bound up, a
    // lower bound down; an exact value by the operator chosen (O0.3)." Three statuses of four;
    // an estimate has no relation to preserve, so the operator's choice stands.
    expect(roundQualified(2.5, S, 'floor')).toBe(2n);
    expect(roundQualified(2.5, S, 'ceil')).toBe(3n);
  });
});

describe('§2.2, the rounding a status requires', () => {
  it('rounds an upper bound up and a lower bound down, whatever the operator chose', () => {
    for (const chosen of ['floor', 'ceil'] as const) {
      expect(roundQualified(2.5, U, chosen), `upper ${chosen}`).toBe(3n);
      expect(roundQualified(2.5, L, chosen), `lower ${chosen}`).toBe(2n);
      expect(roundQualified(-2.5, U, chosen), `upper ${chosen}`).toBe(-2n);
      expect(roundQualified(-2.5, L, chosen), `lower ${chosen}`).toBe(-3n);
    }
  });

  it('rounds an exact value the way the operator chose (O0.3)', () => {
    expect(roundQualified(2.5, E, 'floor')).toBe(2n);
    expect(roundQualified(2.5, E, 'ceil')).toBe(3n);
    expect(roundQualified(-2.5, E, 'floor')).toBe(-3n);
    expect(roundQualified(-2.5, E, 'ceil')).toBe(-2n);
  });

  it('leaves a whole number where it is, and keeps it exact past the double range', () => {
    for (const status of STATUSES) {
      expect(roundQualified(7n, status, 'floor'), status).toBe(7n);
    }
    const huge = 10n ** 30n + 1n;
    expect(roundQualified(huge, U, 'floor')).toBe(huge);
  });

  it('refuses an infinity and a NaN in CPython’s own words', () => {
    expect(() => roundQualified(Number.POSITIVE_INFINITY, E, 'floor')).toThrowError(
      new PyOverflowError('cannot convert float infinity to integer'),
    );
    expect(() => roundQualified(Number.NaN, E, 'ceil')).toThrowError(
      new PyValueError('cannot convert float NaN to integer'),
    );
  });
});

describe('the sum row as the tools write it', () => {
  it('is `derive._status`, membership test by membership test', () => {
    // The function D5 calls, and the only status arithmetic `tools/` does: "estimate absorbs,
    // opposite bounds cancel into an estimate, one-sided bounds survive". N-ary, over a list, and
    // its four readings are the four the docstring names.
    expect(sumStatus([])).toBe(E);
    expect(sumStatus(['exact', 'exact'])).toBe(E);
    expect(sumStatus(['exact', 'upper_bound', 'upper_bound'])).toBe(U);
    expect(sumStatus(['exact', 'lower_bound'])).toBe(L);
    expect(sumStatus(['upper_bound', 'lower_bound'])).toBe(S);
    expect(sumStatus(['estimate', 'upper_bound', 'lower_bound'])).toBe(S);
    expect(sumStatus(['exact', 'estimate'])).toBe(S);
  });

  it('falls through to `exact` for a name the enumeration has not, as the tools do', () => {
    // `_status` is four membership tests over a `set`, so a status outside O0.5's four is simply
    // not one of the three it looks for. D5 hands it what a unit declared; the schema admits only
    // the four, and a port that refused an unknown one would refuse a document the tools derive.
    expect(sumStatus(['exact', 'probably'])).toBe(E);
    expect(sumStatus(['probably', 'lower_bound'])).toBe(L);
  });

  it('flips a bound and leaves the other two alone', () => {
    expect(flipped(U)).toBe(L);
    expect(flipped(L)).toBe(U);
    expect(flipped(E)).toBe(E);
    expect(flipped(S)).toBe(S);
  });
});

describe('the operators §2.2 reads', () => {
  it('gives each of the eleven a row and refuses a name it has not', () => {
    expect(Object.keys(PROPAGATIONS).sort()).toEqual([
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
    expect(() => propagationOf('logarithm')).toThrowError(
      new PyValueError("operator 'logarithm' is unknown to the qualified-value algebra of §2.2"),
    );
    expect(() => propagate('logarithm', [E, E])).toThrowError(PyValueError);
  });

  it('takes exactly two operands in the binary rows', () => {
    for (const operation of ['subtract', 'divide', 'floor_divide', 'ceil_divide', 'modulo']) {
      expect(() => propagate(operation, [E]), operation).toThrowError(
        new PyValueError(`\`${operation}\` takes two operands, not 1 (§2.2)`),
      );
      expect(() => propagate(operation, [E, E, E]), operation).toThrowError(
        new PyValueError(`\`${operation}\` takes two operands, not 3 (§2.2)`),
      );
    }
  });

  it('takes a list in the n-ary row, as the grammar writes it', () => {
    expect(propagate('add', [E, U, L, E])).toBe(S);
    expect(propagate('min', [U, U, U])).toBe(U);
  });
});
