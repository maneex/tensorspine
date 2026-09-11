/**
 * The qualified-value algebra of §2.2: statuses, their propagation through the operator set, and
 * the rounding a status requires.
 *
 * O0.5 states what a status is — "a value stated by a primitive correction (§4.1), by a sparsity
 * bound (§4.5), or supplied to a product from deployment intent (§10.3) carries a **status**:
 * `exact`; `upper_bound` or `lower_bound`, a one-sided guarantee on a non-negative quantity; or
 * `estimate`, no guaranteed relation to the true value" — and §2.2's table states how it
 * propagates, "where every operand is a non-negative real … and any operation with an `S` operand
 * yields `S`".
 *
 * **The tools implement one row of that table.** `derive._status` is the `add`, `multiply`, `min`,
 * `max` row, n-ary: "estimate absorbs, opposite bounds cancel into an estimate, one-sided bounds
 * survive" — and it is the only status arithmetic `tools/` does, because D5's only combination is
 * a sum (its `operations` totals). {@link sumStatus} is that function, written as the tools write
 * it so that parity on D5 is by construction; the rest of the table is the specification's and is
 * here because the specification is the contract, not because a corpus document reaches it. The
 * unit suite proves every cell.
 *
 * **The rows are the grammar's three operation unions**, exactly:
 *
 * | Row | Operators | Reading |
 * |---|---|---|
 * | n-ary | `add`, `multiply`, `min`, `max` | {@link sumStatus} over the operands |
 * | binary | `subtract`, `divide`, `floor_divide`, `ceil_divide` | the second operand's bound
 *   flips, then the same sum: `a − U` is a lower bound, `U − b` an upper one |
 * | binary | `modulo` | **no row in §2.2** — see below |
 * | unary | `negate`, `absolute` | "not applicable to qualified values: a rejection" |
 *
 * The `subtract` and `divide` rows of §2.2 are written cell by cell there; they are one reading —
 * `sum(a, flip(b))` — and the unit suite checks that reading against all sixteen cells of each
 * rather than against the mechanism. The `(b > 0)` beside the division row is a condition on the
 * *values*, which the evaluator decides (a division by zero is `UNRESOLVED`, feature 1.2), not on
 * the statuses this module combines.
 *
 * **Three readings the specification does not state, taken conservatively and recorded as
 * findings.**
 *
 * 1. `modulo` is in the operator set of O0.1, is written **37 times** across the corpus and the
 *    reference base, and has **no row** in §2.2's table. The conservative reading is the one that
 *    cannot invent a guarantee: exact operands give an exact result, and anything else gives an
 *    `estimate` — "no guaranteed relation to the true value" is always a sound thing to say, and
 *    "no propagation turns an estimate into a bound" makes the degradation one-way.
 *    {@link propagationOf} calls it `opaque`, so that the absence of a row is visible in the code
 *    rather than hidden inside a default branch.
 * 2. `negate` and `absolute`: the row spans every column of the table with one cell, "not
 *    applicable to qualified values: a rejection", and O0.5 calls a value carrying *any* status —
 *    `exact` among them — a qualified value. Both readings of that sentence therefore agree on
 *    the strict one: propagating a status through a negation or an absolute value is a rejection
 *    whatever the status is. Nothing is lost: the corpus and the reference base write five of the
 *    eleven operators — `multiply` 73 times, `subtract` 64, `modulo` 37, `floor_divide` 14 and
 *    `add` 10 — and neither `negate` nor `absolute` once.
 * 3. The rounding rule names three statuses of four: "an upper bound up, a lower bound down; an
 *    exact value by the operator chosen (O0.3)". An `estimate` has no direction its status
 *    requires — there is no relation to preserve — so the operator's own choice stands, which is
 *    the only rule available.
 *
 * **What consults this module today**: D5's `operations`, through {@link sumStatus}, and nothing
 * else — because nothing else in the repository composes two statuses. An *expression* of the
 * algebra reads quantities and arguments, which O0.3 makes exact, so the sixteen operation nodes
 * of the reference base's eight cost entries (`multiply` 13, `add` 2, `subtract` 1) combine exact
 * values and carry no status at all; a status meets a status only where a product sums values that
 * each declare one, which is the sum row and no other. The rest of the table is written here
 * because a language is what its specification says and not what its corpus reaches: whoever
 * builds on a qualified value next — the editor showing one, a product that combines two, a
 * deployment intent supplied from outside (§10.3) — finds it already written and already proved.
 */
import { PyOverflowError, PyValueError } from '../expr/errors.js';

/** `epistemic_status`: the four statuses O0.5 declares, in the order the schema writes them. */
export type Status = 'exact' | 'upper_bound' | 'lower_bound' | 'estimate';

/**
 * The statuses this algebra knows, as a set the audit of §1 (d) compares with the schema's
 * `epistemic_status` enumeration: a status the language gained without a reading here, or a
 * reading for a name the language has not, fails the build.
 */
export const STATUSES: readonly Status[] = ['exact', 'upper_bound', 'lower_bound', 'estimate'];

/** How §2.2's table treats one operator: which row of it the operator stands in. */
export type Propagation = 'sum' | 'inverse' | 'opaque' | 'rejection';

/**
 * The row of §2.2's table each operator of O0.1's set stands in.
 *
 * A table keyed by a vocabulary item, which §1 admits for exactly this reason — "an evaluator must
 * say what `floor_divide` does" — and which the audit of §1 (d) holds to the three operation
 * unions of both grammars: eleven operators, eleven readings, no fall-through.
 */
export const PROPAGATIONS: Readonly<Record<string, Propagation>> = {
  add: 'sum',
  multiply: 'sum',
  min: 'sum',
  max: 'sum',
  subtract: 'inverse',
  divide: 'inverse',
  floor_divide: 'inverse',
  ceil_divide: 'inverse',
  modulo: 'opaque',
  negate: 'rejection',
  absolute: 'rejection',
};

/** The row an operator stands in, or the refusal of an operator §2.2 has no reading for. */
export function propagationOf(operation: string): Propagation {
  const row = Object.prototype.hasOwnProperty.call(PROPAGATIONS, operation)
    ? PROPAGATIONS[operation]
    : undefined;
  if (row === undefined) {
    throw new PyValueError(
      `operator '${operation}' is unknown to the qualified-value algebra of §2.2`,
    );
  }
  return row;
}

/**
 * `_status(statuses)`: "the status of a sum of qualified values (§2.2): estimate absorbs, opposite
 * bounds cancel into an estimate, one-sided bounds survive".
 *
 * The tools' own function, and the whole of the `add`, `multiply`, `min`, `max` row — n-ary, as
 * that row's operators are. It is written as they write it, membership test by membership test,
 * so that a status string outside the enumeration falls through to `exact` here as it does there:
 * D5 hands it what a unit declared, and a divergence on an unknown name would be a divergence in
 * the derived document.
 */
export function sumStatus(statuses: Iterable<string>): Status {
  const seen = new Set(statuses);
  if (seen.has('estimate') || (seen.has('upper_bound') && seen.has('lower_bound'))) {
    return 'estimate';
  }
  if (seen.has('upper_bound')) return 'upper_bound';
  if (seen.has('lower_bound')) return 'lower_bound';
  return 'exact';
}

/**
 * The status of the operand on the *right* of a subtraction or a division, as §2.2 reads it.
 *
 * "`a − U`: L · `U − b`: U" and "`a/U`: L · `U/b`: U" say one thing: what bounds the subtrahend or
 * the divisor from above bounds the result from below, and conversely. An exact value and an
 * estimate are unchanged, which is what makes `sum(a, flip(b))` reproduce all sixteen cells of
 * each binary row — including `U − L: U` and `L − U: L`, where the flip turns two opposite bounds
 * into two of the same kind, and `U − U: S`, where it turns two of a kind into opposites.
 */
export function flipped(status: Status): Status {
  if (status === 'upper_bound') return 'lower_bound';
  if (status === 'lower_bound') return 'upper_bound';
  return status;
}

/**
 * The status §2.2's table gives an operation over operands of the given statuses.
 *
 * The arities are the grammar's: `add`, `multiply`, `min` and `max` take a list (their union's
 * `minItems` is 2), the binary operators take exactly two, and `negate` and `absolute` are refused
 * before their one operand is ever looked at.
 */
export function propagate(operation: string, operands: readonly Status[]): Status {
  const row = propagationOf(operation);
  if (row === 'rejection') {
    throw new PyValueError(
      `\`${operation}\` is not applicable to a qualified value: a rejection (§2.2)`,
    );
  }
  if (row === 'sum') return sumStatus(operands);
  const [left, right] = operands;
  if (operands.length !== 2 || left === undefined || right === undefined) {
    throw new PyValueError(
      `\`${operation}\` takes two operands, not ${String(operands.length)} (§2.2)`,
    );
  }
  // `modulo` has no row of its own: exact operands keep an exact result, and any qualified operand
  // leaves a value with no guaranteed relation to the true one.
  if (row === 'opaque') return left === 'exact' && right === 'exact' ? 'exact' : 'estimate';
  return sumStatus([left, flipped(right)]);
}

/**
 * The `if/then/else` row: "the condition ranges over exact values only; the result has the chosen
 * branch's status".
 *
 * Both halves are here. A condition compares values, and a value carrying anything but `exact` is
 * a rejection — the branch taken would otherwise be a guess. The result is the **chosen** branch's
 * status and not a combination of the two: the branch not taken contributes nothing, which is why
 * the other branch's status is not a parameter.
 */
export function conditionalStatus(condition: readonly Status[], chosen: Status): Status {
  for (const status of condition) {
    if (status === 'exact') continue;
    throw new PyValueError(
      `the condition of an \`if\` ranges over exact values only: a \`${status}\` operand is a ` +
        `rejection (§2.2)`,
    );
  }
  return chosen;
}

/** Which way an operator rounds where the status leaves the choice open (O0.3). */
export type Rounding = 'floor' | 'ceil';

/**
 * "A qualified value is rounded in the direction its status requires — an upper bound up, a lower
 * bound down; an exact value by the operator chosen (O0.3)."
 *
 * An `estimate` takes the operator's choice too: its status requires no direction, having no
 * relation to the true value to preserve. An integer is already rounded, and the two refusals are
 * CPython's own, `math.floor` and `math.ceil` converting before they round.
 */
export function roundQualified(
  value: bigint | number,
  status: Status,
  chosen: Rounding,
): bigint {
  const direction: Rounding =
    status === 'upper_bound' ? 'ceil' : status === 'lower_bound' ? 'floor' : chosen;
  if (typeof value === 'bigint') return value;
  if (Number.isNaN(value)) throw new PyValueError('cannot convert float NaN to integer');
  if (!Number.isFinite(value)) {
    throw new PyOverflowError('cannot convert float infinity to integer');
  }
  return BigInt(direction === 'ceil' ? Math.ceil(value) : Math.floor(value));
}
