/**
 * The operator and comparison tables of `tools/expr.py`, and the arithmetic under them.
 *
 * The plan's §1 admits a table keyed by a vocabulary item only in the core, and only to attach
 * semantics to it: "an evaluator must say what `floor_divide` does". {@link OPERATORS} and
 * {@link COMPARISONS} are those two tables, and `tests/audit/semantic-tables.test.ts` proves each
 * key set equal to the schemas' own enumerations — an operator the schemas list without an
 * implementation, or an implementation for a name they do not list, fails the build.
 *
 * What the operators compute is Python's, because the tools are the parity contract (D2) and
 * because the specification's O0.3 leaves rounding to the operator chosen rather than to the
 * host language. The rules that are Python's rather than JavaScript's, all of them exercised by
 * `test/expr/arithmetic.test.ts` against the values `tools/` answers:
 *
 * - **`bool` is an `int`.** `add(true, true)` is `2`, `negate(true)` is `-1`, `absolute(true)` is
 *   `1`, and `true == 1` — but `min(true, 2)` answers `true`, since `min` returns the operand.
 * - **`divide` always answers a float**, `4 / 2` included, and division by zero — integer or
 *   float — raises, which is why `_apply` answers `UNRESOLVED` for it rather than `Infinity`.
 * - **`floor_divide` rounds towards minus infinity**, not towards zero: `-7 // 2` is `-4`. So
 *   `ceil_divide`, written `-((-a) // b)`, is `-3` there; JavaScript's `/` and `%` would give
 *   `-3` and `-1`.
 * - **`modulo` takes the sign of the divisor**: `-7 % 2` is `1`, `7 % -2` is `-1`.
 * - **`add` and `multiply` are `sum` and a product started at an integer**, so an empty argument
 *   list answers `0` and `1`, and a string operand is `0 + str`, a `TypeError`.
 * - **An `int` times a `str` repeats it** (`multiply(['ab', 3])` is `'ababab'`), which is
 *   Python's operator and is reproduced here.
 * - **An operand of the wrong kind raises**, and `apply` turns that into `UNRESOLVED`: the value
 *   it would have fed is not decidable, and the argument that caused it has been refused already
 *   (V3).
 *
 * One divergence is deliberate and stated: `'%d' % 2` is *string formatting* in Python and
 * answers `'2'`, so `modulo` over a format string answers a string there and `UNRESOLVED` here.
 * Implementing printf would be implementing a sublanguage the specification does not name; the
 * conservative answer is to refuse to decide. Every other string on the left of `modulo` raises
 * in Python too, and no value in the corpus or the reference base carries a `%`.
 */
import { comparePythonStrings } from '../schema/index.js';

import { PyIndexError, PyKeyError, PyTypeError, PyValueError, PyZeroDivisionError } from './errors.js';
import { isRecord, pythonTypeName as typeName, UNRESOLVED, type PyValue } from './value.js';

/** A refusal worded as Python words an operator's refusal. */
function unsupported(operator: string, left: PyValue, right?: PyValue): PyTypeError {
  if (right === undefined) {
    return new PyTypeError(`bad operand type for ${operator}: '${typeName(left)}'`);
  }
  return new PyTypeError(
    `unsupported operand type(s) for ${operator}: '${typeName(left)}' and '${typeName(right)}'`,
  );
}

/** Python's integer reading of a value: an `int` as it is, a `bool` as 1 or 0. */
function asInteger(value: PyValue): bigint | undefined {
  if (typeof value === 'bigint') return value;
  if (typeof value === 'boolean') return value ? 1n : 0n;
  return undefined;
}

/** Whether the value is one Python computes with: an `int`, a `bool` or a `float`. */
function isNumber(value: PyValue): value is bigint | boolean | number {
  return typeof value === 'bigint' || typeof value === 'boolean' || typeof value === 'number';
}

/**
 * The two operands under Python's numeric tower: two integers, or two floats.
 *
 * An `int` beyond the range of a double is `Infinity` here where Python raises `OverflowError`;
 * no number of the corpus or the reference base has more than eight digits, and the value model
 * of `value.ts` is what keeps every integer exact up to that point.
 */
function promote(left: PyValue, right: PyValue): [bigint, bigint] | [number, number] | undefined {
  if (!isNumber(left) || !isNumber(right)) return undefined;
  const a = asInteger(left);
  const b = asInteger(right);
  if (a !== undefined && b !== undefined) return [a, b];
  return [a === undefined ? (left as number) : Number(a), b === undefined ? (right as number) : Number(b)];
}

/** The magnitude of `value` with the sign of `sign`, C's `copysign`, which CPython uses. */
function copysign(value: number, sign: number): number {
  const negative = sign < 0 || Object.is(sign, -0);
  return negative ? -Math.abs(value) : Math.abs(value);
}

/** `l + r`. */
export function pyAdd(left: PyValue, right: PyValue): PyValue {
  const pair = promote(left, right);
  if (pair === undefined) throw unsupported('+', left, right);
  const [a, b] = pair;
  return typeof a === 'bigint' ? a + (b as bigint) : a + (b as number);
}

/** `l * r`, string repetition included. */
export function pyMultiply(left: PyValue, right: PyValue): PyValue {
  const pair = promote(left, right);
  if (pair !== undefined) {
    const [a, b] = pair;
    return typeof a === 'bigint' ? a * (b as bigint) : a * (b as number);
  }
  const times = typeof left === 'string' ? asInteger(right) : asInteger(left);
  const text = typeof left === 'string' ? left : right;
  if (times !== undefined && typeof text === 'string') {
    return times <= 0n ? '' : text.repeat(Number(times));
  }
  throw unsupported('*', left, right);
}

/** `l - r`. */
export function pySubtract(left: PyValue, right: PyValue): PyValue {
  const pair = promote(left, right);
  if (pair === undefined) throw unsupported('-', left, right);
  const [a, b] = pair;
  return typeof a === 'bigint' ? a - (b as bigint) : a - (b as number);
}

/**
 * `l / r`: a float, always, and a refusal on a zero divisor.
 *
 * Two integers are converted to doubles and divided, which is the correctly rounded quotient of
 * the two exact values as long as each of them is one a double holds exactly — every number of
 * the corpus and the reference base is, by eight orders of magnitude.
 */
export function pyDivide(left: PyValue, right: PyValue): PyValue {
  const pair = promote(left, right);
  if (pair === undefined) throw unsupported('/', left, right);
  const [a, b] = pair;
  if (typeof a === 'bigint') {
    const divisor = b as bigint;
    if (divisor === 0n) throw new PyZeroDivisionError('division by zero');
    return Number(a) / Number(divisor);
  }
  if ((b as number) === 0) throw new PyZeroDivisionError('float division by zero');
  return a / (b as number);
}

/** `l // r`: towards minus infinity, in Python's own two implementations. */
export function pyFloorDivide(left: PyValue, right: PyValue): PyValue {
  const pair = promote(left, right);
  if (pair === undefined) throw unsupported('//', left, right);
  const [a, b] = pair;
  if (typeof a === 'bigint') {
    const divisor = b as bigint;
    if (divisor === 0n) throw new PyZeroDivisionError('integer division or modulo by zero');
    const quotient = a / divisor;
    // JavaScript's `/` on integers truncates towards zero; Python floors.
    return a % divisor !== 0n && a < 0n !== divisor < 0n ? quotient - 1n : quotient;
  }
  return floatDivmod(a, b as number).quotient;
}

/** `l % r`: the remainder takes the sign of the divisor. */
export function pyModulo(left: PyValue, right: PyValue): PyValue {
  const pair = promote(left, right);
  if (pair === undefined) throw unsupported('%', left, right);
  const [a, b] = pair;
  if (typeof a === 'bigint') {
    const divisor = b as bigint;
    if (divisor === 0n) throw new PyZeroDivisionError('integer division or modulo by zero');
    const rest = a % divisor;
    return rest !== 0n && rest < 0n !== divisor < 0n ? rest + divisor : rest;
  }
  return floatDivmod(a, b as number).rest;
}

/**
 * CPython's `float_divmod`, step for step.
 *
 * `fmod` — JavaScript's `%` on numbers is the same operation — is exact, so `a - rest` is a
 * mathematical multiple of `b`; the division that follows is not exact, which is why CPython
 * snaps the quotient to the nearest integral value rather than flooring the approximation, and
 * why the remainder's sign is forced onto the divisor's, signed zeroes included.
 */
function floatDivmod(a: number, b: number): { quotient: number; rest: number } {
  if (b === 0) throw new PyZeroDivisionError('float floor division by zero');
  let rest = a % b;
  let division = (a - rest) / b;
  if (rest !== 0) {
    if (b < 0 !== rest < 0) {
      rest += b;
      division -= 1;
    }
  } else {
    rest = copysign(0, b);
  }
  let quotient: number;
  if (division !== 0) {
    quotient = Math.floor(division);
    if (division - quotient > 0.5) quotient += 1;
  } else {
    quotient = copysign(0, a / b);
  }
  return { quotient, rest };
}

/** `-l`; a `bool` negates to an `int`, as in Python. */
export function pyNegate(value: PyValue): PyValue {
  const integer = asInteger(value);
  if (integer !== undefined) return -integer;
  if (typeof value === 'number') return -value;
  throw unsupported('unary -', value);
}

/** `abs(l)`; a `bool` answers an `int`. */
export function pyAbsolute(value: PyValue): PyValue {
  const integer = asInteger(value);
  if (integer !== undefined) return integer < 0n ? -integer : integer;
  if (typeof value === 'number') return Math.abs(value);
  throw unsupported('abs()', value);
}

/**
 * The order of two values, `-1`, `0` or `1`, or `undefined` when they are unordered.
 *
 * `undefined` is the answer for a comparison involving a `nan`: Python's `<`, `>`, `<=` and `>=`
 * are all false there, and so is `==`. Two values of kinds Python cannot order raise, which is
 * what `model_condition` catches and turns into `UNRESOLVED` and what `primitive_condition`
 * does not catch.
 *
 * Integers and floats are compared exactly, as Python compares them: the float is split at its
 * decimal point, so an integer beyond 2^53 is not first rounded to a double.
 */
export function pyOrder(left: PyValue, right: PyValue): number | undefined {
  if (typeof left === 'string' && typeof right === 'string') {
    const order = comparePythonStrings(left, right);
    return order < 0 ? -1 : order > 0 ? 1 : 0;
  }
  if (isNumber(left) && isNumber(right)) {
    const a = asInteger(left);
    const b = asInteger(right);
    if (a !== undefined && b !== undefined) return a < b ? -1 : a > b ? 1 : 0;
    if (a === undefined && b === undefined) return orderFloats(left as number, right as number);
    if (b === undefined) return orderMixed(a as bigint, right as number);
    const order = orderMixed(b, left as number);
    return order === undefined ? undefined : -order;
  }
  throw new PyTypeError(
    `'<' not supported between instances of '${typeName(left)}' and '${typeName(right)}'`,
  );
}

function orderFloats(a: number, b: number): number | undefined {
  if (Number.isNaN(a) || Number.isNaN(b)) return undefined;
  return a < b ? -1 : a > b ? 1 : 0;
}

/** An integer against a float, exactly: the float's integer part, then its fraction. */
function orderMixed(a: bigint, b: number): number | undefined {
  if (Number.isNaN(b)) return undefined;
  if (b === Infinity) return -1;
  if (b === -Infinity) return 1;
  const whole = Math.floor(b);
  const integral = BigInt(whole);
  if (a < integral) return -1;
  if (a > integral) return 1;
  return b > whole ? -1 : 0;
}

/**
 * Python's `==`: exact across the numeric kinds, deep over containers, false across kinds.
 *
 * `1 == 1.0` and `true == 1` are true, `1 == '1'` is false, and two records are equal when they
 * carry the same members with equal values — which is what a comparison of a record argument
 * against another does.
 */
export function pyEqual(left: PyValue, right: PyValue): boolean {
  // Python compares identity before content, which is what makes `UNRESOLVED == UNRESOLVED`
  // true there; `NaN === NaN` is false in JavaScript, as `nan == nan` is false in Python.
  if (left === right) return true;
  if (isNumber(left) && isNumber(right)) return pyOrder(left, right) === 0;
  if (typeof left === 'string' || typeof right === 'string') return left === right;
  if (left === null || right === null) return left === right;
  if (Array.isArray(left) || Array.isArray(right)) {
    if (!Array.isArray(left) || !Array.isArray(right)) return false;
    const ours = left as readonly PyValue[];
    const others = right as readonly PyValue[];
    if (ours.length !== others.length) return false;
    return ours.every((one, index) => pyEqual(one, others[index] as PyValue));
  }
  if (!isRecord(left) || !isRecord(right)) return false;
  const names = Object.keys(left);
  if (names.length !== Object.keys(right).length) return false;
  return names.every(
    (name) => Object.hasOwn(right, name) && pyEqual(left[name] as PyValue, right[name] as PyValue),
  );
}

/** The operand at that position, or Python's `IndexError` for an argument list too short. */
function at(args: readonly PyValue[], index: number): PyValue {
  if (index >= args.length) throw new PyIndexError('list index out of range');
  return args[index] as PyValue;
}

/** `min` and `max` keep the first extreme operand, so `min(1, 1.0)` is the integer `1`. */
function extreme(args: readonly PyValue[], wanted: number, name: string): PyValue {
  if (args.length === 0) throw new PyValueError(`${name}() arg is an empty sequence`);
  let best = args[0] as PyValue;
  for (const candidate of args.slice(1)) {
    if (pyOrder(candidate, best) === wanted) best = candidate;
  }
  return best;
}

/**
 * The operators of the algebra, one implementation each, keyed by the names the schemas list
 * (§2.2 O0.1: "addition, subtraction, multiplication, division, floor division, ceiling
 * division, modulo, minimum, maximum, negation, absolute value").
 */
export const OPERATORS: Readonly<Record<string, (args: readonly PyValue[]) => PyValue>> = {
  add: (args) => args.reduce<PyValue>((total, one) => pyAdd(total, one), 0n),
  multiply: (args) => args.reduce<PyValue>((total, one) => pyMultiply(total, one), 1n),
  subtract: (args) => pySubtract(at(args, 0), at(args, 1)),
  divide: (args) => pyDivide(at(args, 0), at(args, 1)),
  floor_divide: (args) => pyFloorDivide(at(args, 0), at(args, 1)),
  // `-((-a) // b)`, which is how `expr.py` writes the ceiling and why it rounds a negative
  // quotient towards zero.
  ceil_divide: (args) => pyNegate(pyFloorDivide(pyNegate(at(args, 0)), at(args, 1))),
  modulo: (args) => pyModulo(at(args, 0), at(args, 1)),
  min: (args) => extreme(args, -1, 'min'),
  max: (args) => extreme(args, 1, 'max'),
  negate: (args) => pyNegate(at(args, 0)),
  absolute: (args) => pyAbsolute(at(args, 0)),
};

/** The comparison operators of the condition language, keyed by the names the schemas list. */
export const COMPARISONS: Readonly<Record<string, (left: PyValue, right: PyValue) => boolean>> = {
  equal: (left, right) => pyEqual(left, right),
  not_equal: (left, right) => !pyEqual(left, right),
  greater: (left, right) => pyOrder(left, right) === 1,
  less: (left, right) => pyOrder(left, right) === -1,
  greater_or_equal: (left, right) => {
    const order = pyOrder(left, right);
    return order === 1 || order === 0;
  },
  less_or_equal: (left, right) => {
    const order = pyOrder(left, right);
    return order === -1 || order === 0;
  },
};

/**
 * One operator over evaluated operands, `expr.py`'s `_apply`.
 *
 * "An operand of the wrong kind, or a division by zero, yields UNRESOLVED: the value it would
 * have fed is not decidable, and the argument that caused it has been refused already (V3)." An
 * operator name the table does not carry answers `UNRESOLVED` too, which is `_apply_raw`'s
 * fall-through; every other refusal — an argument list too short for the operator, an empty
 * `min` — is raised, as Python raises it, because it means the expression is off the grammar.
 */
export function apply(op: PyValue, args: readonly PyValue[]): PyValue {
  // `_apply_raw` compares the name against each operator in turn and falls through, so a name
  // that is not a string — which no grammar admits — is simply no operator.
  const operator = typeof op === 'string' ? OPERATORS[op] : undefined;
  if (operator === undefined) return UNRESOLVED;
  try {
    return operator(args);
  } catch (error) {
    if (error instanceof PyTypeError || error instanceof PyZeroDivisionError) return UNRESOLVED;
    throw error;
  }
}

/** Whether a comparison holds, or a refusal Python would raise for two values it cannot order. */
export function compare(operator: PyValue, left: PyValue, right: PyValue): boolean {
  const comparison = typeof operator === 'string' ? COMPARISONS[operator] : undefined;
  if (comparison === undefined) {
    // `_COMPARISONS[cp['operator']]` on a name the table does not carry is a `KeyError` in the
    // tools: the grammar admits six operators and nothing else reaches here.
    throw new PyKeyError(typeof operator === 'string' ? `'${operator}'` : typeName(operator));
  }
  return comparison(left, right);
}
