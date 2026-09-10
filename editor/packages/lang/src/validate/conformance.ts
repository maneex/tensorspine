/**
 * A value against its declared type and domain: V3, for a quantity and for a primitive argument
 * alike — `_check_type` and `_check_domain` of `tools/validate.py`.
 *
 * "Types, enums, record fields, units and domains conform, for arguments and for quantities,
 * derived ones included; an enum value outside its declared set is rejected; a number literal is
 * an instance of its declared type (`32.0` is not a cardinality); a physical value in `tokens`,
 * `elements`, `bytes` or `operations` is a whole number, only `seconds` being real" (§6, V3).
 *
 * **One table, two callers.** The tools write the type table once and call it from both sides:
 * `check_quantities` hands it a quantity's value under the declared type, `_resolve_record` hands
 * it an argument's. So it lives here rather than under either, and its wording — which says
 * `argument '…'` in both cases, the quantity side rewriting the first occurrence afterwards — is
 * written once.
 *
 * **The record branch is the argument side's.** A `record` type recurses into `_resolve_record`,
 * which applies defaults, refuses unknown fields (V2) and reads `present_when` — the argument
 * resolution of feature 1.6a. A quantity's type is never a record (`quantity_type` is
 * `cardinality | real | physical | enum | boolean`), so the quantity side passes no
 * {@link RecordCheck} and the recursion has no caller here; what does belong here is the refusal
 * of a *scalar* where a record is declared, which is V3's own wording.
 *
 * **Two parameters of `_check_type` are dead.** Its signature is
 * `(v, t, label, evaluate, root, problems, siblings)`, and neither `evaluate` nor `siblings` is
 * read in the body — the record branch passes `lambda x: x` rather than `evaluate` — while `root`
 * is read only by that branch. The port takes what the body uses and lets the record check close
 * over the rest.
 *
 * **Where it raises, the tools raise.** `_check_domain` compares with Python's `>=`, `<=`, `>` and
 * `<`, and `_check_type`'s whole-number test converts with `int()` and `float()`: a value the
 * comparison cannot order (an enum's string against a numeric bound) and a non-finite or
 * astronomical number both raise out of `analyse` in the tools, uncaught. Meaning assumes grammar
 * — no conforming document reaches either — and the port raises there too rather than inventing
 * an answer the tools do not give.
 */
import { PyOverflowError, PyTypeError, PyValueError } from '../expr/errors.js';
import { pyEqual, pyOrder } from '../expr/arithmetic.js';
import { modelValue, type Quantities } from '../expr/model.js';
import {
  isRecord,
  pythonTypeName,
  truthy,
  UNRESOLVED,
  type PyRecord,
  type PyValue,
} from '../expr/value.js';
import { demand, optional } from '../library/access.js';
import { pyRepr, pyStr } from '../library/repr.js';

import { semanticProblem, type SemanticProblem } from './problems.js';

/**
 * The recursion into a declared `record` type: `_resolve_record` over the value's own fields.
 *
 * Feature 1.6a supplies it, closing over the resolved-argument map an absolute argument path is
 * read against (`_check_type`'s `root`) and over the map it fills in place. Nothing of the
 * quantity side has one, and reaching a record without one is a defect of the caller, not a
 * refusal of the document.
 */
export type RecordCheck = (
  /** The value, already known to be a record. */
  value: PyRecord,
  /** The declared type, whose `fields` hold the field declarations. */
  declared: PyValue,
  /** The label the fields are named under: `rope.scaling` for `rope.scaling.kind`. */
  label: string,
  /** The list to append to, in the tools' order. */
  problems: SemanticProblem[],
) => void;

/** No quantities at all: the empty scope `_check_domain` evaluates a bound in. */
const NO_QUANTITIES: Quantities = new Map<string, PyValue>();

/**
 * `_check_type`: one value against one declared type, appending V2/V3 problems in the tools' order.
 *
 * The label is what the message names — a quantity's name, an argument's path — and the messages
 * say `argument '…'` in both cases; `checkQuantities` rewrites the first occurrence of that word,
 * which is what the tools do at the end of `check_quantities`.
 */
export function checkType(
  value: PyValue,
  declared: PyValue,
  label: string,
  problems: SemanticProblem[],
  record?: RecordCheck,
): void {
  const kind = demand(declared, 'kind');
  const refuse = (message: string): void => {
    problems.push(semanticProblem('V3', `argument '${label}' ${message}`));
  };
  if (value === UNRESOLVED) {
    refuse('does not resolve to a value');
    return;
  }
  if (kind === 'record') {
    if (!isRecord(value)) {
      refuse(`= ${pyRepr(value)} is not a record`);
      return;
    }
    if (record === undefined) {
      throw new Error(
        `checkType: '${label}' declares a record type and no record check was supplied ` +
          '(the quantity side has none: a quantity type is a scalar one)',
      );
    }
    record(value, declared, label, problems);
    return;
  }
  if (typeof value === 'boolean') {
    // A `bool` is an `int` in Python, so the boolean case is decided before every numeric one.
    if (kind !== 'boolean') refuse(`= ${pyRepr(value)} is a boolean, not ${pyStr(kind)}`);
    return;
  }
  if (kind === 'boolean') {
    refuse(`= ${pyRepr(value)} is not a boolean`);
  } else if (kind === 'cardinality') {
    // A literal is an instance of its type: 32.0 is not a cardinality (V3).
    if (typeof value !== 'bigint' || value < 0n) {
      refuse(`= ${pyRepr(value)} is not a cardinality (non-negative integer)`);
    }
  } else if (kind === 'real' || kind === 'physical') {
    const unit = kind === 'physical' ? demand(declared, 'unit') : null;
    if (typeof value !== 'bigint' && typeof value !== 'number') {
      refuse(`= ${pyRepr(value)} is not a number` + (unit === null ? '' : ` of ${pyStr(unit)}`));
    } else if (unit !== null && unit !== 'seconds' && !pyEqual(asFloat(value), asInteger(value))) {
      refuse(
        `= ${pyRepr(value)} is not a whole number of ${pyStr(unit)} (only seconds is real)`,
      );
    }
  } else if (kind === 'enum') {
    const values = demand(declared, 'values');
    if (!pyContains(values, value)) {
      refuse(`= ${pyRepr(value)} is not among ${pyStr(values)}`);
    }
  } else {
    problems.push(
      semanticProblem('V3', `argument '${label}': type '${pyStr(kind)}' is unknown to this validator`),
    );
  }
}

/**
 * `_check_domain`: a **model quantity's** value against its declared domain (§2.2).
 *
 * "A bound is a model scalar expression; only a literal bound is checked here (the empty scope
 * leaves a quantity-referencing bound undecidable)" — the scope really is empty, so a bound
 * written over another quantity is skipped rather than resolved, on a document where every
 * quantity has a value. The primitive argument's domain is checked by its own sibling
 * (`_check_argument_domain`, feature 1.6a), which evaluates a bound in the instance's arguments.
 */
export function checkDomain(
  value: PyValue,
  domain: PyValue,
  label: string,
  problems: SemanticProblem[],
): void {
  const refuse = (message: string): void => {
    problems.push(semanticProblem('V3', `argument '${label}' ${message}`));
  };
  if (demand(domain, 'kind') === 'set') {
    const values = demand(domain, 'values');
    if (!pyContains(values, value)) {
      refuse(`= ${pyRepr(value)} is outside the set ${pyStr(values)}`);
    }
    return;
  }
  for (const [edge, said] of [
    ['lower', 'below'],
    ['upper', 'above'],
  ] as const) {
    const bound = optional(domain, edge, null);
    if (bound === null) continue;
    const limit = modelValue(demand(bound, 'value'), NO_QUANTITIES);
    if (limit === UNRESOLVED) continue;
    const inclusive = truthy(demand(bound, 'inclusive'));
    const operator = edge === 'lower' ? (inclusive ? '>=' : '>') : inclusive ? '<=' : '<';
    if (!compares(value, limit, operator)) {
      refuse(
        `= ${pyRepr(value)} is ${said} the domain bound ${pyRepr(limit)} ` +
          `(${inclusive ? 'inclusive' : 'exclusive'})`,
      );
    }
  }
}

/** `left <op> right`, as Python answers it — `False` for anything a `nan` takes part in. */
function compares(left: PyValue, right: PyValue, operator: '>=' | '>' | '<=' | '<'): boolean {
  let order: number | undefined;
  try {
    order = pyOrder(left, right);
  } catch (error) {
    if (error instanceof PyTypeError) {
      // `pyOrder` writes `<`; the operator the line was written with is what Python names.
      throw new PyTypeError(
        `'${operator}' not supported between instances of ` +
          `'${pythonTypeName(left)}' and '${pythonTypeName(right)}'`,
      );
    }
    throw error;
  }
  if (order === undefined) return false;
  if (operator === '>=') return order >= 0;
  if (operator === '>') return order > 0;
  if (operator === '<=') return order <= 0;
  return order < 0;
}

/**
 * `value in container`, as Python reads it: `==` over a list's elements or a dictionary's keys, a
 * substring of a string, and a `TypeError` for anything else. The grammar makes every container
 * this module asks about a list.
 */
function pyContains(container: PyValue, value: PyValue): boolean {
  if (Array.isArray(container)) {
    return (container as readonly PyValue[]).some((one) => pyEqual(value, one));
  }
  if (isRecord(container)) return Object.keys(container).some((name) => pyEqual(value, name));
  if (typeof container === 'string') {
    if (typeof value !== 'string') {
      throw new PyTypeError(
        `'in <string>' requires string as left operand, not ${pythonTypeName(value)}`,
      );
    }
    return container.includes(value);
  }
  throw new PyTypeError(`argument of type '${pythonTypeName(container)}' is not iterable`);
}

/** `float(v)` on a number the grammar admits: exact, or Python's refusal for a huge integer. */
function asFloat(value: bigint | number): number {
  if (typeof value === 'number') return value;
  const asDouble = Number(value);
  if (!Number.isFinite(asDouble)) {
    throw new PyOverflowError('int too large to convert to float');
  }
  return asDouble;
}

/** `int(v)`: truncation towards zero, and Python's two refusals for a non-finite float. */
function asInteger(value: bigint | number): bigint {
  if (typeof value === 'bigint') return value;
  if (Number.isNaN(value)) throw new PyValueError('cannot convert float NaN to integer');
  if (!Number.isFinite(value)) {
    throw new PyOverflowError('cannot convert float infinity to integer');
  }
  return BigInt(Math.trunc(value));
}
