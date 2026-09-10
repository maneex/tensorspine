/**
 * The values the two evaluators of `tools/expr.py` compute with, and what they are in
 * JavaScript.
 *
 * The tools evaluate on the reading CPython's `json` gives a document, so a number in an
 * expression is an `int` or a `float` and the language reads the difference: V3 refuses `32.0`
 * as a cardinality (`isinstance(v, int)`), `divide` always answers a float, `floor_divide` of two
 * integers answers an integer, and the derived documents the parity job compares are written with
 * that distinction in their bytes. JavaScript's `number` cannot carry it — `4` and `4.0` are the
 * same double — so the port carries it in the type:
 *
 * | Python | here | why |
 * |---|---|---|
 * | `int` | `bigint` | arbitrary precision, exactly as Python's; `typeof v === 'bigint'` is `isinstance(v, int)` |
 * | `float` | `number` | a double, exactly as Python's |
 * | `bool` | `boolean` | Python's `bool` is an `int` subclass, so arithmetic reads it as 1 and 0 ({@link arithmetic}) |
 * | `str` | `string` | |
 * | `None` | `null` | which is also JSON `null`, as it is in Python |
 * | `dict` | a plain object | a record argument, whose fields are values |
 * | `list` | an array | no schema puts one in an expression; the model admits it for completeness |
 *
 * Mixing the two numeric types is a `TypeError` in JavaScript, which is a feature here: every
 * place where Python's numeric tower promotes an `int` to a `float` has to be written down rather
 * than happening silently, and {@link arithmetic} is where they all are.
 *
 * `UNRESOLVED` is `expr.py`'s sentinel — "a value that cannot be decided statically: an external
 * quantity that no assignment supplies, or an operand that was refused". It is not an error:
 * each caller decides whether an unresolved value is acceptable at that point, and what is
 * forbidden is deciding silently (I7).
 */
import { isJsonArray, isJsonNumber, isJsonObject, lexemeDenotes, type JsonValue } from '../json/index.js';

import { PyKeyError, PyTypeError } from './errors.js';

/**
 * The sentinel `expr.py` propagates rather than replacing by a guess (I7).
 *
 * A symbol, so that no value a document can hold is ever equal to it — Python relies on the
 * identity of a bare `object()` for the same reason. It does not survive the structured clone
 * that carries a result out of the worker (plan §5.3); the worker protocol encodes it, as it
 * encodes a `bigint`'s distinction from a `number`.
 */
export const UNRESOLVED: unique symbol = Symbol('UNRESOLVED');

/** The type of {@link UNRESOLVED}. */
export type Unresolved = typeof UNRESOLVED;

/**
 * A record: an argument of the `record` kind, or any JSON object an expression reaches.
 *
 * A plain object, as `toPlain`'s reading of the tree is — not the ordered member list the tree
 * keeps (`json/tree.ts`), because nothing an evaluator does depends on the order of a record's
 * members and every caller of this side reads a member by name. The one thing a plain object
 * cannot hold is an integer-like member name in its authored place (`{"10": …, "2": …}`
 * enumerates as `2`, `10`); no name of the grammar can be one — an identifier and an argument
 * path both begin with a letter or an underscore — so no document reaches it, and a document
 * that did would be off the grammar before it reached meaning.
 */
export interface PyRecord {
  readonly [name: string]: PyValue;
}

/**
 * A value of the algebra, as CPython's `json` and the evaluators of `expr.py` see one — the
 * sentinel included.
 *
 * `UNRESOLVED` is a value of the same universe in the tools: `resolve_arguments` writes it into
 * the argument map of an instance whose argument was refused, `static_argument` leaves it inside
 * a record, and `primitive_value` finds it at the end of an argument path and propagates it. A
 * type that excluded it would say something the tools do not, so it is in, and
 * {@link isResolved} is how a caller narrows a value before reading it.
 */
export type PyValue =
  | null
  | boolean
  | bigint
  | number
  | string
  | Unresolved
  | readonly PyValue[]
  | PyRecord;

/** Every value but the sentinel: what a caller has after it has narrowed one. */
export type Resolved = Exclude<PyValue, Unresolved>;

/** Whether a value is a value and not the sentinel. */
export function isResolved(value: PyValue): value is Resolved {
  return value !== UNRESOLVED;
}

/**
 * The name Python's messages give a value's type, which is the name its refusals carry.
 *
 * `UNRESOLVED` answers `object`, as `type(object()).__name__` does.
 */
export function pythonTypeName(value: PyValue): string {
  if (value === UNRESOLVED) return 'object';
  if (value === null) return 'NoneType';
  if (typeof value === 'boolean') return 'bool';
  if (typeof value === 'bigint') return 'int';
  if (typeof value === 'number') return 'float';
  if (typeof value === 'string') return 'str';
  return Array.isArray(value) ? 'list' : 'dict';
}

/** Whether a value is a record — an object that is not an array. */
export function isRecord(value: PyValue): value is PyRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** Whether a value is one of Python's integers: an `int`, or a `bool`, which is one. */
export function isInteger(value: PyValue): value is bigint | boolean {
  return typeof value === 'bigint' || typeof value === 'boolean';
}

/**
 * A member of a record, as Python's `in` and `[]` read one: own members only.
 *
 * A record is a plain object here, as the plain reading of the tree is (`toPlain`), so it carries
 * `Object.prototype`; `'toString' in record` would be true of every record and `record['valueOf']`
 * would answer a function. Python's dictionaries have no such members, so every lookup on this
 * side asks for an own member and nothing else.
 */
export function member(record: PyRecord, name: string): PyValue | undefined {
  return Object.hasOwn(record, name) ? record[name] : undefined;
}

/** Whether a record has that member, as Python's `in` answers. */
export function hasKey(record: PyRecord, name: string): boolean {
  return Object.hasOwn(record, name);
}

/** A member the grammar requires; its absence is the `KeyError` the tools raise there. */
export function demandKey(record: PyRecord, name: string): PyValue {
  if (!Object.hasOwn(record, name)) throw new PyKeyError(`'${name}'`);
  return record[name] as PyValue;
}

/** A value read as a record; anything else is the refusal Python raises for `[…]` on it. */
export function asRecord(value: PyValue): PyRecord {
  if (isRecord(value)) return value;
  throw new PyTypeError(`'${pythonTypeName(value)}' object is not subscriptable`);
}

/** The members of a record, in order, as Python's `.items()` walks a dictionary. */
export function items(value: PyValue): [string, PyValue][] {
  const record = asRecord(value);
  return Object.keys(record).map((name) => [name, record[name] as PyValue]);
}

/**
 * Python's truth of a value, which is not JavaScript's.
 *
 * `expr.py` returns the `boolean` member of a boolean condition as it stands and its callers
 * write `if not truth` and `if truth`, so the truth taken is Python's: zero, the empty string,
 * `None`, an empty container are false and everything else — `nan` included — is true. An empty
 * array and an empty record are the two places JavaScript disagrees, so no caller of the
 * evaluators tests a condition's answer with `if (…)`; it calls this.
 */
export function truthy(value: PyValue): boolean {
  // `UNRESOLVED` is a bare `object()` in the tools, and every object is true there.
  if (value === UNRESOLVED) return true;
  if (value === null || value === false) return false;
  if (typeof value === 'boolean') return value;
  if (typeof value === 'bigint') return value !== 0n;
  if (typeof value === 'number') return value !== 0;
  if (typeof value === 'string') return value.length > 0;
  if (Array.isArray(value)) return value.length > 0;
  return Object.keys(value).length > 0;
}

/**
 * The tree of `json/` as the evaluators read it: the reading CPython's `json.load` gives.
 *
 * Numbers are the one thing that changes. A number the parser read keeps its text, and V3's
 * lexical rule already decided from it whether it is a whole number or a real (`json/number.ts`);
 * the same rule decides here between an `int` and a `float`. The text is used when it denotes
 * exactly this value, as the serializer uses it, so a value changed without its text is read from
 * the value; and a value that is not a whole number is a `float` whatever the node claims,
 * because there is no integer text that denotes it — which is what leaves V3 a `1.5` to refuse
 * rather than an exception to raise.
 */
export function toPython(value: JsonValue): PyValue {
  if (isJsonNumber(value)) {
    if (value.real || !Number.isFinite(value.value) || !Number.isInteger(value.value)) {
      return value.value;
    }
    const lexeme = value.lexeme;
    if (lexeme !== undefined && lexemeDenotes(lexeme, value.value, false)) return BigInt(lexeme);
    return BigInt(value.value);
  }
  if (isJsonArray(value)) return value.map(toPython);
  if (isJsonObject(value)) {
    const record: Record<string, PyValue> = {};
    for (const one of value.members) {
      // `__proto__` is a member like any other to CPython's `json`, and the model schema's
      // `propertyNames` admits the name; plain assignment would set the prototype instead
      // (`json/tree.ts` states the whole rule).
      if (one.name === '__proto__') {
        Object.defineProperty(record, one.name, {
          value: toPython(one.value),
          writable: true,
          enumerable: true,
          configurable: true,
        });
      } else {
        record[one.name] = toPython(one.value);
      }
    }
    return record;
  }
  return value;
}

/**
 * A value back as a node of the tree: the inverse of {@link toPython}, for the derived products,
 * which are written from computed values.
 *
 * An integer is written bare and a float with a fraction or an exponent, which is the rule D12
 * states and the rule V3 reads — so a derived extent of `4096` and one of `4096.0` are different
 * documents, as they are on the Python side.
 */
export function toJsonValue(value: PyValue): JsonValue {
  if (value === UNRESOLVED) {
    throw new PyTypeError('UNRESOLVED has no JSON form: a document holds decided values only');
  }
  if (typeof value === 'bigint') return { kind: 'number', value: Number(value), real: false };
  if (typeof value === 'number') return { kind: 'number', value, real: true };
  if (Array.isArray(value)) return value.map(toJsonValue);
  if (isRecord(value)) {
    return {
      kind: 'object',
      members: Object.keys(value).map((name) => ({
        name,
        value: toJsonValue(value[name] as PyValue),
      })),
    };
  }
  return value as JsonValue;
}

/**
 * The record an expression node is, or `undefined` for a value that is no node at all.
 *
 * Python reads a node with `in`, which is a member test on a dictionary, a membership test on a
 * list and a substring test on a string: a list and a string therefore fall through every branch
 * of `primitive_value` and `model_value` and reach the closing `return UNRESOLVED`, while a
 * number or `None` raises on the first test. The port answers the same, so that a document off
 * the grammar — which the schema stage has already refused — is refused here the same way.
 */
export function expressionNode(value: PyValue): PyRecord | undefined {
  if (isRecord(value)) return value;
  if (typeof value === 'string' || Array.isArray(value)) return undefined;
  throw new PyTypeError(`argument of type '${pythonTypeName(value)}' is not iterable`);
}

/**
 * The record a condition node is; every other value raises, as it does in the tools.
 *
 * A condition has no closing `return UNRESOLVED`: it ends at `c['compare']`, so a string or a
 * list reaches a subscript that raises rather than falling through.
 */
export function conditionNode(value: PyValue): PyRecord {
  if (isRecord(value)) return value;
  if (typeof value === 'string' || Array.isArray(value)) {
    throw new PyTypeError(`${pythonTypeName(value)} indices must be integers`);
  }
  throw new PyTypeError(`argument of type '${pythonTypeName(value)}' is not iterable`);
}

/**
 * What Python's `for x in value` walks: a list's elements, a record's member names, a string's
 * characters. Anything else raises, as Python's iteration protocol does.
 *
 * The evaluators iterate an operator's `args` and a condition's `all` and `any`, and the schemas
 * make all three lists — but a document off the grammar reaches those loops too, and Python's
 * answer there (a record iterated as its names, each of which is a string and so not an
 * expression) is the answer the port gives, rather than a judgement of its own.
 */
export function pyIterate(value: PyValue): PyValue[] {
  if (Array.isArray(value)) return [...(value as readonly PyValue[])];
  if (typeof value === 'string') return [...value];
  if (isRecord(value)) return Object.keys(value);
  throw new PyTypeError(`'${pythonTypeName(value)}' object is not iterable`);
}
