/**
 * `repr` of a value of the algebra.
 *
 * `schema/repr.ts` writes a *tree node* or a plain JavaScript value as CPython writes it, because
 * that is what the schema stage interpolates into its messages. The loader interpolates something
 * else: values as the evaluators hold them (`expr/value.ts`), where Python's `int` is a `bigint`
 * and its `float` a `number` — `default 1e-05 is not among […]`, `version '1.0.2'`,
 * `{'kind': 'window', 'span': 4096}`.
 *
 * The two readings differ on exactly one value: a whole `number`. `pythonRepr` guesses from
 * `Number.isInteger`, because a schema file's numbers carry no float-ness; here there is nothing
 * to guess — a `number` is a float and a `bigint` is an integer — so `4.0` is written `4.0` and
 * not `4`, which is the distinction V3 reads and D12 writes.
 */
import { pythonReprString } from '../schema/repr.js';
import { formatNumber } from '../json/number.js';
import { isRecord, UNRESOLVED, type PyValue } from '../expr/value.js';

/** A value of the algebra as `repr` writes it. */
export function pyRepr(value: PyValue): string {
  if (value === UNRESOLVED) {
    // `expr.py`'s sentinel is a bare `object()`; its `repr` carries an address no port can have.
    return '<object object>';
  }
  if (value === null) return 'None';
  if (value === true) return 'True';
  if (value === false) return 'False';
  if (typeof value === 'bigint') return value.toString();
  if (typeof value === 'number') {
    if (Number.isNaN(value)) return 'nan';
    if (value === Infinity) return 'inf';
    if (value === -Infinity) return '-inf';
    return formatNumber(value, true);
  }
  if (typeof value === 'string') return pythonReprString(value);
  if (Array.isArray(value)) {
    return `[${(value as readonly PyValue[]).map((one) => pyRepr(one)).join(', ')}]`;
  }
  if (isRecord(value)) {
    const members = Object.keys(value).map(
      (name) => `${pythonReprString(name)}: ${pyRepr(value[name] as PyValue)}`,
    );
    return `{${members.join(', ')}}`;
  }
  // `PyValue` has no other alternative; a caller that reached here handed in something else.
  throw new TypeError('a value of the algebra has no other form');
}

/**
 * A value interpolated with `str` rather than `repr` — Python's `f"{v}"`.
 *
 * The two differ for a string, which `str` writes bare: `_provide` writes an axis identity as
 * `identity model.width` and a primitive identity, being a tuple, as `('attention.dense',
 * '1.0.0')`, the elements of a container always being written with `repr`.
 */
export function pyStr(value: PyValue): string {
  return typeof value === 'string' ? value : pyRepr(value);
}
