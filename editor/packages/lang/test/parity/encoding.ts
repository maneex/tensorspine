import {
  formatNumber,
  isRecord,
  member,
  UNRESOLVED,
  type PyRecord,
  type PyValue,
} from '../../src/index.js';

/**
 * The tagged encoding the oracle writes its values in, and the reading back of it.
 *
 * "Values are tagged and written as text (`{"int": "4096"}`, `{"float": "1e-05"}`) and compared as
 * encodings, never as values, because `1` and `1.0` are the same double and only the encoding
 * tells them apart" — feature 1.2's finding, and the reason every parity suite that compares a
 * *value* rather than a message goes through here. It is `generate.py`'s `encode`, inverted.
 */

/** `repr(float)`, which is what the fixture writes; the three non-finite names included. */
export function pythonFloat(value: number): string {
  if (Number.isFinite(value)) return formatNumber(value, true);
  if (Number.isNaN(value)) return 'nan';
  return value > 0 ? 'inf' : '-inf';
}

/** A value as the fixture encodes one — the comparison is over these, never over the values. */
export function encode(value: PyValue): unknown {
  if (value === UNRESOLVED) return { unresolved: true };
  if (value === null) return { none: true };
  if (typeof value === 'boolean') return { bool: value };
  if (typeof value === 'bigint') return { int: value.toString() };
  if (typeof value === 'number') return { float: pythonFloat(value) };
  if (typeof value === 'string') return { str: value };
  if (Array.isArray(value)) return { list: value.map(encode) };
  const record: Record<string, unknown> = {};
  for (const [name, one] of Object.entries(value as PyRecord)) record[name] = encode(one);
  return { record };
}

/** A map of values as the fixture encodes one: `generate.py`'s `encode_map`. */
export function encodeMap(values: ReadonlyMap<string, PyValue>): Record<string, unknown> {
  // `Object.fromEntries` defines properties rather than assigning them, so a quantity called
  // `__proto__` — which the identifier pattern admits — lands as a member like any other.
  return Object.fromEntries([...values].map(([name, one]) => [name, encode(one)]));
}

/** The value an encoding stands for. */
export function decode(encoded: PyValue): PyValue {
  if (!isRecord(encoded)) throw new Error(`not an encoded value: ${JSON.stringify(encoded)}`);
  const text = (name: string): string => member(encoded, name) as string;
  if (member(encoded, 'unresolved') !== undefined) return UNRESOLVED;
  if (member(encoded, 'none') !== undefined) return null;
  if (member(encoded, 'bool') !== undefined) return member(encoded, 'bool') as boolean;
  if (member(encoded, 'int') !== undefined) return BigInt(text('int'));
  if (member(encoded, 'float') !== undefined) {
    const written = text('float');
    if (written === 'nan') return Number.NaN;
    if (written === 'inf') return Number.POSITIVE_INFINITY;
    if (written === '-inf') return Number.NEGATIVE_INFINITY;
    return Number(written);
  }
  if (member(encoded, 'str') !== undefined) return text('str');
  if (member(encoded, 'list') !== undefined) {
    return (member(encoded, 'list') as readonly PyValue[]).map(decode);
  }
  const values = member(encoded, 'record') as PyRecord;
  const record: Record<string, PyValue> = {};
  for (const [name, one] of Object.entries(values)) record[name] = decode(one);
  return record;
}

/** An encoded map as the evaluators take one. */
export function decodeMap(encoded: PyRecord): Map<string, PyValue> {
  return new Map(Object.entries(encoded).map(([name, one]) => [name, decode(one)]));
}

/** An encoded map as a record, which is the shape an assignment and an argument map have. */
export function decodeRecord(encoded: PyRecord): PyRecord {
  const record: Record<string, PyValue> = {};
  for (const [name, one] of Object.entries(encoded)) record[name] = decode(one);
  return record;
}
