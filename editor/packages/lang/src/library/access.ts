/**
 * Reading a document the way the loader's Python reads one.
 *
 * `tools/primitive_library.py` mixes three readings of a member, and they differ exactly where a
 * document is off the grammar: `d['x']` raises, `d.get('x')` answers `None`, and `'x' in d` asks.
 * Which one a line uses is part of what it does, so the three are here rather than folded into
 * one — the loader raises where the tools raise, and answers `None` where they answer it.
 *
 * `get` is the fourth, and it is the port's own: a `.get` that tolerates a container which is not
 * a dictionary at all. It is used where the value being read has already crossed the schema stage
 * and the tolerance only decides what an unreachable state does — never where the tools' own
 * refusal depends on the difference.
 */
import {
  hasKey,
  isRecord,
  member,
  pyIterate,
  type PyRecord,
  type PyValue,
} from '../expr/value.js';
import { PyKeyError, PyTypeError } from '../expr/errors.js';

/** `d['name']`: the member, or the `KeyError` the tools raise for a document off the grammar. */
export function demand(value: PyValue, name: string): PyValue {
  if (!isRecord(value)) {
    throw new PyTypeError(`'${pythonKind(value)}' object is not subscriptable`);
  }
  if (!hasKey(value, name)) throw new PyKeyError(`'${name}'`);
  return member(value, name) as PyValue;
}

/** `d.get('name', fallback)`, on something the grammar makes a dictionary. */
export function optional(value: PyValue, name: string, fallback: PyValue): PyValue {
  if (!isRecord(value)) {
    throw new PyTypeError(`'${pythonKind(value)}' object has no attribute 'get'`);
  }
  return hasKey(value, name) ? (member(value, name) as PyValue) : fallback;
}

/** `'name' in d`, as Python reads it on a dictionary. */
export function has(value: PyValue, name: string): boolean {
  return isRecord(value) && hasKey(value, name);
}

/** A member, `null` when there is none — and `null` when the container is not a record either. */
export function get(value: PyValue, name: string): PyValue {
  return isRecord(value) && hasKey(value, name) ? (member(value, name) as PyValue) : null;
}

/** `d.items()` over a map the grammar keys by an authored name, in declaration order. */
export function entries(value: PyValue): [string, PyValue][] {
  if (!isRecord(value)) {
    throw new PyTypeError(`'${pythonKind(value)}' object has no attribute 'items'`);
  }
  return Object.keys(value).map((name) => [name, member(value, name) as PyValue]);
}

/** The members of a record, or nothing at all when the value is not one. */
export function members(value: PyValue): [string, PyValue][] {
  return isRecord(value) ? entries(value) : [];
}

/** `for x in value`: a list's elements, a record's names, a string's characters. */
export function listOf(value: PyValue): PyValue[] {
  return pyIterate(value);
}

/** A value read as a record, or an empty one: for a place the grammar makes an object. */
export function asRecord(value: PyValue): PyRecord {
  return isRecord(value) ? value : {};
}

/** The name Python's messages give the container's type. */
function pythonKind(value: PyValue): string {
  if (value === null) return 'NoneType';
  if (typeof value === 'boolean') return 'bool';
  if (typeof value === 'bigint') return 'int';
  if (typeof value === 'number') return 'float';
  if (typeof value === 'string') return 'str';
  return Array.isArray(value) ? 'list' : 'dict';
}
