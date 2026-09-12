/**
 * The shapes the schema stage works with: a schema node, a place in an instance, a place in a
 * schema.
 *
 * A schema is read, never edited, so it is plain JavaScript data — `JSON.parse` of the file — and
 * not a document tree: nothing here has to survive a round trip to bytes. An *instance* is the
 * other way round: it is a document or a unit the editor holds, so it is the ordered tree of
 * `json/tree.ts`, which keeps member order and each number's float-ness (D12). The two travel
 * together through the walk, because Ajv reads plain data and the messages read the tree.
 */
import { jsonPointerOf } from '../json/pointer.js';
import { toPlain, type JsonValue } from '../json/tree.js';
import { comparePythonStrings } from './repr.js';

/** A schema object: the keywords of one node of a schema file, in the order the file writes them. */
export interface SchemaObject {
  readonly [keyword: string]: unknown;
}

/** A schema node. `true` and `false` are schemas of their own in 2020-12. */
export type SchemaNode = boolean | SchemaObject;

/** Whether a node is an object rather than one of the two boolean schemas. */
export function isSchemaObject(node: SchemaNode): node is SchemaObject {
  return typeof node === 'object' && node !== null;
}

/** One step of a path: a member name in an object, an index in an array. */
export type PathSegment = string | number;

/**
 * A place in an instance, carried as both readings at once.
 *
 * `tree` is what a message is written from — a number there knows whether the document wrote
 * `16` or `16.0`, and an object knows its member order — and `plain` is what Ajv is handed, since
 * Ajv reads JavaScript data. They denote the same value; `at` walks both in step.
 */
export interface Instance {
  readonly tree: JsonValue;
  readonly plain: unknown;
}

/** An instance from a tree, its plain reading computed once for the whole document. */
export function instanceOf(tree: JsonValue): Instance {
  return { tree, plain: toPlain(tree) };
}

/** The member or element of an instance, both readings kept in step. */
export function at(instance: Instance, segment: PathSegment, tree: JsonValue): Instance {
  const container = instance.plain;
  let plain: unknown = undefined;
  if (typeof segment === 'number') {
    plain = Array.isArray(container) ? container[segment] : undefined;
  } else if (typeof container === 'object' && container !== null) {
    plain = (container as Record<string, unknown>)[segment];
  }
  return { tree, plain };
}

/** A bare value that is its own instance: a member name under `propertyNames`, for one. */
export function literalInstance(value: string): Instance {
  return { tree: value, plain: value };
}

/**
 * A JSON pointer (RFC 6901) from a path: `''` at the root, `/instances/embed`, `~0` and `~1` for
 * the two characters a pointer escapes.
 */
export function pointerOf(path: readonly PathSegment[]): string {
  return jsonPointerOf(path);
}

/**
 * The tools' rendering of a place: the path's segments joined by slashes, `<root>` when there is
 * none. `tools/schema.py`'s `format_error` writes it, and it is what `--validate` prints before
 * the message, so it is part of the wording contract and not a pointer.
 */
export function whereOf(path: readonly PathSegment[]): string {
  return path.length === 0 ? '<root>' : path.map((segment) => String(segment)).join('/');
}

/**
 * The order `sorted(errors, key=lambda e: list(e.absolute_path))` puts two errors in.
 *
 * Python compares the paths element by element and refuses to compare a member name with an
 * index — `sorted` raises `TypeError` on a document whose errors sit at `a/0` and `a/b`. Nothing
 * in the repository produces such a pair, and raising is not an answer the editor can give, so
 * indices are ordered before names; every other case is Python's own.
 */
export function comparePaths(left: readonly PathSegment[], right: readonly PathSegment[]): number {
  const shared = Math.min(left.length, right.length);
  for (let index = 0; index < shared; index += 1) {
    const a = left[index] as PathSegment;
    const b = right[index] as PathSegment;
    if (typeof a === 'number' && typeof b === 'number') {
      if (a !== b) return a < b ? -1 : 1;
    } else if (typeof a === 'string' && typeof b === 'string') {
      const order = comparePythonStrings(a, b);
      if (order !== 0) return order;
    } else {
      return typeof a === 'number' ? -1 : 1;
    }
  }
  return left.length - right.length;
}
