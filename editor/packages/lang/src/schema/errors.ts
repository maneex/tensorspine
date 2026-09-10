/**
 * The error record the structural stage builds, and the two selections the tools make over it.
 *
 * `jsonschema`'s errors are a tree, not a list: `oneOf` and `anyOf` report one error at the
 * branch point and hang the branches' own errors under it as `context`. The tools read that tree
 * in two different ways, and the core has to reproduce both, because they are what the parity
 * contract is written against:
 *
 * - **A model document** (`validate.structural` → `schema.check`) keeps the top-level errors as
 *   they are, sorted by their place in the document. A failed `oneOf` therefore reads as
 *   `{'context': 'analysis.heads'} is not valid under any of the given schemas` — one line, the
 *   branch point, the branches not shown.
 * - **A library unit** (`primitive_library._units` → `schema.deepest`) replaces each top-level
 *   error by the leaf `best_match` descends to, so a failed `oneOf` reads as the cause inside the
 *   branch: `definition/state_ports/kv/rules/4/evolution: 'apend' is not one of [...]`.
 *
 * `best_match` and `relevance` are heuristics whose exact shape decides which leaf is named, so
 * they are ported as they are written — `max` for the choice among siblings, the two smallest for
 * the descent, and the tie that stops the descent — rather than paraphrased.
 */
import { isJsonArray, isJsonNumber, isJsonObject, type JsonValue } from '../json/tree.js';
import { comparePaths, isSchemaObject, type PathSegment, type SchemaNode } from './types.js';

/**
 * One structural error, in the shape `jsonschema` gives it.
 *
 * `path` and `schemaPath` are relative: the walk prepends a segment at every level it returns
 * through, exactly as `descend` does, so an error's own path is relative to the branch point it
 * hangs under and {@link absolutePath} is what names a place in the document.
 */
export interface SchemaError {
  /** The message, in the tools' words. */
  message: string;
  /** The keyword that failed; `null` for the `false` schema, which has none. */
  keyword: string | null;
  /** The keyword's value, as the schema writes it. */
  keywordValue: unknown;
  /** Where in the instance, relative to the error's parent. */
  path: PathSegment[];
  /** Where in the schema, relative to the parent's schema place. */
  schemaPath: PathSegment[];
  /** The branches' errors, when this error is a `oneOf` or an `anyOf`. */
  context: SchemaError[];
  /** The error this one is a branch error of, `null` at the top level. */
  parent: SchemaError | null;
  /** The instance the keyword judged, as a tree node: what a message is written from. */
  instance: JsonValue;
  /** The schema node the keyword belongs to. */
  schema: SchemaNode;
}

/** An error, with the fields the walk fills in afterwards defaulted. */
export function schemaError(
  message: string,
  fields: Partial<Omit<SchemaError, 'message'>> = {},
): SchemaError {
  return {
    message,
    keyword: fields.keyword ?? null,
    keywordValue: fields.keywordValue,
    path: fields.path ?? [],
    schemaPath: fields.schemaPath ?? [],
    context: fields.context ?? [],
    parent: fields.parent ?? null,
    instance: fields.instance ?? null,
    schema: fields.schema ?? true,
  };
}

/** The place in the document an error names: its own path behind every parent's. */
export function absolutePath(error: SchemaError): PathSegment[] {
  const path: PathSegment[] = [];
  for (let current: SchemaError | null = error; current !== null; current = current.parent) {
    path.unshift(...current.path);
  }
  return path;
}

/**
 * `is_type` of the 2020-12 type checker, on a tree node.
 *
 * It is asked one question only — whether an error's instance is of the type its schema declares —
 * and that question ranks errors; it never decides conformance, which is Ajv's (D4). Note that
 * 2020-12 calls a float with no fractional part an integer, which is also what JavaScript's one
 * number type gives; `inf` is not one, in either reading.
 */
export function isType(instance: JsonValue, type: string): boolean {
  switch (type) {
    case 'array':
      return isJsonArray(instance);
    case 'boolean':
      return instance === true || instance === false;
    case 'integer':
      return isJsonNumber(instance) && Number.isInteger(instance.value);
    case 'null':
      return instance === null;
    case 'number':
      return isJsonNumber(instance);
    case 'object':
      return isJsonObject(instance);
    case 'string':
      return typeof instance === 'string';
    default:
      return false;
  }
}

/** `_matches_type`: whether the error's schema declares a type its instance has. */
function matchesType(error: SchemaError): boolean {
  if (!isSchemaObject(error.schema)) return false;
  const expected = error.schema['type'];
  if (typeof expected === 'string') return isType(error.instance, expected);
  if (Array.isArray(expected)) {
    return expected.some((one) => typeof one === 'string' && isType(error.instance, one));
  }
  return false;
}

/** The keywords `by_relevance` calls weak: a `oneOf` or an `anyOf` yields to a sibling. */
const WEAK: ReadonlySet<string> = new Set(['anyOf', 'oneOf']);

/**
 * `relevance`, as a comparison: negative when `left` is the less relevant of the two.
 *
 * The key `jsonschema` sorts by is `(-len(path), path, validator not in weak, validator in
 * strong, not matches_type)`, and larger is better. The fourth component is dropped here because
 * the tools' strong set is empty, so it takes the same value for every error and separates none.
 */
export function compareRelevance(left: SchemaError, right: SchemaError): number {
  if (left.path.length !== right.path.length) return right.path.length - left.path.length;
  const byPath = comparePaths(left.path, right.path);
  if (byPath !== 0) return byPath;
  const leftNotWeak = WEAK.has(left.keyword ?? '') ? 0 : 1;
  const rightNotWeak = WEAK.has(right.keyword ?? '') ? 0 : 1;
  if (leftNotWeak !== rightNotWeak) return leftNotWeak - rightNotWeak;
  const leftTyped = matchesType(left) ? 0 : 1;
  const rightTyped = matchesType(right) ? 0 : 1;
  return leftTyped - rightTyped;
}

/**
 * `best_match`: the error that says most about a refusal.
 *
 * The most relevant of the errors given — the shallowest — and then, while that error has
 * branches, the least relevant of its branches, which is the deepest: `oneOf` only has to match
 * once, so the branch that came closest is the one worth naming. The descent stops when the two
 * least relevant branches are equally relevant, since then no branch came closer than another.
 */
export function bestMatch(errors: readonly SchemaError[]): SchemaError | undefined {
  let best: SchemaError | undefined;
  for (const error of errors) {
    if (best === undefined || compareRelevance(error, best) > 0) best = error;
  }
  if (best === undefined) return undefined;
  while (best.context.length > 0) {
    const smallest = [...best.context].sort(compareRelevance).slice(0, 2);
    const first = smallest[0] as SchemaError;
    const second = smallest[1];
    if (second !== undefined && compareRelevance(first, second) === 0) return best;
    best = first;
  }
  return best;
}

/**
 * `schema.deepest`: each top-level error replaced by the leaf behind it, sorted by place.
 *
 * This is what the library loader prints for a unit that is off the unit schema — the cause a
 * reader can act on, rather than the branch point that noticed it.
 */
export function deepest(errors: readonly SchemaError[]): SchemaError[] {
  const leaves: SchemaError[] = [];
  for (const error of errors) {
    let leaf = bestMatch([error]);
    while (leaf !== undefined && leaf.context.length > 0) leaf = bestMatch(leaf.context);
    if (leaf !== undefined) leaves.push(leaf);
  }
  return sortByPlace(leaves);
}

/** The order `schema.check` returns errors in: by their place in the document, ties as found. */
export function sortByPlace(errors: readonly SchemaError[]): SchemaError[] {
  return [...errors].sort((left, right) => comparePaths(absolutePath(left), absolutePath(right)));
}
