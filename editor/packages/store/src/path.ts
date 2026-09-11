/**
 * Places in the document tree, and the two ways the store gets at one.
 *
 * The document is the model (plan D1): every gesture is an edit at a *place*, and a place is a
 * path of steps — a member name in an object, an index in an array. The same path is rendered
 * three ways, and each rendering has one reader:
 *
 * - **the path itself** (`['compositions', 'decoder', 'instances', 'attn']`) is what a command
 *   takes and what the store navigates with;
 * - **the JSON pointer** (`/compositions/decoder/instances/attn`) is what a problem carries
 *   (the core emits one beside every message, plan §3) and what a source range is found by;
 * - **the key** (`compositions/decoder/instances/attn`) is what the layout sidecar is keyed by
 *   (plan D6, §5.5), which is why it is written here and not invented beside the sidecar.
 *
 * The step type and the pointer are the core's (`PathSegment`, `pointerOf`): one rendering of a
 * place in the workspace, not two.
 *
 * The tree is plain data (feature 0.3), so navigating it is a walk over `members` and array
 * indices. This module walks the frozen tree a command *reads*; `./draft.ts` walks the draft it
 * *writes*, and states there why the two readings cannot be one function.
 */
import {
  isJsonArray,
  isJsonObject,
  pointerOf,
  type JsonArray,
  type JsonObject,
  type JsonValue,
  type PathSegment,
} from '@tensorspine/lang';

/** One step of a path: a member name in an object, an index in an array. */
export type Step = PathSegment;

/** A place in the document, from the root. */
export type Path = readonly Step[];

/**
 * A JSON pointer (RFC 6901) for a place in the document: `''` at the root.
 *
 * The core's own, so that a path the store built and a pointer the core emitted are the same
 * string for the same place.
 */
export { pointerOf };

/**
 * The key the layout sidecar uses for a place: the steps joined by slashes (plan D6 —
 * `instances/embed`, `compositions/decoder/instances/attn`, `interfaces/inputs/tokens`).
 *
 * The root has no key: nothing in the sidecar is positioned at the document itself.
 */
export function keyOf(path: Path): string {
  if (path.length === 0) throw new TypeError('the root has no sidecar key');
  return path.map((step) => String(step)).join('/');
}

/** The steps of a sidecar key, as a path. A step that reads as an index is one. */
export function pathOfKey(key: string): Path {
  return key.split('/').map((step) => (/^(0|[1-9][0-9]*)$/.test(step) ? Number(step) : step));
}

/** Whether `inner` is `outer` itself or a place below it. */
export function isUnder(inner: Path, outer: Path): boolean {
  if (inner.length < outer.length) return false;
  return outer.every((step, position) => inner[position] === step);
}

/** The path with one more step. */
export function child(path: Path, step: Step): Path {
  return [...path, step];
}

/** The place of the container: the path without its last step. */
export function parentOf(path: Path): Path {
  if (path.length === 0) throw new TypeError('the root has no container');
  return path.slice(0, -1);
}

/** The last step of a path — the name or index the place is known by. */
export function lastOf(path: Path): Step {
  const last = path[path.length - 1];
  if (last === undefined) throw new TypeError('the root has no last step');
  return last;
}

/** One step down from a value of the frozen tree, or `undefined` where the step names nothing. */
export function stepInto(value: JsonValue, step: Step): JsonValue | undefined {
  if (typeof step === 'number') {
    return isJsonArray(value) ? value[step] : undefined;
  }
  if (!isJsonObject(value)) return undefined;
  return value.members.find((member) => member.name === step)?.value;
}

/** The value at a place of the frozen tree, or `undefined` where the path names nothing. */
export function nodeAt(tree: JsonValue, path: Path): JsonValue | undefined {
  let current: JsonValue | undefined = tree;
  for (const step of path) {
    if (current === undefined) return undefined;
    current = stepInto(current, step);
  }
  return current;
}

/** Whether the path names something in the tree — what the sidecar's pruning asks (D6). */
export function exists(tree: JsonValue, path: Path): boolean {
  return nodeAt(tree, path) !== undefined;
}

/** The object at a place, or `undefined` where the place is not an object. */
export function objectAt(tree: JsonValue, path: Path): JsonObject | undefined {
  const node = nodeAt(tree, path);
  return node !== undefined && isJsonObject(node) ? node : undefined;
}

/** The array at a place, or `undefined` where the place is not an array. */
export function arrayAt(tree: JsonValue, path: Path): JsonArray | undefined {
  const node = nodeAt(tree, path);
  return node !== undefined && isJsonArray(node) ? node : undefined;
}

/**
 * Raised when a command names a place the document has not, or of the wrong shape.
 *
 * It is not a verdict on the document — those are the core's, and land in Problems (plan §3).
 * It is the store saying that the gesture it was handed does not fit the tree it holds, which is
 * a defect of the caller and never something a user can type.
 */
export class EditError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'EditError';
  }
}
