/**
 * What crosses the worker boundary, and what cannot cross it as it stands.
 *
 * The core answers plain data on purpose — "it survives the structured clone into the worker"
 * (plan §5.3, and features 0.3, 1.7 and 1.6d each state it of their own answer) — and the
 * structured clone carries almost all of it unchanged. Measured on this repository, three things
 * do not:
 *
 * 1. **`UNRESOLVED` is a symbol**, and a symbol is a `DataCloneError`. Feature 1.2 chose the
 *    symbol so that no value a document can hold is ever equal to the sentinel, and wrote down
 *    what it costs: "it does not survive the structured clone that carries a result out of the
 *    worker (plan §5.3); the worker protocol encodes it". This is that encoding.
 * 2. **An `Error` subclass loses its class.** `structuredClone(new PyTypeError(…))` comes back as
 *    a plain `Error` whose `name` is `'Error'` and whose own properties are gone, so a refusal
 *    sent as an exception would arrive stripped of everything a caller reads — the name the tools
 *    raise it under, and the lines a {@link DerivedSchemaError} carries. {@link encodeFailure}
 *    writes them down instead.
 * 3. **A function is a `DataCloneError`**, which is why a {@link SchemaRegistry} and a
 *    {@link LibrarySource} are held in the worker and named by a handle rather than sent.
 *
 * The encoding is a bijection and is proved to be one: every object that already carries the
 * reserved member is escaped, so a document whose own data spells the marker comes back as it
 * was. It allocates nothing where nothing is encoded — the common case, since no corpus document
 * produces an `UNRESOLVED` anywhere in its facts or its products — because a container whose
 * every member encodes to itself is returned as it stands.
 */
import { put } from '../json/tree.js';
import { UNRESOLVED } from '../expr/value.js';

/**
 * The one member name the encoding reserves.
 *
 * `@` is outside the identifier grammar of §5.2 (`^[A-Za-z_][A-Za-z0-9_-]*$`), so no name the
 * model or the unit schema admits can be it — but a safetensors `__metadata__` map and any
 * `additionalProperties` the schemas leave open can carry anything at all, so the encoding does
 * not rely on that and escapes every object that carries the member.
 */
const MARK = '@';

/** {@link MARK}'s value in the sentinel's encoding. */
const UNRESOLVED_TAG = 'unresolved';

/** {@link MARK}'s value in the escape of an object that carries {@link MARK} itself. */
const ESCAPED_TAG = 'escaped';

/** The member the escape keeps the original object under. */
const ESCAPED_VALUE = 'value';

/** A plain object: what the core builds a record with, and nothing else. */
function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (typeof value !== 'object' || value === null) return false;
  const prototype: unknown = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

/**
 * A value ready for `postMessage`: the sentinel replaced, everything else as it stands.
 *
 * `Map`, `Set`, arrays and plain objects are walked; every other cloneable value — a `bigint`, a
 * `Uint8Array`, a string, `null` — is left alone, which is what keeps the walk cheap.
 */
export function encode(value: unknown): unknown {
  if (value === UNRESOLVED) return { [MARK]: UNRESOLVED_TAG };
  if (Array.isArray(value)) {
    const answer = (value as unknown[]).map((one) => encode(one));
    return answer.every((one, index) => one === value[index]) ? value : answer;
  }
  if (value instanceof Map) {
    let moved = false;
    const answer = new Map<unknown, unknown>();
    for (const [key, one] of value) {
      const k = encode(key);
      const v = encode(one);
      if (k !== key || v !== one) moved = true;
      answer.set(k, v);
    }
    return moved ? answer : value;
  }
  if (value instanceof Set) {
    let moved = false;
    const answer = new Set<unknown>();
    for (const one of value) {
      const v = encode(one);
      if (v !== one) moved = true;
      answer.add(v);
    }
    return moved ? answer : value;
  }
  if (isPlainObject(value)) {
    const escaped = Object.hasOwn(value, MARK);
    let moved = escaped;
    const body: Record<string, unknown> = {};
    for (const name of Object.keys(value)) {
      const one = value[name];
      const encoded = encode(one);
      if (encoded !== one) moved = true;
      put(body, name, encoded);
    }
    if (!moved) return value;
    return escaped ? { [MARK]: ESCAPED_TAG, [ESCAPED_VALUE]: body } : body;
  }
  return value;
}

/** The inverse of {@link encode}: what comes off `postMessage`, read back as the core answers it. */
export function decode(value: unknown): unknown {
  if (Array.isArray(value)) {
    const answer = (value as unknown[]).map((one) => decode(one));
    return answer.every((one, index) => one === value[index]) ? value : answer;
  }
  if (value instanceof Map) {
    let moved = false;
    const answer = new Map<unknown, unknown>();
    for (const [key, one] of value) {
      const k = decode(key);
      const v = decode(one);
      if (k !== key || v !== one) moved = true;
      answer.set(k, v);
    }
    return moved ? answer : value;
  }
  if (value instanceof Set) {
    let moved = false;
    const answer = new Set<unknown>();
    for (const one of value) {
      const v = decode(one);
      if (v !== one) moved = true;
      answer.add(v);
    }
    return moved ? answer : value;
  }
  if (isPlainObject(value)) {
    const names = Object.keys(value);
    if (names.length === 1 && names[0] === MARK && value[MARK] === UNRESOLVED_TAG) {
      return UNRESOLVED;
    }
    if (
      names.length === 2 &&
      value[MARK] === ESCAPED_TAG &&
      Object.hasOwn(value, ESCAPED_VALUE) &&
      isPlainObject(value[ESCAPED_VALUE])
    ) {
      return decodeMembers(value[ESCAPED_VALUE]);
    }
    return decodeMembers(value);
  }
  return value;
}

/** Every member decoded, the object itself taken literally — the escape's reading. */
function decodeMembers(value: Record<string, unknown>): Record<string, unknown> {
  let moved = false;
  const body: Record<string, unknown> = {};
  for (const name of Object.keys(value)) {
    const one = value[name];
    const decoded = decode(one);
    if (decoded !== one) moved = true;
    put(body, name, decoded);
  }
  return moved ? body : value;
}

/**
 * What a refusal looks like on the wire.
 *
 * The core's refusals are classes — `PyTypeError`, `ModelError`, `PrimitiveLibraryError`,
 * `DerivedSchemaError`, `HeaderError`, `JsonParseError` — and a structured clone keeps none of
 * that: it answers a plain `Error` named `Error` with no own property. So the name the core
 * raises under travels as data, and so does what the caller reads off the refusal; the client
 * rebuilds a {@link LangFailure} from it.
 */
export interface EncodedFailure {
  /** The `name` the core's class sets: `PyTypeError`, `DerivedSchemaError`, … */
  readonly name: string;
  /** `error.message`, word for word — the tools' wording where the refusal is theirs. */
  readonly message: string;
  /** The stack as the worker saw it, for the log; absent where the platform gives none. */
  readonly stack?: string;
  /**
   * Whatever else the refusal carries, encoded: `DerivedSchemaError`'s `problems` and `document`,
   * `JsonParseError`'s `reason`, `HeaderError`'s `file`. Own enumerable properties alone, which is
   * exactly what a class of the core puts there.
   */
  readonly detail?: Record<string, unknown>;
}

/** A refusal as data. Anything that is not an `Error` is reported as the string it prints as. */
export function encodeFailure(error: unknown): EncodedFailure {
  if (!(error instanceof Error)) {
    return { name: 'Error', message: String(error) };
  }
  const detail: Record<string, unknown> = {};
  let any = false;
  for (const name of Object.keys(error)) {
    // `name`, `message` and `stack` are not own enumerable properties of an `Error` built the
    // ordinary way; a class of the core that assigns `this.name` makes it one, and it is already
    // carried above.
    if (name === 'name' || name === 'message' || name === 'stack') continue;
    put(detail, name, encode((error as unknown as Record<string, unknown>)[name]));
    any = true;
  }
  return {
    name: error.name,
    message: error.message,
    ...(error.stack === undefined ? {} : { stack: error.stack }),
    ...(any ? { detail } : {}),
  };
}

/**
 * A refusal the worker sent, as the caller sees it.
 *
 * It is an `Error` so that it throws and prints like one, and it carries the name the core raised
 * under — `failure.raised` — because that name *is* the distinction the caller reads: a
 * `PyTypeError` out of a derivation is the tools' own exception (feature 1.8c's unguarded public
 * input), a `DerivedSchemaError` is the emitter refusing its own output, a
 * `PrimitiveLibraryError` is the loader's. `instanceof` on the core's classes cannot survive the
 * boundary, so it is not offered.
 */
export class LangFailure extends Error {
  /** The name of the class the worker raised. */
  readonly raised: string;
  /** What that class carried beside its message, decoded. */
  readonly detail: Record<string, unknown>;
  /** The stack as the worker saw it. */
  readonly workerStack: string | undefined;

  constructor(failure: EncodedFailure) {
    super(failure.message);
    this.name = 'LangFailure';
    this.raised = failure.name;
    this.detail =
      failure.detail === undefined ? {} : decodeMembers(failure.detail);
    this.workerStack = failure.stack;
  }
}
