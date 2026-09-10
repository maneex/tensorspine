/**
 * The ordered JSON tree the editor holds a document in.
 *
 * The plan's D1 — the document is the model — puts the `tensorspine/2.0` JSON itself in the
 * store, "as a tree, with member order and number lexemes", and D12 makes the core's serializer
 * the writer of record: an unedited document must save back byte for byte as the corpus writes
 * it. Two things the language relies on are lost by `JSON.parse`, so the tree keeps them:
 *
 * - **Member order.** A plain JavaScript object reorders integer-like keys (`{"10": …, "2": …}`
 *   enumerates as `2`, `10`), and its order is otherwise insertion order only by convention. An
 *   object is therefore an ordered list of members here, and an insert says where it goes.
 * - **The integer/float distinction.** V3 reads a number's spelling: `32.0` is not a cardinality.
 *   Every number carries whether it is real, and the text it was written with, so that `1e-05`,
 *   `1.0` and `0.25` come back as they were authored (D12, plan §7 F7).
 *
 * The shapes are plain objects and arrays — no class, no symbol — so a tree survives the
 * structured clone that carries it to and from the Web Worker the core runs in (plan §5.3), and
 * so Immer can draft it for the store's command log (D13).
 */

/**
 * A JSON number, with what the language reads in it.
 *
 * `value` is the double the text denotes; `real` is V3's lexical reading of the number token —
 * a fraction or an exponent makes a real, a bare whole number a cardinality — and is what the
 * serializer writes from when it has to format the number afresh. `lexeme` is the source text of
 * a parsed number, kept so that an untouched number is re-emitted verbatim; a number the editor
 * makes has none, and its caller supplies `real` (D12: "the float-ness taken from the declared
 * type: a `real` argument is a float, a `cardinality` an integer").
 */
export interface JsonNumber {
  readonly kind: 'number';
  readonly value: number;
  readonly real: boolean;
  readonly lexeme?: string;
}

/** One member of an object: its name and its value, in the position the document writes it. */
export interface JsonMember {
  readonly name: string;
  readonly value: JsonValue;
}

/** A JSON object: its members in order, duplicate names already refused by the parser (V12). */
export interface JsonObject {
  readonly kind: 'object';
  readonly members: readonly JsonMember[];
}

/** A JSON array. Order is the array's own; nothing else is kept. */
export type JsonArray = readonly JsonValue[];

/**
 * A value of the tree. Strings, booleans and null are the JavaScript ones — their JSON form is
 * exact, so there is nothing to keep — while numbers and objects carry what a plain parse loses.
 */
export type JsonValue = string | boolean | null | JsonNumber | JsonObject | JsonArray;

/** Whether a value is an array node. */
export function isJsonArray(value: JsonValue): value is JsonArray {
  return Array.isArray(value);
}

/** Whether a value is an object node. */
export function isJsonObject(value: JsonValue): value is JsonObject {
  return typeof value === 'object' && value !== null && !isJsonArray(value) && value.kind === 'object';
}

/** Whether a value is a number node. */
export function isJsonNumber(value: JsonValue): value is JsonNumber {
  return typeof value === 'object' && value !== null && !isJsonArray(value) && value.kind === 'number';
}

/**
 * A number the editor writes, with its float-ness supplied by the caller (D12).
 *
 * The value is kept as it is given; nothing here decides whether `16` is a cardinality or a real,
 * because JavaScript cannot tell and guessing is what loses `16.0`.
 */
export function jsonNumber(value: number, real: boolean): JsonNumber {
  return { kind: 'number', value, real };
}

/** A whole number: written bare, as a cardinality is (`16`). */
export function jsonInteger(value: number): JsonNumber {
  return jsonNumber(value, false);
}

/** A real: written with a fraction or an exponent, as V3 requires of one (`16.0`, `1e-05`). */
export function jsonReal(value: number): JsonNumber {
  return jsonNumber(value, true);
}

/** An object node from its members, in the order given. */
export function jsonObject(members: readonly JsonMember[] = []): JsonObject {
  return { kind: 'object', members };
}

/** The names of an object's members, in order. */
export function memberNames(node: JsonObject): string[] {
  return node.members.map((member) => member.name);
}

/** The position of a member, or `-1` when the object has no such member. */
export function indexOfMember(node: JsonObject, name: string): number {
  return node.members.findIndex((member) => member.name === name);
}

/** Whether the object has a member of that name. */
export function hasMember(node: JsonObject, name: string): boolean {
  return indexOfMember(node, name) >= 0;
}

/** The value of a member, or `undefined` when the object has no such member. */
export function getMember(node: JsonObject, name: string): JsonValue | undefined {
  return node.members.find((member) => member.name === name)?.value;
}

/**
 * The object with `name` set to `value`.
 *
 * An existing member keeps its position — an edit never moves a member, which is what makes an
 * edited-and-reverted document identical to the one that was read. A new member is appended,
 * unless `at` says which position it takes (§5.5: "new members appended where the gesture put
 * them"). The object is not changed; a new one is returned.
 */
export function withMember(node: JsonObject, name: string, value: JsonValue, at?: number): JsonObject {
  const found = indexOfMember(node, name);
  if (found >= 0) {
    const members = [...node.members];
    members[found] = { name, value };
    return { kind: 'object', members };
  }
  const position = at === undefined ? node.members.length : Math.max(0, Math.min(at, node.members.length));
  const members = [...node.members.slice(0, position), { name, value }, ...node.members.slice(position)];
  return { kind: 'object', members };
}

/** The object without the member of that name; unchanged when it has none. */
export function withoutMember(node: JsonObject, name: string): JsonObject {
  return { kind: 'object', members: node.members.filter((member) => member.name !== name) };
}

/**
 * The tree as plain JavaScript data: objects become plain objects in member order, numbers their
 * values.
 *
 * This is the form Ajv validates and the algorithms of the core read (plan §5.3). It is lossy in
 * exactly the two ways the tree exists to avoid — an integer-like member name would enumerate out
 * of order, and `1.0` becomes `1` — so it is never a form anything is written back from.
 */
export function toPlain(value: JsonValue): unknown {
  if (isJsonNumber(value)) return value.value;
  if (isJsonArray(value)) return value.map(toPlain);
  if (isJsonObject(value)) {
    const plain: Record<string, unknown> = {};
    for (const member of value.members) plain[member.name] = toPlain(member.value);
    return plain;
  }
  return value;
}
