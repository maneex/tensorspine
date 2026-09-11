/**
 * `New Model` and `New Template` (§4.3) — a document made from the schema, not from a literal.
 *
 * > **New Model** creates `{schema, model, primitive_libraries: [<the workspace's default base>],
 * > quantities: {}, constants: {}, instances: {}, compositions: {}, bindings: {values: {},
 * > parameters: {}, constants: {}, states: {}}, interfaces: {inputs: {}, outputs: {}}}` — the
 * > required members of the top-level schema, empty. **New Template** adds `version: "1.0.0"`.
 *
 * That sentence is a *reading of the schema*, and this builds it by reading the schema: the
 * members are `required`, in the order the schema writes them; a member that fixes one value —
 * the `schema` tag — takes that value; a member that holds a map takes an empty map; one that
 * holds a list takes an empty list; one that requires members of its own is built the same way,
 * one level down, which is where `bindings` and `interfaces` come from.
 *
 * **This is where feature 2.1's open point lands.** Its ledger row says: "the store writes no
 * `const` of the language — `tensorspine/2.0` appears nowhere, because no command here makes a
 * document from nothing; 2.6 is where that decision lands." It lands here, and the decision is
 * that it stays true: the tag is `mergeFacts(shape.direct).constant`, read from the very schema
 * the registry loaded, so a workspace carrying a schema of its own makes a document on *that*
 * one. Nothing below names a member of the grammar.
 *
 * **The document it makes is off the grammar, and that is the plan's own skeleton.** The root
 * requires a non-empty `instances` or `compositions` (its `anyOf`), `primitive_libraries` has
 * `minItems: 1`, and `interfaces.inputs` and `.outputs` have `minProperties: 1`. So a new
 * document is refused by Ajv until it has a base, a site and an interface — which is what the
 * Problems panel is for and what D5's "incomplete meaning is the semantic stage's report" says
 * one stage earlier. Recorded as a finding rather than papered over: the alternative would be to
 * invent a site and an interface the author did not ask for.
 */
import { jsonObject, mergeFacts, type JsonMember, type JsonObject, type JsonValue } from '@tensorspine/lang';

import { MODEL_ROLE } from '../document.js';
import type { Shape, SchemaShapes } from '../shape.js';

/** What a new document is given that the schema cannot supply. */
export interface NewDocument {
  /** The `model` identifier — the tab's name, and what the file is called after. */
  readonly name: string;
  /** The base the document resolves from, relative to where the document will be saved. */
  readonly base?: string;
  /**
   * The version a template carries.
   *
   * §4.3: "**New Template** adds `version: "1.0.0"`". A template is a document with external
   * quantities carrying a `version`, and the version is what the primitive that pins it names
   * (§4.6) — so it is the caller's, not the schema's.
   */
  readonly version?: string;
}

/** The version a new template starts at, as §4.3 writes it. */
export const FIRST_VERSION = '1.0.0';

/**
 * A document with the required members of its schema and nothing else.
 *
 * The role is the registry's (`model`), so the same walk makes a unit's skeleton when feature 3.2
 * wants one — it is the schema that differs, not the reading.
 */
export function newDocument(
  shapes: SchemaShapes,
  document: NewDocument,
  role: string = MODEL_ROLE,
): JsonObject {
  const root = shapes.root(role);
  const identifier = nameMember(shapes, role);
  const members: JsonMember[] = [];
  for (const name of requiredOf(shapes, root)) {
    const shape = shapes.member(root, name);
    const facts = mergeFacts(shape.direct.map((place) => place.node));
    // The one member whose value is neither empty nor the caller's: the schema fixes it.
    if (facts.fixed && typeof facts.constant === 'string') {
      members.push({ name, value: facts.constant });
      continue;
    }
    // The identifier the caller supplied goes where the schema asks for a text and nothing else.
    members.push({ name, value: name === identifier ? document.name : emptyOf(shapes, shape) });
  }
  const made = jsonObject(members);
  if (document.base !== undefined) withBase(shapes, made, document.base);
  return finish(shapes, made, document, role);
}

/** The required members of a place, in the order the schema writes them. */
function requiredOf(shapes: SchemaShapes, shape: Shape): readonly string[] {
  const required = shapes.constraintsOf(shape).required;
  return shapes.propertyOrder(shape).filter((name) => required.has(name));
}

/**
 * The members a place declares, in the order the *schema file* writes them.
 *
 * Not `propertyOrder`, which puts the required ones first — the reading a form wants, and the
 * wrong one for placing an optional member among them: the model schema declares `version`
 * between `model` and `primitive_libraries`, and that is where the corpus's own template writes
 * it.
 */
function declaredOrder(shape: Shape): readonly string[] {
  return mergeFacts(shape.all.map((place) => place.node)).members;
}

/**
 * The empty value of a place: a list, a map, or an object with its own required members.
 *
 * "The required members of the top-level schema, empty" is one rule applied twice — `bindings`
 * and `interfaces` are objects that require members, and their members are maps.
 */
function emptyOf(shapes: SchemaShapes, shape: Shape): JsonValue {
  const facts = mergeFacts(shape.direct.map((place) => place.node));
  if (facts.holdsArray) return [];
  const required = requiredOf(shapes, shape);
  if (required.length === 0) return jsonObject();
  return jsonObject(
    required.map((name) => ({ name, value: emptyOf(shapes, shapes.member(shape, name)) })),
  );
}

/** The base the workspace offers, written as the one entry the list requires. */
function withBase(shapes: SchemaShapes, made: JsonObject, base: string): void {
  const root = shapes.root(MODEL_ROLE);
  for (const [index, member] of made.members.entries()) {
    const shape = shapes.member(root, member.name);
    const facts = mergeFacts(shape.direct.map((place) => place.node));
    if (!facts.holdsArray) continue;
    const item = shapes.item(shape, 0);
    const required = requiredOf(shapes, item);
    if (required.length !== 1) continue;
    // The one list the root requires, whose item requires one member: the base's own path.
    (made.members as JsonMember[])[index] = {
      name: member.name,
      value: [jsonObject([{ name: required[0] as string, value: base }])],
    };
    return;
  }
}

/**
 * The member a document is named by — §4.3's "a tab named after `model`".
 *
 * The one *required* member of the root the schema describes as free text and does not fix: the
 * model schema requires `schema` (fixed) and `model` (a `model_id`), so the walk finds `model`
 * and the interface never spells it. The unit schema's own answer is `name`, which is what makes
 * the same reading serve the primitive editor (3.2) unchanged.
 *
 * `null` where the schema has none, or has more than one — which the suites require it not to.
 */
export function nameMember(shapes: SchemaShapes, role: string = MODEL_ROLE): string | null {
  const root = shapes.root(role);
  const required = shapes.constraintsOf(root).required;
  const found = shapes.propertyOrder(root).filter((name) => {
    if (!required.has(name)) return false;
    const facts = mergeFacts(shapes.member(root, name).direct.map((place) => place.node));
    return facts.holdsText && !facts.fixed;
  });
  return found.length === 1 ? (found[0] as string) : null;
}

/**
 * The tag a document of this role carries because its schema fixes it — `tensorspine/2.0`.
 *
 * §4.2 puts it in the status bar, second field, and feature 2.1 refused to write it: "the store
 * writes no `const` of the language". It does not have to. The schema fixes exactly one member of
 * the root, `newDocument` writes whatever that member's `const` is, and this is that value —
 * which is what a caller compares a file's own tag against to ask "is this a model document at
 * all?" without naming the revision anywhere.
 *
 * `null` where the schema fixes no member of its root.
 */
export function fixedTag(shapes: SchemaShapes, role: string = MODEL_ROLE): string | null {
  const root = shapes.root(role);
  for (const name of shapes.propertyOrder(root)) {
    const facts = mergeFacts(shapes.member(root, name).direct.map((place) => place.node));
    if (facts.fixed && typeof facts.constant === 'string') return facts.constant;
  }
  return null;
}

/**
 * The tag a document carries because its schema fixes it, read off the tree.
 *
 * See {@link fixedTag} for the value the schema fixes; this is what the *document* wrote there.
 */
export function tagOf(shapes: SchemaShapes, tree: JsonObject, role: string = MODEL_ROLE): string | null {
  const root = shapes.root(role);
  for (const name of shapes.propertyOrder(root)) {
    const facts = mergeFacts(shapes.member(root, name).direct.map((place) => place.node));
    if (!facts.fixed) continue;
    const written = tree.members.find((member) => member.name === name)?.value;
    if (typeof written === 'string') return written;
  }
  return null;
}

/**
 * The one member of a document's root the schema leaves optional and describes as free text.
 *
 * §4.3 gives New Template a `version` and the model schema declares exactly one such member, so
 * the walk finds it instead of the interface spelling it. It is a *discovery* and not a guess
 * only for as long as there is one of them: `test/documents/skeleton.test.ts` requires that, so a
 * schema that grew a second optional text member fails until somebody says which is meant.
 *
 * `null` where the schema has none, or has more than one.
 */
export function versionMember(shapes: SchemaShapes, role: string = MODEL_ROLE): string | null {
  const root = shapes.root(role);
  const required = shapes.constraintsOf(root).required;
  const found = shapes.propertyOrder(root).filter((name) => {
    if (required.has(name)) return false;
    const facts = mergeFacts(shapes.member(root, name).direct.map((place) => place.node));
    return facts.holdsText && !facts.fixed;
  });
  return found.length === 1 ? (found[0] as string) : null;
}

/** A template's own member: the version §4.3 gives it, written where the schema declares it. */
function finish(
  shapes: SchemaShapes,
  made: JsonObject,
  document: NewDocument,
  role: string,
): JsonObject {
  if (document.version === undefined) return made;
  const name = versionMember(shapes, role);
  if (name === null) return made;
  // It goes where the schema writes it among the others, which for the model schema is after
  // `model` and before `primitive_libraries`.
  const order = declaredOrder(shapes.root(role));
  const at = order.indexOf(name);
  const before = made.members.filter((member) => order.indexOf(member.name) < at);
  const after = made.members.filter((member) => order.indexOf(member.name) >= at);
  return jsonObject([...before, { name, value: document.version }, ...after]);
}
