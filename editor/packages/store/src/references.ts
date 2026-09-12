/**
 * The reference index: every place the document writes a name, found by asking the schema which
 * members hold one.
 *
 * A `tensorspine/2.0` document names things in two ways, and both are the schema's to declare:
 *
 * - **as a key**, in a map whose `propertyNames` is a name definition — `quantities`,
 *   `instances`, `compositions`, a composition's `instances` and `indices`,
 *   `interfaces/inputs`;
 * - **as a tagged value**, in a member whose own schema is that same name definition —
 *   `{"quantity": "d"}`, `{"index": "layer"}`, `{"site": "ffn_r"}`, and the two instance
 *   selectors' `composition` and `instance`.
 *
 * Neither list is written here. {@link referenceTags} walks the schemas and finds them: the
 * *name definitions* are whatever a `propertyNames` points at (`identifier` and
 * `qualified_name` in the model schema), the *tags* are every member bound to one, and the
 * *maps* are every member whose names are one. A schema that gained a construct gains its tag
 * here on the next load, and a tag that left stops being looked for — which is the governing
 * rule of plan §1 applied to the one thing the store has to know about the grammar.
 *
 * **What the schema does not say, stated.** Nothing in the four schemas links the tag `quantity`
 * to the map `quantities`: a `$ref` names a *shape*, not a scope, and no keyword says "this
 * member holds the key of that map". So the index answers *occurrences* — each with its tag,
 * its place and the names written beside it — and a command that renames or deletes is handed
 * the {@link ReferenceSelector}s that say which occurrences name the thing it is about. The
 * caller states the pairing; the store never guesses it from a name. Plan §2.2's
 * `presentation.json`, which is keyed by exactly these anchors, is where the editor will write
 * it down once.
 */
import { isJsonArray, isJsonObject, type JsonObject, type JsonValue } from '@tensorspine/lang';

import { child, isUnder, pointerOf, type Path } from './path.js';
import { SchemaShapes, type Shape } from './shape.js';

/**
 * How a name is written: as the key of a map, or as the value of a tagged member.
 *
 * The second is called `tagged` and not `value` because `value` is a word of the language — one
 * of the argument kinds — and plan §1 (b) is a grep: the interface's sources may not write a
 * vocabulary item as a string, whatever it would have meant there.
 */
export type ReferenceKind = 'key' | 'tagged';

/** One place the document writes a name. */
export interface Occurrence {
  /**
   * The member the name is written under: the tag of a tagged value (`quantity`, `index`,
   * `site`, `instance`, `composition`), or the member the map is (`quantities`, `instances`,
   * `indices`) for a key.
   */
  readonly tag: string;
  /** `key` for a map's name, `tagged` for a tagged member's. */
  readonly kind: ReferenceKind;
  /** The name itself. */
  readonly name: string;
  /** Where it is written: the path of the string for a value, of the member for a key. */
  readonly path: Path;
  /** The anchor of the name definition the schema binds it to (`…/model.json#/$defs/identifier`). */
  readonly form: string;
  /**
   * The names written beside it: for a tagged value, the other tagged values of the same object;
   * for a key, those of the object the map belongs to. The generated instance selector —
   * `{"kind": "generated", "composition":
   * "decoder", "instance": "attn_n"}` gives the `instance` occurrence the qualifier
   * `composition: 'decoder'`, which is what tells a site of one composition from a site of
   * another without reading the word "composition" anywhere.
   */
  readonly qualifiers: Readonly<Record<string, string>>;
}

/** What the schemas declare about names: the definitions, the tags and the maps. */
export interface ReferenceTags {
  /** The anchors of the definitions a `propertyNames` points at, in the order they were found. */
  readonly forms: readonly string[];
  /** Every member whose own schema is a name definition, by tag, with the forms it takes. */
  readonly values: ReadonlyMap<string, ReadonlySet<string>>;
  /** Every member that holds a map keyed by a name definition, by tag, with the forms. */
  readonly keys: ReadonlyMap<string, ReadonlySet<string>>;
}

/** Which occurrences a command is about. */
export interface ReferenceSelector {
  /** The tag: the member the name is written under. */
  readonly tag: string;
  /** Only keys, only values, or (absent) both. */
  readonly kind?: ReferenceKind;
  /** Only occurrences at or below this place — how a composition's scope is stated. */
  readonly under?: Path;
  /**
   * Names that must stand beside the occurrence. A value of `null` requires the qualifier to be
   * *absent*, which is how a root instance selector is told from a generated one without naming
   * either.
   */
  readonly qualifiers?: Readonly<Record<string, string | null>>;
}

/**
 * Every name of one document, indexed.
 *
 * Built once per revision by the store; a command reads it to find what it must rewrite, and the
 * canvas will read it to say what a node is referred to by.
 */
export interface ReferenceIndex {
  /** Every occurrence, in document order. */
  readonly all: readonly Occurrence[];
  /** The occurrences of one name under one tag. */
  of(tag: string, name: string): readonly Occurrence[];
  /** The occurrences a selector names, in document order and each at most once. */
  select(selectors: readonly ReferenceSelector[], name: string): readonly Occurrence[];
}

/**
 * What the schemas say about names, read from the registry.
 *
 * `role` is the document kind the tags are wanted for; a name definition is discovered from
 * *every* schema the registry holds, because a map of one schema may be keyed by a definition of
 * another (the unit schema's `$ref`s into the model schema's `identifier` are that case).
 */
export function referenceTags(shapes: SchemaShapes, role: string): ReferenceTags {
  // The answer is a function of the schemas and the role, and the walk is over every shape a
  // document of that role can be in: it is read once per registry and kept, so that opening a
  // second document costs nothing.
  let byRole = TAGS.get(shapes);
  if (byRole === undefined) {
    byRole = new Map<string, ReferenceTags>();
    TAGS.set(shapes, byRole);
  }
  const known = byRole.get(role);
  if (known !== undefined) return known;
  const found = readTags(shapes, role);
  byRole.set(role, found);
  return found;
}

const TAGS = new WeakMap<SchemaShapes, Map<string, ReferenceTags>>();

function readTags(shapes: SchemaShapes, role: string): ReferenceTags {
  const forms = new Set<string>();
  const values = new Map<string, Set<string>>();
  const keys = new Map<string, Set<string>>();
  const root = shapes.root(role);

  // Pass one: every definition a `propertyNames` points at is a *name* definition. Walking the
  // reachable shapes of the role rather than the schema files keeps a definition nothing reaches
  // out of the answer. A node that is nothing but a `$ref` is the way to the definition and not
  // the definition itself, so it is stepped over — otherwise every place that writes
  // `{"$ref": "#/$defs/identifier"}` would be a name form of its own.
  const reachable = reachableShapes(shapes, root);
  for (const { shape } of reachable) {
    for (const place of shapes.keys(shape).all) {
      if (!isWrapper(place.node)) forms.add(place.anchor);
    }
  }

  // Pass two: every member bound to one of those definitions, and every member that holds a map
  // keyed by one.
  for (const { shape, members } of reachable) {
    for (const name of members) {
      const under = shapes.member(shape, name);
      const form = formOf(under, forms);
      if (form !== null) add(values, name, form);
      const keyForm = formOf(shapes.keys(under), forms);
      if (keyForm !== null) add(keys, name, keyForm);
    }
  }
  return { forms: [...forms], values, keys };
}

/** The index of one document. */
export function referenceIndex(tree: JsonObject, shapes: SchemaShapes, role: string): ReferenceIndex {
  const tags = referenceTags(shapes, role);
  return indexWith(tree, shapes, shapes.root(role), tags);
}

/** The index of one document, when the tags have already been read (they are per schema, not per document). */
export function referenceIndexWith(
  tree: JsonObject,
  shapes: SchemaShapes,
  role: string,
  tags: ReferenceTags,
): ReferenceIndex {
  return indexWith(tree, shapes, shapes.root(role), tags);
}

function indexWith(
  tree: JsonObject,
  shapes: SchemaShapes,
  root: Shape,
  tags: ReferenceTags,
): ReferenceIndex {
  const all: Occurrence[] = [];

  /** The tagged values of one object: what an occurrence inside it is qualified by. */
  const taggedValues = (node: JsonObject, shape: Shape): Record<string, string> => {
    const found: Record<string, string> = Object.create(null) as Record<string, string>;
    for (const member of node.members) {
      if (typeof member.value !== 'string') continue;
      const forms = tags.values.get(member.name);
      if (forms === undefined) continue;
      if (formOf(shapes.member(shape, member.name), forms) === null) continue;
      found[member.name] = member.value;
    }
    return found;
  };

  const walk = (value: JsonValue, shape: Shape, path: Path, qualifiers: Record<string, string>): void => {
    if (isJsonArray(value)) {
      for (const [position, item] of value.entries()) {
        walk(item, shapes.item(shape, position), child(path, position), qualifiers);
      }
      return;
    }
    if (!isJsonObject(value)) return;

    // A map whose names the schema declares to be names: every member is a key occurrence, and
    // the object it belongs to is what qualifies them.
    const nameForms = tags.keys.get(lastName(path));
    const keyForm = nameForms === undefined ? null : formOf(shapes.keys(shape), nameForms);
    const own = taggedValues(value, shape);
    for (const member of value.members) {
      const at = child(path, member.name);
      if (keyForm !== null) {
        all.push({
          tag: lastName(path),
          kind: 'key',
          name: member.name,
          path: at,
          form: keyForm,
          qualifiers,
        });
      }
      const under = shapes.member(shape, member.name);
      if (typeof member.value === 'string') {
        const valueForms = tags.values.get(member.name);
        const form = valueForms === undefined ? null : formOf(under, valueForms);
        if (form !== null) {
          all.push({
            tag: member.name,
            kind: 'tagged',
            name: member.value,
            path: at,
            form,
            qualifiers: beside(own, member.name),
          });
        }
      }
      walk(member.value, under, at, own);
    }
  };

  walk(tree, root, [], Object.create(null) as Record<string, string>);

  const byTag = new Map<string, Occurrence[]>();
  // Where each occurrence stands in the document, so that a selection can be put back in document
  // order without walking every occurrence again: feature 2.8 measured `select` at 27.5 ms over
  // deepseek-v4-pro's 1 245 occurrences and 22 quantities, and a table of declarations (§4.16)
  // asks it once per row.
  const position = new Map<Occurrence, number>();
  for (const [index, occurrence] of all.entries()) {
    position.set(occurrence, index);
    const key = `${occurrence.tag}\u0000${occurrence.name}`;
    const found = byTag.get(key);
    if (found === undefined) byTag.set(key, [occurrence]);
    else found.push(occurrence);
  }

  return {
    all,
    of(tag, name) {
      return byTag.get(`${tag}\u0000${name}`) ?? [];
    },
    select(selectors, name) {
      const found = new Map<string, Occurrence>();
      for (const selector of selectors) {
        for (const occurrence of byTag.get(`${selector.tag}\u0000${name}`) ?? []) {
          if (!matches(occurrence, selector)) continue;
          found.set(pointerOf(occurrence.path), occurrence);
        }
      }
      return [...found.values()].sort(
        (left, right) => (position.get(left) ?? 0) - (position.get(right) ?? 0),
      );
    },
  };
}

/** Whether an occurrence is one the selector is about. */
export function matches(occurrence: Occurrence, selector: ReferenceSelector): boolean {
  if (occurrence.tag !== selector.tag) return false;
  if (selector.kind !== undefined && occurrence.kind !== selector.kind) return false;
  if (selector.under !== undefined && !isUnder(occurrence.path, selector.under)) return false;
  for (const [tag, wanted] of Object.entries(selector.qualifiers ?? {})) {
    const written = occurrence.qualifiers[tag];
    if (wanted === null) {
      if (written !== undefined) return false;
    } else if (written !== wanted) {
      return false;
    }
  }
  return true;
}

/**
 * The name definition a place *is*, among the ones the schemas declare.
 *
 * Read from every candidate, because a member is often declared by an alternative — `{"quantity":
 * "d"}` is one alternative of `argument_value`, and the member `quantity` is a name there as
 * plainly as anywhere. What keeps the reading exact is that it asks for a *name definition* and
 * not for a string: `dtype_expression` is a `dtype` or a `quantity_expression`, neither of which
 * is a name, so the member `dtype` is not one either.
 */
function formOf(shape: Shape, forms: ReadonlySet<string>): string | null {
  for (const place of shape.all) {
    if (forms.has(place.anchor)) return place.anchor;
  }
  return null;
}

/** Whether a schema node is nothing but a way to another: `{"$ref": "…"}` and nothing else. */
function isWrapper(node: Readonly<Record<string, unknown>>): boolean {
  return node['$ref'] !== undefined && Object.keys(node).every((keyword) => keyword === '$ref');
}

/** The last member name of a path — the member a map is written under. `''` at the root. */
function lastName(path: Path): string {
  const last = path[path.length - 1];
  return typeof last === 'string' ? last : '';
}

/** A record of tagged values without the one being described. */
function beside(own: Record<string, string>, tag: string): Record<string, string> {
  const found: Record<string, string> = Object.create(null) as Record<string, string>;
  for (const [name, value] of Object.entries(own)) {
    if (name !== tag) found[name] = value;
  }
  return found;
}

function add(map: Map<string, Set<string>>, tag: string, anchor: string): void {
  const found = map.get(tag);
  if (found === undefined) map.set(tag, new Set([anchor]));
  else found.add(anchor);
}

/** One reachable shape of a role, with the member names its candidates declare. */
interface ReachableShape {
  readonly shape: Shape;
  readonly members: readonly string[];
}

/**
 * Every shape a document of that role can be in, with the members each declares.
 *
 * The walk is over the *schema*, not a document, and is bounded by the anchors already seen —
 * `location` names itself and `scalar_expression` names itself, so without that it would not
 * end.
 */
function reachableShapes(shapes: SchemaShapes, root: Shape): readonly ReachableShape[] {
  const found: ReachableShape[] = [];
  const seen = new Set<string>();
  const queue: Shape[] = [root];
  while (queue.length > 0) {
    const shape = queue.shift();
    if (shape === undefined) continue;
    const key = shapes.keyOfShape(shape);
    if (key === '' || seen.has(key)) continue;
    seen.add(key);
    const members = new Set<string>();
    for (const place of shape.all) {
      const properties = place.node['properties'];
      if (properties !== null && typeof properties === 'object' && !Array.isArray(properties)) {
        for (const name of Object.keys(properties)) members.add(name);
      }
    }
    const names = [...members];
    found.push({ shape, members: names });
    for (const name of names) queue.push(shapes.member(shape, name));
    // The value side of a map, and the items of an array: both are places a document reaches.
    queue.push(shapes.values(shape));
    queue.push(shapes.item(shape, 0));
  }
  return found;
}
