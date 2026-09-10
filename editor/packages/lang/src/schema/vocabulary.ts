/**
 * The vocabulary the schemas carry: every enumeration and every tagged union, by pointer.
 *
 * The plan's governing rule (§1) is that an item of information a schema can state is never
 * hard-coded in the interface: "dtypes, quantity kinds, argument kinds, axis natures and spaces,
 * operator names, comparison operators, condition forms, interface kinds, location forms,
 * evolution rules, access geometries, sharing granularities, communications, transform relations,
 * sensitivities, statuses, cost `per` values: all read from the schemas at startup. A select's
 * options are the schema's `enum`; a tagged union's tags are the `required` keys of its `oneOf`
 * members."
 *
 * This module is where that reading is done, once, for everything that needs it: the generated
 * forms, the presentation audit — which lists every enum value and every `oneOf` member no
 * binding names — and the no-hard-coding test, which needs the literals to look for.
 *
 * Two things are worth stating about the tags. The **discriminating** keys of an alternative are
 * those no sibling alternative requires; when the alternatives are told apart by one key each —
 * `tensor`, `stack`, `concat`, `slice` — that is the chooser's label. When they are not, the
 * vocabulary says so rather than inventing one: `binary_operation_expression` and
 * `nary_operation_expression` both require `op` and `args` and are separated only by which
 * operator names their `op` admits, so both carry the same tags and the union is not
 * discriminated. A `oneOf` whose alternatives have no required key at all — the string form of a
 * `physical_name` item — carries none, and renders generically (§1).
 *
 * A pointer is the anchor `presentation.json` is keyed by: the schema's `$id`, `#`, and the JSON
 * pointer of the node, as `…/model.json#/$defs/instance_definition`.
 */
import type { LoadedSchema } from './registry.js';
import { isSchemaObject, type SchemaNode, type SchemaObject } from './types.js';

/** A scalar a schema writes as an `enum` member or a `const`. */
export type VocabularyValue = string | number | boolean | null;

/** One enumeration of one schema, where it is written. */
export interface VocabularyEnum {
  /** `<$id>#<place>` — the anchor a presentation binding names. */
  readonly pointer: string;
  /** The `$id` of the schema that writes it. */
  readonly schema: string;
  /** The JSON pointer inside that schema: `#/$defs/dtype`. */
  readonly place: string;
  /** The admissible values, in the order the schema writes them. */
  readonly values: readonly VocabularyValue[];
  /** The schema's own `title`, when it carries one. */
  readonly title: string | null;
  /** The schema's own `description`, which is where the interface's help text comes from. */
  readonly description: string | null;
}

/** One alternative of a tagged union. */
export interface VocabularyAlternative {
  /** The JSON pointer of the alternative: `#/$defs/location/oneOf/1`. */
  readonly place: string;
  /** The `$ref` the alternative is, when it is one and nothing else. */
  readonly ref: string | null;
  /** `<$id>#<place>` of what that `$ref` names, followed through a chain of bare refs. */
  readonly target: string | null;
  /** Every key the alternative requires, in the order it writes them. */
  readonly required: readonly string[];
  /** The required keys no sibling alternative requires. */
  readonly discriminating: readonly string[];
  /** The alternative's own `const`, when it is a constant rather than a shape. */
  readonly constant: VocabularyValue | undefined;
  /** What the chooser labels the alternative by: the const, the discriminating keys, or none. */
  readonly tags: readonly string[];
  /** The declared `type`, which is the only label a shape without a required key has. */
  readonly type: string | readonly string[] | null;
  readonly title: string | null;
  readonly description: string | null;
}

/** One `oneOf`: a tagged union, where it is written. */
export interface VocabularyUnion {
  readonly pointer: string;
  readonly schema: string;
  readonly place: string;
  readonly alternatives: readonly VocabularyAlternative[];
  /** Whether every alternative carries a tag no other one carries. */
  readonly discriminated: boolean;
  readonly title: string | null;
  readonly description: string | null;
}

/** Every enumeration and every tagged union of the loaded schemas. */
export interface Vocabulary {
  readonly enums: readonly VocabularyEnum[];
  readonly unions: readonly VocabularyUnion[];
  /** The enumeration at that pointer, `<$id>#<place>` or `#<place>` of the model schema. */
  enumAt(pointer: string): VocabularyEnum | undefined;
  /** The union at that pointer. */
  unionAt(pointer: string): VocabularyUnion | undefined;
  /** Every value of every enumeration, once each: what the no-hard-coding audit looks for. */
  values(): readonly VocabularyValue[];
}

/** The subschema-valued keywords: one schema, a map of schemas, a list of schemas. */
const SINGLE = [
  'additionalProperties',
  'contains',
  'else',
  'if',
  'items',
  'not',
  'propertyNames',
  'then',
  'unevaluatedItems',
  'unevaluatedProperties',
] as const;
const MAPS = ['$defs', 'definitions', 'dependentSchemas', 'patternProperties', 'properties'] as const;
const LISTS = ['allOf', 'anyOf', 'oneOf', 'prefixItems'] as const;

/** A JSON pointer segment, escaped as RFC 6901 escapes it. */
function segment(name: string): string {
  return name.replace(/~/g, '~0').replace(/\//g, '~1');
}

/** The string of a `title` or `description` keyword, or `null`. */
function text(node: SchemaObject, keyword: string): string | null {
  const value = node[keyword];
  return typeof value === 'string' ? value : null;
}

/** Whether a value is one an `enum` or a `const` may hold. */
function isValue(value: unknown): value is VocabularyValue {
  return (
    value === null ||
    typeof value === 'string' ||
    typeof value === 'number' ||
    typeof value === 'boolean'
  );
}

/** The `required` keys of a node, as strings. */
function requiredOf(node: SchemaNode): string[] {
  if (!isSchemaObject(node)) return [];
  const declared = node['required'];
  if (!Array.isArray(declared)) return [];
  return declared.filter((name): name is string => typeof name === 'string');
}

/** Follows a chain of bare `$ref`s to the node that actually says something. */
function follow(
  node: SchemaNode,
  schemaId: string,
  byId: ReadonlyMap<string, LoadedSchema>,
): { node: SchemaNode; pointer: string } | null {
  let current = node;
  let base = schemaId;
  let pointer: string | null = null;
  for (let depth = 0; depth < 8; depth += 1) {
    if (!isSchemaObject(current)) return pointer === null ? null : { node: current, pointer };
    const ref = current['$ref'];
    const bare = typeof ref === 'string' && Object.keys(current).every((key) => key === '$ref');
    if (!bare) return pointer === null ? null : { node: current, pointer };
    const hash = ref.indexOf('#');
    const uri = hash < 0 ? ref : ref.slice(0, hash);
    const inside = hash < 0 ? '#' : ref.slice(hash);
    const target = byId.get(uri === '' ? base : uri);
    if (target === undefined) return null;
    const found = at(target.document, inside);
    if (found === null) return null;
    base = target.id;
    pointer = `${target.id}${inside}`;
    current = found;
  }
  return pointer === null ? null : { node: current, pointer };
}

/** The node a `#/…` pointer names inside a schema document, or `null`. */
function at(document: SchemaObject, pointer: string): SchemaNode | null {
  if (pointer === '#' || pointer === '') return document;
  let node: unknown = document;
  for (const step of pointer.slice(2).split('/')) {
    const name = step.replace(/~1/g, '/').replace(/~0/g, '~');
    if (node === null || typeof node !== 'object') return null;
    node = Array.isArray(node) ? node[Number(name)] : (node as Record<string, unknown>)[name];
    if (node === undefined) return null;
  }
  return node as SchemaNode;
}

/** The alternatives of one `oneOf`, with the tags that tell them apart. */
function alternativesOf(
  branches: readonly SchemaNode[],
  schemaId: string,
  place: string,
  byId: ReadonlyMap<string, LoadedSchema>,
): VocabularyAlternative[] {
  const resolved = branches.map((branch) => {
    const followed = follow(branch, schemaId, byId);
    const node = followed?.node ?? branch;
    const ref =
      isSchemaObject(branch) && typeof branch['$ref'] === 'string' ? branch['$ref'] : null;
    return { branch, node, ref, target: followed?.pointer ?? null };
  });
  const requiredBy = resolved.map((one) => requiredOf(one.node));

  return resolved.map((one, index) => {
    const required = requiredBy[index] as string[];
    const others = new Set(requiredBy.filter((_, other) => other !== index).flat());
    const discriminating = required.filter((name) => !others.has(name));
    const node = one.node;
    const constant = isSchemaObject(node) && isValue(node['const']) ? node['const'] : undefined;
    const declaredType = isSchemaObject(node) ? node['type'] : undefined;
    const type =
      typeof declaredType === 'string'
        ? declaredType
        : Array.isArray(declaredType)
          ? declaredType.filter((one_): one_ is string => typeof one_ === 'string')
          : null;
    const tags =
      constant !== undefined
        ? [String(constant)]
        : discriminating.length > 0
          ? discriminating
          : required;
    return {
      place: `${place}/${String(index)}`,
      ref: one.ref,
      target: one.target,
      required,
      discriminating,
      constant,
      tags,
      type,
      title: isSchemaObject(node) ? text(node, 'title') : null,
      description: isSchemaObject(node) ? text(node, 'description') : null,
    };
  });
}

/** Every enumeration and every tagged union of the schemas, indexed by pointer. */
export function vocabularyOf(schemas: readonly LoadedSchema[]): Vocabulary {
  const byId = new Map(schemas.map((schema) => [schema.id, schema]));
  const enums: VocabularyEnum[] = [];
  const unions: VocabularyUnion[] = [];

  const visit = (node: SchemaNode, schema: LoadedSchema, place: string): void => {
    if (!isSchemaObject(node)) return;
    const values = node['enum'];
    if (Array.isArray(values)) {
      enums.push({
        pointer: `${schema.id}${place}`,
        schema: schema.id,
        place,
        values: values.filter(isValue),
        title: text(node, 'title'),
        description: text(node, 'description'),
      });
    }
    const branches = node['oneOf'];
    if (Array.isArray(branches)) {
      const alternatives = alternativesOf(
        branches as readonly SchemaNode[],
        schema.id,
        `${place}/oneOf`,
        byId,
      );
      const seen = new Map<string, number>();
      for (const alternative of alternatives) {
        for (const tag of alternative.tags) seen.set(tag, (seen.get(tag) ?? 0) + 1);
      }
      unions.push({
        pointer: `${schema.id}${place}`,
        schema: schema.id,
        place,
        alternatives,
        discriminated: alternatives.every(
          (alternative) =>
            alternative.tags.length > 0 &&
            alternative.tags.every((tag) => (seen.get(tag) ?? 0) === 1),
        ),
        title: text(node, 'title'),
        description: text(node, 'description'),
      });
    }
    for (const keyword of SINGLE) {
      const child = node[keyword];
      if (child !== undefined) visit(child as SchemaNode, schema, `${place}/${keyword}`);
    }
    for (const keyword of MAPS) {
      const children = node[keyword];
      if (children === null || typeof children !== 'object' || Array.isArray(children)) continue;
      for (const [name, child] of Object.entries(children)) {
        visit(child as SchemaNode, schema, `${place}/${keyword}/${segment(name)}`);
      }
    }
    for (const keyword of LISTS) {
      const children = node[keyword];
      if (!Array.isArray(children)) continue;
      children.forEach((child, index) => {
        visit(child as SchemaNode, schema, `${place}/${keyword}/${String(index)}`);
      });
    }
  };

  for (const schema of schemas) visit(schema.document, schema, '#');

  const enumsByPointer = new Map(enums.map((one) => [one.pointer, one]));
  const unionsByPointer = new Map(unions.map((one) => [one.pointer, one]));

  return {
    enums,
    unions,
    enumAt: (pointer) => enumsByPointer.get(pointer),
    unionAt: (pointer) => unionsByPointer.get(pointer),
    values() {
      const seen: VocabularyValue[] = [];
      const known = new Set<string>();
      for (const one of enums) {
        for (const value of one.values) {
          const key = `${typeof value}:${String(value)}`;
          if (known.has(key)) continue;
          known.add(key);
          seen.push(value);
        }
      }
      return seen;
    },
  };
}
