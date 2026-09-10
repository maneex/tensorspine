/**
 * The walk: `jsonschema`'s `iter_errors` and `descend`, over the same schema files Ajv compiles.
 *
 * A refusal has to be *explained*, not only pronounced. Ajv pronounces it — `conforms` in
 * `registry.ts` is one call on the whole schema, and that is the verdict (D4) — but Ajv's errors
 * are a flat list with the branches of every `oneOf` in it, while the tools print the error tree
 * `jsonschema` builds and read it in two different ways (`errors.ts`). So the walk runs only when
 * Ajv has already refused, and its job is to say *why*, in the tools' words and at the tools'
 * places.
 *
 * The division of labour is the same at every node:
 *
 * - the **assertions** — `type`, `enum`, `required`, `pattern`, `minItems`, … — are decided by
 *   Ajv, keyword by keyword, in `assertions.ts`;
 * - the **applicators** — `$ref`, `properties`, `items`, `allOf`, `oneOf`, `if` — are here, and
 *   they only compose: which child is judged by which subschema, and how a branch's errors hang
 *   under the branch point.
 *
 * Nothing is passed over in silence. A keyword of the 2020-12 vocabulary that the walk does not
 * implement raises rather than being skipped, so that a schema that starts using
 * `unevaluatedProperties` fails a test instead of quietly validating less than it says; the
 * keywords that are annotations — `title`, `description`, `$defs`, `default` — are skipped, which
 * is exactly what `iter_errors` does with a keyword its validator map does not hold.
 */
import { isJsonArray, isJsonObject, memberNames } from '../json/tree.js';
import {
  additionalPropertiesErrors,
  assertionErrors,
  itemsFalseErrors,
  ASSERTION_KEYWORDS,
  type AssertionEngine,
} from './assertions.js';
import { schemaError, type SchemaError } from './errors.js';
import { pythonRepr } from './repr.js';
import {
  at,
  isSchemaObject,
  literalInstance,
  type Instance,
  type PathSegment,
  type SchemaNode,
  type SchemaObject,
} from './types.js';

/** A schema node with the identity of the document it was found in, so its `$ref`s resolve. */
export interface Resolved {
  readonly node: SchemaNode;
  readonly base: string;
}

/** What the walk needs from outside: how to follow a `$ref`, and who decides an assertion. */
export interface WalkEnv {
  resolve(ref: string, base: string): Resolved;
  readonly assertions: AssertionEngine;
}

/**
 * The keywords `jsonschema`'s 2020-12 validator map holds. A keyword outside this set is an
 * annotation and is skipped; one inside it that the walk does not implement is refused.
 */
export const KNOWN_KEYWORDS: readonly string[] = [
  '$dynamicRef',
  '$ref',
  'additionalProperties',
  'allOf',
  'anyOf',
  'const',
  'contains',
  'dependentRequired',
  'dependentSchemas',
  'enum',
  'exclusiveMaximum',
  'exclusiveMinimum',
  'format',
  'if',
  'items',
  'maxItems',
  'maxLength',
  'maxProperties',
  'maximum',
  'minItems',
  'minLength',
  'minProperties',
  'minimum',
  'multipleOf',
  'not',
  'oneOf',
  'pattern',
  'patternProperties',
  'prefixItems',
  'properties',
  'propertyNames',
  'required',
  'type',
  'unevaluatedItems',
  'unevaluatedProperties',
  'uniqueItems',
];

/** The applicators the walk composes; the rest of {@link KNOWN_KEYWORDS} are the assertions. */
const APPLICATOR_KEYWORDS: readonly string[] = [
  '$ref',
  'additionalProperties',
  'allOf',
  'anyOf',
  'dependentSchemas',
  'if',
  'items',
  'not',
  'oneOf',
  'patternProperties',
  'prefixItems',
  'properties',
  'propertyNames',
];

const KNOWN = new Set(KNOWN_KEYWORDS);
const APPLICATORS = new Set(APPLICATOR_KEYWORDS);
const ASSERTIONS = new Set(ASSERTION_KEYWORDS);

/**
 * The errors already attributed to a keyword, an instance and a schema.
 *
 * `ValidationError._set` fills in only what the raising function left unset, and the one error
 * raised with everything already given is the `false` schema's. The set stands for that
 * distinction without putting a bookkeeping field on the public error record.
 */
const attributed = new WeakSet<SchemaError>();

/** `_set` plus the schema-path bookkeeping `iter_errors` does around a keyword's errors. */
function attribute(error: SchemaError, keyword: string, value: unknown, instance: Instance, node: SchemaNode): void {
  if (!attributed.has(error)) {
    error.keyword = keyword;
    error.keywordValue = value;
    error.instance = instance.tree;
    error.schema = node;
    attributed.add(error);
  }
  // `if` states its verdict through `then` and `else`, and `$ref` through what it points at:
  // neither adds a step of its own to the place in the schema.
  if (keyword !== 'if' && keyword !== '$ref') error.schemaPath.unshift(keyword);
}

/** The `false` schema's own refusal, raised with every field given, as `jsonschema` raises it. */
function falseSchemaError(instance: Instance): SchemaError {
  const error = schemaError(`False schema does not allow ${pythonRepr(instance.tree)}`, {
    keyword: null,
    instance: instance.tree,
    schema: false,
  });
  attributed.add(error);
  return error;
}

/** Whether an object instance has a member of that name. */
function hasName(instance: Instance, name: string): boolean {
  return isJsonObject(instance.tree) && instance.tree.members.some((each) => each.name === name);
}

/** The member of an object instance, both readings in step; asked only for one it has. */
function member(instance: Instance, name: string): Instance {
  const found = isJsonObject(instance.tree)
    ? instance.tree.members.find((each) => each.name === name)?.value
    : undefined;
  return at(instance, name, found ?? null);
}

/** The element of an array instance, both readings in step; asked only for one it has. */
function element(instance: Instance, index: number): Instance {
  const found = isJsonArray(instance.tree) ? instance.tree[index] : undefined;
  return at(instance, index, found ?? null);
}

/**
 * `find_additional_properties`: the members of an object no `properties` entry and no
 * `patternProperties` regex covers, in the order the object writes them.
 */
function additionalNames(node: SchemaObject, instance: Instance): string[] {
  if (!isJsonObject(instance.tree)) return [];
  const declared = node['properties'];
  const known = new Set(
    declared !== null && typeof declared === 'object' ? Object.keys(declared) : [],
  );
  const patterns = node['patternProperties'];
  const regexes =
    patterns !== null && typeof patterns === 'object'
      ? Object.keys(patterns).map((pattern) => new RegExp(pattern, 'u'))
      : [];
  return memberNames(instance.tree).filter(
    (name) => !known.has(name) && !regexes.some((regex) => regex.test(name)),
  );
}

/** `iter_errors`: every error one schema node raises about one instance, in the schema's order. */
export function iterErrors(instance: Instance, schema: SchemaNode, base: string, env: WalkEnv): SchemaError[] {
  if (schema === true) return [];
  if (schema === false) return [falseSchemaError(instance)];
  const errors: SchemaError[] = [];
  for (const keyword of Object.keys(schema)) {
    if (!KNOWN.has(keyword)) continue;
    const raised = APPLICATORS.has(keyword)
      ? applicator(keyword, instance, schema, base, env)
      : ASSERTIONS.has(keyword)
        ? assertion(keyword, instance, schema, env)
        : unsupported(keyword);
    for (const error of raised) {
      attribute(error, keyword, schema[keyword], instance, schema);
      errors.push(error);
    }
  }
  return errors;
}

/**
 * `descend`: `iter_errors` on a subschema, with the step that led to it prepended to the places.
 *
 * The two boolean schemas are answered before the step is prepended, which is what `descend`
 * itself does — it returns from its `False` branch above the loop that prepends. A `false`
 * subschema's refusal therefore names the *parent's* place, not the member's: `properties: {a:
 * false}` on `{"a": 1}` reads `<root>: False schema does not allow 1`. The behaviour is
 * `jsonschema`'s and is reproduced rather than corrected, because the wording and the place are
 * the parity contract (D2).
 */
function descend(
  instance: Instance,
  schema: SchemaNode,
  base: string,
  env: WalkEnv,
  path?: PathSegment,
  schemaPath?: PathSegment,
): SchemaError[] {
  if (schema === true) return [];
  if (schema === false) return [falseSchemaError(instance)];
  const errors = iterErrors(instance, schema, base, env);
  for (const error of errors) {
    if (path !== undefined) error.path.unshift(path);
    if (schemaPath !== undefined) error.schemaPath.unshift(schemaPath);
  }
  return errors;
}

/** `evolve(schema=…).is_valid(instance)`: the walk's own verdict on a subschema. */
function isValid(instance: Instance, schema: SchemaNode, base: string, env: WalkEnv): boolean {
  return iterErrors(instance, schema, base, env).length === 0;
}

/** A keyword of the vocabulary the walk does not compose: refused, never skipped. */
function unsupported(keyword: string): never {
  throw new Error(
    `schema: the keyword '${keyword}' is part of 2020-12 but the walk does not implement it`,
  );
}

/** One assertion keyword, decided by Ajv and phrased as the tools phrase it. */
function assertion(keyword: string, instance: Instance, node: SchemaObject, env: WalkEnv): SchemaError[] {
  return assertionErrors(keyword, node, instance, env.assertions);
}

/** One applicator keyword: which subschema judges what, and how the errors hang together. */
function applicator(
  keyword: string,
  instance: Instance,
  node: SchemaObject,
  base: string,
  env: WalkEnv,
): SchemaError[] {
  const value = node[keyword];
  switch (keyword) {
    case '$ref': {
      const target = env.resolve(String(value), base);
      return descend(instance, target.node, target.base, env);
    }

    case 'properties': {
      const declared = value as Record<string, SchemaNode>;
      const errors: SchemaError[] = [];
      if (!isJsonObject(instance.tree)) return errors;
      for (const [name, subschema] of Object.entries(declared)) {
        if (!hasName(instance, name)) continue;
        errors.push(...descend(member(instance, name), subschema, base, env, name, name));
      }
      return errors;
    }

    case 'patternProperties': {
      const declared = value as Record<string, SchemaNode>;
      const errors: SchemaError[] = [];
      if (!isJsonObject(instance.tree)) return errors;
      for (const [pattern, subschema] of Object.entries(declared)) {
        const regex = new RegExp(pattern, 'u');
        for (const name of memberNames(instance.tree)) {
          if (!regex.test(name)) continue;
          errors.push(...descend(member(instance, name), subschema, base, env, name, pattern));
        }
      }
      return errors;
    }

    case 'additionalProperties': {
      if (!isJsonObject(instance.tree)) return [];
      const extras = additionalNames(node, instance);
      if (isSchemaObject(value as SchemaNode)) {
        // A schema rather than `false`: every additional member is judged by it, in turn.
        const errors: SchemaError[] = [];
        for (const extra of extras) {
          errors.push(...descend(member(instance, extra), value as SchemaNode, base, env, extra));
        }
        return errors;
      }
      if (value === false && extras.length > 0) {
        return additionalPropertiesErrors(node, instance, env.assertions);
      }
      return [];
    }

    case 'propertyNames': {
      if (!isJsonObject(instance.tree)) return [];
      const errors: SchemaError[] = [];
      for (const name of memberNames(instance.tree)) {
        errors.push(...descend(literalInstance(name), value as SchemaNode, base, env));
      }
      return errors;
    }

    case 'prefixItems': {
      if (!isJsonArray(instance.tree)) return [];
      const declared = value as readonly SchemaNode[];
      const errors: SchemaError[] = [];
      const shared = Math.min(declared.length, instance.tree.length);
      for (let index = 0; index < shared; index += 1) {
        const subschema = declared[index] as SchemaNode;
        errors.push(...descend(element(instance, index), subschema, base, env, index, index));
      }
      return errors;
    }

    case 'items': {
      if (!isJsonArray(instance.tree)) return [];
      const prefixItems = node['prefixItems'];
      const prefix = Array.isArray(prefixItems) ? prefixItems.length : 0;
      const total = instance.tree.length;
      if (total - prefix <= 0) return [];
      if (value === false) return itemsFalseErrors(node, instance, env.assertions);
      const errors: SchemaError[] = [];
      for (let index = prefix; index < total; index += 1) {
        errors.push(...descend(element(instance, index), value as SchemaNode, base, env, index));
      }
      return errors;
    }

    case 'allOf': {
      const branches = value as readonly SchemaNode[];
      const errors: SchemaError[] = [];
      branches.forEach((branch, index) => {
        errors.push(...descend(instance, branch, base, env, undefined, index));
      });
      return errors;
    }

    case 'anyOf':
    case 'oneOf':
      return choice(keyword, instance, value as readonly SchemaNode[], base, env);

    case 'not':
      return isValid(instance, value as SchemaNode, base, env)
        ? [
            schemaError(
              `${pythonRepr(instance.tree)} should not be valid under ${pythonRepr(value)}`,
            ),
          ]
        : [];

    case 'if': {
      // `if` never speaks for itself: it chooses which of `then` and `else` judges the instance.
      if (isValid(instance, value as SchemaNode, base, env)) {
        const then = node['then'];
        return then === undefined ? [] : descend(instance, then as SchemaNode, base, env, undefined, 'then');
      }
      const otherwise = node['else'];
      return otherwise === undefined
        ? []
        : descend(instance, otherwise as SchemaNode, base, env, undefined, 'else');
    }

    case 'dependentSchemas': {
      const declared = value as Record<string, SchemaNode>;
      const errors: SchemaError[] = [];
      if (!isJsonObject(instance.tree)) return errors;
      for (const [property, subschema] of Object.entries(declared)) {
        if (!hasName(instance, property)) continue;
        errors.push(...descend(instance, subschema, base, env, undefined, property));
      }
      return errors;
    }

    default:
      return unsupported(keyword);
  }
}

/**
 * `anyOf` and `oneOf`: the branch point.
 *
 * Both stop at the first branch that holds. When none does, they raise one error whose `context`
 * carries every branch's errors — the shape `best_match` descends and the model stage prints as
 * one line. `oneOf` alone then asks whether a later branch holds as well, which is the schema
 * saying two readings of the same value, and names them all.
 */
function choice(
  keyword: 'anyOf' | 'oneOf',
  instance: Instance,
  branches: readonly SchemaNode[],
  base: string,
  env: WalkEnv,
): SchemaError[] {
  const collected: SchemaError[] = [];
  let taken = -1;
  for (let index = 0; index < branches.length; index += 1) {
    const branch = branches[index] as SchemaNode;
    const errors = descend(instance, branch, base, env, undefined, index);
    if (errors.length === 0) {
      taken = index;
      break;
    }
    collected.push(...errors);
  }
  if (taken < 0) {
    const error = schemaError(
      `${pythonRepr(instance.tree)} is not valid under any of the given schemas`,
      { context: collected },
    );
    for (const branchError of collected) branchError.parent = error;
    return [error];
  }
  if (keyword === 'anyOf') return [];
  const also = branches
    .slice(taken + 1)
    .filter((branch) => isValid(instance, branch, base, env))
    .concat([branches[taken] as SchemaNode]);
  if (also.length === 1) return [];
  const reprs = also.map((branch) => pythonRepr(branch)).join(', ');
  return [schemaError(`${pythonRepr(instance.tree)} is valid under each of ${reprs}`)];
}
