/**
 * The one mapping from a keyword to the tools' wording — and the one place Ajv is asked whether
 * a keyword holds.
 *
 * The plan's D4 makes Ajv the validator: "structural conformance is Ajv on the schema files
 * themselves". Every keyword below is decided by Ajv, on a schema built out of that keyword's own
 * value as the file writes it; nothing here re-implements a type check, a pattern match or a
 * uniqueness test. What is implemented here is the *phrasing*: `jsonschema`'s message for the
 * keyword, which is what `--validate` prints and what `tests/rejections` matches on (D2, §7 F1).
 *
 * The keywords are the assertions — the ones that judge a value where they stand. The applicators
 * (`$ref`, `properties`, `oneOf`, `if` …) compose those judgements instead of making one, and
 * they live in `walk.ts`.
 *
 * Ajv's parameters are used where a message names a particular member: which property is missing,
 * which one was not expected. Their *order* is taken from the schema rather than from Ajv, since
 * `jsonschema` walks `required` and `dependentRequired` in the order the file writes them and the
 * message list follows.
 */
import type { ErrorObject } from 'ajv';

import type { JsonValue } from '../json/tree.js';
import { schemaError, type SchemaError } from './errors.js';
import { comparePythonStrings, pythonRepr } from './repr.js';
import type { Instance, SchemaObject } from './types.js';

/**
 * Ajv, as the assertions use it: the errors it raises for one keyword of one schema node.
 *
 * The engine compiles `{<keyword>: <the node's value>}` — with the siblings a keyword cannot be
 * read without, which is `properties` for `additionalProperties: false` and `prefixItems` for
 * `items: false` — and caches the compiled validator against the node. An empty list is Ajv
 * saying the keyword holds.
 */
export interface AssertionEngine {
  failures(node: SchemaObject, keyword: string, data: unknown): readonly ErrorObject[];
}

/** The keywords this module decides. Everything else at a node is an applicator or an annotation. */
export const ASSERTION_KEYWORDS: readonly string[] = [
  'const',
  'dependentRequired',
  'enum',
  'exclusiveMaximum',
  'exclusiveMinimum',
  'format',
  'maxItems',
  'maxLength',
  'maxProperties',
  'maximum',
  'minItems',
  'minLength',
  'minProperties',
  'minimum',
  'multipleOf',
  'pattern',
  'required',
  'type',
  'uniqueItems',
];

/** `str()` of a number, which is what `multipleOf`'s message interpolates instead of `repr`. */
function pythonStr(value: unknown): string {
  return typeof value === 'string' ? value : pythonRepr(value);
}

/** `extras_msg`: the repr'd names and the verb that agrees with how many there are. */
function extrasMessage(extras: readonly string[]): [string, string] {
  return [extras.map((extra) => pythonRepr(extra)).join(', '), extras.length === 1 ? 'was' : 'were'];
}

/** The names Ajv reported as missing, ordered as the schema's own list writes them. */
function inSchemaOrder(reported: readonly string[], declared: readonly unknown[]): string[] {
  const wanted = new Set(reported);
  return declared.filter((name): name is string => typeof name === 'string' && wanted.has(name));
}

/**
 * The errors one assertion keyword of one node raises, in the tools' words.
 *
 * Ajv answers first; the message is written only for what it refused. A keyword Ajv accepts
 * yields nothing, which is how a valid document produces an empty problem list.
 */
export function assertionErrors(
  keyword: string,
  node: SchemaObject,
  instance: Instance,
  engine: AssertionEngine,
): SchemaError[] {
  const value = node[keyword];
  const failures = engine.failures(node, keyword, instance.plain);
  if (failures.length === 0) return [];
  const single = (message: string): SchemaError[] => [schemaError(message)];
  // The instance is interpolated into most of the messages; a node the walk never refuses is
  // never written out, which matters where the instance is a whole composition.
  const subject = (): string => pythonRepr(instance.tree);

  switch (keyword) {
    case 'const':
      return single(`${pythonRepr(value)} was expected`);

    case 'dependentRequired': {
      // One line per missing dependency, in the order the schema declares the dependencies.
      const declared = value as Record<string, readonly unknown[]>;
      const messages: string[] = [];
      for (const [property, dependency] of Object.entries(declared)) {
        const missing = failures
          .filter((failure) => failure.params['property'] === property)
          .map((failure) => String(failure.params['missingProperty']));
        for (const each of inSchemaOrder(missing, dependency)) {
          messages.push(`${pythonRepr(each)} is a dependency of ${pythonRepr(property)}`);
        }
      }
      return messages.map((message) => schemaError(message));
    }

    case 'enum':
      return single(`${subject()} is not one of ${pythonRepr(value)}`);

    case 'exclusiveMaximum':
      return single(`${subject()} is greater than or equal to the maximum of ${pythonRepr(value)}`);

    case 'exclusiveMinimum':
      return single(`${subject()} is less than or equal to the minimum of ${pythonRepr(value)}`);

    case 'format':
      // The tools build their validator without a format checker, so `format` decides nothing;
      // the engine is configured the same way and never reports one. Kept so that the keyword is
      // known rather than unhandled.
      return [];

    case 'maxItems':
      return single(`${subject()} ${value === 0 ? 'is expected to be empty' : 'is too long'}`);

    case 'maxLength':
      return single(`${subject()} ${value === 0 ? 'is expected to be empty' : 'is too long'}`);

    case 'maxProperties':
      return single(`${subject()} ${value === 0 ? 'is expected to be empty' : 'has too many properties'}`);

    case 'maximum':
      return single(`${subject()} is greater than the maximum of ${pythonRepr(value)}`);

    case 'minItems':
      return single(`${subject()} ${value === 1 ? 'should be non-empty' : 'is too short'}`);

    case 'minLength':
      return single(`${subject()} ${value === 1 ? 'should be non-empty' : 'is too short'}`);

    case 'minProperties':
      return single(
        `${subject()} ${value === 1 ? 'should be non-empty' : 'does not have enough properties'}`,
      );

    case 'minimum':
      return single(`${subject()} is less than the minimum of ${pythonRepr(value)}`);

    case 'multipleOf':
      return single(`${subject()} is not a multiple of ${pythonStr(value)}`);

    case 'pattern':
      return single(`${subject()} does not match ${pythonRepr(value)}`);

    case 'required': {
      const declared = value as readonly unknown[];
      const missing = failures.map((failure) => String(failure.params['missingProperty']));
      return inSchemaOrder(missing, declared).map((property) =>
        schemaError(`${pythonRepr(property)} is a required property`),
      );
    }

    case 'type': {
      const types = typeof value === 'string' ? [value] : (value as readonly unknown[]);
      return single(`${subject()} is not of type ${types.map((name) => pythonRepr(name)).join(', ')}`);
    }

    case 'uniqueItems':
      return single(`${subject()} has non-unique elements`);

    default:
      throw new Error(`schema: no wording for the assertion keyword '${keyword}'`);
  }
}

/**
 * `additionalProperties: false` — an assertion, since it judges the object where it stands.
 *
 * Which names are additional is Ajv's answer, from the same `properties` and `patternProperties`
 * the node declares; the message is `jsonschema`'s, with the names sorted, and takes its other
 * form when the node also carries `patternProperties`, because then the reader needs the regexes
 * rather than the list of allowed names.
 */
export function additionalPropertiesErrors(
  node: SchemaObject,
  instance: Instance,
  engine: AssertionEngine,
): SchemaError[] {
  const failures = engine.failures(node, 'additionalProperties', instance.plain);
  if (failures.length === 0) return [];
  const extras = failures
    .map((failure) => String(failure.params['additionalProperty']))
    .sort(comparePythonStrings);
  const patterns = node['patternProperties'];
  if (patterns !== undefined && typeof patterns === 'object' && patterns !== null) {
    const verb = extras.length === 1 ? 'does' : 'do';
    const joined = extras.map((extra) => pythonRepr(extra)).join(', ');
    const regexes = Object.keys(patterns)
      .sort(comparePythonStrings)
      .map((pattern) => pythonRepr(pattern))
      .join(', ');
    return [schemaError(`${joined} ${verb} not match any of the regexes: ${regexes}`, { members: extras })];
  }
  const [joined, verb] = extrasMessage(extras);
  return [
    schemaError(`Additional properties are not allowed (${joined} ${verb} unexpected)`, {
      members: extras,
    }),
  ];
}

/**
 * `items: false` — an assertion too: no element may sit beyond the ones `prefixItems` covers.
 *
 * No schema of the repository writes it; it is here because `jsonschema`'s vocabulary has it, and
 * a keyword the walk cannot phrase is refused rather than passed over in silence.
 */
export function itemsFalseErrors(
  node: SchemaObject,
  instance: Instance,
  engine: AssertionEngine,
): SchemaError[] {
  const failures = engine.failures(node, 'items', instance.plain);
  if (failures.length === 0) return [];
  const prefixItems = node['prefixItems'];
  const prefix = Array.isArray(prefixItems) ? prefixItems.length : 0;
  const items: readonly JsonValue[] = Array.isArray(instance.tree) ? instance.tree : [];
  const extra = items.length - prefix;
  const rest: JsonValue | readonly JsonValue[] =
    extra === 1 ? (items[prefix] as JsonValue) : items.slice(prefix);
  const noun = prefix === 1 ? 'item' : 'items';
  return [
    schemaError(
      `Expected at most ${String(prefix)} ${noun} but found ${String(extra)} extra: ${pythonRepr(rest)}`,
    ),
  ];
}
