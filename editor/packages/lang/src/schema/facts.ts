/**
 * What one schema node asserts, in the shape a generated form reads it.
 *
 * The plan's §1 says a form is generated from the schema and never written: a select's options
 * are the schema's `enum`, a read-only row is its `const`, a number row's bounds are its
 * `minimum` and `maximum`, a map is `additionalProperties` with `propertyNames`. Somebody has to
 * name those keywords, and this module is where they are named — **once**, in the core.
 *
 * Why here and not in the forms. Two of draft 2020-12's own words collide with the language's
 * vocabulary: the keyword `enum` and the type `boolean` are also values of the unit schema's
 * `argument_type.kind`. Catching rule §1 (b) is a whole-literal scan of every interface source,
 * so an interface module that wrote `node['enum']` or `=== 'boolean'` would be reported, and
 * rightly — the scan cannot tell the meta-schema's vocabulary from the language's, and feature
 * 1.12 measured that it must not try (thirty-two of the hundred and one values are ordinary
 * English). The core is the reading of the grammar and is scoped out of that rule by design, so
 * the meta-schema's words live here and the forms read *facts*: `holdsTruth`, `choices`,
 * `constant`. Nothing of the language is named here, only JSON Schema's.
 *
 * A fact is read from **one node**. A place of a document is described by a chain of them — the
 * member's own subschema, what its `$ref` names, the members of its `allOf` — and
 * {@link mergeFacts} takes the chain most specific first, each field from the first node that
 * states it. What a chain's *alternatives* say is deliberately not merged in: the model schema's
 * own root shows why (`anyOf` of two `minProperties`, neither of them the root's own), and the
 * store's `SchemaShapes` keeps the same two readings apart for the same reason.
 */
import { isSchemaObject, type SchemaNode } from './types.js';
import type { VocabularyValue } from './vocabulary.js';

/** The `type` names of draft 2020-12, named here so that no form has to name them. */
const OBJECT = 'object';
const ARRAY = 'array';
const TEXT = 'string';
const WHOLE = 'integer';
const NUMBER = 'number';
const TRUTH = 'boolean';
const NOTHING = 'null';

/** The keywords {@link factsOf} reads. A node that writes none of them states nothing a form can render. */
const READ: readonly string[] = [
  'additionalProperties',
  'const',
  'enum',
  'exclusiveMaximum',
  'exclusiveMinimum',
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
  'oneOf',
  'pattern',
  'prefixItems',
  'properties',
  'propertyNames',
  'type',
  'uniqueItems',
];

/** What a schema node asserts about the value at its place. */
export interface SchemaFacts {
  /** The `type`s the node declares, in the order it writes them; empty when it declares none. */
  readonly types: readonly string[];
  /** Whether the node admits the absent-value type beside another one. */
  readonly nullable: boolean;
  /** Whether the node admits an object. */
  readonly holdsObject: boolean;
  /** Whether the node admits an array. */
  readonly holdsArray: boolean;
  /** Whether the node admits text. */
  readonly holdsText: boolean;
  /** Whether the node admits a whole number — plan §4.12's integer stepper. */
  readonly holdsWholeNumber: boolean;
  /** Whether the node admits a number with a fraction — §4.12's number field. */
  readonly holdsNumber: boolean;
  /** Whether the node admits `true` and `false`. */
  readonly holdsTruth: boolean;
  /** Whether the only type the node declares is the absent one. */
  readonly holdsOnlyNothing: boolean;
  /** The admissible values, in the order the schema writes them; `null` when the node lists none. */
  readonly choices: readonly VocabularyValue[] | null;
  /** Whether the node fixes one value. */
  readonly fixed: boolean;
  /** That value, when it is a scalar the interface can print. */
  readonly constant: VocabularyValue | undefined;
  /** The node's own `title`: a readable label where the definition's name is not one. */
  readonly title: string | null;
  /** The node's own `description`: the help text of plan §1 ("Help text is the schema's"). */
  readonly description: string | null;
  /** The member names the node declares, in the order it writes them. */
  readonly members: readonly string[];
  /** The member names the node requires. */
  readonly required: readonly string[];
  /** Whether the node refuses a member it does not declare. */
  readonly closed: boolean;
  /** Whether the node describes the values of a map (`additionalProperties` as a schema). */
  readonly keyed: boolean;
  /** Whether the node constrains the names of a map (`propertyNames`). */
  readonly named: boolean;
  /** Whether the node describes the elements of an array. */
  readonly listed: boolean;
  /** Whether the node alternates (`oneOf`): a chooser, not a field. */
  readonly alternation: boolean;
  /** Whether the node is conditional (`if`): what a branch adds holds only under it. */
  readonly conditional: boolean;
  /** The lower bound, and whether it excludes its own value. */
  readonly minimum: number | null;
  readonly minimumExcluded: boolean;
  /** The upper bound, and whether it excludes its own value. */
  readonly maximum: number | null;
  readonly maximumExcluded: boolean;
  readonly minLength: number | null;
  readonly maxLength: number | null;
  /** The pattern a text must match, as the schema writes it. Compiling it is `pattern.ts`'s. */
  readonly pattern: string | null;
  readonly minItems: number | null;
  readonly maxItems: number | null;
  readonly uniqueItems: boolean;
  readonly minProperties: number | null;
  readonly maxProperties: number | null;
  /** Every keyword the node writes, in the order it writes them. */
  readonly keywords: readonly string[];
  /** Whether the node states anything a form can render. */
  readonly states: boolean;
}

/** The facts of a node that says nothing: what `true` asserts, and what a missing place answers. */
export const NO_FACTS: SchemaFacts = {
  types: [],
  nullable: false,
  holdsObject: false,
  holdsArray: false,
  holdsText: false,
  holdsWholeNumber: false,
  holdsNumber: false,
  holdsTruth: false,
  holdsOnlyNothing: false,
  choices: null,
  fixed: false,
  constant: undefined,
  title: null,
  description: null,
  members: [],
  required: [],
  closed: false,
  keyed: false,
  named: false,
  listed: false,
  alternation: false,
  conditional: false,
  minimum: null,
  minimumExcluded: false,
  maximum: null,
  maximumExcluded: false,
  minLength: null,
  maxLength: null,
  pattern: null,
  minItems: null,
  maxItems: null,
  uniqueItems: false,
  minProperties: null,
  maxProperties: null,
  keywords: [],
  states: false,
};

/** Whether a value is one an `enum` or a `const` may hold and a row may print. */
function isValue(value: unknown): value is VocabularyValue {
  return (
    value === null ||
    typeof value === 'string' ||
    typeof value === 'number' ||
    typeof value === 'boolean'
  );
}

/** A keyword's value when it is a number, `null` otherwise. */
function numberOf(node: Record<string, unknown>, keyword: string): number | null {
  const value = node[keyword];
  return typeof value === 'number' ? value : null;
}

/** A keyword's value when it is a string, `null` otherwise. */
function textOf(node: Record<string, unknown>, keyword: string): string | null {
  const value = node[keyword];
  return typeof value === 'string' ? value : null;
}

/** The names a `properties`-shaped keyword declares, in the order the schema writes them. */
function namesOf(node: Record<string, unknown>, keyword: string): readonly string[] {
  const declared = node[keyword];
  if (declared === null || typeof declared !== 'object' || Array.isArray(declared)) return [];
  return Object.keys(declared);
}

/** The `type`s a node declares, as strings, in the order it writes them. */
function typesOf(node: Record<string, unknown>): readonly string[] {
  const declared = node['type'];
  if (typeof declared === 'string') return [declared];
  if (Array.isArray(declared)) return declared.filter((one): one is string => typeof one === 'string');
  return [];
}

/** What one schema node asserts. `true` and `false` assert nothing a form can render. */
export function factsOf(schema: SchemaNode): SchemaFacts {
  if (!isSchemaObject(schema)) return NO_FACTS;
  const node = schema as Record<string, unknown>;
  const types = typesOf(node);
  const beside = types.filter((one) => one !== NOTHING);
  const values = node['enum'];
  const constant: unknown = node['const'];
  const additional = node['additionalProperties'];
  const exclusiveLower = numberOf(node, 'exclusiveMinimum');
  const exclusiveUpper = numberOf(node, 'exclusiveMaximum');
  const lower = numberOf(node, 'minimum');
  const upper = numberOf(node, 'maximum');
  const required = node['required'];
  const keywords = Object.keys(node);
  return {
    types,
    nullable: types.includes(NOTHING) && beside.length > 0,
    holdsObject: beside.includes(OBJECT),
    holdsArray: beside.includes(ARRAY),
    holdsText: beside.includes(TEXT),
    holdsWholeNumber: beside.includes(WHOLE),
    holdsNumber: beside.includes(NUMBER),
    holdsTruth: beside.includes(TRUTH),
    holdsOnlyNothing: types.length > 0 && beside.length === 0,
    choices: Array.isArray(values) ? values.filter(isValue) : null,
    fixed: 'const' in node,
    constant: isValue(constant) ? constant : undefined,
    title: textOf(node, 'title'),
    description: textOf(node, 'description'),
    members: namesOf(node, 'properties'),
    required: Array.isArray(required)
      ? required.filter((one): one is string => typeof one === 'string')
      : [],
    closed: additional === false,
    keyed: additional !== undefined && additional !== true && additional !== false,
    named: node['propertyNames'] !== undefined,
    listed: node['items'] !== undefined || node['prefixItems'] !== undefined,
    alternation: Array.isArray(node['oneOf']),
    conditional: node['if'] !== undefined,
    minimum: lower ?? exclusiveLower,
    minimumExcluded: lower === null && exclusiveLower !== null,
    maximum: upper ?? exclusiveUpper,
    maximumExcluded: upper === null && exclusiveUpper !== null,
    minLength: numberOf(node, 'minLength'),
    maxLength: numberOf(node, 'maxLength'),
    pattern: textOf(node, 'pattern'),
    minItems: numberOf(node, 'minItems'),
    maxItems: numberOf(node, 'maxItems'),
    uniqueItems: node['uniqueItems'] === true,
    minProperties: numberOf(node, 'minProperties'),
    maxProperties: numberOf(node, 'maxProperties'),
    keywords,
    states: keywords.some((keyword) => READ.includes(keyword)),
  };
}

/**
 * The facts of a chain of nodes, most specific first: each field from the first node that states
 * it, `members` and `required` accumulated in the chain's own order.
 *
 * The chain is a place's `$ref` chain and its `allOf` members — what the place *is* by
 * definition. An alternative under it is not part of the chain and its constraints are not the
 * place's, which is the rule the store's `SchemaShapes` states and this module keeps.
 */
export function mergeFacts(chain: readonly SchemaNode[]): SchemaFacts {
  const facts = chain.map((node) => factsOf(node));
  if (facts.length === 0) return NO_FACTS;
  const first = <T>(pick: (one: SchemaFacts) => T | null): T | null => {
    for (const one of facts) {
      const found = pick(one);
      if (found !== null) return found;
    }
    return null;
  };
  const any = (pick: (one: SchemaFacts) => boolean): boolean => facts.some(pick);
  const members: string[] = [];
  const required: string[] = [];
  const keywords: string[] = [];
  for (const one of facts) {
    for (const name of one.members) if (!members.includes(name)) members.push(name);
    for (const name of one.required) if (!required.includes(name)) required.push(name);
    for (const keyword of one.keywords) if (!keywords.includes(keyword)) keywords.push(keyword);
  }
  const typed = facts.find((one) => one.types.length > 0) ?? NO_FACTS;
  const fixedBy = facts.find((one) => one.fixed);
  const lower = facts.find((one) => one.minimum !== null);
  const upper = facts.find((one) => one.maximum !== null);
  return {
    types: typed.types,
    nullable: typed.nullable,
    holdsObject: typed.holdsObject,
    holdsArray: typed.holdsArray,
    holdsText: typed.holdsText,
    holdsWholeNumber: typed.holdsWholeNumber,
    holdsNumber: typed.holdsNumber,
    holdsTruth: typed.holdsTruth,
    holdsOnlyNothing: typed.holdsOnlyNothing,
    choices: first((one) => one.choices),
    fixed: fixedBy !== undefined,
    constant: fixedBy?.constant,
    title: first((one) => one.title),
    description: first((one) => one.description),
    members,
    required,
    closed: any((one) => one.closed),
    keyed: any((one) => one.keyed),
    named: any((one) => one.named),
    listed: any((one) => one.listed),
    alternation: any((one) => one.alternation),
    conditional: any((one) => one.conditional),
    minimum: lower?.minimum ?? null,
    minimumExcluded: lower?.minimumExcluded ?? false,
    maximum: upper?.maximum ?? null,
    maximumExcluded: upper?.maximumExcluded ?? false,
    minLength: first((one) => one.minLength),
    maxLength: first((one) => one.maxLength),
    pattern: first((one) => one.pattern),
    minItems: first((one) => one.minItems),
    maxItems: first((one) => one.maxItems),
    uniqueItems: any((one) => one.uniqueItems),
    minProperties: first((one) => one.minProperties),
    maxProperties: first((one) => one.maxProperties),
    keywords,
    states: any((one) => one.states),
  };
}
