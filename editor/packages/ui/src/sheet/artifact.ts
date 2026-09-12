/**
 * The generated argument schema of one primitive, read — plan §1's "generated artifacts are
 * consumed, never regenerated" (F5), and §4.12's type → widget table.
 *
 * > **§4.12's type→widget table is not a table.** Its six keys are `argument_type.kind`'s
 * > enumeration, which catching rule (b) forbids the interface to write — and it does not have to:
 * > `--document primitive-schema` already publishes each primitive's arguments as **JSON Schema**
 * > (vendored, consumed as built, F5) … So **one rule over what a schema asserts is that table**,
 * > and it is the same rule that renders the four schemas.  (feature 2.3)
 *
 * This module is the other half of that sentence: it finds the node of the artifact an argument
 * path names and hands back {@link SchemaFacts}, which {@link widgetOf} turns into the row's
 * literal-mode widget, its bounds and its options — the very function that decides a widget over
 * the four schemas. The integer/real distinction §4.12 says is "never typed in the sheet" is the
 * artifact's own `type`; the unit of a physical value is its own annotation.
 *
 * **What the artifact does not carry, and who answers it instead.** `--document primitive-schema`
 * keeps "what JSON Schema can carry" and annotates the rest: it drops every documentation field
 * (`description`, `value_descriptions`, `deprecated`), it drops `structural`, and it maps a kind
 * onto what JSON Schema can say — a cardinality and a whole-number physical are both `integer`,
 * and a *set* domain on a cardinality becomes an `enum`. Those are the core's answer, carried per
 * argument by `describe` (`ArgumentFact`), and the two readings never overlap.
 *
 * **A primitive the build did not see has none**, which F5 states: "a unit the build did not see —
 * a primitive declared in the editor — is served by the generic walker over its declaration, and
 * the artifact is marked *not generated*". There the row's literal widget is what the *model*
 * grammar says a literal is (a scalar), and the sheet says so.
 */
import {
  factsOf,
  isSchemaObject,
  NO_FACTS,
  parse,
  type JsonValue,
  type SchemaFacts,
  type SchemaObject,
} from '@tensorspine/lang';

/**
 * The keys `primitive_schema.py` annotates with, as it names them.
 *
 * They are the *artifact's* vocabulary and no part of the four schemas: "what JSON Schema cannot
 * express — a bound that names another argument, a relation between arguments (an invariant),
 * conditional presence, the unit of a physical value — is kept in `x-tensorspine-*` annotations".
 * Naming them here is naming what the editor consumes, which is the same act as naming a member
 * of a `Problem`.
 */
const PRESENT_WHEN = 'x-tensorspine-present-when';
const DOMAIN = 'x-tensorspine-domain';
const UNIT = 'x-tensorspine-unit';
const LOWER = 'lower';
const UPPER = 'upper';
const ARGUMENT = 'argument';
const INCLUSIVE = 'inclusive';
/** JSON Schema's own two keywords for the members of an object, read to walk a record's fields. */
const PROPERTIES = 'properties';

/** One bound of a declared domain that names another argument, which JSON Schema cannot state. */
export interface ArgumentBound {
  /** Which edge of the interval it is, as the artifact names it. */
  readonly edge: string;
  /** The argument path the bound names. */
  readonly argument: string;
  readonly inclusive: boolean;
}

/** What the artifact says about the place one argument path names. */
export interface ArgumentSchemaFacts {
  /** What the node asserts — the reading {@link widgetOf} decides a widget from. */
  readonly facts: SchemaFacts;
  /** The unit of a physical argument, as the artifact annotates it. */
  readonly unit?: string;
  /** The bounds the artifact could not state, each naming another argument. */
  readonly bounds: readonly ArgumentBound[];
  /** The applicability condition, as the unit's own condition language writes it. */
  readonly presentWhen?: JsonValue;
}

/** Nothing at all: what a place no artifact describes answers. */
export const NO_ARGUMENT_FACTS: ArgumentSchemaFacts = { facts: NO_FACTS, bounds: [] };

/**
 * One primitive's generated argument schema, indexed by argument path.
 *
 * The walk is JSON Schema's own `properties`, one step per segment of the path, so a record's
 * fields are reached exactly as the artifact writes them and nothing of the *model* grammar (the
 * `record` tag an argument value carries) is read here — the artifact describes values, not the
 * expressions a document writes for them.
 */
export class ArgumentSchema {
  private readonly nodes = new Map<string, SchemaObject>();
  private readonly conditions = new Map<string, JsonValue>();

  constructor(private readonly root: unknown) {}

  /** The artifact's facts for one argument path, or nothing where it describes no such place. */
  at(path: string): ArgumentSchemaFacts {
    const node = this.nodeAt(path);
    if (node === undefined) return NO_ARGUMENT_FACTS;
    const unit = node[UNIT];
    const presentWhen = this.conditionAt(path, node[PRESENT_WHEN]);
    return {
      facts: factsOf(node),
      ...(typeof unit === 'string' ? { unit } : {}),
      bounds: boundsOf(node[DOMAIN]),
      ...(presentWhen === undefined ? {} : { presentWhen }),
    };
  }

  /**
   * The applicability condition as the *editor's* tree, so that §4.13's printer can read it.
   *
   * The artifact is a schema and is read as one — plain JSON, which is what `factsOf` takes — but
   * the condition inside it is a value of the **unit's** own condition language, and every reader
   * of that language works on the lexeme-preserving tree the store holds (feature 0.3, D12). So it
   * is re-read through the core's parser, once per place, rather than a second tree model being
   * invented for the one member that needs it.
   */
  private conditionAt(path: string, condition: unknown): JsonValue | undefined {
    if (condition === undefined) return undefined;
    const held = this.conditions.get(path);
    if (held !== undefined) return held;
    const parsed = parse(JSON.stringify(condition));
    this.conditions.set(path, parsed);
    return parsed;
  }

  private nodeAt(path: string): SchemaObject | undefined {
    const held = this.nodes.get(path);
    if (held !== undefined) return held;
    let node = objectOf(this.root);
    for (const segment of path.split('.')) {
      if (node === undefined) return undefined;
      const properties = node[PROPERTIES];
      if (properties === null || typeof properties !== 'object' || Array.isArray(properties)) {
        return undefined;
      }
      node = objectOf((properties as Record<string, unknown>)[segment]);
    }
    if (node === undefined) return undefined;
    this.nodes.set(path, node);
    return node;
  }
}

/** A value as a schema node, or nothing: the artifact is parsed JSON and is read as it stands. */
function objectOf(value: unknown): SchemaObject | undefined {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return undefined;
  return isSchemaObject(value as SchemaObject) ? (value as SchemaObject) : undefined;
}

/** The two edges of an `x-tensorspine-domain` annotation, in the artifact's own order. */
function boundsOf(domain: unknown): ArgumentBound[] {
  if (domain === null || typeof domain !== 'object' || Array.isArray(domain)) return [];
  const found: ArgumentBound[] = [];
  for (const edge of [LOWER, UPPER]) {
    const bound = (domain as Record<string, unknown>)[edge];
    if (bound === null || typeof bound !== 'object' || Array.isArray(bound)) continue;
    const named = (bound as Record<string, unknown>)[ARGUMENT];
    if (typeof named !== 'string') continue;
    found.push({
      edge,
      argument: named,
      inclusive: (bound as Record<string, unknown>)[INCLUSIVE] === true,
    });
  }
  return found;
}

/** How a primitive's artifact is named in the vendor manifest: `attention.dense@1.0.0`. */
export function primitiveId(name: string, version: string): string {
  return `${name}@${version}`;
}
