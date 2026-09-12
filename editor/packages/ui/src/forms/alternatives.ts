/**
 * The alternatives of a union, as a chooser offers them.
 *
 * Plan §1: "one schema walker renders any `$def`: … `oneOf` (a chooser labelled by the
 * alternative's discriminating required key **or `const`**)". The reading of a union is the
 * core's — `vocabulary()` walks every `oneOf` of every loaded schema and says which required keys
 * tell the alternatives apart (D3, §1's "a tagged union's tags are the `required` keys of its
 * `oneOf` members") — and this module does three things with the answer that a chooser needs and
 * the vocabulary does not do.
 *
 * **It flattens a union of unions.** `argument_value` is `scalar_expression` or
 * `record_argument`, and `scalar_expression` is itself seven alternatives; a chooser that stopped
 * at the outer union would offer "something with no required key" and "record", which is no
 * chooser at all. Flattened, it offers exactly the source modes of plan §4.12 — literal,
 * quantity, index, an expression — with `record` beside them, which is what the artboard's
 * row-states key draws.
 *
 * **It reads the `const` half of §1's sentence.** `quantity_type`'s five alternatives all require
 * `kind` and nothing else; what tells them apart is the `const` that each one's `kind` is fixed
 * to, which is a member's constant and not the alternative's own, so the vocabulary's tags cannot
 * see it. When one member is fixed to a different constant in every alternative, that constant is
 * the label — `cardinality`, `real`, `physical`, `enum`, `boolean` — read from the schema at the
 * moment the chooser is built and written down nowhere.
 *
 * **It keeps the chain each alternative was reached through**, so that a presentation binding
 * written for the *union* serves its alternatives: the expression editor is bound to
 * `scalar_expression`, and it is what edits an `op` tree reached through `argument_value`.
 *
 * Which alternative a written value *is* is not decided here by resemblance: it is
 * `registry.accepts`, one place of the one grammar compiled by the one validator. It has to be.
 * `unary_operation_expression` and `nary_operation_expression` require the same two keys and are
 * separated only by the operators their `op` admits (feature 1.1), so a tag match would answer
 * both and the row would show the wrong arity.
 */
import {
  alternativeLabel,
  isJsonObject,
  mergeFacts,
  type JsonValue,
  type SchemaRegistry,
  type Vocabulary,
  type VocabularyValue,
} from '@tensorspine/lang';
import type { SchemaShapes, Shape } from '@tensorspine/store';

/** One alternative of a union, flattened and with the chain it was reached through. */
export interface Alternative {
  /** What labels it: a discriminating `const`, its discriminating required keys, or its type. */
  readonly label: string;
  /** Every tag the vocabulary gives it; empty when it has none at all. */
  readonly tags: readonly string[];
  /** What it *is*: what its `$ref` names, or where the alternative itself is written. */
  readonly anchor: string;
  /** Where the alternative is written, always — `…#/$defs/location/oneOf/1`. */
  readonly place: string;
  /** The union it belongs to, which is not the outermost one when unions nest. */
  readonly union: string;
  /** The anchors a binding for it is looked up by, most specific first. */
  readonly ancestry: readonly string[];
}

/** A union as a chooser reads it. */
export interface Alternation {
  /** The union's own anchor. */
  readonly anchor: string;
  /** Its alternatives, flattened, in the order the schemas write them. */
  readonly alternatives: readonly Alternative[];
  /** The member whose `const` tells them apart, when one does. */
  readonly discriminator: string | null;
}

/** The union at an anchor, flattened, or `undefined` when the anchor names no union. */
export function alternationAt(
  vocabulary: Vocabulary,
  shapes: SchemaShapes,
  anchor: string,
): Alternation | undefined {
  if (vocabulary.unionAt(anchor) === undefined) return undefined;
  const found: Alternative[] = [];
  gather(vocabulary, anchor, [], new Set<string>(), found);
  const discriminator = discriminatorOf(shapes, found);
  if (discriminator === null) return { anchor, alternatives: found, discriminator };
  return {
    anchor,
    discriminator,
    alternatives: found.map((one) => {
      const fixed = constantsOf(shapes, one.anchor).get(discriminator);
      return fixed === undefined ? one : { ...one, label: printed(fixed) };
    }),
  };
}

function gather(
  vocabulary: Vocabulary,
  anchor: string,
  outer: readonly string[],
  entered: Set<string>,
  into: Alternative[],
): void {
  const union = vocabulary.unionAt(anchor);
  if (union === undefined || entered.has(anchor)) return;
  entered.add(anchor);
  const chain = [anchor, ...outer];
  for (const alternative of union.alternatives) {
    const place = `${union.schema}${alternative.place}`;
    const own = alternative.target ?? place;
    const inner = vocabulary.unionAt(own) === undefined ? undefined : own;
    if (inner !== undefined && !entered.has(inner)) {
      gather(vocabulary, inner, chain, entered, into);
      continue;
    }
    into.push({
      label: alternativeLabel(alternative),
      tags: alternative.tags,
      anchor: own,
      place,
      union: anchor,
      ancestry: own === place ? chain : [own, ...chain],
    });
  }
}

/**
 * The member whose `const` is different in every alternative, or `null`.
 *
 * Every alternative must fix it, and no two to the same value: anything weaker would label two
 * alternatives the same, which is the state the chooser reports rather than papers over.
 */
function discriminatorOf(shapes: SchemaShapes, alternatives: readonly Alternative[]): string | null {
  const first = alternatives[0];
  if (first === undefined || alternatives.length < 2) return null;
  const fixed = alternatives.map((one) => constantsOf(shapes, one.anchor));
  for (const name of [...(fixed[0] ?? new Map<string, VocabularyValue>()).keys()]) {
    const values = fixed.map((one) => one.get(name));
    if (values.some((value) => value === undefined)) continue;
    const printedValues = values.map((value) => printed(value as VocabularyValue));
    if (new Set(printedValues).size === printedValues.length) return name;
  }
  return null;
}

/** The members of a place that are fixed to one value, by name. */
function constantsOf(shapes: SchemaShapes, anchor: string): Map<string, VocabularyValue> {
  const shape = shapes.at(anchor);
  const facts = mergeFacts(shape.direct.map((place) => place.node));
  const found = new Map<string, VocabularyValue>();
  for (const name of facts.members) {
    const member = factsOfMember(shapes, shape, name);
    if (member.fixed && member.constant !== undefined) found.set(name, member.constant);
  }
  return found;
}

/** The facts of one member of a shape. */
function factsOfMember(
  shapes: SchemaShapes,
  shape: Shape,
  name: string,
): ReturnType<typeof mergeFacts> {
  const member = shapes.member(shape, name);
  const chain = member.direct.length > 0 ? member.direct : member.all;
  return mergeFacts(chain.map((place) => place.node));
}

/** A scalar as a label prints it. */
function printed(value: VocabularyValue): string {
  return typeof value === 'string' ? value : JSON.stringify(value);
}

/**
 * The alternative a value **resembles**, by the keys it carries — the reading a repair needs.
 *
 * {@link chosenOf} is the grammar's verdict and stays it; this is the answer to a different
 * question: *which alternative's rows does a form draw* when the value matches none. A value
 * being edited is between two states — a `stack` whose `axis` is still empty is no `stack` to
 * Ajv — and a form that drew nothing there would be a form that could not repair what it had just
 * written itself (the chooser writes the blank of the alternative, and a blank identifier is not
 * one). So the rows shown are the alternative whose discriminating `const` the value carries, or
 * whose required keys it all carries; Ajv's refusal is on the row that breaks it.
 *
 * `undefined` where nothing is written or nothing resembles, which is the honest empty answer.
 */
export function resembling(
  alternation: Alternation,
  value: JsonValue | undefined,
): Alternative | undefined {
  if (value === undefined || !isJsonObject(value)) return undefined;
  const names = new Set(value.members.map((member) => member.name));
  if (alternation.discriminator !== null) {
    const written = value.members.find((member) => member.name === alternation.discriminator)?.value;
    if (typeof written === 'string') {
      const found = alternation.alternatives.find((one) => one.label === written);
      if (found !== undefined) return found;
    }
  }
  return alternation.alternatives.find(
    (one) => one.tags.length > 0 && one.tags.every((tag) => names.has(tag)),
  );
}

/**
 * The alternative a value is, by the grammar's own verdict.
 *
 * `null` when nothing is written or nothing accepts it — which is a state a row shows and never
 * refuses (Q5: the author wires first and fixes afterwards). More than one acceptance is a union
 * whose alternatives overlap; the first is taken and the caller reports it.
 */
export function chosenOf(
  registry: SchemaRegistry,
  alternation: Alternation,
  value: JsonValue | undefined,
): { readonly alternative: Alternative; readonly ambiguous: boolean } | null {
  if (value === undefined) return null;
  const matched = alternation.alternatives.filter((one) => registry.accepts(value, one.anchor));
  const first = matched[0];
  if (first === undefined) return null;
  return { alternative: first, ambiguous: matched.length > 1 };
}
