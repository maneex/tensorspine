/**
 * Which widget edits a place, decided from what its schema asserts.
 *
 * Plan §4.12 gives a table keyed by `argument_type.kind` — a cardinality is an integer stepper, a
 * real a number field, a physical a number field with its unit as a suffix, a boolean a toggle,
 * an enum a select, a record a section — and the table must not be written down anywhere: those
 * six names are an enumeration of the unit schema, and catching rule §1 (b) is a scan for exactly
 * that. It does not have to be. The tools already publish the answer: `--document
 * primitive-schema` writes each primitive's arguments as **JSON Schema**, a cardinality as
 * `{"type": "integer", "minimum": 1}`, a physical as `{"type": "integer",
 * "x-tensorspine-unit": "tokens"}`, an enum as an `enum`, a record as an object — and those files
 * are vendored with the build and consumed as they are (plan §1, F5). So one rule over
 * {@link SchemaFacts}, applied to the four schemas and to a generated argument schema alike,
 * *is* §4.12's table, and the widget of a `physical` argument carries the whole-number/real
 * distinction §4.12 says is "never typed in the sheet".
 *
 * A presentation binding overrides the decision, and only it can: the expression editor, the
 * condition editor, the physical-name token list, the axis rows and the ordered rule list are
 * what a schema cannot say (plan §1, Appendix B).
 */
import type { SchemaFacts } from '@tensorspine/lang';

import type { FormOption } from './types.js';

/** Free text. */
export const TEXT = 'text';
/** A whole number: plan §4.12's integer stepper. */
export const WHOLE = 'integer';
/** A number: §4.12's number field. */
export const NUMBER = 'number';
/** True or false: §4.12's toggle. */
export const TOGGLE = 'toggle';
/** One of an enumerated set: §4.12's select, its options the schema's own. */
export const SELECT = 'select';
/** An object with declared members: §4.12's section header, its fields one level deeper. */
export const SECTION = 'section';
/** An object whose names are the author's: `additionalProperties` with `propertyNames`. */
export const MAP = 'map';
/** An array. */
export const LIST = 'list';
/** A union: the chooser of plan §1, labelled by each alternative's discriminating key. */
export const CHOOSER = 'chooser';
/** A value the schema fixes: read-only. */
export const FIXED = 'constant';
/** A place that admits more than one scalar type: `scalar_literal` is the one of the grammar. */
export const SCALAR = 'scalar';
/** A place whose only admissible value is the absent one. */
export const NOTHING = 'nothing';
/** The generic widget of plan §1: a labelled JSON editor, for a place nothing else answers. */
export const JSON_EDITOR = 'json';

/** The widgets this module decides. A binding may name one that is not here. */
export const WIDGETS: readonly string[] = [
  TEXT,
  WHOLE,
  NUMBER,
  TOGGLE,
  SELECT,
  SECTION,
  MAP,
  LIST,
  CHOOSER,
  FIXED,
  SCALAR,
  NOTHING,
  JSON_EDITOR,
];

/**
 * The widget a place gets from its schema alone.
 *
 * `alternation` is passed rather than read from the facts because a union is recognised through
 * the core's vocabulary — "a tagged union's tags are the `required` keys of its `oneOf` members"
 * is one reading, in one place (plan §1, D3) — and a place can carry a `oneOf` the vocabulary
 * answers for at an anchor further along its chain.
 */
export function widgetOf(facts: SchemaFacts, alternation: boolean): string {
  if (facts.fixed) return FIXED;
  if (facts.choices !== null) return SELECT;
  if (alternation) return CHOOSER;
  // A place with members, closed members or named members is an object whether or not it says
  // so: the walker reads what the schema asserts, not only what it declares.
  if (facts.holdsObject || facts.keyed || facts.named || facts.closed || facts.members.length > 0) {
    return facts.keyed ? MAP : SECTION;
  }
  if (facts.holdsArray || facts.listed) return LIST;
  const scalars = [facts.holdsText, facts.holdsWholeNumber, facts.holdsNumber, facts.holdsTruth];
  if (scalars.filter(Boolean).length > 1) return SCALAR;
  if (facts.holdsText) return TEXT;
  if (facts.holdsWholeNumber) return WHOLE;
  if (facts.holdsNumber) return NUMBER;
  if (facts.holdsTruth) return TOGGLE;
  if (facts.holdsOnlyNothing) return NOTHING;
  return JSON_EDITOR;
}

/** Whether a widget edits one value on the row itself rather than a shape below it. */
export function editsOneValue(widget: string): boolean {
  return (
    widget === TEXT ||
    widget === WHOLE ||
    widget === NUMBER ||
    widget === TOGGLE ||
    widget === SELECT ||
    widget === FIXED ||
    widget === SCALAR ||
    widget === NOTHING
  );
}

/**
 * The blank a mode starts from: the first value the place admits.
 *
 * Not a default of the editor's — the language's defaults are the declaration's and are never
 * written (§4.12) — but the first thing the *grammar* accepts there, so that the document the
 * gesture leaves is on the schema (D5) and the core's own verdict is what judges it. It lives
 * beside {@link widgetOf} because it is the same kind of rule over the same facts, and because
 * both the argument sheet (§4.12) and the expression tree (§4.13) start a value from it.
 */
export function blankOf(
  facts: SchemaFacts,
  options: readonly FormOption[] | undefined,
): string | number | boolean {
  const first = options?.[0]?.value;
  if (first !== undefined && first !== null) return first;
  // A place that admits one type has one blank whatever the order below is; a place that admits
  // several is the grammar's own `scalar_literal` — a literal of an expression — and there the
  // number comes first, because that is what the repository's expressions are made of (678 number
  // literals against 140 booleans, corpus and reference base together).
  if (facts.holdsWholeNumber || facts.holdsNumber) return 0;
  if (facts.holdsTruth) return false;
  return '';
}
