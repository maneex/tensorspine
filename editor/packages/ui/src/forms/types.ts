/**
 * The form model: what a `$def` and a value become, before anything is drawn.
 *
 * Plan §1's first consequence is that **every form is generated** — "one schema walker renders
 * any `$def`: `$ref`, `oneOf` (a chooser labelled by the alternative's discriminating required
 * key or `const`), `enum` (select), `const` (read-only), `type`, `required`, maps
 * (`additionalProperties` with `propertyNames`), arrays (`minItems`, `uniqueItems`), `if/then` (a
 * `required` that appears under a condition …), `description` (tooltip), `minimum`/`maximum`/
 * `minLength`" — and D7 says what the result looks like: **one flat sheet of rows keyed by JSON
 * pointer**, indented by the depth of the path, the error on the row, the conditionals applied as
 * predicates.
 *
 * So a form is a list of {@link FormRow}s and nothing else. No component, no React, no DOM: the
 * rows are what a sheet, a dialog, the argument sheet (§4.12) and the primitive editor (§4.22)
 * all render, and each of those adds only the facts the *core* answers — applicability, effective
 * values, invariant verdicts — which no schema can state.
 *
 * Three things in here are not the schema's, and each is read from `presentation.json`:
 * {@link FormRow.widget} when a binding names one, {@link FormRow.picker} and
 * {@link FormRow.create} beside a select the schema does not enumerate, and
 * {@link FormMode.referent}, which says that the name in this mode refers to a quantity or an
 * index. Nothing else of the interface knows a word of the language.
 */
import type { PathSegment, VocabularyValue } from '@tensorspine/lang';

/**
 * How a value is edited.
 *
 * The generic ones are {@link WIDGETS}, decided from what the schema asserts. A presentation
 * binding may name another — `expression`, `condition`, `token-list`, `axis-rows`,
 * `ordered-list` — and a widget the interface has no rendering for is §1's "unknown constructs
 * get the generic widget" one level up, which is why this is a name and not a closed union.
 */
export type Widget = string;

/** One admissible value of a select: the schema's `enum`, in the order the schema writes it. */
export interface FormOption {
  /** The value itself, as the schema writes it. */
  readonly value: VocabularyValue;
  /** What the option is labelled by: the value printed. */
  readonly label: string;
}

/**
 * One alternative of a union, as a row offers it: the **source mode** of plan §4.12.
 *
 * `{"literal": 32}`, `{"quantity": "d"}`, `{"index": "layer"}`, an `op` tree: the four modes the
 * argument sheet draws are the alternatives of `scalar_expression`, labelled by the key each
 * requires. A mode whose alternative declares one scalar member is edited **on the row itself**
 * ({@link inline}); one that is shaped is edited by the widget bound to the union, or by rows one
 * level deeper when nothing is bound.
 */
export interface FormMode {
  /**
   * What labels it: a discriminating `const`, the alternative's discriminating required key, or
   * — when a presentation binding edits the alternative — that editor's name, which is how the
   * four operation forms of a `scalar_expression` become plan §4.12's one `expression` mode.
   */
  readonly tag: string;
  /**
   * The alternatives it covers: what their `$ref`s name, or where they are written.
   *
   * More than one exactly when they are the same editor's business.
   */
  readonly anchors: readonly string[];
  /** The union the alternatives belong to; not the row's own when unions nest. */
  readonly union: string;
  /** What edits a value in this mode. */
  readonly widget: Widget;
  /** Whether the mode is written as one scalar member, editable on the row itself. */
  readonly inline: boolean;
  /** That member's name, when the alternative has exactly one. */
  readonly member?: string;
  /** The kind of name the member holds, when a binding says (`quantity`, `index`, `argument`). */
  readonly referent?: string;
  /**
   * The alternatives this mode covers that no presentation binding names.
   *
   * Plan §1 (a) lists exactly those as "rendered generically", and a chooser must say the same
   * thing that list says rather than build a second one. Empty when every alternative the mode
   * covers is named — by a symbol at its union, by a binding at its own anchor, or by one at what
   * its `$ref` names. A collapsed mode can be named for some of its alternatives and not others:
   * a condition's `not`, `all` and `any` carry symbols where `compare` does not.
   */
  readonly unbound: readonly string[];
  /** The options of an inline mode whose member is an enumeration. */
  readonly options?: readonly FormOption[];
}

/** What a schema asserts about a value, as a row shows it. */
export interface FormBounds {
  readonly minimum?: number;
  readonly minimumExcluded?: boolean;
  readonly maximum?: number;
  readonly maximumExcluded?: boolean;
  readonly minLength?: number;
  readonly maxLength?: number;
  /** The pattern a name must match, as the schema writes it; compiling it is `pythonRegExp`'s. */
  readonly pattern?: string;
  readonly minItems?: number;
  readonly maxItems?: number;
  readonly uniqueItems?: boolean;
  readonly minProperties?: number;
  readonly maxProperties?: number;
}

/** What a map's own names must be: its `propertyNames`, which is where a new key is judged. */
export interface FormKeys {
  /** The definition the names are: `…#/$defs/identifier`, `…#/$defs/qualified_name`. */
  readonly anchor: string;
  /** Its pattern, as the schema writes it. */
  readonly pattern?: string;
  readonly minLength?: number;
  readonly maxLength?: number;
}

/**
 * A branch under which a member is declared or required.
 *
 * Plan §1 names the case: "`if/then` (a `required` that appears under a condition —
 * `quantity_definition` requires `domain` when the source is `external`)". The row says which
 * branch, whether the branch holds for the value in hand, and what the branch adds — so a sheet
 * can show `domain` as required exactly when it is, and never refuse the edit either way (Q5).
 */
export interface FormCondition {
  /** Where the branch is written: `…#/$defs/quantity_definition/then`. */
  readonly branch: string;
  /** The test it hangs on: its sibling `if`, or the branch itself when it hangs on none. */
  readonly test: string;
  /** Whether the branch applies to the value at the container's place; `null` with no value. */
  readonly holds: boolean | null;
  /** Whether the branch requires the member. */
  readonly requires: boolean;
  /** Whether the branch declares the member at all. */
  readonly declares: boolean;
}

/** One row of a generated form: a place of the value, and everything a sheet shows about it. */
export interface FormRow {
  /** Where in the value, as a JSON pointer (RFC 6901); `''` at the root of the form. */
  readonly path: string;
  /** The same place as steps, which is what a store command takes. */
  readonly steps: readonly PathSegment[];
  /** How far it is indented: the length of the path (D7 — "indent by depth computed from the path"). */
  readonly depth: number;
  /** The label: the schema's `title`, failing that the member name, the map key or the index. */
  readonly label: string;
  /** What edits the value. */
  readonly widget: Widget;
  /** The subschemas that describe the place, most specific first — what a binding is looked up by. */
  readonly anchors: readonly string[];
  /** Whether the container's own schema requires this member. */
  readonly required: boolean;
  /** The branches that declare or require it, when it is declared under a condition. */
  readonly conditions?: readonly FormCondition[];
  /** Whether the value writes it. */
  readonly present: boolean;
  /** The help text: the schema's `description`, most specific first (plan §1). */
  readonly description?: string;
  /** A select's options: the schema's `enum`, in its own order. */
  readonly options?: readonly FormOption[];
  /** The one value a `const` admits: a read-only row. */
  readonly constant?: VocabularyValue;
  /** The bounds the row shows. */
  readonly bounds?: FormBounds;
  /** What the names of a map must be, when the place is one. */
  readonly keys?: FormKeys;
  /** The source modes a union offers, in the schema's order. */
  readonly modes?: readonly FormMode[];
  /** The tag of the mode the value is in; absent when nothing is written or nothing matches. */
  readonly mode?: string;
  /** The value of an inline mode, or of a scalar row, as the tree writes it. */
  readonly written?: VocabularyValue;
  /** The list a select is filled from when the schema enumerates none (`presentation.json`). */
  readonly picker?: string;
  /** The label of the "New …" action beside that picker. */
  readonly create?: string;
  /** The kind of name this row holds, when the value is in a mode that refers to one. */
  readonly referent?: string;
  /** How a figure at this place is shown: a size, a count, a status, a shape. */
  readonly format?: string;
  /** The message of the first problem whose pointer is this row's; the rest go to Problems. */
  readonly error?: string;
  /** Why the row renders generically, when it does; `undefined` when the walker had an answer. */
  readonly generic?: string;
}

/** What a form could not render from the schema alone, for the log (plan §1, "listed"). */
export interface FormNote {
  /** Which kind of gap it is: `undiscriminated`, `untagged`, `unreadable`. */
  readonly code: string;
  /** The row it is about, as a JSON pointer. */
  readonly path: string;
  /** Where in the schema. */
  readonly anchor: string;
  /** What the walker could not do, without repeating the anchor. */
  readonly message: string;
}

/** A generated form: its rows, flat and in order, and what it had no reading for. */
export interface Form {
  readonly rows: readonly FormRow[];
  readonly notes: readonly FormNote[];
}

/** One problem to put on a row: the core emits the pointer beside the message (plan §3). */
export interface FormProblem {
  /** The place, as a JSON pointer relative to the value the form was given. */
  readonly path: string;
  readonly message: string;
}
