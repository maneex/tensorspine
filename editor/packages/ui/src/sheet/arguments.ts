/**
 * The argument property sheet — plan §4.12, artboard S6, inventory §4 "Argument row".
 *
 * > Generated from the primitive's `arguments` (the `argument_declaration` data) and enriched by
 * > the core's `describe` (applicability, effective value, domain and invariant verdicts). Rows are
 * > flat, keyed by the argument path (`rope.scaling.beta_fast`), indented by depth; declaration
 * > order is kept.
 *
 * One row per {@link ArgumentFact}, in the order the core's own walk recorded them, and **three
 * readings meet on it**, none of which this module performs:
 *
 * | What the row shows | Who answers it |
 * |---|---|
 * | applicable, required, structural, effective value, where it came from, the domain verdict, the refusals, the declaration's help | the core's `describe` — `ArgumentFact` |
 * | the source mode the value is in, and the modes it could be in | the generic walker over `argument_value`, at the place of the document the value is written at |
 * | the literal-mode widget, its bounds, its options, its unit | the tools' generated argument schema (F5), through one rule over what a schema asserts |
 *
 * The inventory's §7 is the rule this obeys: *a component that shows a fact never computes it*.
 * Nothing here evaluates a condition, applies a default, decides whether an argument applies or
 * says what a type admits.
 *
 * **What a row's edit writes** is the value's own place — `ArgumentFact.at`, which the core's walk
 * recorded — so the interface never has to know that a record argument is written
 * `{"record": {…}}`. "Defaults are not written": a row left at its default stores nothing, and
 * `Pin value` writes the effective value as a literal, which is the user's call and never the
 * editor's (§4.12).
 */
import {
  isJsonObject,
  languageAnchors,
  pyStr,
  type ArgumentDeprecation,
  type ArgumentFact,
  type ArgumentSource,
  type DomainVerdict,
  type InvariantVerdict,
  type JsonValue,
  type PyValue,
} from '@tensorspine/lang';
import { nodeAt, pointerOf, type Path, type SchemaShapes, type Shape } from '@tensorspine/store';

import {
  formOf,
  MAP,
  SECTION,
  widgetOf,
  type FormBounds,
  type FormContext,
  type FormMode,
  type FormOption,
} from '../forms/index.js';
import { printValue, type PrintContext } from '../expressions/index.js';
import {
  NO_ARGUMENT_FACTS,
  type ArgumentBound,
  type ArgumentSchema,
  type ArgumentSchemaFacts,
} from './artifact.js';

/** What the sheet needs beside the facts to build its rows. */
export interface ArgumentSheetRequest {
  /** The core's per-argument facts, in declaration order (`SiteDescription.arguments.facts`). */
  readonly facts: readonly ArgumentFact[];
  /** The invariant verdicts of the same site, for the block under the rows (§4.12). */
  readonly invariants: readonly InvariantVerdict[];
  /** The document as the store holds it — where each row's written value is read. */
  readonly tree: JsonValue;
  /** The role the document is read under, for the schema reading of a place. */
  readonly role: string;
  readonly context: FormContext;
  readonly shapes: SchemaShapes;
  /** The primitive's generated argument schema, where the build carries one (F5). */
  readonly artifact?: ArgumentSchema | undefined;
  /** Whether the inapplicable rows are shown (§4.11: hidden by default). */
  readonly showInapplicable?: boolean;
  /** Sort the required-and-unset rows first (§4.12: default off, to keep declaration order). */
  readonly problemsFirst?: boolean;
}

/** One row of the argument sheet: a place of the document, and everything shown about it. */
export interface ArgumentRow {
  /** The argument path, which is the row's identity: `rope.scaling.beta_fast`. */
  readonly path: string;
  /** The last segment of the path — what the label column shows. */
  readonly label: string;
  /** How deep the path is: §4.12's "indented by depth". */
  readonly depth: number;
  /** Where the value is written, or would be (the core's `ArgumentFact.at`). */
  readonly at: Path;
  /** The same place as an RFC 6901 pointer, which is what a row is keyed by in the DOM. */
  readonly pointer: string;
  readonly applicable: boolean;
  readonly required: boolean;
  readonly structural: boolean;
  /** The `kind` its declared type carries — what a quantity's own type must match (§4.12). */
  readonly kind: string;
  /** Whether the document writes a value here. */
  readonly written: boolean;
  readonly source: ArgumentSource;
  readonly domain: DomainVerdict;
  /** The source mode the written value is in (§4.12's four, plus `record`); absent when unwritten. */
  readonly mode?: string;
  /** Every mode the place offers, in the schema's order — the row's segmented control. */
  readonly modes: readonly FormMode[];
  /** The scalar the row edits in an inline mode: the literal, the quantity name, the index name. */
  readonly value?: JsonValue;
  /** The expression's text form, when the mode is neither inline nor a record (§4.13). */
  readonly expression?: string;
  /** The effective value as the core resolved it, printed — §4.12's `heads = 32`. */
  readonly effective?: string;
  /** What edits the literal: the artifact's rule, or the grammar's where there is no artifact. */
  readonly widget: string;
  /** Whether the widget came from the grammar alone — F5's "the artifact is not generated". */
  readonly fromGrammar: boolean;
  readonly bounds?: FormBounds;
  readonly options?: readonly FormOption[];
  /** A physical argument's unit, shown as the field's suffix (§4.12). */
  readonly unit?: string;
  /** The bounds that name another argument, with that argument's current value (§4.12). */
  readonly named: readonly NamedBound[];
  /** The applicability condition, in §4.13's text form — the tooltip of an inapplicable row. */
  readonly presentWhen?: string;
  /** The declaration's own words (plan §1: "Help text is the schema's"). */
  readonly description?: string;
  readonly valueDescriptions?: Readonly<Record<string, string>>;
  readonly deprecation?: ArgumentDeprecation;
  /** The first refusal about this argument, as the core worded it; the rest go to Problems. */
  readonly error?: string;
  /** The kind of name an inline mode's value refers to (`quantity`, `index`). */
  readonly referent?: string;
  /** Whether the row has rows under it: a record header (§4.12's section row). */
  readonly record: boolean;
}

/** A domain bound that names another argument: `≤ heads (32)` (§4.12). */
export interface NamedBound extends ArgumentBound {
  /** That argument's effective value at this site, printed; absent where it has none. */
  readonly value?: string;
}

/** The argument sheet: its rows, its invariant block, and the counts the heading shows. */
export interface ArgumentSheet {
  readonly rows: readonly ArgumentRow[];
  /** Every invariant the primitive declares, in declaration order, with its verdict. */
  readonly invariants: readonly InvariantVerdict[];
  /** How many top-level arguments the document writes — S6's `6 set`. */
  readonly set: number;
  /** How many the primitive declares at the top level — S6's `19 declared`. */
  readonly declared: number;
  /** How many rows are hidden because they are inapplicable (§4.11's "Show inapplicable"). */
  readonly hidden: number;
}

/** The rows of one site's arguments (§4.12). */
export function argumentSheet(request: ArgumentSheetRequest): ArgumentSheet {
  const effective = new Map(request.facts.map((fact) => [fact.path, fact]));
  const parents = new Set(
    request.facts.map((fact) => fact.path).filter((path) => path.includes('.')).map(parentPath),
  );
  const print: PrintContext = {
    registry: request.context.registry,
    shapes: request.shapes,
    bindings: request.context.bindings,
  };
  // The anchor the unit's own condition language is written at, named once by the core (§4.13).
  const condition = request.context.shapes.at(languageAnchors(request.context.registry).unitCondition);
  const all = request.facts.map((fact) =>
    rowOf(fact, { ...request, print, parents, effective, condition }),
  );
  const shown = all.filter((row) => row.applicable || request.showInapplicable === true);
  const ordered =
    request.problemsFirst === true
      ? [...shown].sort((one, other) => weight(one) - weight(other))
      : shown;
  return {
    rows: ordered,
    invariants: request.invariants,
    set: request.facts.filter((fact) => !fact.path.includes('.') && fact.written !== undefined)
      .length,
    declared: request.facts.filter((fact) => !fact.path.includes('.')).length,
    hidden: all.length - shown.length,
  };
}

/** "Required arguments without a value are listed first in red" — §4.12's `Sort problems first`. */
function weight(row: ArgumentRow): number {
  if (row.error !== undefined) return 0;
  if (row.required && row.applicable && row.source === 'absent') return 0;
  return 1;
}

/** The path of the record a field belongs to: `rope.scaling.kind` is `rope.scaling`'s. */
function parentPath(path: string): string {
  return path.slice(0, path.lastIndexOf('.'));
}

/** What one row is built from, beside the request. */
interface RowContext extends ArgumentSheetRequest {
  readonly print: PrintContext;
  readonly parents: ReadonlySet<string>;
  readonly effective: ReadonlyMap<string, ArgumentFact>;
  /** The unit schema's `condition`, for the `present_when` a row shows. */
  readonly condition: Shape;
}

function rowOf(fact: ArgumentFact, context: RowContext): ArgumentRow {
  const at = [...fact.at] as Path;
  const value = nodeAt(context.tree, at);
  const shape = shapeAt(context.shapes, at, context.role);
  // One row and no descent: the *facts* are the flat list §4.12 asks for, in the declaration's own
  // order, and a walker's own descent would key a record's fields by the places the grammar writes
  // them (`…/rope/record/theta`) rather than by the argument paths the sheet is keyed by.
  const form = formOf(context.context, {
    shape,
    ...(value === undefined ? {} : { value }),
    label: fact.path,
    limit: 0,
  });
  const head = form.rows[0];
  const artifact = context.artifact?.at(fact.path) ?? NO_ARGUMENT_FACTS;
  const chosen = head?.modes?.find((mode) => mode.tag === head.mode);
  const inline = chosen?.inline === true ? chosen : undefined;
  const fromGrammar = !artifact.facts.states;
  const row: Mutable<ArgumentRow> = {
    path: fact.path,
    label: fact.path.includes('.') ? fact.path.slice(fact.path.lastIndexOf('.') + 1) : fact.path,
    depth: fact.path.split('.').length - 1,
    at,
    pointer: pointerOf(at),
    applicable: fact.applicable,
    required: fact.required,
    structural: fact.structural,
    kind: fact.kind,
    written: value !== undefined,
    source: fact.source,
    domain: fact.domain,
    modes: head?.modes ?? [],
    widget: fromGrammar
      ? (inline?.widget ?? head?.widget ?? '')
      : widgetOf(artifact.facts, artifact.facts.alternation),
    fromGrammar,
    named: namedBounds(artifact.bounds, context.effective),
    record: context.parents.has(fact.path),
  };
  if (head?.mode !== undefined) row.mode = head.mode;
  if (head?.referent !== undefined) row.referent = head.referent;
  if (inline !== undefined && value !== undefined && isJsonObject(value) && inline.member !== undefined) {
    const held = value.members.find((member) => member.name === inline.member);
    if (held !== undefined) row.value = held.value;
  } else if (value !== undefined && chosen !== undefined && printable(chosen)) {
    // §4.13's text form, for the one shaped mode that *is* an expression. A record's mode holds a
    // map of further arguments, and its fields are rows of their own: printing it as an expression
    // would put the whole subtree on one line and say nothing the rows below do not.
    row.expression = printValue(context.print, shape, value);
  }
  // The artifact's options where it has them, the grammar's otherwise: a `set` domain on a
  // cardinality is an `enum` in the artifact, which is exactly the list the row must offer.
  const options = optionsOf(artifact.facts.choices) ?? inline?.options;
  if (options !== undefined) row.options = options;
  const bounds = boundsOf(artifact);
  if (bounds !== undefined) row.bounds = bounds;
  if (artifact.unit !== undefined) row.unit = artifact.unit;
  // The unit's own condition language, printed at the place the *core* names for it: an anchor
  // written in a component would be an item of information the schema states (§1).
  if (artifact.presentWhen !== undefined) {
    row.presentWhen = printValue(context.print, context.condition, artifact.presentWhen);
  }
  // §4.12 prints the effective value "beside the row in muted type": `heads = 32`. A *container*
  // has no such value — a record's fields each carry their own, on their own rows — and printing
  // Python's dict repr there would be a figure nobody can read against the rows below it.
  if (fact.value !== undefined && !isContainer(fact.value)) row.effective = printed(fact.value);
  if (fact.description !== undefined) row.description = fact.description;
  if (fact.valueDescriptions !== undefined) row.valueDescriptions = fact.valueDescriptions;
  if (fact.deprecation !== undefined) row.deprecation = fact.deprecation;
  const problem = fact.problems[0];
  if (problem !== undefined) row.error = problem.message;
  return row;
}

/** Whether a resolved value holds further values rather than being one. */
function isContainer(value: PyValue): boolean {
  return Array.isArray(value) || (value !== null && typeof value === 'object');
}

/** Whether a shaped mode is an expression rather than a container of further rows. */
function printable(mode: FormMode): boolean {
  return !mode.inline && mode.widget !== MAP && mode.widget !== SECTION;
}

/** A mutable row while it is being filled; the rows themselves are read-only. */
type Mutable<T> = { -readonly [K in keyof T]: T[K] };

/** The shape of a place of the document, stepped down from the root (the canvas's own reading). */
function shapeAt(shapes: SchemaShapes, path: Path, role: string): Shape {
  let shape = shapes.root(role);
  for (const step of path) shape = shapes.step(shape, step);
  return shape;
}

/** The bounds a row shows, taken from the artifact's own assertions. */
function boundsOf(artifact: ArgumentSchemaFacts): FormBounds | undefined {
  const facts = artifact.facts;
  if (facts.minimum === null && facts.maximum === null) return undefined;
  return {
    ...(facts.minimum === null ? {} : { minimum: facts.minimum, minimumExcluded: facts.minimumExcluded }),
    ...(facts.maximum === null ? {} : { maximum: facts.maximum, maximumExcluded: facts.maximumExcluded }),
  };
}

/** A select's options, in the order the artifact writes them. */
function optionsOf(choices: readonly (string | number | boolean | null)[] | null): readonly FormOption[] | undefined {
  if (choices === null) return undefined;
  return choices.map((value) => ({ value, label: value === null ? '' : String(value) }));
}

/** Each argument-named bound with the value that argument resolved to at this site (§4.12). */
function namedBounds(
  bounds: readonly ArgumentBound[],
  facts: ReadonlyMap<string, ArgumentFact>,
): NamedBound[] {
  return bounds.map((bound) => {
    const named = facts.get(bound.argument)?.value;
    return { ...bound, ...(named === undefined ? {} : { value: printed(named) }) };
  });
}

/**
 * A value the core resolved, printed.
 *
 * The core's own rendering of a value (`pyStr`) is what every refusal shows and what the invariant
 * block quotes — `heads = 32, kv_heads = 3` — so the effective value beside a row reads as the
 * message about it does, which is the whole point of showing both.
 */
function printed(value: PyValue): string {
  return pyStr(value);
}
