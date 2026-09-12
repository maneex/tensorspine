/**
 * One value of a derived product, written as a table shows it — plan §4.18's renderings.
 *
 * > … with presentation bindings for `bytes` (B/KiB/MiB/GiB with the exact value in the tooltip),
 * > `elements`, `count` (`{tokens: 1.0}` as `1 per tokens`), `status`, `shape`
 * > (`[model.vocabulary=128256, model.width=4096]`), and identifiers as links.
 *
 * Three of those six renderings are feature 2.6's and are read from there, not written again:
 * `bytes`, `elements` and `operations` are `view.py`'s own `fmt_bytes` and `fmt_ops`, and the
 * plan's finding F6 moves them into the editor **once** — "there must not be a second rendering
 * of a byte count". The three this feature adds are the ones 2.6 left it, with the reason it left
 * them: they are not renderings of a *number*.
 *
 *  - **`shape`** is the core's own `shapeText`, which a sheet's `[attention.heads=4096,
 *    model.width=4096]` and an edge's value type already read (`derive/labels.ts`). One rendering,
 *    three readers.
 *  - **`count`** is §4.18's own wording, `1 per tokens`. It is *not* the core's `streamAxis`, and
 *    the difference is stated rather than reconciled: `streamAxis` writes a value's **leading
 *    axis** — `tokens`, `audio/8`, `tokens + pixels/4` — which is the geometry of an array and
 *    what `--view` prints on an edge (F6); a `count` **cell** is the multiplier itself, which is
 *    what §4.18 asks a column to say. Two readings the plan states separately, of one member.
 *    The number is written as `view.py`'s own JavaScript writes one (the review repair
 *    `2d5d031`): a count of `2.0` reads `2`, never `2.0`.
 *  - **`status`** is the chip an epistemic status is drawn as, and a `qualified_value` — "a value
 *    with its status chip" — is found structurally: an object with one number the schema declares
 *    as such and one member this binding marks. No member of the derived schema is named here.
 *
 * **A composite with no binding is written from its own members**, which is §1's "unknown
 * constructs get the generic widget" applied to a figure: a location reads
 * `model.layers.3.self_attn.q_proj.weight`, an endpoint `decoder/attn[layer=0] input`, a payload
 * `k bf16 … + v bf16 …`. Never a blank, and never a number this module made up.
 */
import { pyStr, shapeText, type PyRecord, type PyValue } from '@tensorspine/lang';
import type { Shape } from '@tensorspine/store';

import { figureOf } from '../documents/figures.js';
import type { Binding, Presentation } from '../presentation/index.js';
import { factsOfShape, type FormContext } from '../forms/index.js';

/** What nothing at all is written as — `view.py`'s own em dash, as every other figure writes it. */
export const NOTHING = '—';

/** One cell of a table, a field of a totals strip, or the header's own value. */
export interface DerivedCell {
  /** What is shown. */
  readonly text: string;
  /** The exact figure behind a rounded one — §4.18's "the exact value in the tooltip". */
  readonly exact?: string;
  /** The epistemic status of a qualified value, as a chip beside it. */
  readonly status?: string;
  /** What the identifier in this cell names, where it is one: the binding's own word. */
  readonly names?: string;
  /** The identifier itself, which a link acts on. */
  readonly name?: string;
  /** Whether the cell holds a figure, so a table right-aligns it. */
  readonly figure?: true;
}

/** Whether a value is a record — an object and not an array. */
export function isRecord(value: PyValue): value is PyRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * Whether a value is a list.
 *
 * `Array.isArray` narrows a union to `any[]`, which loses the element type the core states; this
 * keeps it, so nothing below has to be asserted back.
 */
export function isList(value: PyValue): value is readonly PyValue[] {
  return Array.isArray(value);
}

/**
 * A record's members, in the order it writes them.
 *
 * Not `Object.entries`, which widens the core's `UNRESOLVED` sentinel — a *unique* symbol — to
 * `symbol` and so answers a type a `PyValue` is not. Feature 1.2's own decision ("`UNRESOLVED` is
 * a symbol **and** a member of `PyValue`") reaches the interface here, one reading along.
 */
export function membersOf(value: PyRecord): [string, PyValue][] {
  return Object.keys(value).map((name) => [name, value[name] as PyValue]);
}

/** Whether a value is a number of either of the two the core keeps apart (feature 1.2). */
function isNumber(value: PyValue): value is number | bigint {
  return typeof value === 'number' || typeof value === 'bigint';
}

/**
 * The renderings of the formats that are not a number.
 *
 * Bare property keys and no string literal, for the reason feature 2.6 gave the numeric three:
 * `shape` and `count` are the *editor's* own format vocabulary, and a whole-literal scan cannot
 * tell it from the language's. A test holds this table's key set, beside `renderedFormats()`, to
 * the `format` enumeration of `editor/schemas/tensorspine-editor-presentation.schema.json`.
 */
const RENDERINGS: Readonly<Record<string, (value: PyValue) => string>> = {
  shape: (value) => shapeText(value),
  count: (value) => countText(value),
};

/** The format whose value is drawn as a chip rather than written in the cell. */
const STATUS = 'status';

/** Every format this module renders, for the test that closes feature 2.6's own open point. */
export function valueFormats(): string[] {
  return [...Object.keys(RENDERINGS), STATUS].sort();
}

/**
 * `{"tokens": 1.0}` as `1 per tokens` — §4.18's own wording for a stream count.
 *
 * A count with several entries writes them all, in the order the product wrote them; a count the
 * product left open writes nothing at all rather than a zero.
 */
export function countText(value: PyValue): string {
  if (!isRecord(value)) return NOTHING;
  const parts = membersOf(value).map(
    ([stream, one]) => `${isNumber(one) ? String(Number(one)) : pyStr(one)} per ${stream}`,
  );
  return parts.length === 0 ? NOTHING : parts.join(', ');
}

/** The binding at a place, most specific first. */
function bindingAt(shape: Shape, bindings: Presentation): Binding | undefined {
  return bindings.firstOf(shape.all.map((place) => place.anchor));
}

/**
 * A qualified value read structurally: the number the schema declares, and the status beside it.
 *
 * `null` where the object is not one — which includes D5's `parameters` and `state`, whose
 * `status` stands beside *two* numbers and which are therefore a strip of three fields rather than
 * one figure with a chip. The rule is the same one feature 2.6's status bar reads, so the two
 * cannot disagree about what a qualified value is.
 */
export function figureIn(
  value: PyValue,
  shape: Shape,
  context: FormContext,
): { value: PyValue; at: string; status: string } | null {
  if (!isRecord(value)) return null;
  let figure: PyValue | undefined;
  let at: string | undefined;
  let status: string | undefined;
  let numbers = 0;
  for (const [member, one] of membersOf(value)) {
    if (isList(one) || isRecord(one)) return null;
    const place = context.shapes.member(shape, member);
    const facts = factsOfShape(place);
    if (facts.holdsNumber || facts.holdsWholeNumber) {
      numbers += 1;
      figure = one;
      at = member;
      continue;
    }
    if (typeof one === 'string' && bindingAt(place, context.bindings)?.format === STATUS) {
      status = one;
    }
  }
  if (numbers !== 1 || status === undefined || at === undefined) return null;
  return { value: figure ?? null, at, status };
}

/**
 * One value, written as the panel shows it.
 *
 * A qualified value is the figure and its chip: the number is rendered under the binding of the
 * *member* it stands at, failing that under the qualified value's own place — which is where
 * `operations / element` carries the `operations` format the status bar reads (feature 2.6).
 */
export function cellOf(value: PyValue, shape: Shape, context: FormContext): DerivedCell {
  const qualified = figureIn(value, shape, context);
  if (qualified === null) return plainCell(value, shape, context, 0);
  const inside = context.shapes.member(shape, qualified.at);
  const binding = bindingAt(inside, context.bindings) ?? bindingAt(shape, context.bindings);
  if (!isNumber(qualified.value)) return { text: NOTHING, figure: true, status: qualified.status };
  const figure = figureOf(Number(qualified.value), binding);
  return { text: figure.text, exact: figure.exact, figure: true, status: qualified.status };
}

/** How deep a composite cell is written out before it is left as its own JSON. */
const DEPTH = 4;

/** Any other value: a scalar under its binding, or a composite written from its own members. */
function plainCell(value: PyValue, shape: Shape, context: FormContext, depth: number): DerivedCell {
  const binding = bindingAt(shape, context.bindings);
  const render = binding?.format === undefined ? undefined : RENDERINGS[binding.format];
  if (render !== undefined) return { text: render(value) };
  if (value === null || value === undefined) return { text: NOTHING };
  if (isNumber(value)) {
    const figure = figureOf(Number(value), binding);
    return { text: figure.text, exact: figure.exact, figure: true };
  }
  // `value === true || value === false` and not `typeof value === 'boolean'`: `boolean` is a
  // value of `argument_type.kind`, and catching rule §1 (b) is a whole-literal scan that cannot
  // tell a JavaScript type test from a vocabulary item. Features 2.1, 2.5 and 2.11 met the same
  // trap and answered it the same way.
  if (value === true || value === false) return { text: value ? YES : NO };
  if (typeof value === 'string') {
    return binding?.names === undefined
      ? { text: value }
      : { text: value, name: value, names: binding.names };
  }
  return { text: compositeText(value, shape, context, depth) };
}

/** What a truth is written as. Not the schema's vocabulary: a truth has no spelling there. */
const YES = 'yes';
const NO = 'no';

/**
 * A composite, written from its own members.
 *
 * An array writes its items separated by ` + ` — S11's own reading of a state's payload, `k … +
 * v …` — and an object writes its members' values separated by a space, the column's header
 * having already said what they are. Each figure inside is written under the binding at *its* own
 * place, so a payload's bytes read `2.0 KiB` and not the board's `2.00 KiB` (feature 2.9's
 * finding, one artboard along). Past {@link DEPTH} it is written as JSON, which is §1's generic
 * widget and never a blank.
 */
function compositeText(value: PyValue, shape: Shape, context: FormContext, depth: number): string {
  if (depth >= DEPTH) return JSON.stringify(value, (_, one: unknown) =>
    typeof one === 'bigint' ? Number(one) : one,
  );
  if (isList(value)) {
    if (value.length === 0) return NOTHING;
    return value
      .map((one, index) => plainCell(one, context.shapes.item(shape, index), context, depth + 1).text)
      .join(' + ');
  }
  if (!isRecord(value)) return NOTHING;
  const parts = membersOf(value).map(
    ([member, one]) => plainCell(one, context.shapes.member(shape, member), context, depth + 1).text,
  );
  return parts.length === 0 ? NOTHING : parts.join(' ');
}
