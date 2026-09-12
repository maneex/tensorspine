/**
 * What the sheet's rows write — plan §4.4's "every edit has a visible affordance", §4.12's four
 * source modes, its "defaults are not written" and its `Pin value`.
 *
 * Every command here is one of the store's (feature 2.1): `setValue`, `setMemberAt`, `remove`,
 * `insertItem`. What this module adds is the **value** each gesture writes, and it builds every one
 * of them out of what the schemas and the core already said:
 *
 * - the member a mode is written under is that mode's own (`FormMode.member`, the `oneOf`'s
 *   discriminating key) — so the word `literal` is nowhere in the interface, which catching rule
 *   §1 (b) requires of it and would catch;
 * - whether a number is written as a whole one or with a fraction is what the *artifact's* schema
 *   asserts (D12: "the float-ness taken from the declared type"), never a guess about the value;
 * - a blank for a mode is the first thing that place admits — the first option of an enumeration,
 *   `false` for a truth, zero for a number, the empty text otherwise.
 *
 * **Nothing is refused** (Q5): a mode whose list is empty writes nothing and says so, a value the
 * grammar will refuse is written and reported by Ajv on the row, and "wire first, fix later" holds
 * here as it does on the canvas.
 */
import {
  jsonNumber,
  jsonObject,
  type JsonValue,
  type SchemaFacts,
} from '@tensorspine/lang';
import {
  lastOf,
  nodeAt,
  parentOf,
  remove,
  setMemberAt,
  setValue,
  type Command,
  type EditContext,
} from '@tensorspine/store';

import type { FormMode, FormOption } from '../forms/index.js';
import type { ArgumentRow } from './arguments.js';

/**
 * The mode that holds the value itself: inline, and referring to no name.
 *
 * §4.12's `literal`, identified by what it *is* rather than by what it is called — the alternative
 * written as one scalar member that no presentation binding marks as a reference. That is the same
 * reading `Pin value` needs ("writes the effective value as a literal") and the one a widget edits.
 */
export function literalMode(modes: readonly FormMode[]): FormMode | undefined {
  return modes.find((mode) => mode.inline && mode.referent === undefined);
}

/** A mode that holds a name of some kind: the quantity and index modes of §4.12. */
export function referringModes(modes: readonly FormMode[]): FormMode[] {
  return modes.filter((mode) => mode.inline && mode.referent !== undefined);
}

/**
 * Write a value in one mode at a row's place: `{"literal": 32}`, `{"quantity": "d"}`.
 *
 * A row the document already writes is **set** where it stands, so the map keeps its order and an
 * unedited save writes the bytes that were read (D12). A row it does not write is a member the map
 * gains, appended — §5.5's own rule for what a gesture adds ("a new argument after the last") — so
 * that a value put back on its default leaves the document it came from.
 */
export function writeMode(
  context: EditContext,
  row: ArgumentRow,
  mode: FormMode,
  value: JsonValue,
): Command {
  const member = mode.member;
  if (member === undefined) throw new Error(`the mode ${mode.tag} writes no single member`);
  const written = jsonObject([{ name: member, value }]);
  if (nodeAt(context.tree, row.at) !== undefined) {
    return setValue(context, { path: row.at, value: written, label: `Set ${row.path}` });
  }
  const name = lastOf(row.at);
  return setMemberAt(context, {
    path: parentOf(row.at),
    name: typeof name === 'string' ? name : String(name),
    value: written,
    label: `Set ${row.path}`,
  });
}

/**
 * Write an empty record at a row's place: `{"record": {}}`.
 *
 * A record argument is written before its fields can be: D5's rule is that every gesture leaves the
 * document **on the grammar** and lets the semantic stage report what is missing — here V2's
 * `required argument missing 'window.span'`, on the field's own row, which is the report and not a
 * refusal of the gesture (Q5).
 */
export function writeRecord(context: EditContext, row: ArgumentRow, mode: FormMode): Command {
  return writeMode(context, row, mode, jsonObject([]));
}

/** Write a literal at a row's place, with the number form the artifact's type asserts (D12). */
export function writeLiteral(
  context: EditContext,
  row: ArgumentRow,
  value: string | number | boolean,
  facts: SchemaFacts,
): Command {
  const mode = literalMode(row.modes);
  if (mode === undefined) throw new Error(`${row.path} has no literal mode`);
  return writeMode(context, row, mode, literalOf(value, facts));
}

/**
 * A scalar as the document writes it.
 *
 * A number carries its lexeme and its integer/real nature (feature 0.3, D12), and the nature is the
 * declared type's: `holdsWholeNumber` without `holdsNumber` is `--document primitive-schema`'s
 * `{"type": "integer"}`, which a cardinality and a whole-number physical both are.
 */
export function literalOf(value: string | number | boolean, facts: SchemaFacts): JsonValue {
  if (typeof value !== 'number') return value;
  return jsonNumber(value, !(facts.holdsWholeNumber && !facts.holdsNumber));
}

/**
 * "A row left at its default stores nothing" — §4.12, and this is how it gets back there.
 *
 * The member goes from the map; the cascade has nothing to do, an argument value naming nothing.
 */
export function clearValue(context: EditContext, row: ArgumentRow): Command {
  return remove(context, { path: row.at, label: `Clear ${row.path}` });
}

/**
 * `Pin value`: write the effective value as a literal (§4.12).
 *
 * "For a document that must read without the library's default — the user's call, never the
 * editor's", which is why it is a button and not something the sheet does when a row is touched.
 * The value is the **core's**, printed by the core, and parsed back into the literal the declared
 * type admits: a select's option where the place enumerates, a truth where it holds one, a number
 * where it holds a number, the text otherwise.
 */
export function pinValue(context: EditContext, row: ArgumentRow, facts: SchemaFacts): Command {
  const effective = row.effective;
  if (effective === undefined) throw new Error(`${row.path} has no effective value to pin`);
  return writeLiteral(context, row, scalarOf(effective, facts, row.options), facts);
}

/**
 * The blank a mode starts from: the first value the place admits.
 *
 * Not a default of the editor's — the language's defaults are the declaration's and are never
 * written (§4.12) — but the first thing the *grammar* accepts there, so that the document the
 * gesture leaves is on the schema (D5) and the core's own verdict is what judges it.
 */
export function blankOf(facts: SchemaFacts, options: readonly FormOption[] | undefined): string | number | boolean {
  const first = options?.[0]?.value;
  if (first !== undefined && first !== null) return first;
  if (facts.holdsTruth) return false;
  if (facts.holdsWholeNumber || facts.holdsNumber) return 0;
  return '';
}

/**
 * The printed value read back as the scalar the place admits.
 *
 * The core prints a value as Python does (`pyStr`), which is what every refusal and the invariant
 * block show; reading it back is the only place the sheet turns a printed figure into a value, and
 * it does so against what the schema asserts rather than by guessing at the text.
 */
function scalarOf(
  printed: string,
  facts: SchemaFacts,
  options: readonly FormOption[] | undefined,
): string | number | boolean {
  const option = options?.find((one) => String(one.value) === printed)?.value;
  if (option !== undefined && option !== null) return option;
  if (facts.holdsTruth) return printed === TRUE;
  if (facts.holdsWholeNumber || facts.holdsNumber) {
    const number = Number(printed);
    return Number.isNaN(number) ? printed : number;
  }
  return printed;
}

/** How the core prints a truth — Python's own word, which `pyStr` writes. */
const TRUE = 'True';
