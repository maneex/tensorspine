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
  type Path,
  type EditContext,
} from '@tensorspine/store';

import { blankOf, type FormMode, type FormOption } from '../forms/index.js';

export { blankOf };
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
  return writeValue(context, row, jsonObject([{ name: member, value }]));
}

/**
 * Write a whole value at a row's place — what §4.13's expression editor answers.
 *
 * The editor of a shaped mode hands back the tagged JSON itself (`{"op": …, "args": […]}`), not a
 * scalar to be wrapped in a member, so it writes through the same two cases {@link writeMode}
 * does and shares them: **set** where the document already writes the row, so the map keeps its
 * order and an unedited save writes the bytes that were read (D12); **appended** where it does
 * not, which is §5.5's rule for what a gesture adds.
 */
export function writeValue(context: EditContext, row: ArgumentRow, value: JsonValue): Command {
  if (nodeAt(context.tree, row.at) !== undefined) {
    return setValue(context, { path: row.at, value, label: `Set ${row.path}` });
  }
  const name = lastOf(row.at);
  return setMemberAt(context, {
    path: parentOf(row.at),
    name: typeof name === 'string' ? name : String(name),
    value,
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
 * A number as the **author typed it**, in the form they typed it in.
 *
 * V3's own lexical rule, which finding F7's documentation set states: *a number token with a
 * fraction or an exponent is a real, without one a whole number*. So a number a person writes
 * carries the float-ness of what was written, and the declared type is what **judges** it rather
 * than what rewrites it.
 *
 * **This is a departure from D12's parenthesis** ("the float-ness taken from the declared type: a
 * `real` argument is a float") and it is the corpus that asks for it: `llama3-8b` writes
 * `"theta": 500000` for an argument the reference base declares `real`, `model.py` reads it as a
 * Python `int`, D1 carries the `int`, and the Weisfeiler-Lehman signature
 * `tests/signatures/llama3-8b.json` records is taken over `json.dumps` of those arguments — so a
 * writer that took the float-ness from the declaration could not spell that document at all, and
 * feature 2.19 could not build it. D12's rule stands everywhere there is no text to read: a value
 * the editor *computes* (`Pin value`, a blank, a mode's first option) is still written in the form
 * the declaration asks for, which is what {@link literalOf} does.
 *
 * `null` where the text is not a finite number — an empty field, a half-typed `1e`, an infinity —
 * which is a field being typed in and not a value to write.
 */
export function numberAsTyped(text: string, facts: SchemaFacts): JsonValue | null {
  const value = Number(text);
  if (text.trim() === '' || !Number.isFinite(value)) return null;
  // A fraction or an exponent makes it a real; nothing else does, whatever the place admits.
  return literalOf(value, { ...facts, holdsWholeNumber: !/[.eE]/.test(text), holdsNumber: /[.eE]/.test(text) });
}

/**
 * Write a typed number at a row, in the form it was typed in.
 *
 * The caller asks {@link numberAsTyped} first, which is what says whether there is a number to
 * write at all — a field being typed in is not a value.
 */
export function writeTypedNumber(
  context: EditContext,
  row: ArgumentRow,
  text: string,
  facts: SchemaFacts,
): Command {
  const value = numberAsTyped(text, facts);
  if (value === null) throw new Error(`${row.path}: ${JSON.stringify(text)} is not a number`);
  const mode = literalMode(row.modes);
  if (mode === undefined) throw new Error(`${row.path} has no literal mode`);
  return writeMode(context, row, mode, value);
}

/**
 * A scalar as the document writes it.
 *
 * A number carries its lexeme and its integer/real nature (feature 0.3, D12), and the nature is the
 * declared type's: `holdsWholeNumber` without `holdsNumber` is `--document primitive-schema`'s
 * `{"type": "integer"}`, which a cardinality and a whole-number physical both are. Where a person
 * *typed* the number, {@link numberAsTyped} is what reads it instead.
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

/**
 * Write a value at a place the document may not yet have a member for (§5.5's "appended").
 *
 * Every widget that owns a whole subtree writes through it — the rows of a generated form, the
 * expression editor, the physical-name tokens — because the two cases a place can be in are the
 * store's two commands and neither caller should have to know which it is looking at.
 */
export function writeMember(
  context: EditContext,
  at: Path,
  value: JsonValue,
  label: string,
): Command {
  if (nodeAt(context.tree, at) !== undefined) return setValue(context, { path: at, value, label });
  const name = at[at.length - 1];
  return setMemberAt(context, {
    path: at.slice(0, -1),
    name: typeof name === 'string' ? name : String(name ?? ''),
    value,
    label,
  });
}
