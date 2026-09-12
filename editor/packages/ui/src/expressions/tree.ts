/**
 * The tree view of §4.13: a structural editor over the same productions the text form reads.
 *
 * > **Tree view**: a structural editor over the model side's `scalar_expression` … and `condition`
 * > …, and over the unit side's `expression` … and `condition` (with `present`), with operator
 * > selects from the schema enums and arity from the `args` bounds; a node can be wrapped ("wrap
 * > in floor_divide by …"), unwrapped, or replaced. References are picked: quantities and indices
 * > in scope on the model side; declared argument paths … on the unit side.
 *
 * **It is the same grammar.** A row is a production and a value; an operator select's options are
 * the production's own enumeration; the arity is the `minItems`/`maxItems` the schema writes on
 * the operand list; the alternatives a node can be *replaced* by are the productions at its
 * anchor. Nothing here reads a schema or a binding of its own — `language.ts` did that once — so
 * the tree cannot come to mean something the text does not.
 *
 * **Every edit answers a whole value.** A gesture rebuilds the expression from its root and the
 * caller writes it with one named command (D13), which is what makes an edit undoable as one
 * thing and keeps this module free of the store. The rebuild goes through `Production.write`, so
 * a member the grammar puts in a certain order stays in it and a number keeps its lexeme.
 *
 * **No gesture is refused for a semantic reason** (Q5). Wrapping a name in an operator writes the
 * blank second operand the *grammar* accepts, not one the language would call sensible; V3's
 * verdict on it is the core's, and it lands in Problems.
 */
import { isJsonNumber, type JsonValue, type VocabularyValue } from '@tensorspine/lang';

import { blankOf } from '../forms/widget.js';

import {
  grammarAt,
  APPLICATION,
  KEYWORD,
  type Grammar,
  type LanguageContext,
  type Production,
} from './language.js';

/** Where a node stands inside an expression: the operand index at each level, outermost first. */
export type TreePath = readonly number[];

/** One row of the tree view — artboard S7's `.xn`. */
export interface TreeRow {
  /** Where it stands, as operand indices from the root; `[]` is the root itself. */
  readonly path: TreePath;
  /** How far it is indented. */
  readonly depth: number;
  /** The anchor its value is read at: a condition's operand may be an expression. */
  readonly anchor: string;
  /** What the row is: `payload`, `application`, `keyword`, `generic`; `''` where nothing reads it. */
  readonly kind: string;
  /** What the chip says: the operator's name, or the alternative's label. */
  readonly label: string;
  /** The value at this place, as the tree holds it. */
  readonly value: JsonValue;
  /** The operator names this row could carry instead, in the schema's order; empty where none. */
  readonly operators: readonly string[];
  /** The operator it carries now. */
  readonly operator?: string;
  /** Whether the row holds a scalar of its own rather than operands. */
  readonly payload: boolean;
  /** The scalar it holds, when it holds one. */
  readonly written?: VocabularyValue;
  /** Whether that scalar is a name the editor picks rather than a value it types. */
  readonly named: boolean;
  /** What is printed before that name, which is what tells an index from a quantity. */
  readonly prefix: string;
  /** What kind of name, when a `references` binding says: `quantity`, `index`, `argument`. */
  readonly referent?: string;
  /** The keyword that introduces it in its parent, where one does. */
  readonly keyword?: string;
  /** How many operands the row has now, and what the schema admits. */
  readonly operands: number;
  readonly least: number;
  readonly most: number;
  /** Whether the schema admits one more operand here — S7's `+ arg`. */
  readonly extensible: boolean;
  /** Whether an operand may be taken away. */
  readonly reducible: boolean;
}

/** The rows of an expression, depth first, in the order the productions write their operands. */
export function rowsOf(
  context: LanguageContext,
  anchor: string,
  value: JsonValue,
): readonly TreeRow[] {
  const rows: TreeRow[] = [];
  walk(context, anchor, value, [], undefined, rows);
  return rows;
}

function walk(
  context: LanguageContext,
  anchor: string,
  value: JsonValue,
  path: TreePath,
  keyword: string | undefined,
  into: TreeRow[],
): void {
  const grammar = grammarAt(context, anchor);
  const chosen = grammar.chosen(value);
  const production = chosen?.production;
  const parts = chosen?.parts ?? [];
  const list = production?.list;
  const least = list?.minimum ?? production?.operands.length ?? 0;
  const most = list?.maximum ?? production?.operands.length ?? 0;
  const holdsPayload = production !== undefined && production.payload !== undefined;
  const operands = holdsPayload ? 0 : parts.length;
  into.push({
    path,
    depth: path.length,
    anchor,
    kind: production?.kind ?? '',
    label: production?.label ?? '',
    value,
    operators: operatorsAt(grammar, production),
    ...(production?.operator === undefined ? {} : { operator: production.operator }),
    payload: production?.payload !== undefined,
    ...(production?.payload === undefined ? {} : scalarOf(parts[0])),
    named: production?.payload?.named === true,
    prefix: production?.payload?.prefix ?? '',
    ...(production?.payload?.referent === undefined ? {} : { referent: production.payload.referent }),
    ...(keyword === undefined ? {} : { keyword }),
    operands,
    least: holdsPayload ? 0 : least,
    most: holdsPayload ? 0 : most,
    extensible: !holdsPayload && list !== undefined && parts.length < most,
    reducible: !holdsPayload && list !== undefined && parts.length > least,
  });
  if (production === undefined || holdsPayload) return;
  parts.forEach((part, at) => {
    walk(
      context,
      operandAnchor(production, at, anchor),
      part,
      [...path, at],
      production.operands[at]?.keyword,
      into,
    );
  });
}

/**
 * The scalar a row holds, where the value is one the interface can show.
 *
 * The truth test is written `value === true || value === false` and not with the type name, for
 * the reason feature 2.1 recorded: `boolean` is a value of `argument_type.kind` and catching rule
 * §1 (b) is a whole-literal scan that cannot tell a JavaScript type test from a vocabulary item.
 */
function scalarOf(value: JsonValue | undefined): { written?: VocabularyValue } {
  if (value === undefined) return {};
  if (typeof value === 'string' || value === true || value === false || value === null) {
    return { written: value };
  }
  return isJsonNumber(value) ? { written: value.value } : {};
}

/** Where an operand of a production is read: its list's anchor, its own, or the parent's. */
export function operandAnchor(production: Production, at: number, fallback: string): string {
  return production.list?.anchor ?? production.operands[at]?.anchor ?? fallback;
}

/**
 * Which operand of a production can hold a value written at an anchor, or `null`.
 *
 * The two gestures that move a node between places need it, and D5 is why: a conditional's
 * operands are a **condition** and two expressions, and a comparison's are two expressions where
 * the whole is a condition, so "wrap this in a conditional" and "unwrap this comparison" have a
 * right answer and a wrong one. Putting a condition where an expression is written leaves the
 * document off the grammar, which is the one thing every gesture must not do.
 */
export function operandFor(production: Production, anchor: string): number | null {
  if (production.list !== undefined) return production.list.anchor === anchor ? 0 : null;
  const at = production.operands.findIndex((operand) => operand.anchor === anchor);
  return at < 0 ? null : at;
}

/** The operator names a row could carry instead of the one it does, in the schema's order. */
function operatorsAt(grammar: Grammar, production: Production | undefined): readonly string[] {
  if (production?.operator === undefined || production.kind !== APPLICATION) return [];
  const alternative = production.alternative.anchor;
  return grammar.productions
    .filter((one) => one.alternative.anchor === alternative && one.operator !== undefined)
    .map((one) => one.operator as string);
}

/** The value at a path inside an expression, or `undefined` where the path leads nowhere. */
export function nodeAt(
  context: LanguageContext,
  anchor: string,
  value: JsonValue,
  path: TreePath,
): JsonValue | undefined {
  if (path.length === 0) return value;
  const chosen = grammarAt(context, anchor).chosen(value);
  if (chosen === null) return undefined;
  const at = path[0] as number;
  const part = chosen.parts[at];
  if (part === undefined) return undefined;
  return nodeAt(context, operandAnchor(chosen.production, at, anchor), part, path.slice(1));
}

/** The whole expression with the node at a path replaced — what every gesture answers. */
export function replaceAt(
  context: LanguageContext,
  anchor: string,
  value: JsonValue,
  path: TreePath,
  replacement: JsonValue,
): JsonValue {
  if (path.length === 0) return replacement;
  const chosen = grammarAt(context, anchor).chosen(value);
  if (chosen === null) return value;
  const at = path[0] as number;
  const part = chosen.parts[at];
  if (part === undefined) return value;
  const parts = [...chosen.parts];
  parts[at] = replaceAt(
    context,
    operandAnchor(chosen.production, at, anchor),
    part,
    path.slice(1),
    replacement,
  );
  return chosen.production.write(parts);
}

/**
 * The blank value a production starts from: the first thing the *grammar* accepts.
 *
 * Not a default of the editor's — the language's defaults are the declaration's — and not a
 * proposal either: a wrapped operand has to be *something* on the grammar for the core to have a
 * verdict about it at all (D5), and the schema's own first admissible scalar is that something.
 */
export function blankFor(context: LanguageContext, anchor: string, production: Production): JsonValue {
  const payload = production.payload;
  if (payload !== undefined) {
    const facts = payload.facts;
    return production.write([
      payload.named ? UNNAMED : scalarValue(blankOf(facts, undefined), facts.holdsWholeNumber),
    ]);
  }
  const count = Math.max(production.list?.minimum ?? production.operands.length, 0);
  const operands: JsonValue[] = [];
  for (let at = 0; at < count; at += 1) {
    operands.push(blankAt(context, operandAnchor(production, at, anchor)));
  }
  return production.write(operands);
}

/**
 * The name a reference starts from, before one is picked.
 *
 * A name is the one blank the *schema* does not answer: `identifier` is a pattern and not a set,
 * and the empty string does not match it — a blank `{"quantity": ""}` would leave the document off
 * the grammar, which D5 forbids every gesture. So a placeholder that matches the pattern is
 * written and the core reports it for what it is (`[V1] quantity 'name' is not declared`), with
 * the row's own picker beside it; the editor of §4.13 prefers the first name that picker offers
 * and falls back to this where there is none.
 */
export const UNNAMED = 'name';

/** The blank value at an anchor: the first production there, which is the schema's own first. */
export function blankAt(context: LanguageContext, anchor: string): JsonValue {
  const first = grammarAt(context, anchor).productions[0];
  return first === undefined ? null : blankFor(context, anchor, first);
}

/** A scalar as the tree holds it: a number becomes a node with its float-ness from the schema. */
function scalarValue(value: string | number | boolean, whole: boolean): JsonValue {
  if (typeof value !== 'number') return value;
  return { kind: 'number', value, real: !whole && !Number.isInteger(value) };
}

/** The same value with a scalar payload replaced — a name picked, a literal typed. */
export function withPayload(
  context: LanguageContext,
  anchor: string,
  value: JsonValue,
  written: JsonValue,
): JsonValue {
  const chosen = grammarAt(context, anchor).chosen(value);
  if (chosen === null || chosen.production.payload === undefined) return value;
  return chosen.production.write([written]);
}

/**
 * The same application under another operator, keeping every operand the new arity admits.
 *
 * S7's operator select: the options are the enumeration's, and changing `floor_divide` to
 * `subtract` keeps both operands because both take two. Where the new operator takes fewer, the
 * operands past its maximum are dropped; where it takes more, blanks are added — the arity is the
 * schema's `args` bounds and never a rule of this module's.
 */
export function withOperator(
  context: LanguageContext,
  anchor: string,
  value: JsonValue,
  operator: string,
): JsonValue {
  const grammar = grammarAt(context, anchor);
  const chosen = grammar.chosen(value);
  if (chosen === null) return value;
  const wanted = grammar.productions.find(
    (one) => one.operator === operator && one.alternative.anchor === chosen.production.alternative.anchor,
  );
  if (wanted === undefined) return value;
  return withOperands(context, anchor, wanted, chosen.parts);
}

/** A production applied to as many of these operands as the schema admits, blanks for the rest. */
function withOperands(
  context: LanguageContext,
  anchor: string,
  production: Production,
  parts: readonly JsonValue[],
): JsonValue {
  const least = production.list?.minimum ?? production.operands.length;
  const most = production.list?.maximum ?? production.operands.length;
  const kept = parts.slice(0, most);
  while (kept.length < least) kept.push(blankAt(context, operandAnchor(production, kept.length, anchor)));
  return production.write(kept);
}

/**
 * A node wrapped in an operator — S7's `wrap ▸`.
 *
 * "A node can be wrapped ('wrap in floor_divide by …'), unwrapped, or replaced": the node becomes
 * the operand that is **written where it stands** and the rest are blanks, so `d` wrapped in
 * `floor_divide` is `d div 0` and a condition wrapped in a conditional is its `if` — which is on
 * the grammar and wrong in a way the core says out loud. A production with no operand at the
 * node's own anchor cannot hold it, and answers the value unchanged; {@link wrappersAt} is what
 * keeps such a one out of the chooser in the first place.
 */
export function wrapIn(
  context: LanguageContext,
  anchor: string,
  value: JsonValue,
  production: Production,
): JsonValue {
  const at = operandFor(production, anchor);
  if (at === null) return value;
  const parts: JsonValue[] = [];
  const least = production.list?.minimum ?? production.operands.length;
  for (let index = 0; index < Math.max(least, at + 1); index += 1) {
    parts.push(index === at ? value : blankAt(context, operandAnchor(production, index, anchor)));
  }
  return production.write(parts);
}

/** The productions a node can be wrapped in: those with an operand written where it stands. */
export function wrappersAt(
  context: LanguageContext,
  anchor: string,
): readonly Production[] {
  return replacementsAt(context, anchor).filter(
    (one) => isApplication(one) && operandFor(one, anchor) !== null,
  );
}

/**
 * An application replaced by one of its operands — the other half of the wrap.
 *
 * Only an operand written at the node's own anchor: a comparison's sides are expressions where the
 * comparison is a condition, and unwrapping one to its left side would leave a `{"literal": 1}`
 * where the grammar writes a condition. {@link unwrapsTo} is which operand, or none.
 */
export function unwrapTo(context: LanguageContext, anchor: string, value: JsonValue): JsonValue {
  const at = unwrapsTo(context, anchor, value);
  if (at === null) return value;
  return grammarAt(context, anchor).chosen(value)?.parts[at] ?? value;
}

/** The operand an unwrap would keep, or `null` where none of them stands where the node does. */
export function unwrapsTo(
  context: LanguageContext,
  anchor: string,
  value: JsonValue,
): number | null {
  const chosen = grammarAt(context, anchor).chosen(value);
  if (chosen === null || chosen.parts.length === 0) return null;
  const at = operandFor(chosen.production, anchor);
  return at !== null && at < chosen.parts.length ? at : null;
}

/** One more operand, where the schema admits one — S7's `+ arg`. */
export function withMoreOperands(
  context: LanguageContext,
  anchor: string,
  value: JsonValue,
): JsonValue {
  const chosen = grammarAt(context, anchor).chosen(value);
  if (chosen === null || chosen.production.list === undefined) return value;
  if (chosen.parts.length >= chosen.production.list.maximum) return value;
  const added = blankAt(context, chosen.production.list.anchor);
  return chosen.production.write([...chosen.parts, added]);
}

/** One operand fewer, where the schema admits it. */
export function withoutOperand(
  context: LanguageContext,
  anchor: string,
  value: JsonValue,
  at: number,
): JsonValue {
  const chosen = grammarAt(context, anchor).chosen(value);
  if (chosen === null || chosen.production.list === undefined) return value;
  if (chosen.parts.length <= chosen.production.list.minimum) return value;
  return chosen.production.write(chosen.parts.filter((_, index) => index !== at));
}

/**
 * What a node can be replaced by: every production at its anchor, one entry per operator.
 *
 * The chooser of §4.13's tree view. An operator the file binds no symbol for is here like any
 * other — it is the schema's enumeration that decides the list, and the symbol only decides how
 * the text form writes it.
 */
export function replacementsAt(context: LanguageContext, anchor: string): readonly Production[] {
  return grammarAt(context, anchor).productions;
}

/** What a production is called in a chooser: its operator's name, or the alternative's label. */
export function productionLabel(production: Production): string {
  return production.operator ?? production.label;
}

/** Whether a production writes several operands rather than one scalar. */
function isApplication(production: Production): boolean {
  return production.kind === APPLICATION || production.kind === KEYWORD;
}
