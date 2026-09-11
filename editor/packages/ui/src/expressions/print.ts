/**
 * The text form of an expression and of a condition — plan §4.13, artboard S7.
 *
 * > `d div heads`
 * > `$layer mod 5 = 4 and $layer >= 4`
 * > `heads mod kv_heads = 0`
 *
 * The folded canvas of §4.7 needs it in four places, which is why it is written here and not in
 * feature 2.11: a **guard badge** prints the site's `when`, a **composition header** prints
 * `layer ∈ [0, 32)`, a **structural summary** prints `width=d`, and a **boundary handle** prints
 * `attn_n[layer=0].input`. Feature 2.11 owns the editors — the tree view, the strict parser, the
 * reference pickers — and this is the half of §4.13 both of them read, so there is one convention
 * and not two: an operator's symbol, its form and its precedence are `presentation.json`'s, and
 * the parser 2.11 writes reads the same three.
 *
 * **Nothing here knows an operator.** The walk is over the schema: which alternative a value is is
 * `registry.accepts` (the grammar's own verdict, feature 2.3's rule), which member holds the
 * operator is the member whose binding carries `symbols`, and what it prints as is that symbol.
 * An operator the file binds nothing for prints as `name(args)`, which is §4.13's own fallback and
 * is why `min` and `max` carry no binding. Catching rule §1 (b) is what makes that the only
 * possible implementation: this file may not write `floor_divide`.
 *
 * **A name is printed with the prefix its place binds**, which is how `$layer` is an index where
 * `d` is a quantity — a distinction no schema states, both being an `identifier` under the one
 * member of a one-member object.
 *
 * **Parentheses come from the precedence the symbols carry** and from nothing else. An infix
 * symbol with no precedence parenthesises every nested application, which is always correct; the
 * five levels `presentation.json` gives the language's operators are what reproduce S7's text
 * without a single pair.
 */
import {
  isJsonArray,
  isJsonNumber,
  isJsonObject,
  pyStr,
  toPython,
  type JsonValue,
  type SchemaRegistry,
} from '@tensorspine/lang';
import type { SchemaShapes, Shape } from '@tensorspine/store';

import { alternationAt, chosenOf } from '../forms/alternatives.js';
import type { Binding, Presentation, SymbolBinding } from '../presentation/index.js';

/** What a printing needs: the schemas it walks, the shapes it steps through, the bindings. */
export interface PrintContext {
  readonly registry: SchemaRegistry;
  readonly shapes: SchemaShapes;
  readonly bindings: Presentation;
}

/** The text of a value at a place of a schema, and how tightly the result binds. */
interface Printed {
  readonly text: string;
  /**
   * The precedence of the outermost infix application, or `null` where the text is atomic — a
   * name, a number, a function call, a parenthesised group. `null` never needs parentheses.
   */
  readonly binds: number | null;
}

/** What is printed for a value the walk cannot read at all. */
export const UNPRINTABLE = '…';

/** The text form of the value at a place of a schema. */
export function printValue(context: PrintContext, shape: Shape, value: JsonValue): string {
  return print(context, shape, value).text;
}

/** The text form of a value at an anchor — the entry point a badge and a header use. */
export function printAt(context: PrintContext, anchor: string, value: JsonValue): string {
  return printValue(context, context.shapes.at(anchor), value);
}

/** The binding of a place, most specific first, as every reader of `presentation.json` takes it. */
function bindingOf(bindings: Presentation, shape: Shape): Binding | undefined {
  return bindings.firstOf(shape.all.map((place) => place.anchor));
}

/** A scalar as the text form writes it: a name bare, a number by its own lexeme (feature 0.3). */
function scalar(value: JsonValue): string {
  if (typeof value === 'string') return value;
  if (value === true || value === false) return String(value);
  if (value === null) return 'null';
  if (isJsonNumber(value)) return value.lexeme ?? pyStr(toPython(value));
  return UNPRINTABLE;
}

/** The text of one value, with the precedence of what it printed. */
function print(context: PrintContext, shape: Shape, value: JsonValue): Printed {
  if (!isJsonObject(value)) return { text: scalar(value), binds: null };
  const { registry, shapes, bindings } = context;

  // Which alternative the value is, is the grammar's verdict and never a resemblance (2.3).
  const alternation = alternationAt(registry.vocabulary(), shapes, unionAnchorOf(context, shape));
  const chosen = alternation === undefined ? null : chosenOf(registry, alternation, value);
  const at = chosen === null ? shape : shapes.at(chosen.alternative.anchor);

  // A union may bind its alternatives by tag — `all` prints `and`, `not` prints `not` — which is
  // a symbol on the *union's* own binding keyed by the member name (feature 2.2).
  const union = bindingOf(bindings, shape);
  for (const member of value.members) {
    const tagged = union?.symbols?.get(member.name);
    if (tagged === undefined) continue;
    return applied(tagged, member.name, operandsOf(context, at, member.name, member.value));
  }

  // Otherwise the operator is the member whose own binding carries the symbols of an enumeration,
  // and the operands are the other members, in the schema's order.
  const members = value.members.filter((one) => shapes.propertyOrder(at).includes(one.name));
  const order = members.length > 0 ? members : value.members;
  let symbol: { name: string; binding: SymbolBinding | undefined } | null = null;
  const operands: Printed[] = [];
  for (const member of order) {
    const memberShape = shapes.member(at, member.name);
    const symbols = bindingOf(bindings, memberShape)?.symbols;
    if (symbols !== undefined && typeof member.value === 'string') {
      symbol = { name: member.value, binding: symbols.get(member.value) };
      continue;
    }
    if (isJsonArray(member.value)) {
      const item = shapes.item(memberShape, 0);
      for (const one of member.value) operands.push(print(context, item, one));
      continue;
    }
    // A member holding an object whose own members carry the operator — a `compare` — is walked
    // into rather than printed: the alternative's one member *is* the application.
    if (isJsonObject(member.value) && symbol === null && operands.length === 0 && order.length === 1) {
      const inner = print(context, memberShape, member.value);
      return inner;
    }
    operands.push(print(context, memberShape, member.value));
  }

  if (symbol !== null) return applied(symbol.binding, symbol.name, operands);

  // No operator at all: a one-member alternative is the name or the literal it holds, printed
  // with whatever prefix its place binds (`$layer`).
  const only = order[0];
  if (order.length === 1 && only !== undefined) {
    const memberShape = shapes.member(at, only.name);
    const prefix = bindingOf(bindings, memberShape)?.prefix ?? '';
    const inner = operands[0] ?? print(context, memberShape, only.value);
    return { text: `${prefix}${inner.text}`, binds: prefix === '' ? inner.binds : null };
  }

  // Several members and nothing that says how they combine: the generic form, named by the
  // alternative the grammar chose, which is what §1 asks of a construct with no binding.
  const name = chosen?.alternative.label ?? '';
  return { text: `${name}(${operands.map((one) => one.text).join(', ')})`, binds: null };
}

/** The operands of a tag-bound application: a list of them, or the single value the tag holds. */
function operandsOf(
  context: PrintContext,
  shape: Shape,
  member: string,
  value: JsonValue,
): Printed[] {
  const memberShape = context.shapes.member(shape, member);
  if (!isJsonArray(value)) return [print(context, memberShape, value)];
  const item = context.shapes.item(memberShape, 0);
  return value.map((one) => print(context, item, one));
}

/** One application, printed in the form its symbol names. */
function applied(
  symbol: SymbolBinding | undefined,
  name: string,
  operands: readonly Printed[],
): Printed {
  const listed = operands.map((one) => one.text).join(', ');
  // §4.13: "an operator without a symbol renders and parses as `name(args)`".
  if (symbol === undefined) return { text: `${name}(${listed})`, binds: null };
  if (symbol.form === PREFIX) {
    const only = operands[0];
    if (only === undefined) return { text: symbol.text, binds: null };
    // `not true` needs the space and `-d` does not: a symbol ending in a word character runs into
    // its argument, one ending in punctuation does not. A typographic rule, not the language's.
    const gap = /\w$/.test(symbol.text) ? ' ' : '';
    return { text: `${symbol.text}${gap}${wrap(only, Infinity)}`, binds: null };
  }
  if (symbol.form === INFIX && operands.length >= 2) {
    const binds = symbol.precedence ?? null;
    const floor = binds ?? Infinity;
    return { text: operands.map((one) => wrap(one, floor)).join(` ${symbol.text} `), binds };
  }
  return { text: `${symbol.text}(${listed})`, binds: null };
}

/** An operand, parenthesised where its own application binds more loosely than its parent. */
function wrap(operand: Printed, floor: number): string {
  return operand.binds !== null && operand.binds < floor ? `(${operand.text})` : operand.text;
}

/**
 * The anchor a place is read as a union by: the first of its chain that *is* one.
 *
 * A member holding an expression is reached through its own property anchor
 * (`…/comparison_condition/properties/compare/properties/left`), which names no union at all; the
 * union is one `$ref` along, at `#/$defs/scalar_expression`. Taking the first anchor of the chain
 * would leave every nested application undiscriminated, and the operator's symbol with it — found
 * by this suite's own S7 line, which printed `modulo($layer, 5)`.
 */
function unionAnchorOf(context: PrintContext, shape: Shape): string {
  const vocabulary = context.registry.vocabulary();
  for (const place of shape.all) {
    if (vocabulary.unionAt(place.anchor) !== undefined) return place.anchor;
  }
  return shape.all[0]?.anchor ?? '';
}

/** The `form` an operator printed before its argument carries. */
const PREFIX = 'prefix';

/** The `form` an operator printed between its arguments carries. */
const INFIX = 'infix';
