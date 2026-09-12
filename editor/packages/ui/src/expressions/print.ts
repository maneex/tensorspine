/**
 * The text form of an expression and of a condition — plan §4.13, artboard S7.
 *
 * > `d div heads`
 * > `$layer mod 5 = 4 and $layer >= 4`
 * > `heads mod kv_heads = 0`
 *
 * The folded canvas of §4.7 needs it in four places, which is why it was written at feature 2.9
 * and not at 2.11: a **guard badge** prints the site's `when`, a **composition header** prints
 * `layer ∈ [0, 32)`, a **structural summary** prints `width=d`, and a **boundary handle** prints
 * `attn_n[layer=0].input`. Feature 2.11 added the other half — the strict parser — and with it the
 * one table both halves read (`language.ts`): an operator's symbol, its form, its precedence and a
 * name's prefix are `presentation.json`'s, derived once against the schemas, and the printer and
 * the parser are two renderings of that one derivation.
 *
 * **Nothing here knows an operator.** The walk is over the grammar: which production a value is
 * written by is `Grammar.chosen` (the schema's own verdict through `registry.accepts`, feature
 * 2.3's rule), and what it prints as is the symbol that production carries. An operator the file
 * binds nothing for prints as `name(args)`, which is §4.13's own fallback and is why `min` and
 * `max` carry no binding. Catching rule §1 (b) is what makes that the only possible
 * implementation: this file may not write `floor_divide`.
 *
 * **Parentheses come from the precedence the symbols carry and from the parser's own reading.**
 * An operand that binds more loosely is parenthesised, which is the ordinary rule; an operand of
 * *equal* precedence is parenthesised where it stands to the right, because the parser reads a
 * chain left to right; and the first operand of an n-ary operator is parenthesised when it is an
 * application of that same operator, because the parser would otherwise flatten the two into one
 * — `a * (b * c)`, `(a * b) * c` and `a * b * c` are three documents the reference base writes
 * between them, and the text form has to keep them apart.
 *
 * **A space where two texts would run together.** `not true` needs one and `-d` does not, and the
 * difference is not typographic: `nottrue` is one word and `-1` is one number. The lexer is asked
 * ({@link runsTogether}), so the rule is right for whatever symbols the file carries.
 */
import { isJsonNumber, isJsonObject, pyStr, toPython, type JsonValue } from '@tensorspine/lang';
import type { Shape } from '@tensorspine/store';

import {
  grammarAt,
  APPLICATION,
  FUNCTION,
  INFIX,
  KEYWORD,
  LOOSEST,
  PAYLOAD,
  PREFIX,
  unionAnchorOf,
  type LanguageContext,
  type Production,
} from './language.js';
import { quoted, runsTogether } from './lex.js';
import { callName } from './parse.js';

/** What a printing needs: the schemas it walks, the shapes it steps through, the bindings. */
export type PrintContext = LanguageContext;

/** The text of a value at a place of a schema, and how tightly the result binds. */
interface Printed {
  readonly text: string;
  /**
   * The precedence of the outermost infix application, or `null` where the text is atomic — a
   * name, a number, a function call, a parenthesised group. `null` never needs parentheses.
   */
  readonly binds: number | null;
  /** The production that wrote it, which is what says whether a list would absorb it. */
  readonly production: Production | null;
  /** How many operands that production was given. */
  readonly parts: number;
}

/** What is printed for a value the grammar has no reading of at all. */
export const UNPRINTABLE = '…';

/** The text form of the value at a place of a schema. */
export function printValue(context: PrintContext, shape: Shape, value: JsonValue): string {
  return printAt(context, unionAnchorOf(context, shape), value);
}

/** The text form of a value at an anchor — the entry point a badge, a row and an editor use. */
export function printAt(context: PrintContext, anchor: string, value: JsonValue): string {
  return print(context, anchor, value).text;
}

/** The text of one value, with the precedence of what it printed. */
function print(context: PrintContext, anchor: string, value: JsonValue): Printed {
  // A scalar standing where an expression is expected is not a literal *of* one — every
  // alternative of both languages is an object — so it is some other place a caller prints
  // through here (an interface's kind, a name, an enum value) and it is written as it stands.
  if (!isJsonObject(value)) return atom(scalar(value, false));
  const grammar = grammarAt(context, anchor);
  const chosen = grammar.chosen(value);
  if (chosen === null) return atom(UNPRINTABLE);
  const { production, parts } = chosen;

  if (production.kind === PAYLOAD) {
    const payload = production.payload;
    const text = `${payload?.prefix ?? ''}${scalar(parts[0] ?? null, payload?.named !== true)}`;
    return atom(text);
  }

  if (production.kind === KEYWORD) {
    const written = production.operands.map((operand, at) => {
      const inner = print(context, operand.anchor, parts[at] ?? null);
      return `${operand.keyword ?? ''} ${inner.text}`;
    });
    return { text: written.join(' '), binds: LOOSEST, production, parts: parts.length };
  }

  if (production.kind === APPLICATION && production.payload !== undefined) {
    // A tagged production whose one member holds a name rather than an operand: `present(path)`.
    const name = scalar(parts[0] ?? null, !production.payload.named);
    return atom(`${production.symbol?.text ?? production.label}(${name})`);
  }

  const operandAnchor = (at: number): string =>
    production.list?.anchor ?? production.operands[at]?.anchor ?? anchor;
  const operands = parts.map((part, at) => print(context, operandAnchor(at), part));
  return applied(grammar.marks, production, operands);
}

/** A text that binds nothing: a name, a literal, a call, a parenthesised group. */
function atom(text: string): Printed {
  return { text, binds: null, production: null, parts: 0 };
}

/**
 * A scalar as the text form writes it.
 *
 * A *name* is bare; a literal that is text is **quoted**, which §4.13 asks for ("a literal is a
 * number, `true`, `false` or a quoted string") and the reference base's own data insists on:
 * `causal` is a literal in `mask = causal` and an argument in `causal = true`, and one of the two
 * readings would be lost otherwise. A number is written by its own lexeme (feature 0.3), so that
 * `1e-05` stays `1e-05`, and by Python's repr where the editor made it (D12).
 */
function scalar(value: JsonValue, quote: boolean): string {
  if (typeof value === 'string') return quote ? quoted(value) : value;
  if (value === true || value === false) return String(value);
  if (value === null) return UNPRINTABLE;
  if (isJsonNumber(value)) return value.lexeme ?? pyStr(toPython(value));
  return UNPRINTABLE;
}

/** One application, printed in the form its symbol names. */
function applied(
  marks: readonly string[],
  production: Production,
  operands: readonly Printed[],
): Printed {
  const symbol = production.symbol;
  const listed = operands.map((one) => one.text).join(', ');
  const name = callName(production) ?? production.label;
  // §4.13: "an operator without a symbol renders and parses as `name(args)`".
  if (symbol === undefined) return atom(`${name}(${listed})`);
  if (symbol.form === PREFIX) {
    const only = operands[0];
    if (only === undefined) return atom(symbol.text);
    const operand = wrap(only, production, 0, Infinity);
    const gap = runsTogether(symbol.text, operand, marks) ? ' ' : '';
    return atom(`${symbol.text}${gap}${operand}`);
  }
  if (symbol.form === INFIX && operands.length >= 2) {
    const binds = symbol.precedence ?? null;
    const floor = binds ?? Infinity;
    const text = operands.map((one, at) => wrap(one, production, at, floor)).join(` ${symbol.text} `);
    return { text, binds, production, parts: operands.length };
  }
  // A function form — and an infix symbol with fewer operands than it can stand between, which
  // the grammar admits (`all` takes one) and no repository file writes. It keeps its production,
  // because a *call* of the operator that would absorb it still has to be parenthesised where a
  // chain of that operator begins: `and(true) and x` would otherwise read as two clauses.
  if (symbol.form === FUNCTION || symbol.form === INFIX) {
    return { text: `${symbol.text}(${listed})`, binds: null, production, parts: operands.length };
  }
  // A form the interface has no rendering for: §1's "unknown constructs get the generic widget",
  // which here is the same fallback an operator with no symbol at all gets.
  return atom(`${name}(${listed})`);
}

/**
 * An operand, parenthesised exactly where the parser would read the text differently without it.
 *
 * Three reasons, and no fourth: it binds more loosely; it binds equally and stands to the right
 * of a chain the parser reads left to right; or it is an application of the very operator that
 * would absorb it into one list.
 */
function wrap(operand: Printed, production: Production, at: number, floor: number): string {
  const parenthesised = `(${operand.text})`;
  // The operand a chain of this very operator would absorb into itself, whatever it is written as:
  // a nested application of it, or a *call* of it with too few operands to stand between.
  const list = production.list;
  const absorbed =
    at === 0 && list !== undefined && list.maximum > 2 && operand.production === production;
  if (operand.binds === null) return absorbed ? parenthesised : operand.text;
  if (operand.binds < floor) return parenthesised;
  if (operand.binds > floor) return operand.text;
  return at > 0 || absorbed ? parenthesised : operand.text;
}
