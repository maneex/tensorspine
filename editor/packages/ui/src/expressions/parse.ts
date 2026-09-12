/**
 * The strict parser of §4.13: text → tagged JSON, and nothing else.
 *
 * > an infix rendering for reading and typing, with a parser that produces only tagged JSON …
 * > The parser is strict: `a / b` is `divide` (a real), never a floor — the language's
 * > explicit-rounding rule is the core's to enforce (V3), the editor merely shows the effective
 * > type. Round-trip is exact: text → JSON → text is the identity on the corpus's expressions and
 * > on every expression of the reference base (a test).
 *
 * **It reads the same table the printer reads.** Every symbol, every precedence, every form and
 * every prefix comes from `language.ts`, which derives them from the schemas and
 * `presentation.json`; this module holds no operator, no tag and no member name of either
 * language. A second table here is the one defect the feature could not survive, since the round
 * trip is a property of the *pair*.
 *
 * **Strict means it invents nothing.** `/` builds `divide` because that is the symbol bound to
 * `divide`; the rounding operators have their own symbols and are written by whoever wants them.
 * Nothing is coerced, nothing is rounded, and an operator the file binds no symbol for is written
 * and read as `name(args)`.
 *
 * **A refusal is a row, never a blocked keystroke** (Q5). The answer carries the tagged value
 * *or* the problems with their ranges, and the caller keeps the text either way: a half-typed
 * expression is something the author is in the middle of, and §4.17's panel is where it belongs.
 *
 * ## Three readings that depend on where a token stands, stated
 *
 * - **A word-shaped symbol is a symbol where one can stand, and a name everywhere else.** `div`,
 *   `mod`, `and` and `or` are claimed in infix position only, `abs`, `min`, `present` only before
 *   a `(`, and `not`, `if`, `then` and `else` wherever a primary begins. No name of the corpus or
 *   of the reference base is one of those words (measured: 116 distinct names, no collision), and
 *   a quantity that was would be writable in the tree view and not in the text one.
 * - **A signed number is a literal in prefix position and a subtraction after an operand.** `-1`
 *   is `{"literal": -1}` and `a -1` is `a - 1`; the printer writes `- 1` for the negation of one,
 *   which is what makes the pair exact in both directions.
 * - **The operand of a prefix symbol is a primary.** `not` carries no precedence, so the printer
 *   parenthesises every application under it and the parser reads no further than the printer
 *   writes. `not a = b` is therefore a refusal naming the `=`, not a silent regrouping.
 */
import { isNumberLexeme, jsonNumber, lexemeIsReal, type JsonValue } from '@tensorspine/lang';

import {
  grammarAt,
  APPLICATION,
  FUNCTION,
  GENERIC,
  INFIX,
  KEYWORD,
  PAYLOAD,
  PREFIX,
  type Grammar,
  type LanguageContext,
  type Production,
} from './language.js';
import { CLOSE, COMMA, lex, MARK, NUMBER, OPEN, QUOTED, splitSign, WORD, type Token } from './lex.js';

/** What the parser could not read, and where. */
export interface ExpressionProblem {
  /** What is wrong, in the editor's own words — the text is the author's, not the language's. */
  readonly message: string;
  /** The range in the text the message is about. */
  readonly from: number;
  readonly to: number;
}

/** What a parse answers: the tagged value, or what stopped it. */
export interface Parsed {
  /** The tagged JSON, absent where the text could not be read. */
  readonly value?: JsonValue;
  /** Everything that stopped it; empty on success. */
  readonly problems: readonly ExpressionProblem[];
}

/** The two spellings of JSON's boolean, which is the grammar a literal is written in. */
const TRUE = 'true';
const FALSE = 'false';

/**
 * How deeply a text may nest before the parser calls it a refusal.
 *
 * A descent is a stack frame and the field this reads is one a reader can paste into: without a
 * bound, a thousand parentheses are a `RangeError` out of a keystroke rather than a row in
 * Problems, which is the one thing Q5 does not admit. Far above anything the language writes — the
 * deepest expression of the corpus and of the reference base nests four — and the same kind of
 * backstop the form walker carries (feature 2.3's `LIMIT`).
 */
const DEPTH = 64;

/** The text of an expression or a condition, read as the value at an anchor. */
export function parseAt(context: LanguageContext, anchor: string, text: string): Parsed {
  const grammar = grammarAt(context, anchor);
  const lexed = lex(text, grammar.marks);
  if (lexed.refusal !== undefined) {
    const { message, at } = lexed.refusal;
    return { problems: [{ message, from: at, to: at + 1 }] };
  }
  const reader = new Reader(context, [...lexed.tokens], text.length);
  try {
    const node = reader.at(anchor, 0);
    const left = reader.peek();
    if (left !== undefined) {
      throw new Refusal(`'${left.text}' is more than the expression needs`, left.from, left.to);
    }
    return { value: node.value, problems: [] };
  } catch (error) {
    if (error instanceof Refusal) {
      return { problems: [{ message: error.message, from: error.from, to: error.to }] };
    }
    throw error;
  }
}

/** What stopped a parse, with the range it stopped at. */
class Refusal extends Error {
  constructor(
    message: string,
    readonly from: number,
    readonly to: number,
  ) {
    super(message);
    this.name = 'Refusal';
  }
}

/** One parsed value, with what produced it: a combination needs both. */
interface Node {
  readonly value: JsonValue;
  /** The production that wrote it, where one did. */
  readonly production: Production | null;
  /** Its operands, as that production wrote them. */
  readonly parts: readonly JsonValue[];
  /**
   * Whether a chain of the same operator may absorb it.
   *
   * Parentheses stop it, which is what keeps `a * (b * c)` and `a * b * c` apart; so does having
   * been written as a *call*, because `and(true)` names a one-element `all` that a chain would
   * otherwise swallow. Both are facts about the text and about neither value.
   */
  readonly grouped: boolean;
}

/** The recursive descent, over whatever grammar the anchors answer. */
class Reader {
  private at_ = 0;

  constructor(
    private readonly context: LanguageContext,
    private readonly tokens: Token[],
    private readonly end: number,
  ) {}

  peek(ahead = 0): Token | undefined {
    return this.tokens[this.at_ + ahead];
  }

  /** The value at an anchor, reading no operator that binds more loosely than `minimum`. */
  at(anchor: string, minimum: number, depth = 0): Node {
    if (depth > DEPTH) {
      const token = this.peek();
      throw new Refusal(
        `the expression nests deeper than ${String(DEPTH)} levels`,
        token?.from ?? this.end,
        token?.to ?? this.end,
      );
    }
    const grammar = grammarAt(this.context, anchor);
    let left = this.delegated(grammar, minimum, depth) ?? this.leading(grammar, depth);
    for (;;) {
      const found = this.infixAhead(grammar, minimum, anchor);
      if (found === null) return left;
      this.at_ += 1;
      const operand = found.production.list?.anchor ?? found.production.operands[1]?.anchor ?? anchor;
      const right = this.at(operand, found.precedence + 1, depth + 1);
      left = combine(found.production, left, right);
    }
  }

  /**
   * A production of this anchor whose operands are written at another one — a comparison, whose
   * two sides are expressions where the whole is a condition.
   *
   * It is tried first and rewound, because the token that begins it is the token that begins an
   * expression and nothing tells them apart in advance: `true` alone is a boolean condition and
   * `true = x` is a comparison of two literals. One speculative read, one rewind, no search.
   */
  private delegated(grammar: Grammar, minimum: number, depth: number): Node | null {
    const productions = grammar.productions.filter(
      (one) =>
        one.symbol?.form === INFIX &&
        one.operands.length === 2 &&
        one.operands[0]?.anchor !== grammar.anchor &&
        (one.symbol.precedence ?? 0) >= minimum,
    );
    if (productions.length === 0) return null;
    const from = this.at_;
    const left = productions[0]?.operands[0]?.anchor as string;
    try {
      const first = this.at(left, 0, depth + 1);
      const ahead = this.peek();
      const production =
        ahead === undefined || ahead.kind === QUOTED
          ? undefined
          : productions.find((one) => one.symbol?.text === ahead.text);
      if (production === undefined) {
        this.at_ = from;
        return null;
      }
      this.at_ += 1;
      const right = this.at(production.operands[1]?.anchor as string, 0, depth + 1);
      return {
        value: production.write([first.value, right.value]),
        production,
        parts: [first.value, right.value],
        grouped: false,
      };
    } catch (error) {
      if (!(error instanceof Refusal)) throw error;
      this.at_ = from;
      return null;
    }
  }

  /** The infix production the next token names, or `null`. */
  private infixAhead(
    grammar: Grammar,
    minimum: number,
    anchor: string,
  ): { production: Production; precedence: number } | null {
    let ahead = this.peek();
    if (ahead === undefined) return null;
    // `a -1` is the subtraction a reader means; the sign the lexer took is given back here, where
    // an operand already stands to the left.
    const split = splitSign(ahead);
    if (split !== null) {
      const claims = grammar.productions.some(
        (one) => one.symbol?.form === INFIX && one.symbol.text === split[0].text,
      );
      if (claims) {
        this.tokens.splice(this.at_, 1, split[0], split[1]);
        ahead = split[0];
      }
    }
    const token = ahead;
    if (token.kind === QUOTED || token.kind === NUMBER) return null;
    for (const production of grammar.productions) {
      const symbol = production.symbol;
      if (symbol === undefined || symbol.form !== INFIX || symbol.text !== token.text) continue;
      if (production.operands.length === 2 && production.operands[0]?.anchor !== anchor) continue;
      const precedence = symbol.precedence ?? 0;
      if (precedence < minimum) continue;
      return { production, precedence };
    }
    return null;
  }

  /** A primary: what begins a value at this anchor. */
  private leading(grammar: Grammar, depth: number): Node {
    const token = this.peek();
    if (token === undefined) {
      throw new Refusal('the expression stops before it says anything', this.end, this.end);
    }
    if (token.kind === MARK && token.text === OPEN) {
      this.at_ += 1;
      const inside = this.at(grammar.anchor, 0, depth + 1);
      this.expect(CLOSE);
      return { ...inside, grouped: true };
    }
    const prefixed = this.prefixed(grammar, token, depth);
    if (prefixed !== null) return prefixed;
    const keyword = this.keyword(grammar, token, depth);
    if (keyword !== null) return keyword;
    const called = this.called(grammar, token, depth);
    if (called !== null) return called;
    // A literal is read before a name: `true` is the boolean the grammar admits, not a quantity
    // that happens to be spelled like one. The three spellings this shadows — `true`, `false` and
    // a bare number — are JSON's own, and no name of the repository is one of them.
    const scalar = this.scalar(grammar, token);
    if (scalar !== null) return scalar;
    const named = this.named(grammar, token);
    if (named !== null) return named;
    throw new Refusal(`'${token.text}' begins nothing this place admits`, token.from, token.to);
  }

  /** `-d`, `not true`: a symbol before its one operand, which is a primary and no more. */
  private prefixed(grammar: Grammar, token: Token, depth: number): Node | null {
    if (token.kind === QUOTED || token.kind === NUMBER) return null;
    const production = grammar.productions.find(
      (one) => one.symbol?.form === PREFIX && one.symbol.text === token.text,
    );
    if (production === undefined) return null;
    this.at_ += 1;
    const anchor = production.list?.anchor ?? production.operands[0]?.anchor ?? grammar.anchor;
    const operand = this.leading(grammarAt(this.context, anchor), depth + 1);
    const parts = [operand.value];
    return { value: production.write(parts), production, parts, grouped: false };
  }

  /** `if … then … else …`: one keyword before each operand, in the schema's own member order. */
  private keyword(grammar: Grammar, token: Token, depth: number): Node | null {
    const production = grammar.productions.find(
      (one) => one.kind === KEYWORD && one.operands[0]?.keyword === token.text && token.kind === WORD,
    );
    if (production === undefined) return null;
    this.at_ += 1;
    const parts: JsonValue[] = [];
    production.operands.forEach((operand, at) => {
      if (at > 0) this.expect(operand.keyword as string);
      parts.push(this.at(operand.anchor, 0, depth + 1).value);
    });
    return { value: production.write(parts), production, parts, grouped: false };
  }

  /** `abs(d)`, `min(a, b)`, `present(chunk)`: a name, then the operands in parentheses. */
  private called(grammar: Grammar, token: Token, depth: number): Node | null {
    if (token.kind !== WORD || this.peek(1)?.text !== OPEN) return null;
    const production = grammar.productions.find((one) => callName(one) === token.text);
    if (production === undefined) return null;
    this.at_ += 2;
    const parts: JsonValue[] = [];
    if (this.peek()?.text !== CLOSE) {
      for (;;) {
        parts.push(this.operand(production, parts.length, depth));
        if (this.peek()?.text !== COMMA) break;
        this.at_ += 1;
      }
    }
    this.expect(CLOSE);
    this.countOperands(production, parts.length, token);
    return { value: production.write(parts), production, parts, grouped: true };
  }

  /** One operand of a call: a value at its anchor, or the bare name a payload production holds. */
  private operand(production: Production, at: number, depth: number): JsonValue {
    const payload = production.payload;
    if (payload !== undefined) {
      const token = this.peek();
      if (token === undefined || token.kind !== WORD) {
        throw new Refusal(
          `${production.label} takes a name`,
          token?.from ?? this.end,
          token?.to ?? this.end,
        );
      }
      this.at_ += 1;
      return token.text;
    }
    const anchor = production.list?.anchor ?? production.operands[at]?.anchor;
    if (anchor === undefined) {
      const token = this.peek();
      throw new Refusal(
        `${production.label} takes ${String(production.operands.length)} of them`,
        token?.from ?? this.end,
        token?.to ?? this.end,
      );
    }
    return this.at(anchor, 0, depth + 1).value;
  }

  /** A name, with the prefix its place binds: `d` is a quantity where `$layer` is an index. */
  private named(grammar: Grammar, token: Token): Node | null {
    const prefix = token.kind === MARK ? token.text : '';
    const word = prefix === '' ? token : this.peek(1);
    if (word === undefined || word.kind !== WORD) return null;
    const production = grammar.productions.find(
      (one) => one.kind === PAYLOAD && one.payload?.named === true && one.payload.prefix === prefix,
    );
    if (production === undefined) return null;
    this.at_ += prefix === '' ? 1 : 2;
    const parts = [word.text];
    return { value: production.write(parts), production, parts, grouped: false };
  }

  /** A literal: a number by its own lexeme, `true`, `false`, or a quoted text. */
  private scalar(grammar: Grammar, token: Token): Node | null {
    for (const production of grammar.productions) {
      const facts = production.kind === PAYLOAD ? production.payload?.facts : undefined;
      if (facts === undefined || production.payload?.named === true) continue;
      if (token.kind === NUMBER && (facts.holdsNumber || facts.holdsWholeNumber)) {
        if (!isNumberLexeme(token.text)) return null;
        this.at_ += 1;
        // The lexeme is kept: `1e-05` is not `0.00001` and `1.0` is not `1` (D12, feature 0.3).
        const parts: JsonValue[] = [
          { ...jsonNumber(Number(token.text), lexemeIsReal(token.text)), lexeme: token.text },
        ];
        return { value: production.write(parts), production, parts, grouped: false };
      }
      if (token.kind === QUOTED && facts.holdsText) {
        this.at_ += 1;
        const parts: JsonValue[] = [token.denotes ?? ''];
        return { value: production.write(parts), production, parts, grouped: false };
      }
      if (token.kind === WORD && facts.holdsTruth && (token.text === TRUE || token.text === FALSE)) {
        this.at_ += 1;
        const parts: JsonValue[] = [token.text === TRUE];
        return { value: production.write(parts), production, parts, grouped: false };
      }
    }
    return null;
  }

  /** What the schema admits of a call's length, reported where the author can see it. */
  private countOperands(production: Production, count: number, token: Token): void {
    const list = production.list;
    const least = production.payload !== undefined ? 1 : (list?.minimum ?? production.operands.length);
    const most = production.payload !== undefined ? 1 : (list?.maximum ?? production.operands.length);
    if (count >= least && count <= most) return;
    const wanted =
      least === most
        ? String(least)
        : most === Infinity
          ? `${String(least)} or more`
          : `${String(least)} to ${String(most)}`;
    throw new Refusal(
      `${production.label} takes ${wanted} operands, not ${String(count)}`,
      token.from,
      token.to,
    );
  }

  private expect(text: string): void {
    const token = this.peek();
    if (token === undefined) {
      throw new Refusal(`'${text}' is missing from the end of the expression`, this.end, this.end);
    }
    if (token.text !== text || token.kind === QUOTED) {
      throw new Refusal(`'${text}' was expected here`, token.from, token.to);
    }
    this.at_ += 1;
  }
}

/**
 * The name a production is written as a call by: its symbol's text, or — §4.13's own fallback,
 * "an operator without a symbol renders and parses as `name(args)`" — the operator's own name.
 *
 * A symbol whose `form` the interface has no rendering for falls to the same fallback, so that
 * the printer and the parser agree about a binding neither of them understands (§1).
 */
export function callName(production: Production): string | undefined {
  if (production.kind === GENERIC) return production.label;
  if (production.kind !== APPLICATION) return undefined;
  const symbol = production.symbol;
  if (symbol === undefined) return production.operator;
  const known = symbol.form === INFIX || symbol.form === PREFIX || symbol.form === FUNCTION;
  return known ? symbol.text : production.operator;
}

/**
 * One infix step, left to right — and the one place an n-ary operator is flattened.
 *
 * `a * b * c` is the flat `multiply(a, b, c)` the reference base writes, and `a * (b * c)` is the
 * nesting it also writes: the parentheses the author wrote are what tells them apart, and the
 * printer puts them back for exactly this reason. An operator whose list admits no third operand
 * — `subtract`, the divisions, `modulo` — nests, which is left associativity and the only reading
 * of `a - b - c` the language has.
 */
function combine(production: Production, left: Node, right: Node): Node {
  const list = production.list;
  const flattens =
    list !== undefined &&
    !left.grouped &&
    left.production === production &&
    left.parts.length + 1 <= list.maximum;
  const parts = flattens ? [...left.parts, right.value] : [left.value, right.value];
  return { value: production.write(parts), production, parts, grouped: false };
}
