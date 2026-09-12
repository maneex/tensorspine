/**
 * The tokens of §4.13's text form.
 *
 * The text form is read and written by two halves of one module — the printer (`print.ts`) and
 * the parser (`parse.ts`) — and both of them need the same answer to one question: *where does a
 * token end?* The printer needs it because a symbol and its operand must not run together
 * (`- 1` is the negation of one, `-1` is minus one, and the two are different documents), and the
 * parser needs it because that is what a parser is.
 *
 * **Nothing here knows an operator.** The symbols are handed in — they are `presentation.json`'s,
 * gathered by `language.ts` from the schema place each one is bound at — and this module only
 * says how a run of characters is cut. The four kinds it cuts into are the four §4.13 names:
 * "Identifiers are quantities (model side) or argument paths (unit side); `$name` is an index; a
 * literal is a number, `true`, `false` or a quoted string" — a word, a number, a quoted string —
 * with the grouping punctuation of `name(args)` beside them.
 *
 * **A word-shaped symbol is lexed as a word.** `div`, `mod`, `and`, `not`, `abs` all match the
 * identifier pattern of the grammar (`^[A-Za-z_][A-Za-z0-9_-]*$`), so cutting them here would
 * mean deciding, in the lexer, that `div` is never a quantity — a decision that belongs to the
 * place a token stands in, and which the parser makes with the grammar in hand.
 *
 * **A name is dotted.** The unit side's references are argument paths (`rope.scaling.kind`), the
 * model side's are plain identifiers; both are one token here and the *schema* is what refuses a
 * dot where a `qualified_name` is not admitted — a refusal that lands on the row, as Q5 asks,
 * rather than in a lexer that would call it a syntax error.
 *
 * **A sign belongs to the number it touches.** `-1` is one token and `- 1` is two, because
 * `{"literal": -1}` and the negation of `{"literal": 1}` are two different documents and the text
 * form has to keep them apart: a lexer that never signed a number would make the first unwritable
 * and a text edit would quietly turn it into the second. Where a signed number stands after a
 * complete operand — `a -1` — it is the subtraction the reader meant, and the parser splits the
 * token back ({@link splitSign}); nowhere else does the reading depend on a space. The lexeme is
 * kept exactly as written, because feature 0.3's whole point is that `1e-05` is not `0.00001`.
 */

/** A word: an identifier, a dotted argument path, or a word-shaped symbol the parser will claim. */
export const WORD = 'word';
/** A number token, without a sign; its text is the lexeme a `JsonNumber` keeps (feature 0.3). */
export const NUMBER = 'number';
/** A quoted string — §4.13's own form for a literal that is text. */
export const QUOTED = 'quoted';
/** A symbol handed in from `presentation.json`, or one of the text form's three punctuation marks. */
export const MARK = 'mark';
/** A character no rule cuts: reported where it stands, never thrown away in silence. */
export const UNKNOWN = 'unknown';

/** The three marks the text form writes itself: grouping, and the separator of `name(a, b)`. */
export const OPEN = '(';
export const CLOSE = ')';
export const COMMA = ',';

/** One token: what it is, its text, and where it stands. */
export interface Token {
  /** {@link WORD}, {@link NUMBER}, {@link QUOTED}, {@link MARK} or {@link UNKNOWN}. */
  readonly kind: string;
  /** The token's own text, exactly as written — a number's lexeme, a mark's symbol. */
  readonly text: string;
  /** What a quoted string denotes, with its escapes read; only a {@link QUOTED} token has one. */
  readonly denotes?: string;
  /** Where the token starts in the source, and where it ends — a problem's range. */
  readonly from: number;
  readonly to: number;
  /** Whether anything at all stands between this token and the one before it. */
  readonly spaced: boolean;
}

/** What a lexing answers: the tokens, and the first character no rule could cut. */
export interface Lexed {
  readonly tokens: readonly Token[];
  /** The message and position of the first refusal, when the text carries one. */
  readonly refusal?: { readonly message: string; readonly at: number };
}

/** Whether a character may begin a word: the first character of the grammar's identifier pattern. */
function beginsWord(character: string): boolean {
  return /[A-Za-z_]/.test(character);
}

/** Whether a character may continue one. */
function continuesWord(character: string): boolean {
  return /[A-Za-z0-9_-]/.test(character);
}

/**
 * The tokens of a text, given the symbols the grammar at the place admits.
 *
 * The symbols are matched longest first, so that `<=` is one token where `<` and `=` are two, and
 * word-shaped ones are left to the word rule — a symbol that is a word is claimed by the parser,
 * at the position where it can be one.
 */
export function lex(text: string, symbols: readonly string[]): Lexed {
  const marks = [...symbols, OPEN, CLOSE, COMMA]
    .filter((symbol) => !beginsWord(symbol))
    .sort((left, right) => right.length - left.length);
  const tokens: Token[] = [];
  let at = 0;
  let spaced = true;
  while (at < text.length) {
    const character = text[at] as string;
    if (/\s/.test(character)) {
      at += 1;
      spaced = true;
      continue;
    }
    const from = at;
    if (beginsWord(character)) {
      at += 1;
      while (at < text.length && (continuesWord(text[at] as string) || dotsIntoWord(text, at))) at += 1;
      tokens.push({ kind: WORD, text: text.slice(from, at), from, to: at, spaced });
      spaced = false;
      continue;
    }
    if (/[0-9]/.test(character) || (character === SIGN && /[0-9]/.test(text[at + 1] ?? ''))) {
      at = endOfNumber(text, character === SIGN ? at + 1 : at);
      tokens.push({ kind: NUMBER, text: text.slice(from, at), from, to: at, spaced });
      spaced = false;
      continue;
    }
    if (character === '"') {
      const read = readQuoted(text, at);
      if (read === null) {
        return { tokens, refusal: { message: 'a quoted literal that is never closed', at: from } };
      }
      tokens.push({ kind: QUOTED, text: text.slice(from, read.to), denotes: read.denotes, from, to: read.to, spaced });
      at = read.to;
      spaced = false;
      continue;
    }
    const mark = marks.find((one) => text.startsWith(one, at));
    if (mark !== undefined) {
      at += mark.length;
      tokens.push({ kind: MARK, text: mark, from, to: at, spaced });
      spaced = false;
      continue;
    }
    at += 1;
    tokens.push({ kind: UNKNOWN, text: character, from, to: at, spaced });
    return { tokens, refusal: { message: `'${character}' begins nothing the text form writes`, at: from } };
  }
  return { tokens };
}

/**
 * The one character a number may carry before its digits.
 *
 * Named here and not read from the bindings: it is JSON's own number grammar (`-?` before the
 * integer part), which is what a literal's lexeme is written in, and not a symbol of the
 * language. Whether the same character is also an operator's symbol is `presentation.json`'s
 * business and no concern of this rule.
 */
const SIGN = '-';

/** Whether the dot at this position joins two parts of one dotted name (`rope.scaling.kind`). */
function dotsIntoWord(text: string, at: number): boolean {
  return text[at] === '.' && beginsWord(text[at + 1] ?? '');
}

/** Where a number token ends: the JSON number grammar, minus the sign the parser owns. */
function endOfNumber(text: string, from: number): number {
  let at = from;
  while (/[0-9]/.test(text[at] ?? '')) at += 1;
  if (text[at] === '.' && /[0-9]/.test(text[at + 1] ?? '')) {
    at += 1;
    while (/[0-9]/.test(text[at] ?? '')) at += 1;
  }
  if (/[eE]/.test(text[at] ?? '')) {
    let after = at + 1;
    if (/[-+]/.test(text[after] ?? '')) after += 1;
    if (/[0-9]/.test(text[after] ?? '')) {
      at = after;
      while (/[0-9]/.test(text[at] ?? '')) at += 1;
    }
  }
  return at;
}

/** A quoted literal, read with JSON's own escapes, or `null` where it is never closed. */
function readQuoted(text: string, from: number): { denotes: string; to: number } | null {
  let at = from + 1;
  let denotes = '';
  while (at < text.length) {
    const character = text[at] as string;
    if (character === '"') return { denotes, to: at + 1 };
    if (character !== '\\') {
      denotes += character;
      at += 1;
      continue;
    }
    const escaped = text[at + 1];
    if (escaped === undefined) return null;
    if (escaped === 'u') {
      const digits = text.slice(at + 2, at + 6);
      if (!/^[0-9a-fA-F]{4}$/.test(digits)) return null;
      denotes += String.fromCharCode(Number.parseInt(digits, 16));
      at += 6;
      continue;
    }
    const known = ESCAPES.get(escaped);
    if (known === undefined) return null;
    denotes += known;
    at += 2;
  }
  return null;
}

/** JSON's own escapes, read and written by the two halves of this module. */
const ESCAPES = new Map<string, string>([
  ['"', '"'],
  ['\\', '\\'],
  ['/', '/'],
  ['b', '\b'],
  ['f', '\f'],
  ['n', '\n'],
  ['r', '\r'],
  ['t', '\t'],
]);

/**
 * A signed number token read as the sign and the number it is made of, or `null`.
 *
 * The one place the reading depends on where a token stands: after a complete operand, `a -1` is
 * the subtraction a reader means and not two operands running together. The parser asks for the
 * split exactly there; in every other position a signed number is the literal it looks like.
 */
export function splitSign(token: Token): readonly [Token, Token] | null {
  if (token.kind !== NUMBER || !token.text.startsWith(SIGN)) return null;
  const at = token.from + SIGN.length;
  return [
    { kind: MARK, text: SIGN, from: token.from, to: at, spaced: token.spaced },
    { kind: NUMBER, text: token.text.slice(SIGN.length), from: at, to: token.to, spaced: false },
  ];
}

/**
 * A text literal, quoted as §4.13 writes one.
 *
 * "A literal is a number, `true`, `false` or a **quoted string**" — and the quoting is not a
 * decoration: `causal` is a literal in `mask = causal` and a *name* in `causal = true`, both of
 * which the reference base writes, so a text form that printed the first bare would read it back
 * as the second. JSON's own escapes are used, since the value goes into a JSON document.
 */
export function quoted(value: string): string {
  return JSON.stringify(value);
}

/**
 * Whether two texts, written one after the other with nothing between them, would be cut
 * differently than as those two.
 *
 * This is the printer's spacing rule, and it is a *question about the lexer* rather than a
 * typographic taste: `not` and `true` written together are the one word `nottrue`, and `-` and
 * `1` written together are the one number `-1` — which is a different document from the negation
 * of `1`. Asking the lexer is what makes the printer's output parse back to what it printed,
 * whatever symbols `presentation.json` carries.
 */
export function runsTogether(left: string, right: string, symbols: readonly string[]): boolean {
  if (left === '' || right === '') return false;
  const joined = lex(`${left}${right}`, symbols);
  const first = joined.tokens[0];
  return first === undefined || first.text !== left;
}
