/**
 * The lexeme-preserving parser (plan D12).
 *
 * It reads what CPython's `json` reads, because that is what the repository's tools read: the
 * JSON grammar plus the three non-finite names `NaN`, `Infinity` and `-Infinity`, with duplicate
 * member names refused rather than silently resolved to the last value — `tools/model.py`'s
 * `_pairs` and `tools/primitive_library.py`'s `read_json` both install that refusal, and V12 is
 * the rule ("a document or unit with duplicate member names in any object is rejected").
 *
 * Two details are deliberately CPython's, so that a message the editor shows is a message the
 * tools would print:
 *
 * - **The wording.** A duplicate reads `duplicate member name 'model' (V12)`, which is
 *   `model.py`'s text and the substring `tests/rejections/models.json` matches; a syntax error
 *   reads `Expecting ',' delimiter: line 3 column 5 (char 42)`, which is the shape and the
 *   position `json.JSONDecodeError` carries and `primitive_library.read_json` re-raises as
 *   `<path>: <message> (V12)`. `reason` holds the text without the position, which is what the
 *   library loader's own wrapper needs (feature 1.3).
 * - **Which duplicate is named.** `json.load(object_pairs_hook=…)` calls the hook when an object
 *   closes, so in `{"a": 1, "a": 2, "b": {"x": 1, "x": 2}}` it is the inner `x` that is refused,
 *   not the outer `a` that appears first. The check here runs at the closing brace for the same
 *   reason and names the same member.
 *
 * Of those two, only the duplicate's wording is a parity contract: `tests/rejections` has a case
 * for it (`models/v12-duplicate-member-name.json`, matched on `duplicate member name 'model'`)
 * and none for a malformed document. The syntax messages are CPython 3.11's all the same, read
 * off the interpreter and pinned case by case in the tests, so that a unit the loader cannot read
 * is reported in the words `--validate` would have used.
 *
 * The parser keeps every number's source text and reads its float-ness from the token's own
 * spelling (V3: a fraction or an exponent makes a real), so that an unedited document written
 * back is written byte for byte as it was read.
 *
 * There is one reading beside V12's, and it has one caller: `duplicates: 'last'` is the plain
 * `json.load`, which `tools/schema.py`'s `check` uses. The library loader runs the schema stage
 * before `read_json` and therefore needs that reading, so that a unit which is both off-schema and
 * holds a duplicate is refused for being off-schema, as the tools refuse it (feature 1.3).
 */
import { NON_FINITE_LEXEMES } from './number.js';
import type { JsonMember, JsonObject, JsonValue } from './tree.js';

/**
 * A text that cannot be read as the tools read it: a syntax error, or a duplicate member name.
 *
 * `message` is what the tools print — `duplicate member name 'x' (V12)` for a duplicate, and
 * `<reason>: line L column C (char N)` for a syntax error. `duplicateMember` separates the two
 * cases, as `tools/validate.py` does when it catches `ModelError` alone; the position is carried
 * in every case, for the source view's marker and for the Problems panel.
 */
export class JsonParseError extends Error {
  /** The text without the position: `Expecting value`, `duplicate member name 'x'`. */
  readonly reason: string;
  /** 1-based, as CPython counts them. */
  readonly line: number;
  readonly column: number;
  /** 0-based offset in the text, CPython's `char`. */
  readonly offset: number;
  /** The member name refused, when a duplicate is what was refused; `null` otherwise. */
  readonly duplicateMember: string | null;

  constructor(
    reason: string,
    position: { line: number; column: number; offset: number },
    duplicateMember: string | null = null,
  ) {
    super(
      duplicateMember === null
        ? `${reason}: line ${String(position.line)} column ${String(position.column)} (char ${String(position.offset)})`
        : `${reason} (V12)`,
    );
    this.name = 'JsonParseError';
    this.reason = reason;
    this.line = position.line;
    this.column = position.column;
    this.offset = position.offset;
    this.duplicateMember = duplicateMember;
  }
}

/** CPython's `json` whitespace: space, tab, newline, carriage return. */
const WHITESPACE = new Set([' ', '\t', '\n', '\r']);

/** The escapes CPython's scanner decodes; the solidus among them, as JSON allows. */
const ESCAPES: Readonly<Record<string, string>> = {
  '"': '"',
  '\\': '\\',
  '/': '/',
  b: '\b',
  f: '\f',
  n: '\n',
  r: '\r',
  t: '\t',
};

/** CPython's `NUMBER_RE`, anchored: the longest number token at this position. */
const NUMBER = /-?(?:0|[1-9][0-9]*)(\.[0-9]+)?([eE][-+]?[0-9]+)?/y;

const HEX = /^[0-9a-fA-F]{4}$/;

/** How a duplicate member name is read. */
export interface ParseOptions {
  /**
   * `refuse` (the default) is V12: `json.load(object_pairs_hook=…)`, which both `model.py` and
   * `primitive_library.py` install. `last` is the plain `json.load` — the first occurrence's
   * place, the last occurrence's value, as `dict(pairs)` builds it.
   *
   * The lenient reading exists for one place and is named there: `primitive_library._units` runs
   * `schema.check` on a unit *before* `read_json`, and `schema.check` reads the file with a plain
   * `json.load`. So a unit that is both off-schema and holds a duplicate is refused for being
   * off-schema, and the library loader needs that reading to say so.
   */
  readonly duplicates?: 'refuse' | 'last';
}

/**
 * The text as an ordered tree, with every number's lexeme and float-ness kept.
 *
 * Throws {@link JsonParseError} for a text the tools would refuse, with the tools' wording.
 */
export function parse(text: string, options: ParseOptions = {}): JsonValue {
  const state = new Scanner(text, options.duplicates ?? 'refuse');
  const value = state.value(state.skipWhitespace(0));
  const end = state.skipWhitespace(state.index);
  if (end !== text.length) state.fail('Extra data', end);
  return value;
}

class Scanner {
  readonly text: string;
  readonly duplicates: 'refuse' | 'last';
  /** Where the last value read ended. */
  index = 0;

  constructor(text: string, duplicates: 'refuse' | 'last' = 'refuse') {
    this.text = text;
    this.duplicates = duplicates;
  }

  /**
   * The refusal, positioned as CPython positions one: the line and the column both 1-based, the
   * offset the character count, so that `line 1 column 1 (char 0)` is the start of a text.
   */
  fail(reason: string, at: number, duplicateMember: string | null = null): never {
    const before = at > 0 ? this.text.lastIndexOf('\n', at - 1) : -1;
    let line = 1;
    for (let i = 0; i < at && i < this.text.length; i += 1) if (this.text[i] === '\n') line += 1;
    throw new JsonParseError(reason, { line, column: at - before, offset: at }, duplicateMember);
  }

  skipWhitespace(from: number): number {
    let i = from;
    while (i < this.text.length) {
      const character = this.text[i];
      if (character === undefined || !WHITESPACE.has(character)) break;
      i += 1;
    }
    return i;
  }

  /** The value at `from`, leaving `index` after it. */
  value(from: number): JsonValue {
    const character = this.text[from];
    if (character === '"') return this.string(from);
    if (character === '{') return this.object(from);
    if (character === '[') return this.array(from);
    if (this.text.startsWith('null', from)) {
      this.index = from + 4;
      return null;
    }
    if (this.text.startsWith('true', from)) {
      this.index = from + 4;
      return true;
    }
    if (this.text.startsWith('false', from)) {
      this.index = from + 5;
      return false;
    }
    NUMBER.lastIndex = from;
    const number = NUMBER.exec(this.text);
    if (number !== null && number.index === from) {
      const lexeme = number[0];
      this.index = from + lexeme.length;
      // V3's lexical rule, read off the token: a fraction or an exponent makes a real.
      const real = number[1] !== undefined || number[2] !== undefined;
      return { kind: 'number', value: Number(lexeme), real, lexeme };
    }
    // The three names CPython reads as floats, tried after the number token, as its scanner does.
    for (const lexeme of NON_FINITE_LEXEMES) {
      if (this.text.startsWith(lexeme, from)) {
        this.index = from + lexeme.length;
        return { kind: 'number', value: Number(lexeme), real: true, lexeme };
      }
    }
    this.fail('Expecting value', from);
  }

  string(from: number): string {
    const chunks: string[] = [];
    let chunkStart = from + 1;
    let i = chunkStart;
    for (;;) {
      if (i >= this.text.length) this.fail('Unterminated string starting at', from);
      const code = this.text.charCodeAt(i);
      if (code === 0x22) {
        chunks.push(this.text.slice(chunkStart, i));
        this.index = i + 1;
        return chunks.join('');
      }
      if (code === 0x5c) {
        chunks.push(this.text.slice(chunkStart, i));
        const escape = this.text[i + 1];
        if (escape === undefined) this.fail('Unterminated string starting at', from);
        if (escape === 'u') {
          const hex = this.text.slice(i + 2, i + 6);
          if (!HEX.test(hex)) this.fail('Invalid \\uXXXX escape', i + 1);
          chunks.push(String.fromCharCode(Number.parseInt(hex, 16)));
          i += 6;
        } else {
          const decoded = ESCAPES[escape];
          if (decoded === undefined) this.fail('Invalid \\escape', i);
          chunks.push(decoded);
          i += 2;
        }
        chunkStart = i;
        continue;
      }
      // Strict reading, as `json.load` is by default: a raw control character is refused.
      if (code < 0x20) this.fail('Invalid control character at', i);
      i += 1;
    }
  }

  object(from: number): JsonObject {
    const members: JsonMember[] = [];
    const nameAt: number[] = [];
    let i = this.skipWhitespace(from + 1);
    if (this.text[i] === '}') {
      this.index = i + 1;
      return this.closed(members, nameAt);
    }
    for (;;) {
      if (this.text[i] !== '"') this.fail('Expecting property name enclosed in double quotes', i);
      const at = i;
      const name = this.string(i);
      i = this.skipWhitespace(this.index);
      if (this.text[i] !== ':') this.fail("Expecting ':' delimiter", i);
      i = this.skipWhitespace(i + 1);
      const value = this.value(i);
      members.push({ name, value });
      nameAt.push(at);
      i = this.skipWhitespace(this.index);
      if (this.text[i] === '}') {
        this.index = i + 1;
        return this.closed(members, nameAt);
      }
      if (this.text[i] !== ',') this.fail("Expecting ',' delimiter", i);
      i = this.skipWhitespace(i + 1);
    }
  }

  /**
   * The object, once its members are read: the duplicate check runs here, at the closing brace,
   * where `json.load`'s `object_pairs_hook` runs it.
   *
   * Under `duplicates: 'last'` the members are folded as `dict(pairs)` folds them instead — a
   * repeated name keeps the place of its first occurrence and the value of its last, which is
   * what assigning into a dictionary does.
   */
  closed(members: JsonMember[], nameAt: number[]): JsonObject {
    if (this.duplicates === 'last') return { kind: 'object', members: folded(members) };
    const seen = new Set<string>();
    for (const [position, member] of members.entries()) {
      if (seen.has(member.name)) {
        this.fail(`duplicate member name '${member.name}'`, nameAt[position] ?? 0, member.name);
      }
      seen.add(member.name);
    }
    return { kind: 'object', members };
  }

  array(from: number): JsonValue[] {
    const items: JsonValue[] = [];
    let i = this.skipWhitespace(from + 1);
    if (this.text[i] === ']') {
      this.index = i + 1;
      return items;
    }
    for (;;) {
      items.push(this.value(i));
      i = this.skipWhitespace(this.index);
      if (this.text[i] === ']') {
        this.index = i + 1;
        return items;
      }
      if (this.text[i] !== ',') this.fail("Expecting ',' delimiter", i);
      i = this.skipWhitespace(i + 1);
    }
  }
}

/** `dict(pairs)`: one member per name, at its first place, with its last value. */
function folded(members: readonly JsonMember[]): JsonMember[] {
  const at = new Map<string, number>();
  const out: JsonMember[] = [];
  for (const member of members) {
    const known = at.get(member.name);
    if (known === undefined) {
      at.set(member.name, out.length);
      out.push(member);
    } else {
      out[known] = member;
    }
  }
  return out;
}
