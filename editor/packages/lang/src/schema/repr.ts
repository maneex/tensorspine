/**
 * `repr`, as CPython writes it.
 *
 * The tools' structural stage prints the messages of `jsonschema`, and almost every one of them
 * is built by interpolating a value: `'sideways' is not among [...]`, `{'context': …} is not
 * valid under any of the given schemas`, `[] should be non-empty`. Those texts are the parity
 * contract (the plan's D2 and §7 F1 — `tests/rejections` matches on them), so the core has to
 * write a value exactly as `repr` writes it, quotes, escapes and all.
 *
 * Three things separate `repr` from `JSON.stringify`:
 *
 * - **Quotes.** A string is written between apostrophes, unless it contains one and no double
 *   quote, in which case double quotes are used and the apostrophe is not escaped.
 * - **Printability.** Only the unprintable characters are escaped, and printability is Unicode's:
 *   `é` is written as itself, U+2028 — a line separator — is escaped, and so is DEL. The
 *   escape is `\xNN`, `\uNNNN` or `\UNNNNNNNN`, by code point, in lower-case hexadecimal.
 * - **Numbers.** `True`, `False` and `None` are Python's names, a float always shows a fraction
 *   or an exponent (`1.0`, `1e-05`) and a whole number never does. Which of the two a number is
 *   is never guessed: the tree carries it (D12, `json/number.ts`), because `16` and `16.0` are
 *   one double and only the document's spelling separates them.
 *
 * A value taken from a schema file is plain JavaScript data rather than a tree node — a schema is
 * read, never edited — so a number found there is written whole when it is whole. Every numeric
 * literal of the five schemas is a whole number (bounds of 0, 1, 2, 3, 120 and 512), so nothing
 * of the parity contract rests on that reading.
 */
import { formatNumber, lexemeDenotes } from '../json/number.js';
import { isJsonArray, isJsonNumber, isJsonObject, type JsonValue } from '../json/tree.js';

/** The characters `repr` never writes as themselves, beyond the quote and the backslash. */
const NAMED_ESCAPES: ReadonlyMap<string, string> = new Map([
  ['\t', '\\t'],
  ['\n', '\\n'],
  ['\r', '\\r'],
]);

/**
 * Unicode's unprintable characters, as `str.isprintable` reads them: the Other categories (Cc,
 * Cf, Cs, Co, Cn) and the separators (Zl, Zp, Zs). The ASCII space is the one separator Python
 * calls printable, and it is excluded below rather than here.
 */
const UNPRINTABLE = /[\p{C}\p{Zl}\p{Zp}\p{Zs}]/u;

/** Whether the code point is one `repr` writes as itself. */
function isPrintable(character: string): boolean {
  return character === ' ' || !UNPRINTABLE.test(character);
}

/** One code point escaped as `repr` escapes it: `\xNN`, `\uNNNN` or `\UNNNNNNNN`. */
function escapeCodePoint(character: string): string {
  const code = character.codePointAt(0) ?? 0;
  if (code < 0x100) return `\\x${code.toString(16).padStart(2, '0')}`;
  if (code < 0x10000) return `\\u${code.toString(16).padStart(4, '0')}`;
  return `\\U${code.toString(16).padStart(8, '0')}`;
}

/**
 * The order Python puts two strings in: by code point.
 *
 * JavaScript's `<` compares UTF-16 code units, so a character of the private-use area sorts after
 * an astral one where Python sorts it before. Two places depend on the order — the names an
 * `additionalProperties` message lists, and the places two problems are reported in — and both
 * are the tools' order, not JavaScript's.
 */
export function comparePythonStrings(left: string, right: string): number {
  if (left === right) return 0;
  const leftPoints = [...left];
  const rightPoints = [...right];
  const shared = Math.min(leftPoints.length, rightPoints.length);
  for (let index = 0; index < shared; index += 1) {
    const a = (leftPoints[index] as string).codePointAt(0) ?? 0;
    const b = (rightPoints[index] as string).codePointAt(0) ?? 0;
    if (a !== b) return a < b ? -1 : 1;
  }
  return leftPoints.length - rightPoints.length;
}

/** A string as `repr` writes it, quote chosen and every unprintable character escaped. */
export function pythonReprString(text: string): string {
  const quote = text.includes("'") && !text.includes('"') ? '"' : "'";
  let out = quote;
  for (const character of text) {
    if (character === quote || character === '\\') {
      out += `\\${character}`;
      continue;
    }
    const named = NAMED_ESCAPES.get(character);
    if (named !== undefined) {
      out += named;
      continue;
    }
    out += isPrintable(character) ? character : escapeCodePoint(character);
  }
  return out + quote;
}

/** A number as `repr` writes it: `nan`, `inf` and `-inf` have names, the rest is `json/number`. */
function pythonReprNumber(value: number, real: boolean): string {
  if (Number.isNaN(value)) return 'nan';
  if (value === Infinity) return 'inf';
  if (value === -Infinity) return '-inf';
  return formatNumber(value, real);
}

/**
 * A value of a document tree, or of a schema file, as `repr` writes it.
 *
 * A tree node carries its own float-ness; a plain JavaScript number does not, so a whole one is
 * written whole. Anything the two forms cannot express — a function, `undefined` — is refused
 * rather than written as something else, since it can only come from a caller's mistake.
 */
export function pythonRepr(value: unknown): string {
  if (value === null) return 'None';
  if (value === true) return 'True';
  if (value === false) return 'False';
  if (typeof value === 'string') return pythonReprString(value);
  if (typeof value === 'number') return pythonReprNumber(value, !Number.isInteger(value));
  if (isJsonNumber(value as JsonValue)) {
    const node = value as { value: number; real: boolean; lexeme?: string };
    // An integer's digits, where the node kept them. CPython reads `12345678901234567890` as an
    // `int` and `repr` prints it whole; `node.value` is the nearest double to it, whose digits are
    // different ones. The text is only taken where it still denotes this number and says the same
    // thing about its float-ness — a `1E5` stays the float `100000.0`, as `repr` writes it — and
    // it goes through `BigInt` so that `-0` reads `0`, which is the integer CPython read.
    if (!node.real && node.lexeme !== undefined && lexemeDenotes(node.lexeme, node.value, false)) {
      return BigInt(node.lexeme).toString();
    }
    return pythonReprNumber(node.value, node.real);
  }
  if (isJsonArray(value as JsonValue)) {
    const items = value as readonly unknown[];
    return `[${items.map((item) => pythonRepr(item)).join(', ')}]`;
  }
  if (isJsonObject(value as JsonValue)) {
    const node = value as { members: readonly { name: string; value: JsonValue }[] };
    const members = node.members.map(
      (member) => `${pythonReprString(member.name)}: ${pythonRepr(member.value)}`,
    );
    return `{${members.join(', ')}}`;
  }
  if (typeof value === 'object') {
    const members = Object.entries(value as Record<string, unknown>).map(
      ([name, member]) => `${pythonReprString(name)}: ${pythonRepr(member)}`,
    );
    return `{${members.join(', ')}}`;
  }
  throw new TypeError(`${typeof value} has no JSON form and no repr`);
}
