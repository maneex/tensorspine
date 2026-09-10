/**
 * The writer of record (plan D12, §5.5).
 *
 * The corpus and the reference base are written by Python's
 * `json.dumps(document, indent=2, ensure_ascii=False)` with a trailing newline — measured over
 * all 146 files of `data/models/` and `data/primitive-library/`, every one of which this writer
 * reproduces byte for byte. That is: two-space indentation, one member or item to a line, `": "`
 * between a name and its value, `{}` and `[]` for the empty ones, text as it stands with only
 * JSON's own escapes, and member order as authored.
 *
 * A number is re-emitted with the text it was read with, so that `1e-05`, `1.0` and `1e+16` do
 * not turn into `0.00001`, `1` and `1e16` on the first save; a number the editor made, or one whose
 * value no longer agrees with its text, is formatted by the rule of `./number.ts` from the
 * float-ness the tree carries.
 *
 * JavaScript's own escaping is Python's here: `JSON.stringify` escapes `"`, `\`, and the control
 * characters, with `\b \f \n \r \t` named and the rest as `\u00xx` in lower-case hexadecimal —
 * exactly `json.encoder.encode_basestring` — and leaves every other character alone, which is
 * what `ensure_ascii=False` does with the `§`, `→`, `θ` and `−` the reference base is written
 * with.
 */
import { formatNumber, lexemeDenotes } from './number.js';
import { isJsonArray, isJsonNumber, isJsonObject, type JsonValue } from './tree.js';

const INDENT = '  ';

/**
 * The tree as the interchange format writes it, ending in a newline.
 *
 * Throws a `TypeError` for anything that is not a value of the tree — a bare JavaScript number
 * among them, since nothing but its caller knows whether it is a cardinality or a real (D12).
 */
export function serialize(value: JsonValue): string {
  const out: string[] = [];
  write(value, '', '', out);
  out.push('\n');
  return out.join('');
}

function write(value: JsonValue, indent: string, path: string, out: string[]): void {
  if (typeof value === 'string') {
    out.push(JSON.stringify(value));
    return;
  }
  if (typeof value === 'boolean') {
    out.push(value ? 'true' : 'false');
    return;
  }
  if (value === null) {
    out.push('null');
    return;
  }
  if (isJsonNumber(value)) {
    out.push(numberText(value.lexeme, value.value, value.real));
    return;
  }
  if (isJsonArray(value)) {
    if (value.length === 0) {
      out.push('[]');
      return;
    }
    const inner = indent + INDENT;
    out.push('[\n');
    for (const [position, item] of value.entries()) {
      if (position > 0) out.push(',\n');
      out.push(inner);
      write(item, inner, `${path}/${String(position)}`, out);
    }
    out.push('\n', indent, ']');
    return;
  }
  if (isJsonObject(value)) {
    if (value.members.length === 0) {
      out.push('{}');
      return;
    }
    const inner = indent + INDENT;
    out.push('{\n');
    for (const [position, member] of value.members.entries()) {
      if (position > 0) out.push(',\n');
      out.push(inner, JSON.stringify(member.name), ': ');
      write(member.value, inner, `${path}/${pointerSegment(member.name)}`, out);
    }
    out.push('\n', indent, '}');
    return;
  }
  throw new TypeError(
    typeof value === 'number'
      ? `a bare number at ${where(path)}: build it with jsonInteger or jsonReal, so that its float-ness is stated`
      : `${describe(value)} at ${where(path)} is not a value of the tree`,
  );
}

/** The lexeme when it still denotes the value, the formatted value otherwise. */
function numberText(lexeme: string | undefined, value: number, real: boolean): string {
  if (lexeme !== undefined && lexemeDenotes(lexeme, value, real)) return lexeme;
  return formatNumber(value, real);
}

/** The pointer an error names; the root has none, so it is named. */
function where(path: string): string {
  return path === '' ? 'the root' : path;
}

/** RFC 6901's escaping, for the pointer an error names. */
function pointerSegment(name: string): string {
  return name.replace(/~/g, '~0').replace(/\//g, '~1');
}

function describe(value: unknown): string {
  if (value === undefined) return 'undefined';
  if (typeof value === 'object') {
    const kind: unknown = (value as { kind?: unknown }).kind;
    return typeof kind === 'string' ? `an object of kind '${kind}'` : 'an object of no kind';
  }
  return `a ${typeof value}`;
}
