/**
 * JSON pointers (RFC 6901), written from the segments of a place.
 *
 * A pointer names a place of a *document*, which is what the whole editor navigates by: a
 * problem carries one ({@link Problem.path}), the store keys a place by one, `presentation.json`
 * keys a binding by one into a schema, and the source view of §4.10 turns one into a range of
 * text. The rule is three characters long and was written three times before this module — in
 * `schema/pointer.ts`, inside `schema/types.ts`'s `pointerOf`, and here — so it is written once
 * and the other two read it.
 *
 * It lives under `json/` and not under `schema/` because that is the direction the core's
 * imports run: `schema/` reads `json/`, never the other way round, and the span map of
 * `spans.ts` needs an escaper while it parses.
 */

/**
 * One segment of a pointer, escaped: `~` becomes `~0` and `/` becomes `~1`.
 *
 * Those are the only two escapes RFC 6901 has, and the order matters — escaping the tilde first
 * is what stops `~1` in a name from being read back as a slash.
 */
export function pointerSegment(name: string): string {
  return name.replace(/~/g, '~0').replace(/\//g, '~1');
}

/**
 * A JSON pointer from a path: `''` at the root, `/instances/embed`, `/families/0`.
 *
 * A numeric segment is an array index and is written as it stands; a name is escaped.
 */
export function jsonPointerOf(path: readonly (string | number)[]): string {
  return path
    .map((segment) => (typeof segment === 'number' ? `/${String(segment)}` : `/${pointerSegment(segment)}`))
    .join('');
}
