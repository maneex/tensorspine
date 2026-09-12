/**
 * The **minimal** text edit between two renderings of one document — plan §4.10.
 *
 * > an edit on the canvas or in a form updates the source text minimally (the serializer
 * > preserves member order and number lexemes and rewrites the changed subtree).
 *
 * The serializer renders the whole document (D12: it is the writer of record and it reproduces
 * the corpus's bytes), so what "minimally" can mean here is not *how the text is produced* but
 * *how it is applied*: the text a gesture elsewhere produced differs from the one the pane holds
 * in one stretch, and replacing only that stretch is what keeps the caret, the selection, the
 * folded regions, the scroll position and the editor's own undo stack where the reader left them.
 * Replacing the whole text would move all five on every keystroke made anywhere else.
 *
 * It is a function of two strings and nothing else, which is why it is here and not in the
 * component: a diff is a rule, and a rule has one implementation and a test.
 */

/** One replacement: the stretch of the old text that goes, and what takes its place. */
export interface TextEdit {
  /** The first character replaced. */
  readonly start: number;
  /** The character after the last one replaced; equal to {@link start} for an insertion. */
  readonly end: number;
  /** What is written there; empty for a deletion. */
  readonly text: string;
}

/**
 * The one replacement that turns `before` into `after`, or `null` where they are already equal.
 *
 * The longest common prefix and the longest common suffix are taken and what is left between
 * them is the edit. The two are not allowed to overlap — a text that gained a repetition
 * (`aa` → `aaa`) has a prefix and a suffix that would otherwise claim the same characters — so
 * the suffix is capped at what the prefix left of each side.
 *
 * **Neither boundary is allowed to fall inside a surrogate pair.** The offsets are UTF-16 code
 * units, which is what a text editor counts in, and a position in the middle of a pair is one an
 * editor snaps to the nearest boundary — which would apply the edit somewhere else than it was
 * computed for. The corpus and the reference base carry non-ASCII text written as it stands
 * (`ensure_ascii=False`, feature 0.3), so this is a real boundary and not a theoretical one.
 */
export function minimalEdit(before: string, after: string): TextEdit | null {
  if (before === after) return null;
  let prefix = 0;
  const shortest = Math.min(before.length, after.length);
  while (prefix < shortest && before.charCodeAt(prefix) === after.charCodeAt(prefix)) prefix += 1;
  if (prefix > 0 && isHighSurrogate(before.charCodeAt(prefix - 1))) prefix -= 1;
  let suffix = 0;
  const room = shortest - prefix;
  while (
    suffix < room &&
    before.charCodeAt(before.length - 1 - suffix) === after.charCodeAt(after.length - 1 - suffix)
  ) {
    suffix += 1;
  }
  if (suffix > 0 && isLowSurrogate(before.charCodeAt(before.length - suffix))) suffix -= 1;
  return {
    start: prefix,
    end: before.length - suffix,
    text: after.slice(prefix, after.length - suffix),
  };
}

/** The first half of a surrogate pair. */
function isHighSurrogate(code: number): boolean {
  return code >= 0xd800 && code <= 0xdbff;
}

/** The second half of a surrogate pair. */
function isLowSurrogate(code: number): boolean {
  return code >= 0xdc00 && code <= 0xdfff;
}
