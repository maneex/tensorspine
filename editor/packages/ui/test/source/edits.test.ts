import { describe, expect, it } from 'vitest';

import { parse, serialize } from '@tensorspine/lang';

import { minimalEdit } from '../../src/source/edits.js';
import { corpusText, filesUnder } from '../documents/source.js';

// Feature 2.17, plan §4.10: "an edit on the canvas or in a form updates the source text
// minimally". The serializer renders the whole document (D12), so what "minimally" decides is
// how the new text is *applied* to the pane — one replacement of the stretch that differs, so
// that the caret, the selection, the folded regions and Monaco's own undo stack survive an edit
// made anywhere else.
//
// The claim is exact and it is checked as such: applying the edit to the old text gives the new
// one, character for character, on every pair these cases build — including the pairs a real
// gesture makes, which are a corpus document and the same document with one value changed.

/** `before` with the edit applied — what a text editor does with the answer. */
function applied(before: string, edit: { start: number; end: number; text: string } | null): string {
  if (edit === null) return before;
  return before.slice(0, edit.start) + edit.text + before.slice(edit.end);
}

describe('the minimal edit', () => {
  it('is nothing where the two texts are equal', () => {
    expect(minimalEdit('{"a": 1}', '{"a": 1}')).toBeNull();
  });

  it('replaces only the stretch that differs', () => {
    const edit = minimalEdit('{"heads": 32}', '{"heads": 16}');
    expect(edit).toEqual({ start: 10, end: 12, text: '16' });
  });

  it('inserts with an empty range, and deletes with an empty text', () => {
    expect(minimalEdit('[1, 3]', '[1, 2, 3]')).toEqual({ start: 4, end: 4, text: '2, ' });
    expect(minimalEdit('[1, 2, 3]', '[1, 3]')).toEqual({ start: 4, end: 7, text: '' });
  });

  it('does not let the prefix and the suffix claim the same characters', () => {
    // `aa` → `aaa` has a common prefix of two and a common suffix of two, over a text of two.
    expect(applied('aa', minimalEdit('aa', 'aaa'))).toBe('aaa');
    expect(applied('aaa', minimalEdit('aaa', 'aa'))).toBe('aa');
    expect(applied('', minimalEdit('', 'x'))).toBe('x');
    expect(applied('x', minimalEdit('x', ''))).toBe('');
  });

  it('never cuts a surrogate pair, at either end', () => {
    // A text editor counts in UTF-16 code units and snaps a position inside a pair to its
    // boundary, which would apply the edit somewhere else than it was computed for.
    const before = '{"note": "a\u{1F600}b"}';
    const after = '{"note": "a\u{1F601}b"}';
    const edit = minimalEdit(before, after);
    expect(edit).not.toBeNull();
    expect(isBoundary(before, edit?.start ?? 0)).toBe(true);
    expect(isBoundary(before, edit?.end ?? 0)).toBe(true);
    expect(applied(before, edit)).toBe(after);
  });

  it('applies back to the new text on every pair of corpus documents', () => {
    // Every corpus document against every other: 15 × 15 pairs of real, large, similar texts,
    // which is the shape of what a gesture produces.
    const texts = Object.values(filesUnder('data/models'));
    expect(texts.length).toBeGreaterThan(14);
    for (const before of texts) {
      for (const after of texts) {
        expect(applied(before, minimalEdit(before, after))).toBe(after);
      }
    }
  });

  it('is a single small stretch for the edit a gesture actually makes', () => {
    // The case the pane meets a hundred times an hour: one value of one document changed, and the
    // serializer's own rendering of the result (D12).
    const before = corpusText('llama3-8b');
    const tree = parse(before);
    const after = serialize(edited(tree));
    const edit = minimalEdit(before, after);
    expect(edit).not.toBeNull();
    expect(applied(before, edit)).toBe(after);
    // Two characters out of forty thousand: a whole-text replacement is what this exists to avoid.
    expect((edit?.end ?? 0) - (edit?.start ?? 0)).toBeLessThan(8);
    expect(edit?.text.length).toBeLessThan(8);
  });
});

/** Whether an offset is not inside a surrogate pair. */
function isBoundary(text: string, at: number): boolean {
  if (at <= 0 || at >= text.length) return true;
  const before = text.charCodeAt(at - 1);
  return !(before >= 0xd800 && before <= 0xdbff);
}

/** `llama3-8b` with `decoder`'s one literal `32` replaced — one value, one place. */
function edited(tree: ReturnType<typeof parse>): ReturnType<typeof parse> {
  const text = serialize(tree);
  return parse(text.replace('"literal": 32', '"literal": 16'));
}
