import { describe, expect, it } from 'vitest';

import { JsonParseError, parse, spansOf } from '../../src/json/parse.js';
import { jsonPointerOf } from '../../src/json/pointer.js';
import { isJsonArray, isJsonObject, type JsonValue } from '../../src/json/tree.js';
import { interchangeFiles, readRepositoryFile } from './repository.js';

// Feature 2.17, plan §4.10: the source view reveals the *range* of a place — "Show in JSON from
// any selection reveals its range", a problem's pointer becomes a marker under the text it is
// about. A pointer is what every other part of the editor navigates by (a problem carries one,
// the store keys a place by one), so what the source view needs from the core is the map from a
// pointer to the two offsets its text stands between.
//
// It is the **parser's** map and not a second reading: the same scanner, the same refusals, one
// `Map.set` per value. The claim this file makes is that the map is exact — every span is a slice
// of the text that parses back to the value at that place — and it makes it over every file of
// the corpus and of the reference base, which is the whole of what the editor opens.

const files = interchangeFiles();

/** Every pointer a tree carries, the root included — the set the map must have exactly. */
function pointersOf(tree: JsonValue, at: readonly (string | number)[] = []): string[] {
  const here = [jsonPointerOf(at)];
  if (isJsonArray(tree)) return [...here, ...tree.flatMap((item, index) => pointersOf(item, [...at, index]))];
  if (isJsonObject(tree)) {
    return [...here, ...tree.members.flatMap((member) => pointersOf(member.value, [...at, member.name]))];
  }
  return here;
}

describe('the span of a place', () => {
  const text = '{\n  "model": "llama3-8b",\n  "eps": 1e-05,\n  "families": [ "norm" ]\n}\n';
  const spans = spansOf(text);

  it('is the value’s own extent, and the slice parses back to it', () => {
    const eps = spans.get('/eps');
    expect(eps).toBeDefined();
    expect(text.slice(eps?.start, eps?.end)).toBe('1e-05');
  });

  it('carries the member name beside it, quotes included', () => {
    const model = spans.get('/model');
    expect(text.slice(model?.nameStart, model?.nameEnd)).toBe('"model"');
    expect(text.slice(model?.start, model?.end)).toBe('"llama3-8b"');
  });

  it('names the root as the empty pointer, over the whole document', () => {
    const root = spans.get('');
    expect(root?.start).toBe(0);
    expect(text.slice(root?.start, root?.end)).toBe(text.trimEnd());
    // The root is nobody's member, so it carries no name.
    expect(root?.nameStart).toBeUndefined();
  });

  it('gives an array item its index, and no name', () => {
    const item = spans.get('/families/0');
    expect(text.slice(item?.start, item?.end)).toBe('"norm"');
    expect(item?.nameStart).toBeUndefined();
    expect(text.slice(spans.get('/families')?.start, spans.get('/families')?.end)).toBe('[ "norm" ]');
  });

  it('escapes a member name as RFC 6901 escapes it', () => {
    const odd = spansOf('{"a/b": 1, "c~d": 2}');
    expect([...odd.keys()].sort()).toEqual(['', '/a~1b', '/c~0d']);
  });

  it('refuses what the parser refuses, in the same words', () => {
    expect(() => spansOf('{"a": }')).toThrow(JsonParseError);
    expect(() => spansOf('{"a": 1, "a": 2}')).toThrow(/duplicate member name 'a' \(V12\)/);
  });
});

describe('over every file the editor opens', () => {
  it('is a map with one span per place of the tree, and no other key', () => {
    for (const path of files) {
      const text = readRepositoryFile(path);
      const spans = spansOf(text);
      expect([...spans.keys()].sort(), path).toEqual(pointersOf(parse(text)).sort());
    }
  });

  it('slices a text that parses back to exactly the value at that place', () => {
    // The strongest statement the map can make, and the one the source view rests on: a range is
    // not "about" a place, it *is* the place. Checked on every file, at every place.
    for (const path of files) {
      const text = readRepositoryFile(path);
      const tree = parse(text);
      for (const [pointer, span] of spansOf(text)) {
        const segments = pointer === '' ? [] : pointer.slice(1).split('/').map(unescape);
        const expected = valueAt(tree, segments);
        expect(parse(text.slice(span.start, span.end)), `${path} ${pointer}`).toEqual(expected);
      }
    }
  }, 120_000);

  it('puts a member’s name before its value and inside its parent', () => {
    for (const path of files) {
      const text = readRepositoryFile(path);
      const spans = spansOf(text);
      for (const [pointer, span] of spans) {
        if (span.nameStart === undefined || span.nameEnd === undefined) continue;
        expect(span.nameStart, `${path} ${pointer}`).toBeLessThan(span.start);
        expect(span.nameEnd, `${path} ${pointer}`).toBeLessThanOrEqual(span.start);
        const parent = spans.get(pointer.slice(0, pointer.lastIndexOf('/')));
        expect(parent?.start, `${path} ${pointer}`).toBeLessThan(span.nameStart);
        expect(parent?.end, `${path} ${pointer}`).toBeGreaterThanOrEqual(span.end);
      }
    }
  }, 120_000);
});

/** A pointer segment read back: `~1` is a slash, `~0` a tilde, in that order. */
function unescape(segment: string): string {
  return segment.replace(/~1/g, '/').replace(/~0/g, '~');
}

/** The value a pointer names: a segment is an index where the container is an array. */
function valueAt(tree: JsonValue, segments: readonly string[]): JsonValue | undefined {
  let here: JsonValue | undefined = tree;
  for (const segment of segments) {
    if (here === undefined) return undefined;
    if (isJsonArray(here)) here = here[Number(segment)];
    else if (isJsonObject(here)) here = here.members.find((one) => one.name === segment)?.value;
    else return undefined;
  }
  return here;
}
