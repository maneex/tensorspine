import { describe, expect, it } from 'vitest';

import { parse, serialize, spansOf } from '@tensorspine/lang';
import { schemaProblem } from '@tensorspine/lang/api';
import type { Problem } from '@tensorspine/lang/api';

import { markersFor, markRange, revealRange } from '../../src/source/markers.js';
import { corpusText, registry } from '../documents/source.js';

// Feature 2.17, plan §4.10 and §4.17: "structural errors at their range", "Show in JSON from any
// selection reveals its range", and the third of §4.17's three navigations, which feature 2.8
// left here because it needs a text editor.
//
// Every row the editor holds carries a JSON pointer; the core answers where a pointer stands in a
// text; this turns the two into characters. What the cases below hold is that the characters are
// the right ones — a marker under the member a refusal is about, a reveal over the member *and*
// its value, as S16 draws it — and that the rows they are made from are the **core's**, taken
// from Ajv on the repository's own schemas and never re-derived here.

const MODEL = 'llama3-8b';

/** The grammar's own rows for a text, as §5.4's synchronous stage answers them. */
function structural(text: string): readonly Problem[] {
  return registry()
    .structural(parse(text), 'model')
    .map((row) => schemaProblem(row, 'llama3-8b.json'));
}

describe('the range of a place', () => {
  const text = '{\n  "model": "llama3-8b",\n  "instances": {\n    "final_n": {\n      "primitive": {}\n    }\n  }\n}\n';
  const spans = spansOf(text);

  it('reveals the member and its value — what S16 highlights', () => {
    const range = revealRange(spans, '/instances/final_n');
    expect(text.slice(range?.start, range?.end)).toBe('"final_n": {\n      "primitive": {}\n    }');
  });

  it('marks the member’s name where its value runs past one line', () => {
    const range = markRange(spans, '/instances/final_n', text);
    expect(text.slice(range?.start, range?.end)).toBe('"final_n"');
  });

  it('marks the member and its value where the two are on one line', () => {
    const range = markRange(spans, '/model', text);
    expect(text.slice(range?.start, range?.end)).toBe('"model": "llama3-8b"');
  });

  it('marks the first line of a place that has no name of its own', () => {
    const range = markRange(spans, '', text);
    expect(text.slice(range?.start, range?.end)).toBe('{');
  });

  it('answers nothing for a place the text has not', () => {
    // A hoisted binding names a place no file holds (feature 2.8), and a row about another file
    // names none of this one's. Neither is a marker.
    expect(markRange(spans, '/bindings/values/decoder.attn.norm_in', text)).toBeNull();
    expect(revealRange(spans, '/quantities/d')).toBeNull();
  });
});

describe('the markers of a document', () => {
  it('puts an unknown member’s marker under the member, not under the document', () => {
    // `additionalProperties` reports the place of the **object** and names the extras in its
    // prose, because that is where `jsonschema` puts them and the wording is the parity contract.
    // The core carries the names beside the line; this is what they are for.
    const text = corpusText(MODEL).replace('"quantities": {', '"kernels": {},\n  "quantities": {');
    const rows = structural(text);
    expect(rows.length).toBeGreaterThan(0);
    expect(rows[0]?.message).toContain("'kernels' was unexpected");
    expect(rows[0]?.path).toBe('');
    expect(rows[0]?.members).toEqual(['kernels']);

    const markers = markersFor(rows, spansOf(text), text);
    expect(markers).toHaveLength(1);
    expect(text.slice(markers[0]?.start, markers[0]?.end)).toBe('"kernels": {}');
    expect(markers[0]?.source).toBe('schema');
    expect(markers[0]?.severity).toBe('error');
  });

  it('marks the value a refusal is about where the refusal names a place', () => {
    const text = corpusText(MODEL).replace('"families": [\n        "norm"\n      ]', '"families": "norm"');
    const rows = structural(text);
    expect(rows.length).toBeGreaterThan(0);
    const markers = markersFor(rows, spansOf(text), text);
    expect(markers.length).toBeGreaterThan(0);
    expect(text.slice(markers[0]?.start, markers[0]?.end)).toBe('"families": "norm"');
  });

  it('leaves out a row whose place the text has not, and keeps the others', () => {
    const text = corpusText(MODEL);
    const spans = spansOf(text);
    const elsewhere: Problem = {
      code: 'V5',
      message: '[V5] decoder.attn.norm_in: streams disagree',
      path: '/bindings/values/decoder.attn.norm_in',
      severity: 'error',
      source: 'semantic',
    };
    const here: Problem = {
      code: 'V2',
      message: "[V2] embed: required argument missing 'width'",
      path: '/instances/embed',
      severity: 'error',
      source: 'semantic',
    };
    expect(markersFor([elsewhere, here], spans, text)).toHaveLength(1);
    expect(markersFor([elsewhere, here], spans, text)[0]?.code).toBe('V2');
  });

  it('is empty for a corpus document, which the core refuses nothing of', () => {
    const text = corpusText(MODEL);
    expect(structural(text)).toEqual([]);
    expect(markersFor(structural(text), spansOf(text), text)).toEqual([]);
  });

  it('reveals a place of the bytes the editor writes, name first', () => {
    // The pane's own invariant: a place the document declares is a place the pane can reveal, and
    // what it reveals begins at the member's name. Checked over the bytes the editor actually
    // writes (D12) rather than over the file, so a reveal is right after an edit as well as after
    // an open.
    const text = serialize(parse(corpusText(MODEL)));
    const spans = spansOf(text);
    for (const [pointer, name] of [
      ['/model', '"model"'],
      ['/quantities/d', '"d"'],
      ['/instances/embed', '"embed"'],
      ['/compositions/decoder', '"decoder"'],
    ]) {
      const range = revealRange(spans, pointer ?? '');
      expect(range, pointer).not.toBeNull();
      expect(text.slice(range?.start ?? 0, range?.end ?? 0), pointer).toMatch(new RegExp(`^${name ?? ''}: `));
    }
    // The root has no name and is the whole document, trailing newline excepted.
    const root = revealRange(spans, '');
    expect(text.slice(root?.start, root?.end)).toBe(text.trimEnd());
  });
});
