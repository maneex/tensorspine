import { describe, expect, it } from 'vitest';

import { multiSheet } from '../../src/sheet/multi.js';
import { sheetOf, siteOf } from './source.js';

/**
 * §4.11's multi-selection sheet — the intersection of the selected sites' editable rows.
 *
 * Features 2.10 and 2.12 both left it "waiting on a selection that can hold more than one place";
 * §4.4's rubber band is that selection (feature 2.14), so the intersection is asserted here over
 * the corpus's own sites: two of one primitive, two of different ones, and a pair that disagrees.
 */

const LLAMA = 'llama3-8b';
const GEMMA = 'gemma3n-kvshare';

function selection(model: string, wheres: readonly string[]): ReturnType<typeof multiSheet> {
  return multiSheet(wheres.map((where) => ({ site: siteOf(model, where), sheet: sheetOf(model, where) })));
}

describe('two sites of one primitive', () => {
  // `attn_n` and `ffn_n` of `llama3-8b`'s decoder are both `norm.rms`, and both write `width=d`.
  const multi = selection(LLAMA, ['decoder/attn_n[layer=0]', 'decoder/ffn_n[layer=0]']);

  it('names the sites, the primitive they share and the version they pin', () => {
    expect(multi.count).toBe(2);
    expect(multi.names).toEqual(['attn_n', 'ffn_n']);
    expect(multi.primitive).toBe('norm.rms');
    expect(multi.version).toBe('1.0.0');
  });

  it('has every argument in common, and none of them differs', () => {
    expect(multi.rows.map((row) => row.path)).toEqual(['width', 'eps', 'zero_centered']);
    expect(multi.dropped).toBe(0);
    expect(multi.rows.filter((row) => row.differs)).toEqual([]);
    // The row the sheet draws is the first site's, and the other is carried beside it so that an
    // edit reaches both (D13's one command).
    const width = multi.rows.find((row) => row.path === 'width');
    expect(width?.row.at.join('/')).toBe('compositions/decoder/instances/attn_n/arguments/width');
    expect(width?.also.map((row) => row.at.join('/'))).toEqual([
      'compositions/decoder/instances/ffn_n/arguments/width',
    ]);
  });

  it('keeps the families they share', () => {
    expect(multi.families).toEqual(['norm']);
    expect(multi.familiesDiffer).toBe(false);
  });
});

describe('two sites of one primitive that disagree', () => {
  // `attn` and `attn_full` of `gemma3n-kvshare` are both `attention.dense`; one writes a window
  // and the other does not, and their guards differ.
  const multi = selection(GEMMA, ['decoder/attn[layer=0]', 'decoder/attn_full[layer=4]']);

  it('marks the rows the sites write differently, and only those', () => {
    const differing = multi.rows.filter((row) => row.differs).map((row) => row.path);
    // `attn` writes `window` and `attn_full` does not, so the record's own row disagrees.
    expect(differing).toEqual(['window']);
    expect(differing).not.toContain('heads');
    expect(differing).not.toContain('mask');
  });

  it('leaves out a field one site has and the other has not', () => {
    // "A record argument's fields are resolved only where the record is written" (feature 2.10),
    // so `window.span` is a row of `attn` alone and the intersection is the record's own row.
    expect(multi.rows.map((row) => row.path)).toContain('window');
    expect(multi.rows.map((row) => row.path)).not.toContain('window.span');
    expect(multi.dropped).toBe(1);
    expect(multi.primitive).toBe('attention.dense');
  });
});

describe('two sites of different primitives', () => {
  // `attn_n` is a `norm.rms` and `attn` an `attention.dense`: the intersection is the arguments
  // both declarations happen to carry, which is what §4.11 calls "a common argument".
  const multi = selection(LLAMA, ['decoder/attn_n[layer=0]', 'decoder/attn[layer=0]']);

  it('says the primitive is not one, and keeps what the declarations share', () => {
    expect(multi.primitive).toBeNull();
    expect(multi.rows.map((row) => row.path)).toEqual(['width']);
    expect(multi.dropped).toBeGreaterThan(0);
  });

  it('keeps the families in common and says the rest differ', () => {
    expect(multi.familiesDiffer).toBe(true);
  });
});

describe('a selection of one, and of none', () => {
  it('answers the sheet of one site with every row and nothing differing', () => {
    const multi = selection(LLAMA, ['decoder/attn_n[layer=0]']);
    expect(multi.count).toBe(1);
    expect(multi.rows.every((row) => !row.differs && row.also.length === 0)).toBe(true);
  });

  it('answers nothing for an empty selection', () => {
    expect(multiSheet([]).count).toBe(0);
    expect(multiSheet([]).rows).toEqual([]);
  });
});
