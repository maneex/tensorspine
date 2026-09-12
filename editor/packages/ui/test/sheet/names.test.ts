import { describe, expect, it } from 'vitest';

import type { Path } from '@tensorspine/store';

import { outlineOf } from '../../src/explorer/outline.js';
import { presentation } from '../../src/presentation/index.js';
import { namesFor } from '../../src/sheet/names.js';
import { MODEL, context, sheetOf, siteOf, tree } from './source.js';

/**
 * What the quantity and index selects offer — plan §4.12's two referring modes.
 *
 * > **quantity** | a select of the document's quantities **whose type kind matches the argument's**
 * > … **index** | a select of the indices **in scope** (inside a composition only)
 *
 * The feature's own block names the first case: "quantity mode lists only cardinality quantities
 * for `heads`". `llama3-8b` declares nine quantities — seven cardinalities, one real (`eps`) and
 * one enum (`precision`) — so the list is a real filter and not an empty claim.
 */

const LLAMA = 'llama3-8b';
const ATTN = 'decoder/attn[layer=0]';

function outline(name: string): ReturnType<typeof outlineOf> {
  return outlineOf({
    tree: tree(name),
    shapes: context.shapes,
    bindings: presentation(),
    role: MODEL,
    openAll: true,
  });
}

function rowAt(name: string, where: string, path: string): { at: Path; kind: string } {
  const row = sheetOf(name, where).rows.find((one) => one.path === path);
  if (row === undefined) throw new Error(`no row ${path}`);
  return { at: row.at, kind: row.kind };
}

describe('the quantity mode of a cardinality argument', () => {
  it('lists only the cardinality quantities, and every one of them', () => {
    const row = rowAt(LLAMA, ATTN, 'heads');
    const names = namesFor({
      outline: outline(LLAMA),
      referent: 'quantity',
      at: row.at,
      kind: row.kind,
      tree: tree(LLAMA),
      role: MODEL,
      context,
      shapes: context.shapes,
    }).map((one) => one.name);
    expect(names).toEqual(['d', 'ffn', 'heads', 'kv_heads', 'head_dim', 'layers', 'vocab']);
    expect(names).not.toContain('eps');
    expect(names).not.toContain('precision');
  });

  it('lists the real quantity for a real argument and nothing else', () => {
    const row = rowAt(LLAMA, 'final_n', 'eps');
    const names = namesFor({
      outline: outline(LLAMA),
      referent: 'quantity',
      at: row.at,
      kind: row.kind,
      tree: tree(LLAMA),
      role: MODEL,
      context,
      shapes: context.shapes,
    }).map((one) => one.name);
    expect(names).toEqual(['eps']);
  });

  it('lists every quantity when the row asks for no kind at all', () => {
    const row = rowAt(LLAMA, ATTN, 'heads');
    const names = namesFor({
      outline: outline(LLAMA),
      referent: 'quantity',
      at: row.at,
      tree: tree(LLAMA),
      role: MODEL,
      context,
      shapes: context.shapes,
    }).map((one) => one.name);
    expect(names).toHaveLength(9);
  });
});

describe('the index mode', () => {
  it('offers a composition’s index to a site inside it', () => {
    const row = rowAt(LLAMA, ATTN, 'heads');
    const found = namesFor({
      outline: outline(LLAMA),
      referent: 'index',
      at: row.at,
      tree: tree(LLAMA),
      role: MODEL,
      context,
      shapes: context.shapes,
    });
    expect(found.map((one) => one.name)).toEqual(['layer']);
    expect(found[0]?.scope).toBe('decoder');
  });

  it('offers none to a root instance — §4.12’s "inside a composition only"', () => {
    const row = rowAt(LLAMA, 'final_n', 'width');
    expect(
      namesFor({
        outline: outline(LLAMA),
        referent: 'index',
        at: row.at,
        tree: tree(LLAMA),
        role: MODEL,
        context,
        shapes: context.shapes,
      }),
    ).toEqual([]);
  });
});

describe('what the sheet knows of a row’s kind', () => {
  it('is the declaration’s own word, which the artifact cannot carry', () => {
    // A cardinality and a whole-number physical are both `{"type": "integer"}` in the generated
    // schema; the two are told apart only by the declaration, which is why `describe` answers it.
    expect(rowAt(LLAMA, ATTN, 'heads').kind).toBe('cardinality');
    expect(rowAt('voxtral-realtime', 'decoder/attn[layer=0]', 'window.span').kind).toBe('physical');
    expect(siteOf(LLAMA, ATTN).arguments.facts.find((one) => one.path === 'mask')?.kind).toBe('enum');
  });
});
