import { readdirSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { nodeAt, pathOfKey, pointerOf } from '@tensorspine/store';

import {
  COMPUTED,
  GUARDED,
  outlineOf,
  SHARED,
  TEMPLATE,
  type OutlineRow,
} from '../../src/explorer/outline.js';
import { bindings, corpus, linesOf, outline, repositoryRoot, shapes } from './source.js';

// Feature 2.7 — the model explorer's outline (§4.5), as the model behind the tree.
//
// The feature's own test is "the tree for llama3-8b matches S1's structure and counts", and this
// is where that is asked exhaustively: the rows, their order, their depth, their counts, their
// figures and their marks, over the repository's own corpus. The browser layer
// (`apps/web/e2e/explorer.spec.ts`) asks what only a browser can answer — that clicking a row
// selects it, that clicking a name renames the JSON, that the axe pass finds nothing.

/** Every corpus document, by the name the suites read it under. */
function corpusNames(): string[] {
  const found: string[] = [];
  const walk = (at: string, prefix: string): void => {
    for (const entry of readdirSync(join(repositoryRoot, at), { withFileTypes: true })) {
      if (entry.isDirectory()) walk(`${at}/${entry.name}`, `${prefix}${entry.name}/`);
      else if (entry.name.endsWith('.json')) found.push(`${prefix}${entry.name.slice(0, -5)}`);
    }
  };
  walk('data/models', '');
  return found.sort();
}

describe('the outline of llama3-8b', () => {
  // S1's own left panel, row for row. The three departures from the board are stated where the
  // module states them: the label of `primitive_libraries` is the schema's member name and not
  // the board's shorter *Libraries*; a composition's index is a row of its own rather than a
  // note in the tail; and the figure a document computes is marked `computed` rather than
  // `derived`, which is a value of `quantity_source`'s own enumeration and so may not be typed
  // into interface source (§1 (b)).
  const rows = outline('llama3-8b');

  it('draws S1’s rows, in S1’s order, at S1’s depths', () => {
    expect(linesOf(rows)).toEqual([
      '▾ llama3-8b · tensorspine/2.0',
      '  ▸ Primitive libraries (1) · ../primitive-library/',
      '  ▾ Quantities (9)',
      '      d 4096',
      '      ffn 14336',
      '      heads 32',
      '      kv_heads 8',
      '      head_dim 128 computed',
      '      layers 32',
      '      vocab 128256',
      '      eps 1e-05',
      '      precision bf16',
      '  ▸ Constants (0)',
      '  ▾ Instances (3)',
      '      embed · embed',
      '      final_n · norm.rms',
      '      lm_head · lm_head',
      '  ▾ Compositions (1)',
      '    ▾ decoder',
      '        indices 1 · instances 6 · values 8 · parameters 9 · states 1',
      '        layer',
      '        attn_n · norm.rms',
      '        attn · attention.dense',
      '        attn_r · residual.add',
      '        ffn_n · norm.rms',
      '        ffn · ffn.gated',
      '        ffn_r · residual.add',
      '      ▸ Bindings · values 8 · parameters 9 · states 1',
      '  ▸ Bindings · values 4 · parameters 3 · constants 0 · states 0',
      '  ▸ Interfaces · inputs 1 · outputs 1',
    ]);
  });

  it('opens what S1 opens and closes what S1 closes, with nothing said about any of them', () => {
    const open = (pointer: string): boolean =>
      rows.find((row) => row.pointer === pointer)?.open ?? false;
    // A non-empty map of the document's own names is open; a list, an empty map and an object of
    // maps are closed. Which is the state the board draws, and it is read from the schema.
    expect(open('/quantities')).toBe(true);
    expect(open('/instances')).toBe(true);
    expect(open('/compositions')).toBe(true);
    expect(open('/compositions/decoder')).toBe(true);
    expect(open('/primitive_libraries')).toBe(false);
    expect(open('/constants')).toBe(false);
    expect(open('/bindings')).toBe(false);
    expect(open('/interfaces')).toBe(false);
  });

  it('counts what the document holds, member for member', () => {
    const count = (pointer: string): number | undefined =>
      rows.find((row) => row.pointer === pointer)?.count;
    expect(count('/quantities')).toBe(9);
    expect(count('/constants')).toBe(0);
    expect(count('/instances')).toBe(3);
    expect(count('/compositions')).toBe(1);
    expect(count('/primitive_libraries')).toBe(1);
    // Bindings and Interfaces hold maps rather than names of their own, so they carry no count
    // and say what each of their maps holds — which is what S1 writes in their tail.
    expect(count('/bindings')).toBeUndefined();
    expect(count('/interfaces')).toBeUndefined();
  });

  it('says what a quantity resolves to, in the document’s own lexeme, and marks what follows', () => {
    const quantity = (name: string): OutlineRow | undefined =>
      rows.find((row) => row.pointer === `/quantities/${name}`);
    expect(quantity('d')?.figure).toBe('4096');
    // Python's repr, which is what the file writes and what `view.py` prints — not `0.00001`.
    expect(quantity('eps')?.figure).toBe('1e-05');
    expect(quantity('precision')?.figure).toBe('bf16');
    // A literal that declares how it follows from the others (V11 checks the two agree): the
    // value is the document's and the mark says it is not freely chosen.
    expect(quantity('head_dim')?.figure).toBe('128');
    expect(quantity('head_dim')?.marks).toEqual([COMPUTED]);
    expect(quantity('d')?.marks).toEqual([]);
  });

  it('tints a swatch by what the row holds, never by what it is called', () => {
    const tint = (pointer: string): string | undefined =>
      rows.find((row) => row.pointer === pointer)?.tint;
    expect(tint('/compositions/decoder')).toBe('group');
    expect(tint('/quantities/d')).toBe('figure');
    expect(tint('/instances/embed')).toBeUndefined();
  });

  it('carries the reference rules a rename and a delete are computed from', () => {
    const site = rows.find((row) => row.pointer === '/compositions/decoder/instances/attn');
    expect(site?.declares).toBe('site');
    expect(site?.scope).toEqual({ path: ['compositions', 'decoder'], name: 'decoder' });
    expect(site?.declaredAt).toBe(
      'https://tensorspine.dev/schema/2.0/model.json#/$defs/composition_definition/properties/instances',
    );
    const quantity = rows.find((row) => row.pointer === '/quantities/d');
    expect(quantity?.declares).toBe('quantity');
    expect(quantity?.scope).toBeUndefined();
  });
});

describe('the outline over the corpus', () => {
  const names = corpusNames();

  it('reads the fifteen documents the repository carries', () => {
    expect(names.length).toBeGreaterThanOrEqual(15);
  });

  it('names a place of the document in every row, and the document’s own name at the root', () => {
    for (const name of names) {
      const tree = corpus(name);
      // Everything open, so that every row the tree can draw is walked at least once.
      const rows = outlineOf({ tree, shapes: shapes(), bindings: bindings(), openAll: true });
      expect(rows.length, name).toBeGreaterThan(1);
      for (const row of rows) {
        if (row.kind === 'note') continue;
        expect(nodeAt(tree, row.path), `${name} ${row.pointer}`).toBeDefined();
        expect(pointerOf(row.path), name).toBe(row.pointer);
        if (row.named) {
          expect(String(row.path[row.path.length - 1]), `${name} ${row.pointer}`).toBe(row.label);
        }
      }
      const root = rows[0] as OutlineRow;
      expect(root.kind, name).toBe('document');
      expect(root.tail, name).toContain('tensorspine/2.0');
    }
  });

  it('counts a container’s own entries, whatever the document', () => {
    for (const name of names) {
      const tree = corpus(name);
      for (const row of outlineOf({ tree, shapes: shapes(), bindings: bindings(), openAll: true })) {
        if (row.count === undefined) continue;
        const node = nodeAt(tree, row.path);
        const held = Array.isArray(node)
          ? node.length
          : typeof node === 'object' && node !== null && 'members' in node
            ? node.members.length
            : -1;
        expect(held, `${name} ${row.pointer}`).toBe(row.count);
      }
    }
  });
});

describe('what a row says about itself', () => {
  it('marks a guarded site with the mark §4.5 names, from the editor bound to a condition', () => {
    const rows = outline('gemma3n-kvshare');
    const guarded = rows
      .filter((row) => row.marks.includes(GUARDED))
      .map((row) => String(row.path[row.path.length - 1]));
    expect(guarded).toEqual(['attn', 'ffn_sparse', 'attn_full', 'ffn']);
  });

  it('marks an identity several members share, and leaves one member unmarked', () => {
    const rows = outline('gemma3n-kvshare', { toggled: new Set(['/bindings']) });
    const shared = rows
      .filter((row) => row.marks.includes(SHARED))
      .map((row) => String(row.path[row.path.length - 1]));
    expect(shared).toEqual(['tied_embeddings', 'shared.sliding.kv', 'shared.full.kv']);
    // Two families are not a membership, and one member is not a sharing.
    expect(rows.find((row) => row.pointer === '/instances/embed')?.marks).toEqual([]);
    expect(
      rows.find((row) => row.pointer === '/bindings/parameters/expand.projection')?.marks,
    ).toEqual([]);
  });

  it('marks an instance of a template primitive, from the names the core answers', () => {
    // `shieldstral-3b-composite` instantiates the `decoder.causal_yarn` template (§0's own
    // acceptance case). Which primitives pin a template is the *library*'s answer, handed in;
    // with none, nothing is marked, which is what a document whose bases are not gathered shows.
    const templates = new Set(['decoder.causal_yarn']);
    const marked = outline('shieldstral-3b-composite', { templates })
      .filter((row) => row.marks.includes(TEMPLATE))
      .map((row) => row.label);
    expect(marked).toEqual(['text']);
    expect(
      outline('shieldstral-3b-composite').some((row) => row.marks.includes(TEMPLATE)),
    ).toBe(false);
    // And no instance of an ordinary primitive is marked by a set that names one.
    expect(
      outline('llama3-8b', { templates: new Set(['norm.rms']) })
        .filter((row) => row.marks.includes(TEMPLATE))
        .map((row) => row.label),
    ).toEqual(['final_n', 'attn_n', 'ffn_n']);
  });

  it('leaves a template’s external quantities without a figure, having nothing to resolve them', () => {
    const rows = outline('decoder-causal-yarn/1.0.0');
    const external = rows.filter((row) => row.pointer.startsWith('/quantities/'));
    expect(external.length).toBe(8);
    expect(external.every((row) => row.figure === undefined)).toBe(true);
    // A template carries its version beside the tag its schema fixes (§4.3's badge reads the
    // same member).
    expect(rows[0]?.tail).toBe('tensorspine/2.0 · 1.0.0');
  });

  it('carries the red dot of §4.5 on the row a problem names and on every row above it', () => {
    const rows = outline('llama3-8b', {
      problems: ['/compositions/decoder/instances/attn/arguments/heads'],
    });
    const dotted = rows.filter((row) => row.problem).map((row) => row.pointer);
    expect(dotted).toEqual([
      '',
      '/compositions',
      '/compositions/decoder',
      '/compositions/decoder/instances/attn',
    ]);
  });
});

describe('the filter box and the toggles', () => {
  it('keeps what matches and everything above it, and nothing else', () => {
    const rows = outline('llama3-8b', { filter: 'attn' });
    expect(linesOf(rows)).toEqual([
      '▾ llama3-8b · tensorspine/2.0',
      '  ▾ Compositions (1)',
      '    ▾ decoder',
      '        attn_n · norm.rms',
      '        attn · attention.dense',
      '        attn_r · residual.add',
      // A filter opens what the default leaves closed, which is how a name inside the
      // composition's own bindings is found at all.
      '      ▾ Bindings',
      '        ▾ Values (8)',
      '            attn_n.carry ⚑',
      '            attn_r.a_carry ⚑',
      '            attn.norm_in',
      '            attn_r.b',
      '        ▾ Parameters (9)',
      '            attn_n.weight',
      '            attn.q',
      '            attn.k',
      '            attn.v',
      '            attn.out',
      '        ▾ States (1)',
      '            attn.kv',
    ]);
  });

  it('finds a name inside a composition that the default would have left closed', () => {
    const rows = outline('gemma3n-kvshare', { filter: 'laurel' });
    expect(rows.some((row) => row.pointer === '/compositions/decoder/instances/laurel')).toBe(true);
    expect(rows.some((row) => row.pointer === '/quantities/laurel_rank')).toBe(true);
  });

  it('reads a toggle as a deviation from the default, in both directions', () => {
    const closed = outline('llama3-8b', { toggled: new Set(['/quantities']) });
    expect(closed.some((row) => row.pointer === '/quantities/d')).toBe(false);
    expect(closed.find((row) => row.pointer === '/quantities')?.tail).toBe(
      'd 4096 · ffn 14336 · heads 32 · …',
    );
    const opened = outline('llama3-8b', { toggled: new Set(['/constants', '/interfaces']) });
    expect(opened.find((row) => row.pointer === '/interfaces')?.open).toBe(true);
    expect(opened.some((row) => row.pointer === '/interfaces/inputs/tokens')).toBe(true);
  });

  it('keys a toggle by a place of the document, so the sidecar’s keys and a pointer agree', () => {
    for (const row of outline('llama3-8b')) {
      if (row.kind === 'note' || row.path.length === 0) continue;
      expect(pathOfKey(row.path.map((step) => String(step)).join('/'))).toEqual(row.path);
    }
  });
});
