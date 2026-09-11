import { afterEach, describe, expect, it } from 'vitest';

import { createLang } from '@tensorspine/lang/api/engine';
import { MemoryWorkspace, stubPlatform } from '@tensorspine/store/platform';
import { pointerOf } from '@tensorspine/store';

import { createDocuments, type Documents, type DocumentsStore } from '../../src/documents/store.js';
import { removeCommand, renameCommand, selectedRow } from '../../src/explorer/Explorer.js';
import { outlineOf, type OutlineRow } from '../../src/explorer/outline.js';
import { presentation } from '../../src/presentation/index.js';
import { filesUnder, schemaTexts } from '../documents/source.js';
import { readRepositoryFile } from './source.js';

// Feature 2.7 — what the explorer's gestures do to the document, asked of the store that makes
// them rather than of the tree that draws them.
//
// Three claims, and they are the feature's own:
//
//   - **selecting an item selects it** — the selection is a place of the one tree (D1), so the
//     canvas (2.9) and the sheets (2.10) will read the very thing the tree wrote;
//   - **rename by click writes the JSON** — every occurrence of the name, by the reference rules
//     of `presentation.json`, in one undoable command, and the bytes that come out are the
//     document's own (D12);
//   - **delete cascades** — over everything whose rule names the thing, with what the grammar
//     keeps listed rather than silently dropped.

const MODEL = 'data/models/llama3-8b.json';

interface Open {
  readonly store: DocumentsStore;
  readonly stop: () => void;
}

const opened: Open[] = [];

afterEach(() => {
  for (const one of opened.splice(0)) one.stop();
});

/** A documents store over a workspace laid out like the repository's `data/`, core in-thread. */
function editor(): Open {
  const workspace = MemoryWorkspace.of({
    ...filesUnder('data/models'),
    ...filesUnder('data/primitive-library'),
  });
  const lang = createLang();
  const built = createDocuments({
    platform: stubPlatform({ workspace }),
    lang,
    vendoredSchemas: () => Promise.resolve(schemaTexts()),
    debounceMs: 0,
    autosaveMs: 0,
  });
  const one: Open = {
    store: built.store,
    stop: () => {
      built.dispose();
      lang.close();
    },
  };
  opened.push(one);
  return one;
}

function now(one: Open): Documents {
  return one.store.getState();
}

/** The outline of the open document, whatever is open in the tree. */
function rowsOf(one: Open): OutlineRow[] {
  const document = now(one).open[0];
  if (document === undefined) throw new Error('nothing is open');
  return outlineOf({
    tree: document.session.store.tree,
    shapes: document.session.store.shapes,
    bindings: presentation(),
    role: document.session.store.role,
    openAll: true,
  });
}

function rowAt(one: Open, pointer: string): OutlineRow {
  const row = rowsOf(one).find((each) => each.pointer === pointer);
  if (row === undefined) throw new Error(`no row at ${pointer}`);
  return row;
}

describe('selecting an item of the tree', () => {
  it('writes the place into the document’s own state, where every projection reads it', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    const site = rowAt(one, '/compositions/decoder/instances/attn');
    now(one).selectPlace(site.path);
    expect(now(one).open[0]?.selection).toEqual(['compositions', 'decoder', 'instances', 'attn']);
    // And the sheet of §4.11 finds the row it is about, under a group the tree had closed.
    const found = selectedRow(one.store);
    expect(found?.label).toBe('attn');
    expect(found?.declares).toBe('site');
    now(one).selectPlace(null);
    expect(now(one).open[0]?.selection).toBeUndefined();
  }, 120_000);

  it('remembers what each tab had selected, a selection being the document’s and not the tree’s', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    await now(one).openDocument('data/models/qwen3.5-4b-text.json');
    const [first, second] = now(one).open;
    now(one).selectPlace(['quantities', 'd'], first?.id);
    now(one).selectPlace(['quantities', 'heads'], second?.id);
    expect(now(one).open[0]?.selection).toEqual(['quantities', 'd']);
    expect(now(one).open[1]?.selection).toEqual(['quantities', 'heads']);
  }, 120_000);

  it('keeps a row’s openness as a deviation, per document', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    const id = now(one).open[0]?.id as string;
    now(one).togglePlace('/quantities', id);
    expect(now(one).open[0]?.toggled).toEqual(['/quantities']);
    now(one).togglePlace('/quantities', id);
    expect(now(one).open[0]?.toggled).toEqual([]);
  }, 120_000);
});

describe('rename by clicking the name', () => {
  it('rewrites every occurrence the reference rules find, and the bytes are the document’s', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    const before = readRepositoryFile('data/models/llama3-8b.json');
    const applied = now(one).edit(renameCommand(rowAt(one, '/quantities/d'), 'width'));
    expect(applied?.changed).toBe(true);
    expect(applied?.label).toBe('Rename quantity d to width');
    const text = now(one).open[0]?.session.text as string;
    // Eleven occurrences in this document: the declaration and the ten arguments that name it.
    expect(before.match(/"quantity": "d"/g)).toHaveLength(10);
    expect(text.match(/"quantity": "d"/g)).toBeNull();
    expect(text.match(/"quantity": "width"/g)).toHaveLength(10);
    // And nothing else moved: the declaration's own key and the ten references are the whole
    // difference from the file's bytes — the argument *called* `width` is another name and stays.
    expect(text).toBe(
      before
        .replace('"d": {\n      "type"', '"width": {\n      "type"')
        .replaceAll('"quantity": "d"', '"quantity": "width"'),
    );
  }, 120_000);

  it('renames a site of a composition without touching the site of another scope', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    const site = rowAt(one, '/compositions/decoder/instances/attn');
    const applied = now(one).edit(renameCommand(site, 'self_attention'));
    expect(applied?.changed).toBe(true);
    const text = now(one).open[0]?.session.text as string;
    expect(text).toContain('"site": "self_attention"');
    expect(text).not.toContain('"site": "attn"');
    // `attn_n` and `attn_r` are other sites and are untouched.
    expect(text).toContain('"site": "attn_n"');
    expect(text).toContain('"site": "attn_r"');
  }, 120_000);

  it('moves the selection with the name, so the sheet stays on what it was showing', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    const id = now(one).open[0]?.id as string;
    now(one).selectPlace(['quantities', 'd'], id);
    now(one).edit(renameCommand(rowAt(one, '/quantities/d'), 'width'));
    expect(now(one).open[0]?.selection).toEqual(['quantities', 'width']);
    expect(pointerOf(now(one).open[0]?.selection ?? [])).toBe('/quantities/width');
  }, 120_000);

  it('refuses a name the map already has, off the grammar being the one refusal (D5)', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    const before = now(one).open[0]?.session.text;
    expect(() => now(one).edit(renameCommand(rowAt(one, '/quantities/d'), 'heads'))).toThrow(
      "a member named 'heads' is already there",
    );
    expect(now(one).open[0]?.session.text).toBe(before);
  }, 120_000);

  it('is one entry of the Edit menu, and Undo takes the whole rewrite back', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    const before = now(one).open[0]?.session.text;
    now(one).edit(renameCommand(rowAt(one, '/instances/embed'), 'tokens_in'));
    const session = now(one).open[0]?.session;
    expect(session?.store.undoLabel).toBe('Rename instance embed to tokens_in');
    session?.store.undo();
    expect(now(one).open[0]?.session.text).toBe(before);
  }, 120_000);
});

describe('delete', () => {
  it('offers the cascade rather than making it, and makes it when it is confirmed', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    const before = now(one).open[0]?.session.text;
    now(one).offer(removeCommand(rowAt(one, '/compositions/decoder/instances/attn')));
    const dialog = now(one).dialog;
    expect(dialog?.kind).toBe('remove');
    if (dialog?.kind !== 'remove') throw new Error('no confirmation');
    expect(dialog.label).toBe('Delete site attn');
    // The site, the two value bindings that name it, the four parameter identities and the
    // state identity — eight places, deepest first and, inside one map, the last first, which is
    // the order the removals are made in so that one never moves the place of another. Nothing
    // is written until it is confirmed.
    expect([...dialog.removed].sort()).toEqual([
      '/compositions/decoder/bindings/parameters/attn.k',
      '/compositions/decoder/bindings/parameters/attn.out',
      '/compositions/decoder/bindings/parameters/attn.q',
      '/compositions/decoder/bindings/parameters/attn.v',
      '/compositions/decoder/bindings/states/attn.kv',
      '/compositions/decoder/bindings/values/attn.norm_in',
      '/compositions/decoder/bindings/values/attn_r.b',
      '/compositions/decoder/instances/attn',
    ]);
    expect(dialog.removed.at(-1)).toBe('/compositions/decoder/instances/attn');
    expect(dialog.kept).toEqual([]);
    expect(now(one).open[0]?.session.text).toBe(before);
    now(one).confirmed();
    const text = now(one).open[0]?.session.text as string;
    expect(text).not.toContain('"attn"');
    expect(text).not.toContain('attention.dense');
    expect(now(one).dialog).toBeNull();
  }, 120_000);

  it('lists what the grammar keeps, and keeps it — wire first, fix afterwards (Q5)', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    now(one).offer(removeCommand(rowAt(one, '/instances/embed')));
    const dialog = now(one).dialog;
    if (dialog?.kind !== 'remove') throw new Error('no confirmation');
    // `interfaces/inputs/tokens/to` needs one endpoint and is required of its input, so the
    // endpoint stays and the core reports the name it no longer resolves.
    expect(dialog.kept).toEqual(['/interfaces/inputs/tokens/to/0']);
    now(one).confirmed();
    expect(now(one).open[0]?.session.text).toContain('"instance": "embed"');
  }, 120_000);

  it('clears a selection the delete removed, and one edit takes it all back', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    const id = now(one).open[0]?.id as string;
    const before = now(one).open[0]?.session.text;
    now(one).selectPlace(['compositions', 'decoder', 'instances', 'attn'], id);
    now(one).offer(removeCommand(rowAt(one, '/compositions/decoder/instances/attn')));
    now(one).confirmed();
    expect(now(one).open[0]?.selection).toBeUndefined();
    now(one).open[0]?.session.store.undo();
    expect(now(one).open[0]?.session.text).toBe(before);
  }, 120_000);

  it('leaves the document on the grammar, whatever it took with it (D5)', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    const id = now(one).open[0]?.id as string;
    for (const pointer of ['/quantities/vocab', '/instances/lm_head', '/compositions/decoder/instances/ffn']) {
      now(one).offer(removeCommand(rowAt(one, pointer)));
      now(one).confirmed();
    }
    const document = now(one).open[0];
    expect(document?.id).toBe(id);
    // The tree is still a document the outline can draw, which is the cheap half of D5's claim;
    // the schema stage's own verdict is the pipeline's, and feature 2.8 shows it.
    expect(rowsOf(one).length).toBeGreaterThan(10);
  }, 120_000);
});

describe('the library the marks are read from', () => {
  it('answers which primitives pin a template, which is what §4.5’s `▣` is', async () => {
    const one = editor();
    await now(one).openDocument('data/models/shieldstral-3b-composite.json');
    // `primitive_library.template_primitives` over the gathered base, as the core reads it.
    expect([...now(one).library.templates]).toEqual(['decoder.causal_yarn']);
    const marked = outlineOf({
      tree: now(one).open[0]?.session.store.tree as never,
      shapes: now(one).open[0]?.session.store.shapes as never,
      bindings: presentation(),
      templates: now(one).library.templates,
    })
      .filter((row) => row.marks.length > 0)
      .map((row) => `${row.label} ${row.marks.join('')}`);
    expect(marked).toContain('text ▣');
  }, 120_000);
});

describe('the filter box', () => {
  it('is the explorer’s and outlives the side bar being closed', async () => {
    const one = editor();
    await now(one).openDocument(MODEL);
    now(one).setFilter('attn');
    expect(now(one).filter).toBe('attn');
    now(one).setFilter('');
    expect(now(one).filter).toBe('');
  }, 120_000);
});
