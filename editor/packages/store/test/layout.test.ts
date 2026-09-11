import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

import { loadSchemas, parse, type SchemaRegistry } from '@tensorspine/lang';
import { describe, expect, it } from 'vitest';

import { rename } from '../src/commands.js';
import { DocumentStore } from '../src/document.js';
import {
  emptyLayout,
  LAYOUT_SCHEMA,
  LayoutStore,
  moved,
  prune,
  readLayout,
  resolves,
  writeLayout,
  type Layout,
} from '../src/layout.js';
import { corpusText, corpusTree, editorRoot, shapes } from './source.js';
import { QUANTITY, site } from './selectors.js';

// The sidecar of plan D6 and §5.5: what the canvas knows and the document may not carry. Two
// claims are load-bearing and are what this suite is about — "a sidecar key that names a path the
// document no longer has is dropped with a log line" (D6), and "the sidecar keys follow" a rename
// (§3) — and the third is the project's own rule that every schema ships with its `.md`.

const SCHEMA_FILE = 'schemas/tensorspine-editor-layout.schema.json';
const COMPANION = 'schemas/TENSORSPINE-EDITOR-LAYOUT.md';

function editorFile(path: string): string {
  return readFileSync(join(editorRoot, path), 'utf8');
}

let sidecarRegistry: SchemaRegistry | null = null;

/** The editor's own schemas, as the application loads them beside the language's. */
function sidecarSchemas(): SchemaRegistry {
  sidecarRegistry ??= loadSchemas([{ path: SCHEMA_FILE, text: editorFile(SCHEMA_FILE) }], {
    origin: 'editor/schemas',
  });
  return sidecarRegistry;
}

function conforms(layout: Layout): boolean {
  return sidecarSchemas().conforms(parse(writeLayout(layout)), 'layout');
}

describe('the sidecar’s own schema', () => {
  it('is in the editor’s directory, with the companion note the project’s rule asks for', () => {
    expect(existsSync(join(editorRoot, SCHEMA_FILE))).toBe(true);
    expect(existsSync(join(editorRoot, COMPANION))).toBe(true);
    expect(editorFile(COMPANION).length).toBeGreaterThan(1000);
  });

  it('fixes the `schema` member the store writes', () => {
    const declared = JSON.parse(editorFile(SCHEMA_FILE)) as {
      properties: { schema: { const: string } };
    };
    expect(declared.properties.schema.const).toBe(LAYOUT_SCHEMA);
    expect(LAYOUT_SCHEMA).toBe('tensorspine-editor-layout/1');
  });

  it('admits what the store writes, and refuses what it does not', () => {
    expect(conforms(emptyLayout())).toBe(true);
    const full: Layout = {
      schema: LAYOUT_SCHEMA,
      positions: {
        'instances/embed': { x: 10, y: 20 },
        'compositions/decoder/instances/attn': { x: -5.5, y: 0 },
        'interfaces/inputs/tokens': { x: 0, y: 1 },
      },
      collapsed: ['compositions/decoder'],
      viewport: { x: 0, y: 0, zoom: 1.25 },
      preview_assignment: { layers: 26, eps: 0.00001, mode: 'text', flag: true },
      expanded_view: { filters: { families: ['decoder'], indices: { layer: { from: 0, to: 3 } } } },
    };
    expect(conforms(full)).toBe(true);
    const registry = sidecarSchemas();
    expect(registry.conforms(parse('{"positions": {}, "collapsed": []}'), 'layout')).toBe(false);
    expect(
      registry.conforms(parse('{"schema": "other/1", "positions": {}, "collapsed": []}'), 'layout'),
    ).toBe(false);
    expect(
      registry.conforms(
        parse(`{"schema": "${LAYOUT_SCHEMA}", "positions": {"a": {"x": 1}}, "collapsed": []}`),
        'layout',
      ),
    ).toBe(false);
  });
});

describe('a sidecar key names a place of the document', () => {
  const tree = corpusTree('llama3-8b');

  it('resolves what the document has, and nothing else', () => {
    expect(resolves(tree, 'instances/embed')).toBe(true);
    expect(resolves(tree, 'compositions/decoder/instances/attn')).toBe(true);
    expect(resolves(tree, 'interfaces/inputs/tokens')).toBe(true);
    expect(resolves(tree, 'interfaces/inputs/tokens/to/0')).toBe(true);
    expect(resolves(tree, 'instances/gone')).toBe(false);
    expect(resolves(tree, '')).toBe(false);
  });

  it('is dropped when it no longer resolves, with the line D6 asks for', () => {
    const layout: Layout = {
      schema: LAYOUT_SCHEMA,
      positions: {
        'instances/embed': { x: 1, y: 2 },
        'instances/was_here': { x: 3, y: 4 },
      },
      collapsed: ['compositions/decoder', 'compositions/gone'],
    };
    const { layout: pruned, dropped } = prune(layout, tree);
    expect(Object.keys(pruned.positions)).toEqual(['instances/embed']);
    expect(pruned.collapsed).toEqual(['compositions/decoder']);
    expect(dropped.map((one) => one.key)).toEqual(['instances/was_here', 'compositions/gone']);
    expect(dropped.map((one) => one.where)).toEqual(['positions', 'collapsed']);
    for (const one of dropped) {
      expect(one.message).toContain(one.key);
      expect(one.message).toContain('the document has no such place');
    }
  });

  it('is the same sidecar when nothing was stale', () => {
    const layout: Layout = {
      schema: LAYOUT_SCHEMA,
      positions: { 'instances/embed': { x: 1, y: 2 } },
      collapsed: [],
    };
    const { layout: pruned, dropped } = prune(layout, tree);
    expect(dropped).toEqual([]);
    expect(pruned).toBe(layout);
  });

  it('leaves the preview assignment alone: its keys are quantities, not places', () => {
    const layout: Layout = {
      ...emptyLayout(),
      preview_assignment: { layers: 26, gone_quantity: 1 },
    };
    expect(prune(layout, tree).layout.preview_assignment).toEqual({ layers: 26, gone_quantity: 1 });
  });
});

describe('the keys follow a rename (plan §3)', () => {
  it('carries the place that moved and everything under it', () => {
    const layout: Layout = {
      schema: LAYOUT_SCHEMA,
      positions: {
        'compositions/decoder/instances/attn': { x: 1, y: 1 },
        'compositions/decoder/instances/attn/arguments/heads': { x: 2, y: 2 },
        'compositions/decoder/instances/attn_n': { x: 3, y: 3 },
      },
      collapsed: ['compositions/decoder/instances/attn'],
    };
    const after = moved(layout, [
      {
        from: ['compositions', 'decoder', 'instances', 'attn'],
        to: ['compositions', 'decoder', 'instances', 'attention'],
      },
    ]);
    expect(Object.keys(after.positions)).toEqual([
      'compositions/decoder/instances/attention',
      'compositions/decoder/instances/attention/arguments/heads',
      // A key that only begins with the same letters is not under it.
      'compositions/decoder/instances/attn_n',
    ]);
    expect(after.collapsed).toEqual(['compositions/decoder/instances/attention']);
    expect(moved(layout, [])).toBe(layout);
  });

  it('follows the moves a rename command answers', () => {
    const document = DocumentStore.open(corpusText('llama3-8b'), shapes());
    const canvas = new LayoutStore({
      schema: LAYOUT_SCHEMA,
      positions: { 'compositions/decoder/instances/attn': { x: 10, y: 20 } },
      collapsed: [],
    });
    const applied = document.run((context) =>
      rename(context, {
        path: ['compositions', 'decoder', 'instances', 'attn'],
        to: 'attention',
        references: site('decoder'),
      }),
    );
    canvas.follow(applied.moves);
    expect(canvas.layout.positions).toEqual({
      'compositions/decoder/instances/attention': { x: 10, y: 20 },
    });
    expect(prune(canvas.layout, document.tree).dropped).toEqual([]);
  });
});

describe('the sidecar’s own history (D13)', () => {
  it('records a move, a collapse and a viewport, each under its own name', () => {
    const canvas = new LayoutStore();
    canvas.move(['instances', 'embed'], { x: 1, y: 2 });
    expect(canvas.undoLabel).toBe('Move instances/embed');
    canvas.collapse(['compositions', 'decoder'], true);
    expect(canvas.layout.collapsed).toEqual(['compositions/decoder']);
    canvas.collapse(['compositions', 'decoder'], false);
    expect(canvas.layout.collapsed).toEqual([]);
    canvas.look({ x: 0, y: 0, zoom: 2 });
    expect(canvas.layout.viewport).toEqual({ x: 0, y: 0, zoom: 2 });
    expect(canvas.revision).toBe(4);
  });

  it('undoes a move without touching the document, and a document edit without moving a node', () => {
    // D13: "layout moves are recorded in a separate log so an undo of a semantic edit does not
    // shuffle positions".
    const document = DocumentStore.open(corpusText('llama3-8b'), shapes());
    const text = document.text;
    const canvas = new LayoutStore();
    canvas.move(['instances', 'embed'], { x: 1, y: 2 });

    document.run((context) =>
      rename(context, { path: ['quantities', 'd'], to: 'width', references: QUANTITY }),
    );
    expect(canvas.layout.positions).toEqual({ 'instances/embed': { x: 1, y: 2 } });

    document.undo();
    expect(document.text).toBe(text);
    expect(canvas.layout.positions).toEqual({ 'instances/embed': { x: 1, y: 2 } });

    canvas.move(['instances', 'embed'], { x: 9, y: 9 });
    canvas.undo();
    expect(canvas.layout.positions).toEqual({ 'instances/embed': { x: 1, y: 2 } });
    expect(document.text).toBe(text);
    expect(document.canUndo).toBe(false);
  });

  it('drops the stale keys as an edit of its own, and answers the lines', () => {
    const document = DocumentStore.open(corpusText('llama3-8b'), shapes());
    const canvas = new LayoutStore({
      schema: LAYOUT_SCHEMA,
      positions: { 'instances/embed': { x: 1, y: 2 }, 'instances/was_here': { x: 3, y: 4 } },
      collapsed: [],
    });
    const dropped = canvas.prune(document.tree);
    expect(dropped.map((one) => one.key)).toEqual(['instances/was_here']);
    expect(canvas.layout.positions).toEqual({ 'instances/embed': { x: 1, y: 2 } });
    expect(canvas.undoLabel).toBe('Drop stale layout keys');
    expect(canvas.prune(document.tree)).toEqual([]);
    expect(canvas.revision).toBe(1);
  });

  it('resets every override, which is what “Reset layout” is', () => {
    const canvas = new LayoutStore();
    canvas.move(['instances', 'embed'], { x: 1, y: 2 });
    canvas.collapse(['compositions', 'decoder'], true);
    canvas.reset();
    expect(canvas.layout.positions).toEqual({});
    expect(canvas.layout.collapsed).toEqual(['compositions/decoder']);
  });
});

describe('the sidecar’s file', () => {
  it('is written with two spaces and a trailing newline, and read back as it was', () => {
    const layout: Layout = {
      schema: LAYOUT_SCHEMA,
      positions: { 'instances/embed': { x: 1.5, y: -2 } },
      collapsed: ['compositions/decoder'],
      viewport: { x: 0, y: 0, zoom: 1 },
      preview_assignment: { layers: 26 },
    };
    const text = writeLayout(layout);
    expect(text.endsWith('\n')).toBe(true);
    expect(text).toContain('\n  "positions": {');
    expect(readLayout(text)).toEqual(layout);
  });

  it('reads a file another version wrote for what it can, rather than refusing it', () => {
    // A sidecar is a convenience: a file the editor cannot fully read must not stop a document
    // from opening. What the schema judges is a separate question, and the caller asks it.
    const read = readLayout(
      '{"schema": "tensorspine-editor-layout/1", "positions": {"a/b": {"x": 1, "y": 2}, "bad": 3},' +
        ' "collapsed": ["c/d", 7], "viewport": {"x": 0}, "extra": true}',
    );
    expect(read.positions).toEqual({ 'a/b': { x: 1, y: 2 } });
    expect(read.collapsed).toEqual(['c/d']);
    expect(read.viewport).toBeUndefined();
    expect(readLayout('[]')).toEqual(emptyLayout());
  });
});
