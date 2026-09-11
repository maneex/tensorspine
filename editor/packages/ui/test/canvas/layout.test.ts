import { describe, expect, it } from 'vitest';

import { emptyLayout, LayoutStore } from '@tensorspine/store';

import {
  collapsedGroups,
  expandedFromSidecar,
  fit,
  hasManualMoves,
  layoutKey,
  place,
  routes,
} from '../../src/canvas/layout.js';
import { CONTEXT_MENU, entriesFor } from '../../src/canvas/menu.js';
import { ROLE } from '../../src/canvas/model.js';
import { commandById } from '../../src/shell/commands.js';
import { everyComposition, modelOf, reading } from './source.js';

/**
 * Where the boxes go, and what a manual move does to that — plan §4.7, D6.
 *
 * The layout is ELK's (feature 0.4's module, unchanged); what is asserted here is the *editor's*
 * half of it: that a box the sidecar carries a position for is put there instead, that nothing is
 * written until the author moves something, and that a wire runs from the box it leaves to the box
 * it enters whatever either of them has been dragged to.
 */

describe('placing the folded canvas', () => {
  it('places every box of llama3-8b, top to bottom, inside one drawing', async () => {
    const model = modelOf('llama3-8b');
    const placement = await place({ model });
    expect(placement.boxes.size).toBe(model.boxes.length);
    expect(placement.width).toBeGreaterThan(0);
    expect(placement.height).toBeGreaterThan(0);
    for (const box of placement.boxes.values()) {
      expect(box.x).toBeGreaterThanOrEqual(0);
      expect(box.y).toBeGreaterThanOrEqual(0);
      expect(box.manual).toBe(false);
    }
  });

  it('reads the document top to bottom: the input first, the output last', async () => {
    const model = modelOf('llama3-8b');
    const placement = await place({ model });
    const tokens = placement.boxes.get('/interfaces/inputs/tokens');
    const logits = placement.boxes.get('/interfaces/outputs/logits');
    const decoder = placement.boxes.get('/compositions/decoder');
    expect(tokens).toBeDefined();
    expect(logits).toBeDefined();
    expect((tokens?.y ?? 0) < (decoder?.y ?? 0)).toBe(true);
    expect((decoder?.y ?? 0) < (logits?.y ?? 0)).toBe(true);
  });

  it('puts a composition’s sites inside its own box when it is open', async () => {
    const model = modelOf('llama3-8b', { collapsed: new Set() });
    const placement = await place({ model });
    const group = placement.boxes.get('/compositions/decoder');
    expect(group).toBeDefined();
    for (const box of model.boxes) {
      if (box.parent !== '/compositions/decoder') continue;
      const placed = placement.boxes.get(box.pointer);
      expect(placed).toBeDefined();
      expect(placed!.x).toBeGreaterThanOrEqual(group!.x);
      expect(placed!.y).toBeGreaterThanOrEqual(group!.y);
      expect(placed!.x + placed!.width).toBeLessThanOrEqual(group!.x + group!.width + 1);
      expect(placed!.y + placed!.height).toBeLessThanOrEqual(group!.y + group!.height + 1);
    }
  });

  it('puts a box the sidecar moved where the sidecar says, and says it was manual (D6)', async () => {
    const model = modelOf('llama3-8b');
    const placement = await place({
      model,
      positions: { 'instances/embed': { x: 720, y: 40 } },
    });
    const embed = placement.boxes.get('/instances/embed');
    expect(embed?.x).toBe(720);
    expect(embed?.y).toBe(40);
    expect(embed?.manual).toBe(true);
    expect(placement.boxes.get('/instances/lm_head')?.manual).toBe(false);
  });

  it('keys the sidecar by the document’s own path, as D6 writes one', () => {
    expect(layoutKey('/compositions/decoder/instances/attn')).toBe(
      'compositions/decoder/instances/attn',
    );
  });

  it('takes the size the browser measured over the size the model estimated', async () => {
    const model = modelOf('llama3-8b');
    const placement = await place({
      model,
      measured: new Map([['/instances/embed', { width: 400, height: 200 }]]),
    });
    expect(placement.boxes.get('/instances/embed')?.width).toBe(400);
    expect(placement.boxes.get('/instances/embed')?.height).toBe(200);
  });

  it('ignores a measurement of nothing, which is what a headless DOM answers', async () => {
    const model = modelOf('llama3-8b');
    const placement = await place({
      model,
      measured: new Map([['/instances/embed', { width: 0, height: 0 }]]),
    });
    expect(placement.boxes.get('/instances/embed')?.width).toBeGreaterThan(0);
  });

  it('lays the largest folded canvas out well inside the budget (feature 0.4: 8–35 ms)', async () => {
    const model = modelOf('deepseek-v4-pro');
    const placement = await place({ model });
    expect(placement.boxes.size).toBeGreaterThan(20);
    // Feature 1.11's rule for a budget a suite runs beside ninety files: a regression guard at
    // three times it. The folded canvas is the one thing §4.7 lays out on an author's own path.
    expect(placement.milliseconds).toBeLessThan(600);
  });
});

describe('the wires of a placement', () => {
  it('runs each from the box it leaves to the box it enters, top to bottom (S1)', async () => {
    const model = modelOf('llama3-8b');
    const placement = await place({ model });
    const drawn = routes(model, placement);
    expect(drawn.length).toBe(model.wires.length);
    for (const route of drawn) {
      const wire = model.wires.find((one) => one.id === route.id);
      const from = placement.boxes.get(wire?.from ?? '');
      const to = placement.boxes.get(wire?.to ?? '');
      expect(route.from.y).toBe((from?.y ?? 0) + (from?.height ?? 0));
      expect(route.to.y).toBe(to?.y ?? 0);
      expect(route.d.startsWith('M')).toBe(true);
    }
  });

  it('fans several wires leaving one box across its width, as S1 draws them', async () => {
    const model = modelOf('llama3-8b');
    const placement = await place({ model });
    const drawn = routes(model, placement);
    const fromEmbed = drawn.filter((route) => {
      const wire = model.wires.find((one) => one.id === route.id);
      return wire?.from === '/instances/embed';
    });
    expect(fromEmbed.length).toBe(2);
    expect(fromEmbed[0]?.from.x).not.toBe(fromEmbed[1]?.from.x);
  });

  it('follows a box the author moved rather than a route ELK computed', async () => {
    const model = modelOf('llama3-8b');
    const placement = await place({ model, positions: { 'instances/embed': { x: 900, y: 900 } } });
    const drawn = routes(model, placement);
    const entry = drawn.find((route) => {
      const wire = model.wires.find((one) => one.id === route.id);
      return wire?.label === 'decoder.entry';
    });
    expect(entry?.from.x).toBeGreaterThan(900);
  });
});

describe('Zoom to Fit', () => {
  it('shows the whole drawing, centred, and never magnifies it', async () => {
    const model = modelOf('llama3-8b');
    const placement = await place({ model });
    const wide = fit(placement, { width: 10_000, height: 10_000 });
    expect(wide.zoom).toBe(1);
    const narrow = fit(placement, { width: 300, height: 300 });
    expect(narrow.zoom).toBeLessThan(1);
    expect(placement.width * narrow.zoom).toBeLessThanOrEqual(300);
    expect(placement.height * narrow.zoom).toBeLessThanOrEqual(300);
  });

  it('answers something drawable for a drawing of nothing', () => {
    expect(fit({ boxes: new Map(), width: 0, height: 0, milliseconds: 0 }, { width: 100, height: 100 })).toEqual({
      zoom: 1,
      x: 0,
      y: 0,
    });
  });
});

describe('the collapsed set of §4.7', () => {
  it('draws every composition shut until the reader opens one', () => {
    const folded = reading('llama3-8b').folded;
    expect([...collapsedGroups(folded, new Set())]).toEqual(['/compositions/decoder']);
    expect([...collapsedGroups(folded, new Set(['/compositions/decoder']))]).toEqual([]);
    expect([...everyComposition(folded)]).toEqual(['/compositions/decoder']);
  });

  it('reads a sidecar that names collapsed groups as the arrangement it is (D6)', () => {
    const folded = reading('llama3-8b').folded;
    expect([...expandedFromSidecar(folded, emptyLayout())]).toEqual([]);
    const arranged = { ...emptyLayout(), collapsed: ['compositions/nothing'] };
    expect([...expandedFromSidecar(folded, arranged)]).toEqual(['/compositions/decoder']);
  });

  it('writes nothing until a move or a fold: a Save with no overrides writes no sidecar', () => {
    const layout = new LayoutStore();
    expect(hasManualMoves(layout.layout)).toBe(false);
    layout.move(['instances', 'embed'], { x: 10, y: 20 });
    expect(hasManualMoves(layout.layout)).toBe(true);
    expect(layout.layout.positions['instances/embed']).toEqual({ x: 10, y: 20 });
    layout.reset();
    expect(hasManualMoves(layout.layout)).toBe(false);
  });
});

describe('the context menu of §4.7', () => {
  it('offers §4.7’s own thirteen entries, in §4.7’s order', () => {
    expect(CONTEXT_MENU.map((entry) => entry.label)).toEqual([
      'Edit',
      'Rename',
      'Duplicate',
      'Delete',
      'Drill In',
      'Add to Composition…',
      'Extract to Composition',
      'Extract to Template',
      'Edit Primitive',
      'Upgrade Pin…',
      'Show in Explorer',
      'Show in JSON',
      'Copy D1 identifier',
    ]);
  });

  it('mirrors every entry it calls a command in the menu bar (§4.4’s own rule)', () => {
    for (const entry of CONTEXT_MENU) {
      if (!entry.command) continue;
      expect(commandById(entry.id), entry.label).toBeDefined();
    }
  });

  it('names the six §4.4 does not carry, so none of them passes for one', () => {
    const own = CONTEXT_MENU.filter((entry) => !entry.command).map((entry) => entry.label);
    expect(own).toEqual([
      'Edit',
      'Drill In',
      'Add to Composition…',
      'Extract to Composition',
      'Show in Explorer',
      'Copy D1 identifier',
    ]);
    for (const entry of CONTEXT_MENU) {
      if (entry.command) continue;
      expect(commandById(entry.id), entry.label).toBeUndefined();
    }
  });

  it('offers a terminal what a terminal has, and a composition what it has', () => {
    expect(entriesFor(ROLE.terminal, ROLE).map((entry) => entry.label)).toEqual([
      'Edit',
      'Rename',
      'Delete',
      'Show in Explorer',
      'Show in JSON',
    ]);
    const group = entriesFor(ROLE.group, ROLE).map((entry) => entry.label);
    expect(group).not.toContain('Edit Primitive');
    expect(group).toContain('Drill In');
    expect(entriesFor(ROLE.node, ROLE).map((entry) => entry.label)).toContain('Edit Primitive');
  });
});
