import { describe, expect, it } from 'vitest';

import { drillModel, type DrillModel } from '../../src/canvas/drill.js';
import { withGhosts } from '../../src/canvas/Canvas.js';
import { compositionEntries, drillEntriesFor, entriesFor, portEntries } from '../../src/canvas/menu.js';
import { canvasModel, ROLE, SIDE } from '../../src/canvas/model.js';
import { NO_PLACEMENT, type Placement } from '../../src/canvas/layout.js';
import { bindings, reading, registry, shapes } from './source.js';

/**
 * What the drill-in draws — plan §4.8, artboards S4 and S5, over the corpus's own documents.
 *
 * The model is where the drawing is decided, so this is where the drawing is asserted: the cards,
 * the ghost column, the pinned terminals, the strip's bounds and the alternation rows. What the
 * *core* answered — which sites there are, what exists at an index — has its own suite one package
 * along (`packages/lang/test/describe/drill.test.ts`); nothing is asserted twice.
 */

function model(name: string, composition: string, options: { derived?: boolean } = {}): DrillModel {
  const read = reading(name);
  const answer = drillModel({
    tree: read.tree,
    composition,
    folded: read.folded,
    facts: read.facts,
    derived: options.derived === false ? null : read.derived,
    problems: read.problems,
    registry,
    shapes,
    bindings,
  });
  if (answer === null) throw new Error(`${name} declares no composition ${composition}`);
  return answer;
}

describe('the drill-in of llama3-8b › decoder (S4)', () => {
  const drill = model('llama3-8b', 'decoder');

  it('draws the six sites as cards, and nothing of the document’s top level', () => {
    const cards = drill.canvas.boxes.filter((box) => box.role === ROLE.node);
    expect(cards.map((box) => box.name)).toEqual([
      'attn_n',
      'attn',
      'attn_r',
      'ffn_n',
      'ffn',
      'ffn_r',
    ]);
    // A card of the drill-in is §4.7's own card, with the facts `describe` answered for the
    // representative iteration: the ports, the slot chips and the derived line.
    const attn = cards.find((box) => box.name === 'attn');
    expect(attn?.primitive).toBe('attention.dense');
    expect(attn?.inputs.map((port) => port.name)).toEqual(['input']);
    expect(attn?.slots.map((slot) => slot.name)).toEqual(['q', 'k', 'v', 'out', 'kv']);
    expect(attn?.derived).toBe('80.0 MiB params · 4.0 KiB / cached position');
    // Every card is drawn at the top of the drawing: a drill-in is one composition, so nothing
    // hangs inside anything.
    expect(cards.every((box) => box.parent === null)).toBe(true);
  });

  it('prints the index strip’s bounds as the document writes them (§4.8)', () => {
    expect(drill.ranges.map((range) => range.name)).toEqual(['layer']);
    const [layer] = drill.ranges;
    expect(layer?.bounds.map((bound) => `${bound.name} ${bound.text}`)).toEqual([
      'start 0',
      'stop 32',
      'step 1',
    ]);
    expect(layer?.bounds.map((bound) => bound.at.join('/'))).toEqual([
      'compositions/decoder/indices/layer/start',
      'compositions/decoder/indices/layer/stop',
      'compositions/decoder/indices/layer/step',
    ]);
    expect(layer?.values).toHaveLength(32);
    expect(drill.points).toHaveLength(32);
  });

  it('draws one ghost column on the left, with the two carry wires through it', () => {
    expect(drill.ghosts).toHaveLength(1);
    const [ghost] = drill.ghosts;
    expect(ghost?.name).toBe('ffn_r');
    expect(ghost?.side).toBe(SIDE.left);
    // §4.13's own text form for the override, printed by the one printer (feature 2.9's rule).
    expect(ghost?.label).toBe('[$layer - 1]');
    expect(ghost?.primitive).toBe('residual.add');
    const through = drill.canvas.wires.filter((wire) => wire.from === ghost?.id);
    expect(through.map((wire) => wire.label)).toEqual(['attn_n.carry', 'attn_r.a_carry']);
    expect(ghost?.targets).toHaveLength(2);
  });

  it('pins the boundary edges as terminals, with the rules on each (S4)', () => {
    const terminals = [...drill.terminals.values()];
    expect(terminals.map((one) => one.label)).toEqual([
      '◁ embed.output at layer = 0',
      '▷ final_n.input at layer = layers - 1',
    ]);
    expect(terminals[0]?.links.map((link) => link.text)).toEqual([
      'decoder.entry → attn_n.input',
      'decoder.entry.a → attn_r.a',
    ]);
    expect(terminals[1]?.links.map((link) => link.text)).toEqual(['final_n.in ← ffn_r.output']);
    // A terminal is a box of the layout, so the wires reach it and ELK puts it above or below.
    for (const terminal of terminals) {
      expect(drill.canvas.byPointer.get(terminal.id)?.role).toBe(ROLE.terminal);
    }
  });

  it('labels a guarded edge with its condition, in §4.13’s text form', () => {
    const carry = drill.canvas.wires.find((wire) => wire.label === 'attn_n.carry');
    expect(carry?.guard).toBe('$layer >= 1');
    expect(drill.guards.get('/compositions/decoder/bindings/values/attn_n.carry')).toBe('$layer >= 1');
  });

  it('has no alternation row, because no site of this composition is guarded', () => {
    expect(drill.rows).toEqual([]);
    expect(drill.unguarded).toBe(6);
    expect(drill.expanded).toBe(true);
  });
});

describe('the drill-in of gemma3n-kvshare › decoder (S5)', () => {
  const drill = model('gemma3n-kvshare', 'decoder');

  it('lists one row per guarded site, with its guard and its count', () => {
    expect(drill.rows.map((row) => `${row.name} ${String(row.at)} of ${String(row.of)}`)).toEqual([
      'attn 24 of 30',
      'ffn_sparse 10 of 30',
      'attn_full 6 of 30',
      'ffn 20 of 30',
    ]);
    expect(drill.rows.map((row) => row.guard)).toEqual([
      '$layer mod 5 != 4',
      '$layer < 10',
      '$layer mod 5 = 4 and $layer >= 4',
      '$layer >= 10',
    ]);
    expect(drill.unguarded).toBe(13);
  });

  it('fills a cell where D1 emitted the node, and only there', () => {
    const attn = drill.rows.find((row) => row.name === 'attn');
    const full = drill.rows.find((row) => row.name === 'attn_full');
    expect(attn?.cells).toHaveLength(30);
    expect(attn?.cells.filter((on) => on)).toHaveLength(24);
    // The two alternate: `attn` at every index but every fifth from four, `attn_full` there.
    for (const [at, on] of (attn?.cells ?? []).entries()) {
      expect(on).toBe(!(full?.cells[at] ?? false));
    }
  });

  it('says nothing about presence when nothing has been derived (§5.4’s freshness)', () => {
    const undrawn = model('gemma3n-kvshare', 'decoder', { derived: false });
    expect(undrawn.expanded).toBe(false);
    expect(undrawn.rows.map((row) => row.at)).toEqual([0, 0, 0, 0]);
    expect(undrawn.presence.points).toEqual([]);
  });
});

describe('the drill-in of a composition that names an instance outside it', () => {
  it('pins the root instance `gemma3n-kvshare`’s `aux` rule names as a terminal', () => {
    const drill = model('gemma3n-kvshare', 'decoder');
    const terminals = [...drill.terminals.values()];
    expect(terminals.some((one) => one.label.startsWith('◁ embed.auxiliary'))).toBe(true);
    // A scoped rule may name a root instance (`scoped_value_endpoint`'s second alternative), and
    // three corpus documents do; the drawing pins it exactly as it pins a top-level rule.
    const pinned = terminals.find((one) => one.label.startsWith('◁ embed.auxiliary'));
    expect(pinned?.links.map((link) => link.rule)).toContain('aux');
  });
});

describe('the ghost columns, placed beside the drawing', () => {
  it('shifts the drawing right and puts a left column in the room that opens', () => {
    const drill = model('llama3-8b', 'decoder');
    const before: Placement = {
      ...NO_PLACEMENT,
      boxes: new Map(
        drill.canvas.boxes.map((box, at) => [
          box.pointer,
          { pointer: box.pointer, x: 100, y: at * 120, width: 214, height: 90, manual: false },
        ]),
      ),
      width: 314,
      height: 800,
    };
    const after = withGhosts(before, drill.ghosts);
    const ghost = after.boxes.get(drill.ghosts[0]?.id ?? '');
    expect(ghost?.x).toBe(0);
    // Every box moved right by the column's width and the gap, so nothing is drawn over it.
    for (const [pointer, box] of before.boxes) {
      expect(after.boxes.get(pointer)?.x).toBe(box.x + 224);
    }
    expect(after.width).toBeGreaterThan(before.width);
    // The column sits level with the wires that reach it, not at the top of the drawing.
    expect(ghost?.y).toBeGreaterThan(0);
  });

  it('changes nothing where there is no ghost at all', () => {
    expect(withGhosts(NO_PLACEMENT, [])).toBe(NO_PLACEMENT);
  });
});

describe('a composition opened in place on the folded canvas (S3)', () => {
  it('draws its sites small and its own edges, and leaves the carries to the drill-in', () => {
    const read = reading('llama3-8b');
    const open = new Set(
      read.folded.nodes.filter((node) => node.children.length > 0).map((node) => node.pointer),
    );
    const shut = new Set<string>();
    const model = canvasModel({
      tree: read.tree,
      folded: read.folded,
      facts: read.facts,
      derived: read.derived,
      problems: read.problems,
      collapsed: shut,
      registry,
      shapes,
      bindings,
    });
    void open;
    const sites = model.boxes.filter((box) => box.parent !== null);
    expect(sites).toHaveLength(6);
    // S3's `.minode`: 176 px wide, against the 214 of a card.
    expect(sites.every((box) => box.width === 176)).toBe(true);

    // The six scoped edges that run between two sites at the current iteration; the two carries
    // are not among them — "only drawn in the drill-in (S4), where the ghost column has room".
    const inside = model.wires.filter((wire) => wire.id.startsWith('/compositions/decoder/'));
    expect(inside.map((wire) => wire.label)).toEqual([
      'attn.norm_in',
      'attn_r.b',
      'ffn_n.in',
      'ffn_r.a',
      'ffn.norm_in',
      'ffn_r.b',
    ]);
  });

  it('draws none of them while the composition is collapsed (feature 2.9’s own drawing)', () => {
    const read = reading('llama3-8b');
    const model = canvasModel({
      tree: read.tree,
      folded: read.folded,
      facts: read.facts,
      derived: read.derived,
      problems: read.problems,
      collapsed: new Set(
        read.folded.nodes.filter((node) => node.children.length > 0).map((node) => node.pointer),
      ),
      registry,
      shapes,
      bindings,
    });
    expect(model.wires.filter((wire) => wire.id.startsWith('/compositions/'))).toEqual([]);
  });
});

describe('the menus of §4.8 and §4.20', () => {
  it('offers §4.20’s two moves on an instance and neither on a composition', () => {
    const node = entriesFor(ROLE.node, ROLE).map((entry) => entry.label);
    const group = entriesFor(ROLE.group, ROLE).map((entry) => entry.label);
    // "turns selected roots into a new composition", "moves a root instance into a composition":
    // both are gestures on an instance, and a composition is what they make.
    expect(node).toContain('Extract to Composition');
    expect(node).toContain('Add to Composition…');
    expect(group).not.toContain('Extract to Composition');
    expect(group).not.toContain('Add to Composition…');
    expect(group).toContain('Drill In');
  });

  it('offers a site of a drill-in the complementary duplicate, and not the way into one', () => {
    const entries = drillEntriesFor(ROLE.node, ROLE).map((entry) => entry.label);
    expect(entries).toContain('Duplicate with complementary guard');
    expect(entries).not.toContain('Drill In');
    expect(entries).not.toContain('Add to Composition…');
    expect(entries).not.toContain('Extract to Composition');
    // It is offered beside `Duplicate`, which is the gesture it is a form of.
    expect(entries.indexOf('Duplicate with complementary guard')).toBe(entries.indexOf('Duplicate') + 1);
  });

  it('offers "Connect from previous iteration…" on an output of a drill-in, per index', () => {
    expect(portEntries('outputs').map((entry) => entry.label)).toEqual(['Expose as output…']);
    expect(portEntries('outputs', ['layer']).map((entry) => entry.label)).toEqual([
      'Expose as output…',
      'Connect from previous iteration…',
    ]);
    expect(portEntries('outputs', ['row', 'col']).map((entry) => entry.label)).toEqual([
      'Expose as output…',
      'Connect from previous row…',
      'Connect from previous col…',
    ]);
    // An input port has nothing to carry from: a connection runs from a producer.
    expect(portEntries('inputs', ['layer']).map((entry) => entry.label)).toEqual(['Expose as input…']);
  });

  it('offers one entry per composition of the document, and none where there are none', () => {
    expect(compositionEntries(['decoder', 'encoder']).map((entry) => entry.label)).toEqual([
      'decoder',
      'encoder',
    ]);
    expect(compositionEntries([])).toEqual([]);
  });
});
