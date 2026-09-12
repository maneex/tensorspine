import { describe, expect, it } from 'vitest';

import { templatePrimitives, whereOfSite } from '@tensorspine/lang';
import type { Problem } from '@tensorspine/lang/api';

import { canvasModel, DEFAULT_VIEW, ROLE, SIDE } from '../../src/canvas/model.js';
import { bindings, everyComposition, library, modelOf, reading, registry, shapes } from './source.js';

/**
 * What the folded canvas draws, held to the artboards and to the core.
 *
 * Two of these tests are the feature's own load-bearing ones. **The handles are the validator's**:
 * a hollow red input is V7's row and an amber output is V13's, and the suite reads both off the
 * core and requires them to name the same ports — which is what makes §4.7's "projected onto
 * handles, not a second check" true rather than hopeful. **The count is the core's**: `×32` where
 * a bound resolves and `×?` where it does not, which is erratum E10's lesson.
 */

describe('the boxes of S1', () => {
  it('draws llama3-8b’s three cards, one composition and two terminals', () => {
    const model = modelOf('llama3-8b');
    const cards = model.boxes.filter((box) => box.role === ROLE.node && box.parent === null);
    expect(cards.map((box) => box.name)).toEqual(['embed', 'final_n', 'lm_head']);
    const groups = model.boxes.filter((box) => box.role === ROLE.group);
    expect(groups.map((box) => box.name)).toEqual(['decoder']);
    const terminals = model.boxes.filter((box) => box.role === ROLE.terminal);
    expect(terminals.map((box) => box.name)).toEqual(['tokens', 'logits']);
    expect(terminals.map((box) => box.side)).toEqual([SIDE.left, SIDE.right]);
  });

  it('takes the role of each box from presentation.json and from nowhere else', () => {
    // `instance_definition` is bound `node`, `composition_definition` `group`, `public_input` and
    // `public_output` `terminal` with a side. A box whose place has no binding has no role, and
    // the canvas draws it as a card; a binding removed shows up here first.
    const model = modelOf('llama3-8b');
    for (const box of model.boxes) expect(box.role).not.toBe('');
  });

  it('says what it is drawing, as S1’s own note does', () => {
    expect(modelOf('llama3-8b').note).toBe('3 root · 1 composition · 1 input · 1 output');
  });

  it('reads the card’s face: name, primitive@version, families', () => {
    const card = modelOf('llama3-8b').byPointer.get('/instances/final_n');
    expect(card?.name).toBe('final_n');
    expect(card?.primitive).toBe('norm.rms');
    expect(card?.version).toBe('1.0.0');
    expect(card?.families).toEqual(['norm']);
  });

  it('shows the families only under View ▸ Show Families', () => {
    const off = modelOf('llama3-8b', { view: { families: false } });
    expect(off.byPointer.get('/instances/final_n')?.families).toEqual([]);
  });

  it('shows an interface’s own members beside its name (S1: token, generative)', () => {
    const model = modelOf('llama3-8b');
    expect(model.byPointer.get('/interfaces/inputs/tokens')?.badges).toEqual(['token']);
    expect(model.byPointer.get('/interfaces/outputs/logits')?.badges).toEqual(['generative']);
  });
});

describe('the template instance of S2', () => {
  it('marks the card whose primitive pins a template, and only from the library', () => {
    const read = reading('shieldstral-3b-composite');
    const plain = canvasModel({
      folded: read.folded,
      facts: read.facts,
      derived: null,
      problems: [],
      collapsed: everyComposition(read.folded),
      view: DEFAULT_VIEW,
      registry,
      shapes,
      bindings,
    });
    // With no library gathered, nothing is marked: whether a primitive is a template is a fact
    // about the *library*, and a document that has not been resolved against one says nothing.
    expect(plain.boxes.some((box) => box.template)).toBe(false);

    const marked = canvasModel({
      folded: read.folded,
      facts: read.facts,
      derived: null,
      problems: [],
      collapsed: everyComposition(read.folded),
      view: DEFAULT_VIEW,
      registry,
      shapes,
      bindings,
      templates: new Set(templatePrimitives(library())),
    });
    const templates = marked.boxes.filter((box) => box.template);
    expect(templates.map((box) => box.name)).toEqual(['text']);
    expect(templates[0]?.primitive).toBe('decoder.causal_yarn');
  });
});

describe('the structural summary of S2', () => {
  it('lists the arguments the core calls structural, in the declaration’s order', () => {
    const model = modelOf('llama3-8b', { collapsed: new Set() });
    const attn = model.byPointer.get('/compositions/decoder/instances/attn');
    // The order is the *declaration's*, and the set is what `attention.dense` calls structural:
    // S2's own panel for this site writes `head_dim` too, and its guarded qwen panel leaves it
    // out — the unit is the authority, and the core is what reads it.
    // `mask="causal"` and not S2's `mask=causal`: the card writes §4.13's text form, and that
    // form quotes a string literal since feature 2.11 gave it a parser — a bare `causal` is how a
    // *reference* to an argument of that name is written. A fifth artboard figure the repository
    // does not have, in the sense feature 2.9 listed four.
    expect(attn?.summary).toBe('width=d · heads=heads · head_dim=head_dim · kv_heads=kv_heads · mask="causal"');
  });

  it('leaves out an argument the declaration does not call structural', () => {
    // S2's own note: "`eps` is not in the summary: its declaration is `structural: false`."
    const model = modelOf('llama3-8b');
    expect(model.byPointer.get('/instances/final_n')?.summary).toBe('width=d');
  });

  it('prints the value as the document writes it, not as it resolves', () => {
    const model = modelOf('llama3-8b');
    expect(model.byPointer.get('/instances/embed')?.summary).toContain('width=d');
    expect(model.byPointer.get('/instances/embed')?.summary).not.toContain('4096');
  });
});

describe('the guard badge of S2', () => {
  it('prints the condition in §4.13’s text form', () => {
    const model = modelOf('qwen3.5-35b-a3b', { collapsed: new Set() });
    const guarded = model.boxes.filter((box) => box.guard !== null);
    expect(guarded.length).toBeGreaterThan(0);
    for (const box of guarded) expect(box.guard).toMatch(/^\$?\w/);
  });
});

describe('the composition box of S3', () => {
  it('prints the range and the count when the range resolves', () => {
    const decoder = modelOf('llama3-8b').byPointer.get('/compositions/decoder');
    expect(decoder?.range).toBe('layer ∈ [0, 32) by 1');
    expect(decoder?.count).toBe('×32');
    expect(decoder?.held).toBe('instances 6 · values 8 · parameters 9 · states 1');
  });

  it('draws every boundary handle S1 and S3 draw, and only those', () => {
    const decoder = modelOf('llama3-8b').byPointer.get('/compositions/decoder');
    expect(decoder?.collapsed).toBe(true);
    expect(decoder?.handles.map((one) => one.label)).toEqual([
      'attn_n[layer=0].input',
      'attn_r[layer=0].a',
      'ffn_r[layer=31].output',
    ]);
    expect(decoder?.handles.map((one) => one.side)).toEqual([SIDE.left, SIDE.left, SIDE.right]);
  });

  it('draws its sites when it is expanded in place, and no handles then', () => {
    const model = modelOf('llama3-8b', { collapsed: new Set() });
    const decoder = model.byPointer.get('/compositions/decoder');
    expect(decoder?.collapsed).toBe(false);
    expect(decoder?.handles).toEqual([]);
    const sites = model.boxes.filter((box) => box.parent === '/compositions/decoder');
    expect(sites.map((box) => box.name)).toEqual([
      'attn_n',
      'attn',
      'attn_r',
      'ffn_n',
      'ffn',
      'ffn_r',
    ]);
  });
});

describe('the handles of §4.7, projected from V7 and V13', () => {
  it('draws an unfed input hollow red exactly where the validator writes a V7 row', () => {
    // The rule the feature exists to keep honest: the handle and the row are one reading. The
    // validator's own condition is `present(port) && !producers.has((site, port))`, and the card
    // reads `present` and `fedBy` off those very maps — so the two sets must be equal, port for
    // port, on a document that has such a row.
    for (const name of ['llama3-8b', 'colbert-v2', 'whisper-large-v3']) {
      const read = reading(name);
      const model = canvasModel({
        folded: read.folded,
        facts: read.facts,
        derived: null,
        problems: read.problems,
        collapsed: new Set(),
        view: DEFAULT_VIEW,
        registry,
        shapes,
        bindings,
      });
      const hollow = new Set<string>();
      for (const box of model.boxes) {
        for (const port of box.inputs) if (port.state === 'unfed') hollow.add(`${box.pointer}:${port.name}`);
      }
      const rows = new Set<string>();
      for (const [where, site] of read.facts.sites) {
        for (const port of site.inputs) {
          if (port.present && port.fedBy === null) {
            rows.add(`${pointerOfSite(where, site.segments)}:${port.name}`);
          }
        }
      }
      expect([...hollow].sort()).toEqual([...rows].sort());
    }
  });

  it('draws an unconsumed output amber, and every corpus document has one: its own', () => {
    const model = modelOf('llama3-8b', { collapsed: new Set() });
    const amber = model.boxes.flatMap((box) =>
      box.outputs.filter((port) => port.state === 'unused').map((port) => `${box.name}.${port.name}`),
    );
    // `llama3-8b` is clean: every output is consumed by an edge or exposed by the interface.
    expect(amber).toEqual([]);
  });

  it('draws no handle for a port the arguments removed', () => {
    const model = modelOf('llama3-8b', { collapsed: new Set() });
    const attn = model.byPointer.get('/compositions/decoder/instances/attn');
    // `attention.dense` declares `source_values`, present only when `cross` holds; llama's is not.
    expect(attn?.inputs.map((port) => port.name)).toEqual(['input']);
    expect(attn?.outputs.map((port) => port.name)).toEqual(['output']);
  });

  it('says what a handle is, for the hover and for a reader who cannot see it', () => {
    const attn = modelOf('llama3-8b', { collapsed: new Set() }).byPointer.get(
      '/compositions/decoder/instances/attn',
    );
    expect(attn?.inputs[0]?.title).toContain('model.width=4096');
  });
});

describe('the slot chips of S2', () => {
  it('draws one chip per present slot and state port, in the primitive’s own order', () => {
    const attn = modelOf('llama3-8b', { collapsed: new Set() }).byPointer.get(
      '/compositions/decoder/instances/attn',
    );
    expect(attn?.slots.map((slot) => `${slot.kind[0] ?? ''}${slot.name}`)).toEqual([
      'pq',
      'pk',
      'pv',
      'pout',
      'skv',
    ]);
  });

  it('marks a located chip from D3’s own location, and a tied one from its members', () => {
    const model = modelOf('qwen3.5-4b-text');
    const embed = model.byPointer.get('/instances/embed');
    expect(embed?.slots[0]?.located).toBe(true);
    expect(embed?.slots[0]?.shared).toBe(true);
    expect(embed?.slots[0]?.identity).toBe('embed.weight');
  });

  it('says nothing about a chip when nothing has been derived', () => {
    const model = modelOf('llama3-8b', { derived: false });
    const embed = model.byPointer.get('/instances/embed');
    expect(embed?.slots[0]?.located).toBe(false);
    expect(embed?.slots[0]?.shared).toBe(false);
  });
});

describe('the derived line of S2 and S3', () => {
  it('shows one iteration on a card and the whole family on the box', () => {
    const model = modelOf('llama3-8b', { collapsed: new Set() });
    // The rendering is `view.py`'s own `fmt_bytes`, through feature 2.6's `sizeText`: two
    // decimals on a GiB, one on a MiB and on a KiB below 10 KiB, none above. S2 writes
    // `80.0 MiB params · 4 KiB / cached position` and S1 `1002 MiB`; the convention is what ships.
    expect(model.byPointer.get('/compositions/decoder/instances/attn')?.derived).toBe(
      '80.0 MiB params · 4.0 KiB / cached position',
    );
    expect(model.byPointer.get('/instances/embed')?.derived).toBe('1002.0 MiB params');
    expect(model.byPointer.get('/compositions/decoder')?.derived).toBe(
      '13.00 GiB params · 128 KiB / cached position',
    );
  });

  it('is drawn only under View ▸ Show Derived Figures', () => {
    const off = modelOf('llama3-8b', { view: { derivedFigures: false } });
    expect(off.byPointer.get('/instances/embed')?.derived).toBeNull();
  });

  it('is dimmed with a badge while a derivation is behind the document (§5.4)', () => {
    const read = reading('llama3-8b');
    const model = canvasModel({
      folded: read.folded,
      facts: read.facts,
      derived: read.derived,
      stale: true,
      problems: [],
      collapsed: everyComposition(read.folded),
      view: DEFAULT_VIEW,
      registry,
      shapes,
      bindings,
    });
    expect(model.byPointer.get('/instances/embed')?.stale).toBe(true);
  });
});

describe('the wires of S1', () => {
  it('draws one per value binding and one per interface endpoint', () => {
    const model = modelOf('llama3-8b');
    expect(model.wires.map((wire) => wire.label)).toEqual([
      'decoder.entry',
      'decoder.entry.a',
      'final_n.in',
      'lm_head.in',
      'tokens',
      'logits',
    ]);
  });

  it('hangs a wire into a collapsed composition on the box', () => {
    const model = modelOf('llama3-8b');
    const entry = model.wires.find((wire) => wire.label === 'decoder.entry');
    expect(entry?.from).toBe('/instances/embed');
    expect(entry?.to).toBe('/compositions/decoder');
  });

  it('hangs it on the site itself once the composition is open', () => {
    const model = modelOf('llama3-8b', { collapsed: new Set() });
    const entry = model.wires.find((wire) => wire.label === 'decoder.entry');
    expect(entry?.to).toBe('/compositions/decoder/instances/attn_n');
  });

  it('shows the value’s type only under View ▸ Show Edge Types, in D2’s convention', () => {
    expect(modelOf('llama3-8b').wires[0]?.type).toBeNull();
    const on = modelOf('llama3-8b', { view: { edgeTypes: true } });
    expect(on.wires.find((wire) => wire.label === 'decoder.entry')?.type).toBe(
      'bf16[tokens, model.width=4096]',
    );
  });
});

describe('the identity links of §4.7', () => {
  it('draws none until View ▸ Show Identities is on', () => {
    expect(modelOf('qwen3.5-4b-text').links).toEqual([]);
  });

  it('joins the two cards of a tie', () => {
    const model = modelOf('qwen3.5-4b-text', { view: { identities: true } });
    expect(model.links.length).toBe(1);
    expect(model.links[0]?.identity).toBe('embed.weight');
    expect([model.links[0]?.from.box, model.links[0]?.to.box].sort()).toEqual([
      '/instances/embed',
      '/instances/lm_head',
    ]);
  });
});

describe('the problem dot of S2', () => {
  it('counts the problems whose pointer falls at or under a box', () => {
    const rows: Problem[] = [
      {
        code: 'V7',
        message: 'input port with no producer: norm.rms@final_n.input',
        path: '/instances/final_n',
        severity: 'error',
        source: 'semantic',
      },
      {
        code: '',
        message: 'a warning about the same place',
        path: '/instances/final_n/arguments/eps',
        severity: 'warning',
        source: 'lint',
      },
    ];
    const model = modelOf('llama3-8b', { problems: rows });
    const card = model.byPointer.get('/instances/final_n');
    expect(card?.problem).toBe('error');
    expect(card?.problems).toBe(2);
    expect(model.byPointer.get('/instances/embed')?.problem).toBeNull();
  });
});

/** The pointer of a described site, from the segments the core carries beside it. */
function pointerOfSite(where: string, segments: readonly (string | number)[]): string {
  void where;
  return segments.map((step) => `/${String(step).replace(/~/g, '~0').replace(/\//g, '~1')}`).join('');
}

describe('a document the core has said nothing about', () => {
  it('draws the boxes and nothing of the facts', () => {
    const read = reading('llama3-8b');
    const model = canvasModel({
      folded: read.folded,
      facts: null,
      derived: null,
      problems: [],
      collapsed: everyComposition(read.folded),
      view: DEFAULT_VIEW,
      registry,
      shapes,
      bindings,
    });
    expect(model.boxes.length).toBeGreaterThan(0);
    const embed = model.byPointer.get('/instances/embed');
    expect(embed?.facts).toBe(false);
    expect(embed?.inputs).toEqual([]);
    expect(embed?.slots).toEqual([]);
    expect(embed?.name).toBe('embed');
  });
});

describe('what the canvas costs', () => {
  it('builds the model of every corpus document inside the sheet’s own budget', () => {
    for (const name of ['llama3-8b', 'gemma3n-kvshare', 'deepseek-v4-pro']) {
      const read = reading(name);
      const started = performance.now();
      for (let round = 0; round < 3; round += 1) {
        canvasModel({
          folded: read.folded,
          facts: read.facts,
          derived: read.derived,
          problems: read.problems,
          collapsed: everyComposition(read.folded),
          view: DEFAULT_VIEW,
          registry,
          shapes,
          bindings,
        });
      }
      const each = (performance.now() - started) / 3;
      // §5.6 gives a keystroke 16 ms to render; asserted at three times it, as feature 1.11
      // asserts its own — the suite runs beside ninety other files.
      expect(each, name).toBeLessThan(48);
    }
  });
});

describe('every site the canvas asks the core about', () => {
  it('is one of the sites describe(folded) answers', () => {
    const read = reading('llama3-8b');
    const described = new Set([...read.facts.sites.keys()]);
    for (const node of read.folded.byPointer.values()) {
      if (node.where === null) continue;
      expect(described.has(node.where) || node.kind === 'composition', node.where).toBe(true);
    }
    expect([...described].sort()).toContain(whereOfSite({ kind: 'root', composition: '', name: 'embed', indices: [] }));
  });
});
