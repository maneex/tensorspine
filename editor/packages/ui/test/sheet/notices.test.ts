import { describe, expect, it } from 'vitest';

import { serialize } from '@tensorspine/lang';
import { DocumentStore } from '@tensorspine/store';

import { fixesFor } from '../../src/problems/fixes.js';
import { absentSlotBindings, NOTICE, slotSites } from '../../src/problems/notices.js';
import { NO_ARGUMENT_FACTS } from '../../src/sheet/artifact.js';
import { writeLiteral } from '../../src/sheet/edits.js';
import { artifactOf, context, described, facts, freshTree, sheetFor } from './source.js';

/**
 * The structural-change notice and its fix — plan §3, §4.12 and §4.17's first editor row.
 *
 * > After an edit of a structural argument the core's `describe` returns the new port/slot/state
 * > set; the editor lists every binding that now names an absent element as a Problem with a fix
 * > action ("Rebind q → q_gated"), and every newly present element without a binding as the usual
 * > V7 problem.
 *
 * The edit is made on the row, the facts are the core's for the edited document, and the notice's
 * words are the feature's own block's.
 */

const LLAMA = 'llama3-8b';
const ATTN = 'decoder/attn[layer=0]';

/** `llama3-8b` with `output_gate` turned on from the sheet's own row. */
function gated(): { store: DocumentStore; facts: ReturnType<typeof facts> } {
  const store = new DocumentStore(freshTree(LLAMA), context.shapes);
  const sheet = sheetFor(LLAMA, store.tree, ATTN);
  const row = sheet.rows.find((one) => one.path === 'output_gate');
  if (row === undefined) throw new Error('no output_gate row');
  const artifact = artifactOf('attention.dense', '1.0.0');
  store.apply(
    writeLiteral(store.context, row, true, (artifact?.at(row.path) ?? NO_ARGUMENT_FACTS).facts),
  );
  return { store, facts: described(LLAMA, store.tree) };
}

describe('a binding that names a slot the arguments no longer create', () => {
  it('is nothing at all while the document is as it was written', () => {
    expect(
      absentSlotBindings({ sites: slotSites(facts(LLAMA)), index: new DocumentStore(freshTree(LLAMA), context.shapes).context.index }),
    ).toEqual([]);
  });

  it('is one notice, named as the document writes the binding', () => {
    const { store, facts: after } = gated();
    const rows = absentSlotBindings({
      sites: slotSites(after),
      index: store.context.index,
      file: 'models/llama3-8b.json',
    });
    expect(rows).toHaveLength(1);
    const row = rows[0];
    expect(row?.message).toBe('binding attn.q names a slot the arguments no longer create');
    expect(row?.rule).toBe(NOTICE.absentSlot);
    expect(row?.source).toBe('editor');
    expect(row?.severity).toBe('notice');
    // The place the *document* writes it at, not the place §5.2 rule 7 hoists it to.
    expect(row?.path).toBe('/compositions/decoder/bindings/parameters/attn.q');
    expect(row?.node).toBe('decoder/attn[layer=0].q');
  });

  it('carries "Rebind q → q_gated", and applying it writes the new slot’s name', () => {
    const { store, facts: after } = gated();
    const sites = slotSites(after);
    const rows = absentSlotBindings({ sites, index: store.context.index });
    const fixes = fixesFor(rows[0] as never, { rows: [], slots: sites, index: store.context.index });
    expect(fixes.map((fix) => fix.action.title)).toEqual(['Rebind q → q_gated']);

    store.apply((fixes[0] as never as { make: (c: never) => never }).make(store.context as never));
    expect(serialize(store.tree)).toContain('"parameter": "q_gated"');
    // The rule keeps the name its author gave it: a binding's name is the author's (feature 2.9).
    expect(serialize(store.tree)).toContain('"attn.q": {');
    // And the core agrees: the slot that was absent is bound, and nothing is left unbound.
    const again = described(LLAMA, store.tree);
    const site = again.sites.get(ATTN);
    expect(site?.parameters.find((slot) => slot.name === 'q_gated')?.boundBy).toBe('decoder.attn.q');
    expect(absentSlotBindings({ sites: slotSites(again), index: store.context.index })).toEqual([]);
  });

  it('offers no fix where more than one slot of the kind is free', () => {
    const { store, facts: after } = gated();
    const sites = slotSites(after).map((site) =>
      site.where !== ATTN
        ? site
        : {
            ...site,
            // A second free parameter slot of the same kind: nothing here can say which goes with
            // which, and §4.17's rule is that a fix exists only where the repair is unambiguous.
            slots: [...site.slots, { name: 'q_other', kind: 'parameter', present: true, boundBy: null }],
          },
    );
    const rows = absentSlotBindings({ sites, index: store.context.index });
    expect(fixesFor(rows[0] as never, { rows: [], slots: sites, index: store.context.index })).toEqual([]);
  });
});

describe('the sheet after the same edit', () => {
  it('shows `q` absent and `q_gated` present, which is what the notice reads', () => {
    const { store } = gated();
    const after = described(LLAMA, store.tree).sites.get(ATTN);
    expect(after?.parameters.filter((slot) => slot.present).map((slot) => slot.name)).toEqual([
      'q_gated',
      'k',
      'v',
      'out',
    ]);
  });
});
