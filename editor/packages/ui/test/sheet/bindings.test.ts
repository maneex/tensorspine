import { describe, expect, it } from 'vitest';

import { identityFacts, type SiteKey } from '@tensorspine/lang';
import { DocumentStore, type Path } from '@tensorspine/store';

import { addDeclaration } from '../../src/sheet/add.js';
import { bindPrivately, tieTo, type HeldMember } from '../../src/sheet/bindings.js';
import { analysisOf, context, freshTree, library, MODEL } from './source.js';

/**
 * What "Bind privately" and "Tie to…" write — §4.11's two gestures, §4.14's identities.
 *
 * The corpus is the fixture and the round trip is the assertion: untie `lm_head.weight` from
 * `qwen3.5-4b-text`'s tied embedding and tie it back, and the document is the one that was read,
 * byte for byte. That is the property the e2e case of feature 2.13 states through the interface;
 * here it is stated over the commands, where a failure names the command rather than a click.
 */

/** A document held as the editor holds one, so a command can be applied and the bytes read back. */
function session(name: string): DocumentStore {
  return new DocumentStore(freshTree(name), context.shapes);
}

/** The site of one member of an identity, as the core names it. */
function memberOf(name: string, rule: string, at: number): { site: SiteKey; slot: string; held: HeldMember } {
  const facts = identityFacts(analysisOf(name), library(), { rule, state: false });
  const member = facts?.members[at];
  if (member === undefined) throw new Error(`${name}: ${rule} has no member ${String(at)}`);
  return {
    site: member.site,
    slot: member.slot,
    held: { rule: ['bindings', 'parameters', rule] as Path, at: member.at },
  };
}

describe('a slot leaving and joining an identity', () => {
  it('unties lm_head.weight and ties it back — the document is the one that was read', () => {
    const store = session('qwen3.5-4b-text');
    const before = store.text;
    const member = memberOf('qwen3.5-4b-text', 'embed.weight', 1);

    // "Bind privately (creates `<site>.<slot>`)": the slot leaves the tie and gets an identity of
    // its own, at the top level, with the symbol the grammar requires there.
    const added = store.run((edit) =>
      bindPrivately(edit, {
        target: { site: member.site, slot: member.slot, state: false },
        held: member.held,
      }),
    );
    expect(added).not.toBeNull();
    const untied = JSON.parse(store.text) as {
      bindings: { parameters: Record<string, { members: unknown[]; tensor?: { name: string } }> };
    };
    expect(Object.keys(untied.bindings.parameters)).toEqual([
      'embed.weight',
      'final_n.weight',
      'lm_head.weight',
    ]);
    expect(untied.bindings.parameters['embed.weight']?.members).toHaveLength(1);
    expect(untied.bindings.parameters['lm_head.weight']?.tensor).toEqual({ name: 'lm_head.weight' });
    expect(untied.bindings.parameters['lm_head.weight']?.members).toEqual([
      { instance: { kind: 'root', instance: 'lm_head' }, parameter: 'weight' },
    ]);

    // "Tie to…": the slot joins the identity `embed.weight` declares, and the rule it leaves —
    // now with no member at all — goes with it, which is the store's own collapse rule.
    store.run((edit) =>
      tieTo(edit, {
        target: { site: member.site, slot: member.slot, state: false },
        held: { rule: ['bindings', 'parameters', 'lm_head.weight'] as Path, at: 0 },
        into: ['bindings', 'parameters', 'embed.weight'] as Path,
      }),
    );
    expect(store.text).toBe(before);
  });

  it('writes a scoped rule for a site of a composition, naming the site alone', () => {
    const store = session('llama3-8b');
    const member = memberOf('llama3-8b', 'decoder.attn.q', 0);
    // The rule it is in today is written inside the composition, which is where the core's
    // reading points and where a new one goes too.
    const held: HeldMember = {
      rule: ['compositions', 'decoder', 'bindings', 'parameters', 'attn.q'] as Path,
      at: member.held.at,
    };
    const added = store.run((edit) =>
      bindPrivately(edit, {
        target: { site: member.site, slot: member.slot, state: false },
        held,
        name: 'attn.query',
      }),
    );
    expect(added).not.toBeNull();
    const written = JSON.parse(store.text) as {
      compositions: {
        decoder: { bindings: { parameters: Record<string, { members: unknown[]; tensor?: unknown }> } };
      };
    };
    const rules = written.compositions.decoder.bindings.parameters;
    // §5.2 rule 7: a scoped rule needs no symbol — its identity is `decoder.attn.query`.
    expect(rules['attn.query']).toEqual({ members: [{ site: 'attn', parameter: 'q' }] });
    // And the rule it left had one member, so the grammar's own `minItems` took it with it.
    expect(Object.keys(rules)).not.toContain('attn.q');
  });

  it('ties a scoped slot into a top-level identity with the selector the grammar asks for', () => {
    const store = session('llama3-8b');
    const member = memberOf('llama3-8b', 'decoder.attn.q', 0);
    store.run((edit) =>
      tieTo(edit, {
        target: { site: member.site, slot: member.slot, state: false },
        held: {
          rule: ['compositions', 'decoder', 'bindings', 'parameters', 'attn.q'] as Path,
          at: member.held.at,
        },
        into: ['bindings', 'parameters', 'embed.weight'] as Path,
      }),
    );
    const written = JSON.parse(store.text) as {
      bindings: { parameters: Record<string, { members: unknown[] }> };
    };
    // Outside its own composition a site is named by a selector, at the point the chip stands for.
    expect(written.bindings.parameters['embed.weight']?.members[1]).toEqual({
      instance: {
        kind: 'generated',
        composition: 'decoder',
        instance: 'attn',
        indices: { layer: { literal: 0 } },
      },
      parameter: 'q',
    });
  });

  it('binds an unbound slot without unbinding anything', () => {
    const store = session('llama3-8b');
    const member = memberOf('llama3-8b', 'embed.weight', 0);
    const added = store.run((edit) =>
      bindPrivately(edit, {
        target: { site: member.site, slot: 'weight', state: false },
        held: null,
        name: 'embed.copy',
      }),
    );
    expect(added?.label).toBe('Bind embed.weight privately');
    const written = JSON.parse(store.text) as {
      bindings: { parameters: Record<string, unknown> };
    };
    // Both rules stand: the document now binds one slot twice, which is V7's line and not the
    // editor's to refuse (Q5).
    expect(Object.keys(written.bindings.parameters)).toEqual([
      'embed.weight',
      'final_n.weight',
      'lm_head.weight',
      'embed.copy',
    ]);
  });

});

describe('an identity created from the Identities list (§4.14)', () => {
  it('is the map’s own Add, with a name the grammar admits', () => {
    const store = session('llama3-8b');
    const at = ['bindings', 'parameters'] as Path;
    const added = store.run((edit) =>
      addDeclaration(edit, { context, role: MODEL, path: at }),
    );
    // The binding maps are the one kind of map `presentation.json` gives no word to — a rule's
    // name is a label and nothing refers to it (feature 2.2) — so the entry is proposed under the
    // word the table itself uses. The empty name the map would otherwise take is what
    // `propertyNames` refuses, and nobody meant it.
    expect(added.label).toBe('Add entry');
    const written = JSON.parse(store.text) as { bindings: { parameters: Record<string, unknown> } };
    expect(Object.keys(written.bindings.parameters)).toContain('entry');
    // D5's skeleton: the required members of a `parameter_binding`, each the first value its own
    // schema admits — which V2 and the grammar then report on their own rows.
    expect(written.bindings.parameters['entry']).toEqual({ tensor: { name: '' }, members: [] });
  });
});
