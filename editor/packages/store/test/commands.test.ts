import { isJsonObject, jsonInteger, jsonObject, type JsonValue } from '@tensorspine/lang';
import { describe, expect, it } from 'vitest';

import {
  addToMap,
  connect,
  insertItem,
  remove,
  rename,
  sameValue,
  setMemberAt,
  setValue,
  unique,
} from '../src/commands.js';
import { DocumentStore } from '../src/document.js';
import { EditError, nodeAt, pointerOf, type Path } from '../src/path.js';
import { corpusText, corpusTree, shapes } from './source.js';
import { COMPOSITION, INDEX, QUANTITY, ROOT_INSTANCE, site } from './selectors.js';

// The JSON edit each gesture of plan §4.7 makes, on the document the plan's own examples are
// taken from. `llama3-8b` is 9 sites, one composition, three root instances, one public input
// and one public output: every shape a command has to deal with, small enough to state exactly.

function store(name = 'llama3-8b'): DocumentStore {
  return DocumentStore.open(corpusText(name), shapes());
}

function names(value: JsonValue | undefined): readonly string[] {
  return value !== undefined && isJsonObject(value) ? value.members.map((member) => member.name) : [];
}

function places(paths: readonly Path[]): string[] {
  return paths.map((path) => pointerOf(path));
}

describe('setting a value', () => {
  it('writes it where the path says, and names the row', () => {
    const one = store();
    const applied = one.run((context) =>
      setValue(context, { path: ['quantities', 'd', 'source', 'value'], value: jsonInteger(2048) }),
    );
    expect(applied.label).toBe('Set value');
    expect(nodeAt(one.tree, ['quantities', 'd', 'source', 'value'])).toEqual({
      kind: 'number',
      value: 2048,
      real: false,
    });
    // The member keeps its place: an edit never moves one.
    expect(names(nodeAt(one.tree, ['quantities', 'd', 'source']))).toEqual(['kind', 'value']);
  });

  it('refuses a place the document has not — a defect of the caller, not a verdict', () => {
    const one = store();
    expect(() => one.run((context) => setValue(context, { path: ['nowhere'], value: null }))).toThrow(
      EditError,
    );
  });
});

describe('adding a member and an item', () => {
  it('puts a new member where the gesture says, and leaves an old one where it was', () => {
    const one = store();
    one.run((context) =>
      setMemberAt(context, {
        path: ['instances', 'embed'],
        name: 'when',
        value: jsonObject([{ name: 'boolean', value: true }]),
      }),
    );
    expect(names(nodeAt(one.tree, ['instances', 'embed']))).toEqual([
      'primitive',
      'arguments',
      'families',
      'when',
    ]);
    one.run((context) =>
      setMemberAt(context, {
        path: ['instances', 'embed'],
        name: 'weights_location_prefix',
        value: [],
        at: 0,
      }),
    );
    expect(names(nodeAt(one.tree, ['instances', 'embed']))[0]).toBe('weights_location_prefix');
  });

  it('adds a family at the end and at a place', () => {
    const one = store();
    one.run((context) =>
      insertItem(context, { path: ['instances', 'embed', 'families'], value: 'extra' }),
    );
    expect(nodeAt(one.tree, ['instances', 'embed', 'families'])).toEqual([
      'input',
      'embedding',
      'extra',
    ]);
    one.run((context) =>
      insertItem(context, { path: ['instances', 'embed', 'families'], value: 'first', at: 0 }),
    );
    expect(nodeAt(one.tree, ['instances', 'embed', 'families', 0])).toBe('first');
  });
});

describe('adding an instance — the palette’s drop (D5)', () => {
  const placeholder = {
    // Written in the wrong order on purpose: the schema's order is what comes out.
    families: ['norm'] as JsonValue,
    arguments: jsonObject([]) as JsonValue,
    primitive: jsonObject([
      { name: 'name', value: 'norm.rms' },
      { name: 'version', value: '1.0.0' },
    ]) as JsonValue,
  };

  it('writes the required members, in the schema’s own order', () => {
    const one = store();
    const applied = one.run((context) =>
      addToMap(context, { path: ['instances'], name: 'probe', values: placeholder }),
    );
    expect(applied.label).toBe('Add probe');
    expect(names(nodeAt(one.tree, ['instances', 'probe']))).toEqual([
      'primitive',
      'arguments',
      'families',
    ]);
    expect(names(nodeAt(one.tree, ['instances']))).toEqual(['embed', 'final_n', 'lm_head', 'probe']);
  });

  it('adds a site of a composition the same way', () => {
    const one = store();
    one.run((context) =>
      addToMap(context, {
        path: ['compositions', 'decoder', 'instances'],
        name: 'probe',
        values: placeholder,
      }),
    );
    expect(nodeAt(one.tree, ['compositions', 'decoder', 'instances', 'probe'])).toBeDefined();
  });

  it('refuses a set of members the schema does not require, and a name already there', () => {
    const one = store();
    expect(() =>
      one.run((context) =>
        addToMap(context, {
          path: ['instances'],
          name: 'probe',
          values: { primitive: placeholder.primitive },
        }),
      ),
    ).toThrow(/requires arguments, families/);
    expect(() =>
      one.run((context) => addToMap(context, { path: ['instances'], name: 'embed', values: placeholder })),
    ).toThrow(EditError);
  });
});

describe('connecting two ports — the canvas’s drag (§4.7, Q5)', () => {
  const tree = corpusTree('llama3-8b');
  const from = nodeAt(tree, ['bindings', 'values', 'lm_head.in', 'from']) as JsonValue;
  const to = nodeAt(tree, ['bindings', 'values', 'lm_head.in', 'to']) as JsonValue;
  const other = nodeAt(tree, ['bindings', 'values', 'final_n.in', 'to']) as JsonValue;

  it('names the binding as asked, uniquified against the map', () => {
    const one = store();
    const applied = one.run((context) =>
      connect(context, {
        path: ['bindings', 'values'],
        name: 'lm_head.in',
        values: { from, to: other },
        into: 'to',
      }),
    );
    expect(applied.label).toBe('Connect lm_head.in_2');
    expect(names(nodeAt(one.tree, ['bindings', 'values']))).toContain('lm_head.in_2');
  });

  it('replaces the edge that already fed the input, and says which (V7)', () => {
    const one = store();
    const applied = one.run((context) =>
      connect(context, {
        path: ['bindings', 'values'],
        name: 'again',
        values: { from, to },
        into: 'to',
      }),
    );
    expect(applied.replaced).toEqual(['bindings', 'values', 'lm_head.in']);
    const written = names(nodeAt(one.tree, ['bindings', 'values']));
    expect(written).not.toContain('lm_head.in');
    expect(written).toContain('again');
  });

  it('gives the replaced binding its own name back', () => {
    const one = store();
    const applied = one.run((context) =>
      connect(context, {
        path: ['bindings', 'values'],
        name: 'lm_head.in',
        values: { from, to },
        into: 'to',
      }),
    );
    expect(applied.replaced).toEqual(['bindings', 'values', 'lm_head.in']);
    expect(applied.label).toBe('Connect lm_head.in');
    expect(names(nodeAt(one.tree, ['bindings', 'values']))).toContain('lm_head.in');
    expect(names(nodeAt(one.tree, ['bindings', 'values']))).not.toContain('lm_head.in_2');
  });

  it('takes the replacement back with one undo — the gesture is one command (D13)', () => {
    const one = store();
    const before = one.text;
    one.run((context) =>
      connect(context, {
        path: ['bindings', 'values'],
        name: 'again',
        values: { from, to },
        into: 'to',
      }),
    );
    one.undo();
    expect(one.text).toBe(before);
  });

  it('writes the binding’s members in the schema’s order', () => {
    const one = store();
    one.run((context) =>
      connect(context, {
        path: ['bindings', 'values'],
        name: 'guarded',
        values: {
          to: other,
          when: jsonObject([{ name: 'boolean', value: true }]),
          from,
        },
        into: 'to',
      }),
    );
    expect(names(nodeAt(one.tree, ['bindings', 'values', 'guarded']))).toEqual(['from', 'to', 'when']);
  });
});

describe('renaming (plan §3: “rewrites every reference … and the sidecar keys follow”)', () => {
  it('rewrites the quantity `d` where a grep of the file finds it', () => {
    const one = store();
    const before = corpusText('llama3-8b');
    const grep = before.match(/"quantity": "d"/g) ?? [];
    expect(grep).toHaveLength(10);

    const applied = one.run((context) =>
      rename(context, { path: ['quantities', 'd'], to: 'width', references: QUANTITY }),
    );
    expect(applied.label).toBe('Rename d to width');
    const after = one.text;
    expect(after.match(/"quantity": "d"/g)).toBeNull();
    expect(after.match(/"quantity": "width"/g)).toHaveLength(grep.length);
    // The declaration keeps its place in the map, because member order is the document's.
    expect(names(nodeAt(one.tree, ['quantities']))).toEqual([
      'width',
      'ffn',
      'heads',
      'kv_heads',
      'head_dim',
      'layers',
      'vocab',
      'eps',
      'precision',
    ]);
    // Nothing else moved: the two files differ in the ten references and the one declaration.
    expect(after.split('\n').filter((line, position) => line !== before.split('\n')[position])).toHaveLength(
      11,
    );
  });

  it('answers the places that moved, so the sidecar can follow', () => {
    const one = store();
    const applied = one.run((context) =>
      rename(context, {
        path: ['compositions', 'decoder', 'instances', 'attn'],
        to: 'attention',
        references: site('decoder'),
      }),
    );
    expect(applied.moves).toEqual([
      {
        from: ['compositions', 'decoder', 'instances', 'attn'],
        to: ['compositions', 'decoder', 'instances', 'attention'],
      },
    ]);
    expect(one.text).toContain('"site": "attention"');
    expect(one.text).not.toContain('"site": "attn"');
  });

  it('rewrites a generated selector’s site, and leaves another composition’s alone', () => {
    // `shieldstral-3b` has two compositions; a site renamed in one is a site of that one.
    const one = store('gemma3n-kvshare');
    const before = one.text;
    const composition = names(nodeAt(one.tree, ['compositions']))[0] as string;
    const first = names(nodeAt(one.tree, ['compositions', composition, 'instances']))[0] as string;
    one.run((context) =>
      rename(context, {
        path: ['compositions', composition, 'instances', first],
        to: 'renamed_site',
        references: site(composition),
      }),
    );
    expect(one.text).not.toBe(before);
    expect(one.text).toContain('"renamed_site"');
  });

  it('rewrites an index where it is written, and every key that assigns it', () => {
    // An index is named nineteen times in `llama3-8b`: thirteen `{"index": "layer"}`, its own
    // range, two overrides on scoped endpoints and three assignments written from outside the
    // composition. The last six are *keys* of an `indices` map, which is why a rename of an index
    // is not a rewrite of strings.
    const one = store();
    const before = corpusText('llama3-8b');
    expect(before.match(/"layer"/g)).toHaveLength(19);
    expect(before.match(/"index": "layer"/g)).toHaveLength(13);
    expect(before.match(/"layer": \{/g)).toHaveLength(6);

    one.run((context) =>
      rename(context, {
        path: ['compositions', 'decoder', 'indices', 'layer'],
        to: 'depth',
        references: INDEX('decoder'),
      }),
    );
    const after = one.text;
    expect(after).not.toContain('"layer"');
    expect(after.match(/"depth"/g)).toHaveLength(19);
    expect(after.match(/"index": "depth"/g)).toHaveLength(13);
    expect(after.match(/"depth": \{/g)).toHaveLength(6);
    expect(after.split('\n')).toHaveLength(before.split('\n').length);
  });

  it('rewrites a composition’s name where a selector states it', () => {
    const one = store();
    one.run((context) =>
      rename(context, { path: ['compositions', 'decoder'], to: 'blocks', references: COMPOSITION }),
    );
    expect(one.text).toContain('"composition": "blocks"');
    expect(one.text).not.toContain('"composition": "decoder"');
    expect(one.text).toContain('"blocks": {');
  });

  it('refuses a name the map already carries — V12 is what that would be', () => {
    const one = store();
    expect(() =>
      one.run((context) => rename(context, { path: ['quantities', 'd'], to: 'ffn', references: QUANTITY })),
    ).toThrow(EditError);
  });
});

describe('deleting (plan §3: “the delete command cascades over every selector that names it”)', () => {
  it('lists and removes every binding naming the site `attn`', () => {
    const one = store();
    const applied = one.run((context) =>
      remove(context, {
        path: ['compositions', 'decoder', 'instances', 'attn'],
        references: site('decoder'),
      }),
    );
    expect(applied.label).toBe('Delete attn');
    expect(places(applied.cascade?.removed ?? []).sort()).toEqual(
      [
        '/compositions/decoder/instances/attn',
        '/compositions/decoder/bindings/values/attn.norm_in',
        '/compositions/decoder/bindings/values/attn_r.b',
        '/compositions/decoder/bindings/parameters/attn.q',
        '/compositions/decoder/bindings/parameters/attn.k',
        '/compositions/decoder/bindings/parameters/attn.v',
        '/compositions/decoder/bindings/parameters/attn.out',
        '/compositions/decoder/bindings/states/attn.kv',
      ].sort(),
    );
    expect(applied.cascade?.kept).toEqual([]);
    const text = one.text;
    expect(text).not.toContain('"site": "attn"');
    expect(text).not.toContain('"attn": {');
    // The sites that only *shared* a prefix of the name are untouched.
    expect(text).toContain('"attn_n"');
    expect(text).toContain('"attn_r"');
  });

  it('removes the member of an identity and the binding the member leaves empty', () => {
    // `attn.q` binds one member; the member alone cannot go, because `members` needs one and is
    // required of the binding — so the binding goes, and not the list it required.
    const one = store();
    const applied = one.run((context) =>
      remove(context, {
        path: ['compositions', 'decoder', 'instances', 'attn'],
        references: site('decoder'),
      }),
    );
    expect(places(applied.cascade?.removed ?? [])).toContain(
      '/compositions/decoder/bindings/parameters/attn.q',
    );
    expect(places(applied.cascade?.removed ?? [])).not.toContain(
      '/compositions/decoder/bindings/parameters/attn.q/members/0',
    );
  });

  it('takes one member of a shared identity and leaves the identity standing', () => {
    // `qwen3.5-4b-text` ties `embed.weight` and `lm_head.weight` into one tensor. Deleting
    // `lm_head` costs the identity its second member and no more: `members` still has one, which
    // is what the grammar asks of it.
    const one = store('qwen3.5-4b-text');
    const applied = one.run((context) =>
      remove(context, { path: ['instances', 'lm_head'], references: ROOT_INSTANCE }),
    );
    expect(places(applied.cascade?.removed ?? [])).toContain(
      '/bindings/parameters/embed.weight/members/1',
    );
    expect(places(applied.cascade?.removed ?? [])).not.toContain('/bindings/parameters/embed.weight');
    expect(one.text).toContain('"embed.weight"');
    // Its one public output reads `lm_head`, and `interfaces/outputs` needs a member: the output
    // stays, naming an instance that has gone, and V1 is what says so.
    expect(places(applied.cascade?.kept ?? [])).toEqual(['/interfaces/outputs/main']);
    expect(one.text).toContain('"main": {');
  });

  it('keeps what it cannot remove without leaving the grammar, and says so (D5)', () => {
    const one = store();
    const applied = one.run((context) =>
      remove(context, { path: ['instances', 'embed'], references: ROOT_INSTANCE }),
    );
    expect(places(applied.cascade?.removed ?? []).sort()).toEqual(
      [
        '/instances/embed',
        '/bindings/values/decoder.entry',
        '/bindings/values/decoder.entry.a',
        '/bindings/parameters/embed.weight',
      ].sort(),
    );
    // The one public input feeds `embed` alone; dropping its endpoint would empty `to`, dropping
    // `tokens` would empty `inputs`, and `inputs` is required. So the endpoint stays, the
    // document is still on the grammar, and V1 is what reports the dangling name.
    expect(places(applied.cascade?.kept ?? [])).toEqual(['/interfaces/inputs/tokens/to/0']);
    expect(one.text).toContain('"tokens"');
  });

  it('does nothing, and says why, when the gesture has no answer on the grammar', () => {
    const one = store();
    const before = one.text;
    const applied = one.run((context) =>
      remove(context, { path: ['interfaces', 'inputs', 'tokens'] }),
    );
    expect(applied.changed).toBe(false);
    expect(applied.cascade?.removed).toEqual([]);
    expect(places(applied.cascade?.kept ?? [])).toEqual(['/interfaces/inputs/tokens']);
    expect(one.text).toBe(before);
  });

  it('removes a whole composition and the bindings that name its sites', () => {
    const one = store();
    const applied = one.run((context) =>
      remove(context, { path: ['compositions', 'decoder'], references: COMPOSITION }),
    );
    expect(places(applied.cascade?.removed ?? [])).toContain('/compositions/decoder');
    expect(places(applied.cascade?.removed ?? [])).toContain('/bindings/values/decoder.entry');
    expect(one.text).not.toContain('"composition": "decoder"');
  });

  it('takes the whole cascade back with one undo', () => {
    const one = store();
    const before = one.text;
    one.run((context) =>
      remove(context, {
        path: ['compositions', 'decoder', 'instances', 'attn'],
        references: site('decoder'),
      }),
    );
    one.undo();
    expect(one.text).toBe(before);
  });
});

describe('the two readings a command needs of a value', () => {
  it('tells a name the map has not', () => {
    expect(unique('attn', [])).toBe('attn');
    expect(unique('attn', ['attn'])).toBe('attn_2');
    expect(unique('attn', ['attn', 'attn_2'])).toBe('attn_3');
  });

  it('compares two values on what they mean, not on how they were written', () => {
    expect(sameValue(jsonInteger(4), { kind: 'number', value: 4, real: false, lexeme: '4' })).toBe(true);
    // A real and a cardinality are not the same value: V3 reads the spelling.
    expect(sameValue(jsonInteger(4), { kind: 'number', value: 4, real: true })).toBe(false);
    expect(sameValue(jsonObject([{ name: 'a', value: 'x' }]), jsonObject([{ name: 'a', value: 'x' }]))).toBe(
      true,
    );
    // Member order is the document's, so two objects that write it differently are not one value.
    expect(
      sameValue(
        jsonObject([
          { name: 'a', value: 'x' },
          { name: 'b', value: 'y' },
        ]),
        jsonObject([
          { name: 'b', value: 'y' },
          { name: 'a', value: 'x' },
        ]),
      ),
    ).toBe(false);
    expect(sameValue(['a'], ['a'])).toBe(true);
    expect(sameValue(['a'], ['a', 'b'])).toBe(false);
    expect(sameValue(null, false)).toBe(false);
  });
});
