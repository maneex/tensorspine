import { describe, expect, it } from 'vitest';

import { isJsonObject, parse, serialize, type JsonValue } from '../../src/json/index.js';
import {
  identityFacts,
  identityMember,
  identitySymbol,
  locationAxisPlaces,
  type Analysis,
} from '../../src/index.js';
import { corpus, describedCorpus, library } from './source.js';

/**
 * What §4.14's location editor asks the core for, over the three documents feature 2.13 names.
 *
 * The claim under test is the one the inventory's §7 makes about every fact on a sheet: *the
 * component shows it, the core answers it*. So each case below is a fact the editor would
 * otherwise have had to compute — which axes a `stack` may name, what an `{index}` token may
 * name, what a location evaluates to for layer 0 and layer 31, which dtypes V14 admits — read
 * off the corpus's own documents and the reference base, never off a fixture written for it.
 */

function analysisOf(name: string): Analysis {
  const analysis = describedCorpus(name).analysis;
  if (analysis === null) throw new Error(`${name}: no analysis`);
  return analysis;
}

describe('the facts of one parameter identity', () => {
  it('names llama3-8b’s decoder.attn.q, its instances and the names each evaluates to', () => {
    const facts = identityFacts(analysisOf('llama3-8b'), library, {
      rule: 'decoder.attn.q',
      state: false,
    });
    expect(facts).not.toBeNull();
    if (facts === null) return;

    // One instance per layer — the rule fires once per point of the composition's grid — and the
    // identity each carries is the name §5.2 rule 7 gives it, which is what D3's rows are keyed by.
    expect(facts.instances).toHaveLength(32);
    expect(facts.instances[0]?.identity).toBe('decoder.attn.q[layer=0]');
    expect(facts.instances[31]?.identity).toBe('decoder.attn.q[layer=31]');
    expect(facts.instances[0]?.indices).toEqual([{ name: 'layer', value: '0' }]);

    // S8's own preview: the evaluated names for layer 0 and layer 31.
    expect(facts.instances[0]?.names).toEqual(['model.layers.0.self_attn.q_proj.weight']);
    expect(facts.instances[31]?.names).toEqual(['model.layers.31.self_attn.q_proj.weight']);
    expect(facts.instances.every((one) => one.located)).toBe(true);
    expect(facts.instances[0]?.slices).toEqual([]);

    // The slot the location is written against — "the first resolved member's slot" — and the
    // axes of its shape **as stored**, which is what `evaluate_location` resolves an axis against.
    expect(facts.members.map((one) => `${one.where}.${one.slot}`)).toEqual([
      'decoder/attn[layer=0].q',
    ]);
    expect(facts.members[0]?.present).toBe(true);
    expect(facts.axes).toEqual(['heads_flat', 'feature']);
    expect(facts.shape?.map((axis) => axis.extent)).toEqual([4096n, 4096n]);

    // What a token may name: the index the rule fires under, and the axes a `{coordinate}` names.
    expect(facts.tokens).toEqual([
      { member: 'index', names: ['layer'], declared: true },
      { member: 'coordinate', names: ['heads_flat', 'feature'], declared: false },
    ]);

    // V14's own set, for the role the slot declares.
    expect(facts.dtypes).toContain('bf16');
    expect(facts.dtypes).not.toContain('bool');
  });

  it('offers `multiplicity` on gemma3n’s expand.projection, which declares one', () => {
    const facts = identityFacts(analysisOf('gemma3n-kvshare'), library, {
      rule: 'expand.projection',
      state: false,
    });
    expect(facts).not.toBeNull();
    if (facts === null) return;

    // §3.4: "a slot that declares a multiplicity … is stored with one axis before its shape axes —
    // local name `multiplicity`". The editor's axis list is that shape's, so the leading name is
    // there without the interface knowing the word.
    expect(facts.axes[0]).toBe('multiplicity');
    expect(facts.axes).toEqual(['multiplicity', 'feature']);
    expect(facts.shape?.[0]?.extent).toBe(3n);

    // The stack is expanded over its coordinates: three physical names, one per copy.
    expect(facts.instances).toHaveLength(1);
    expect(facts.instances[0]?.names).toEqual([
      'model.language_model.altup_projections.0.weight',
      'model.language_model.altup_projections.1.weight',
      'model.language_model.altup_projections.2.weight',
    ]);
    // A root instance's rule fires in no index environment at all.
    expect(facts.tokens[0]).toEqual({ member: 'index', names: [], declared: true });
    expect(facts.tokens[1]?.names).toEqual(['multiplicity', 'feature']);
  });

  it('reads a tie as one identity with two members — qwen3.5-4b-text’s embed.weight', () => {
    const facts = identityFacts(analysisOf('qwen3.5-4b-text'), library, {
      rule: 'embed.weight',
      state: false,
    });
    expect(facts?.members.map((one) => `${one.where}.${one.slot}`)).toEqual([
      'embed.weight',
      'lm_head.weight',
    ]);
    // Both members' roles are read: the dtype set is the intersection V14 compares against.
    expect(facts?.members.every((one) => one.roles.length === 1)).toBe(true);
    expect(facts?.dtypes.length).toBeGreaterThan(0);
    expect(facts?.instances).toHaveLength(1);
    expect(facts?.instances[0]?.names).toEqual(['model.language_model.embed_tokens.weight']);
  });

  it('reads a slice as a region, with its offset and its extent', () => {
    const facts = identityFacts(analysisOf('qwen3.5-35b-a3b'), library, {
      rule: 'vision.attn.k',
      state: false,
    });
    expect(facts).not.toBeNull();
    if (facts === null) return;
    const first = facts.instances[0];
    expect(first?.names).toEqual([]);
    expect(first?.slices[0]?.offset).toBe('1152');
    expect(first?.slices[0]?.extent).toBe('1152');
    expect(first?.located).toBe(true);
  });

  it('answers nothing for a rule no binding carries', () => {
    expect(
      identityFacts(analysisOf('llama3-8b'), library, { rule: 'nothing.at.all', state: false }),
    ).toBeNull();
  });
});

describe('where a location names an axis', () => {
  /** One binding rule of a corpus document, as the store holds it: an ordered tree. */
  const ruleOf = (name: string, at: readonly string[]): JsonValue => {
    let value: JsonValue = parse(corpus(name));
    for (const step of at) {
      const member = isJsonObject(value)
        ? value.members.find((each) => each.name === step)
        : undefined;
      if (member === undefined) throw new Error(`${name}: no rule at ${at.join('/')}`);
      value = member.value;
    }
    return value;
  };

  it('answers the places the select of §4.14 is filled at, and nowhere else', () => {
    // A plain `tensor` names no axis at all.
    expect(
      locationAxisPlaces(
        ruleOf('llama3-8b', ['compositions', 'decoder', 'bindings', 'parameters', 'attn.q']),
      ),
    ).toEqual([]);
    // A `stack` names one, and a `slice` names one.
    expect(
      locationAxisPlaces(ruleOf('gemma3n-kvshare', ['bindings', 'parameters', 'expand.projection'])),
    ).toEqual(['/location/stack/axis']);
    expect(
      locationAxisPlaces(
        ruleOf('qwen3.5-35b-a3b', ['compositions', 'vision', 'bindings', 'parameters', 'attn.k']),
      ),
    ).toEqual(['/location/slice/axis']);
  });

  it('follows the tree, which is what an author editing a location has', () => {
    // The grammar refuses a blank axis, so the analysis answers nothing about this document at
    // all — and the place is exactly what the author needs to repair it. The walk is the
    // grammar's own recursion, so a stack of slices names both.
    const written = parse(
      '{"members": [], "location": {"stack": {"axis": "", "part": {"slice": ' +
        '{"tensor": ["w"], "axis": "feature", "offset": {"literal": 0}}}}}}',
    );
    expect(locationAxisPlaces(written)).toEqual([
      '/location/stack/axis',
      '/location/stack/part/slice/axis',
    ]);
    // A concat names its own axis and walks every part.
    const concat = parse(
      '{"location": {"concat": {"axis": "feature", "parts": [{"tensor": ["a"]}, ' +
        '{"stack": {"axis": "multiplicity", "part": {"tensor": ["b"]}}}]}}}',
    );
    expect(locationAxisPlaces(concat)).toEqual([
      '/location/concat/axis',
      '/location/concat/parts/1/stack/axis',
    ]);
    // A rule that writes no location names nothing.
    expect(locationAxisPlaces(parse('{"members": []}'))).toEqual([]);
  });
});

describe('the facts of one state identity', () => {
  it('carries the members and the payload roles, and no location at all', () => {
    const facts = identityFacts(analysisOf('gemma3n-kvshare'), library, {
      rule: 'shared.sliding.kv',
      state: true,
    });
    expect(facts).not.toBeNull();
    if (facts === null) return;
    expect(facts.state).toBe(true);
    // A state identity has no location: no shape to write one against, no axes, no tokens.
    expect(facts.shape).toBeNull();
    expect(facts.axes).toEqual([]);
    expect(facts.tokens).toEqual([]);
    expect(facts.instances.every((one) => one.located)).toBe(false);
    // V14 reads every payload component's role on the state side, so a `kv` port answers two.
    expect(facts.members.length).toBeGreaterThan(1);
    expect(facts.members[0]?.roles.length).toBe(2);
    expect(facts.dtypes.length).toBeGreaterThan(0);
  });
});

describe('the endpoint a gesture writes', () => {
  it('names a site alone inside its own composition, and with a selector outside it', () => {
    const analysis = analysisOf('llama3-8b');
    const facts = identityFacts(analysis, library, { rule: 'decoder.attn.q', state: false });
    const site = facts?.members[0]?.site;
    expect(site).toBeDefined();
    if (site === undefined) return;

    // §5.2: a scoped rule's site endpoint "selects the generated instance at the current indices",
    // so the rule written inside `decoder` names the site and nothing else.
    expect(serialize(identityMember(site, 'q', { state: false, scope: 'decoder' }))).toBe(
      '{\n  "site": "attn",\n  "parameter": "q"\n}\n',
    );
    // Written anywhere else, the same member is the selector the grammar asks for, at the point
    // the chip stands for.
    expect(serialize(identityMember(site, 'q', { state: false }))).toBe(
      [
        '{',
        '  "instance": {',
        '    "kind": "generated",',
        '    "composition": "decoder",',
        '    "instance": "attn",',
        '    "indices": {',
        '      "layer": {',
        '        "literal": 0',
        '      }',
        '    }',
        '  },',
        '  "parameter": "q"',
        '}',
        '',
      ].join('\n'),
    );
  });

  it('names a root instance, and a state port by the member the grammar gives it', () => {
    const analysis = analysisOf('qwen3.5-4b-text');
    const facts = identityFacts(analysis, library, { rule: 'embed.weight', state: false });
    const site = facts?.members[1]?.site;
    expect(site).toBeDefined();
    if (site === undefined) return;
    expect(serialize(identityMember(site, 'weight', { state: false }))).toBe(
      [
        '{',
        '  "instance": {',
        '    "kind": "root",',
        '    "instance": "lm_head"',
        '  },',
        '  "parameter": "weight"',
        '}',
        '',
      ].join('\n'),
    );
    expect(serialize(identityMember(site, 'kv', { state: true }))).toContain('"state": "kv"');
    expect(serialize(identitySymbol('lm_head.weight'))).toBe(
      '{\n  "name": "lm_head.weight"\n}\n',
    );
  });
});
