import { describe, expect, it } from 'vitest';

import {
  describe as describeDocument,
  describeAnalysis,
  analyse,
  toPython,
  whereOfSite,
  type Description,
  type PyValue,
  type SiteDescription,
} from '../../src/index.js';
import { parse } from '../../src/json/index.js';
import { corpus, describedCorpus, library, schemas, siteOf, sites } from './source.js';

// `describe` (feature 1.6d): the facts §4.7, §4.11 and §4.12 show, and §7 of the component
// inventory forbids a component to compute — "which ports, slots and states exist", "whether an
// argument applies", "what a shape evaluates to", "compatible partners for tying and sharing".
//
// The shapes themselves are held to the tools' own products by the parity suite
// (`test/parity/describe.test.ts`, D2's and D3's rows and D4's rule for every corpus node). What
// is here is what a *reader* of the answer needs and the products do not state: presence and
// absence, the source of each fact, the applying entries, and the compatibility lists — pinned on
// the corpus documents the design pass itself corrected its figures against (inventory §0, E7 to
// E9: `attention.dense` partitions on `attention.heads` and `attention.kv_heads`, both
// `all_reduce`, granularity `div(heads, kv_heads)` = 4 on llama; its `kv` port has four ordered
// rules and a causal, non-cross, window-less instance selects the **fourth**).

describe('describe gates on the grammar', () => {
  it('answers the schema stage and nothing else for a document off the grammar', () => {
    // "Meaning assumes grammar" (feature 1.2): `analyse` would raise on this, and the editor calls
    // `describe` on every keystroke, so the answer is a report and never an exception.
    const tree = parse('{"schema": "tensorspine/2.0", "model": 3}');
    const answer = describeDocument(tree, { schemas, library });
    expect(answer.conforms).toBe(false);
    expect(answer.analysis).toBe(null);
    expect(answer.sites.size).toBe(0);
    expect(answer.structural.length).toBeGreaterThan(0);
  });

  it('describes a corpus document with no problem at all', () => {
    const answer = describedCorpus('llama3-8b');
    expect(answer.conforms).toBe(true);
    expect(answer.analysis?.problems).toEqual([]);
    expect(answer.sites.size).toBe(195);
  });

  it('answers the same over an analysis a caller already has', () => {
    const tree = parse(corpus('llama3-8b'));
    const one = describeDocument(tree, { schemas, library });
    const other = describeAnalysis(analyse(toPython(tree), library));
    expect(shape(other)).toEqual(shape(one));
  });
});

describe('the ports of a site', () => {
  const attn = siteOf('llama3-8b', 'decoder/attn[layer=0]');

  it('draws the present ones and leaves out the ones a condition removes', () => {
    // `source_values` is present only on a cross-attention instance (§4.1's `present_when`).
    expect(attn.inputs.map((port) => [port.name, port.present])).toEqual([
      ['input', true],
      ['source_values', false],
    ]);
    expect(attn.outputs.map((port) => [port.name, port.present])).toEqual([['output', true]]);
  });

  it('evaluates the shape of a present port and none of an absent one', () => {
    const input = attn.inputs.find((port) => port.name === 'input');
    expect(input?.shape?.map((axis) => [axis.axis, axis.extent])).toEqual([
      ['model.width', 4096n],
    ]);
    expect(attn.inputs.find((port) => port.name === 'source_values')?.shape).toBe(null);
  });

  it('names what feeds an input and whether an output is consumed', () => {
    expect(attn.inputs.find((port) => port.name === 'input')?.fedBy).toBe('decoder.attn.norm_in');
    expect(attn.outputs.find((port) => port.name === 'output')?.consumed).toBe(true);
    // A public input is a producer like any other, named as the validator names it.
    expect(siteOf('llama3-8b', 'embed').inputs[0]?.fedBy).toBe('input:tokens');
  });

  it('carries the declared kind beside the domain the graph resolved', () => {
    // `embed.tokens` declares `token`; `attn.input` declares `inherit` and receives one.
    expect(siteOf('llama3-8b', 'embed').inputs[0]?.kind).toBe('token');
    const input = attn.inputs.find((port) => port.name === 'input');
    expect(input?.kind).toBe('inherit');
    expect(input?.domain).toEqual(['token', 'tokens']);
  });
});

describe('the slots of a site', () => {
  const attn = siteOf('llama3-8b', 'decoder/attn[layer=0]');

  it('draws the present ones, with the role, the identity and the binding', () => {
    const present = attn.parameters.filter((slot) => slot.present);
    expect(present.map((slot) => slot.name)).toEqual(['q', 'k', 'v', 'out']);
    expect(present.map((slot) => slot.identity)).toEqual([
      'decoder.attn.q[layer=0]',
      'decoder.attn.k[layer=0]',
      'decoder.attn.v[layer=0]',
      'decoder.attn.out[layer=0]',
    ]);
    expect(present[0]?.boundBy).toBe('decoder.attn.q');
    expect(present[0]?.role).toBe('attention.qkv_projection');
  });

  it('keeps an absent slot in the list, so a binding that names one can be found', () => {
    // Plan §3: "turning `output_gate` on replaces the slot `q` by `q_gated`" — the editor lists
    // the bindings that now name an absent element, which needs the absent ones to be answered.
    const gated = attn.parameters.find((slot) => slot.name === 'q_gated');
    expect(gated?.present).toBe(false);
    expect(gated?.shape).toBe(null);
    expect(gated?.identity).toBe(null);
  });

  it('answers the shape as stored, the storage axis of a multiplicity first', () => {
    const shared = siteOf('qwen3.5-35b-a3b', 'decoder/mlp[layer=0]').parameters.find(
      (slot) => slot.name === 'shared_gate',
    );
    expect(shared?.shape?.map((axis) => [axis.axis, axis.extent])).toEqual([
      ['storage.multiplicity', 1n],
      ['ffn.inner', 512n],
      ['model.width', 2048n],
    ]);
    expect(shared?.shape?.[0]?.nature).toBe('storage');
    expect(shared?.multiplicity).toBe(1n);
    // A slot that declares none says so, rather than answering the count 1 the products write.
    expect(attn.parameters.find((slot) => slot.name === 'q')?.multiplicity).toBe(null);
  });
});

describe('the state ports of a site', () => {
  it('answers the rule that applies — the fourth of `attention.dense`, on a causal instance', () => {
    // Inventory §0, erratum E8: "4 ordered rules; a causal, non-cross, window-less instance
    // selects rule 4."
    const kv = siteOf('llama3-8b', 'decoder/attn[layer=0]').states[0];
    expect(kv?.name).toBe('kv');
    expect(kv?.present).toBe(true);
    expect(kv?.ruleIndex).toBe(3);
    expect([kv?.evolution, kv?.access, kv?.sharing]).toEqual([
      'append',
      'logical_position',
      'by_position',
    ]);
    expect(kv?.indexedBySource).toBe(false);
    expect(kv?.stream).toEqual(['token', 'tokens']);
    expect(kv?.keyAxes).toEqual(['instance.session', 'instance.branch']);
    expect(kv?.operations).toEqual(['append', 'read']);
    expect(kv?.written).toBe(true);
    expect(kv?.carriedAcross).toBe(false);
    expect(kv?.identity).toBe('decoder.attn.kv[layer=0]');
  });

  it('answers a state indexed by a source port, with that port and its stream', () => {
    const kv = siteOf('whisper-large-v3', 'decoder/cross_attn[layer=0]').states[0];
    expect(kv?.ruleIndex).toBe(0);
    expect(kv?.indexedBySource).toBe(true);
    expect(kv?.indexedByPort).toBe('source_values');
    expect(kv?.stream).toEqual(['position', 'audio']);
  });

  it('answers the rule a chunked instance selects, and not the fallback', () => {
    // `llama4-scout` alternates chunked and full attention: the chunked instance takes the
    // **second** of `attention.dense`'s four rules — `window` / `ring` / `within_span`, its span
    // the `chunk.span` argument — and the full one the fourth. The four are mutually exclusive by
    // construction, so this is where the corpus shows the ordering doing work.
    const chunked = siteOf('llama4-scout', 'decoder/attn[layer=0]').states[0];
    expect(chunked?.ruleIndex).toBe(1);
    expect([chunked?.evolution, chunked?.access, chunked?.sharing]).toEqual([
      'window',
      'ring',
      'within_span',
    ]);
    expect(chunked?.span).toBe(8192n);
    const full = siteOf('llama4-scout', 'decoder/attn_full[layer=3]').states[0];
    expect(full?.ruleIndex).toBe(3);
    expect(full?.span).toBe(null);
  });

  it('answers a window rule with its evaluated span', () => {
    const history = siteOf('voxtral-realtime', 'conv_frontend').states.find(
      (state) => state.name === 'conv1_history',
    );
    expect(history?.evolution).toBe('window');
    expect(typeof history?.span).toBe('bigint');
    expect(history?.carriedAcross).toBe(true);
  });

  it('evaluates the payload of a present port and none of an absent one', () => {
    const kv = siteOf('llama3-8b', 'decoder/attn[layer=0]').states[0];
    expect(kv?.payload.map((component) => component.name)).toEqual(['k', 'v']);
    expect(kv?.payload[0]?.shape.map((axis) => [axis.axis, axis.extent])).toEqual([
      ['attention.kv_heads', 8n],
      ['attention.head_dim', 128n],
    ]);
    // The composite's vision attention is `mask: none`: its `kv` port is absent, and nothing
    // about it is evaluated.
    const vision = siteOf('shieldstral-3b-composite', 'vision/attn[layer=0]').states[0];
    expect(vision?.present).toBe(false);
    expect(vision?.rule).toBe(null);
    expect(vision?.payload).toEqual([]);
  });
});

describe('the partition options and the cost entries that apply', () => {
  it('answers the five of `attention.dense`, with the granularity evaluated', () => {
    // Inventory §0, erratum E7: "`attention.heads` and `attention.kv_heads`, both `all_reduce`;
    // granularity `div(heads, kv_heads)` = 4 for llama; 5 options with `instance.session` and the
    // two payload axes."
    const attn = siteOf('llama3-8b', 'decoder/attn[layer=0]');
    expect(
      attn.partitions.map((option) => [option.target, option.communication, option.granularity]),
    ).toEqual([
      [{ argument_axis: 'attention.heads' }, ['all_reduce'], 4n],
      [{ argument_axis: 'attention.kv_heads' }, ['all_reduce'], 1n],
      [{ instance_key_axis: 'instance.session' }, ['none'], 1n],
      [
        { payload_axis: { state: 'kv', component: 'k', axis: 'attention.kv_heads' } },
        ['none'],
        1n,
      ],
      [
        { payload_axis: { state: 'kv', component: 'v', axis: 'attention.kv_heads' } },
        ['none'],
        1n,
      ],
    ]);
    expect(attn.partitions.map((option) => option.index)).toEqual([0, 1, 2, 3, 4]);
  });

  it('answers the cost entries whose condition holds, with the expression evaluated', () => {
    const attn = siteOf('llama3-8b', 'decoder/attn[layer=0]');
    expect(attn.costs.map((entry) => [entry.per, entry.status, entry.value])).toEqual([
      ['cached_position', 'exact', 16384n],
    ]);
    // A primitive that declares no cost entry at all answers none.
    expect(siteOf('llama3-8b', 'final_n').costs).toEqual([]);
  });

  it('leaves out the entries whose condition does not hold', () => {
    // `attention.latent_compressed` declares three cost entries, two under `present index` and
    // one under its negation: `deepseek-v4-pro`'s two families of attention sites take opposite
    // branches, which is the one place in the corpus where an entry is dropped.
    const withIndex = siteOf('deepseek-v4-pro', 'main/attn_csa[layer=11]');
    const without = siteOf('deepseek-v4-pro', 'main/attn_hca[layer=10]');
    expect(withIndex.costs.map((entry) => [entry.index, entry.per, entry.value])).toEqual([
      [0, 'element', 268435456n],
      [1, 'cached_position', 16384n],
    ]);
    expect(without.costs.map((entry) => [entry.index, entry.per, entry.value])).toEqual([
      [2, 'cached_position', 262144n],
    ]);
  });
});

describe('a template instance', () => {
  const text = siteOf('shieldstral-3b-composite', 'text');

  it('carries the interface its expansion resolved', () => {
    expect(text.interface).not.toBe(null);
    expect([...(text.interface?.inputs.keys() ?? [])]).toEqual(['hidden']);
    expect([...(text.interface?.outputs.keys() ?? [])]).toEqual(['hidden_out']);
    // The stream is the *template's* own: its public input `hidden` introduces one of that name,
    // and the expansion is what resolves the output onto it (§4.6).
    expect(text.interface?.outputs.get('hidden_out')?.stream).toBe('hidden');
    expect(text.interface?.outputs.get('hidden_out')?.kind).toBe('token');
  });

  it('has the template’s interfaces as its ports, and no slot or state of its own', () => {
    expect(text.inputs.map((port) => port.name)).toEqual(['hidden']);
    expect(text.outputs.map((port) => port.name)).toEqual(['hidden_out']);
    expect(text.parameters).toEqual([]);
    expect(text.constants).toEqual([]);
    expect(text.states).toEqual([]);
    expect(text.partitions).toEqual([]);
  });

  it('is the only site of the corpus that carries one', () => {
    for (const [where, site] of sites('llama3-8b')) {
      expect(site.interface, where).toBe(null);
    }
  });
});

describe('the arguments are feature 1.6a’s facts, composed and not copied', () => {
  it('answers the per-argument facts the sheet of §4.12 shows', () => {
    const attn = siteOf('llama3-8b', 'decoder/attn[layer=0]');
    const facts = new Map(attn.arguments.facts.map((fact) => [fact.path, fact]));
    expect(facts.get('heads')?.source).toBe('given');
    expect(facts.get('heads')?.value).toBe(32n);
    expect(facts.get('kv_heads')?.value).toBe(8n);
    // `window` applies when `mask = causal` (inventory §0, erratum E4) and is not supplied here:
    // applicable, absent, and no refusal — which is what "Show inapplicable" needs to tell apart.
    expect(facts.get('window')?.applicable).toBe(true);
    expect(facts.get('window')?.source).toBe('absent');
    expect(facts.get('window')?.value).toBe(undefined);
    // `cross` is supplied and false, so `source_values` is the port that does not exist.
    expect(facts.get('cross')?.value).toBe(false);
    expect(facts.get('rope.theta')?.value).toBe(500000n);
    // The invariants are the V8 block's verdicts, holding ones included.
    expect(attn.arguments.invariants.every((one) => one.verdict === 'holds')).toBe(true);
    expect(attn.arguments.invariants.length).toBeGreaterThan(0);
  });

  it('is the very object the analysis recorded inside the walk', () => {
    const answer = describedCorpus('llama3-8b');
    for (const [id, site] of answer.sites) {
      expect(site.arguments).toBe(answer.analysis?.resolved.get(id)?.arguments);
    }
  });
});

describe('compatible partners for tying and sharing', () => {
  it('offers the identity a parameter slot may join, and no other', () => {
    // `llama3-8b` binds `embed.weight` and `lm_head.weight` as two identities; the primitives
    // declare both `shareable` for the role `embedding.table`, with the same stored shape, so
    // either may join the other's identity. `final_n.weight` is `exclusive`.
    const embed = siteOf('llama3-8b', 'embed').parameters[0];
    expect(embed?.identity).toBe('embed.weight');
    expect(embed?.tiesWith).toEqual([{ identity: 'lm_head.weight', rule: 'lm_head.weight' }]);
    const head = siteOf('llama3-8b', 'lm_head').parameters[0];
    expect(head?.tiesWith).toEqual([{ identity: 'embed.weight', rule: 'embed.weight' }]);
    expect(siteOf('llama3-8b', 'final_n').parameters[0]?.tiesWith).toEqual([]);
  });

  it('offers nothing where the only other identity is incompatible', () => {
    // On `qwen3.5-4b-text` the two are already one identity, so the list of an *other* identity
    // to join is empty — and why `final_n.weight` is not in it is `check`'s answer, not a row of
    // the list (inventory §7: "with the reason a candidate is excluded").
    const embed = siteOf('qwen3.5-4b-text', 'embed').parameters[0];
    expect(embed?.identity).toBe('embed.weight');
    expect(embed?.tiesWith).toEqual([]);
  });

  it('offers the state identities a port may share, and no other', () => {
    // Every `decoder/attn[layer=i].kv` of `llama3-8b` selects the same rule under the same
    // stream with the same payload, so each may join any of the other thirty-one.
    const kv = siteOf('llama3-8b', 'decoder/attn[layer=0]').states[0];
    expect(kv?.sharesWith).toHaveLength(31);
    expect(kv?.sharesWith.map((one) => one.identity)).not.toContain('decoder.attn.kv[layer=0]');
    expect(kv?.sharesWith[0]).toEqual({
      identity: 'decoder.attn.kv[layer=1]',
      rule: 'decoder.attn.kv',
    });
  });

  it('keeps the two families of `whisper-large-v3` apart', () => {
    // A self-attention `kv` appends along the token stream; a cross-attention `kv` is indexed by
    // its `source_values` port and appends along the audio stream. V9 refuses a shared
    // allocation, so neither family is offered the other's identities.
    const self = siteOf('whisper-large-v3', 'decoder/self_attn[layer=0]').states[0];
    const cross = siteOf('whisper-large-v3', 'decoder/cross_attn[layer=0]').states[0];
    expect(self?.sharesWith).toHaveLength(31);
    expect(cross?.sharesWith).toHaveLength(31);
    expect(self?.sharesWith.every((one) => one.rule === 'decoder.self_attn.kv')).toBe(true);
    expect(cross?.sharesWith.every((one) => one.rule === 'decoder.cross_attn.kv')).toBe(true);
  });

  it('offers no partner to an absent slot or port', () => {
    const attn = siteOf('shieldstral-3b-composite', 'vision/attn[layer=0]');
    expect(attn.states[0]?.present).toBe(false);
    expect(attn.states[0]?.sharesWith).toEqual([]);
    expect(attn.parameters.find((slot) => slot.name === 'q_gated')?.tiesWith).toEqual([]);
  });
});

/** Two descriptions compared where they are comparable: the facts, not the objects. */
function shape(answer: Description): unknown {
  return [...answer.sites].map(([id, site]) => [id, summary(site)]);
}

/** One site's facts, as a value a deep comparison can read. */
function summary(site: SiteDescription): unknown {
  const token = (value: PyValue): unknown =>
    typeof value === 'bigint' ? `int:${value}` : value;
  return {
    where: whereOfSite(site.key),
    inputs: site.inputs.map((port) => [port.name, port.present, token(port.kind)]),
    outputs: site.outputs.map((port) => [port.name, port.present, port.consumed]),
    parameters: site.parameters.map((slot) => [slot.name, slot.present, slot.identity, slot.tiesWith]),
    states: site.states.map((state) => [state.name, state.ruleIndex, state.sharesWith]),
    partitions: site.partitions.map((option) => [option.index, token(option.granularity)]),
    costs: site.costs.map((entry) => [entry.index, token(entry.value)]),
  };
}
