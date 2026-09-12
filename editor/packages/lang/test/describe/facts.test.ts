import { describe, expect, it } from 'vitest';

import { analyse } from '../../src/validate/index.js';
import { toPython } from '../../src/expr/value.js';
import { parse } from '../../src/json/index.js';
import { derive } from '../../src/derive/index.js';
import { describeAnalysis, siteDerived } from '../../src/describe/index.js';
import { whereOfSite } from '../../src/validate/index.js';
import { corpus, describedCorpus, library, schemas, siteOf, sites } from './source.js';

/**
 * What `describe` gained for the sheet of §4.11 and §4.12 — feature 2.10.
 *
 * Three additions, and each is a fact the sheet must not compute for itself (inventory §7):
 *
 *  - the per-argument documentation and place, which the walk that resolved the arguments already
 *    held (`ArgumentFact.at`, `kind`, `required`, `description`, `valueDescriptions`,
 *    `deprecation`);
 *  - the count of a state port's ordered rules, so a row can say *which of how many* applies;
 *  - the compatibility knob feature 1.11 measured the need for, which leaves the partner lists out
 *    and changes nothing else.
 *
 * And `siteDerived`, the products' own rows for one site — read, never summed.
 */

const LLAMA = 'llama3-8b';
const ATTN = 'decoder/attn[layer=0]';

describe('the per-argument facts of the sheet', () => {
  const site = siteOf(LLAMA, ATTN);
  const fact = (path: string) => {
    const found = site.arguments.facts.find((one) => one.path === path);
    if (found === undefined) throw new Error(`no fact ${path}`);
    return found;
  };

  it('says where the value is written, or where it would be', () => {
    expect(fact('heads').at).toEqual([
      'compositions',
      'decoder',
      'instances',
      'attn',
      'arguments',
      'heads',
    ]);
    // A record's field is written under the tag the grammar gives it, which is exactly what an
    // interface must not have to know: the walk that resolved it says so.
    expect(fact('rope.theta').at).toEqual([
      'compositions',
      'decoder',
      'instances',
      'attn',
      'arguments',
      'rope',
      'record',
      'theta',
    ]);
    // A member the document does not write still has the place it would be written at.
    expect(fact('cross').at).toEqual([
      'compositions',
      'decoder',
      'instances',
      'attn',
      'arguments',
      'cross',
    ]);
  });

  it('says the kind its declared type carries, which no generated schema can', () => {
    expect(fact('heads').kind).toBe('cardinality');
    expect(fact('scale').kind).toBe('real');
    expect(fact('mask').kind).toBe('enum');
    expect(fact('cross').kind).toBe('boolean');
    expect(fact('rope').kind).toBe('record');
  });

  it('says what the declaration requires, present_when and all', () => {
    expect(fact('width').required).toBe(true);
    expect(fact('kv_heads').required).toBe(false);
    // Required *under* its condition: the generated schema keeps it out of `required` because it
    // cannot be unconditional there, and the sheet's badge is about the declaration.
    expect(fact('chunk').required).toBe(true);
    expect(fact('chunk').applicable).toBe(false);
  });

  it('carries the declaration’s own words: the description and the values’ (§1)', () => {
    expect(fact('mask').description).toBe('Which positions a query may attend to.');
    expect(Object.keys(fact('mask').valueDescriptions ?? {})).toEqual([
      'causal',
      'chunked',
      'none',
    ]);
    // No unit of the reference base is deprecated, so nothing carries the third — stated rather
    // than asserted from an absence somewhere else.
    expect(fact('mask').deprecation).toBeUndefined();
  });

  it('counts a state port’s ordered rules beside the one that applies', () => {
    const kv = siteOf(LLAMA, ATTN).states[0];
    expect(kv?.rules).toBe(4);
    expect(kv?.ruleIndex).toBe(3);
  });
});

describe('the compatibility knob (feature 1.11’s finding)', () => {
  const documents = ['llama3-8b', 'qwen3.5-4b-text', 'gemma3n-kvshare'];

  for (const name of documents) {
    it(`${name}: leaves the partner lists out and changes nothing else`, () => {
      const tree = parse(corpus(name));
      const analysis = analyse(toPython(tree), library);
      const full = describeAnalysis(analysis);
      const narrowed = describeAnalysis(analysis, { compatibility: false });
      expect([...narrowed.sites.keys()]).toEqual([...full.sites.keys()]);
      for (const [key, site] of full.sites) {
        const other = narrowed.sites.get(key);
        expect(other).toBeDefined();
        // The lists are empty, and the identity a slot *belongs to* is filled either way — it is
        // what a chip and a row show, and it costs one pass over the identity instances.
        expect(other?.parameters.map((slot) => slot.tiesWith)).toEqual(site.parameters.map(() => []));
        expect(other?.states.map((state) => state.sharesWith)).toEqual(site.states.map(() => []));
        expect(other?.parameters.map((slot) => slot.identity)).toEqual(
          site.parameters.map((slot) => slot.identity),
        );
        expect(other?.states.map((state) => state.identity)).toEqual(
          site.states.map((state) => state.identity),
        );
        // Everything else, member for member.
        expect({ ...other, parameters: [], states: [] }).toEqual({ ...site, parameters: [], states: [] });
      }
    });
  }

  it('is on unless a caller says otherwise', () => {
    const embed = siteOf('llama3-8b', 'embed').parameters[0];
    expect(embed?.tiesWith).toEqual([{ identity: 'lm_head.weight', rule: 'lm_head.weight' }]);
  });
});

describe('what the products say about one site', () => {
  it('is D3’s rows, D4’s, D5’s corrections and D1’s count, read as they stand', () => {
    const tree = parse(corpus(LLAMA));
    const derived = derive(tree, { schemas, library });
    const found = siteDerived(derived, ATTN);
    expect(found.tensors.map((row) => row.slot)).toEqual(['q', 'k', 'v', 'out']);
    expect(found.tensors[0]?.bytes).toBe(33554432n);
    expect(found.states.map((row) => row.port)).toEqual(['kv']);
    expect(found.states[0]?.instanceKey).toEqual(['layer', 'instance.session', 'instance.branch']);
    expect(found.corrections.map((row) => row.per)).toEqual(['cached_position']);
    expect(found.nodes).toBe(32);
    expect(found.acrossPositions).toBe(true);
  });

  it('answers nothing for a site the products do not name', () => {
    const derived = derive(parse(corpus(LLAMA)), { schemas, library });
    const found = siteDerived(derived, 'decoder/attn[layer=99]');
    expect(found.tensors).toEqual([]);
    expect(found.nodes).toBe(32);
    expect(found.acrossPositions).toBeNull();
  });

  it('attributes a row to the site its member names and to no other', () => {
    const derived = derive(parse(corpus(LLAMA)), { schemas, library });
    for (const where of sites(LLAMA).keys()) {
      for (const row of siteDerived(derived, where).tensors) {
        expect(row.identity).toContain(where.replace(/\//g, '.').replace(/\[.*$/, ''));
      }
    }
  });
});

describe('every corpus document', () => {
  it('describes every declared argument with a place and a kind', () => {
    for (const name of ['llama3-8b', 'deepseek-v4-pro', 'whisper-large-v3']) {
      for (const site of describedCorpus(name).sites.values()) {
        for (const fact of site.arguments.facts) {
          expect(fact.at.length, `${whereOfSite(site.key)} ${fact.path}`).toBeGreaterThan(1);
          expect(fact.kind, `${whereOfSite(site.key)} ${fact.path}`).not.toBe('');
        }
      }
    }
  });
});
