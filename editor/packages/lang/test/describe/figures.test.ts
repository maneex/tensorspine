import { describe, expect, it } from 'vitest';

import { parse } from '../../src/json/index.js';
import {
  basesOf,
  boxOfSite,
  derive,
  derivedFacts,
  describe as describeDocument,
  foldedGraph,
  loadLibrary,
  noDerivedFacts,
  slotKey,
  splitMember,
  tiedCount,
  toPython,
  whereOfSite,
  type FoldedGraph,
  type PyValue,
} from '../../src/index.js';
import { nodeSource } from '../library/source.js';
import { repositoryRoot } from '../json/repository.js';
import { repositorySchemas } from '../schema/repository.js';
import { corpus, library, schemas } from './source.js';

/**
 * What a card's derived line says, and what an identity link joins — held to the products.
 *
 * The rule these tests exist for is the component inventory's §7: **"No component adds, converts
 * or rounds a byte count."** A card's `80.0 MiB params` is a sum of four of D3's rows, so the sum
 * is computed here, where D3's own totals are, and asserted against them.
 */

const source = nodeSource(repositoryRoot);

/** One corpus document derived, with the folded graph its figures are attributed to. */
function derivedOf(name: string): { derived: PyValue; graph: FoldedGraph } {
  const path = `data/models/${name}.json`;
  const tree = parse(corpus(name));
  const { bases, problem } = basesOf(path, toPython(tree));
  if (problem !== null) throw new Error(`${name}: ${problem.message}`);
  const gathered = loadLibrary(bases, { schemas: repositorySchemas(), source });
  const derived = derive(tree, { schemas: repositorySchemas(), library: gathered });
  return { derived, graph: foldedGraph(tree) };
}

describe('the derived line of a card', () => {
  it('sums D3 over llama3-8b’s decoder/attn as S2 draws it: 80 MiB and 4 KiB a position', () => {
    const { derived, graph } = derivedOf('llama3-8b');
    const facts = derivedFacts(derived, graph);
    // A card stands for one representative iteration (§4.8), so S2's figure is one layer's.
    const attn = facts.figures.get('/compositions/decoder/instances/attn');
    expect(attn?.bytes).toBe(80n * 1024n * 1024n);
    expect(attn?.bytesPerCachedPosition).toBe(4096n);
    expect(attn?.tensors).toBe(4);
    expect(attn?.states).toBe(1);
  });

  it('sums D3 over the whole family on the group box, as S3 draws it: 13 GiB and 128 KiB', () => {
    const { derived, graph } = derivedOf('llama3-8b');
    const facts = derivedFacts(derived, graph);
    const decoder = facts.figures.get('/compositions/decoder');
    // S3 prints `13.00 GiB`, which is this figure rounded: 13 GiB exactly is 13958643712, and the
    // 512 KiB above it are the thirty-two pairs of norm weights.
    expect(decoder?.bytes).toBe(13959168000n);
    expect(decoder?.bytesPerCachedPosition).toBe(32n * 4096n);
    expect(decoder?.tensors).toBe(32 * 9);
    expect(decoder?.states).toBe(32);
  });

  it('sums D3 over the root instances S1 draws', () => {
    const { derived, graph } = derivedOf('llama3-8b');
    const facts = derivedFacts(derived, graph);
    expect(facts.figures.get('/instances/embed')?.bytes).toBe(1050673152n);
    expect(facts.figures.get('/instances/final_n')?.bytes).toBe(8192n);
    expect(facts.figures.get('/instances/final_n')?.bytesPerCachedPosition).toBeNull();
  });

  it('counts every identity once over the whole document — D3’s own totals', () => {
    for (const name of ['llama3-8b', 'qwen3.5-4b-text', 'gemma3n-kvshare']) {
      const { derived, graph } = derivedOf(name);
      const facts = derivedFacts(derived, graph);
      const totals = (derived as Record<string, Record<string, Record<string, PyValue>>>)['d3']?.[
        'totals'
      ]?.['bytes'];
      expect(facts.bytes).toBe(totals);
    }
  });

  it('shows a tied identity on both cards and counts it once in the document', () => {
    const { derived, graph } = derivedOf('qwen3.5-4b-text');
    const facts = derivedFacts(derived, graph);
    expect(tiedCount(derived)).toBe(1);
    const embed = facts.memberships.get(slotKey('/instances/embed', 'weight'));
    const head = facts.memberships.get(slotKey('/instances/lm_head', 'weight'));
    expect(embed?.identity).toBe(head?.identity);
    expect(embed?.shared).toBe(true);
    expect(head?.shared).toBe(true);
    // Both cards show the same 1.18 GiB, because each says what its own instance uses…
    expect(facts.figures.get('/instances/embed')?.bytes).toBe(1271398400n);
    expect(facts.figures.get('/instances/lm_head')?.bytes).toBe(1271398400n);
    // …and the document's own sum still counts it once, so the two cards together are more than
    // the document's total less everything else.
    const roots = ['/instances/embed', '/instances/lm_head'];
    const cards = roots.reduce((sum, box) => sum + (facts.figures.get(box)?.bytes ?? 0n), 0n);
    expect(cards).toBe(2n * 1271398400n);
  });

  it('marks a located identity from D3’s own location', () => {
    const { derived, graph } = derivedOf('llama3-8b');
    const facts = derivedFacts(derived, graph);
    expect(facts.memberships.get(slotKey('/instances/embed', 'weight'))?.located).toBe(true);
    expect(facts.figures.get('/instances/embed')?.located).toBe(1);
  });
});

describe('the identity links of §4.7', () => {
  it('joins the members of a tie', () => {
    const { derived, graph } = derivedOf('qwen3.5-4b-text');
    const links = derivedFacts(derived, graph).links.filter((link) => !link.state);
    expect(links.length).toBe(1);
    expect(links[0]?.places.map((place) => place.box).sort()).toEqual([
      '/instances/embed',
      '/instances/lm_head',
    ]);
  });

  it('draws no link where every member is one card (gemma3n’s shared.sliding.kv)', () => {
    // `shared.sliding.kv` shares one state across nine iterations of `decoder/attn`, and the
    // folded canvas draws `decoder/attn` as one card: a dashed line from a chip to itself says
    // nothing, and §4.8's drill-in is where the nine iterations are drawn apart. A card stands
    // for its representative iteration, so a chip says what *that* iteration binds — which for
    // `gemma3n-kvshare` is not one of the nine.
    const { derived, graph } = derivedOf('gemma3n-kvshare');
    const facts = derivedFacts(derived, graph);
    expect(facts.links.filter((link) => link.state)).toEqual([]);
    expect(facts.memberships.get(slotKey('/compositions/decoder/instances/attn', 'kv'))).toBeDefined();
    expect(facts.figures.get('/compositions/decoder')?.states).toBe(20);
  });

  it('marks a chip shared where the card’s own identity has several members', () => {
    const { derived, graph } = derivedOf('qwen3.5-4b-text');
    const facts = derivedFacts(derived, graph);
    const shared = [...facts.memberships.values()].filter((one) => one.shared);
    expect(shared.length).toBe(2);
  });

  it('joins two cards only', () => {
    const { derived, graph } = derivedOf('qwen3.5-4b-text');
    for (const link of derivedFacts(derived, graph).links) {
      expect(new Set(link.places.map((place) => place.box)).size).toBeGreaterThan(1);
    }
  });
});

describe('the value types View ▸ Show Edge Types shows', () => {
  it('names every D2 value by the identifier a folded handle carries', () => {
    const { derived, graph } = derivedOf('llama3-8b');
    const facts = derivedFacts(derived, graph);
    const entry = graph.edges.find((edge) => edge.rule === 'decoder.entry');
    expect(entry?.from?.value).toBe('embed.output');
    expect(facts.types.get('embed.output')).toBe('bf16[tokens, model.width=4096]');
  });
});

describe('attributing a member to a box', () => {
  it('reads a member identifier as a site and a slot', () => {
    expect(splitMember('embed.weight')).toEqual({ site: 'embed', slot: 'weight' });
    expect(splitMember('decoder/attn[layer=0].q')).toEqual({
      site: 'decoder/attn[layer=0]',
      slot: 'q',
    });
    expect(splitMember('weight')).toBeNull();
  });

  it('finds the box by the longest prefix the graph has', () => {
    const graph = foldedGraph(parse(corpus('llama3-8b')));
    expect(boxOfSite(graph, 'embed')).toBe('/instances/embed');
    expect(boxOfSite(graph, 'decoder/attn[layer=17]')).toBe('/compositions/decoder/instances/attn');
    expect(boxOfSite(graph, 'nobody/here')).toBeNull();
  });

  it('puts a template instance’s whole expansion on the instance’s own card', () => {
    const { derived, graph } = derivedOf('shieldstral-3b-composite');
    const facts = derivedFacts(derived, graph);
    const text = facts.figures.get('/instances/text');
    expect(text).toBeDefined();
    // The template's sites are `text/decoder/attn[layer=0]`, and the author edits one card.
    expect(text?.tensors).toBeGreaterThan(100);
    expect(text?.states).toBe(26);
  });

  it('answers nothing for a document nothing has been derived for', () => {
    const facts = noDerivedFacts();
    expect(facts.figures.size).toBe(0);
    expect(facts.bytes).toBeNull();
    expect(facts.links).toEqual([]);
  });
});

describe('describe over the folded canvas', () => {
  it('answers one site per declared instance, not one per iteration', () => {
    const tree = parse(corpus('llama3-8b'));
    const whole = describeDocument(tree, { schemas, library });
    const folded = describeDocument(tree, { schemas, library, folded: true });
    expect(whole.sites.size).toBe(195);
    expect(folded.sites.size).toBe(9);
    expect([...folded.sites.values()].map((site) => whereOfSite(site.key)).sort()).toEqual([
      'decoder/attn[layer=0]',
      'decoder/attn_n[layer=0]',
      'decoder/attn_r[layer=0]',
      'decoder/ffn[layer=0]',
      'decoder/ffn_n[layer=0]',
      'decoder/ffn_r[layer=0]',
      'embed',
      'final_n',
      'lm_head',
    ]);
  });

  it('takes the first iteration a guard leaves, never one that is not there', () => {
    const tree = parse(corpus('gemma3n-kvshare'));
    const path = 'data/models/gemma3n-kvshare.json';
    const { bases } = basesOf(path, toPython(tree));
    const gathered = loadLibrary(bases, { schemas: repositorySchemas(), source });
    const folded = describeDocument(tree, { schemas: repositorySchemas(), library: gathered, folded: true });
    const graph = foldedGraph(tree);
    const guarded = [...graph.byPointer.values()].filter(
      (node) => node.kind === 'site' && node.guard !== null,
    );
    expect(guarded.length).toBeGreaterThan(0);
    // Every guarded site still has a card's worth of facts: the representative is a site that
    // fired, which a caller naming `[layer=0]` could not have known.
    for (const site of guarded) {
      const described = [...folded.sites.values()].filter(
        (one) => one.key.composition === site.segments[1] && one.key.name === site.name,
      );
      expect(described.length).toBe(1);
    }
  });
});
