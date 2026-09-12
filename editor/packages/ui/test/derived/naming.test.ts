import { describe, expect, it } from 'vitest';

import { keptBy, namedIn, namedKinds, subjectKinds } from '../../src/derived/naming.js';
import {
  derivedReading,
  sectionsOf,
  TABLE,
  tableRows,
  type DerivedSection,
} from '../../src/derived/products.js';
import { presentation } from '../../src/presentation/index.js';
import { context, CORPUS, derivedOf, presentationSchema } from './source.js';

// §4.18's selection filter: "restricts every tab to the selected node, identity or split". It is
// driven by the schema and the bindings — a row is kept when a string it writes, at a place the
// derived schema types as an identifier, names the subject — so nothing below knows that D3 has
// `tensors` or that a split's `block` is a list of nodes.

const llama = derivedOf('llama3-8b');
const products = derivedReading(llama, context).products;

/** Every section of every product of one corpus document. */
function everySection(name: string): { product: string; section: DerivedSection }[] {
  return derivedReading(derivedOf(name), context).products.flatMap((product) =>
    sectionsOf(product, context).map((section) => ({ product: product.member, section })),
  );
}

/** How many rows of a product a subject keeps, and how many there are. */
function held(name: string, subject: Parameters<typeof keptBy>[1]): Record<string, string> {
  const found: Record<string, string> = {};
  for (const { product, section } of everySection(name)) {
    if (section.kind !== TABLE) continue;
    const all = tableRows(section, context, () => true, 0);
    const kept = tableRows(section, context, (entry) => keptBy(entry.named, subject), 0);
    found[`${product}.${section.name}`] = `${String(kept.kept)}/${String(all.kept)}`;
  }
  return found;
}

describe('what a row names, read off the schema and the bindings', () => {
  it('finds a D3 row’s identity and its members, each under what it names', () => {
    const d3 = sectionsOf(
      products.find((one) => one.member === 'd3') as (typeof products)[number],
      context,
    );
    const tensors = d3.find((one) => one.name === 'tensors');
    const entry = tensors?.kind === TABLE ? tensors.entries[0] : undefined;
    expect(entry?.named).toEqual([
      { kind: 'identity', name: 'embed.weight' },
      { kind: 'reference', name: 'embed.weight' },
    ]);
  });

  it('finds a split’s own name and every node of its block', () => {
    const d6 = sectionsOf(
      products.find((one) => one.member === 'd6') as (typeof products)[number],
      context,
    );
    const splits = d6.find((one) => one.name === 'graph_splits');
    const entry = splits?.kind === TABLE ? splits.entries[0] : undefined;
    const kinds = new Set((entry?.named ?? []).map((one) => one.kind));
    expect([...kinds].sort()).toEqual(['node', 'split']);
    expect(entry?.named[0]).toEqual({ kind: 'split', name: 'decoder[layer<=0]' });
  });

  it('reads a name nested inside a composite — a payload, a location, an endpoint', () => {
    const d1 = sectionsOf(
      products.find((one) => one.member === 'd1') as (typeof products)[number],
      context,
    );
    const edges = d1.find((one) => one.name === 'edges');
    const entry = edges?.kind === TABLE ? edges.entries[0] : undefined;
    expect(entry?.named).toEqual([
      { kind: 'node', name: 'decoder/attn[layer=0]' },
      { kind: 'node', name: 'decoder/attn_r[layer=0]' },
    ]);
  });

  it('names nothing where the schema types nothing', () => {
    const shape = context.shapes.root('derived');
    expect(namedIn({ whatever: 'text' }, context.shapes.member(shape, 'model'), context)).toEqual([]);
  });
});

describe('a node subject: the declaration the reader selected, and its iterations', () => {
  const attn = { kind: 'node', name: 'decoder/attn', label: 'attn' };

  it('keeps every iteration of the declared site, in every product', () => {
    // The selection is a *place* of the document and a place denotes a family of D1 nodes (§4.8,
    // feature 2.9's "a card is one iteration; a group box is the whole family"), so the subject is
    // the declared site and the rows are its thirty-two.
    expect(held('llama3-8b', attn)).toEqual({
      // Thirty-two nodes, two edges and four tensors each, one state and one cost correction per
      // layer, and five partition options — which is erratum E7's own count for `attention.dense`.
      'd1.nodes': '32/195',
      'd1.edges': '64/258',
      'd1.interfaces · inputs': '0/1',
      'd1.interfaces · outputs': '0/1',
      'd2.streams': '0/1',
      'd2.values': '64/196',
      'd2.graph_splits': '1/38',
      'd3.tensors': '128/291',
      'd4.states': '32/32',
      'd5.corrections': '32/32',
      'd5.sparsity': '0/1',
      // D5's splits carry a byte figure and no node at all, so no node subject ever keeps one —
      // which is the schema saying so, not a rule of the filter's.
      'd5.graph_splits': '0/38',
      // A split is kept where its *block* holds the node: thirty-six of the thirty-eight.
      'd6.graph_splits': '36/38',
      // Empty in this document, and a table all the same: the columns are the schema's.
      'd6.information_loss': '0/0',
      'd6.partition_options': '160/323',
    });
  });

  it('keeps nothing of a product that names another site', () => {
    const embed = { kind: 'node', name: 'embed', label: 'embed' };
    const rows = held('llama3-8b', embed);
    expect(rows['d4.states']).toBe('0/32');
    expect(rows['d3.tensors']).toBe('1/291');
    // `embed` is the first node of every block, so every split of D6 keeps it.
    expect(rows['d6.graph_splits']).toBe('38/38');
  });

  it('keeps a composition’s every site, since a composition is a prefix of them', () => {
    const decoder = { kind: 'node', name: 'decoder', label: 'decoder' };
    const rows = held('llama3-8b', decoder);
    expect(rows['d1.nodes']).toBe('192/195');
    expect(rows['d3.tensors']).toBe('288/291');
    expect(rows['d6.partition_options']).toBe('320/323');
  });

  it('keeps the rows of a template instance’s whole expansion', () => {
    // `shieldstral-3b-composite` writes its text tower as a template instance, and D3 names those
    // tensors `text/decoder/…`: a reader who selected `text` selected all of them.
    const rows = held('shieldstral-3b-composite', { kind: 'node', name: 'text', label: 'text' });
    const [kept = '0', total = '0'] = (rows['d3.tensors'] ?? '').split('/');
    expect(Number(kept)).toBeGreaterThan(0);
    expect(Number(kept)).toBeLessThan(Number(total));
  });
});

describe('an identity subject, and a split subject', () => {
  it('keeps an identity’s own instances, and nothing else', () => {
    const rows = held('llama3-8b', {
      kind: 'identity',
      name: 'decoder.attn.q',
      label: 'decoder.attn.q',
    });
    expect(rows['d3.tensors']).toBe('32/291');
    expect(rows['d4.states']).toBe('0/32');
  });

  it('keeps a split’s own rows across D2, D5 and D6', () => {
    const rows = held('llama3-8b', {
      kind: 'split',
      name: 'decoder[layer<=0]',
      label: 'decoder[layer<=0]',
    });
    expect(rows['d2.graph_splits']).toBe('1/38');
    expect(rows['d5.graph_splits']).toBe('1/38');
    expect(rows['d6.graph_splits']).toBe('1/38');
  });

  it('keeps every row when nothing is selected', () => {
    const all = held('llama3-8b', null);
    for (const [where, figure] of Object.entries(all)) {
      const [kept, total] = figure.split('/');
      expect(kept, where).toBe(total);
    }
  });
});

describe('the filter’s own vocabulary, held to the file that states it', () => {
  it('reads only kinds `presentation.json`’s schema admits', () => {
    const admitted = presentationSchema().$defs.binding.properties['names']?.enum ?? [];
    expect(namedKinds().filter((one) => !admitted.includes(one))).toEqual([]);
    expect(subjectKinds().filter((one) => !admitted.includes(one))).toEqual([]);
  });

  it('reads every kind the shipped bindings actually write', () => {
    // The other direction: a `names` binding whose kind nothing compares would be a link with no
    // behaviour and a filter that silently ignores a column.
    const written = new Set(
      presentation()
        .anchors.map((anchor) => presentation().at(anchor)?.names)
        .filter((one): one is string => one !== undefined),
    );
    expect([...written].sort().filter((one) => !namedKinds().includes(one))).toEqual([]);
  });

  // The whole corpus, derived and walked product by product: ten seconds under the full run's
  // load, which the default five-second budget is no claim about (feature 2.13's own reading).
  it('holds every corpus document’s every product, and refuses none of them', { timeout: 120_000 }, () => {
    for (const name of CORPUS) {
      const rows = held(name, { kind: 'node', name: 'nothing-is-called-this', label: 'x' });
      for (const [where, figure] of Object.entries(rows)) {
        expect(figure.startsWith('0/'), `${name} ${where}`).toBe(true);
      }
    }
  });
});
