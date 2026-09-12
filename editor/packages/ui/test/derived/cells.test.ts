import { describe, expect, it } from 'vitest';

import { renderedFormats } from '../../src/documents/figures.js';
import {
  cellOf,
  countText,
  NOTHING,
  valueFormats,
  type DerivedCell,
} from '../../src/derived/cells.js';
import { derivedReading, sectionsOf, TABLE, tableRows, TOTALS } from '../../src/derived/products.js';
import { presentation } from '../../src/presentation/index.js';
import { context, derivedOf, presentationSchema } from './source.js';

// §4.18's renderings: "presentation bindings for `bytes` …, `elements`, `count` (`{tokens: 1.0}`
// as `1 per tokens`), `status`, `shape` (`[model.vocabulary=128256, model.width=4096]`), and
// identifiers as links". Three of the six are feature 2.6's and are read from there; these are the
// three it left, and the check that the six now have exactly six renderings between them.

const llama = derivedOf('llama3-8b');
const products = derivedReading(llama, context).products;

/** The sections of one product of `llama3-8b`. */
function sectionsFor(member: string) {
  const product = products.find((one) => one.member === member);
  if (product === undefined) throw new Error(`no product ${member}`);
  return sectionsOf(product, context);
}

/** One row of a section, as the panel draws it. */
function rowOf(member: string, section: string, at: number): readonly DerivedCell[] {
  const found = sectionsFor(member).find((one) => one.name === section);
  if (found === undefined) throw new Error(`no section ${section}`);
  const rows = tableRows(found, context, () => true, at + 1);
  return rows.rows[at]?.cells ?? [];
}

/** The columns of a table section, so a case can name a cell by its member. */
function columnsOf(member: string, section: string): readonly string[] {
  const found = sectionsFor(member).find((one) => one.name === section);
  return found?.kind === TABLE ? found.columns.map((one) => one.name) : [];
}

describe('the three renderings this feature adds, and the six the editor now has', () => {
  it('renders every format its own schema admits, and nothing else', () => {
    // Feature 2.6 wrote: "`count`, `status` and `shape` are the Derived panel's (2.15) and are not
    // written here on faith". This is that open point closed: the two tables together cover the
    // enumeration exactly, so a seventh format would fail here until it had a rendering.
    const admitted = presentationSchema().$defs.binding.properties['format']?.enum ?? [];
    const rendered = [...new Set([...renderedFormats(), ...valueFormats()])].sort();
    expect(rendered).toEqual([...admitted].sort());
  });

  it('writes a count as §4.18 writes one, and a whole multiplier without its point', () => {
    // `1 per tokens` is §4.18's own wording. The number is written as `view.py`'s own JavaScript
    // writes one — the review repair `2d5d031`'s rule, one figure along — so a count of `2.0`
    // reads `2` and never `2.0`.
    expect(countText({ tokens: 1 })).toBe('1 per tokens');
    expect(countText({ audio: 0.125 })).toBe('0.125 per audio');
    expect(countText({ tokens: 1, pixels: 0.25 })).toBe('1 per tokens, 0.25 per pixels');
    expect(countText(null)).toBe(NOTHING);
    expect(countText({})).toBe(NOTHING);
  });

  it('writes a shape as the core writes one, in the D3 table', () => {
    const columns = columnsOf('d3', 'tensors');
    const cells = rowOf('d3', 'tensors', 0);
    expect(cells[columns.indexOf('shape')]?.text).toBe(
      '[model.vocabulary=128256, model.width=4096]',
    );
  });

  it('writes a qualified value as its figure with its status chip', () => {
    const operations = sectionsFor('d5').find((one) => one.name === 'operations');
    expect(operations?.kind).toBe(TOTALS);
    const fields = operations?.kind === TOTALS ? operations.fields : [];
    expect(fields.map((one) => `${one.name} ${one.cell.text} ${one.cell.status ?? ''}`)).toEqual([
      'element 15.01 Gop exact',
      'cached_position 524 288 op exact',
      'sequence — exact',
      'invocation — exact',
    ]);
  });

  it('leaves D5’s parameters a strip of three, since its status stands beside two numbers', () => {
    // Stated because it is the one place the structural reading of a qualified value says no: a
    // `status` beside *two* figures is not one figure with a chip, and inventing a pairing would
    // be the component deciding which number the status is about.
    const parameters = sectionsFor('d5').find((one) => one.name === 'parameters');
    const fields = parameters?.kind === TOTALS ? parameters.fields : [];
    expect(fields.map((one) => `${one.name} ${one.cell.text}`)).toEqual([
      'elements 8.03 G',
      'bytes 14.96 GiB',
      'status exact',
    ]);
  });
});

describe('what a cell says about the value it holds', () => {
  it('writes nothing at all as the em dash, never as a blank', () => {
    const columns = columnsOf('d3', 'tensors');
    // `embed.weight` carries a sparsity unit; the next tensor does not, and its cell says so.
    expect(rowOf('d3', 'tensors', 1)[columns.indexOf('sparsity')]?.text).toBe(NOTHING);
    expect(rowOf('d3', 'tensors', 1)[columns.indexOf('location')]?.text).not.toBe(NOTHING);
  });

  it('writes a fraction as the language writes one, never rounded to nothing', () => {
    // D3's `activated_fraction` is `7.79690618762475e-06`; `fmt_int`'s rounding would have made it
    // `0`, which is a figure the product does not carry (the component inventory's §7).
    const columns = columnsOf('d3', 'tensors');
    expect(rowOf('d3', 'tensors', 0)[columns.indexOf('sparsity')]?.text).toBe(
      '0 model.vocabulary 1 128 256 7.79690618762475e-06',
    );
  });

  it('writes a truth as a word, since the language spells one nowhere', () => {
    const columns = columnsOf('d3', 'tensors');
    expect(rowOf('d3', 'tensors', 0)[columns.indexOf('tied')]?.text).toBe('no');
  });

  it('writes a composite from its own members — a location, an endpoint, a payload', () => {
    const tensors = columnsOf('d3', 'tensors');
    const located = rowOf('d3', 'tensors', 0)[tensors.indexOf('location')];
    expect(located?.text).toBe('model.embed_tokens.weight');
    const edges = columnsOf('d1', 'edges');
    const edge = rowOf('d1', 'edges', 0);
    expect(edge[edges.indexOf('from')]?.text).toBe('decoder/attn[layer=0] output');
    const states = columnsOf('d4', 'states');
    expect(rowOf('d4', 'states', 0)[states.indexOf('payload')]?.text).toBe(
      'k state.kv bf16 [attention.kv_heads=8, attention.head_dim=128] 1 024 2.0 KiB + ' +
        'v state.kv bf16 [attention.kv_heads=8, attention.head_dim=128] 1 024 2.0 KiB',
    );
  });

  it('marks an identifier as a link, with what the binding says it names', () => {
    const columns = columnsOf('d3', 'tensors');
    const cells = rowOf('d3', 'tensors', 0);
    expect(cells[columns.indexOf('identity')]).toMatchObject({
      names: 'identity',
      name: 'embed.weight',
    });
    expect(cells[columns.indexOf('members')]?.names).toBeUndefined();
    const splits = columnsOf('d6', 'graph_splits');
    expect(rowOf('d6', 'graph_splits', 0)[splits.indexOf('graph_split')]).toMatchObject({
      names: 'split',
    });
  });

  it('renders a place with no binding at all as the product’s own number', () => {
    const cell = cellOf(
      42n,
      context.shapes.member(context.shapes.root('derived'), 'nothing-of-the-schema'),
      context,
    );
    expect(cell).toEqual({ text: '42', exact: '42', figure: true });
  });

  it('keeps the exact figure beside every rounded one', () => {
    const columns = columnsOf('d3', 'tensors');
    const cells = rowOf('d3', 'tensors', 0);
    const bytes = cells[columns.indexOf('bytes')];
    // `1002.0 MiB`, which is `view.py`'s own `fmt_bytes` — feature 2.9 measured that S1 writes
    // `1002 MiB` and that the board, not the repository, is what is wrong there.
    expect(bytes?.text).toBe('1002.0 MiB');
    expect(bytes?.exact).toBe('1 050 673 152');
  });

  it('is the binding that decides, and the binding is the one data file', () => {
    // The rule of §1 in the concrete: the byte figures of the derived schema carry `format:
    // "bytes"` in `presentation.json`, and removing one would leave the raw number — which is what
    // the audit of `tests/audit/presentation.test.ts` exists to catch.
    const anchor =
      'https://tensorspine.dev/schema/2.1/derived.json#/$defs/d3/properties/tensors/items/properties/bytes';
    expect(presentation().at(anchor)?.format).toBe('bytes');
  });
});
