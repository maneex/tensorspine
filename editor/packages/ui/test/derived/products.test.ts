import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { factsOfShape } from '../../src/forms/index.js';
import {
  derivedReading,
  LIST,
  sectionsOf,
  TABLE,
  tableRows,
  TOTALS,
  type DerivedSection,
} from '../../src/derived/products.js';
import { context, CORPUS, derivedOf, repositoryRoot } from './source.js';

// §4.18's six tabs, found rather than listed: "a tab per product (D1–D6, titled with the
// products' names from the specification), each rendered generically from the derived schema —
// an array of objects becomes a table whose columns are the schema's properties in order, a map a
// keyed table". Every figure below is `tools/tensorspine --derive`'s own, through the core.

/** The derived schema as a document, so the suite can hold a table's columns to it. */
function derivedSchema(): Record<string, never> {
  return JSON.parse(
    readFileSync(join(repositoryRoot, 'schemas', 'tensorspine-derived.schema.json'), 'utf8'),
  ) as Record<string, never>;
}

/** The section of that name, or a failure naming what there was. */
function sectionOf(sections: readonly DerivedSection[], name: string): DerivedSection {
  const found = sections.find((one) => one.name === name);
  if (found === undefined) {
    throw new Error(`no section ${name} among ${sections.map((one) => one.name).join(', ')}`);
  }
  return found;
}

const llama = derivedOf('llama3-8b');
const products = derivedReading(llama, context).products;

describe('the six products of §4.18, found in the derived document', () => {
  it('is six, in the schema’s own order, with the specification’s names', () => {
    expect(products.map((one) => `${one.short} ${one.label}`)).toEqual([
      'D1 Derived Computation Graph',
      'D2 Derived Value Shapes and Lifetimes',
      'D3 Derived Parameter Tensor Inventory',
      'D4 Derived State Inventory and Behavior',
      'D5 Derived Logical Resource Requirements and Costs',
      'D6 Derived Decomposition Options',
    ]);
  });

  it('takes each one’s short name from the member the document writes it under', () => {
    // `D3` is `d3` in capitals, which is the schema's own name for the member and not a label a
    // component chose: a seventh product would be `D7` on the day the schema declared it.
    expect(products.map((one) => one.member)).toEqual(['d1', 'd2', 'd3', 'd4', 'd5', 'd6']);
    for (const product of products) expect(product.short).toBe(product.member.toUpperCase());
  });

  it('takes the line under the name from the schema’s own `description` (§1)', () => {
    const d3 = products.find((one) => one.member === 'd3');
    expect(d3?.description).toBe(
      'Parameter tensors (§7): one entry per identity instance; a tied tensor once.',
    );
  });

  it('leaves the envelope out of the tabs and names the assignment in the header', () => {
    // `schema`, `model` and `primitive_libraries` carry no binding and are therefore no tab; the
    // assignment carries `header` and is the one thing §4.18 asks the header for beside freshness.
    const reading = derivedReading(llama, context);
    expect(reading.header.map((one) => one.label)).toEqual(['assignment']);
    expect(reading.header[0]?.cell.text).toBe('—');
  });
});

describe('a product’s sections, from what the schema asserts at each member', () => {
  const d3 = sectionsOf(
    products.find((one) => one.member === 'd3') as (typeof products)[number],
    context,
  );

  it('makes an array of objects a table whose columns are the schema’s properties in order', () => {
    const tensors = sectionOf(d3, 'tensors');
    expect(tensors.kind).toBe(TABLE);
    const schema = derivedSchema() as unknown as {
      $defs: { d3: { properties: { tensors: { items: { properties: Record<string, unknown> } } } } };
    };
    const declared = Object.keys(schema.$defs.d3.properties.tensors.items.properties);
    expect(tensors.kind === TABLE ? tensors.columns.map((one) => one.name) : []).toEqual(declared);
    expect(declared).toEqual([
      'identity',
      'members',
      'primitive',
      'slot',
      'role',
      'sensitivity',
      'dtype',
      'shape',
      'multiplicity',
      'elements',
      'bytes',
      'tied',
      'sparsity',
      'location',
    ]);
  });

  it('makes an object of scalars a strip of totals, with D3’s own figures', () => {
    const totals = sectionOf(d3, 'totals');
    expect(totals.kind).toBe(TOTALS);
    const fields = totals.kind === TOTALS ? totals.fields : [];
    expect(fields.map((one) => `${one.name} ${one.cell.text}`)).toEqual([
      'tensors 291',
      'elements 8.03 G',
      'bytes 14.96 GiB',
      'tied 0',
    ]);
    // `tied` reads `0` and not the em dash: nothing at all is what `view.py`'s `fmt_bytes` writes
    // for a zero *size*, which is that binding's own rule, and a count with no binding is the
    // product's own number. §4.18's "the exact value in the tooltip" is beside the rounded one.
    expect(fields.find((one) => one.name === 'bytes')?.cell.exact).toBe('16 060 522 496');
  });

  it('lists all 291 of `llama3-8b`’s tensors, which is D3’s own total', () => {
    const tensors = sectionOf(d3, 'tensors');
    const rows = tableRows(tensors, context, () => true, 10_000);
    expect(rows.total).toBe(291);
    expect(rows.kept).toBe(291);
  });

  it('makes a map of objects a keyed table whose first column is the map’s own names', () => {
    const d1 = sectionsOf(
      products.find((one) => one.member === 'd1') as (typeof products)[number],
      context,
    );
    const nodes = sectionOf(d1, 'nodes');
    expect(nodes.kind).toBe(TABLE);
    const columns = nodes.kind === TABLE ? nodes.columns : [];
    expect(columns[0]).toMatchObject({ name: 'nodes', key: true, names: 'node' });
    expect(columns.slice(1).map((one) => one.name)).toEqual([
      'primitive',
      'arguments',
      'families',
      'across_positions',
    ]);
    const rows = tableRows(nodes, context, () => true, 3);
    expect(rows.total).toBe(195);
    // The map's own order, which is `d1.emit`'s and not a sort of this module's.
    expect(rows.rows[0]?.cells[0]).toEqual({
      text: 'decoder/attn[layer=0]',
      name: 'decoder/attn[layer=0]',
      names: 'node',
    });
  });

  it('makes a map of scalars a strip, not a table with a nameless column', () => {
    const d4 = sectionsOf(
      products.find((one) => one.member === 'd4') as (typeof products)[number],
      context,
    );
    const evolution = sectionOf(d4, 'totals · by_evolution');
    expect(evolution.kind).toBe(TOTALS);
    expect(
      evolution.kind === TOTALS ? evolution.fields.map((one) => `${one.name} ${one.cell.text}`) : [],
    ).toEqual(['append 32']);
  });

  it('makes an array of scalars a list — D1’s topological order', () => {
    const d1 = sectionsOf(
      products.find((one) => one.member === 'd1') as (typeof products)[number],
      context,
    );
    const order = sectionOf(d1, 'topological_order');
    expect(order.kind).toBe(LIST);
    const rows = tableRows(order, context, () => true, 2);
    expect(rows.total).toBe(195);
    expect(rows.rows.map((row) => row.cells[0]?.text)).toEqual(['embed', 'decoder/attn_n[layer=0]']);
  });

  it('splits an object that holds both — D2’s peak, its values and its byte counts', () => {
    const d2 = sectionsOf(
      products.find((one) => one.member === 'd2') as (typeof products)[number],
      context,
    );
    const named = d2.map((one) => `${one.kind} ${one.name}`);
    expect(named).toContain('totals peak_live');
    expect(named).toContain('list peak_live · values');
    expect(named).toContain('totals peak_live · bytes_per_invocation');
    const peak = sectionOf(d2, 'peak_live');
    expect(
      peak.kind === TOTALS ? peak.fields.map((one) => `${one.name} ${one.cell.text}`) : [],
    ).toEqual(['node lm_head', 'bytes_per_element 509 KiB']);
  });
});

describe('D4’s table, over the document the feature’s block names', () => {
  it('has one row per layer — 32 states for `llama3-8b`', () => {
    const d4 = sectionsOf(
      products.find((one) => one.member === 'd4') as (typeof products)[number],
      context,
    );
    const states = sectionOf(d4, 'states');
    expect(tableRows(states, context, () => true, 10_000).total).toBe(32);
    const totals = sectionOf(d4, 'totals');
    expect(
      totals.kind === TOTALS
        ? totals.fields.filter((one) => one.name === 'append_bytes_per_cached_position')[0]?.cell
            .text
        : '',
    ).toBe('128 KiB');
  });
});

describe('every corpus document, every product', () => {
  // The same load guard as the suite next door: the corpus derived, and every section of every
  // product of every document built.
  it('builds its sections, and every column of every table is a member the schema declares', { timeout: 120_000 }, () => {
    for (const name of CORPUS) {
      const found = derivedReading(derivedOf(name), context);
      expect(found.products.map((one) => one.member), name).toEqual([
        'd1',
        'd2',
        'd3',
        'd4',
        'd5',
        'd6',
      ]);
      for (const product of found.products) {
        for (const section of sectionsOf(product, context)) {
          if (section.kind !== TABLE) continue;
          for (const column of section.columns) {
            if (column.key === true) continue;
            // The column exists in the schema at the place it is drawn from, which is what makes
            // "the columns are the schema's properties in order" a fact rather than a wish.
            expect(factsOfShape(column.shape).states || factsOfShape(column.shape).members.length > 0,
              `${name} ${product.member} ${section.name} ${column.name}`).toBe(true);
          }
          // Every row renders, and renders as many cells as there are columns.
          const rows = tableRows(section, context, () => true, 5);
          for (const row of rows.rows) {
            expect(row.cells, `${name} ${product.member} ${section.name}`).toHaveLength(
              section.columns.length,
            );
          }
        }
      }
    }
  });
});
