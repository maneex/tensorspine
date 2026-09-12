import { describe, expect, it } from 'vitest';

import {
  derivedReading,
  sectionsOf,
  TABLE,
  tableRows,
  TOTALS,
} from '../../src/derived/products.js';
import { context, derivedOf } from '../derived/source.js';

/**
 * The Derived panel's six tabs, as lines — the snapshot layer of plan §0.3.
 *
 * The panel is generated from the derived schema, so *what it shows* is a function of that schema
 * and of `presentation.json` alone: a member added to a product, a column renamed, a figure that
 * lost its binding, all show up here as a reviewed diff rather than as a discovery in front of a
 * reader. Two documents, because one alone would not carry a template instance's sub-graph or the
 * separated states of a shared identity.
 */

/** One product, written as its sections and their columns. */
function linesOf(name: string): string[] {
  const lines: string[] = [];
  for (const product of derivedReading(derivedOf(name), context).products) {
    lines.push(`${product.short} ${product.label}`);
    for (const section of sectionsOf(product, context)) {
      if (section.kind === TOTALS) {
        lines.push(
          `  totals ${section.name === '' ? '·' : section.name}: ` +
            section.fields
              .map((field) => `${field.name}=${field.cell.text}${field.cell.status ?? ''}`)
              .join(' '),
        );
        continue;
      }
      const rows = tableRows(section, context, () => true, 0);
      const columns = section.kind === TABLE ? section.columns.map((one) => one.name) : ['·'];
      lines.push(`  ${section.kind} ${section.name} [${String(rows.total)}]: ${columns.join(', ')}`);
    }
  }
  return lines;
}

describe('the Derived panel', () => {
  it('shows llama3-8b as the snapshot records', () => {
    expect(linesOf('llama3-8b')).toMatchSnapshot();
  });

  it('shows gemma3n-kvshare as the snapshot records — its shared states included', () => {
    expect(linesOf('gemma3n-kvshare')).toMatchSnapshot();
  });
});
