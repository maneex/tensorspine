import { describe, expect, it } from 'vitest';

import { pyStr } from '@tensorspine/lang';

import type { ArgumentRow } from '../../src/sheet/arguments.js';
import { instanceOf, sheetOf } from '../sheet/source.js';

/**
 * The sheet of `decoder/attn`, as lines — artboard S6, read as a diff rather than as a picture.
 *
 * Catching rule §1 (c) one level along: a schema change, a library change or a change to what the
 * tools generate shows up here as a reviewed diff instead of as a surprise in a component. The
 * rows are the interface's own model and every fact on them is the core's or the artifact's, so a
 * line that moves says which of the three moved.
 */

function lineOf(row: ArgumentRow): string {
  const marks: string[] = [];
  if (!row.applicable) marks.push('inapplicable');
  if (row.required) marks.push('required');
  if (row.structural) marks.push('structural');
  if (row.record) marks.push('record');
  marks.push(row.source);
  if (row.mode !== undefined) marks.push(`mode ${row.mode}`);
  if (row.value !== undefined) marks.push(`is ${valueOf(row)}`);
  if (row.effective !== undefined) marks.push(`= ${row.effective}`);
  if (row.unit !== undefined) marks.push(row.unit);
  if (row.options !== undefined) marks.push(`[${row.options.map((one) => String(one.value)).join('|')}]`);
  for (const bound of row.named) {
    marks.push(`${bound.edge} ${bound.inclusive ? '<=' : '<'} ${bound.argument} (${bound.value ?? '?'})`);
  }
  if (row.presentWhen !== undefined) marks.push(`when ${row.presentWhen}`);
  if (row.error !== undefined) marks.push(`! ${row.error}`);
  return `${'  '.repeat(row.depth)}${row.widget} ${row.label}  ${marks.join(' · ')}`;
}

function valueOf(row: ArgumentRow): string {
  const value = row.value;
  if (value === null || value === undefined) return '';
  if (typeof value === 'object') return 'lexeme' in value ? String(value.lexeme) : '';
  return String(value);
}

describe('the argument sheet of decoder/attn (S6)', () => {
  it('has these rows', () => {
    expect(sheetOf('llama3-8b', 'decoder/attn[layer=0]', { showInapplicable: true }).rows.map(lineOf))
      .toMatchSnapshot();
  });

  it('has these invariants', () => {
    expect(
      sheetOf('llama3-8b', 'decoder/attn[layer=0]').invariants.map(
        (one) => `${one.verdict} ${pyStr(one.description)} — ${one.shown}`,
      ),
    ).toMatchSnapshot();
  });
});

describe('the instance sheet of decoder/attn (S6)', () => {
  it('has these sections', () => {
    const sheet = instanceOf('llama3-8b', 'decoder/attn[layer=0]');
    expect({
      identity: `${sheet.name} · ${sheet.composition ?? ''} · ${sheet.primitive}@${sheet.version}`,
      ports: [...sheet.inputs, ...sheet.outputs].map(
        (row) => `${row.side} ${row.name} ${row.present ? `[${row.shape}]` : 'absent'}`,
      ),
      parameters: sheet.parameters.map(
        (row) => `${row.name} ${row.present ? (row.identity ?? 'unbound') : 'absent'}`,
      ),
      states: sheet.states.map(
        (row) => `${row.name} rule ${String(row.rule)} of ${String(row.rules)} ${row.identity ?? ''}`,
      ),
      partitions: sheet.partitions.map(
        (row) => `${row.target} ${row.communication.join(',')} /${row.granularity}`,
      ),
      derived: {
        tensors: sheet.derived.tensors.map((row) => `${row.identity} ${String(row.bytes)}`),
        states: sheet.derived.states.map(
          (row) => `${row.identity} ${String(row.bytesPerCachedPosition)}`,
        ),
        nodes: sheet.derived.nodes,
        acrossPositions: sheet.derived.acrossPositions,
      },
    }).toMatchSnapshot();
  });
});
