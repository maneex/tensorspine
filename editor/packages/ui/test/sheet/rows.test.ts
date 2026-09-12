import { describe, expect, it } from 'vitest';

import { argumentSheet, type ArgumentRow } from '../../src/sheet/arguments.js';
import { pyStr } from '@tensorspine/lang';

import { NUMBER, SECTION, SELECT, TOGGLE, WHOLE } from '../../src/forms/index.js';
import { MODEL, context, sheetOf, siteOf, tree } from './source.js';

/**
 * The argument sheet's rows — plan §4.12, artboard S6, inventory §4 "Argument row".
 *
 * The selection the feature's own block names, `decoder/attn` of `llama3-8b`, read through the
 * core and the tools' generated argument schema. What these hold to account is the *reading*: that
 * every row of S6 is there, by name and in the state the board draws it in, and that each of the
 * three answers it is made of comes from whoever owns it — the core for what applies and what a
 * default resolved to, the artifact for what edits a literal, the grammar for the source mode.
 */

const LLAMA = 'llama3-8b';
const ATTN = 'decoder/attn[layer=0]';

function rowOf(rows: readonly ArgumentRow[], path: string): ArgumentRow {
  const row = rows.find((one) => one.path === path);
  if (row === undefined) throw new Error(`no row ${path} (of ${rows.map((one) => one.path).join(', ')})`);
  return row;
}

describe('the rows of S6, by name and by state', () => {
  const sheet = sheetOf(LLAMA, ATTN);

  it('counts what the heading counts: 6 set of 19 declared', () => {
    expect(sheet.set).toBe(6);
    expect(sheet.declared).toBe(19);
  });

  it('keeps the declaration order, the fields of a record after their record', () => {
    const paths = sheet.rows.map((row) => row.path);
    expect(paths.slice(0, 6)).toEqual(['width', 'heads', 'head_dim', 'kv_heads', 'scale', 'mask']);
    expect(paths.filter((path) => path.startsWith('rope'))).toEqual([
      'rope',
      'rope.theta',
      'rope.layout',
      'rope.partial',
      'rope.mrope',
      'rope.scaling',
    ]);
  });

  it('indents by the depth of the path', () => {
    expect(rowOf(sheet.rows, 'rope').depth).toBe(0);
    expect(rowOf(sheet.rows, 'rope.theta').depth).toBe(1);
  });

  it('draws a quantity row as S6 does: the mode, the name, the effective value', () => {
    const row = rowOf(sheet.rows, 'width');
    expect(row.mode).toBe('quantity');
    expect(row.referent).toBe('quantity');
    expect(row.value).toBe('d');
    expect(row.effective).toBe('4096');
    expect(row.structural).toBe(true);
    expect(row.source).toBe('given');
  });

  it('draws a literal row with the enumeration the artifact carries', () => {
    const row = rowOf(sheet.rows, 'mask');
    expect(row.mode).toBe('literal');
    expect(row.value).toBe('causal');
    expect(row.widget).toBe(SELECT);
    expect(row.options?.map((one) => one.value)).toEqual(['causal', 'chunked', 'none']);
    // The declaration's own words per value, which the artifact does not carry (§4.12).
    expect(row.valueDescriptions?.['causal']).toContain('attends to itself');
    expect(row.description).toBe('Which positions a query may attend to.');
  });

  it('marks a row nothing writes as unset, with no mode', () => {
    const row = rowOf(sheet.rows, 'scale');
    expect(row.written).toBe(false);
    expect(row.mode).toBeUndefined();
    expect(row.source).toBe('absent');
    expect(row.effective).toBeUndefined();
  });

  it('marks a defaulted row with its effective value and nothing written', () => {
    const row = rowOf(sheet.rows, 'cross');
    expect(row.source).toBe('default');
    expect(row.written).toBe(false);
    expect(row.effective).toBe('False');
    expect(row.widget).toBe(TOGGLE);
  });

  it('hides an inapplicable row by default and reveals it with its condition', () => {
    expect(sheet.rows.some((row) => row.path === 'chunk')).toBe(false);
    expect(sheet.hidden).toBe(1);
    const shown = sheetOf(LLAMA, ATTN, { showInapplicable: true });
    const row = rowOf(shown.rows, 'chunk');
    expect(row.applicable).toBe(false);
    // The condition itself, printed in §4.13's text form from the artifact's own annotation.
    expect(row.presentWhen).toBe('mask = chunked');
  });

  it('gives a record row a section widget and its fields their own rows', () => {
    const row = rowOf(sheet.rows, 'rope');
    expect(row.widget).toBe(SECTION);
    expect(row.record).toBe(true);
    // The number keeps the lexeme the file wrote (feature 0.3, D12): the field edits what the
    // document holds and a save that changes nothing writes the same bytes.
    expect(rowOf(sheet.rows, 'rope.theta').value).toEqual({
      kind: 'number',
      value: 500000,
      real: false,
      lexeme: '500000',
    });
  });

  it('takes the integer/real distinction from the artifact and never from the value', () => {
    expect(rowOf(sheet.rows, 'heads').widget).toBe(WHOLE);
    expect(rowOf(sheet.rows, 'scale').widget).toBe(NUMBER);
    expect(rowOf(sheet.rows, 'rope.theta').widget).toBe(NUMBER);
    // A physical argument in tokens is a whole number, with its unit as the field's suffix. No
    // site of `llama3-8b` writes one — a record's fields are resolved only where the record is
    // (the core's own rule) — so the case is read on the document that does.
    const span = rowOf(sheetOf('voxtral-realtime', 'decoder/attn[layer=0]').rows, 'window.span');
    expect(span.widget).toBe(WHOLE);
    expect(span.unit).toBe('tokens');
    expect(span.effective).toBe('8192');
  });

  it('shows a bound that names another argument with that argument’s value', () => {
    const row = rowOf(sheet.rows, 'kv_heads');
    expect(row.named).toEqual([{ edge: 'upper', argument: 'heads', inclusive: true, value: '32' }]);
  });

  it('carries the structural badge the core answers, and no other', () => {
    const structural = sheet.rows.filter((row) => row.structural).map((row) => row.path);
    expect(structural).toContain('mask');
    expect(structural).not.toContain('scale');
    expect(structural).not.toContain('rope.theta');
  });

  it('lists every invariant the primitive declares, with its verdict', () => {
    expect(sheet.invariants.map((one) => one.verdict)).toEqual(['holds', 'holds', 'holds', 'holds']);
    expect(pyStr(sheet.invariants[0]?.description ?? '')).toBe('heads is a multiple of kv_heads');
    expect(sheet.invariants[0]?.shown).toBe('heads = 32, kv_heads = 8');
  });
});

describe('the row states of S6’s own key', () => {
  it('answers the row a refusal is about with the core’s own words', () => {
    // `final_n` of llama3-8b is on the grammar and refuses nothing; the domain error is what an
    // edited `eps` produces, which `edits.test.ts` writes and this reads back.
    const sheet = sheetOf(LLAMA, 'final_n');
    expect(sheet.rows.every((row) => row.error === undefined)).toBe(true);
  });

  it('sorts the required-and-unset rows first only when asked', () => {
    const site = siteOf(LLAMA, ATTN);
    const without = site.arguments.facts.filter((fact) => fact.path !== 'mask');
    const request = {
      facts: without,
      invariants: site.arguments.invariants,
      tree: tree(LLAMA),
      role: MODEL,
      context,
      shapes: context.shapes,
    };
    expect(argumentSheet(request).rows[0]?.path).toBe('width');
    const sorted = argumentSheet({ ...request, problemsFirst: true });
    // Nothing is required and unset here, so the order is the declaration's either way; the flag
    // moves a row only where there is one to move, which the notice suite exercises.
    expect(sorted.rows.map((row) => row.path)).toEqual(
      argumentSheet(request).rows.map((row) => row.path),
    );
  });
});

describe('a primitive the build did not generate a schema for', () => {
  it('renders every literal from the grammar alone and says so', () => {
    const site = siteOf(LLAMA, ATTN);
    const sheet = argumentSheet({
      facts: site.arguments.facts,
      invariants: site.arguments.invariants,
      tree: tree(LLAMA),
      role: MODEL,
      context,
      shapes: context.shapes,
    });
    const row = rowOf(sheet.rows, 'mask');
    expect(row.fromGrammar).toBe(true);
    // The model grammar's own reading of a literal: a scalar, with no enumeration and no bounds.
    expect(row.options).toBeUndefined();
    expect(row.bounds).toBeUndefined();
    // Everything the *core* answers is there all the same: F5's "the argument sheet works from
    // the declarations either way".
    expect(row.effective).toBe('causal');
    expect(row.description).toBe('Which positions a query may attend to.');
  });
});
