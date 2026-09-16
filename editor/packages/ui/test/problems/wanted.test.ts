import { describe, expect, it } from 'vitest';

import { primitiveCatalog } from '@tensorspine/lang';
import { PROBLEM_SEVERITY, PROBLEM_SOURCE } from '@tensorspine/lang/api';

import { FIX, NOTICE, wantedPrimitives } from '../../src/problems/notices.js';
import { fixesFor } from '../../src/problems/fixes.js';
import { library } from '../canvas/source.js';

// The editor's own row for a primitive the author asked the project to declare — feature 2.21's
// *add to the project*, §4.17's `editor` source.
//
// The row exists for one reason the core cannot supply: V1 says a document names a primitive no
// base provides, and it says nothing about whether the author *meant* to declare it. That answer
// comes from the confirm, so the row carries it — and goes as soon as either half of its
// condition stops holding.

const catalog = new Set(primitiveCatalog(library()).map((one) => one.id));
const AT = '/instances/attn/primitive';
const WANTED = { pointer: AT, name: 'lab.attention', version: '1.0.0' };

/** What the document writes at the place, as the field reads it. */
function writes(held: { readonly name: string; readonly version: string } | null) {
  return () => held;
}

describe('a primitive the author asked the project to declare', () => {
  it('is one notice of the editor’s own, at the place the confirm was raised on', () => {
    const rows = wantedPrimitives({
      wanted: [WANTED],
      catalog,
      written: writes({ name: WANTED.name, version: WANTED.version }),
      file: 'models/lab.json',
    });
    expect(rows.length).toBe(1);
    const row = rows[0];
    expect(row?.rule).toBe(NOTICE.wantedPrimitive);
    expect(row?.source).toBe(PROBLEM_SOURCE.editor);
    expect(row?.severity).toBe(PROBLEM_SEVERITY.notice);
    expect(row?.path).toBe(AT);
    expect(row?.file).toBe('models/lab.json');
    expect(row?.code).toBe('');
    expect(row?.message).toContain(`${WANTED.name}@${WANTED.version}`);
  });

  it('declares the repair it is owed, and says what it is waiting for', () => {
    const row = wantedPrimitives({
      wanted: [WANTED],
      catalog,
      written: writes({ name: WANTED.name, version: WANTED.version }),
    })[0];
    expect(row?.fixes?.length).toBe(1);
    expect(row?.fixes?.[0]?.kind).toBe(FIX.createPrimitive);
    expect(row?.fixes?.[0]?.title).toContain(WANTED.name);
    // What it awaits is on the row, so a reader learns it without clicking anything.
    expect(row?.detail?.[0]?.message).toContain('neither is built yet');
  });

  it('is claimed by no provider yet, which is what makes the panel draw it as words', () => {
    const row = wantedPrimitives({
      wanted: [WANTED],
      catalog,
      written: writes({ name: WANTED.name, version: WANTED.version }),
    })[0];
    expect(row).toBeDefined();
    expect(fixesFor(row as NonNullable<typeof row>, { rows: [] })).toEqual([]);
  });

  it('goes when a base gains the identity — nothing has to withdraw it', () => {
    const held = primitiveCatalog(library())[0];
    expect(
      wantedPrimitives({
        wanted: [{ pointer: AT, name: held?.name ?? '', version: held?.version ?? '' }],
        catalog,
        written: writes({ name: held?.name ?? '', version: held?.version ?? '' }),
      }),
    ).toEqual([]);
  });

  it('goes when the place stops naming it — retyped, or deleted with its instance', () => {
    expect(
      wantedPrimitives({
        wanted: [WANTED],
        catalog,
        written: writes({ name: 'lab.something_else', version: '1.0.0' }),
      }),
    ).toEqual([]);
    expect(wantedPrimitives({ wanted: [WANTED], catalog, written: writes(null) })).toEqual([]);
  });

  it('says one identity once, however many places want it', () => {
    const rows = wantedPrimitives({
      wanted: [WANTED, { ...WANTED, pointer: '/instances/attn2/primitive' }],
      catalog,
      written: writes({ name: WANTED.name, version: WANTED.version }),
    });
    expect(rows.length).toBe(1);
  });
});
