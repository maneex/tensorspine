import { describe, expect, it } from 'vitest';

import { primitiveCatalog, primitiveVersions, type PrimitiveIdentity } from '@tensorspine/lang';

import {
  carriedIdentity,
  identityOf,
  nearestIdentity,
  offersOf,
  PICKER,
  PRIMITIVE_IDENTITY,
  readIdentity,
  suggestIdentities,
  type IdentityOffers,
} from '../../src/library/primitives.js';
import { presentation } from '../../src/presentation/index.js';
import { library } from '../canvas/source.js';

// The one list a primitive is chosen from — feature 2.21, plan §4.6.
//
// **Nothing below types an identity as a fixture.** The catalog is the loader's, over the
// repository's own reference base, and the expectations are read out of it: a suite that wrote
// `attention.dense@1.0.0` as data it *offers* would be the very literal catching rule §1 (b)
// scans the interface for, one directory along. What the suite does type is what a **person**
// types — `att`, `attention.dens` — which is the input, not the answer.

const catalog: readonly PrimitiveIdentity[] = primitiveCatalog(library());
const versions = primitiveVersions(library());

/** The offers as the interface builds them: the catalog, and feature 2.10's version list. */
const offers: IdentityOffers = {
  identities: catalog,
  versions: (name) => versions.get(name) ?? [],
};

/** Every identity of the catalog whose name begins with a text — the answer, read not written. */
function beginning(text: string): string[] {
  return catalog.filter((one) => one.id.startsWith(text)).map((one) => one.id);
}

describe('the suggest list over the catalog', () => {
  it('offers, for `att`, every identity of the library beginning so and nothing else', () => {
    const offered = suggestIdentities(offers, 'att');
    // The answer is what the base carries, and the base carries more than one of them — a
    // single-answer case would not show that the list is a list.
    expect(offered).toEqual(beginning('att'));
    expect(offered.length).toBeGreaterThan(1);
    expect(offered.every((id) => id.includes('@'))).toBe(true);
    // And they are the library's, not the document's: each is a key of `by_id`.
    for (const id of offered) expect(catalog.some((one) => one.id === id), id).toBe(true);
  });

  it('offers the whole catalog for nothing typed, in the loader’s order', () => {
    expect(suggestIdentities(offers, '')).toEqual(catalog.map((one) => one.id));
  });

  it('offers a name’s versions once an `@` has settled the name half', () => {
    const first = catalog[0];
    expect(first).toBeDefined();
    const name = first?.name ?? '';
    expect(suggestIdentities(offers, `${name}@`)).toEqual(
      (versions.get(name) ?? []).map((version) => `${name}@${version}`),
    );
  });

  it('offers what merely carries the text after what begins with it', () => {
    // A reader who remembers the second segment types it; the list still finds the identity, and
    // the ones that begin with the text come first.
    const held = catalog.find((one) => one.name.includes('.'));
    expect(held).toBeDefined();
    const tail = (held?.name ?? '').split('.')[1] ?? '';
    const offered = suggestIdentities(offers, tail);
    expect(offered).toContain(held?.id);
    const begins = offered.filter((id) => id.startsWith(tail));
    expect(offered.slice(0, begins.length)).toEqual(begins);
  });

  it('offers nothing from a catalog that holds nothing — never another picker’s list', () => {
    expect(suggestIdentities(offersOf([]), 'att')).toEqual([]);
  });
});

describe('what a text means', () => {
  const held = { name: '', version: '' };

  it('reads an identity of the catalog as itself, both halves', () => {
    const first = catalog[0];
    expect(first).toBeDefined();
    const verdict = readIdentity(offers, first?.id ?? '', held);
    expect(verdict.kind).toBe('known');
    expect(verdict.kind === 'known' ? verdict.reference : null).toEqual({
      name: first?.name,
      version: first?.version,
    });
  });

  it('keeps the version the document holds when only a name is typed', () => {
    const first = catalog[0];
    const standing = { name: 'something.else', version: first?.version ?? '' };
    const verdict = readIdentity(offers, first?.name ?? '', standing);
    expect(verdict.kind).toBe('known');
    expect(verdict.kind === 'known' ? verdict.reference.version : '').toBe(first?.version);
  });

  it('asks rather than refuses when the catalog does not carry the name (Q5)', () => {
    const verdict = readIdentity(offers, 'not.a.primitive@1.0.0', held);
    expect(verdict.kind).toBe('unknown');
  });

  it('answers nothing for a text that names nothing, which is the field’s own revert', () => {
    expect(readIdentity(offers, '', held).kind).toBe('nothing');
    expect(readIdentity(offers, '@1.0.0', held).kind).toBe('nothing');
    expect(readIdentity(offers, 'a@b@c', held).kind).toBe('nothing');
  });

  it('reads a reference with neither half written as the empty field', () => {
    expect(identityOf({ name: '', version: '' })).toBe('');
  });
});

describe('the nearest identity, for a typo', () => {
  it('finds the one a letter away', () => {
    const wanted = catalog.find((one) => one.name.length > 6);
    expect(wanted).toBeDefined();
    const mistyped = (wanted?.name ?? '').slice(0, -1);
    expect(
      nearestIdentity(offers, { name: mistyped, version: wanted?.version ?? '' }),
    ).toBe(wanted?.id);
  });

  it('finds none where nothing is near, rather than offering noise as help', () => {
    expect(nearestIdentity(offers, { name: 'zzz', version: '1.0.0' })).toBeNull();
  });

  it('keeps the typed version where the nearest name carries it', () => {
    const wanted = catalog.find((one) => one.name.length > 6);
    const nearest = nearestIdentity(offers, {
      name: (wanted?.name ?? '').slice(0, -1),
      version: wanted?.version ?? '',
    });
    expect(nearest).toContain(`@${wanted?.version ?? ''}`);
  });
});

describe('the drop and the sheet read one list', () => {
  it('pins the catalog’s version where a drag carries a name alone', () => {
    const first = catalog[0];
    expect(carriedIdentity(offers, { primitive: first?.name ?? '' })).toEqual({
      name: first?.name,
      version: first?.version,
    });
  });

  it('keeps a version a drag names, since a palette may pin one', () => {
    const first = catalog[0];
    expect(carriedIdentity(offers, { primitive: first?.name ?? '', version: '9.9.9' })).toEqual({
      name: first?.name,
      version: '9.9.9',
    });
  });

  it('lands a name the catalog does not carry, which V1 is what reports (Q5)', () => {
    expect(carriedIdentity(offers, { primitive: 'not.a.primitive' })).toEqual({
      name: 'not.a.primitive',
      version: '',
    });
  });

  it('fails both readings when the one list is emptied — they share it', () => {
    // The claim of the feature: one list, three readers. A catalog that lost its contents loses
    // the sheet's suggestions *and* the drop's version, and neither has a second source to fall
    // back on.
    const none = offersOf([]);
    const first = catalog[0];
    expect(suggestIdentities(offers, first?.name ?? '').length).toBeGreaterThan(0);
    expect(suggestIdentities(none, first?.name ?? '')).toEqual([]);
    expect(carriedIdentity(offers, { primitive: first?.name ?? '' }).version).toBe(first?.version);
    expect(carriedIdentity(none, { primitive: first?.name ?? '' }).version).toBe('');
    expect(readIdentity(offers, first?.id ?? '', { name: '', version: '' }).kind).toBe('known');
    expect(readIdentity(none, first?.id ?? '', { name: '', version: '' }).kind).toBe('unknown');
  });
});

describe('what the field costs, per keystroke and per commit', () => {
  // §5.6 gives a keystroke 16 ms to reach the screen. The suggest list is rebuilt on every one of
  // them and the nearest match is asked for once per commit, so both are measured rather than
  // assumed — the second especially, since it is an edit distance over every name of the base.
  const runs = 200;

  it('answers the suggest list far inside the keystroke budget', () => {
    const started = performance.now();
    for (let i = 0; i < runs; i += 1) suggestIdentities(offers, 'att');
    const each = (performance.now() - started) / runs;
    console.log(`suggestIdentities over ${String(catalog.length)} identities: ${each.toFixed(3)} ms`);
    expect(each).toBeLessThan(16);
  });

  it('answers the nearest match inside a commit, over every name of the base', () => {
    const wanted = catalog[0];
    const started = performance.now();
    for (let i = 0; i < runs; i += 1) {
      nearestIdentity(offers, { name: (wanted?.name ?? '').slice(0, -1), version: '1.0.0' });
    }
    const each = (performance.now() - started) / runs;
    console.log(`nearestIdentity over ${String(catalog.length)} identities: ${each.toFixed(3)} ms`);
    expect(each).toBeLessThan(16);
  });
});

describe('the names this module holds are the presentation file’s, not a schema’s', () => {
  it('names the widget the shipped file binds at a pinned primitive', () => {
    const bound = presentation().at(
      'https://tensorspine.dev/schema/2.0/model.json#/$defs/primitive_reference',
    );
    expect(bound?.widget).toBe(PRIMITIVE_IDENTITY);
    expect(bound?.picker).toBe(PICKER.primitives);
    expect(bound?.create).toBeDefined();
  });

  it('names every picker the shipped file uses, and no more', () => {
    const used = new Set(
      presentation()
        .anchors.map((anchor) => presentation().at(anchor)?.picker)
        .filter((one): one is string => one !== undefined),
    );
    const named = new Set<string>(Object.values(PICKER));
    for (const picker of used) expect(named, picker).toContain(picker);
  });
});
