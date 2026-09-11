import { describe, expect, it } from 'vitest';

import { presentation, PresentationError, readPresentation } from '../../src/presentation/load.js';
import { readRepositoryFile } from './source.js';

// Reading `presentation.json` — the one data file plan §1 admits in the interface.
//
// The decision this suite states is the loader's split: a **member name** the format does not
// carry is a refusal, because a mistyped `widgit` would otherwise do nothing at all and nobody
// would learn of it; a member **value** the interface does not know is not, because §1's "unknown
// constructs get the generic widget" is the same answer for a rendering nobody has written yet.
// The schema in `editor/schemas/` states the admissible values, and the audit layer holds the
// shipped file to it.

const file = presentation();

describe('the file the application ships', () => {
  it('reads, and keeps the anchors in the order the file writes them', () => {
    expect(file.anchors.length).toBeGreaterThan(40);
    expect(new Set(file.anchors).size).toBe(file.anchors.length);
    const written = [...readRepositoryFile('editor/packages/ui/src/presentation.json').matchAll(
      /^ {2}"([^"]+)":/gm,
    )].map((match) => match[1]);
    expect(file.anchors).toEqual(written);
  });

  it('is read once and kept', () => {
    expect(presentation()).toBe(file);
  });

  it('answers a binding by its anchor, and nothing for one it does not carry', () => {
    const anchor = file.anchors[0] as string;
    expect(file.at(anchor)).toBeDefined();
    expect(file.at(`${anchor}/nowhere`)).toBeUndefined();
  });

  it('answers the first of a chain, so a place wins over the definition it refers to', () => {
    // The store answers a place's schema chain most specific first; `firstOf` takes it in that
    // order. Proved on a pair the file actually carries: the composition's own `indices` is a
    // declaration, and `index_ranges` — what it refers to — is not bound at all.
    const place = 'https://tensorspine.dev/schema/2.0/model.json#/$defs/composition_definition/properties/indices';
    const definition = 'https://tensorspine.dev/schema/2.0/model.json#/$defs/index_ranges';
    expect(file.at(definition)).toBeUndefined();
    expect(file.firstOf([place, definition])).toBe(file.at(place));
    expect(file.firstOf([definition, place])).toBe(file.at(place));
    expect(file.firstOf([definition])).toBeUndefined();
  });
});

describe('what the loader refuses', () => {
  const refuses = (source: unknown, message: string | RegExp): void => {
    expect(() => readPresentation(source)).toThrowError(PresentationError);
    expect(() => readPresentation(source)).toThrowError(message);
  };

  it('a file that is not an object of bindings', () => {
    refuses([], /an object of bindings by anchor/);
    refuses('a', /an object of bindings by anchor/);
    refuses(null, /an object of bindings by anchor/);
  });

  it('a binding that is not an object, or says nothing', () => {
    refuses({ 'a#': 1 }, /a#: expected a binding/);
    refuses({ 'a#': {} }, /a#: a binding says nothing/);
  });

  it('a member the format does not carry — the typo that would silently do nothing', () => {
    refuses({ 'a#': { widgit: 'expression' } }, /a#: a binding carries no member 'widgit'/);
    refuses({ 'a#': { roles: 'node' } }, /a binding carries no member 'roles'/);
  });

  it('a member whose value is of the wrong shape', () => {
    refuses({ 'a#': { role: 1 } }, /a#\.role: expected a name/);
    refuses({ 'a#': { role: '' } }, /a#\.role: expected a name/);
    refuses({ 'a#': { face: 'primitive' } }, /a#\.face: expected a list of names/);
    refuses({ 'a#': { face: [1] } }, /a#\.face\[0\]: expected a name/);
    refuses({ 'a#': { structuralSummary: 'yes' } }, /a#\.structuralSummary: expected true or false/);
    refuses({ 'a#': { symbols: [] } }, /a#\.symbols: expected symbols by name/);
    refuses({ 'a#': { symbols: {} } }, /a#\.symbols: names no symbol/);
    refuses({ 'a#': { refers: {} } }, /a#\.refers: expected reference rules/);
  });

  it('a symbol without its text or without its place', () => {
    refuses({ 'a#': { symbols: { x: 'y' } } }, /a#\.symbols\.x: expected a symbol/);
    refuses(
      { 'a#': { symbols: { x: { text: '+' } } } },
      /a#\.symbols\.x: a symbol needs the place it prints in/,
    );
    refuses(
      { 'a#': { symbols: { x: { form: 'infix' } } } },
      /a#\.symbols\.x: a symbol needs the text it prints as/,
    );
    refuses(
      { 'a#': { symbols: { x: { text: '+', form: 'infix', extra: 1 } } } },
      /a#\.symbols\.x: a symbol carries no member 'extra'/,
    );
  });

  it('a reference rule without the member a name is written under', () => {
    refuses({ 'a#': { refers: [{}] } }, /a#\.refers\[0\]: a reference rule needs the member/);
    refuses({ 'a#': { refers: ['x'] } }, /a#\.refers\[0\]: expected a reference rule/);
    refuses(
      { 'a#': { refers: [{ tag: 'site', scope: 'x' }] } },
      /a#\.refers\[0\]: a reference rule carries no member 'scope'/,
    );
    refuses({ 'a#': { refers: [{ tag: 'site', kind: 2 }] } }, /a#\.refers\[0\]: 'kind' expects a name/);
  });
});

describe('what the loader lets through', () => {
  it('a rendering the interface does not know yet, which renders generically', () => {
    // §1: "A `$def`, an enum value or an operator without a presentation binding renders
    // generically … never fails, and is listed in the log." The same answer serves a *binding*
    // whose widget this build has no component for: the file is data, and a component that has no
    // reading of a value falls back exactly as it does for an unbound construct.
    const read = readPresentation({ 'a#': { widget: 'sankey', role: 'portal', format: 'furlongs' } });
    expect(read.at('a#')?.widget).toBe('sankey');
    expect(read.at('a#')?.role).toBe('portal');
  });

  it('a referent nothing refers to, which is a statement and not a gap', () => {
    const read = readPresentation({ 'a#': { declares: 'output', refers: [] } });
    expect(read.at('a#')?.refers).toEqual([]);
  });
});
