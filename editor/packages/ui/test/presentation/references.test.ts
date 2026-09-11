import { isJsonObject, parse } from '@tensorspine/lang';
import { referenceIndex, SchemaShapes, type ReferenceSelector } from '@tensorspine/store';
import { describe, expect, it } from 'vitest';

import {
  COMPOSITION,
  CONSTANT,
  INDEX,
  PUBLIC_INPUT,
  QUANTITY,
  ROOT_INSTANCE,
  site,
} from '../../../store/test/selectors.js';
import { presentation, PresentationError } from '../../src/presentation/load.js';
import { referenceSelectors, type Scope } from '../../src/presentation/selectors.js';
import type { Binding } from '../../src/presentation/types.js';
import { readRepositoryFile, registry } from './source.js';

// Feature 2.1's first finding, discharged.
//
//   > The schemas say *which* members hold a name; no keyword says what the name *refers to*. …
//   > So the index answers occurrences and the commands take `ReferenceSelector`s; that pairing
//   > belongs in `presentation.json` (2.2), keyed by these anchors.
//
// `packages/store/test/selectors.ts` is the reading feature 2.1 was built against, written as
// constants beside its suites because no file existed to hold them. This suite requires the
// bindings to *be* that reading — selector for selector — and then requires both to find the same
// occurrences in a corpus document, so the agreement is not an agreement about notation.

const MODEL = 'https://tensorspine.dev/schema/2.0/model.json';
const file = presentation();

/** The binding at an anchor, or a failure naming the anchor rather than an undefined read. */
function bindingAt(anchor: string): Binding {
  const found = file.at(anchor);
  if (found === undefined) throw new Error(`${anchor} is bound to nothing`);
  return found;
}

const DECLARATIONS = {
  quantity: `${MODEL}#/properties/quantities`,
  constant: `${MODEL}#/properties/constants`,
  instance: `${MODEL}#/properties/instances`,
  composition: `${MODEL}#/properties/compositions`,
  site: `${MODEL}#/$defs/composition_definition/properties/instances`,
  index: `${MODEL}#/$defs/composition_definition/properties/indices`,
  input: `${MODEL}#/$defs/interfaces/properties/inputs`,
  output: `${MODEL}#/$defs/interfaces/properties/outputs`,
} as const;

/** The scope a site or an index of `decoder` is read against. */
const decoder: Scope = { path: ['compositions', 'decoder'], name: 'decoder' };

describe('the bindings state what a name refers to', () => {
  it('names the kind of thing each map declares', () => {
    for (const [declares, anchor] of Object.entries(DECLARATIONS)) {
      expect(bindingAt(anchor).declares, anchor).toBe(declares);
    }
  });

  it('reproduces the selectors feature 2.1 was built against, one for one', () => {
    const same = (anchor: string, expected: readonly ReferenceSelector[], scope?: Scope): void => {
      expect(referenceSelectors(bindingAt(anchor), scope), anchor).toEqual(expected);
    };
    same(DECLARATIONS.quantity, QUANTITY);
    same(DECLARATIONS.constant, CONSTANT);
    same(DECLARATIONS.instance, ROOT_INSTANCE);
    same(DECLARATIONS.composition, COMPOSITION);
    same(DECLARATIONS.site, site('decoder'), decoder);
    same(DECLARATIONS.index, INDEX('decoder'), decoder);
    same(DECLARATIONS.input, PUBLIC_INPUT);
  });

  it('says that nothing names a public output, which is an answer and not a gap', () => {
    expect(referenceSelectors(bindingAt(DECLARATIONS.output))).toEqual([]);
  });

  it('refuses a scoped rule with no scope, rather than selecting the whole document', () => {
    expect(() => referenceSelectors(bindingAt(DECLARATIONS.site))).toThrowError(PresentationError);
    expect(() => referenceSelectors(bindingAt(DECLARATIONS.site))).toThrowError(
      /read against an enclosing declaration/,
    );
  });
});

describe('the selectors find what the store finds', () => {
  const text = readRepositoryFile('data/models/llama3-8b.json');
  const tree = parse(text);
  if (!isJsonObject(tree)) throw new TypeError('llama3-8b is not an object');
  const index = referenceIndex(tree, new SchemaShapes(registry()), 'model');

  const places = (selectors: readonly ReferenceSelector[], name: string): string[] =>
    index.select(selectors, name).map((one) => `${one.tag}:${one.kind}:${one.path.join('/')}`);

  const both = (anchor: string, expected: readonly ReferenceSelector[], name: string, scope?: Scope) => {
    const found = places(referenceSelectors(bindingAt(anchor), scope), name);
    expect(found, `${anchor} ${name}`).toEqual(places(expected, name));
    return found;
  };

  it('finds a quantity where feature 2.1 measured ten occurrences of `d`', () => {
    expect(both(DECLARATIONS.quantity, QUANTITY, 'd')).toHaveLength(10);
  });

  it('finds a root instance, and no site, which is what the absent qualifier is for', () => {
    expect(both(DECLARATIONS.instance, ROOT_INSTANCE, 'embed').length).toBeGreaterThan(0);
    // `attn_n` is a site of `decoder`, named from the top level by generated selectors — every
    // one of which writes a `composition` beside the `instance`. The root rule requires that
    // qualifier to be *absent*, so it reaches none of them, and the two readings agree on that
    // too; the site rule, read against `decoder`, reaches them all.
    expect(index.of('instance', 'attn_n').length).toBeGreaterThan(0);
    expect(both(DECLARATIONS.instance, ROOT_INSTANCE, 'attn_n')).toEqual([]);
    expect(both(DECLARATIONS.site, site('decoder'), 'attn_n', decoder).length).toBeGreaterThan(0);
  });

  it('finds a composition, its sites and its indices', () => {
    expect(both(DECLARATIONS.composition, COMPOSITION, 'decoder').length).toBeGreaterThan(0);
    expect(both(DECLARATIONS.site, site('decoder'), 'attn', decoder).length).toBeGreaterThan(0);
    expect(both(DECLARATIONS.index, INDEX('decoder'), 'layer', decoder).length).toBeGreaterThan(0);
  });

  it('finds the sites of one composition only, which is what the scope is for', () => {
    const elsewhere: Scope = { path: ['compositions', 'nowhere'], name: 'nowhere' };
    expect(places(referenceSelectors(bindingAt(DECLARATIONS.site), elsewhere), 'attn')).toEqual([]);
  });
});
