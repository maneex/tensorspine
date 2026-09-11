import { describe, expect, it } from 'vitest';

import { mergeFacts, toPlain } from '@tensorspine/lang';
import {
  FIRST_VERSION,
  nameMember,
  newDocument,
  tagOf,
  versionMember,
} from '../../src/documents/skeleton.js';
import { corpusTree, MODEL, registry, shapes } from '../source.js';

// §4.3's New Model and New Template, and the three readings they rest on.
//
// The plan writes the skeleton out as a sentence; what is asked here is that the sentence is a
// *reading of the schema* and that the code reads it rather than repeating it — which is where
// feature 2.1's open point lands ("the store writes no `const` of the language … 2.6 is where
// that decision lands").

const UNIT = 'primitive-library-unit';

describe('the three members a document is read by', () => {
  it('names the document by the one required member the schema leaves as free text', () => {
    expect(nameMember(shapes(), MODEL)).toBe('model');
  });

  it('finds the same reading on the unit schema, which is what the primitive editor needs', () => {
    expect(nameMember(shapes(), UNIT)).toBe('name');
  });

  it('names the version by the one optional member the schema leaves as free text', () => {
    expect(versionMember(shapes(), MODEL)).toBe('version');
  });

  it('reads the revision tag off the document, never out of this source', () => {
    // The tag is `tensorspine/2.0` in the corpus; what matters is that it is what the schema
    // fixes, read off the file — so the value is taken from the schema here too rather than
    // typed, which is the very rule the function exists to keep (§1, feature 2.1's refusal).
    const fixed = mergeFacts(
      shapes().member(shapes().root(MODEL), 'schema').direct.map((place) => place.node),
    ).constant;
    expect(typeof fixed).toBe('string');
    expect(tagOf(shapes(), corpusTree('llama3-8b'), MODEL)).toBe(fixed);
    // And a document that carries something else there shows what it carries, not what we knew.
    const own = newDocument(shapes(), { name: 'x' });
    expect(tagOf(shapes(), { ...own, members: [{ name: 'schema', value: 'other/9.9' }] }, MODEL)).toBe(
      'other/9.9',
    );
  });
});

describe('New Model', () => {
  it('writes the required members of the top-level schema, empty, in the schema’s order', () => {
    const made = newDocument(shapes(), { name: 'untitled' });
    expect(toPlain(made)).toEqual({
      schema: tagOf(shapes(), made, MODEL),
      model: 'untitled',
      primitive_libraries: [],
      quantities: {},
      constants: {},
      instances: {},
      compositions: {},
      bindings: { values: {}, parameters: {}, constants: {}, states: {} },
      interfaces: { inputs: {}, outputs: {} },
    });
  });

  it('writes the workspace’s default base as the one entry the list requires', () => {
    const made = newDocument(shapes(), { name: 'untitled', base: '../primitive-library/' });
    expect(toPlain(made)).toMatchObject({
      primitive_libraries: [{ base: '../primitive-library/' }],
    });
  });

  it('is refused by the grammar in exactly three ways, which is what the plan’s skeleton is', () => {
    // §4.3 writes the skeleton out and the schema refuses it: `primitive_libraries` has
    // `minItems: 1`, `interfaces.inputs` and `.outputs` have `minProperties: 1`, and the root's
    // own `anyOf` wants a non-empty `instances` or `compositions`. A new document is therefore
    // incomplete and says so in the Problems panel, which is D5's own answer one stage earlier.
    // Recorded as a test rather than papered over: the alternative is to invent a site and an
    // interface the author never asked for.
    const problems = registry().structural(newDocument(shapes(), { name: 'untitled' }), MODEL);
    expect(problems.map((one) => one.path).sort()).toEqual([
      '',
      '/interfaces/inputs',
      '/interfaces/outputs',
      '/primitive_libraries',
    ]);
    // With a base — which a workspace that holds one supplies — three are left, and every one of
    // them is something the author has still to write.
    const withBase = newDocument(shapes(), { name: 'untitled', base: '../primitive-library/' });
    expect(registry().structural(withBase, MODEL).map((one) => one.path).sort()).toEqual([
      '',
      '/interfaces/inputs',
      '/interfaces/outputs',
    ]);
  });
});

describe('New Template', () => {
  it('adds the version §4.3 gives it, where the schema writes that member', () => {
    const made = newDocument(shapes(), { name: 'decoder-causal-yarn', version: FIRST_VERSION });
    expect(made.members.map((one) => one.name)).toEqual([
      'schema',
      'model',
      'version',
      'primitive_libraries',
      'quantities',
      'constants',
      'instances',
      'compositions',
      'bindings',
      'interfaces',
    ]);
    expect(toPlain(made)).toMatchObject({ version: '1.0.0' });
  });

  it('writes the members in the order the corpus’s own template writes them', () => {
    const corpus = corpusTree('decoder-causal-yarn/1.0.0').members.map((one) => one.name);
    const made = newDocument(shapes(), { name: 'x', version: FIRST_VERSION }).members.map(
      (one) => one.name,
    );
    expect(made.filter((name) => corpus.includes(name))).toEqual(
      corpus.filter((name) => made.includes(name)),
    );
  });
});
