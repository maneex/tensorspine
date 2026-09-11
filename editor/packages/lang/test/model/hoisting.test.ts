import { describe, expect, it } from 'vitest';

import { toJsonValue, toPython, type PyValue } from '../../src/expr/index.js';
import { parse, toPlain } from '../../src/json/index.js';
import {
  hoisted,
  HoistRecorder,
  hoistingOf,
  loadModel,
  normalise,
  NO_HOISTING,
  writtenPlace,
  type Hoisting,
} from '../../src/model/index.js';
import { readRepositoryFile } from '../json/repository.js';
import { document } from './document.js';

// §5.2 rule 7 read backwards — feature 2.8.
//
// `normalise` hoists every composition-scoped rule to the top level before any other rule
// applies, so every place the validator names is a place of the *normalised* document. Feature
// 2.7 found the consequence in the interface: `[V5] decoder.attn.norm_in` carries
// `/bindings/values/decoder.attn.norm_in`, which no file on disk has. What the map here answers
// is where each such place was written, and the rule it is held to is that it says nothing the
// hoist does not do: the record is written by `hoist` itself, member by member, as it writes
// them.
//
// Two claims run through every case below:
//
//   * **a copied place keeps its tail** — a guard, a port, a declared identity: the whole subtree
//     corresponds, so a pointer inside one maps segment for segment;
//   * **a built place loses it** — the endpoint object, the generated selector, an identity
//     §5.2 rule 7 names after the rule: there is nothing below it in the document, and the answer
//     is the nearest written place with `exact: false` beside it.

/** The hoisting of a document written as text. */
function hoistingOfText(text: string): Hoisting {
  return hoistingOf(toPython(parse(text)));
}

/** The written place of a pointer, as `<path>` or `<path> (approximate)`. */
function written(hoisting: Hoisting, pointer: string): string {
  const place = writtenPlace(hoisting, pointer);
  if (place === null) return '(not hoisted)';
  return place.exact ? place.path : `${place.path} (approximate)`;
}

describe('what `normalise` records while it hoists', () => {
  it('answers nothing for a document that carries no scoped binding', () => {
    const text = document({});
    expect(hoisted(hoistingOfText(text))).toBe(false);
    expect(hoistingOfText(text)).toBe(NO_HOISTING);
    expect(written(NO_HOISTING, '/bindings/values/c.link')).toBe('(not hoisted)');
  });

  it('maps a scoped value rule, its guard and its ports to the places they were written', () => {
    const hoisting = hoistingOfText(
      document({
        bindings:
          '{"values": {"link": {"when": {"boolean": true}, ' +
          '"from": {"site": "a", "port": "out"}, "to": {"site": "b", "port": "in"}}}}',
      }),
    );
    const at = '/bindings/values/c.link';
    const from = '/compositions/c/bindings/values/link';
    expect(written(hoisting, at)).toBe(from);
    expect(written(hoisting, `${at}/when`)).toBe(`${from}/when`);
    expect(written(hoisting, `${at}/when/boolean`)).toBe(`${from}/when/boolean`);
    expect(written(hoisting, `${at}/from/port`)).toBe(`${from}/from/port`);
    // The site name inside the generated selector is the endpoint's own `site` member.
    expect(written(hoisting, `${at}/from/instance/instance`)).toBe(`${from}/from/site`);
    // `for_each` is the composition's own `indices`, copied: a pointer into it keeps its tail.
    expect(written(hoisting, `${at}/for_each/i/stop`)).toBe('/compositions/c/indices/i/stop');
  });

  it('says where it could not be exact, rather than inventing a place', () => {
    const hoisting = hoistingOfText(
      document({
        bindings: '{"values": {"link": {"from": {"site": "a", "port": "out"}, "to": {"site": "b", "port": "in"}}}}',
      }),
    );
    const at = '/bindings/values/c.link';
    const from = '/compositions/c/bindings/values/link';
    // `kind` and `composition` are the hoist's own words: the endpoint wrote neither.
    expect(written(hoisting, `${at}/from/instance/kind`)).toBe(`${from}/from (approximate)`);
    expect(written(hoisting, `${at}/from/instance/composition`)).toBe(`${from}/from (approximate)`);
    // And an index nobody overrode is the hoist's own `{index: i}`: nothing in the document
    // wrote it, so the nearest written place is the composition's declaration of the index — and
    // the answer says it is not that place.
    expect(written(hoisting, `${at}/from/instance/indices/i`)).toBe(
      '/compositions/c/indices/i (approximate)',
    );
    expect(written(hoisting, `${at}/from/instance/indices/i/index`)).toBe(
      '/compositions/c/indices/i (approximate)',
    );
  });

  it('keeps an overridden index expression, which the endpoint did write', () => {
    const hoisting = hoistingOfText(
      document({
        bindings:
          '{"values": {"carry": {"from": {"site": "a", "indices": {"i": {"op": "subtract", ' +
          '"args": [{"index": "i"}, {"literal": 1}]}}, "port": "out"}, ' +
          '"to": {"site": "b", "port": "in"}}}}',
      }),
    );
    const from = '/compositions/c/bindings/values/carry';
    expect(written(hoisting, '/bindings/values/c.carry/from/instance/indices/i/args/1')).toBe(
      `${from}/from/indices/i/args/1`,
    );
  });

  it('passes an explicit selector through, subtree and all', () => {
    const hoisting = hoistingOfText(
      document({
        bindings:
          '{"values": {"link": {"from": {"instance": {"kind": "root", "instance": "root"}, "port": "out"}, ' +
          '"to": {"site": "b", "port": "in"}}}}',
      }),
    );
    expect(written(hoisting, '/bindings/values/c.link/from/instance/instance')).toBe(
      '/compositions/c/bindings/values/link/from/instance/instance',
    );
  });

  it('maps a parameter rule’s members, its slot names, its dtype and its location', () => {
    const hoisting = hoistingOfText(
      document({
        bindings:
          '{"parameters": {"a.w": {"members": [{"site": "a", "parameter": "w"}, ' +
          '{"site": "b", "parameter": "w"}], "dtype": {"literal": "bf16"}, ' +
          '"tensor": {"name": "shared", "indices": {}}, ' +
          '"location": {"whole": {"name": ["w"]}}}}}',
      }),
    );
    const at = '/bindings/parameters/c.a.w';
    const from = '/compositions/c/bindings/parameters/a.w';
    expect(written(hoisting, at)).toBe(from);
    expect(written(hoisting, `${at}/members/1/parameter`)).toBe(`${from}/members/1/parameter`);
    expect(written(hoisting, `${at}/members/1/instance/instance`)).toBe(`${from}/members/1/site`);
    expect(written(hoisting, `${at}/dtype/literal`)).toBe(`${from}/dtype/literal`);
    expect(written(hoisting, `${at}/tensor/name`)).toBe(`${from}/tensor/name`);
    expect(written(hoisting, `${at}/location/whole/name/0`)).toBe(`${from}/location/whole/name/0`);
  });

  it('stands an identity §5.2 rule 7 named for the rule that did not write one', () => {
    const hoisting = hoistingOfText(
      document({
        bindings: '{"states": {"a.kv": {"members": [{"site": "a", "state": "kv"}]}}}',
      }),
    );
    const from = '/compositions/c/bindings/states/a.kv';
    expect(written(hoisting, '/bindings/states/c.a.kv/identity')).toBe(`${from} (approximate)`);
    expect(written(hoisting, '/bindings/states/c.a.kv/identity/name')).toBe(`${from} (approximate)`);
  });

  it('maps a constant rule’s own member', () => {
    const hoisting = hoistingOfText(
      document({
        bindings:
          '{"constants": {"a.k": {"members": [{"site": "a", "constant": "k"}], "constant": "kappa"}}}',
      }),
    );
    expect(written(hoisting, '/bindings/constants/c.a.k/constant')).toBe(
      '/compositions/c/bindings/constants/a.k/constant',
    );
  });
});

describe('over the corpus', () => {
  const CORPUS = [
    'llama3-8b',
    'colbert-v2',
    'deepseek-v4-pro',
    'gemma3n-kvshare',
    'whisper-large-v3',
    'shieldstral-3b-composite',
  ];

  /** Every place a map answers, resolved against the document as written. */
  function resolves(tree: PyValue, pointer: string): boolean {
    let node: PyValue = tree;
    if (pointer === '') return true;
    for (const step of pointer.slice(1).split('/')) {
      const name = step.replace(/~1/g, '/').replace(/~0/g, '~');
      if (Array.isArray(node)) {
        const index = Number(name);
        if (!Number.isInteger(index) || index < 0 || index >= node.length) return false;
        node = node[index] as PyValue;
        continue;
      }
      if (node === null || typeof node !== 'object') return false;
      const held = (node as Record<string, PyValue>)[name];
      if (held === undefined) return false;
      node = held;
    }
    return true;
  }

  it.each(CORPUS)('every place %s’s map answers is a place the file has', (name) => {
    const text = readRepositoryFile(`data/models/${name}.json`);
    const tree = toPython(parse(text));
    const hoisting = hoistingOf(tree);
    expect(hoisted(hoisting)).toBe(true);
    for (const [at, place] of hoisting.places) {
      expect(resolves(tree, place.path), `${at} -> ${place.path}`).toBe(true);
    }
  });

  it.each(CORPUS)('every hoisted rule of %s is answered, and nothing else is', (name) => {
    const text = readRepositoryFile(`data/models/${name}.json`);
    const written = toPython(parse(text));
    const hoisting = hoistingOf(written);
    const model = loadModel(text);
    const bindings = model['bindings'] as Record<string, PyValue>;
    let hoistedRules = 0;
    for (const [kind, rules] of Object.entries(bindings)) {
      for (const rule of Object.keys(rules as Record<string, PyValue>)) {
        const at = `/bindings/${kind}/${rule.replace(/~/g, '~0').replace(/\//g, '~1')}`;
        const place = writtenPlace(hoisting, at);
        // A rule the author wrote at the top level is answered by nothing; a hoisted one is
        // answered by the scoped rule it came from, and by that rule exactly.
        const scoped = place !== null;
        if (!scoped) continue;
        hoistedRules += 1;
        expect(place.exact).toBe(true);
        expect(place.path.startsWith('/compositions/')).toBe(true);
        expect(resolves(written, place.path)).toBe(true);
      }
    }
    expect(hoistedRules).toBeGreaterThan(0);
  });

  it('answers llama3-8b’s own scoped rules, which is feature 2.7’s finding', () => {
    const text = readRepositoryFile('data/models/llama3-8b.json');
    const hoisting = hoistingOf(toPython(parse(text)));
    expect(written(hoisting, '/bindings/values/decoder.attn.norm_in')).toBe(
      '/compositions/decoder/bindings/values/attn.norm_in',
    );
    expect(written(hoisting, '/bindings/states/decoder.attn.kv')).toBe(
      '/compositions/decoder/bindings/states/attn.kv',
    );
    // A rule the document writes at the top level with a dot in its name is not a hoisted one:
    // the map is built from the compositions, never from splitting the name.
    expect(written(hoisting, '/bindings/values/decoder.entry')).toBe('(not hoisted)');
    expect(written(hoisting, '/bindings/values/decoder.entry.a')).toBe('(not hoisted)');
  });
});

describe('the record is a by-product of the hoist', () => {
  it('changes nothing about what `normalise` answers', () => {
    // The map costs the expansion nothing and decides nothing: the document the recorder saw is
    // the document every other stage reads.
    for (const name of ['llama3-8b', 'gemma3n-kvshare', 'colbert-v2']) {
      const text = readRepositoryFile(`data/models/${name}.json`);
      const tree = toPython(parse(text));
      const plain = (value: PyValue): string => JSON.stringify(toPlain(toJsonValue(value)));
      expect(plain(normalise(tree))).toBe(plain(normalise(tree, new HoistRecorder())));
    }
  });
});
