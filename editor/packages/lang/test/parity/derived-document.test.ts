import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { expect, it, suite } from 'vitest';

import {
  basesOf,
  derive,
  expand,
  loadLibrary,
  parse,
  toJsonValue,
  toPython,
  type Library,
  type PyRecord,
  type PyValue,
} from '../../src/index.js';
import { nodeSource } from '../library/source.js';
import { repositorySchemas } from '../schema/repository.js';
import { agrees, at, derivedCorpus, generated, inCI } from './derived.js';
import { oracleOut, readOracleManifest, repositoryRoot } from './oracle.js';

// Parity of the whole derived document (feature 1.8e): `derive.products`' own answer, envelope and
// six products, for every corpus document and the template under its documented assignment.
//
// The per-product suites (`d1`, `d2.test.ts` … `d6.test.ts`) each compare their own member; this
// one compares what a consumer actually receives — `derive(tree, …)` against
// `out/derive/<name>.derived.json`, the document `tools/tensorspine --derive` wrote — and it is
// not the sum of the others. Three things live only here:
//
//   * the **envelope**: `schema`, `model`, `primitive_libraries`, `assignment`, then the five
//     products in numerical order, which is the order `document.update` appends them in and
//     therefore part of the document's bytes;
//   * the **order the products are computed in**, which is not their numbering — D3, D4, D2, D5,
//     then D1 and the agreement, then D6 — and which the assembly is required to keep because D5
//     reads the three inventories and D6 reads D2, D4 and D1's published order;
//   * the **self-check**, `d1.self_check`: the document validated against the derived schema before
//     it is returned. `derive` raising instead of returning is what "a document it cannot vouch
//     for is not written" means for a core that writes nothing.
//
// The comparison is `derived.ts`'s: the readings deep-equal, then the two written with the core's
// serializer and compared as text. The recorded file's own bytes are not the expectation — `--derive`
// writes with `indent=1` where every other emitter of the repository writes `indent=2` — so both
// sides go through the serializer and what is compared is the document, not the tool's layout.

const schemas = repositorySchemas();
const source = nodeSource(repositoryRoot);
const libraries = new Map<string, Library>();

/** The bases a document declares, gathered once per set: every suite here derives several times. */
function libraryFor(path: string, document: PyValue): Library {
  const bases = [...basesOf(path, document).bases];
  const key = bases.join('|');
  const held = libraries.get(key);
  if (held !== undefined) return held;
  const gathered = loadLibrary(bases, { schemas, source });
  libraries.set(key, gathered);
  return gathered;
}

/** The whole derived document of one corpus document, from its text. */
function derived(path: string, assignment: PyRecord): PyRecord {
  const tree = parse(readFileSync(join(repositoryRoot, path), 'utf8'));
  return derive(tree, { schemas, library: libraryFor(path, toPython(tree)), assignment });
}

/** The recorded document, read as a document is read: `1e-05` a float, `4096` a whole number. */
function recorded(relative: string): PyValue {
  return toPython(parse(readFileSync(join(oracleOut, relative), 'utf8')));
}

suite('the derived document against the tools', () => {
  it.runIf(inCI)('has the oracle to compare with', () => {
    expect(generated).toBe(true);
  });

  it.skipIf(!generated)(
    'derives every corpus document and the template to the document the tools wrote',
    { timeout: 600_000 },
    () => {
      const manifest = readOracleManifest();
      expect(manifest.documents).toHaveLength(15);
      for (const one of manifest.documents) {
        const expected = recorded(one.derived);
        const answer = derived(one.path, at(expected, 'assignment') as PyRecord);
        agrees(answer, expected, one.path);
      }
    },
  );

  it.skipIf(!generated)(
    'writes the head the schema requires and the six products in the document’s own order',
    { timeout: 600_000 },
    () => {
      // The envelope is D1's — `d1.emit` builds it and `products` appends to it — so the five
      // products land after `d1` in the order they are *written*, which is numerical and not the
      // order they are computed in. Nothing else is added at the head: a template instance adds
      // `instances` inside the graph, and the `assignment` is the external quantities the document
      // declares, empty for a concrete model and the template's eight names for the template.
      for (const one of readOracleManifest().documents) {
        const expected = recorded(one.derived);
        const answer = derived(one.path, at(expected, 'assignment') as PyRecord);
        expect(Object.keys(answer), one.name).toEqual([
          'schema',
          'model',
          'primitive_libraries',
          'assignment',
          'd1',
          'd2',
          'd3',
          'd4',
          'd5',
          'd6',
        ]);
        expect(answer['schema'], one.name).toBe('tensorspine-derived/2.1');
        expect(Object.keys(answer), one.name).toEqual(Object.keys(expected as PyRecord));
      }
    },
  );

  it.skipIf(!generated)(
    'vouches for what it returns: every document is on the derived schema',
    { timeout: 600_000 },
    () => {
      // `derive` runs this itself and raises when it fails, so a green run of the suite above
      // already proves it. It is stated separately because it is the *reason* the suite above can
      // be trusted: the check is the emitter's, over the document as a whole, and a product that
      // wrote a member the schema has not would be refused here rather than compared away.
      for (const one of derivedCorpus()) {
        const answer = derived(one.path, at(one.expected, 'assignment') as PyRecord);
        expect(schemas.structural(toJsonValue(answer), 'derived'), one.path).toEqual([]);
      }
    },
  );

  it.skipIf(!generated)(
    'carries D1 exactly as `--d1` emits it, re-expanded rather than read off the analysis',
    { timeout: 600_000 },
    () => {
      // `products` calls `d1.emit` again instead of reading the validator's expansion: the two
      // resolve an instance's arguments differently (feature 1.7), and `_consistent` is what holds
      // them together. The `d1` of the derived document is therefore the emitter's own, which this
      // states by building it a second time and requiring the two to be the same graph.
      for (const one of derivedCorpus()) {
        const assignment = at(one.expected, 'assignment') as PyRecord;
        const answer = derived(one.path, assignment);
        const emitted = expand(one.derivation.document, libraryFor(one.path, one.derivation.document), {
          assignment,
        });
        expect(answer['d1'], one.path).toEqual(emitted['d1']);
      }
    },
  );
});
