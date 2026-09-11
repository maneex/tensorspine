import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe as suite, expect, it } from 'vitest';

import { parse } from '../../src/json/index.js';
import {
  basesOf,
  describe as describeDocument,
  loadLibrary,
  toPython,
  UNRESOLVED,
  whereOfSite,
  type DescribedShape,
  type Description,
  type Library,
  type PyRecord,
  type PyValue,
  type SiteDescription,
  type StateDescription,
} from '../../src/index.js';
import { repositorySchemas } from '../schema/repository.js';
import { nodeSource } from '../library/source.js';
import { oracleGenerated, oracleOut, readOracleManifest, repositoryRoot } from './oracle.js';

// Parity of `describe` (feature 1.6d): the facts the editor asks per site, against the products
// the tools derive from the same documents.
//
// The feature's own comparison, stated in its block: "For every corpus node, `describe`'s shapes
// equal D2's and D3's rows and the state rule equals D4's". The oracle already writes
// `--derive`'s answer for every corpus document and for the template under its documented
// assignment, so the expectation is the repository's own derived document — not a transcription
// of `derive.py` — and the three products are read where they name a node:
//
//   D2 `values`   one row per produced value, `<node>.<port>`: its `shape` and its `role` are the
//                 output port's, evaluated (`_shape(port['shape'], args)`), and its `domain` is
//                 the pair the V5 block resolved. A row named by a public input is the *input
//                 port* it feeds, so it is compared as an input port's shape.
//   D3 `tensors`  one row per parameter identity instance, written against its **first** member's
//                 slot: the shape **as stored** (`_shape(param['shape'], args, multiplicity)`),
//                 the role, and the multiplicity. That first member is what the row states, so
//                 that member's slot description is what it is compared with.
//   D4 `states`   one row per state identity instance, written against its first member's port:
//                 the rule that applies — evolution, access, sharing, `indexed_by`, span, stride
//                 — the key axes and the payload shapes.
//
// `_num`'s reading is applied here and not in the core: the products write `None` where an extent
// or a span did not resolve to a number, while `describe` answers the value it evaluated to,
// which is a fact the sheet shows and a blank is not.
//
// **The composite is compared on the intersection.** `derive` expands every template instance
// before deriving (§5.1), so `shieldstral-3b-composite`'s D2, D3 and D4 rows name the template's
// own sites under the instance's prefix, and the instance's own site is gone. `describe` works on
// the un-expanded graph — what the author edits — so the rows whose node is not a site of this
// document are skipped, and the suite states how many of each document's rows were compared, so
// that a document falling silently out of the comparison is a failure.

const inCI = process.env['CI'] !== undefined && process.env['CI'] !== '';
const generated = oracleGenerated();

const schemas = repositorySchemas();
const source = nodeSource(repositoryRoot);
const libraries = new Map<string, Library>();

function libraryFor(bases: readonly string[]): Library {
  const key = bases.join('|');
  const held = libraries.get(key);
  if (held !== undefined) return held;
  const library = loadLibrary(bases, { schemas, source });
  libraries.set(key, library);
  return library;
}

/** One derived document, read as a document is read: `1e-05` a float, `4096` a whole number. */
function derived(path: string): PyValue {
  return toPython(parse(readFileSync(join(oracleOut, path), 'utf8')));
}

/**
 * `describe` over one corpus document, under the assignment the tools derived it with.
 *
 * The assignment is read from the *derived document*, which records it (§7): it comes back
 * through the same lexeme-preserving reading as every other number, so `eps` stays the float
 * `1e-05` and `width` the whole number 3072 — the distinction feature 1.2 keeps, and the one a
 * plain `JSON.parse` of the manifest would lose.
 */
const describedOnce = new Map<string, Description>();

function described(path: string, assignment: PyRecord): Description {
  const held = describedOnce.get(path);
  if (held !== undefined) return held;
  const answer = describeOnce(path, assignment);
  describedOnce.set(path, answer);
  return answer;
}

function describeOnce(path: string, assignment: PyRecord): Description {
  const tree = parse(readFileSync(join(repositoryRoot, path), 'utf8'));
  const model = toPython(tree);
  const { bases, problem } = basesOf(path, model);
  expect(problem, path).toBe(null);
  return describeDocument(tree, {
    schemas,
    library: libraryFor(bases),
    ...(Object.keys(assignment).length === 0 ? {} : { assignment }),
  });
}

/** `_num(v)`: "a number, and `None` for anything else" — the reading D2, D3 and D4 write with. */
function num(value: PyValue): PyValue {
  if (value === UNRESOLVED || typeof value === 'boolean') return null;
  return typeof value === 'bigint' || typeof value === 'number' ? value : null;
}

/** A described shape in the products' own form: `[{axis, extent}]`, with `factors` where declared. */
function rows(shape: DescribedShape | null): PyValue {
  if (shape === null) return [];
  return shape.map((axis) => {
    const row: Record<string, PyValue> = { axis: axis.axis, extent: num(axis.extent) };
    if (axis.factors !== undefined) row['factors'] = rows(axis.factors);
    return row;
  });
}

/** The sites of one description, by the identifier D1 and the products name a node with. */
function byNode(description: Description): Map<string, SiteDescription> {
  const sites = new Map<string, SiteDescription>();
  for (const [, site] of description.sites) sites.set(whereOfSite(site.key), site);
  return sites;
}

/** `<node>.<port>` split at its last dot, as `d6` splits a member. */
function split(id: string): { node: string; port: string } {
  const at = id.lastIndexOf('.');
  return { node: id.slice(0, at), port: id.slice(at + 1) };
}

/** A member of a product row, as the fixture writes it: a string. */
function text(value: PyValue): string {
  if (typeof value !== 'string') throw new TypeError('a product names a node with a string');
  return value;
}

/** The rule facts D4 writes about a state, from `describe`'s answer. */
function ruleOf(state: StateDescription): Record<string, PyValue> {
  return {
    evolution: state.evolution,
    access: state.access,
    sharing: state.sharing,
    indexed_by_source: state.indexedBySource,
    indexed_by_port: state.indexedByPort,
    span: num(state.span),
    stride: num(state.stride),
  };
}

/** The same facts as the derived document writes them. */
function ruleOfRow(row: PyValue, get: (name: string) => PyValue): Record<string, PyValue> {
  void row;
  return {
    evolution: get('evolution'),
    access: get('access'),
    sharing: get('sharing'),
    indexed_by_source: get('indexed_by_source'),
    indexed_by_port: get('indexed_by_port'),
    span: get('span'),
    stride: get('stride'),
  };
}

/** A member of a record, or `null` where the record has none — the products omit nothing they set. */
function at(record: PyValue, name: string): PyValue {
  if (record === null || typeof record !== 'object' || Array.isArray(record)) {
    throw new TypeError('a product row is a record');
  }
  const held = (record as PyRecord)[name];
  return held === undefined ? null : held;
}

/** Every row of one product of a derived document. */
function product(document: PyValue, section: string, name: string): readonly PyValue[] {
  return at(at(document, section), name) as readonly PyValue[];
}

suite('describe against the derived products', () => {
  it.runIf(inCI)('has the oracle to compare with', () => {
    expect(generated).toBe(true);
  });

  it.skipIf(!generated)("answers every corpus node's shapes as D2 and D3 write them", { timeout: 120_000 }, () => {
    let values = 0;
    let tensors = 0;
    const skipped: string[] = [];
    for (const one of readOracleManifest().documents) {
      const document = derived(one.derived);
      const description = described(one.path, at(document, 'assignment') as PyRecord);
      expect(description.conforms, one.path).toBe(true);
      expect(
        (description.analysis?.problems ?? []).length,
        `${one.path}: the corpus is clean`,
      ).toBe(0);
      const sites = byNode(description);
      let here = 0;

      // D2: one row per value. A row carrying `input` is a public input, whose shape is "the
      // shape of the port it feeds (V4 makes every fed port agree)".
      for (const row of product(document, 'd2', 'values')) {
        const id = text(at(row, 'value'));
        const isInput = at(row, 'input') !== null;
        const where = isInput
          ? split(text(at(row, 'to') !== null ? ((at(row, 'to') as PyValue[])[0] as PyValue) : ''))
          : split(id);
        const site = sites.get(where.node);
        if (site === undefined) {
          skipped.push(one.name);
          continue;
        }
        const port = (isInput ? site.inputs : site.outputs).find(
          (one) => one.name === where.port,
        );
        expect(port, `${one.name}: ${id}`).toBeDefined();
        const found = port as NonNullable<typeof port>;
        expect(rows(found.shape), `${one.name}: ${id} shape`).toEqual(at(row, 'shape'));
        expect(found.role, `${one.name}: ${id} role`).toEqual(at(row, 'role'));
        if (!isInput) {
          const domain = at(row, 'domain');
          expect(
            found.domain === null ? null : { kind: found.domain[0], stream: found.domain[1] },
            `${one.name}: ${id} domain`,
          ).toEqual(domain);
        }
        values += 1;
        here += 1;
      }

      // D3: one row per parameter identity instance, written against its first member's slot.
      for (const row of product(document, 'd3', 'tensors')) {
        const first = (at(row, 'members') as readonly PyValue[])[0] as PyValue;
        const where = split(text(first));
        const site = sites.get(where.node);
        if (site === undefined) {
          skipped.push(one.name);
          continue;
        }
        const slot = site.parameters.find((one) => one.name === where.port);
        expect(slot, `${one.name}: ${text(first)}`).toBeDefined();
        const found = slot as NonNullable<typeof slot>;
        expect(rows(found.shape), `${one.name}: ${text(first)} stored shape`).toEqual(
          at(row, 'shape'),
        );
        expect(found.role, `${one.name}: ${text(first)} role`).toEqual(at(row, 'role'));
        // `_num(primitive_value(param['multiplicity'], args)) if 'multiplicity' in param else 1`
        expect(
          found.multiplicity === null ? 1n : num(found.multiplicity),
          `${one.name}: ${text(first)} multiplicity`,
        ).toEqual(at(row, 'multiplicity'));
        expect(found.present, `${one.name}: ${text(first)} present`).toBe(true);
        tensors += 1;
        here += 1;
      }
      expect(here, `${one.name}: rows compared`).toBeGreaterThan(0);
    }
    // Every value and every tensor of the corpus, less those inside an expanded template — and
    // the composite is the only document that has any, so no other can fall silently out of the
    // comparison.
    expect([...new Set(skipped)]).toEqual(['shieldstral-3b-composite']);
    expect(values).toBeGreaterThan(2000);
    expect(tensors).toBeGreaterThan(2000);
  });

  it.skipIf(!generated)('answers the rule that applies to every corpus state as D4 writes it', { timeout: 120_000 }, () => {
    let states = 0;
    const skipped: string[] = [];
    for (const one of readOracleManifest().documents) {
      const document = derived(one.derived);
      const description = described(one.path, at(document, 'assignment') as PyRecord);
      const sites = byNode(description);
      for (const row of product(document, 'd4', 'states')) {
        const first = (at(row, 'members') as readonly PyValue[])[0] as PyValue;
        const where = split(text(first));
        const site = sites.get(where.node);
        if (site === undefined) {
          skipped.push(one.name);
          continue;
        }
        const state = site.states.find((port) => port.name === where.port);
        expect(state, `${one.name}: ${text(first)}`).toBeDefined();
        const found = state as NonNullable<typeof state>;
        expect(found.present, `${one.name}: ${text(first)} present`).toBe(true);
        expect(ruleOf(found), `${one.name}: ${text(first)} rule`).toEqual(
          ruleOfRow(row, (name) => at(row, name)),
        );
        // "the identity's indices × the primitive's `key_axes`" (§4.4): the port's own are the
        // tail of the instance key D4 writes.
        const key = at(row, 'instance_key') as readonly PyValue[];
        expect(key.slice(key.length - found.keyAxes.length), `${one.name}: key axes`).toEqual(
          found.keyAxes,
        );
        // The payload, component by component, in the order the declaration writes them.
        expect(
          found.payload.map((component) => ({
            component: component.name,
            role: component.role,
            shape: rows(component.shape),
          })),
          `${one.name}: ${text(first)} payload`,
        ).toEqual(
          (at(row, 'payload') as readonly PyValue[]).map((component) => ({
            component: at(component, 'component'),
            role: at(component, 'role'),
            shape: at(component, 'shape'),
          })),
        );
        // `sorted({o['effect'] for o in port['operations'].values()})`
        expect(found.operations, `${one.name}: ${text(first)} operations`).toEqual(
          at(row, 'operations'),
        );
        // The stream the state grows along, as D4 resolves it.
        const stream = at(row, 'stream');
        expect(
          found.stream === null ? null : { kind: found.stream[0], stream: found.stream[1] },
          `${one.name}: ${text(first)} stream`,
        ).toEqual(stream);
        states += 1;
      }
    }
    expect([...new Set(skipped)]).toEqual(['shieldstral-3b-composite']);
    expect(states).toBeGreaterThan(300);
  });
});
