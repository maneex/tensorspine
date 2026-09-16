import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { expect, test } from '@playwright/test';

import { foreignKeys, namesIn, renameDerived, UndeclaredPlace, type Renaming } from './derived-names.js';

/**
 * The normalisation the phase-2 exit compares through — feature 2.19, the other half of
 * `scratch.spec.ts`.
 *
 * `derived-names.ts` is a renaming of the names a *document* declares, applied to a derived
 * document at the places that hold one. What holds it to account is here, and it is four things:
 *
 * 1. **It is a renaming, and nothing else** — over every derived document the oracle wrote,
 *    prefixing every name it can see and then taking the prefix off again gives the document back,
 *    member for member. A rewrite that lost anything, doubled anything, or reached into a figure
 *    could not round-trip.
 * 2. **It reaches every name it can see** — after the prefixing there is no name of the map's own
 *    domain left at any place the tables call a name. The same walk answers both questions, so
 *    "what it can see" is not an assertion made twice.
 * 3. **It hides no real difference** — a byte figure moved, a port renamed, a node dropped: each
 *    still shows through the renaming.
 * 4. **Its guard fires, and its one assumption is held to the schema** — a string at a place
 *    neither table names is a refusal rather than silence, and every *key* the walk takes for a
 *    member of the schema is one: read out of `schemas/tensorspine-derived.schema.json` itself, so
 *    a map the tables do not classify is a decision somebody takes rather than a name missed.
 *
 * And the fifth thing, which is what the exit test actually rests on: over the corpus's own
 * derived document, renaming the **binding rules** moves exactly one place, `d1/edges/#/rule`.
 * That is the claim "what the exit normalises" in the concrete, asked of the document rather than
 * written in a comment.
 */

const repository = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', '..');
const oracleOut = join(repository, 'editor/tests/oracle/out/derive');
const oracleGenerated = existsSync(oracleOut);

/** What the worked example below reads of a derived document. */
interface Read {
  readonly model: string;
  readonly d1: {
    readonly nodes: Record<string, { readonly families: readonly string[] }>;
    readonly edges: readonly { readonly rule: string }[];
    readonly interfaces: { readonly inputs: Record<string, unknown> };
  };
  readonly d2: {
    readonly values: readonly { readonly value: string }[];
    readonly streams: Record<string, unknown>;
  };
  readonly d3: { readonly tensors: readonly { readonly identity: string }[] };
  readonly d6: { readonly graph_splits: readonly { readonly graph_split: string }[] };
}

/** Every derived document the oracle wrote, by the model it belongs to. */
function derivedDocuments(): { name: string; document: unknown }[] {
  return readdirSync(oracleOut)
    .filter((file) => file.endsWith('.derived.json'))
    .sort()
    .map((file) => ({
      name: file.slice(0, -'.derived.json'.length),
      document: JSON.parse(readFileSync(join(oracleOut, file), 'utf8')) as unknown,
    }));
}

/** The classes a renaming carries, in the order the module declares them. */
const CLASSES = ['compositions', 'sites', 'indices', 'rules', 'families', 'interfaces', 'quantities'] as const;

/** A renaming that prefixes every name a document carries — total, by construction. */
function prefixed(document: unknown, prefix: string): Renaming {
  const found = namesIn(document);
  const made: Record<string, unknown> = { model: `${prefix}${found.model[0] ?? ''}` };
  for (const held of CLASSES) {
    made[held] = Object.fromEntries((found[held] ?? []).map((name) => [name, `${prefix}${name}`]));
  }
  return made;
}

/** The renaming that takes it back. */
function inverse(renaming: Renaming, model: string): Renaming {
  const made: Record<string, unknown> = { model };
  for (const held of CLASSES) {
    const map = (renaming[held] ?? {});
    made[held] = Object.fromEntries(Object.entries(map).map(([from, to]) => [to, from]));
  }
  return made;
}

/** Every place at which two JSON trees differ, as a list of paths. */
function differences(one: unknown, other: unknown, path = ''): string[] {
  if (Array.isArray(one) || Array.isArray(other)) {
    if (!Array.isArray(one) || !Array.isArray(other) || one.length !== other.length) return [path];
    return one.flatMap((held, at) => differences(held, other[at], `${path}/#`));
  }
  if (one !== null && other !== null && typeof one === 'object' && typeof other === 'object') {
    const left = Object.keys(one);
    const right = Object.keys(other);
    if (left.length !== right.length) return [`${path}/$key`];
    const found: string[] = [];
    for (const [at, key] of left.entries()) {
      if (key !== right[at]) found.push(`${path}/$key`);
      found.push(
        ...differences(
          (one as Record<string, unknown>)[key],
          (other as Record<string, unknown>)[right[at] as string],
          // A map key is one segment of the place, and a node identifier holds slashes, so it is
          // escaped as the module's own walk escapes it.
          `${path}/${key === right[at] ? key.replaceAll('/', '~1') : '*'}`,
        ),
      );
    }
    return found;
  }
  return one === other ? [] : [path];
}

/** The distinct places, as patterns, at which two trees differ. */
function places(one: unknown, other: unknown): string[] {
  const seen = new Set(differences(one, other).map((place) => place.replace(/\/[^/]*$/, (tail) => tail)));
  return [...seen].sort();
}

test.describe('the renaming of a derived document', () => {
  test.skip(!oracleGenerated, 'the oracle has not been generated in this working copy');

  test('is a renaming and nothing else: it round-trips over every derived document', () => {
    const documents = derivedDocuments();
    expect(documents.length).toBeGreaterThanOrEqual(15);
    for (const { name, document } of documents) {
      const renaming = prefixed(document, 'z_');
      const renamed = renameDerived(document, renaming);
      expect(renamed, `${name}: the renaming does something`).not.toEqual(document);
      const model = (document as { model: string }).model;
      expect(renameDerived(renamed, inverse(renaming, model)), `${name}: and undoes it`).toEqual(
        document,
      );
    }
  });

  test('reaches every name it can see, at every place the tables call one', () => {
    for (const { name, document } of derivedDocuments()) {
      const renaming = prefixed(document, 'z_');
      const found = namesIn(renameDerived(document, renaming));
      for (const held of CLASSES) {
        const domain = new Set(Object.keys((renaming[held] ?? {})));
        expect(
          (found[held] ?? []).filter((one) => domain.has(one)),
          `${name}: ${held} left behind`,
        ).toEqual([]);
      }
      expect(found.model, `${name}: the model identifier`).toEqual([renaming.model]);
    }
  });

  test('moves exactly one place when the binding rules are what is renamed', () => {
    // The claim the exit test rests on: a value rule's name is a label nothing refers to (feature
    // 2.2), and the one place a derived document writes one is the rule an emitted edge came from.
    const document = JSON.parse(
      readFileSync(join(oracleOut, 'llama3-8b.derived.json'), 'utf8'),
    ) as unknown;
    const edges = (document as { d1: { edges: { rule: string }[] } }).d1.edges;
    const written = [...new Set(edges.map((edge) => edge.rule))].sort();
    const renaming: Renaming = {
      rules: Object.fromEntries(written.map((one) => [one, `renamed.${one}`])),
    };
    expect(written.length).toBe(12);
    expect(places(document, renameDerived(document, renaming))).toEqual(['/d1/edges/#/rule']);
  });

  test('rewrites each form as its own header says, on `llama3-8b`', () => {
    const document = JSON.parse(
      readFileSync(join(oracleOut, 'llama3-8b.derived.json'), 'utf8'),
    ) as unknown;
    const renaming: Renaming = {
      model: 'llama',
      compositions: { decoder: 'stack' },
      sites: { attn: 'sequence_operator_site', embed: 'tokens_in', lm_head: 'head' },
      indices: { layer: 'depth' },
      rules: { 'decoder.attn.q': 'stack.attention.query', 'decoder.entry': 'stack.in' },
      families: { norm: 'normalisation' },
      interfaces: { tokens: 'in', logits: 'out' },
    };
    const renamed = renameDerived(document, renaming) as Read;

    expect(renamed.model).toBe('llama');
    // A node identifier: the composition, the site and the index inside its bracket.
    expect(Object.keys(renamed.d1.nodes)).toContain('stack/sequence_operator_site[depth=0]');
    expect(Object.keys(renamed.d1.nodes)).toContain('tokens_in');
    // A family, at the place `d1` writes one; and a rule, whole — where the map does not carry the
    // whole name, the composition's own prefix moves on its own, which is what carries the rules a
    // composition scopes when the composition is renamed and they are not.
    expect(renamed.d1.nodes['tokens_in']?.families).toEqual(['input', 'embedding']);
    expect(renamed.d1.edges[0]?.rule).toBe('stack.attn_r.b');
    expect(renamed.d1.edges.map((edge) => edge.rule)).toContain('stack.in');
    // A value reference keeps the port, which is the primitive's name and not the document's.
    expect(renamed.d2.values[0]?.value).toBe('tokens_in.output');
    // The interfaces, and the stream an input introduces (§2.3).
    expect(Object.keys(renamed.d1.interfaces.inputs)).toEqual(['in']);
    expect(Object.keys(renamed.d2.streams)).toEqual(['in']);
    // An identity name: the rule, whole, with its evaluated indices beside it.
    const identities = renamed.d3.tensors.map((one) => one.identity);
    expect(identities).toContain('stack.attention.query[depth=0]');
    expect(identities).toContain('stack.attn_n.weight[depth=0]');
    // A graph split: a composition's layer prefix, and a family's.
    const splits = renamed.d6.graph_splits.map((one) => one.graph_split);
    expect(splits).toContain('stack[depth<=0]');
    expect(splits).toContain('family:normalisation');
  });

  test('hides no difference a renaming could not have made', () => {
    const document = JSON.parse(
      readFileSync(join(oracleOut, 'llama3-8b.derived.json'), 'utf8'),
    ) as Record<string, Record<string, Record<string, unknown>[]>>;
    const renaming = prefixed(document, 'z_');
    const renamed = renameDerived(document, renaming);
    for (const [what, moved] of [
      [
        'a byte figure',
        (held: typeof document) => {
          (held['d3']?.['tensors']?.[0] as Record<string, unknown>)['bytes'] = 1;
        },
      ],
      [
        'a port, which is the primitive’s name and not the document’s',
        (held: typeof document) => {
          ((held['d1']?.['edges']?.[0] as Record<string, Record<string, unknown>>)['to'] as Record<
            string,
            unknown
          >)['port'] = 'elsewhere';
        },
      ],
      [
        'a node the graph no longer emits',
        (held: typeof document) => {
          held['d1']?.['edges']?.pop();
        },
      ],
    ] as const) {
      const changed = JSON.parse(JSON.stringify(document)) as typeof document;
      moved(changed);
      expect(renameDerived(changed, renaming), what).not.toEqual(renamed);
    }
  });

  test('refuses a string at a place neither table names', () => {
    const document = JSON.parse(
      readFileSync(join(oracleOut, 'llama3-8b.derived.json'), 'utf8'),
    ) as Record<string, unknown>;
    (document['d1'] as Record<string, unknown>)['provenance'] = 'a member the schema has not';
    expect(() => renameDerived(document, {})).toThrow(UndeclaredPlace);
    expect(() => renameDerived(document, {})).toThrow('/d1/provenance');
  });

  test('takes a key at an undeclared place for a member of the schema, and it is one', () => {
    // The walk's one assumption, asked of the schemas rather than asserted: every map of the
    // derived schema is in the tables, so a key at a place neither table names is the *name of a
    // member* — `input`, `to`, `kind`, `bytes`. A map the tables do not classify would show up
    // here as a key no schema declares.
    const declared = memberNames();
    for (const { name, document } of derivedDocuments()) {
      const foreign = [...new Set(foreignKeys(document).map((one) => one.key))].sort();
      expect(foreign.length, `${name}: something is read as a member`).toBeGreaterThan(10);
      expect(
        foreign.filter((key) => !declared.has(key)),
        `${name}: keys no schema declares as a member`,
      ).toEqual([]);
    }
  });
});

/** Every member name the model and derived schemas declare — what a `properties` map is keyed by. */
function memberNames(): ReadonlySet<string> {
  const found = new Set<string>();
  const walk = (node: unknown): void => {
    if (Array.isArray(node)) {
      for (const one of node) walk(one);
      return;
    }
    if (node === null || typeof node !== 'object') return;
    const held = node as Record<string, unknown>;
    const properties = held['properties'];
    if (properties !== null && typeof properties === 'object' && !Array.isArray(properties)) {
      for (const name of Object.keys(properties)) found.add(name);
    }
    for (const one of Object.values(held)) walk(one);
  };
  for (const file of ['tensorspine-derived.schema.json', 'tensorspine.schema.json']) {
    walk(JSON.parse(readFileSync(join(repository, 'schemas', file), 'utf8')));
  }
  return found;
}
