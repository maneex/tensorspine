import { loadSchemas } from '@tensorspine/lang';
import { describe, expect, it } from 'vitest';

import { matches, referenceIndex, referenceTags, type Occurrence } from '../src/references.js';
import { SchemaShapes } from '../src/shape.js';
import { nodeAt, pointerOf, type Path } from '../src/path.js';
import { corpusNames, corpusTree, MODEL, readRepositoryFile, shapes } from './source.js';

// The reference index is the one thing the store has to know about the grammar, and plan §1 says
// it may not know it by heart: "an item of information that can be inferred from a schema is
// never hard-coded". So the tags are *found* — the name definitions are whatever a
// `propertyNames` points at, the tags are the members bound to one — and this suite reads the
// schema file a second way to say what the answer must be.

const MODEL_SCHEMA = 'https://tensorspine.dev/schema/2.0/model.json';
const IDENTIFIER = `${MODEL_SCHEMA}#/$defs/identifier`;
const QUALIFIED = `${MODEL_SCHEMA}#/$defs/qualified_name`;

interface RawSchema {
  readonly [keyword: string]: unknown;
}

/**
 * A second reading of the model schema, by hand: every member whose subschema is a bare `$ref`
 * to a definition, and every member whose subschema is a map keyed by one.
 *
 * It is deliberately naive — a walk of the file's own keywords, one `$ref` followed by name, no
 * union flattening, no memoisation — so that it shares nothing with the module under test but
 * the file it reads.
 */
function byHand(): {
  readonly names: ReadonlySet<string>;
  readonly values: Map<string, Set<string>>;
  readonly keys: Map<string, Set<string>>;
} {
  const document = JSON.parse(readRepositoryFile('schemas/tensorspine.schema.json')) as RawSchema;
  const names = new Set<string>();
  const values = new Map<string, Set<string>>();
  const keys = new Map<string, Set<string>>();
  const refOf = (node: unknown): string | null => {
    if (node === null || typeof node !== 'object' || Array.isArray(node)) return null;
    const ref = (node as RawSchema)['$ref'];
    return typeof ref === 'string' ? `${MODEL_SCHEMA}#${ref.slice(1)}` : null;
  };
  /** What a member's subschema is, with a chain of bare `$ref`s followed by name. */
  const behind = (node: unknown): RawSchema | null => {
    let current = node;
    for (let step = 0; step < 8; step += 1) {
      if (current === null || typeof current !== 'object' || Array.isArray(current)) return null;
      const ref = (current as RawSchema)['$ref'];
      if (typeof ref !== 'string') return current as RawSchema;
      const defs = document['$defs'] as RawSchema;
      current = defs[ref.slice('#/$defs/'.length)];
    }
    return null;
  };
  const add = (map: Map<string, Set<string>>, tag: string, anchor: string): void => {
    const found = map.get(tag);
    if (found === undefined) map.set(tag, new Set([anchor]));
    else found.add(anchor);
  };
  const walk = (node: unknown): void => {
    if (Array.isArray(node)) {
      for (const item of node) walk(item);
      return;
    }
    if (node === null || typeof node !== 'object') return;
    const one = node as RawSchema;
    const named = refOf(one['propertyNames']);
    if (named !== null) names.add(named);
    const properties = one['properties'];
    if (properties !== null && typeof properties === 'object' && !Array.isArray(properties)) {
      for (const [tag, sub] of Object.entries(properties as RawSchema)) {
        const ref = refOf(sub);
        if (ref !== null) add(values, tag, ref);
        const target = behind(sub);
        if (target !== null) {
          const under = refOf(target['propertyNames']);
          if (under !== null) add(keys, tag, under);
        }
      }
    }
    for (const value of Object.values(one)) walk(value);
  };
  walk(document);
  return { names, values, keys };
}

/** The tags the hand reading finds, kept to the ones that are name definitions. */
function expected(): { values: Map<string, Set<string>>; keys: Map<string, Set<string>> } {
  const { names, values, keys } = byHand();
  const keep = (map: Map<string, Set<string>>): Map<string, Set<string>> => {
    const found = new Map<string, Set<string>>();
    for (const [tag, anchors] of map) {
      const admitted = new Set([...anchors].filter((anchor) => names.has(anchor)));
      if (admitted.size > 0) found.set(tag, admitted);
    }
    return found;
  };
  return { values: keep(values), keys: keep(keys) };
}

const tags = referenceTags(shapes(), MODEL);

describe('the tagged keys the schema declares', () => {
  it('finds the name definitions by what a `propertyNames` points at', () => {
    expect([...tags.forms].sort()).toEqual([IDENTIFIER, QUALIFIED].sort());
    expect(byHand().names).toEqual(new Set([IDENTIFIER, QUALIFIED]));
  });

  it('finds every member that holds a name, and no other', () => {
    const found = new Map([...tags.values].map(([tag, forms]) => [tag, [...forms].sort()]));
    const wanted = new Map([...expected().values].map(([tag, forms]) => [tag, [...forms].sort()]));
    expect([...found.keys()].sort()).toEqual([...wanted.keys()].sort());
    for (const [tag, forms] of wanted) expect(found.get(tag), tag).toEqual(forms);
    // The ones plan §2.1 names, so that a walk that quietly found nothing cannot pass.
    for (const tag of ['quantity', 'index', 'instance', 'composition', 'site']) {
      expect(found.get(tag), tag).toEqual([IDENTIFIER]);
    }
  });

  it('finds every map whose names are names, and no other', () => {
    const found = new Map([...tags.keys].map(([tag, forms]) => [tag, [...forms].sort()]));
    const wanted = new Map([...expected().keys].map(([tag, forms]) => [tag, [...forms].sort()]));
    expect([...found.keys()].sort()).toEqual([...wanted.keys()].sort());
    for (const [tag, forms] of wanted) expect(found.get(tag), tag).toEqual(forms);
    expect(found.get('quantities')).toEqual([IDENTIFIER]);
    expect(found.get('instances')).toEqual([IDENTIFIER]);
    // A binding's rule name is a qualified name, a quantity's a plain identifier: one map of the
    // grammar keys both, and the two are told apart by the definition and not by the member.
    expect(found.get('constants')).toEqual([IDENTIFIER, QUALIFIED].sort());
  });

  it('is a reading and not a memory: a schema that gains a tag gains it here', () => {
    const grown = {
      $schema: 'https://json-schema.org/draft/2020-12/schema',
      $id: 'https://example.test/toy.json',
      type: 'object',
      properties: {
        things: {
          type: 'object',
          propertyNames: { $ref: '#/$defs/identifier' },
          additionalProperties: { $ref: '#/$defs/thing' },
        },
      },
      $defs: {
        identifier: { type: 'string', pattern: '^[A-Za-z_][A-Za-z0-9_-]*$' },
        thing: {
          type: 'object',
          properties: { gadget: { $ref: '#/$defs/identifier' }, note: { type: 'string' } },
        },
      },
    };
    const toy = new SchemaShapes(
      loadSchemas([{ path: 'toy.json', text: JSON.stringify(grown) }], { origin: 'toy' }),
    );
    const found = referenceTags(toy, 'toy');
    expect(found.forms).toEqual(['https://example.test/toy.json#/$defs/identifier']);
    expect([...found.values.keys()]).toEqual(['gadget']);
    expect([...found.keys.keys()]).toEqual(['things']);

    const shrunk = JSON.parse(JSON.stringify(grown)) as typeof grown;
    delete (shrunk.$defs.thing.properties as Record<string, unknown>)['gadget'];
    const smaller = referenceTags(
      new SchemaShapes(
        loadSchemas([{ path: 'toy.json', text: JSON.stringify(shrunk) }], { origin: 'toy' }),
      ),
      'toy',
    );
    expect([...smaller.values.keys()]).toEqual([]);
  });
});

describe('the index of a document', () => {
  const tree = corpusTree('llama3-8b');
  const index = referenceIndex(tree, shapes(), MODEL);

  it('finds the quantity `d` where a grep of the file finds it', () => {
    const grep = readRepositoryFile('data/models/llama3-8b.json').match(/"quantity": "d"/g) ?? [];
    expect(index.of('quantity', 'd')).toHaveLength(grep.length);
    expect(grep).toHaveLength(10);
    for (const occurrence of index.of('quantity', 'd')) {
      expect(occurrence.kind).toBe('tagged');
      expect(occurrence.form).toBe(IDENTIFIER);
    }
  });

  it('finds the declaration as a key, under the map’s own member', () => {
    const declaration = index
      .of('quantities', 'd')
      .filter((one) => pointerOf(one.path) === '/quantities/d');
    expect(declaration).toHaveLength(1);
    expect(declaration[0]?.kind).toBe('key');
  });

  it('qualifies a generated selector by the composition written beside it', () => {
    const generated = index
      .of('instance', 'attn_n')
      .filter((one) => one.qualifiers['composition'] !== undefined);
    expect(generated.length).toBeGreaterThan(0);
    for (const one of generated) expect(one.qualifiers['composition']).toBe('decoder');
    const root = index.of('instance', 'embed');
    expect(root.length).toBeGreaterThan(0);
    for (const one of root) expect(one.qualifiers['composition']).toBeUndefined();
  });

  it('tells a name of the language from a primitive’s identity by the definition, not the member', () => {
    const primitives = index.all.filter((one) => one.tag === 'name');
    expect(primitives.length).toBeGreaterThan(0);
    for (const one of primitives) expect(one.form).toBe(QUALIFIED);
    // `embed` is both a root instance and a primitive of the base: one name, two definitions, and
    // a rename of the instance must not touch the primitive it instantiates.
    const both = index.all.filter((one) => one.name === 'embed');
    expect(new Set(both.map((one) => one.form))).toEqual(new Set([IDENTIFIER, QUALIFIED]));
  });

  it('selects by scope, by kind and by the absence of a qualifier', () => {
    const scoped = index.select(
      [{ tag: 'site', kind: 'tagged', under: ['compositions', 'decoder'] }],
      'attn',
    );
    expect(scoped.length).toBeGreaterThan(0);
    for (const one of scoped) expect(one.path[1]).toBe('decoder');
    expect(index.select([{ tag: 'site', kind: 'tagged', under: ['instances'] }], 'attn')).toEqual([]);
    expect(
      index.select([{ tag: 'instance', kind: 'tagged', qualifiers: { composition: null } }], 'attn_n'),
    ).toEqual([]);
    expect(
      index.select([{ tag: 'instance', kind: 'tagged', qualifiers: { composition: 'decoder' } }], 'attn_n')
        .length,
    ).toBeGreaterThan(0);
  });

  it('answers each occurrence once, in document order, however many selectors name it', () => {
    const twice = index.select(
      [
        { tag: 'quantity', kind: 'tagged' },
        { tag: 'quantity' },
      ],
      'd',
    );
    expect(twice).toHaveLength(10);
    const order = index.all.filter((one) => twice.includes(one));
    expect(order).toEqual(twice);
  });

  it('matches a selector against an occurrence on its own', () => {
    const one = index.of('quantity', 'd')[0] as Occurrence;
    expect(matches(one, { tag: 'quantity' })).toBe(true);
    expect(matches(one, { tag: 'index' })).toBe(false);
    expect(matches(one, { tag: 'quantity', kind: 'key' })).toBe(false);
    expect(matches(one, { tag: 'quantity', qualifiers: { composition: 'decoder' } })).toBe(false);
  });
});

describe('the index of every corpus document', () => {
  const names = corpusNames();

  it('reads the fifteen documents the repository holds', () => {
    expect(names.length).toBe(15);
  });

  it('records a name that is really written where it says it is', () => {
    for (const name of names) {
      const tree = corpusTree(name);
      const index = referenceIndex(tree, shapes(), MODEL);
      expect(index.all.length, name).toBeGreaterThan(0);
      for (const one of index.all) {
        if (one.kind === 'tagged') {
          expect(nodeAt(tree, one.path), `${name} ${pointerOf(one.path)}`).toBe(one.name);
        } else {
          expect(lastStep(one.path), `${name} ${pointerOf(one.path)}`).toBe(one.name);
          expect(nodeAt(tree, one.path), `${name} ${pointerOf(one.path)}`).toBeDefined();
        }
        expect(tags.forms).toContain(one.form);
      }
    }
  });
});

function lastStep(path: Path): string {
  return String(path[path.length - 1]);
}
