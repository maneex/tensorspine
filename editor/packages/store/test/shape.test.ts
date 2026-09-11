import { describe, expect, it } from 'vitest';

import { referenceTags } from '../src/references.js';
import { SchemaShapes } from '../src/shape.js';
import { registry, shapes, MODEL } from './source.js';

// `SchemaShapes` is the store's reading of the schemas, and two of its answers had no suite of
// their own until a form needed them.
//
// The first is a defect feature 2.2 found and stated rather than fixed: **the root was not
// followed**. `root()` built its shape from the root node alone, while `$ref` chains and `allOf`
// members are flattened by `follow`/`gather`, which it did not call. The model schema's root
// carries neither, so nothing showed; the unit schema's root reaches its `definition` through
// `allOf: [{if, then}, …]`, one branch per `kind`, so `definition` reduced to `{"type":
// "object"}`, no `propertyNames` of the unit schema was reachable, and `referenceTags` answered
// nothing at all. The second is `at(anchor)`, which a generated form needs because a form starts
// at a `$def` and not at a document root.
//
// Both are read against the schema files themselves, never against a list written here.

const UNIT = 'primitive-library-unit';
const UNIT_SCHEMA = 'https://tensorspine.dev/schema/2.0/primitive-library-unit.json';
const MODEL_SCHEMA = 'https://tensorspine.dev/schema/2.0/model.json';

describe('the root of a schema is followed like any other place', () => {
  it('flattens the unit root’s `allOf`, so its four definitions are reachable', () => {
    const root = shapes().root(UNIT);
    // The chain: the root, then one `allOf` member per `kind` the schema admits.
    expect(root.direct.map((place) => place.pointer)).toEqual([
      '',
      '/allOf/0',
      '/allOf/1',
      '/allOf/2',
      '/allOf/3',
    ]);
    const definition = shapes().member(root, 'definition');
    const reached = definition.all.map((place) => place.pointer);
    for (const named of [
      '/$defs/primitive_definition',
      '/$defs/axis_definition',
      '/$defs/precision_role_definition',
      '/$defs/base_definition',
    ]) {
      expect(reached, named).toContain(named);
    }
  });

  it('leaves the conditional branches out of what the place requires', () => {
    // The rule the module states: a constraint written inside a branch is not the place's own.
    // The root requires its four members and nothing a `then` adds.
    const required = [...shapes().constraintsOf(shapes().root(UNIT)).required].sort();
    expect(required).toEqual(['definition', 'kind', 'name', 'schema']);
    const definition = shapes().member(shapes().root(UNIT), 'definition');
    expect([...shapes().constraintsOf(definition).required]).toEqual([]);
  });

  it('answers the unit side’s reference tags, which it could not before', () => {
    const tags = referenceTags(shapes(), UNIT);
    // A tag is discovered, never listed: whatever a `propertyNames` points at is a *name*, and a
    // member bound to that same definition *holds* one. Every `propertyNames` the unit schema
    // writes points at the model schema's `identifier`, so the nine maps and the seven members
    // below are the whole answer, and they are the answer the walk finds — not one written here.
    expect([...tags.keys.keys()].sort()).toEqual([
      'arguments',
      'constants',
      'fields',
      'inputs',
      'operations',
      'outputs',
      'parameters',
      'payload',
      'state_ports',
    ]);
    expect([...tags.values.keys()].sort()).toEqual([
      'component',
      'from_port',
      'id',
      'name',
      'port',
      'state',
      'to_port',
    ]);
  });

  it('names an axis and a role by no tag, because no map is keyed by a qualified name', () => {
    // Worth stating rather than discovering: a port's `role` and a shape axis's `axis` are
    // `qualified_name`s, and the unit schema keys no map by one, so neither is a *name* under the
    // discovery rule and neither is found as a reference. What they refer to lives in another
    // base's units, which is a cross-base pairing `presentation.json` states (it already binds
    // both anchors to a picker) and the unit store of feature 3.2 reads.
    const tags = referenceTags(shapes(), UNIT);
    for (const named of ['axis', 'role', 'argument']) {
      expect([...tags.values.keys()], named).not.toContain(named);
    }
  });

  it('leaves the model root exactly as it was', () => {
    // The model schema's root carries no `$ref` and no `allOf`, so following it changes nothing:
    // the fix is the unit side's alone, and the corpus suites say so by staying green.
    const root = shapes().root(MODEL);
    expect(root.direct.map((place) => place.pointer)).toEqual(['']);
    const tags = referenceTags(shapes(), MODEL);
    expect([...tags.keys.keys()].length).toBe(13);
  });

  it('answers the empty shape for a role no schema of the registry carries', () => {
    expect(shapes().root('nothing-of-the-kind')).toEqual({ direct: [], all: [] });
  });
});

describe('the shape of the place an anchor names', () => {
  const shapesOf = (): SchemaShapes => shapes();

  it('follows the `$ref` chain of a definition', () => {
    const shape = shapesOf().at(`${MODEL_SCHEMA}#/$defs/instance_definition`);
    expect(shape.direct.map((place) => place.anchor)).toEqual([
      `${MODEL_SCHEMA}#/$defs/instance_definition`,
    ]);
    expect(shapesOf().propertyOrder(shape)).toContain('primitive');
  });

  it('reaches a place inside a definition, not only the definition', () => {
    const shape = shapesOf().at(`${UNIT_SCHEMA}#/$defs/argument_declaration/properties/type`);
    // The member's own subschema is a bare `$ref`, so the chain carries the target too.
    expect(shape.direct.map((place) => place.pointer)).toEqual([
      '/$defs/argument_declaration/properties/type',
      '/$defs/argument_type',
      '/$defs/argument_type/allOf/0',
      '/$defs/argument_type/allOf/1',
      '/$defs/argument_type/allOf/2',
      '/$defs/argument_type/allOf/3',
      '/$defs/argument_type/allOf/4',
      '/$defs/argument_type/allOf/5',
    ]);
  });

  it('agrees with the root of a role when the anchor is that root', () => {
    const byRole = shapesOf().root(MODEL);
    const byAnchor = shapesOf().at(`${MODEL_SCHEMA}#`);
    expect(byAnchor.all.map((place) => place.anchor)).toEqual(
      byRole.all.map((place) => place.anchor),
    );
  });

  it('answers the empty shape for an anchor that names nothing', () => {
    expect(shapesOf().at('not-an-anchor')).toEqual({ direct: [], all: [] });
    expect(shapesOf().at('https://elsewhere.example/x.json#/$defs/a')).toEqual({
      direct: [],
      all: [],
    });
    expect(shapesOf().at(`${MODEL_SCHEMA}#/$defs/no_such_definition`)).toEqual({
      direct: [],
      all: [],
    });
    // `true` and `false` are schemas of draft 2020-12 and describe no place a form stands at.
    expect(registry().byId(MODEL_SCHEMA)).toBeDefined();
  });
});
