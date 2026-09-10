import { describe, expect, it } from 'vitest';

import { filesUnder, readRepositoryFile } from '../json/repository.js';
import { repositorySchemas } from './repository.js';

// `vocabulary()` is what the plan's §1 rests on: "dtypes, quantity kinds, argument kinds, axis
// natures and spaces, operator names, comparison operators, condition forms, interface kinds,
// location forms … all read from the schemas at startup. A select's options are the schema's
// `enum`; a tagged union's tags are the `required` keys of its `oneOf` members."
//
// So the suite counts. It walks the schema files itself — a second, independent reading — and
// requires the vocabulary to hold exactly the enumerations and unions that walk finds, in the
// same places, with the same values. A schema that gains an enum and a vocabulary that does not
// is the failure this catches; the committed snapshot at the end is what makes such a change a
// reviewed diff rather than a surprise.

const registry = repositorySchemas();
const vocabulary = registry.vocabulary();

const MODEL = 'https://tensorspine.dev/schema/2.0/model.json';
const UNIT = 'https://tensorspine.dev/schema/2.0/primitive-library-unit.json';

/** The subschema-valued keywords, as the walk of `vocabulary.ts` reads them. */
const SINGLE = [
  'additionalProperties',
  'contains',
  'else',
  'if',
  'items',
  'not',
  'propertyNames',
  'then',
  'unevaluatedItems',
  'unevaluatedProperties',
];
const MAPS = ['$defs', 'definitions', 'dependentSchemas', 'patternProperties', 'properties'];
const LISTS = ['allOf', 'anyOf', 'oneOf', 'prefixItems'];

/** Every `enum` and every `oneOf` of the repository's schemas, found by reading the files here. */
function readSchemasDirectly(): { enums: [string, unknown[]][]; unions: [string, number][] } {
  const enums: [string, unknown[]][] = [];
  const unions: [string, number][] = [];
  const visit = (node: unknown, id: string, place: string): void => {
    if (node === null || typeof node !== 'object' || Array.isArray(node)) return;
    const object = node as Record<string, unknown>;
    if (Array.isArray(object['enum'])) enums.push([`${id}${place}`, object['enum']]);
    if (Array.isArray(object['oneOf'])) unions.push([`${id}${place}`, object['oneOf'].length]);
    for (const keyword of SINGLE) {
      if (object[keyword] !== undefined) visit(object[keyword], id, `${place}/${keyword}`);
    }
    for (const keyword of MAPS) {
      const children = object[keyword];
      if (children === null || typeof children !== 'object' || Array.isArray(children)) continue;
      for (const [name, child] of Object.entries(children)) {
        visit(child, id, `${place}/${keyword}/${name.replace(/~/g, '~0').replace(/\//g, '~1')}`);
      }
    }
    for (const keyword of LISTS) {
      const children = object[keyword];
      if (!Array.isArray(children)) continue;
      children.forEach((child, index) => {
        visit(child, id, `${place}/${keyword}/${String(index)}`);
      });
    }
  };
  for (const path of filesUnder('schemas').filter((name) => name.endsWith('.json'))) {
    const document = JSON.parse(readRepositoryFile(path)) as Record<string, unknown>;
    visit(document, String(document['$id']), '#');
  }
  return { enums, unions };
}

const direct = readSchemasDirectly();

describe('vocabulary()', () => {
  it('holds every enumeration the schema files carry, and only those', () => {
    expect(vocabulary.enums).toHaveLength(direct.enums.length);
    expect(direct.enums.length).toBe(38);
    for (const [pointer, values] of direct.enums) {
      const found = vocabulary.enumAt(pointer);
      expect(found, pointer).toBeDefined();
      expect(found?.values, pointer).toEqual(values);
    }
  });

  it('holds every tagged union the schema files carry, and only those', () => {
    expect(vocabulary.unions).toHaveLength(direct.unions.length);
    expect(direct.unions.length).toBe(31);
    for (const [pointer, count] of direct.unions) {
      const found = vocabulary.unionAt(pointer);
      expect(found, pointer).toBeDefined();
      expect(found?.alternatives, pointer).toHaveLength(count);
    }
  });

  it('lists the dtype enum', () => {
    const dtype = vocabulary.enumAt(`${MODEL}#/$defs/dtype`);
    expect(dtype?.values).toEqual([
      'bool',
      'u4',
      'i4',
      'u8',
      'i8',
      'i16',
      'i32',
      'i64',
      'fp4',
      'f8e4m3',
      'f8e4m3fn',
      'f8e5m2',
      'bf16',
      'f16',
      'f32',
      'f64',
    ]);
    expect(dtype?.values).toHaveLength(16);
  });

  it('lists the operator enums, one per arity, and the comparison operators', () => {
    const unary = vocabulary.enumAt(`${MODEL}#/$defs/unary_operation_expression/properties/op`);
    const binary = vocabulary.enumAt(`${MODEL}#/$defs/binary_operation_expression/properties/op`);
    const nary = vocabulary.enumAt(`${MODEL}#/$defs/nary_operation_expression/properties/op`);
    expect(unary?.values).toEqual(['negate', 'absolute']);
    expect(binary?.values).toEqual(['subtract', 'divide', 'ceil_divide', 'floor_divide', 'modulo']);
    expect(nary?.values).toEqual(['add', 'multiply', 'min', 'max']);
    for (const schema of [MODEL, UNIT]) {
      const comparison = vocabulary.enumAt(
        `${schema}#/$defs/comparison_condition/properties/compare/properties/operator`,
      );
      expect(comparison?.values, schema).toEqual([
        'equal',
        'not_equal',
        'less',
        'less_or_equal',
        'greater',
        'greater_or_equal',
      ]);
    }
  });

  it('lists the condition tags, one per form, each discriminating', () => {
    // The two schemas do not carry the same forms: a primitive's condition may test whether an
    // optional argument is `present`, which a document's condition has no need of.
    const forms: Record<string, string[]> = {
      [MODEL]: ['boolean', 'not', 'all', 'any', 'compare'],
      [UNIT]: ['boolean', 'not', 'all', 'any', 'present', 'compare'],
    };
    for (const schema of [MODEL, UNIT]) {
      const condition = vocabulary.unionAt(`${schema}#/$defs/condition`);
      expect(condition, schema).toBeDefined();
      expect(condition?.alternatives.map((one) => one.tags.join('+')), schema).toEqual(
        forms[schema],
      );
      expect(condition?.discriminated, schema).toBe(true);
      // Every form is a `$ref`, and the vocabulary names what each one points at.
      for (const alternative of condition?.alternatives ?? []) {
        expect(alternative.ref, schema).toMatch(/^#\/\$defs\//);
        expect(alternative.target, schema).toBe(`${schema}${alternative.ref ?? ''}`);
      }
    }
  });

  it('lists the location forms', () => {
    const location = vocabulary.unionAt(`${MODEL}#/$defs/location`);
    expect(location?.alternatives).toHaveLength(4);
    expect(location?.alternatives.map((one) => one.tags.join('+'))).toEqual([
      'tensor',
      'stack',
      'concat',
      'slice',
    ]);
    expect(location?.discriminated).toBe(true);
    expect(location?.description).toContain('Where the identity');
  });

  it('says when a union is not told apart by its required keys', () => {
    // The three operation expressions all require `op` and `args`; only the operator names their
    // `op` admits separate them, which no required key states. The vocabulary says so rather
    // than inventing a tag (plan §1: an unknown construct renders generically).
    const expression = vocabulary.unionAt(`${MODEL}#/$defs/scalar_expression`);
    expect(expression?.discriminated).toBe(false);
    const tags = expression?.alternatives.map((one) => one.tags.join('+')) ?? [];
    expect(tags).toEqual(['literal', 'quantity', 'index', 'op+args', 'op+args', 'op+args', 'if+then+else']);

    // A `physical_name` item may be a bare string, which requires nothing at all.
    const item = vocabulary.unionAt(`${MODEL}#/$defs/physical_name/items`);
    expect(item?.alternatives.map((one) => one.tags)).toEqual([[], ['index'], ['coordinate']]);
    expect(item?.alternatives[0]?.type).toBe('string');
    expect(item?.discriminated).toBe(false);
  });

  it('reads a const alternative as its own tag', () => {
    const target = vocabulary.unionAt(`${UNIT}#/$defs/partition_target`);
    const any = target?.alternatives.find((one) => one.required.includes('any_axis'));
    expect(any?.tags).toEqual(['any_axis']);
    // The constant lives one level down, on the property the alternative requires.
    const self = vocabulary.unionAt(`${UNIT}#/$defs/indexing_source`)?.alternatives[0];
    expect(self?.required).toEqual(['self']);
  });

  it('offers every enum value once, for the no-hard-coding audit', () => {
    const values = vocabulary.values();
    expect(new Set(values).size).toBe(values.length);
    expect(values).toContain('bf16');
    expect(values).toContain('ceil_divide');
    expect(values.length).toBe(
      new Set(direct.enums.flatMap(([, list]) => list.map((one) => `${typeof one}:${String(one)}`)))
        .size,
    );
  });

  it('is the vocabulary the repository publishes', () => {
    // The committed snapshot: a schema change shows up here as a reviewed diff (plan §1 c).
    const summary = {
      enums: vocabulary.enums.map((one) => ({ pointer: one.pointer, values: one.values })),
      unions: vocabulary.unions.map((one) => ({
        pointer: one.pointer,
        discriminated: one.discriminated,
        alternatives: one.alternatives.map((alternative) => ({
          tags: alternative.tags,
          ref: alternative.ref,
        })),
      })),
    };
    expect(summary).toMatchSnapshot();
  });
});
