import { describe, expect, it } from 'vitest';

import {
  identityKey,
  libraryUnits,
  loadLibrary,
  templateInterface,
  templateInterfaces,
  toPrimitiveExpression,
  type Library,
} from '../../src/library/index.js';
import type { PyValue } from '../../src/expr/index.js';
import { repositorySchemas } from '../schema/repository.js';
import { nodeSource, REFERENCE_BASE } from './source.js';

// `validate.template_interface`: the primitive a template primitive presents to a caller (§4.6).
// The parity suite compares the whole answer for the reference base's own template against the
// tools; these are the rules that answer is made of, on templates small enough to read.

const schemas = repositorySchemas();
const library: Library = loadLibrary([REFERENCE_BASE], { schemas, source: nodeSource() });

/** A template document with the two members the interface reads. */
function template(quantities: PyValue, interfaces: PyValue = { inputs: {}, outputs: {} }): PyValue {
  return { schema: 'tensorspine/2.0', quantities, interfaces };
}

const external = (type: PyValue, over: PyValue = {}): PyValue => ({
  type,
  source: { kind: 'external', ...(over as Record<string, PyValue>) },
});

describe('the reference base’s template primitive', () => {
  const key = identityKey('decoder.causal_yarn', '1.0.0');
  const definition = library.byId.get(key)?.definition as PyValue;
  const document = library.templates.get(key)?.document as PyValue;
  const answered = templateInterface(definition, document) as Record<string, PyValue>;

  it('exposes one argument per external quantity, in the template’s own order', () => {
    expect(Object.keys(answered['arguments'] as Record<string, PyValue>)).toEqual([
      'width',
      'layers',
      'heads',
      'kv_heads',
      'head_dim',
      'inner',
      'eps',
      'precision',
    ]);
  });

  it('carries each quantity’s type and domain, all required for want of a default', () => {
    const args = answered['arguments'] as Record<string, Record<string, PyValue>>;
    expect(args['eps']?.['type']).toEqual({ kind: 'real' });
    expect(args['eps']?.['required']).toBe(true);
    expect(args['eps']?.['structural']).toBe(true);
    expect(args['precision']?.['domain']).toEqual({ kind: 'set', values: ['bf16', 'f16', 'f32'] });
    expect(args['width']?.['domain']).toEqual({
      kind: 'interval',
      lower: { value: { literal: 1n }, inclusive: true },
    });
  });

  it('exposes the public interfaces as ports, the outputs inheriting their domain', () => {
    expect(answered['ports']).toEqual({
      inputs: { hidden: { role: 'activation.hidden', domain: { kind: 'token', from: { self: true } } } },
      outputs: {
        hidden_out: {
          role: 'activation.hidden',
          domain: { kind: 'inherit', from: { self: true } },
        },
      },
    });
  });

  it('is what the library answers for the identity, without a document being read again', () => {
    expect(templateInterfaces(library).get(key)).toEqual(answered);
    expect([...templateInterfaces(library).keys()]).toEqual([key]);
  });

  it('is one of the 130 units the base carries', () => {
    expect(libraryUnits(library)).toHaveLength(130);
    expect(
      libraryUnits(library).filter((one) => one.name === 'decoder.causal_yarn'),
    ).toHaveLength(1);
  });

  it('owns no parameter, constant, state or partition option of its own', () => {
    expect(answered['parameters']).toEqual({});
    expect(answered['constants']).toEqual({});
    expect(answered['state_ports']).toEqual({});
    expect(answered['partition_options']).toEqual([]);
    expect(answered['version']).toBe('1.0.0');
  });
});

describe('a template default', () => {
  it('makes its argument optional when it can be written over the arguments', () => {
    const answered = templateInterface(
      { version: '2.0.0' },
      template({
        width: external({ kind: 'cardinality' }),
        heads: external({ kind: 'cardinality' }, { default: { quantity: 'width' } }),
      }),
    ) as Record<string, Record<string, Record<string, PyValue>>>;
    expect(answered['arguments']?.['heads']).toEqual({
      type: { kind: 'cardinality' },
      required: false,
      structural: true,
      default: { argument: 'width' },
    });
    // Python assigns into the dictionary, so `required` keeps the place it was written in.
    expect(Object.keys(answered['arguments']?.['heads'] as Record<string, PyValue>)).toEqual([
      'type',
      'required',
      'structural',
      'default',
    ]);
  });

  it('leaves its argument required when the default reads something a caller cannot supply', () => {
    const answered = templateInterface(
      { version: '1.0.0' },
      template({
        width: external({ kind: 'cardinality' }),
        inner: external({ kind: 'cardinality' }, { default: { quantity: 'derived' } }),
        derived: {
          type: { kind: 'cardinality' },
          source: { kind: 'derived', derivation: { literal: 4n } },
        },
      }),
    ) as Record<string, Record<string, Record<string, PyValue>>>;
    expect(answered['arguments']?.['inner']?.['required']).toBe(true);
    expect(answered['arguments']?.['inner']).not.toHaveProperty('default');
    // A derived quantity is not an argument of the interface either.
    expect(Object.keys(answered['arguments'] as Record<string, PyValue>)).toEqual(['width', 'inner']);
  });
});

describe('a template expression read as a primitive expression', () => {
  const quantities = template({
    width: external({ kind: 'cardinality' }),
    four: { type: { kind: 'cardinality' }, source: { kind: 'literal', value: 4n } },
    derived: { type: { kind: 'cardinality' }, source: { kind: 'derived', derivation: { literal: 1n } } },
  });

  it('keeps a literal as it stands', () => {
    expect(toPrimitiveExpression({ literal: 32n }, quantities)).toEqual({ literal: 32n });
  });

  it('turns an external quantity into the argument the caller supplies', () => {
    expect(toPrimitiveExpression({ quantity: 'width' }, quantities)).toEqual({ argument: 'width' });
  });

  it('inlines a literal quantity’s value', () => {
    expect(toPrimitiveExpression({ quantity: 'four' }, quantities)).toEqual({ literal: 4n });
  });

  it('gives up on a derived quantity and on a quantity the template does not declare', () => {
    expect(toPrimitiveExpression({ quantity: 'derived' }, quantities)).toBeNull();
    expect(toPrimitiveExpression({ quantity: 'absent' }, quantities)).toBeNull();
  });

  it('rewrites an operator when every operand can be rewritten, and gives up otherwise', () => {
    expect(
      toPrimitiveExpression(
        { op: 'multiply', args: [{ quantity: 'width' }, { quantity: 'four' }] },
        quantities,
      ),
    ).toEqual({ op: 'multiply', args: [{ argument: 'width' }, { literal: 4n }] });
    expect(
      toPrimitiveExpression(
        { op: 'multiply', args: [{ quantity: 'width' }, { quantity: 'derived' }] },
        quantities,
      ),
    ).toBeNull();
    // An index is a model-side expression a call site has nothing to put in its place.
    expect(toPrimitiveExpression({ index: 'layer' }, quantities)).toBeNull();
  });
});
