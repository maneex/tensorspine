import { describe, expect, it } from 'vitest';

import { decode, encode, encodeFailure, LangFailure } from '../../src/api/codec.js';
import { DerivedSchemaError } from '../../src/derive/products.js';
import { PyTypeError } from '../../src/expr/errors.js';
import { UNRESOLVED, type PyValue } from '../../src/expr/value.js';
import { JsonParseError } from '../../src/json/parse.js';

// What the worker boundary drops, and what the codec puts back. Every case here was measured
// against `structuredClone` first: the encoding exists because the clone refused the value, and
// the suite asserts the refusal as well as the repair, so that a platform which one day carried
// a symbol would show up as a passing clone rather than as a silent change.

/** What `postMessage` would do to a value, without a port. */
function crossed(value: unknown): unknown {
  return structuredClone(value);
}

/** Encode, cross the boundary, decode: what a call of the API costs its answer. */
function roundTrip(value: unknown): unknown {
  return decode(crossed(encode(value)));
}

describe('what the structured clone refuses', () => {
  it('refuses the sentinel, which is why the encoding exists', () => {
    expect(() => crossed(UNRESOLVED)).toThrow(/could not be cloned/);
    expect(() => crossed({ heads: UNRESOLVED })).toThrow(/could not be cloned/);
  });

  it('carries a bigint, so Python’s int/float distinction needs no encoding', () => {
    // Feature 1.2's decision travels as it stands: `32n` and `32` stay apart across the boundary.
    expect(crossed({ whole: 32n, real: 32 })).toEqual({ whole: 32n, real: 32 });
    expect(typeof (crossed({ whole: 32n }) as { whole: unknown }).whole).toBe('bigint');
  });

  it('strips an Error of its class, its name and everything it carried', () => {
    const back = crossed(new PyTypeError('unsupported operand type(s)')) as Error;
    expect(back).toBeInstanceOf(Error);
    expect(back).not.toBeInstanceOf(PyTypeError);
    expect(back.name).toBe('Error');
  });

  it('refuses a function, which is why a registry and a source are held by handle', () => {
    expect(() => crossed({ conforms: () => true })).toThrow(/could not be cloned/);
  });
});

describe('the encoding', () => {
  it('carries the sentinel there and back', () => {
    expect(roundTrip(UNRESOLVED)).toBe(UNRESOLVED);
    expect(roundTrip({ heads: UNRESOLVED, kv_heads: 8n })).toEqual({
      heads: UNRESOLVED,
      kv_heads: 8n,
    });
    expect(roundTrip([UNRESOLVED, 1n, 'a'])).toEqual([UNRESOLVED, 1n, 'a']);
  });

  it('carries it inside a Map and a Set, which is what the answers are made of', () => {
    const stats = new Map<string, PyValue>([['sites', 9n]]);
    expect(roundTrip(stats)).toEqual(stats);
    const sites = new Map<string, PyValue>([['attn', UNRESOLVED]]);
    expect(roundTrip(sites)).toEqual(sites);
    expect(roundTrip(new Set([UNRESOLVED, 'a']))).toEqual(new Set([UNRESOLVED, 'a']));
  });

  it('escapes an object whose own data spells the marker', () => {
    // `@` is outside the identifier grammar of §5.2, so no name the schemas admit can be it — but
    // a safetensors `__metadata__` map carries whatever the file writes, so the encoding escapes
    // rather than relying on the grammar.
    for (const original of [
      { '@': 'unresolved' },
      { '@': 'escaped', value: 1n },
      { '@': 'escaped', value: { '@': 'unresolved' } },
      { '@': 1n },
      { '@': 'unresolved', more: UNRESOLVED },
      { nested: { '@': 'unresolved' } },
    ]) {
      expect(roundTrip(original)).toEqual(original);
    }
  });

  it('keeps a member named __proto__ a member', () => {
    // The defence features 0.3, 1.1 and 1.6a each had to make: plain assignment would set the
    // prototype and the member would vanish from the reading.
    const original: Record<string, unknown> = {};
    Object.defineProperty(original, '__proto__', {
      value: UNRESOLVED,
      writable: true,
      enumerable: true,
      configurable: true,
    });
    const back = roundTrip(original) as Record<string, unknown>;
    expect(Object.keys(back)).toEqual(['__proto__']);
    expect(Object.getOwnPropertyDescriptor(back, '__proto__')?.value).toBe(UNRESOLVED);
  });

  it('keeps member order, which D12’s bytes depend on', () => {
    const original = { schema: 'tensorspine/2.0', model: 'llama3_8b', quantities: {} };
    expect(Object.keys(roundTrip(original) as object)).toEqual(Object.keys(original));
  });

  it('allocates nothing where nothing is encoded', () => {
    // The common case: no corpus document produces an `UNRESOLVED` anywhere in its facts or its
    // products, so the walk must not copy a derived document of three megabytes to carry it.
    const value = { a: [1n, { b: 'c' }], m: new Map([['k', 2n]]), s: new Set([3n]) };
    expect(encode(value)).toBe(value);
    expect(decode(value)).toBe(value);
  });

  it('leaves the bytes a header is read from alone', () => {
    const bytes = new Uint8Array([1, 2, 3]);
    expect(encode(bytes)).toBe(bytes);
    expect(roundTrip(bytes)).toEqual(bytes);
  });
});

describe('a refusal on the wire', () => {
  it('keeps the name the core raised under', () => {
    const failure = new LangFailure(
      encodeFailure(new PyTypeError("unsupported operand type(s) for *: 'NoneType' and 'int'")),
    );
    expect(failure.raised).toBe('PyTypeError');
    expect(failure.message).toBe("unsupported operand type(s) for *: 'NoneType' and 'int'");
    expect(failure).toBeInstanceOf(Error);
  });

  it('keeps what the refusal carried beside its message', () => {
    const refusal = new DerivedSchemaError(
      [{ code: 'schema', message: 'None is not of type "string"', path: '/d4/states/0/writer', segments: [], keyword: 'type' }],
      { d4: null },
    );
    const failure = new LangFailure(encodeFailure(refusal));
    expect(failure.raised).toBe('DerivedSchemaError');
    expect(failure.detail['problems']).toEqual(refusal.problems);
    expect(failure.detail['document']).toEqual(refusal.document);
  });

  it('keeps a parse refusal’s own reason beside the tools’ wording', () => {
    // Feature 0.3's two wordings: `model.py` raises the `(V12)` line and the loader wraps the bare
    // `reason`, so both travel — the message for parity, the reason for the loader's own form.
    const raised = new JsonParseError(
      "duplicate member name 'x'",
      { line: 1, column: 1, offset: 0 },
      'x',
    );
    const failure = new LangFailure(encodeFailure(raised));
    expect(failure.raised).toBe('JsonParseError');
    expect(failure.message).toBe("duplicate member name 'x' (V12)");
    expect(failure.detail['reason']).toBe("duplicate member name 'x'");
    expect(failure.detail['duplicateMember']).toBe('x');
  });

  it('reports what is not an Error as the text it prints as', () => {
    expect(new LangFailure(encodeFailure('a string')).message).toBe('a string');
  });
});
