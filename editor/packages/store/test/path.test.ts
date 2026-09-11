import { whereOf } from '@tensorspine/lang';
import { describe, expect, it } from 'vitest';

import {
  arrayAt,
  child,
  exists,
  isUnder,
  keyOf,
  lastOf,
  nodeAt,
  objectAt,
  parentOf,
  pathOfKey,
  pointerOf,
  stepInto,
} from '../src/path.js';
import { corpusTree } from './source.js';

// A place in the document is rendered three ways — the path, the JSON pointer a problem carries,
// and the key the layout sidecar is written with (plan D6) — and the three have to agree, since
// the sidecar's pruning resolves a key against the tree and a problem's pointer is what the
// canvas navigates by.

describe('a place of the document', () => {
  it('is a JSON pointer, escaped as RFC 6901 escapes it', () => {
    expect(pointerOf([])).toBe('');
    expect(pointerOf(['instances', 'embed'])).toBe('/instances/embed');
    expect(pointerOf(['bindings', 'values', 'decoder.entry'])).toBe('/bindings/values/decoder.entry');
    expect(pointerOf(['interfaces', 'inputs', 'tokens', 'to', 0])).toBe('/interfaces/inputs/tokens/to/0');
    expect(pointerOf(['a/b', 'c~d'])).toBe('/a~1b/c~0d');
  });

  it('is the sidecar key plan D6 writes, and the core’s own rendering of a place', () => {
    for (const path of [
      ['instances', 'embed'],
      ['compositions', 'decoder', 'instances', 'attn'],
      ['interfaces', 'inputs', 'tokens'],
      ['interfaces', 'inputs', 'tokens', 'to', 0],
    ]) {
      expect(keyOf(path)).toBe(whereOf(path));
    }
    expect(keyOf(['instances', 'embed'])).toBe('instances/embed');
    expect(() => keyOf([])).toThrow(TypeError);
  });

  it('reads a key back, an index as an index', () => {
    expect(pathOfKey('compositions/decoder/instances/attn')).toEqual([
      'compositions',
      'decoder',
      'instances',
      'attn',
    ]);
    expect(pathOfKey('interfaces/inputs/tokens/to/0')).toEqual([
      'interfaces',
      'inputs',
      'tokens',
      'to',
      0,
    ]);
    // A name that reads as a number is not one: the grammar's identifiers start with a letter or
    // an underscore, so no member of a document can be mistaken for an index.
    expect(pathOfKey('a/01')).toEqual(['a', '01']);
  });

  it('knows what is under what', () => {
    expect(isUnder(['compositions', 'decoder', 'instances'], ['compositions', 'decoder'])).toBe(true);
    expect(isUnder(['compositions', 'decoder'], ['compositions', 'decoder'])).toBe(true);
    expect(isUnder(['compositions', 'decoder'], ['compositions', 'other'])).toBe(false);
    expect(isUnder(['compositions'], ['compositions', 'decoder'])).toBe(false);
  });

  it('is built and taken apart step by step', () => {
    expect(child(['instances'], 'embed')).toEqual(['instances', 'embed']);
    expect(parentOf(['instances', 'embed'])).toEqual(['instances']);
    expect(lastOf(['interfaces', 'inputs', 'tokens', 'to', 0])).toBe(0);
    expect(() => parentOf([])).toThrow(TypeError);
    expect(() => lastOf([])).toThrow(TypeError);
  });
});

describe('reading the tree at a place', () => {
  const tree = corpusTree('llama3-8b');

  it('finds a member, an item and nothing at all', () => {
    expect(objectAt(tree, ['quantities', 'd'])).toBeDefined();
    expect(arrayAt(tree, ['instances', 'embed', 'families'])).toHaveLength(2);
    expect(nodeAt(tree, ['quantities', 'd', 'source', 'value'])).toEqual({
      kind: 'number',
      value: 4096,
      real: false,
      lexeme: '4096',
    });
    expect(nodeAt(tree, ['quantities', 'nowhere'])).toBeUndefined();
    expect(nodeAt(tree, ['quantities', 'd', 'source', 'value', 'deeper'])).toBeUndefined();
    expect(exists(tree, ['compositions', 'decoder', 'instances', 'attn'])).toBe(true);
    expect(exists(tree, ['compositions', 'decoder', 'instances', 'gone'])).toBe(false);
  });

  it('steps into an object by name and into an array by index, and nothing else', () => {
    const families = nodeAt(tree, ['instances', 'embed', 'families']);
    expect(families).toBeDefined();
    expect(stepInto(families!, 0)).toBe('input');
    expect(stepInto(families!, 'input')).toBeUndefined();
    expect(stepInto('a string', 0)).toBeUndefined();
  });

  it('takes an object for an object and an array for an array, never the other', () => {
    expect(objectAt(tree, ['instances', 'embed', 'families'])).toBeUndefined();
    expect(arrayAt(tree, ['instances', 'embed'])).toBeUndefined();
  });
});
