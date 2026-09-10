import { describe, expect, it } from 'vitest';

import { basename, dirname, isAbsolute, join, normalise, relative } from '../../src/library/index.js';

// The loader computes a path in four places and each one decides what it refuses: a unit's
// identity is its path relative to its section root, a base's templates location is `templates`
// resolved against the base, a pinned template is `<location>/<name>/<version>.json`, and every
// refusal names the file it was computed for. So the arithmetic is `posixpath`'s, and these are
// its answers, read off CPython 3.11.

describe('join', () => {
  it('separates with exactly one slash and restarts on an absolute component', () => {
    expect(join('a', 'b')).toBe('a/b');
    expect(join('a/', 'b')).toBe('a/b');
    expect(join('a', '/b')).toBe('/b');
    expect(join('', 'b')).toBe('b');
    expect(join('a', '')).toBe('a/');
    expect(join('data/models', 'decoder-causal-yarn', '1.0.0.json')).toBe(
      'data/models/decoder-causal-yarn/1.0.0.json',
    );
  });
});

describe('normalise', () => {
  it('is `normpath`: `.` dropped, `..` resolved, separators collapsed', () => {
    expect(normalise('data/primitive-library/../models/')).toBe('data/models');
    expect(normalise('a/./b/../c')).toBe('a/c');
    expect(normalise('a//b')).toBe('a/b');
    expect(normalise('')).toBe('.');
    expect(normalise('../../x')).toBe('../../x');
  });

  it('keeps two leading separators and collapses three, as POSIX requires', () => {
    expect(normalise('//a/b')).toBe('//a/b');
    expect(normalise('///a')).toBe('/a');
  });

  it('cannot climb above the root', () => {
    expect(normalise('/../x')).toBe('/x');
  });
});

describe('dirname and basename', () => {
  it('answer what `posixpath` answers, separators and all', () => {
    expect(dirname('a/b')).toBe('a');
    expect(dirname('/a')).toBe('/');
    expect(dirname('a')).toBe('');
    expect(dirname('/')).toBe('/');
    expect(dirname('a/b/')).toBe('a/b');
    expect(dirname('//a')).toBe('//');
    expect(dirname('a//b')).toBe('a');
    expect(basename('a/b.json')).toBe('b.json');
    expect(basename('b.json')).toBe('b.json');
  });
});

describe('relative', () => {
  it('is the suffix when one path is under the other — the loader’s only case', () => {
    expect(relative('a/b/c.json', 'a/b')).toBe('c.json');
    expect(relative('base/primitives/attention/dense/1.0.0.json', 'base/primitives')).toBe(
      'attention/dense/1.0.0.json',
    );
  });

  it('answers `.` and climbs with `..` in the general case', () => {
    expect(relative('a/b', 'a/b')).toBe('.');
    expect(relative('a/c', 'a/b')).toBe('../c');
    expect(relative('/x/y', '/x/a/b')).toBe('../../y');
  });

  it('refuses to mix an absolute path with a relative one, having no working directory', () => {
    expect(() => relative('/a', 'a')).toThrow(TypeError);
  });
});

describe('isAbsolute', () => {
  it('is POSIX’s: a leading separator and nothing else', () => {
    expect(isAbsolute('/a')).toBe(true);
    expect(isAbsolute('a')).toBe(false);
    expect(isAbsolute('C:\\a')).toBe(false);
  });
});
