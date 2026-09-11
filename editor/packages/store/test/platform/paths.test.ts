import { describe, expect, it } from 'vitest';

import {
  fromPosix,
  isUnder,
  join,
  nameOf,
  normalise,
  parentOf,
  POSIX_ROOT,
  resolveFrom,
  segmentsOf,
  toPosix,
  WORKSPACE_ROOT,
} from '../../src/platform/paths.js';
import { PlatformError } from '../../src/platform/errors.js';

// The path arithmetic of a workspace (feature 2.4, plan §5.2 `resolve`). Two things are held to
// account here: that a relative reference resolves against the file it is *written in*, which is
// what `primitive_libraries[].base` means, and that the seam with the core's own spelling — where
// the root is `.` and not `''` — is crossed by a conversion and never by a comparison on text.

describe('a workspace path', () => {
  it('drops empty segments and `.`, and applies `..`', () => {
    expect(normalise('models/llama3-8b.json')).toBe('models/llama3-8b.json');
    expect(normalise('/models//./llama3-8b.json')).toBe('models/llama3-8b.json');
    expect(normalise('models/')).toBe('models');
    expect(normalise('a/b/../c')).toBe('a/c');
    expect(normalise('')).toBe(WORKSPACE_ROOT);
    expect(normalise('.')).toBe(WORKSPACE_ROOT);
  });

  it('refuses a path that leaves the workspace rather than clamping it', () => {
    // A workspace path is user input — a base a document declares, a file a drop carried — and a
    // silently clamped `../..` is a path the caller never wrote.
    expect(() => normalise('../outside')).toThrow(PlatformError);
    expect(() => normalise('a/../../outside')).toThrow(/leaves the workspace/);
  });

  it('answers its segments, its parent and its name', () => {
    expect(segmentsOf('a/b/c.json')).toEqual(['a', 'b', 'c.json']);
    expect(segmentsOf('')).toEqual([]);
    expect(parentOf('models/llama3-8b.json')).toBe('models');
    expect(parentOf('llama3-8b.json')).toBe(WORKSPACE_ROOT);
    expect(nameOf('models/llama3-8b.json')).toBe('llama3-8b.json');
    expect(nameOf('')).toBe('');
    expect(join('models', 'llama3-8b.json')).toBe('models/llama3-8b.json');
    expect(join(WORKSPACE_ROOT, 'llama3-8b.json')).toBe('llama3-8b.json');
  });

  it('says what lies under a directory, the root holding everything', () => {
    expect(isUnder('models/llama3-8b.json', 'models')).toBe(true);
    expect(isUnder('models', 'models')).toBe(true);
    expect(isUnder('models-2/x.json', 'models')).toBe(false);
    expect(isUnder('anything/at/all', WORKSPACE_ROOT)).toBe(true);
  });
});

describe('a relative reference', () => {
  it('resolves against the directory of the file it is written in', () => {
    // The corpus's own case: `"base": "../primitive-library/"` in `data/models/llama3-8b.json`.
    expect(resolveFrom('models/llama3-8b.json', '../primitive-library/')).toBe('primitive-library');
    // And a base manifest's `"templates": "../models/"`, written in the manifest itself.
    expect(resolveFrom('primitive-library/primitive-library.json', '../models/')).toBe('models');
    // A sidecar beside its document (D6).
    expect(resolveFrom('models/llama3-8b.json', 'llama3-8b.layout.json')).toBe(
      'models/llama3-8b.layout.json',
    );
  });

  it('reads a reference that names the folder it sits in as that folder', () => {
    expect(resolveFrom('models/llama3-8b.json', './')).toBe('models');
    expect(resolveFrom('llama3-8b.json', '.')).toBe(WORKSPACE_ROOT);
  });

  it('refuses one that climbs out of the workspace', () => {
    // One refusal class for the whole platform, so a caller writes one `catch` and reads the
    // reason: a path that leaves the workspace is `bad-path`, as a mistyped one is.
    const refusal = (): unknown => resolveFrom('llama3-8b.json', '../elsewhere');
    expect(refusal).toThrow(PlatformError);
    try {
      refusal();
    } catch (error) {
      expect((error as PlatformError).reason).toBe('bad-path');
    }
  });
});

describe('the seam with the core', () => {
  it('converts the root between the two spellings and leaves every other path alone', () => {
    // `os.path.normpath` writes a relative path with nothing left in it as `.`; a workspace writes
    // it `''`. A document at the root of its workspace declaring the folder it sits in reaches
    // exactly that case, which is the review repair `0232559`'s.
    expect(toPosix(WORKSPACE_ROOT)).toBe(POSIX_ROOT);
    expect(fromPosix(POSIX_ROOT)).toBe(WORKSPACE_ROOT);
    expect(toPosix('primitive-library')).toBe('primitive-library');
    expect(fromPosix('primitive-library')).toBe('primitive-library');
    expect(fromPosix('data/./primitive-library/')).toBe('data/primitive-library');
  });

  it('refuses an absolute path, which a workspace has no way to name', () => {
    expect(() => fromPosix('/etc/passwd')).toThrow(PlatformError);
  });
});
