import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import {
  identityKey,
  loadLibrary,
  primitiveCatalog,
  primitiveVersions,
  splitIdentity,
  type Library,
} from '../../src/library/index.js';
import { parse } from '../../src/json/index.js';
import { isJsonObject, type JsonValue } from '../../src/json/tree.js';
import { repositorySchemas } from '../schema/repository.js';
import { repositoryRoot } from '../json/repository.js';
import { nodeSource, REFERENCE_BASE } from './source.js';

// The catalog a chooser offers — feature 2.21.
//
// `by_id` is the loader's own index of the gathered set, and the editor needs three things of it
// that no component may invent: which identities there are, which base each came from, and how
// the pair `name@version` splits back into the two members of `primitive_reference`. The loader
// already says the third in a sentence beside `identityKey` — "`@` cannot occur in either half" —
// and this suite is what turns that sentence into a fact, over the whole reference base and over
// every `primitive` member of the corpus.

const schemas = repositorySchemas();
const source = nodeSource();

function reference(): Library {
  return loadLibrary([REFERENCE_BASE], { schemas, source });
}

/** Every `primitive` member of every corpus document, as the file writes the two halves. */
function corpusReferences(): { file: string; name: string; version: string }[] {
  const directory = join(repositoryRoot, 'data', 'models');
  const found: { file: string; name: string; version: string }[] = [];
  const walk = (value: JsonValue | undefined, file: string): void => {
    if (Array.isArray(value)) {
      for (const item of value) walk(item as JsonValue, file);
      return;
    }
    if (value === undefined || !isJsonObject(value)) return;
    const name = value.members.find((one) => one.name === 'name')?.value;
    const version = value.members.find((one) => one.name === 'version')?.value;
    if (
      value.members.length === 2 &&
      typeof name === 'string' &&
      typeof version === 'string' &&
      /^\d+\.\d+\.\d+$/.test(version)
    ) {
      found.push({ file, name, version });
    }
    for (const member of value.members) walk(member.value, file);
  };
  for (const file of readdirSync(directory).filter((one) => one.endsWith('.json')).sort()) {
    walk(parse(readFileSync(join(directory, file), 'utf8')), file);
  }
  return found;
}

describe('the catalog of the reference base', () => {
  const catalog = primitiveCatalog(reference());

  it('is every identity `by_id` carries, with the base each came from', () => {
    const library = reference();
    expect(catalog.length).toBe(library.byId.size);
    // Appendix A's own figure for the base the corpus pins, read rather than written down.
    expect(catalog.length).toBe(36);
    for (const identity of catalog) {
      expect(library.byId.get(identity.id), identity.id).toBeDefined();
      expect(identity.base).toBe(REFERENCE_BASE);
      expect(identity.id).toBe(identityKey(identity.name, identity.version));
    }
  });

  it('is in the loader’s own order: by name, then by version', () => {
    const names = catalog.map((one) => one.id);
    expect([...names].sort()).toEqual(names);
  });

  it('marks the one primitive of the base that pins a template (§4.6’s `▣`)', () => {
    const pinned = catalog.filter((one) => one.template);
    expect(pinned.length).toBe(1);
    expect(reference().templates.has(pinned[0]?.id ?? '')).toBe(true);
  });

  it('agrees with `primitiveVersions`, which is the version half’s own list', () => {
    const versions = primitiveVersions(reference());
    for (const [name, held] of versions) {
      expect(
        catalog.filter((one) => one.name === name).map((one) => one.version),
        name,
      ).toEqual(held);
    }
    expect(new Set(catalog.map((one) => one.name)).size).toBe(versions.size);
  });

  it('carries a lab’s own base beside the reference one, each identity saying which', () => {
    // Q6: "a lab's base ships with its model". Two bases, and the catalog says which is which —
    // which is what §4.6's "one section per base" needs and what a name alone cannot answer.
    const lab = join(repositoryRoot, 'tests', 'rejections', 'primitive-library', 'unknown-axis');
    const both = primitiveCatalog(loadLibrary([REFERENCE_BASE, lab], { schemas, source }));
    expect(new Set(both.map((one) => one.base))).toEqual(new Set([REFERENCE_BASE, lab]));
    expect(both.length).toBeGreaterThan(catalog.length);
  });
});

describe('`name@version` splits back into the two members it was made of', () => {
  it('round-trips every identity of the reference base', () => {
    for (const identity of primitiveCatalog(reference())) {
      expect(splitIdentity(identity.id), identity.id).toEqual({
        name: identity.name,
        version: identity.version,
      });
    }
  });

  it('round-trips every pinned primitive the corpus writes', () => {
    const written = corpusReferences();
    // The corpus does pin primitives, and in quantity: a suite that silently found none would
    // pass on nothing at all.
    expect(written.length).toBeGreaterThan(100);
    for (const one of written) {
      expect(splitIdentity(identityKey(one.name, one.version)), `${one.file} ${one.name}`).toEqual({
        name: one.name,
        version: one.version,
      });
    }
  });

  it('answers nothing for a text that is no key, which is the caller’s to decide about', () => {
    expect(splitIdentity('norm.rms')).toBeNull();
    expect(splitIdentity('')).toBeNull();
    expect(splitIdentity('@1.0.0')).toBeNull();
    expect(splitIdentity('norm.rms@')).toBeNull();
    expect(splitIdentity('a@b@c')).toBeNull();
  });

  it('holds the loader’s own sentence: `@` occurs in neither half, over both schemas', () => {
    // The sentence is a consequence of two patterns, so the patterns are what is read: a
    // `qualified_name` and a `semantic_version` admit no `@`. A grammar that admitted one would
    // make one field over two members ambiguous, and this is what would say so.
    const model = JSON.parse(
      readFileSync(join(repositoryRoot, 'schemas', 'tensorspine.schema.json'), 'utf8'),
    ) as { $defs: Record<string, { pattern?: string }> };
    for (const name of ['qualified_name', 'semantic_version']) {
      const pattern = model.$defs[name]?.pattern;
      expect(pattern, name).toBeDefined();
      expect(new RegExp(pattern ?? '').test('a@b'), name).toBe(false);
    }
  });
});
