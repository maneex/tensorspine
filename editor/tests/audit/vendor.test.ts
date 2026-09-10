import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, relative, resolve, sep } from 'node:path';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { VendorError, defaultVendorRoot, vendor, type VendorManifest } from '../../scripts/vendor.js';
import { editorRoot, readEditorFile } from './tree.js';

// Feature 0.2 of the implementation plan: what the static application ships beside its own
// code — the repository's schemas, the corpus and the reference base as the Examples workspace
// (plan D11), and the artifacts the tools generate and the editor never regenerates (plan F5).
//
// The audit runs the script itself, into a temporary directory, and holds what it wrote to five
// claims: the manifest names every file and every digest is that file's; the copies are the
// repository's trees, whole and byte for byte, under the `$id`s the plan names; the
// argument-schema map has one entry per non-template primitive version of the reference base;
// the Examples workspace resolves its bases where it stands; and the script refuses, clearly and
// without touching anything, when the tools are not there.
//
// It needs python3 and jsonschema, as `pnpm oracle` does: the two generated artifacts are the
// tools' own output, and no fixture stands in for them.

const repositoryRoot = resolve(editorRoot, '..');

/** The four schemas the plan's §1 names, and the `$id` each one declares. A version segment
 *  that moves here is a deliberate signal: the registry of feature 1.1 loads by `$id`. */
const NAMED_SCHEMAS: Record<string, string> = {
  'schemas/tensorspine.schema.json': 'https://tensorspine.dev/schema/2.0/model.json',
  'schemas/tensorspine-primitive-library-unit.schema.json':
    'https://tensorspine.dev/schema/2.0/primitive-library-unit.json',
  'schemas/tensorspine-documentation.schema.json':
    'https://tensorspine.dev/schema/2.0/documentation.json',
  'schemas/tensorspine-derived.schema.json': 'https://tensorspine.dev/schema/2.1/derived.json',
};

interface Unit {
  name: string;
  kind: string;
  definition: { version?: string; template?: unknown };
}

interface Model {
  primitive_libraries?: { base?: string }[];
}

/** Every file below `root`, as `/`-separated paths, sorted. */
function walk(root: string, at = ''): string[] {
  const found: string[] = [];
  for (const entry of readdirSync(at === '' ? root : join(root, at), { withFileTypes: true })) {
    const path = at === '' ? entry.name : `${at}/${entry.name}`;
    if (entry.isDirectory()) found.push(...walk(root, path));
    else found.push(path);
  }
  return found.sort();
}

function sha256(bytes: Buffer): string {
  return createHash('sha256').update(bytes).digest('hex');
}

function readJson<T>(path: string): T {
  return JSON.parse(readFileSync(path, 'utf8')) as T;
}

/** Every primitive unit of the vendored reference base, with its identity and whether it is a
 *  template — read from the units themselves, never from a path or a count written here. */
function primitiveUnits(): { id: string; template: boolean }[] {
  const root = join(out, manifest.examples.primitive_library, 'primitives');
  return walk(root)
    .filter((name) => name.endsWith('.json'))
    .map((name) => {
      const unit = readJson<Unit>(join(root, name));
      return {
        id: `${unit.name}@${unit.definition.version ?? ''}`,
        template: unit.definition.template !== undefined,
      };
    });
}

/** A checkout-shaped directory whose tools are whatever `tool` says — or absent. */
function fakeRepository(tool: string | null): string {
  const root = mkdtempSync(join(tmpdir(), 'tensorspine-fake-'));
  for (const directory of ['schemas', 'data/models', 'data/primitive-library']) {
    mkdirSync(join(root, directory), { recursive: true });
    writeFileSync(join(root, directory, 'placeholder.json'), '{}\n');
  }
  if (tool !== null) {
    mkdirSync(join(root, 'tools'), { recursive: true });
    writeFileSync(join(root, 'tools', 'tensorspine'), tool);
  }
  return root;
}

let out: string;
let manifest: VendorManifest;
const temporary: string[] = [];

beforeAll(() => {
  out = mkdtempSync(join(tmpdir(), 'tensorspine-vendor-'));
  temporary.push(out);
  // What an earlier run left behind. The script rebuilds its directory; it never adds to one.
  writeFileSync(join(out, 'stale.json'), '{"stale": true}\n');
  mkdirSync(join(out, 'schemas'), { recursive: true });
  writeFileSync(join(out, 'schemas', 'gone.schema.json'), '{}\n');
  manifest = vendor({ out, quiet: true }).manifest;
}, 120_000);

afterAll(() => {
  for (const path of temporary) rmSync(path, { recursive: true, force: true });
});

describe('the vendor manifest', () => {
  it('names every file written, and nothing else', () => {
    const written = walk(out).filter((path) => path !== 'vendor.json');
    expect(manifest.files.map((file) => file.path)).toEqual(written);
    expect(new Set(written).size).toBe(written.length);
    expect(written.length).toBeGreaterThan(0);
  });

  it('carries the digest and the length of each file', () => {
    const wrong: string[] = [];
    for (const file of manifest.files) {
      const bytes = readFileSync(join(out, file.path));
      if (bytes.length !== file.bytes || sha256(bytes) !== file.sha256) wrong.push(file.path);
    }
    expect(wrong).toEqual([]);
  });

  it('says which commit the bytes come from, and whether the tree still matched it', () => {
    expect(manifest.repository_commit).toMatch(/^[0-9a-f]{40}$/);
    expect(typeof manifest.repository_dirty).toBe('boolean');
    expect(manifest.tools.command).toBe('tools/tensorspine');
    expect(manifest.tools.jsonschema).toMatch(/^\d+\.\d+/);
  });

  it('accounts for every set of files it declares', () => {
    const counted = manifest.sets.reduce((total, set) => total + set.files, 0);
    expect(counted).toBe(manifest.files.length);
    for (const set of manifest.sets) expect(existsSync(join(out, set.path))).toBe(true);
  });

  it('keeps no file of an earlier run', () => {
    expect(existsSync(join(out, 'stale.json'))).toBe(false);
    expect(existsSync(join(out, 'schemas', 'gone.schema.json'))).toBe(false);
    const paths = manifest.files.map((file) => file.path);
    expect(paths).not.toContain('stale.json');
    expect(paths).not.toContain('schemas/gone.schema.json');
  });
});

describe('the copied trees', () => {
  it('are the repository’s files, byte for byte', () => {
    let compared = 0;
    for (const file of manifest.files) {
      if (!file.path.startsWith('schemas/') && !file.path.startsWith('data/')) continue;
      // The vendor root mirrors the repository's paths, so the source is named by the same path.
      expect(sha256(readFileSync(join(repositoryRoot, file.path)))).toBe(file.sha256);
      compared += 1;
    }
    expect(compared).toBeGreaterThan(0);
  });

  it('hold the whole corpus, its versioned subdirectory included', () => {
    const repository = walk(join(repositoryRoot, 'data', 'models'));
    expect(walk(join(out, manifest.examples.models))).toEqual(repository);
    // The template document lives under `decoder-causal-yarn/1.0.0.json`: the copy descends.
    expect(repository.some((name) => name.includes('/'))).toBe(true);
  });

  it('hold the whole reference base', () => {
    expect(walk(join(out, manifest.examples.primitive_library))).toEqual(
      walk(join(repositoryRoot, 'data', 'primitive-library')),
    );
  });

  it('hold the whole schema directory', () => {
    expect(walk(join(out, 'schemas'))).toEqual(walk(join(repositoryRoot, 'schemas')));
  });
});

describe('the vendored schemas', () => {
  it('carry the `$id`s the plan names', () => {
    const declared = new Map(manifest.schemas.map((schema) => [schema.path, schema.id]));
    for (const [path, id] of Object.entries(NAMED_SCHEMAS)) {
      expect(declared.get(path)).toBe(id);
      expect(readJson<{ $id?: string }>(join(out, path)).$id).toBe(id);
    }
  });

  it('index the whole directory by `$id`, the plan’s four and whatever else it holds', () => {
    // `schemas/` holds a fifth schema the plan does not name — the unit-fixture format — and
    // §1 vendors the directory, so it travels too. Every schema there must still be indexable.
    for (const schema of manifest.schemas) {
      expect(schema.id).toMatch(/^https:\/\/tensorspine\.dev\/schema\//);
    }
    const ids = manifest.schemas.map((schema) => schema.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids).toEqual(expect.arrayContaining(Object.values(NAMED_SCHEMAS)));
  });
});

describe('the generated argument schemas', () => {
  it('has one entry per non-template primitive version of the reference base', () => {
    const expected = primitiveUnits()
      .filter((unit) => !unit.template)
      .map((unit) => unit.id)
      .sort();
    expect(Object.keys(manifest.primitive_schemas).sort()).toEqual(expected);
    expect(expected.length).toBeGreaterThan(0);
  });

  it('skips the template versions, as `--document primitive-schema` does (F5)', () => {
    // The tools generate no argument schema for a template primitive: a template's external
    // quantities are its argument schema, and F5 is the change set that will close the gap.
    const templates = primitiveUnits().filter((unit) => unit.template);
    expect(templates.length).toBeGreaterThan(0);
    for (const unit of templates) {
      expect(Object.keys(manifest.primitive_schemas)).not.toContain(unit.id);
    }
  });

  it('names each schema by the identity the file itself declares', () => {
    for (const [id, path] of Object.entries(manifest.primitive_schemas)) {
      const schema = readJson<{ $id?: string }>(join(out, path));
      const [name, version] = id.split('@');
      expect(schema.$id).toBe(`https://tensorspine.dev/primitive/${name ?? ''}/${version ?? ''}.json`);
    }
  });
});

describe('the Examples workspace', () => {
  it('is laid out like `data/`, so every document resolves its base where it stands', () => {
    const models = join(out, manifest.examples.models);
    const documents = walk(models).filter((name) => name.endsWith('.json'));
    expect(documents.length).toBeGreaterThan(0);
    const unresolved: string[] = [];
    for (const name of documents) {
      const document = readJson<Model>(join(models, name));
      for (const declared of document.primitive_libraries ?? []) {
        const base = resolve(join(models, name), '..', declared.base ?? '');
        if (!existsSync(join(base, 'primitive-library.json'))) {
          unresolved.push(`${name} -> ${declared.base ?? ''}`);
        }
      }
    }
    expect(unresolved).toEqual([]);
  });

  it('lands where the application serves it, and the build puts it there', () => {
    expect(relative(editorRoot, defaultVendorRoot).split(sep).join('/')).toBe(
      'apps/web/public/vendor',
    );
    const scripts = (JSON.parse(readEditorFile('package.json')) as { scripts: Record<string, string> })
      .scripts;
    expect(scripts['vendor']).toContain('scripts/vendor.ts');
    expect(scripts['build']).toContain('vendor');
    expect(readEditorFile('.gitignore')).toContain('apps/web/public/vendor/');
  });
});

describe('the vendored library reference', () => {
  it('covers every unit of the reference base', () => {
    const reference = readFileSync(join(out, manifest.primitive_library_reference), 'utf8');
    expect(reference.length).toBeGreaterThan(0);
    const root = join(out, manifest.examples.primitive_library, 'primitives');
    const missing = walk(root)
      .filter((name) => name.endsWith('.json'))
      .map((name) => readJson<Unit>(join(root, name)).name)
      .filter((name) => !reference.includes(name));
    expect(missing).toEqual([]);
  });
});

describe('a refusal', () => {
  it('says the tools are missing, and touches nothing', () => {
    const root = fakeRepository(null);
    const target = mkdtempSync(join(tmpdir(), 'tensorspine-untouched-'));
    temporary.push(root, target);
    writeFileSync(join(target, 'previous.txt'), 'an earlier vendor\n');

    expect(() => vendor({ repositoryRoot: root, out: target, quiet: true })).toThrowError(VendorError);
    try {
      vendor({ repositoryRoot: root, out: target, quiet: true });
    } catch (error) {
      expect((error as Error).message).toContain('tools/tensorspine');
      expect((error as Error).message).toContain('never falls back on stale files');
    }
    expect(readFileSync(join(target, 'previous.txt'), 'utf8')).toBe('an earlier vendor\n');
  });

  it('carries the tools’ own words, and leaves no half vendor behind', () => {
    const root = fakeRepository('import sys\nsys.stderr.write("primitive library refused: no base\\n")\nsys.exit(1)\n');
    const target = join(mkdtempSync(join(tmpdir(), 'tensorspine-partial-')), 'vendor');
    temporary.push(root, target);

    let message = '';
    try {
      vendor({ repositoryRoot: root, out: target, quiet: true });
    } catch (error) {
      expect(error).toBeInstanceOf(VendorError);
      message = (error as Error).message;
    }
    expect(message).toContain('--document primitive-schema');
    expect(message).toContain('primitive library refused: no base');
    expect(existsSync(target)).toBe(false);
  });
});
