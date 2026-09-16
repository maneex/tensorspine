import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { createLang } from '@tensorspine/lang/api/engine';
import type { Lang, SchemasHandle } from '@tensorspine/lang/api';

import {
  gatherBases,
  gatherSchemas,
  schemaDifferences,
  WORKSPACE_SCHEMAS,
} from '../../src/documents/gather.js';
import { ABSENT, MemoryWorkspace, ReadOnlyWorkspace, textsOf } from '../../src/platform/index.js';
import { readTree } from '../../src/platform/tree.js';
import { corpusText, editorRoot, repositoryRoot } from '../source.js';

// What a document needs read before it can be judged — and, at the centre of it, **the templates
// hand-over feature 2.4 left to this one**.
//
// A base's `templates` location resolves against the base and may leave it (`"templates":
// "../models/"`), and the loader opens the template document a template primitive pins. So the
// files under a base are not all the loader needs, and the workspace has to know where the rest
// are *before* it hands anything over. The answer taken here is that the **core** says where: one
// manifest read, one load, and `templates` named in `packages/lang` where every other member of
// the language is named.

/** Every `.json` of a repository directory, by its repository path. */
function filesUnder(directory: string): Record<string, string> {
  const found: Record<string, string> = {};
  const walk = (at: string, prefix: string): void => {
    for (const entry of readdirSync(join(repositoryRoot, at), { withFileTypes: true })) {
      const path = `${at}/${entry.name}`;
      if (entry.isDirectory()) walk(path, `${prefix}${entry.name}/`);
      else if (entry.name.endsWith('.json')) found[path] = readFileSync(join(repositoryRoot, path), 'utf8');
    }
  };
  walk(directory, '');
  return found;
}

/** A workspace laid out exactly like the repository's `data/`, which is what §4.3 says one is. */
function dataWorkspace(): MemoryWorkspace {
  return MemoryWorkspace.of({
    ...filesUnder('data/models'),
    ...filesUnder('data/primitive-library'),
  });
}

const MODEL = 'data/models/llama3-8b.json';

let held: { lang: Lang; schemas: SchemasHandle } | null = null;

async function core(): Promise<{ lang: Lang; schemas: SchemasHandle }> {
  if (held !== null) return held;
  const lang = createLang();
  const files: Record<string, string> = {};
  for (const name of readdirSync(join(repositoryRoot, 'schemas'))) {
    if (name.endsWith('.json')) {
      files[`schemas/${name}`] = readFileSync(join(repositoryRoot, 'schemas', name), 'utf8');
    }
  }
  const loaded = await lang.loadSchemas(files, { origin: 'schemas' });
  held = { lang, schemas: loaded.handle };
  return held;
}

describe('the bases a document declares', () => {
  it('are the core’s answer, resolved against the document’s own directory', async () => {
    const { lang } = await core();
    const tree = await lang.parse(corpusText('llama3-8b'));
    const answered = await lang.documentBases(tree, MODEL);
    expect(answered.bases).toEqual(['data/primitive-library']);
    expect(answered.problems).toEqual([]);
  });

  it('carry the loader’s own refusal where a document cannot be resolved from at all', async () => {
    const { lang } = await core();
    const tree = await lang.parse('{"schema": "tensorspine/1.0"}');
    const answered = await lang.documentBases(tree, MODEL);
    expect(answered.bases).toEqual([]);
    expect(answered.problems).toHaveLength(1);
    expect(answered.problems[0]?.message).toContain('expected tensorspine/2.0');
  });
});

describe('the templates a base pins', () => {
  it('are read from the manifest by the core, before the base is handed over', async () => {
    const { lang, schemas } = await core();
    const base = { base: 'data/primitive-library', files: filesUnder('data/primitive-library') };
    expect(await lang.baseTemplates([base], schemas)).toEqual(['data/models']);
  });

  it('cost one manifest read, not a load: a base of its manifest alone answers the same', async () => {
    const { lang, schemas } = await core();
    const whole = filesUnder('data/primitive-library');
    const manifest = 'data/primitive-library/primitive-library.json';
    const alone = { base: 'data/primitive-library', files: { [manifest]: whole[manifest] ?? '' } };
    expect(await lang.baseTemplates([alone], schemas)).toEqual(['data/models']);
  });

  it('are nothing where the base declares none', async () => {
    const { lang, schemas } = await core();
    expect(await lang.baseTemplates([{ base: 'nowhere', files: {} }], schemas)).toEqual([null]);
  });
});

describe('what a document has read for it', () => {
  it('is the base, and the template documents its manifest points outside it at', async () => {
    const { lang, schemas } = await core();
    const place = dataWorkspace();
    const tree = await lang.parse((await place.read(MODEL)).text);
    const gathered = await gatherBases(lang, schemas, place, tree, MODEL);
    expect(gathered.bases).toHaveLength(1);
    const files = Object.keys(gathered.bases[0]?.files ?? {});
    // Every unit of the base…
    const units = Object.keys(filesUnder('data/primitive-library'));
    expect(files.filter((path) => path.startsWith('data/primitive-library/')).sort()).toEqual(
      units.sort(),
    );
    // …and the template document the base pins, which is outside it.
    expect(files).toContain('data/models/decoder-causal-yarn/1.0.0.json');
  });

  it('lets the load resolve the template pin, which a base of its own files alone cannot', async () => {
    const { lang, schemas } = await core();
    const place = dataWorkspace();
    const tree = await lang.parse((await place.read(MODEL)).text);

    const bare = { base: 'data/primitive-library', files: filesUnder('data/primitive-library') };
    const without = await lang.loadLibrary([bare], schemas);
    // The loader's own words for a pin whose document was not handed over — which is exactly the
    // refusal "a base handed over without them is refused for a reason the workspace, not the
    // base, is responsible for" names.
    expect(without.problems.map((one) => one.message).join('\n')).toContain(
      "template 'decoder-causal-yarn' 1.0.0 is not at data/models/decoder-causal-yarn/1.0.0.json",
    );

    const gathered = await gatherBases(lang, schemas, place, tree, MODEL);
    const withThem = await lang.loadLibrary(gathered.bases, schemas);
    expect(withThem.problems).toEqual([]);
    expect(withThem.library.templates.size).toBe(1);
  });

  it('hands a base the workspace does not hold over empty, and lets the loader say so', async () => {
    const { lang, schemas } = await core();
    const place = MemoryWorkspace.of({ 'models/x.json': corpusText('llama3-8b') });
    const tree = await lang.parse((await place.read('models/x.json')).text);
    const gathered = await gatherBases(lang, schemas, place, tree, 'models/x.json');
    expect(gathered.bases).toEqual([{ base: 'primitive-library', files: {} }]);
  });
});

describe('a base fetched from an address and standing in the workspace (feature 2.20)', () => {
  /** The acceptance fixture base of feature 1.13, which is what a laboratory's own base looks like. */
  function fixtureBase(): Record<string, string> {
    const root = join(editorRoot, 'tests', 'fixtures', 'base');
    const found: Record<string, string> = {};
    const walk = (at: string, prefix: string): void => {
      for (const entry of readdirSync(at, { withFileTypes: true })) {
        if (entry.isDirectory()) walk(join(at, entry.name), `${prefix}${entry.name}/`);
        else if (entry.name.endsWith('.json')) {
          found[`${prefix}${entry.name}`] = readFileSync(join(at, entry.name), 'utf8');
        }
      }
    };
    walk(root, '');
    return found;
  }

  /** Where a document that uses the fetched base stands, in a workspace laid out like `data/`. */
  const MODEL_AT = 'data/models/uses-fixture.json';

  /** `declared-constant.json`, with its two bases written where this workspace keeps them. */
  function document(where: string): string {
    const text = readFileSync(
      join(editorRoot, 'tests', 'fixtures', 'models', 'declared-constant.json'),
      'utf8',
    );
    return text
      .replace('../../../../data/primitive-library/', '../primitive-library/')
      .replace('../base/', `../${where}/`);
  }

  it('is gathered from the mount, and the loader reads it as a base like any other', async () => {
    const { lang, schemas } = await core();
    const place = dataWorkspace();
    // The workspace holds the corpus and the reference base and knows nothing of `lab-base`: the
    // mount is what fills the path, and the path is what the document pins.
    await place.write(MODEL_AT, document('lab-base'), ABSENT);
    const tree = await lang.parse((await place.read(MODEL_AT)).text);
    const mounts = [
      { at: 'data/lab-base', texts: fixtureBase(), address: 'https://lab.example/lab-base/vendor.json' },
    ];
    const gathered = await gatherBases(lang, schemas, place, tree, MODEL_AT, mounts);
    expect(gathered.bases.map((one) => one.base)).toEqual([
      'data/primitive-library',
      'data/lab-base',
    ]);
    expect(Object.keys(gathered.bases[1]?.files ?? {}).sort()).toEqual([
      'data/lab-base/primitive-library.json',
      'data/lab-base/primitives/fixture/position_bias/1.0.0.json',
    ]);
    // Data, not code: the loader puts every unit through the unit schema and its cross-references,
    // and answers a library with the fetched base's own primitive in it.
    const loaded = await lang.loadLibrary(gathered.bases, schemas, { forDocument: MODEL_AT });
    expect(loaded.problems).toEqual([]);
    expect([...loaded.library.byId.keys()]).toContain('fixture.position_bias@1.0.0');
  });

  it('is what the document is judged against: the same document without it is a V1', async () => {
    const { lang, schemas } = await core();
    const place = dataWorkspace();
    await place.write(MODEL_AT, document('lab-base'), ABSENT);
    const tree = await lang.parse((await place.read(MODEL_AT)).text);
    const gathered = await gatherBases(lang, schemas, place, tree, MODEL_AT);
    expect(gathered.bases[1]).toEqual({ base: 'data/lab-base', files: {} });
    const loaded = await lang.loadLibrary(gathered.bases, schemas, { forDocument: MODEL_AT });
    // The loader's own words, in the tools' wording: a base that is not there is a rejection.
    expect(loaded.problems.map((one) => one.message).join('\n')).toContain(
      "primitive library base 'data/lab-base' does not exist (V1)",
    );
  });

  it('answers for what stands under it, whichever side of the mount the base is', async () => {
    const { lang, schemas } = await core();
    const place = dataWorkspace();
    await place.write(MODEL_AT, document('bases/lab'), ABSENT);
    const tree = await lang.parse((await place.read(MODEL_AT)).text);
    const mounts = [
      { at: 'data/bases/lab', texts: fixtureBase(), address: 'https://lab.example/lab/vendor.json' },
    ];
    const gathered = await gatherBases(lang, schemas, place, tree, MODEL_AT, mounts);
    expect(Object.keys(gathered.bases[1]?.files ?? {})).toContain(
      'data/bases/lab/primitives/fixture/position_bias/1.0.0.json',
    );
  });
});

describe('the schemas a workspace is read against', () => {
  it('are the build’s where the workspace carries none (§1)', async () => {
    const vendored = { 'schemas/tensorspine.schema.json': '{}' };
    const gathered = await gatherSchemas(dataWorkspace(), vendored);
    expect(gathered.fromWorkspace).toBe(false);
    expect(gathered.files).toBe(vendored);
  });

  it('are the workspace’s own where it carries some, which §1 admits', async () => {
    const own = MemoryWorkspace.of({ [`${WORKSPACE_SCHEMAS}/tensorspine.schema.json`]: '{"$id": "x"}' });
    const gathered = await gatherSchemas(own, { 'schemas/tensorspine.schema.json': '{}' });
    expect(gathered.fromWorkspace).toBe(true);
    expect(Object.keys(gathered.files)).toEqual([`${WORKSPACE_SCHEMAS}/tensorspine.schema.json`]);
  });

  it('report every difference with the build’s copy as a line for the log, never a refusal', () => {
    const lines = schemaDifferences(
      { 'schemas/a.json': '{"a": 1}', 'schemas/c.json': '{}' },
      { 'vendor/schemas/a.json': '{"a": 2}', 'vendor/schemas/b.json': '{}' },
    );
    expect(lines.map((one) => one.message)).toEqual([
      'schemas/a.json differs from the a.json the build vendors',
      "schemas/c.json is the workspace's own; the build vendors no c.json",
      "the workspace's schemas carry no b.json; the build's copy is not used",
    ]);
  });
});

describe('reading a set of files', () => {
  it('uses the workspace’s own bulk read where it has one, and answers the same map', async () => {
    const texts = { 'a/one.json': '1', 'a/two.json': '2', 'a/b/three.json': '3' };
    const asked: string[][] = [];
    const set = textsOf('examples', 'x', 'x', texts);
    const bulk = new ReadOnlyWorkspace(
      {
        ...set,
        readMany: async (paths) => {
          asked.push([...paths]);
          const found: Record<string, { text: string; revision: string }> = {};
          for (const path of paths) {
            const one = await set.read(path);
            if (one !== null) found[path] = one;
          }
          return found;
        },
      },
      () => Promise.resolve(),
    );
    const one = new ReadOnlyWorkspace(set, () => Promise.resolve());
    expect(await readTree(bulk, '', { suffix: '.json' })).toEqual(texts);
    expect(await readTree(one, '', { suffix: '.json' })).toEqual(texts);
    expect(asked).toHaveLength(1);
    expect(asked[0]?.sort()).toEqual(Object.keys(texts).sort());
  });
});
