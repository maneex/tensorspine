import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import type { Lang, LibraryHandle } from '@tensorspine/lang/api';

import { schemaTexts } from '../documents/source.js';

import { bannerOf, BANNERS } from '../../src/problems/banner.js';
import { core, corpusPath, readRepositoryFile, referenceFiles, verdictOf } from './source.js';
import { parse } from '@tensorspine/lang';

/** The reference base with one unit whose declared name disagrees with the file it is in. */
async function brokenLibrary(): Promise<LibraryHandle> {
  const unit = 'data/primitive-library/primitives/norm/rms/1.0.0.json';
  const files = { ...referenceFiles() };
  const text = files[unit];
  if (text === undefined) throw new Error(`${unit} is not in the reference base`);
  files[unit] = text.replace('"name": "norm.rms"', '"name": "norm.wrong"');
  const gathered = await lang.loadLibrary(
    [{ base: 'data/primitive-library', files }],
    (await lang.loadSchemas(schemaTexts(), { origin: 'schemas' })).handle,
  );
  expect(gathered.problems.length).toBeGreaterThan(0);
  return gathered.handle;
}

// §4.17's banner, and plan §3's continuation behind it. Each of the four is a state the core
// actually puts a document in, reached by running the real validator over a real document; the
// banner is the editor's own words about the core's own answer, and nothing else.

let lang: Lang;
let library: LibraryHandle;
let stop: () => void;

beforeAll(async () => {
  const session = await core();
  lang = session.lang;
  library = session.library;
  stop = session.stop;
}, 120_000);

afterAll(() => {
  stop();
});

describe('a document the core is happy with', () => {
  it('carries no banner', async () => {
    const { verdict } = await verdictOf(lang, library, 'llama3-8b');
    expect(bannerOf(verdict)).toBeNull();
    expect(bannerOf(null)).toBeNull();
  });
});

describe('a document off the grammar', () => {
  it('says meaning assumes grammar, and that nothing after the schema stage ran (D5)', async () => {
    const tree = parse('{"schema": "tensorspine/2.0"}');
    const verdict = await lang.validate(tree, 'scratch.json', { library });
    expect(verdict.stagesRun).toEqual(['schema']);
    const banner = bannerOf(verdict);
    expect(banner?.kind).toBe('off-grammar');
    expect(banner?.tone).toBe('stop');
    expect(banner?.head).toBe(BANNERS['off-grammar'].head);
  });
});

describe('a library that would not gather', () => {
  it('says the rows below are the core’s continuation, which the tools do not print', async () => {
    // A base with a unit whose declared identity disagrees with the file it is in: the loader
    // refuses it, in the tools' own words, and `--validate` would stop there. The core keeps
    // checking what remains decidable and marks every later row `afterRefusal` — plan §3's
    // "extra rows, never fewer", which the parity job does not compare.
    const broken = await brokenLibrary();
    const verdict = await lang.validate(
      parse(readRepositoryFile('data/models/llama3-8b.json')),
      corpusPath('llama3-8b'),
      { library: broken },
    );
    expect(verdict.problems.some((one) => one.source === 'library')).toBe(true);
    expect(verdict.problems.some((one) => one.afterRefusal === true)).toBe(true);
    const banner = bannerOf(verdict);
    expect(banner?.kind).toBe('after-refusal');
    expect(banner?.tone).toBe('stop');
  });
});

describe('a template with no assignment', () => {
  it('says it is skipped rather than failed (§4.6, I7)', async () => {
    const text = readRepositoryFile('data/models/decoder-causal-yarn/1.0.0.json');
    const verdict = await lang.validate(
      parse(text),
      'data/models/decoder-causal-yarn/1.0.0.json',
      { library },
    );
    expect(verdict.needsAssignment).toBeDefined();
    const banner = bannerOf(verdict);
    expect(banner?.kind).toBe('needs-assignment');
    expect(banner?.tone).toBe('info');
  });
});

describe('an assignment the document refuses', () => {
  it('says nothing was analysed under it', async () => {
    const text = readRepositoryFile('data/models/decoder-causal-yarn/1.0.0.json');
    const verdict = await lang.validate(
      parse(text),
      'data/models/decoder-causal-yarn/1.0.0.json',
      {
        library,
        // Every external quantity has a value, and one of them is not admissible.
        assignment: {
          width: 3072n,
          layers: 26n,
          heads: 32n,
          kv_heads: 8n,
          head_dim: 128n,
          inner: 9216n,
          eps: 5,
          precision: 'bf16',
        },
      },
    );
    expect(verdict.needsAssignment).toBeUndefined();
    expect(verdict.stagesRun).not.toContain('semantic');
    expect(verdict.problems.length).toBeGreaterThan(0);
    const banner = bannerOf(verdict);
    expect(banner?.kind).toBe('assignment-refused');
    expect(banner?.tone).toBe('warn');
  });
});

describe('the four are a closed set', () => {
  it('each says a head and a body, and each carries a tone and a mark', () => {
    for (const [kind, said] of Object.entries(BANNERS)) {
      expect(said.head, kind).not.toBe('');
      expect(said.body, kind).not.toBe('');
      expect(['stop', 'warn', 'info'], kind).toContain(said.tone);
      expect(said.mark, kind).not.toBe('');
    }
  });
});
