import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { LangCancelled, LangFailure, LangHandleError } from '../../src/api/index.js';
import type { Lang, LibraryHandle, Problem, SchemasHandle } from '../../src/api/index.js';
import { UNRESOLVED, type PyRecord, type PyValue } from '../../src/expr/value.js';
import { serialize } from '../../src/json/serialize.js';
import { toJsonValue } from '../../src/expr/value.js';
import { corpus, corpusPath, loaded, referenceBase, schemaFiles, open, type Deployment } from './source.js';

// The conformance suite of the `Lang` API: every call of plan Appendix C, run against **both**
// implementations — the core in this thread, and the core behind a structured-clone boundary.
// What it proves is not that the core is right (the parity suites do that) but that the seam
// carries its answers: the same verdicts, the same facts, the same products, the same refusals,
// and the sentinel that a structured clone would have dropped.

const DEPLOYMENTS: Deployment[] = ['in-process', 'worker'];

for (const deployment of DEPLOYMENTS) {
  describe(`the Lang API (${deployment})`, () => {
    let lang: Lang;
    let library: LibraryHandle;
    let schemas: SchemasHandle;
    let stop: () => void;

    beforeAll(async () => {
      const session = await loaded(deployment);
      lang = session.lang;
      library = session.library;
      schemas = session.schemas;
      stop = session.stop;
    }, 120_000);

    afterAll(() => {
      stop();
    });

    // --- the vocabulary ------------------------------------------------------

    it('loads the repository’s schemas and answers what is indexed', async () => {
      const answer = await lang.loadSchemas(schemaFiles(), { origin: 'schemas' });
      expect(answer.origin).toBe('schemas');
      const roles = answer.schemas.map((one) => one.role).sort();
      // The four the plan names, and the fifth the directory carries (feature 0.2).
      expect(roles).toEqual([
        'derived',
        'documentation',
        'fixture',
        'model',
        'primitive-library-unit',
      ]);
      await lang.release(answer.handle);
    });

    it('gathers the reference base clean, with its template interface', async () => {
      const answer = await lang.loadLibrary([referenceBase()], (await lang.loadSchemas(schemaFiles())).handle);
      expect(answer.problems).toEqual([]);
      expect(answer.library.byId.size).toBe(36);
      expect(answer.library.axes.size).toBeGreaterThan(0);
      expect(answer.library.precision.size).toBeGreaterThan(0);
      expect([...answer.library.templates.keys()]).toEqual(['decoder.causal_yarn@1.0.0']);
      await lang.release(answer.handle);
    });

    it('refuses to work under a handle it does not hold', async () => {
      const gone: LibraryHandle = { kind: 'library', id: 'library-does-not-exist' };
      await expect(lang.validateUnit(null, 'x.json', gone)).rejects.toThrow(
        deployment === 'in-process' ? LangHandleError : LangFailure,
      );
    });

    // --- the document --------------------------------------------------------

    it('parses and serializes a corpus document back to its own bytes (D12)', async () => {
      const text = corpus('llama3-8b');
      const tree = await lang.parse(text);
      expect(await lang.serialize(tree)).toBe(text);
    });

    it('reports a duplicate member name in the tools’ words (V12)', async () => {
      const refusal = lang.parse('{"model": 1, "model": 2}');
      await expect(refusal).rejects.toThrow(/duplicate member name 'model' \(V12\)/);
    });

    it('validates the corpus clean, stage by stage', async () => {
      const tree = await lang.parse(corpus('llama3-8b'));
      const verdict = await lang.validate(tree, corpusPath('llama3-8b'), { library });
      expect(verdict.problems).toEqual([]);
      expect(verdict.stagesRun).toEqual(['schema', 'library', 'assignment', 'semantic']);
      expect(verdict.needsAssignment).toBeUndefined();
      expect(verdict.stats.get('instances')).toBeDefined();
      // The counters are Python's integers and they cross as such (feature 1.2).
      expect(typeof verdict.stats.get('instances')).toBe('bigint');
    });

    it('stops at the grammar: meaning assumes grammar', async () => {
      const tree = await lang.parse('{"schema": "tensorspine/2.0"}');
      const verdict = await lang.validate(tree, 'scratch.json', { library });
      expect(verdict.stagesRun).toEqual(['schema']);
      expect(verdict.problems.length).toBeGreaterThan(0);
      expect(verdict.problems.every((one) => one.source === 'schema')).toBe(true);
      expect(verdict.problems.every((one) => one.severity === 'error')).toBe(true);
      expect(verdict.stats.size).toBe(0);
    });

    it('refuses a document in the tools’ own wording, with the pointer beside it', async () => {
      const tree = await lang.parse(corpus('llama3-8b').replace('"norm.rms"', '"norm.rmz"'));
      const verdict = await lang.validate(tree, corpusPath('llama3-8b'), { library });
      const first = verdict.problems[0] as Problem;
      expect(first.source).toBe('semantic');
      expect(first.code).toBe('V1');
      expect(first.message).toMatch(/^\[V1\] /);
      expect(first.message).toContain('primitive absent from primitive library');
      expect(first.path).toMatch(/^\/(instances|compositions)\//);
      expect(first.file).toBe(corpusPath('llama3-8b'));
    });

    it('skips the semantic stage where an external quantity has no value', async () => {
      const tree = await lang.parse(corpus('decoder-causal-yarn/1.0.0'));
      const verdict = await lang.validate(tree, 'data/models/decoder-causal-yarn/1.0.0.json', {
        library,
      });
      expect(verdict.stagesRun).toEqual(['schema', 'library', 'assignment']);
      expect(verdict.needsAssignment?.unset).toContain('layers');
      expect(verdict.needsAssignment?.report).toMatch(/^needs --assign for /);
      expect(verdict.problems).toEqual([]);
    });

    it('runs the lint stage over the set the caller names, and not otherwise', async () => {
      const tree = await lang.parse(corpus('llama3-8b'));
      const path = corpusPath('llama3-8b');
      const alone = await lang.validate(tree, path, { library });
      expect(alone.stagesRun).not.toContain('lint');

      const linted = await lang.validate(tree, path, {
        library,
        lint: [{ path, text: corpus('llama3-8b') }],
      });
      expect(linted.stagesRun).toContain('lint');
      // Feature 1.10: the answer is a function of the set, so one document leaves the rest of the
      // base uncalled — thirty-odd curation rows, every one a warning and none a refusal.
      expect(linted.problems.some((one) => one.source === 'lint')).toBe(true);
      expect(linted.problems.filter((one) => one.source === 'lint').every((one) => one.severity === 'warning')).toBe(true);
      expect(
        linted.problems.some((one) => one.rule === 'uncalled' && one.message.includes('called by none of the 1 model(s) linted')),
      ).toBe(true);
    });

    // --- describe and check ---------------------------------------------------

    it('warns where the document declares a base the session did not gather', async () => {
      // The session's own guard, and the one place it is not the tools' wording: the tools load
      // the library *from the document* on every run, so the two can never disagree there, while
      // the API gathers once a session so that a keystroke does not pay the load's 261 ms. It is a
      // warning, so no verdict moves.
      const tree = await lang.parse(corpus('llama3-8b'));
      const verdict = await lang.validate(tree, 'somewhere/else/llama3-8b.json', { library });
      const warning = verdict.problems.find((one) => one.rule === 'bases');
      expect(warning?.severity).toBe('warning');
      expect(warning?.source).toBe('library');
      expect(warning?.message).toContain("does not carry the base 'somewhere/primitive-library'");
      expect(verdict.problems.filter((one) => one.severity === 'error')).toEqual([]);
    });

    it('validates a unit of a base spelled with a trailing separator', async () => {
      // `data/primitive-library` and `data/primitive-library/` are one directory: `memorySource`
      // says so on the way in, and a base the workspace hands over from a directory picker is as
      // likely to carry the separator as not. The unit is the base's own, unchanged.
      const base = referenceBase();
      const gathered = await lang.loadLibrary([{ ...base, base: `${base.base}/` }], schemas);
      expect(gathered.problems).toEqual([]);
      const path = 'data/primitive-library/primitives/norm/rms/1.0.0.json';
      const text = base.files[path];
      if (text === undefined) throw new Error('missing corpus fixture');
      const unit = await lang.parse(text);
      expect(await lang.validateUnit(unit, path, gathered.handle)).toEqual([]);
      await lang.release(gathered.handle);
    });

    it('refuses a unit that lies under no gathered base, naming where it does lie', async () => {
      const base = referenceBase();
      const text = base.files['data/primitive-library/primitives/norm/rms/1.0.0.json'];
      if (text === undefined) throw new Error('missing corpus fixture');
      const unit = await lang.parse(text);
      const problems = await lang.validateUnit(unit, 'elsewhere/norm/rms/1.0.0.json', library);
      expect(problems.map((one) => one.message)).toEqual([
        'elsewhere/norm/rms/1.0.0.json: not under primitives/, axes/ or precision/ of elsewhere/norm/rms',
      ]);
    });

    it('leaves the analysis where check can read it, on one revision', async () => {
      const tree = await lang.parse(corpus('llama3-8b'));
      const path = corpusPath('llama3-8b');
      const verdict = await lang.validate(tree, path, { library, revision: 7 });
      expect(verdict.problems).toEqual([]);
      // §5.4's pipeline: `validate` and `describe` on one revision read one `analyse`, and `check`
      // reads what they left. What this holds is the seam — that a `check` answers at all after a
      // `validate` alone, which is only true if the analysis stayed.
      const facts = await lang.describe(tree, path, { library, revision: 7, only: ['embed'] });
      expect([...facts.sites.keys()]).toEqual(['embed']);
      const decision = await lang.check(path, {
        location: { identity: 'not_an_identity', location: { tensor: 'x' } },
      });
      expect(decision.ok).toBe(false);
      expect(decision.unknown).toBeTruthy();
    });

    it('describes every site of a corpus document', async () => {
      const tree = await lang.parse(corpus('llama3-8b'));
      const facts = await lang.describe(tree, corpusPath('llama3-8b'), { library });
      expect(facts.conforms).toBe(true);
      expect(facts.structural).toEqual([]);
      // Keyed by the identifier §5.2 rule 2 gives a site, which is what D1 lists.
      expect(facts.sites.has('embed')).toBe(true);
      expect(facts.sites.has('decoder/attn[layer=0]')).toBe(true);
      const attn = facts.sites.get('decoder/attn[layer=0]');
      expect(attn?.primitive).toBe('attention.dense');
      expect(attn?.inputs.map((port) => port.name)).toContain('input');
      expect(attn?.states.map((state) => state.name)).toEqual(['kv']);
      // Feature 1.6d's reading, carried across the boundary unchanged: a causal, non-cross,
      // window-less instance selects the fourth of `attention.dense`'s four ordered rules.
      expect(attn?.states[0]?.ruleIndex).toBe(3);
    });

    it('describes only the sites the caller asked for', async () => {
      const tree = await lang.parse(corpus('llama3-8b'));
      const one = await lang.describe(tree, corpusPath('llama3-8b'), {
        library,
        only: ['decoder/attn[layer=0]'],
      });
      expect([...one.sites.keys()]).toEqual(['decoder/attn[layer=0]']);
      const all = await lang.describe(tree, corpusPath('llama3-8b'), { library });
      expect(all.sites.get('decoder/attn[layer=0]')).toEqual(one.sites.get('decoder/attn[layer=0]'));
    });

    it('reports a template as needing its assignment rather than raising on it', async () => {
      // Every call site of the tools checks `missing_assignment` before it analyses — "a template
      // with no assignment is skipped, not failed" — and `describe` is called on every keystroke,
      // so it takes the same gate instead of letting the expansion's `index 'layer': stop does not
      // resolve to a value` out.
      const path = 'data/models/decoder-causal-yarn/1.0.0.json';
      const tree = await lang.parse(corpus('decoder-causal-yarn/1.0.0'));
      const facts = await lang.describe(tree, path, { library });
      expect(facts.conforms).toBe(true);
      expect(facts.sites.size).toBe(0);
      expect(facts.needsAssignment?.unset).toContain('layers');

      const decision = await lang.check(path, {
        location: { identity: 'wq[layer=0]', location: { tensor: 'x' } },
      });
      expect(decision.ok).toBe(false);
      expect(decision.unknown).toContain('its assignment is incomplete or refused');
    });

    it('describes a template under an assignment', async () => {
      const path = 'data/models/decoder-causal-yarn/1.0.0.json';
      const tree = await lang.parse(corpus('decoder-causal-yarn/1.0.0'));
      // `tests/signature.py`'s documented assignment for `decoder-causal-yarn@1.0.0`, which is
      // what the template's own suites read it under (feature 0.1's finding).
      const assignment = {
        width: 3072n,
        layers: 26n,
        heads: 32n,
        kv_heads: 8n,
        head_dim: 128n,
        inner: 9216n,
        eps: 1e-5,
        precision: 'bf16',
      };
      const facts = await lang.describe(tree, path, { library, assignment });
      expect(facts.needsAssignment).toBeUndefined();
      expect(facts.sites.size).toBeGreaterThan(0);
    });

    it('answers the schema’s problems and no facts for a document off the grammar', async () => {
      const tree = await lang.parse('{"schema": "tensorspine/2.0"}');
      const facts = await lang.describe(tree, 'scratch.json', { library });
      expect(facts.conforms).toBe(false);
      expect(facts.sites.size).toBe(0);
      expect(facts.structural.length).toBeGreaterThan(0);
    });

    it('checks a candidate edge against the reading it holds', async () => {
      const tree = await lang.parse(corpus('llama3-8b'));
      const path = corpusPath('llama3-8b');
      const facts = await lang.describe(tree, path, { library });
      const attn = facts.sites.get('decoder/attn[layer=0]');
      const embed = facts.sites.get('embed');
      if (attn === undefined || embed === undefined) throw new Error('the sites are missing');

      const verdict = await lang.check(path, {
        edge: {
          from: { site: embed.key, port: 'output' },
          to: { site: attn.key, port: 'input' },
        },
      });
      // V7: the input is already fed by the document's own binding, so the candidate is refused —
      // with the reason, never by refusing the gesture (Q5, "wire first, fix later").
      expect(verdict.ok).toBe(false);
      expect(verdict.problems.map((one) => one.code)).toContain('V7');
    });

    it('refuses to check a document it has not read', async () => {
      await expect(lang.check('nothing-was-read.json', { edge: { from: { site: { kind: 'root', composition: '', name: 'a', indices: [] }, port: 'p' }, to: { site: { kind: 'root', composition: '', name: 'b', indices: [] }, port: 'q' } } })).rejects.toThrow(
        /no reading is held/,
      );
    });

    // --- the products ---------------------------------------------------------

    it('expands a corpus document to D1, envelope and all', async () => {
      // What D1 *is* is feature 1.7's parity suite, byte for byte against `--d1`; what this holds
      // is that the seam carries it — the envelope, the nodes, and the integers as integers.
      const tree = await lang.parse(corpus('llama3-8b'));
      const one = await lang.expand(tree, { library });
      expect(Object.keys(one)).toEqual([
        'schema',
        'model',
        'primitive_libraries',
        'assignment',
        'd1',
      ]);
      const d1 = one['d1'] as PyRecord;
      expect(Object.keys(d1['nodes'] as PyRecord)).toContain('decoder/attn[layer=0]');
      expect(serialize(toJsonValue(one)).endsWith('\n')).toBe(true);
    });

    it('derives a corpus document and validates it against the derived schema', async () => {
      const tree = await lang.parse(corpus('llama3-8b'));
      const derived = await lang.derive(tree, corpusPath('llama3-8b'), { library });
      expect(Object.keys(derived)).toEqual([
        'schema',
        'model',
        'primitive_libraries',
        'assignment',
        'd1',
        'd2',
        'd3',
        'd4',
        'd5',
        'd6',
      ]);
      const totals = (derived['d3'] as PyRecord)['totals'] as PyRecord;
      // `--derive` writes 291 tensors and 16 060 522 496 bytes for llama3-8b, as whole numbers:
      // the int/float distinction feature 1.2 keeps is what crosses as a `bigint`.
      expect(totals['tensors']).toBe(291n);
      expect(totals['bytes']).toBe(16_060_522_496n);
    });

    it('refuses to derive a document that does not validate, in the tools’ words', async () => {
      const tree = await lang.parse(corpus('llama3-8b').replace('"norm.rms"', '"norm.rmz"'));
      await expect(lang.derive(tree, 'scratch.json', { library })).rejects.toThrow(
        /^not valid, no products: \[V1\] /,
      );
    });

    // --- the checkpoint and the expressions ------------------------------------

    it('reads a safetensors header and checks a derived document against it', async () => {
      const header = {
        __metadata__: { format: 'pt' },
        'model.embed_tokens.weight': { dtype: 'BF16', shape: [128256, 4096], data_offsets: [0, 1] },
      };
      const json = new TextEncoder().encode(JSON.stringify(header));
      const bytes = new Uint8Array(8 + json.length);
      new DataView(bytes.buffer).setBigUint64(0, BigInt(json.length), true);
      bytes.set(json, 8);

      const read = await lang.readHeader(bytes, 'model.safetensors');
      expect(read.entries['model.embed_tokens.weight']?.dtype).toBe('bf16');
      expect(read.entries['model.embed_tokens.weight']?.known).toBe(true);

      const tree = await lang.parse(corpus('llama3-8b'));
      const derived = await lang.derive(tree, corpusPath('llama3-8b'), { library });
      const report = await lang.checkCheckpoint(derived, read.entries);
      // llama3-8b locates every tensor, and this header holds one of them: every other location
      // is missing from the checkpoint, and the one that is there matches.
      expect(report.stats.physical).toBe(1);
      expect(report.stats.unnamed).toBe(0);
      expect(report.errors.every((one) => one.source === 'checkpoint' && one.code === 'V17')).toBe(true);
      expect(report.rows.length).toBe(report.errors.length + report.warnings.length);
    });

    it('evaluates a model expression against a document’s quantities', async () => {
      const quantities = new Map<string, PyValue>([
        ['d', 4096n],
        ['heads', 32n],
      ]);
      const value = await lang.evaluate(
        { op: 'floor_divide', args: [{ quantity: 'd' }, { quantity: 'heads' }] },
        quantities,
      );
      expect(value).toBe(128n);
    });

    it('carries the sentinel across the boundary rather than dropping it', async () => {
      // Feature 1.2: `UNRESOLVED` is a symbol and a symbol is a `DataCloneError`. This is the
      // call where the editor meets one — an expression over a quantity nothing supplies.
      const value = await lang.evaluate({ quantity: 'nothing' }, new Map());
      expect(value).toBe(UNRESOLVED);
    });

    // --- the session’s own state -------------------------------------------------

    it('forgets a document, and then has nothing to check against', async () => {
      const tree = await lang.parse(corpus('llama3-8b'));
      const path = corpusPath('llama3-8b');
      await lang.describe(tree, path, { library });
      await lang.forget(path);
      await expect(
        lang.check(path, {
          location: { identity: 'wq[layer=0]', location: { tensor: 'x' } },
        }),
      ).rejects.toThrow(/no reading is held/);
    });

    it('answers nothing once it is closed', async () => {
      const other = open(deployment);
      other.close();
      await expect(other.parse('{}')).rejects.toThrow(LangCancelled);
    });
  });
}

// How a base is *spelled* is path arithmetic in the session, which is one body for both
// deployments (`api/session.ts`); it is exercised once rather than twice, and the two calls that
// read it — `validateUnit` and the missing-base warning — are covered against both above.
describe('the spellings of one base', () => {
  let lang: Lang;
  let schemas: SchemasHandle;
  let stop: () => void;

  beforeAll(async () => {
    const session = await loaded('in-process');
    lang = session.lang;
    schemas = session.schemas;
    stop = session.stop;
  }, 120_000);

  afterAll(() => {
    stop();
  });

  const UNIT = 'data/primitive-library/primitives/norm/rms/1.0.0.json';

  it('reads a unit of the base under every spelling `normalise` makes one path of', async () => {
    const base = referenceBase();
    const text = base.files[UNIT];
    if (text === undefined) throw new Error('missing corpus fixture');
    const unit = await lang.parse(text);
    for (const spelling of [
      'data/primitive-library',
      'data/primitive-library/',
      'data/primitive-library//',
      './data/primitive-library',
      'data/./primitive-library',
      'data/models/../primitive-library',
    ]) {
      const gathered = await lang.loadLibrary([{ ...base, base: spelling }], schemas);
      expect(gathered.problems, spelling).toEqual([]);
      expect(await lang.validateUnit(unit, UNIT, gathered.handle), spelling).toEqual([]);
      await lang.release(gathered.handle);
    }
  });

  it('recognises the base a document declares under any of them', async () => {
    // `llama3-8b.json` writes `"../primitive-library/"`, which `basesOf` normalises; a base
    // gathered under the document's own spelling must not then read as one the session has not
    // got. The warning is the session's own guard and it says a real thing — the test beside it
    // holds that — so it must not also say it of a base that is there.
    const base = referenceBase();
    const tree = await lang.parse(corpus('llama3-8b'));
    for (const spelling of ['data/primitive-library', 'data/primitive-library/', './data/primitive-library']) {
      const gathered = await lang.loadLibrary([{ ...base, base: spelling }], schemas);
      const verdict = await lang.validate(tree, corpusPath('llama3-8b'), { library: gathered.handle });
      expect(verdict.problems, spelling).toEqual([]);
      await lang.release(gathered.handle);
    }
  });

  it('reads a unit against the innermost base that contains it', async () => {
    // Two bases, one inside the other: the unit belongs to the inner one, whose `primitives/`
    // section root is what gives it its identity. Taking the outer would name it `norm.rms.1.0.0`
    // under a section it is not in, which is the identity check's whole subject (§8.2).
    const base = referenceBase();
    const inner = 'data/primitive-library/primitives/norm/';
    const text = base.files[UNIT];
    if (text === undefined) throw new Error('missing corpus fixture');
    const unit = await lang.parse(text);
    const gathered = await lang.loadLibrary(
      [{ ...base, base: 'data/primitive-library' }, { base: inner, files: {} }],
      schemas,
    );
    const problems = await lang.validateUnit(unit, UNIT, gathered.handle);
    expect(problems.map((one) => one.message)).toEqual([
      `${UNIT}: not under primitives/, axes/ or precision/ of ${inner}`,
    ]);
    await lang.release(gathered.handle);
  });

  it('does not read a sibling directory as inside a base', async () => {
    // `data/primitive-library` begins with the text of `data/primitive-lib` and is not inside it:
    // what separates a directory from its sibling is the separator, so that is what is compared.
    const base = referenceBase();
    const manifest = base.files['data/primitive-library/primitive-library.json'];
    const text = base.files[UNIT];
    if (manifest === undefined || text === undefined) throw new Error('missing corpus fixture');
    const unit = await lang.parse(text);
    const sibling = {
      base: 'data/primitive-lib',
      files: { 'data/primitive-lib/primitive-library.json': manifest },
    };
    const gathered = await lang.loadLibrary([sibling], schemas);
    const problems = await lang.validateUnit(unit, UNIT, gathered.handle);
    expect(problems.map((one) => one.message)).toEqual([
      `${UNIT}: not under primitives/, axes/ or precision/ of data/primitive-library/primitives/norm/rms`,
    ]);
    await lang.release(gathered.handle);
  });
});
