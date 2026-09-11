import { afterEach, describe, expect, it } from 'vitest';

import {
  connectLang,
  LangCancelled,
  LangFailure,
  type Lang,
  type LibraryHandle,
  type Progress,
} from '../../src/api/index.js';
import { createLang, serveLang, type LangHost } from '../../src/api/engine.js';
import { BLANK_MODEL, SCRATCH } from '../derive/source.js';
import { corpus, corpusPath, referenceBase, schemaFiles } from './source.js';

// The worker: request ids, cancellation, one in-flight derivation per document, progress.
//
// The boundary here is a `MessageChannel` — the host on one port, the client on the other. It is
// not a thread, and it is the strongest thing the unit layer can be: `packages/lang` names its own
// imports with the `.js` specifiers a browser bundler resolves and plain Node does not, so a Node
// worker thread cannot load the core at all. What a channel *is* is a true structured-clone
// boundary, measured: a `postMessage` between two ports of one channel serializes, refuses a
// symbol with the same `DataCloneError` a thread would, and hands the other side a copy. The
// thread itself is `apps/web/e2e/worker.spec.ts`, where a real `Worker` runs the built bundle.

interface Connected {
  readonly lang: Lang;
  readonly host: LangHost;
  readonly close: () => void;
}

/**
 * Where the scratch document is opened from.
 *
 * Its `primitive_libraries` are written relative to `data/models/`, as every corpus document's
 * are: `../primitive-library/` resolves to the reference base and `../../scratch/base/` to the
 * base the case adds. Opening it anywhere else resolves them elsewhere, which is exactly what
 * the session's own bases guard reports.
 */
const SCRATCH_MODEL_PATH = 'data/models/scratch-blank.json';

const open: Connected[] = [];

function connect(): Connected {
  const channel = new MessageChannel();
  const host = serveLang(channel.port2);
  const lang = connectLang(channel.port1, {
    onClose: () => {
      host.stop();
      channel.port1.close();
      channel.port2.close();
    },
  });
  const connected: Connected = { lang, host, close: () => lang.close() };
  open.push(connected);
  return connected;
}

/** A connection with the schemas and the reference base loaded, and a document parsed. */
async function ready(): Promise<{
  readonly lang: Lang;
  readonly library: LibraryHandle;
  readonly tree: unknown;
  readonly path: string;
}> {
  const { lang } = connect();
  const schemas = await lang.loadSchemas(schemaFiles(), { origin: 'schemas' });
  const library = await lang.loadLibrary([referenceBase()], schemas.handle);
  const tree = await lang.parse(corpus('llama3-8b'));
  return { lang, library: library.handle, tree, path: corpusPath('llama3-8b') };
}

afterEach(() => {
  while (open.length > 0) open.pop()?.close();
});

describe('the worker round trip', () => {
  it('answers each request under its own id, whatever order the answers come in', async () => {
    const { lang } = await ready();
    // Three calls of very different weights, posted at once: what must hold is that each promise
    // gets its own answer, not that they finish in order.
    const [first, second, third] = await Promise.all([
      lang.parse('{"a": 1}'),
      lang.serialize({
        kind: 'object',
        members: [{ name: 'b', value: { kind: 'number', value: 2, real: false, lexeme: '2' } }],
      }),
      lang.evaluate({ literal: 3n }, new Map()),
    ]);
    expect(first).toMatchObject({ kind: 'object' });
    expect(second).toBe('{\n  "b": 2\n}\n');
    expect(third).toBe(3n);
  }, 120_000);

  it('survives a refusal the core raises, and answers the next call', async () => {
    const { lang } = await ready();
    await expect(lang.parse('{')).rejects.toBeInstanceOf(LangFailure);
    expect(await lang.serialize(null)).toBe('null\n');
  }, 120_000);

  it('reports the stages a validation crosses as it crosses them', async () => {
    const { lang, library, tree, path } = await ready();
    const seen: Progress[] = [];
    const verdict = await lang.validate(tree as never, path, {
      library,
      onProgress: (progress) => seen.push(progress),
    });
    expect(seen.map((one) => one.stage)).toEqual(['schema', 'library', 'assignment', 'semantic']);
    expect(seen.every((one) => one.call === 'validate' && one.total === 4)).toBe(true);
    expect(seen.map((one) => one.done)).toEqual([0, 1, 2, 3]);
    expect(verdict.stagesRun).toEqual(seen.map((one) => one.stage));
  }, 120_000);
});

describe('cancellation', () => {
  it('never starts a call the caller gave up on before it ran', async () => {
    const { lang, library, tree, path } = await ready();
    const abort = new AbortController();
    const derivation = lang.derive(tree as never, path, { library, signal: abort.signal });
    // The session yields to the task queue before its heavy stage, which is what gives the cancel
    // message a turn to arrive: aborting in the same tick reaches the worker first.
    abort.abort();
    await expect(derivation).rejects.toBeInstanceOf(LangCancelled);
    await expect(derivation).rejects.toMatchObject({ reason: 'requested' });
  }, 120_000);

  it('refuses a call whose signal is already aborted, without posting it', async () => {
    const { lang } = await ready();
    const abort = new AbortController();
    abort.abort();
    await expect(
      lang.derive(null, 'x.json', {
        library: { kind: 'library', id: 'nothing' },
        signal: abort.signal,
      }),
    ).rejects.toBeInstanceOf(LangCancelled);
  }, 120_000);

  it('cancels the superseded derivation of a document and drops its result', async () => {
    const { lang, library, tree, path } = await ready();
    const older = lang.derive(tree as never, path, { library });
    const newer = lang.derive(tree as never, path, { library });

    await expect(older).rejects.toBeInstanceOf(LangCancelled);
    await expect(older).rejects.toMatchObject({ reason: 'superseded', call: 'derive' });
    // The newer one answers: one in-flight derivation per document, and it is the last asked for.
    const derived = await newer;
    expect(Object.keys(derived)).toContain('d6');
  }, 120_000);

  it('supersedes per document, not per session', async () => {
    const { lang } = connect();
    const schemas = await lang.loadSchemas(schemaFiles(), { origin: 'schemas' });
    const library = await lang.loadLibrary([referenceBase()], schemas.handle);
    const one = await lang.parse(corpus('llama3-8b'));
    const other = await lang.parse(corpus('shieldstral-3b'));
    const first = lang.derive(one, corpusPath('llama3-8b'), { library: library.handle });
    const second = lang.derive(other, corpusPath('shieldstral-3b'), { library: library.handle });
    const [a, b] = await Promise.all([first, second]);
    expect(Object.keys(a)).toContain('d6');
    expect(Object.keys(b)).toContain('d6');
  }, 120_000);

  it('answers a check during a drag ahead of a derivation that has not begun', async () => {
    // §5.6 gives `check` twenty milliseconds and a derivation two seconds. A strict
    // first-in-first-out would have made the first unmeetable behind the second; what makes it
    // meetable is that the derivation waits at its yield while the cheap call, which has none,
    // runs to the end.
    const { lang, library, tree, path } = await ready();
    await lang.describe(tree as never, path, { library, only: [] });
    const order: string[] = [];
    const derivation = lang
      .derive(tree as never, path, { library })
      .then(() => order.push('derive'));
    const verdict = lang
      .check(path, { location: { identity: 'wq[layer=0]', location: { tensor: 'x' } } })
      .then(() => order.push('check'));
    await Promise.all([derivation, verdict]);
    expect(order).toEqual(['check', 'derive']);
  }, 120_000);

  it('cancels everything outstanding when the proxy is closed', async () => {
    const { lang, library, tree, path } = await ready();
    const derivation = lang.derive(tree as never, path, { library });
    lang.close();
    await expect(derivation).rejects.toMatchObject({ reason: 'closed' });
  }, 120_000);
});

describe('a refusal the tools themselves raise', () => {
  // Feature 1.8e settled it and left this half to feature 1.11: a *public input's* byte size is
  // taken unguarded where a produced value's is guarded, so a port shape that does not resolve
  // raises `TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'` out of the whole
  // derivation. "`derive` neither catches it nor turns it into a blank — the products would be a
  // fiction either way — and the worker of feature 1.11 is what catches it."
  //
  // The decision taken here: the worker catches it and answers it as a **refusal**, keeping the
  // name the core raised under and the tools' own words. It does not become a `Problem` of the
  // validation — the document validates, and saying otherwise would be inventing a verdict the
  // tools do not give — and it does not blank the figure. The panel of §4.18 shows it through
  // `derivationRow`, which is the one place the core's wording becomes a row.

  async function scratched(): Promise<{ lang: Lang; library: LibraryHandle }> {
    const { lang } = connect();
    const schemas = await lang.loadSchemas(schemaFiles(), { origin: 'schemas' });
    const base = referenceBase();
    const library = await lang.loadLibrary(
      [base, { base: 'scratch/base', files: SCRATCH }],
      schemas.handle,
    );
    expect(library.problems).toEqual([]);
    return { lang, library: library.handle };
  }

  it('comes back as a refusal in Python’s own words, and does not kill the worker', async () => {
    const { lang, library } = await scratched();
    const tree = await lang.parse(BLANK_MODEL);
    const refusal = await lang
      .derive(tree, SCRATCH_MODEL_PATH, { library })
      .then(() => null)
      .catch((error: unknown) => error as LangFailure);
    expect(refusal).toBeInstanceOf(LangFailure);
    expect(refusal?.raised).toBe('PyTypeError');
    expect(refusal?.message).toBe("unsupported operand type(s) for *: 'NoneType' and 'int'");

    // The worker is still there, and the document the raise came from still validates: the
    // refusal is the derivation's, not the document's.
    const verdict = await lang.validate(tree, SCRATCH_MODEL_PATH, { library });
    expect(verdict.problems).toEqual([]);
    expect(verdict.stagesRun).toContain('semantic');
  }, 120_000);

  it('is the same refusal in this thread, which is what makes the worker’s answer faithful', async () => {
    const lang = createLang();
    const schemas = await lang.loadSchemas(schemaFiles(), { origin: 'schemas' });
    const library = await lang.loadLibrary(
      [referenceBase(), { base: 'scratch/base', files: SCRATCH }],
      schemas.handle,
    );
    const tree = await lang.parse(BLANK_MODEL);
    await expect(lang.derive(tree, SCRATCH_MODEL_PATH, { library: library.handle })).rejects.toThrow(
      "unsupported operand type(s) for *: 'NoneType' and 'int'",
    );
    lang.close();
  }, 120_000);
});
