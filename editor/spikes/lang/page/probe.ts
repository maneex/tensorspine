import { startLang } from '../../../apps/web/src/lang/connect.ts';
import { LangCancelled, type Lang, type LibraryHandle, type Progress } from '@tensorspine/lang/api';

/**
 * What the browser layer drives, and nothing more.
 *
 * The page computes nothing of its own: every answer below is the core's, reached through the
 * `Lang` interface over a real `Worker`. What each probe is *for* is written beside it, because
 * each answers a claim of §5.3 or §5.6 that one thread cannot answer.
 */

interface Loaded {
  readonly lang: Lang;
  readonly library: LibraryHandle;
  readonly documents: Record<string, string>;
}

interface Material {
  readonly schemas: Record<string, string>;
  readonly base: { base: string; files: Record<string, string> };
  readonly documents: Record<string, string>;
}

let session: Promise<Loaded> | null = null;

/** The schemas and the reference base, loaded into a worker-backed `Lang` once per page. */
function load(): Promise<Loaded> {
  session ??= (async () => {
    const material = (await (await fetch('./material.json')).json()) as Material;
    const lang = startLang();
    const schemas = await lang.loadSchemas(material.schemas, { origin: 'schemas' });
    const library = await lang.loadLibrary([material.base], schemas.handle);
    if (library.problems.length > 0) {
      throw new Error(`the reference base did not load: ${library.problems[0]?.message ?? ''}`);
    }
    return { lang, library: library.handle, documents: material.documents };
  })();
  return session;
}

/** What every probe answers: a verdict the test reads, and how long it took. */
export interface Outcome {
  readonly ok: boolean;
  readonly detail: Record<string, unknown>;
  readonly ms: number;
}

async function timed(run: () => Promise<Record<string, unknown>>): Promise<Outcome> {
  const start = performance.now();
  try {
    const detail = await run();
    return { ok: true, detail, ms: performance.now() - start };
  } catch (error) {
    return {
      ok: false,
      detail: { error: error instanceof Error ? `${error.name}: ${error.message}` : String(error) },
      ms: performance.now() - start,
    };
  }
}

const probe = {
  /** The round trip: a document opened, validated and derived in the worker. */
  pipeline: (name: string): Promise<Outcome> =>
    timed(async () => {
      const { lang, library, documents } = await load();
      const text = documents[name] as string;
      const path = `data/models/${name}.json`;
      const tree = await lang.parse(text);
      const stages: Progress[] = [];
      const verdict = await lang.validate(tree, path, {
        library,
        revision: 1,
        onProgress: (progress) => stages.push(progress),
      });
      const one = await lang.expand(tree, { library });
      const derived = await lang.derive(tree, path, { library, revision: 1 });
      const facts = await lang.describe(tree, path, { library, revision: 1, only: ['embed'] });
      return {
        // The bytes the editor would write back are the bytes the file holds (D12).
        roundTrips: (await lang.serialize(tree)) === text,
        stages: stages.map((stage) => stage.stage),
        problems: verdict.problems.length,
        stats: Object.fromEntries(
          [...verdict.stats].map(([key, value]) => [key, String(value as bigint | number)]),
        ),
        nodes: Object.keys((one['d1'] as Record<string, unknown>)['nodes'] as object).length,
        products: Object.keys(derived),
        sites: [...facts.sites.keys()],
      };
    }),

  /**
   * Whether the worker's thread is really another thread.
   *
   * A derivation of the largest corpus document is 230 ms of synchronous work. If it ran on the
   * page's thread, a `requestAnimationFrame` loop would stop for that long; in a worker it does
   * not miss a frame. That is §5.3's "validation and derivation never block the UI", asked of a
   * browser rather than asserted.
   */
  nonBlocking: (name: string): Promise<Outcome> =>
    timed(async () => {
      const { lang, library, documents } = await load();
      const tree = await lang.parse(documents[name] as string);
      const path = `data/models/${name}.json`;
      let frames = 0;
      let running = true;
      const tick = (): void => {
        frames += 1;
        if (running) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
      const start = performance.now();
      await lang.derive(tree, path, { library });
      const elapsed = performance.now() - start;
      running = false;
      return { frames, elapsed, framesPerSecond: (frames / elapsed) * 1000 };
    }),

  /**
   * One in-flight derivation per document (§5.3): a second call cancels the first.
   *
   * Across a real thread, which is where it matters: the cancel is a message that has to reach a
   * worker already at work and overtake it.
   */
  superseded: (name: string): Promise<Outcome> =>
    timed(async () => {
      const { lang, library, documents } = await load();
      const tree = await lang.parse(documents[name] as string);
      const path = `data/models/${name}.json`;
      const older = lang.derive(tree, path, { library });
      const newer = lang.derive(tree, path, { library });
      const first = await older.then(
        () => ({ cancelled: false, reason: null as string | null }),
        (error: unknown) => ({
          cancelled: error instanceof LangCancelled,
          reason: error instanceof LangCancelled ? error.reason : String(error),
        }),
      );
      const second = await newer;
      return { first, second: Object.keys(second) };
    }),

  /** A caller's `AbortSignal`, across the thread. */
  aborted: (name: string): Promise<Outcome> =>
    timed(async () => {
      const { lang, library, documents } = await load();
      const tree = await lang.parse(documents[name] as string);
      const path = `data/models/${name}.json`;
      const abort = new AbortController();
      const derivation = lang.derive(tree, path, { library, signal: abort.signal });
      abort.abort();
      return await derivation.then(
        () => ({ cancelled: false, reason: null as string | null }),
        (error: unknown) => ({
          cancelled: error instanceof LangCancelled,
          reason: error instanceof LangCancelled ? error.reason : String(error),
        }),
      );
    }),

  /** A refusal the core raises comes back as a refusal, and the worker answers the next call. */
  refusal: (): Promise<Outcome> =>
    timed(async () => {
      const { lang } = await load();
      const refused = await lang.parse('{').then(
        () => null,
        (error: unknown) => (error instanceof Error ? `${error.name}: ${error.message}` : null),
      );
      return { refused, after: await lang.serialize(null) };
    }),
};

declare global {
  interface Window {
    langProbe: typeof probe;
  }
}

window.langProbe = probe;
const out = document.querySelector('#out');
if (out instanceof HTMLElement) {
  out.textContent = 'ready';
  out.dataset['state'] = 'ready';
}
