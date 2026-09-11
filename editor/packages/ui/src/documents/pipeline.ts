/**
 * §5.4's pipeline for one open document — feature 2.6.
 *
 * ```
 * gesture ──► command (patches) ──► document tree ──┬──► Ajv (sync) ──► Problems (schema)
 *                                                   └──► core.validate (300 ms) ──► Problems
 *                                                             └─ on success ──► core.derive
 * ```
 *
 * What this module owns is the *running* of it: the debounce, the cancellation of a reading a
 * newer revision superseded, and the freshness every product is measured against. What it does
 * not own is any verdict: `validate` and `derive` are the core's, and the figures they produce
 * are read, never recomputed (the component inventory's §7).
 *
 * **The revision is what opts a call into reuse.** `CallOptions.revision` is the store's own
 * counter, and the session behind the worker keeps the analysis of one `(path, revision)` so that
 * a validation and the derivation after it pay for `analyse` once. The caller promises that one
 * revision names one tree, which is exactly what `DocumentStore.revision` is.
 *
 * **Nothing is published for a revision that has moved.** Every answer is checked against the
 * revision it was asked for before it is written back — the lesson of the review repair `4540ff4`
 * one level up, and the same reason: a figure computed for a document the editor has moved on
 * from is worse than no figure, because it looks fresh.
 */
import type { JsonValue, PyRecord } from '@tensorspine/lang';
import { LangCancelled } from '@tensorspine/lang/api';
import type { Lang, LibraryHandle, Verdict } from '@tensorspine/lang/api';

/** How long a keystroke waits before the semantic stage runs (§5.6: "debounced at 300 ms"). */
export const VALIDATE_DEBOUNCE_MS = 300;

/** Where a derivation stands, for the bar's second pill and the status bar's fourth field. */
export type DerivationState = 'waiting' | 'running' | 'fresh' | 'stale' | 'failed' | 'skipped';

/** What the pipeline has to say about a document, at the revision it says it for. */
export interface Reading {
  /** The document revision every field below was computed for (§5.4's freshness). */
  readonly revision: number;
  /** True while the semantic stage is running or waiting for its debounce. */
  readonly checking: boolean;
  /** The core's verdict, or `null` before the first one answers. */
  readonly verdict: Verdict | null;
  readonly derivation: DerivationState;
  /** The derived document, or `null` where none has been computed for this revision. */
  readonly derived: PyRecord | null;
  /** The revision the derived document was computed for — older than {@link revision} is stale. */
  readonly derivedAt: number;
  /** What a refusal of the derivation said, in the core's words. */
  readonly failure: string | null;
}

/** A reading of a document nothing has been asked about yet. */
export function noReading(revision: number): Reading {
  return {
    revision,
    checking: false,
    verdict: null,
    derivation: 'waiting',
    derived: null,
    derivedAt: -1,
    failure: null,
  };
}

/** What the pipeline is given for one document. */
export interface PipelineOptions {
  readonly lang: Lang;
  readonly library: LibraryHandle;
  readonly path: string;
  /** The tree and its revision, read at the moment the run starts. */
  readonly read: () => { tree: JsonValue; revision: number };
  /** Called whenever the reading changed — the store's `set`. */
  readonly publish: (reading: (before: Reading) => Reading) => void;
  /** How long to wait before the semantic stage; zero in a suite that wants it now. */
  readonly debounceMs?: number;
}

/**
 * The pipeline of one open document: one validation and one derivation in flight, both dropped
 * when the document moves under them.
 */
export class Pipeline {
  private timer: ReturnType<typeof setTimeout> | null = null;
  private running: AbortController | null = null;
  private stopped = false;

  constructor(private readonly options: PipelineOptions) {}

  /** Run now: `Validate Now` (F7), and what an open does. */
  now(): void {
    this.schedule(0);
  }

  /** Run after the debounce: what an edit does (§5.4). */
  soon(): void {
    this.schedule(this.options.debounceMs ?? VALIDATE_DEBOUNCE_MS);
  }

  /** Stop: a closed tab, a workspace replaced. */
  stop(): void {
    this.stopped = true;
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = null;
    this.running?.abort();
    this.running = null;
  }

  private schedule(delay: number): void {
    if (this.stopped) return;
    if (this.timer !== null) clearTimeout(this.timer);
    const { revision } = this.options.read();
    // The bar says "checking" from the gesture, not from the moment the debounce expires: the
    // figures beside it are already about an older document and must not read as current. A
    // document nothing has derived yet is not *stale* — it is undone, which is a different word.
    this.options.publish((before) => ({
      ...before,
      revision,
      checking: true,
      derivation:
        before.derived === null || before.derivedAt === revision ? before.derivation : 'stale',
    }));
    this.timer = setTimeout(() => {
      this.timer = null;
      void this.run();
    }, delay);
  }

  private async run(): Promise<void> {
    this.running?.abort();
    const control = new AbortController();
    this.running = control;
    const { lang, library, path, publish } = this.options;
    const { tree, revision } = this.options.read();
    const mine = (): boolean => !this.stopped && !control.signal.aborted && this.options.read().revision === revision;
    try {
      const verdict = await lang.validate(tree, path, {
        library,
        revision,
        signal: control.signal,
      });
      if (!mine()) return;
      publish((before) => ({ ...before, revision, checking: false, verdict }));
      if (verdict.problems.some((one) => one.severity === 'error')) {
        publish((before) => ({ ...before, derivation: 'skipped', failure: null }));
        return;
      }
      if (verdict.needsAssignment !== undefined) {
        publish((before) => ({ ...before, derivation: 'skipped', failure: null }));
        return;
      }
      publish((before) => ({ ...before, derivation: 'running' }));
      const derived = await lang.derive(tree, path, { library, revision, signal: control.signal });
      if (!mine()) return;
      publish((before) => ({
        ...before,
        derived,
        derivedAt: revision,
        derivation: 'fresh',
        failure: null,
      }));
    } catch (error) {
      // A cancelled call is not a failure: a newer revision replaced it, or the tab closed.
      if (error instanceof LangCancelled) return;
      if (!mine()) return;
      publish((before) => ({
        ...before,
        checking: false,
        derivation: 'failed',
        failure: error instanceof Error ? error.message : String(error),
      }));
    } finally {
      if (this.running === control) this.running = null;
    }
  }
}

/** How many problems of each weight a verdict carries — the bar's first pill (§4.2). */
export function countsOf(verdict: Verdict | null): { errors: number; warnings: number } {
  if (verdict === null) return { errors: 0, warnings: 0 };
  let errors = 0;
  let warnings = 0;
  for (const problem of verdict.problems) {
    if (problem.severity === 'error') errors += 1;
    else if (problem.severity === 'warning') warnings += 1;
  }
  return { errors, warnings };
}
