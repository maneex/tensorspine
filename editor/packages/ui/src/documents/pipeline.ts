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
import type { JsonValue, PyRecord, SchemaRegistry } from '@tensorspine/lang';
import { derivationRow, LangCancelled, PROBLEM_SOURCE, schemaProblem } from '@tensorspine/lang/api';
import type {
  Lang,
  LibraryHandle,
  LintDocuments,
  Problem,
  Verdict,
} from '@tensorspine/lang/api';

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
  /**
   * The grammar's rows, for **this** revision, from Ajv on the interface's own thread.
   *
   * §5.4 puts the schema stage there — "Ajv (sync) ──► Problems (schema)", "< 5 ms on a corpus
   * document" — so a member the grammar refuses is a row before the keystroke has finished, and
   * not three hundred milliseconds later with everything else. The registry is the page's own
   * (feature 2.6 builds one beside the worker's for exactly this), and the rows are the same rows
   * the worker would answer: one implementation, two threads, and a suite that says so.
   *
   * `null` where the page has no registry at all — the stub platform CI builds against reads no
   * schemas (feature 2.4) — and there the verdict's own schema rows are the only ones there are.
   */
  readonly structural: readonly Problem[] | null;
  /** The core's verdict, or `null` before the first one answers. */
  readonly verdict: Verdict | null;
  /** The revision {@link verdict} was computed for — older than {@link revision} is stale. */
  readonly verdictAt: number;
  readonly derivation: DerivationState;
  /** The derived document, or `null` where none has been computed for this revision. */
  readonly derived: PyRecord | null;
  /** The revision the derived document was computed for — older than {@link revision} is stale. */
  readonly derivedAt: number;
  /** What a refusal of the derivation said, as the row §4.17 shows it. */
  readonly failure: Problem | null;
  /**
   * The lint stage's rows, and the revision they were run for.
   *
   * Not part of the debounced run, and that is this feature's decision: `--lint`'s answer is a
   * function of the **set** of documents it is given (feature 1.10), the only set under which the
   * repository's own corpus lints clean is the whole of it, and linting fourteen documents costs
   * 781 ms — one `analyse` per document — against §5.6's 300 ms for a keystroke's validation.
   * So `Model ▸ Lint` (§4.4) runs it over the workspace's documents, the rows stand until it is
   * run again, and they are marked stale as every other product is when the document moves.
   */
  readonly lint: { readonly revision: number; readonly problems: readonly Problem[] } | null;
  /**
   * The editor's own rows about this document — §4.17's `editor` source, at `notice`.
   *
   * Computed where the verdict is, and for the same revision: they read the document's text and
   * the editor's own events, never a rule of §6, so they cost an outline and a reference index
   * (2.7 measured both at a few milliseconds) and they are recomputed when the core answers
   * rather than on every keystroke.
   */
  readonly notices: readonly Problem[];
}

/** A reading of a document nothing has been asked about yet. */
export function noReading(revision: number): Reading {
  return {
    revision,
    checking: false,
    structural: null,
    verdict: null,
    verdictAt: -1,
    notices: [],
    derivation: 'waiting',
    derived: null,
    derivedAt: -1,
    failure: null,
    lint: null,
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
  /** The page's own registry, for §5.4's synchronous schema stage. */
  readonly registry?: SchemaRegistry | null;
  /** The role the document is read under — `model` for a `tensorspine/2.0` document. */
  readonly role?: string;
  /** The documents a lint run reads, asked for when one is asked for (feature 1.10). */
  readonly lintSet?: () => Promise<LintDocuments>;
  /** The editor's own rows about the document, asked for when the core has answered. */
  readonly notices?: (tree: JsonValue) => readonly Problem[];
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

  /**
   * `Model ▸ Lint` (§4.4): the same validation, with the lint stage, over the set the caller names.
   *
   * One call and not a second pipeline, because `--lint`'s own staging is `validate`'s: the
   * grammar, the library, the assignment, the meaning, then lint — and asking for lint alone
   * would be a second implementation of that order (D3). What changes is that the rows are kept
   * apart, with the revision they were computed for, so that they read as stale rather than
   * vanishing on the next keystroke.
   */
  lint(): void {
    void this.run(true);
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
    const { tree, revision } = this.options.read();
    // §5.4's first branch: Ajv on the interface's own thread, now, before anything is waited for.
    const structural = this.structural(tree);
    // The bar says "checking" from the gesture, not from the moment the debounce expires: the
    // figures beside it are already about an older document and must not read as current. A
    // document nothing has derived yet is not *stale* — it is undone, which is a different word.
    this.options.publish((before) => ({
      ...before,
      revision,
      checking: true,
      structural,
      derivation:
        before.derived === null || before.derivedAt === revision ? before.derivation : 'stale',
    }));
    this.timer = setTimeout(() => {
      this.timer = null;
      void this.run();
    }, delay);
  }

  /**
   * The grammar's rows for a tree, here and now.
   *
   * The page has a registry of its own — the same files the worker indexed (feature 2.6) — and
   * `conforms` is 0.93–2.80 ms on the corpus against §5.6's five. `structural` walks only after a
   * refusal, which is the reading feature 1.1 built and the reading the worker takes.
   */
  private structural(tree: JsonValue): readonly Problem[] | null {
    const registry = this.options.registry;
    if (registry === null || registry === undefined) return null;
    const role = this.options.role ?? MODEL_ROLE;
    if (registry.conforms(tree, role)) return [];
    return registry.structural(tree, role).map((one) => schemaProblem(one, this.options.path));
  }

  private async run(withLint = false): Promise<void> {
    this.running?.abort();
    const control = new AbortController();
    this.running = control;
    const { lang, library, path, publish } = this.options;
    const { tree, revision } = this.options.read();
    const mine = (): boolean => !this.stopped && !control.signal.aborted && this.options.read().revision === revision;
    const lintSet = withLint ? await this.options.lintSet?.() : undefined;
    try {
      const verdict = await lang.validate(tree, path, {
        library,
        revision,
        signal: control.signal,
        ...(lintSet === undefined ? {} : { lint: lintSet }),
      });
      if (!mine()) return;
      const notices = this.options.notices?.(tree) ?? [];
      publish((before) => ({
        ...before,
        revision,
        checking: false,
        notices,
        verdict: withoutStages(verdict, this.options.registry != null),
        verdictAt: revision,
        ...(lintSet === undefined
          ? {}
          : {
              lint: {
                revision,
                problems: verdict.problems.filter((one) => one.source === PROBLEM_SOURCE.lint),
              },
            }),
      }));
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
        // The core's own words, as the row §4.18 and §4.17 show: "the tools *print* them; the
        // core raises them, because a document it cannot vouch for is not returned".
        failure: derivationRow(
          error instanceof Error ? error : new Error(String(error)),
          path,
        ),
      }));
    } finally {
      if (this.running === control) this.running = null;
    }
  }
}

/**
 * The role a `tensorspine/2.0` document is read under.
 *
 * The registry's own word for the model schema's role, which is what `structural` takes; a caller
 * that holds a session passes the session's own, and this is what a pipeline built without one
 * falls back to.
 */
const MODEL_ROLE = 'model';

/**
 * The verdict without the rows another field of the reading carries.
 *
 * The **lint** rows are kept apart with the revision they were computed for
 * ({@link Reading.lint}), so that a keystroke that re-validates does not silently drop findings
 * the reader asked for — it marks them stale instead. The **schema** rows are dropped where the
 * page ran Ajv itself: they are the same rows from the same implementation, and the page's are
 * the ones that are current. Leaving either in would show it twice.
 */
function withoutStages(verdict: Verdict, ownSchema: boolean): Verdict {
  const dropped = (one: (typeof verdict.problems)[number]): boolean =>
    one.source === PROBLEM_SOURCE.lint || (ownSchema && one.source === PROBLEM_SOURCE.schema);
  if (!verdict.problems.some(dropped)) return verdict;
  return { ...verdict, problems: verdict.problems.filter((one) => !dropped(one)) };
}

/**
 * Every row a reading holds, in the order §4.17's table lists their sources.
 *
 * The grammar's first, because §5.4 runs it first and it is the only one that is always current;
 * then what the core's staged validation answered, then the derivation's refusal, then the lint
 * findings the reader asked for, then the editor's own notices. A row computed for an older
 * revision than the document's is `stale` — §5.4's freshness, "anything computed for an older
 * revision is shown dimmed".
 */
export function rowsOf(reading: Reading): { problem: Problem; stale: boolean }[] {
  const rows: { problem: Problem; stale: boolean }[] = [];
  for (const one of reading.structural ?? []) rows.push({ problem: one, stale: false });
  const behind = reading.verdictAt !== reading.revision;
  for (const one of reading.verdict?.problems ?? []) rows.push({ problem: one, stale: behind });
  if (reading.failure !== null) rows.push({ problem: reading.failure, stale: behind });
  if (reading.lint !== null) {
    const old = reading.lint.revision !== reading.revision;
    for (const one of reading.lint.problems) rows.push({ problem: one, stale: old });
  }
  for (const one of reading.notices) rows.push({ problem: one, stale: behind });
  return rows;
}

/** How many problems of each weight a reading carries — the bar's first pill (§4.2, §4.17). */
export function countsOf(reading: Reading | Verdict | null): { errors: number; warnings: number } {
  if (reading === null) return { errors: 0, warnings: 0 };
  const problems =
    'problems' in reading ? reading.problems : rowsOf(reading).map((one) => one.problem);
  let errors = 0;
  let warnings = 0;
  for (const problem of problems) {
    if (problem.severity === 'error') errors += 1;
    else if (problem.severity === 'warning') warnings += 1;
  }
  return { errors, warnings };
}
