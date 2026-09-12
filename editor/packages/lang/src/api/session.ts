/**
 * The session: one implementation of every call of the `Lang` API, driven twice.
 *
 * The in-process `Lang` (`core.ts`) calls it directly and the worker's host (`host.ts`) calls it
 * from messages, so there is one body per call and the two deployments cannot drift. What the
 * session adds to the core is only what a *session* has and a pure function has not:
 *
 * - **What cannot cross the boundary is held here.** A `SchemaRegistry` answers through methods
 *   and carries compiled Ajv validators; a gathered library is read through a `LibrarySource`.
 *   Functions are a `DataCloneError`, so both are held and named by a handle (`api/types.ts`).
 * - **The analysis of one revision is kept.** "The pipeline of §5.4 pays it once and
 *   `describeAnalysis` is what lets it" (feature 1.6d): `validate`, `describe` and `check` on one
 *   `(path, revision)` read one `analyse`, which costs 26–82 ms on the corpus. The caller opts in
 *   by passing the store's revision counter; without one nothing is reused.
 * - **A heavy stage yields before it runs.** The core is synchronous — once `derive` is entered
 *   it runs to the end — so the one moment a cancellation can be noticed is *between* stages. The
 *   session yields to the task queue there, which is what lets a cancel message be delivered, what
 *   lets a superseded derivation be dropped before it starts, and what lets a `check` during a
 *   drag answer ahead of a derivation that has not begun.
 * - **One in-flight derivation per document** (§5.3). A second `derive` for a path cancels the
 *   first; the first's result is dropped even if it had already been computed.
 *
 * Everything else is the core's, unchanged: no rule of the language is decided here, and every
 * refusal keeps the wording its tool gives it.
 */
import { checkCheckpoint as checkCheckpointOf } from '../artifact/check.js';
import {
  readHeader as readHeaderOf,
  type CheckpointHeaders,
  type HeaderRead,
} from '../artifact/header.js';
import { expand as expandOf } from '../d1/expand.js';
import {
  checkCandidate,
  describeAnalysis,
  identityFacts,
  type Candidate,
  type IdentityRequest,
  type SiteDescription,
  type Verdict as CandidateVerdict,
} from '../describe/index.js';
import { derive as deriveOf } from '../derive/products.js';
import { modelValue, type Env, type Quantities } from '../expr/model.js';
import { toPython, type PyRecord, type PyValue } from '../expr/value.js';
import { put, type JsonValue } from '../json/tree.js';
import { parse as parseOf } from '../json/parse.js';
import { serialize as serializeOf } from '../json/serialize.js';
import {
  baseTemplates as baseTemplatesOf,
  basesOf,
  loadLibrary as loadLibraryOf,
  missingBases,
  type Library,
  type LibraryContext,
} from '../library/load.js';
import { dirname, normalise } from '../library/paths.js';
import { hoistingOf, type Hoisting } from '../model/index.js';
import { pyRepr } from '../library/repr.js';
import { memorySource, type LibrarySource } from '../library/source.js';
import { validateUnit as validateUnitOf, type UnitLocation } from '../library/unit.js';
import { distinctFindings, lint as lintOf, type LintDocument } from '../lint/lint.js';
import {
  loadSchemas as loadSchemasOf,
  type SchemaRegistry,
  type StructuralProblem,
} from '../schema/registry.js';
import { analyse, type Analysis } from '../validate/bindings/index.js';
import { assignmentNeeded, checkAssignment } from '../validate/quantities.js';

import { checkpointRow, libraryRow, lintRow, schemaProblem, semanticRow } from './problems.js';
import {
  LangCancelled,
  LangHandleError,
  type CancelReason,
  type CheckpointReport,
  type DocumentBases,
  type Facts,
  type LibraryBaseFiles,
  type LibraryHandle,
  type LibraryLoaded,
  type Problem,
  type Progress,
  type SchemasHandle,
  type SchemasLoaded,
  type Stage,
  type Verdict,
} from './types.js';

/**
 * What a call is given to watch and to report through.
 *
 * It is not part of the transport: a function cannot cross the boundary, so the host builds one
 * per request from the messages it receives and the in-process proxy builds one from the caller's
 * `AbortSignal` and `onProgress`.
 */
export interface Control {
  /** The refusal to raise where the caller has given up on this call, else `null`. */
  cancelled(): LangCancelled | null;
  /** Report a stage boundary. */
  report(progress: Progress): void;
}

/** A control that never cancels and reports nowhere: what a call without options is given. */
export const UNWATCHED: Control = {
  cancelled: () => null,
  report: () => undefined,
};

/**
 * Yield to the task queue.
 *
 * A *macro*task, not a microtask: a `message` event is delivered as a task, so a microtask
 * checkpoint would run the whole call before a cancel could arrive. This is the only place the
 * session gives the platform a turn, and it is what makes a cancellation observable at all.
 */
export function yieldToTasks(): Promise<void> {
  return new Promise<void>((resolve) => {
    setTimeout(resolve, 0);
  });
}

/** The arguments every call that reads a document takes, once the functions are stripped off. */
export interface DocumentOptions {
  readonly library: LibraryHandle;
  readonly assignment?: PyRecord | undefined;
  readonly revision?: number | undefined;
}

/** {@link DocumentOptions} with the sites a caller wants described. */
export interface DescribeArguments extends DocumentOptions {
  readonly only?: readonly string[] | undefined;
  /** One site per declared instance: the folded canvas of §4.7 (`DescribeOptions.folded`). */
  readonly folded?: boolean | undefined;
  /** Whether to fill the compatibility lists (`DescribeOptions.compatibility`); default true. */
  readonly compatibility?: boolean | undefined;
  /** The identity §4.14's sheet is open on (`DescribeOptions.identity`). */
  readonly identity?: IdentityRequest | undefined;
}

/** {@link DocumentOptions} with the set the lint stage reads (feature 1.10: the answer is the set's). */
export interface ValidateArguments extends DocumentOptions {
  readonly lint?: readonly LintDocument[] | undefined;
}

/**
 * The one member of a document a hoisted place can sit under: `normalise` writes every scoped
 * rule into `bindings`, and nowhere else.
 *
 * It is the guard that keeps the map off the hot path, not a reading of the rule — `writtenPlace`
 * answers `null` for every pointer the hoist did not write, so a row that skips the question is a
 * row the map would have left alone.
 */
const HOISTED_UNDER = '/bindings/';

/** The schemas, held because they cannot be sent. */
interface HeldSchemas {
  readonly handle: SchemasHandle;
  readonly registry: SchemaRegistry;
}

/**
 * One base of a gathered library: the path the caller wrote, and the path the session compares.
 *
 * `data/primitive-library` and `data/primitive-library/` name one directory. The loader reads the
 * same files under either — `memorySource` normalises every key and every lookup, and `placeOf`
 * normalises both of its arguments — so a containment test that compared the *text* would answer
 * differently for two spellings of one base, and the unit would be refused for the caller's
 * punctuation. The normalised form is what containment is decided on; the caller's own spelling is
 * what a refusal names, because that is the path the caller holds and will go looking for.
 */
interface HeldBase {
  /** As the caller wrote it: what the loader was given, and what a diagnostic names. */
  readonly path: string;
  /** `normalise` of it: what containment is decided on. */
  readonly root: string;
}

/** A gathered library, held with what gathered it. */
interface HeldLibrary {
  readonly handle: LibraryHandle;
  readonly schemas: HeldSchemas;
  readonly bases: readonly HeldBase[];
  readonly library: Library;
  readonly context: LibraryContext;
  /**
   * The refusals the *gather* carried, beside the ones the loaded library carries.
   *
   * A base that is not there is one: `load_for` raises it before the loader reads a file, so the
   * gathered library is empty and carries no refusal of its own — and a `validate` against it
   * would otherwise report every primitive as absent with nothing saying why. The tools load the
   * library **from the document** on every run and print that refusal as the document's, so
   * reporting it at every `validate` is their reading, not an invention; the API gathers once a
   * session (§5.6's 300 ms) and this is what keeps the two the same.
   */
  readonly refusals: readonly Problem[];
}

/** What one document's last reading left behind, for the next call on the same revision. */
interface HeldReading {
  /**
   * Which *installation* this is: a number nothing else in the session ever carries again.
   *
   * `validate` yields to the task queue between its grammar stage and its semantic one — that is
   * the whole reason a cancel can be delivered at all — and a keystroke's `describe`, a second
   * `validate` or a `forget` lands in that gap. The call that resumes therefore has to ask whether
   * the reading it installed is still the reading the session holds, and no field of the reading
   * itself can answer that: a revision can be *reused* — a document closed and reopened counts
   * from 1 again — and two calls can legitimately share one. So the installation is numbered, and
   * publishing an analysis is allowed only under the number it was read for. A completed copy
   * keeps the number, which is what lets two calls of one revision share the analysis rather than
   * each computing it (feature 1.6d).
   */
  readonly generation: number;
  /** The revision the caller named, or `-1` where it named none. */
  readonly revision: number;
  /** The library handle it was read against, so a different one is not answered from this. */
  readonly library: string;
  /** The assignment it was read under, as `pyRepr` writes it. */
  readonly assignment: string;
  readonly conforms: boolean;
  readonly structural: readonly StructuralProblem[];
  /** `null` where the grammar refused the document, or where nothing has analysed it yet. */
  readonly analysis: Analysis | null;
}

/** The session: one per worker, one per in-process proxy. */
export class LangSession {
  private next = 0;
  /** The counter behind {@link HeldReading.generation}: it only ever goes up. */
  private generations = 0;
  private readonly schemas = new Map<string, HeldSchemas>();
  private readonly libraries = new Map<string, HeldLibrary>();
  private readonly readings = new Map<string, HeldReading>();
  /**
   * The derivation in flight for each document: "one in-flight derivation per document" (§5.3).
   *
   * The reason travels with the cancel, because the three ways a derivation ends are three
   * different things to tell the caller: a second `derive` for the path *superseded* it, a
   * `forget` dropped the document under it, and a `clear` — the proxy closing — *closed* it.
   */
  private readonly deriving = new Map<string, { cancel: (reason: CancelReason) => void }>();

  // --- the vocabulary the interface reads --------------------------------------

  /** `loadSchemas(files)`: the registry by `$id`, held, with what the forms and the audits read. */
  loadSchemas(
    files: Readonly<Record<string, string>>,
    options: { readonly origin?: string } = {},
  ): SchemasLoaded {
    const registry = loadSchemasOf(
      Object.entries(files).map(([path, text]) => ({ path, text })),
      options.origin === undefined ? {} : { origin: options.origin },
    );
    const handle: SchemasHandle = { kind: 'schemas', id: this.identity('schemas') };
    this.schemas.set(handle.id, { handle, registry });
    return {
      handle,
      origin: registry.origin,
      schemas: registry.schemas.map((one) => ({ path: one.path, id: one.id, role: one.role })),
    };
  }

  /**
   * `loadLibrary(bases)`: the bases gathered, held, and answered with every refusal they carry.
   *
   * The bases arrive as their files because the core opens nothing (§5.3: "the core is pure —
   * the UI reads files through `Platform` and hands the core texts and trees"), so the source
   * behind the loader is the one built from those bytes.
   */
  loadLibrary(
    bases: readonly LibraryBaseFiles[],
    schemas: SchemasHandle,
    options: { readonly modelsBase?: string | null; readonly forDocument?: string } = {},
  ): LibraryLoaded {
    const held = this.heldSchemas(schemas);
    const files: Record<string, string> = {};
    for (const base of bases) {
      for (const [path, text] of Object.entries(base.files)) put(files, path, text);
    }
    const source: LibrarySource = memorySource(files);
    const context: LibraryContext = {
      schemas: held.registry,
      source,
      ...(options.modelsBase === undefined ? {} : { modelsBase: options.modelsBase }),
    };
    const paths = bases.map((base) => base.base);
    // `load_for`'s own guard, which `load` assumes has run: the tools open what they are given
    // and raise what `open()` raises, because a base that does not exist has already been
    // rejected (V1) one call up. The caller here gathers its own bases out of a workspace, so the
    // guard is made here — with the core's function, in the core's words — rather than left to a
    // `LibrarySourceError` escaping out of a read.
    const absent = missingBases(options.forDocument ?? '', paths, context);
    if (absent.length > 0) {
      const handle: LibraryHandle = { kind: 'library', id: this.identity('library') };
      const empty = loadLibraryOf([], context);
      const refusals = absent.map((one) => libraryRow(one));
      this.libraries.set(handle.id, {
        handle,
        schemas: held,
        bases: paths.map((path) => ({ path, root: normalise(path) })),
        library: empty,
        context,
        refusals,
      });
      return { handle, library: empty, problems: refusals };
    }
    // The loader is given the caller's own spellings, which is what its refusals name.
    const library = loadLibraryOf(paths, context);
    const handle: LibraryHandle = { kind: 'library', id: this.identity('library') };
    const heldBases = paths.map((path) => ({ path, root: normalise(path) }));
    this.libraries.set(handle.id, {
      handle,
      schemas: held,
      bases: heldBases,
      library,
      context,
      refusals: [],
    });
    return { handle, library, problems: library.problems.map((one) => libraryRow(one)) };
  }

  /**
   * `bases_of`: the bases a document declares, resolved against the document's own directory.
   *
   * The call the workspace makes **before** it can read anything for a document: it has the
   * document's text and has to know which folders to gather. The answer is the loader's own, so
   * that no part of the interface reads `primitive_libraries` for itself (D3, §1).
   */
  documentBases(tree: JsonValue, path: string): DocumentBases {
    const resolved = basesOf(path, recordOf(tree));
    return {
      bases: resolved.bases,
      problems: resolved.problem === null ? [] : [libraryRow(resolved.problem)],
    };
  }

  /**
   * Where each base keeps its template documents, read from the manifest each one carries.
   *
   * The second half of the same question, asked once the bases' own files are in hand. A base's
   * templates location is resolved against the base and **may leave it** — the reference base
   * says `"templates": "../models/"` — so a base handed to {@link loadLibrary} with its own files
   * alone is refused for a reason the workspace, not the base, is responsible for. This is the
   * computation `loadLibrary` does, without the load: one manifest read per base.
   */
  baseTemplates(
    bases: readonly LibraryBaseFiles[],
    schemas: SchemasHandle,
  ): (string | null)[] {
    const held = this.heldSchemas(schemas);
    const files: Record<string, string> = {};
    for (const base of bases) {
      for (const [path, text] of Object.entries(base.files)) put(files, path, text);
    }
    const context: LibraryContext = { schemas: held.registry, source: memorySource(files) };
    return bases.map((base) => baseTemplatesOf(base.base, context));
  }

  /** `validateUnit(unit, path, bases)`: the loader's verdict on one unit, live (D15, §4.22). */
  validateUnit(unit: JsonValue, path: string, library: LibraryHandle): Problem[] {
    const held = this.heldLibrary(library);
    const where: UnitLocation = { path, base: baseOf(held.bases, path) };
    return validateUnitOf(unit, where, held.library, held.context).map((one) => libraryRow(one));
  }

  // --- the document --------------------------------------------------------------

  /** `parse(text)`: the ordered tree with its number lexemes (D12). */
  parse(text: string): JsonValue {
    return parseOf(text);
  }

  /** `serialize(tree)`: the corpus's bytes (D12). */
  serialize(tree: JsonValue): string {
    return serializeOf(tree);
  }

  /**
   * `validate(tree, path, assignment?)`: the stages `--validate` runs, in its order.
   *
   * `run`'s own staging, reproduced: the grammar, and nothing after it if it refused ("meaning
   * assumes grammar"); the library the document resolves against; the assignment, which *skips*
   * the semantic stage rather than failing it when an external quantity has no value ("a template
   * with no assignment is skipped, not failed: it is a family of graphs, and refusing it would
   * report a defect where there is none"); then `analyse`. What is added is plan §3's
   * continuation — a library refusal does not stop the walk here, and every row after it is
   * marked `afterRefusal` — and the lint stage, over the set the caller names.
   */
  async validate(
    tree: JsonValue,
    path: string,
    options: ValidateArguments,
    control: Control = UNWATCHED,
  ): Promise<Verdict> {
    const held = this.heldLibrary(options.library);
    const stagesRun: Stage[] = [];
    const problems: Problem[] = [];
    const empty: ReadonlyMap<string, PyValue> = new Map<string, PyValue>();
    const total = options.lint === undefined ? 4 : 5;
    const stage = (name: Stage): void => {
      control.report({ call: 'validate', stage: name, done: stagesRun.length, total });
      stagesRun.push(name);
    };

    stage('schema');
    const reading = this.read(tree, path, options, held);
    watch(control);
    if (!reading.conforms) {
      return {
        stagesRun,
        problems: reading.structural.map((one) => schemaProblem(one, path)),
        stats: empty,
      };
    }

    const document = recordOf(tree);

    stage('library');
    const refused = held.library.problems.length > 0 || held.refusals.length > 0;
    for (const one of held.refusals) problems.push(one);
    for (const one of held.library.problems) problems.push(libraryRow(one));
    for (const one of basesMissing(held, document, path)) problems.push(one);

    stage('assignment');
    const needs = assignmentNeeded(document, options.assignment);
    if (needs.unset.length > 0) {
      return { stagesRun, problems, stats: empty, needsAssignment: needs };
    }
    const wrong = checkAssignment(document, options.assignment);
    if (wrong.length > 0) {
      for (const one of wrong) problems.push(semanticRow(one, { file: path, afterRefusal: refused }));
      return { stagesRun, problems, stats: empty };
    }

    stage('semantic');
    await yieldToTasks();
    watch(control);
    const analysis = this.analyse(document, path, options, held, reading);
    // §5.2 rule 7, read backwards: the stage's own pointers name the *normalised* document, and a
    // panel navigates in the one the editor holds. The hoist's record is asked for once, and only
    // where a row could name a hoisted place at all — a document with no scoped binding pays
    // nothing, and neither does one whose refusals are all about quantities or interfaces.
    let record: Hoisting | null = null;
    const mapping = (): Hoisting => (record ??= hoistingOf(document));
    for (const one of analysis.problems) {
      problems.push(
        semanticRow(one, {
          file: path,
          afterRefusal: refused,
          ...(one.path.startsWith(HOISTED_UNDER) ? { hoisting: mapping() } : {}),
        }),
      );
    }

    if (options.lint !== undefined) {
      stage('lint');
      await yieldToTasks();
      watch(control);
      const findings = lintOf(options.lint, {
        schemas: held.schemas.registry,
        library: held.library,
      });
      for (const one of distinctFindings(findings)) problems.push(lintRow(one));
    }

    return { stagesRun, problems, stats: analysis.stats };
  }

  /**
   * `describe(tree, path, assignment?)`: the facts the sheets, the cards and the chips show.
   *
   * The analysis it reads stays here: the whole `Description` of the largest corpus document
   * costs 45 ms to clone and its sites alone 25 (`editor/spikes/timings.md`), so what travels is
   * what §5.3 lists and nothing else — and `check` reads the analysis that stayed.
   *
   * Three gates before it, not one. Feature 1.6d's — the grammar, because "meaning assumes
   * grammar" and the editor calls this on every keystroke — and the assignment's two, because
   * every call site of the tools checks `missing_assignment` before it analyses and a template
   * read without one raises `Unassigned` out of the expansion of its own index ranges, and
   * because a value the document's own declaration refuses raises out of the same expansion:
   * `layers = 1.5` reaches `range` as `'float' object cannot be interpreted as an integer`.
   *
   * All three are {@link validate}'s own gates, in its order, and none is a verdict this invents:
   * they are reported, as {@link Facts.structural}, {@link Facts.needsAssignment} and
   * {@link Facts.assignmentRefused}. The last matters because the assignment sheet is edited while
   * this is called: what a half-typed value must produce is a row under the field, not an
   * exception out of the graph.
   */
  describe(
    tree: JsonValue,
    path: string,
    options: DescribeArguments,
    control: Control = UNWATCHED,
  ): Facts {
    const held = this.heldLibrary(options.library);
    const reading = this.read(tree, path, options, held);
    watch(control);
    if (!reading.conforms) {
      return {
        conforms: false,
        structural: reading.structural.map((one) => schemaProblem(one, path)),
        sites: new Map(),
      };
    }
    const document = recordOf(tree);
    // The gate every call site of the tools takes before it analyses (§4.6, I7): a template with
    // no assignment is skipped, not failed, and the skip is reported rather than raised.
    const needs = assignmentNeeded(document, options.assignment);
    if (needs.unset.length > 0) {
      return { conforms: true, structural: [], sites: new Map(), needsAssignment: needs };
    }
    // And the gate `validate` takes next, with the same call and the same wording: `analyse`
    // expands the document's index ranges with the assigned values, so a value the declaration
    // refuses is a refusal here and not an exception out of `range`.
    const wrong = checkAssignment(document, options.assignment);
    if (wrong.length > 0) {
      return {
        conforms: true,
        structural: [],
        sites: new Map(),
        assignmentRefused: wrong.map((one) => semanticRow(one, { file: path })),
      };
    }
    const analysis = this.analyse(document, path, options, held, reading);
    const description = describeAnalysis(analysis, {
      ...(options.only === undefined ? {} : { only: options.only }),
      ...(options.folded === undefined ? {} : { folded: options.folded }),
      ...(options.compatibility === undefined ? {} : { compatibility: options.compatibility }),
    });
    const sites = new Map<string, SiteDescription>();
    for (const site of description.sites.values()) sites.set(site.where, site);
    return {
      conforms: true,
      structural: [],
      sites,
      // The second question of one reading (§4.14): the identity the sheet has open, answered
      // from the analysis the sites were described from.
      ...(options.identity === undefined
        ? {}
        : { identity: identityFacts(analysis, held.library, options.identity) }),
    };
  }

  /**
   * `check(path, candidate)`: the verdict on one candidate edit, over the reading already held.
   *
   * §5.3 writes `check(tree, candidate)`; feature 1.6d measured why it cannot be: `analyse` alone
   * is 10–93 ms on the corpus and a drag asks per hovered handle, so the verdict reads the
   * analysis the session holds for that document. The caller therefore calls `describe` or
   * `validate` on it first, which is what §5.4's pipeline does anyway.
   */
  check(path: string, candidate: Candidate): CandidateVerdict {
    const reading = this.readings.get(path);
    if (reading === undefined) {
      throw new LangHandleError(
        `no reading is held for ${path}: call describe or validate on it first`,
      );
    }
    if (!reading.conforms) {
      return {
        ok: false,
        problems: [],
        unknown: 'the document is off the grammar, so nothing about a candidate edit is decidable',
      };
    }
    if (reading.analysis === null) {
      // On the grammar and never analysed: the assignment stage is the only thing that stops a
      // reading there — an external quantity with no value, or one the document's own domain
      // refuses (§4.6). Nothing about the graph is decided until it has one.
      return {
        ok: false,
        problems: [],
        unknown:
          'the document has not been analysed: its assignment is incomplete or refused (§4.6), ' +
          'so nothing about a candidate edit is decidable',
      };
    }
    return checkCandidate(reading.analysis, candidate);
  }

  /** `expand(tree, assignment?)`: D1, the graph `--d1` writes. */
  async expand(
    tree: JsonValue,
    options: DocumentOptions,
    control: Control = UNWATCHED,
  ): Promise<PyRecord> {
    const held = this.heldLibrary(options.library);
    control.report({ call: 'expand', stage: 'expansion', done: 0, total: 1 });
    await yieldToTasks();
    watch(control);
    return expandOf(toPython(tree), held.library, {
      ...(options.assignment === undefined ? {} : { assignment: options.assignment }),
    });
  }

  /**
   * `derive(tree, assignment?)`: the derived document, validated against the derived schema.
   *
   * **One stage, and that is a decision.** The tools' derivation is one function whose order —
   * D3, D4, D2, D5, then D1 and the agreement, then D6 — is feature 1.8e's, and sequencing it
   * here to report a finer progress would be a second implementation of that order (D3). So the
   * progress is one stage, the cancellation is checked before it and after it, and the budget it
   * is held to is §5.6's 2 s: measured 58–284 ms over the corpus, in a worker, never on the
   * interface's thread.
   *
   * **One in-flight derivation per document** (§5.3): a second call for the same path cancels the
   * first, whose result is dropped whether or not it had already been computed.
   */
  async derive(
    tree: JsonValue,
    path: string,
    options: DocumentOptions,
    control: Control = UNWATCHED,
  ): Promise<PyRecord> {
    const held = this.heldLibrary(options.library);
    const state: { cancelled: LangCancelled | null } = { cancelled: null };
    this.deriving.get(path)?.cancel('superseded');
    const entry = {
      cancel: (reason: CancelReason): void => {
        state.cancelled = new LangCancelled('derive', reason);
      },
    };
    this.deriving.set(path, entry);
    const guard = (): void => {
      if (state.cancelled !== null) throw state.cancelled;
      watch(control);
    };
    try {
      control.report({ call: 'derive', stage: 'products', done: 0, total: 1 });
      await yieldToTasks();
      guard();
      const answer = deriveOf(tree, {
        schemas: held.schemas.registry,
        library: held.library,
        ...(options.assignment === undefined ? {} : { assignment: options.assignment }),
      });
      // The result of a superseded derivation is dropped, not returned: by the time it finished,
      // the document it describes is not the document the editor holds.
      guard();
      return answer;
    } finally {
      if (this.deriving.get(path) === entry) this.deriving.delete(path);
    }
  }

  // --- the checkpoint ------------------------------------------------------------

  /** `checkCheckpoint(derived, headers)`: V17 against what the files hold (§4.19). */
  checkCheckpoint(derived: PyValue, headers: CheckpointHeaders): CheckpointReport {
    const answer = checkCheckpointOf(derived, headers);
    return {
      errors: answer.errors.map((one) => checkpointRow(one)),
      warnings: answer.warnings.map((one) => checkpointRow(one)),
      stats: answer.stats,
      rows: [...answer.errors, ...answer.warnings],
    };
  }

  /** `readHeader(bytes, file)`: a safetensors header, in the language's dtype names. */
  readHeader(bytes: Uint8Array, file: string): HeaderRead {
    return readHeaderOf(bytes, file);
  }

  // --- expressions ---------------------------------------------------------------

  /**
   * `evaluate(e, quantities, env?)`: the **model** side of `expr.py`.
   *
   * There are two evaluators and the signature names this one: the model side reads a document's
   * quantities and the composition indices in scope. The primitive side reads an instance's
   * resolved arguments, which is not a `quantities` map at all — its answers reach the interface
   * through `describe`'s facts, where the walk that resolved them recorded them (feature 1.6a).
   */
  evaluate(expression: PyValue, quantities: Quantities, env?: Env): PyValue {
    return modelValue(expression, quantities, env);
  }

  // --- the session's own state ---------------------------------------------------

  /** Drop what is held for a document: a closed tab, a file reverted. */
  forget(path: string): void {
    this.readings.delete(path);
    // The document this derivation describes is gone, so its result is dropped like any other
    // that no longer describes what the editor holds.
    this.deriving.get(path)?.cancel('superseded');
  }

  /** Drop a registry or a gathered library: a workspace closed, a base reloaded. */
  release(handle: SchemasHandle | LibraryHandle): void {
    if (handle.kind === 'schemas') this.schemas.delete(handle.id);
    else this.libraries.delete(handle.id);
  }

  /** Everything held, dropped: the proxy closed. */
  clear(): void {
    this.schemas.clear();
    this.libraries.clear();
    this.readings.clear();
    // `closed`, not `superseded`: nothing replaced these derivations, the service they were asked
    // of has ended, and that is what the caller's `catch` reads.
    for (const [, entry] of this.deriving) entry.cancel('closed');
    this.deriving.clear();
  }

  // --- what the calls share ------------------------------------------------------

  private identity(kind: string): string {
    this.next += 1;
    return `${kind}-${String(this.next)}`;
  }

  private heldSchemas(handle: SchemasHandle): HeldSchemas {
    const held = this.schemas.get(handle.id);
    if (held === undefined) throw new LangHandleError(`no schemas are held under ${handle.id}`);
    return held;
  }

  private heldLibrary(handle: LibraryHandle): HeldLibrary {
    const held = this.libraries.get(handle.id);
    if (held === undefined) throw new LangHandleError(`no library is held under ${handle.id}`);
    return held;
  }

  /** The grammar stage, from the memo when the caller named a revision it already read. */
  private read(
    tree: JsonValue,
    path: string,
    options: DocumentOptions,
    held: HeldLibrary,
  ): HeldReading {
    const kept = this.kept(path, options, held);
    if (kept !== null) return kept;
    const structural = held.schemas.registry.structural(tree, 'model');
    this.generations += 1;
    const reading: HeldReading = {
      generation: this.generations,
      revision: options.revision ?? -1,
      library: held.handle.id,
      assignment: assignmentKey(options.assignment),
      conforms: structural.length === 0,
      structural,
      analysis: null,
    };
    this.readings.set(path, reading);
    return reading;
  }

  /**
   * The semantic stage, from the memo where one was kept for this very revision.
   *
   * **What this publishes into may not be what its caller read.** `validate` suspends before it
   * gets here, and in that gap the session can have been given a newer document under the same
   * path — a keystroke's `describe`, a second `validate` — or told to drop it entirely by
   * `forget`. So the analysis is published only while the reading it was read for is still the
   * one held ({@link HeldReading.generation}). A superseded call still *answers*: its result is
   * a true reading of the document it was handed, and the caller asked for it. What it may not do
   * is leave that reading behind as the session's, because `check` reads what the session holds
   * and would then judge a candidate edit against a graph the editor has moved on from — and,
   * where the newer document is off the grammar, would decide what nothing can decide.
   */
  private analyse(
    document: PyRecord,
    path: string,
    options: DocumentOptions,
    held: HeldLibrary,
    reading: HeldReading,
  ): Analysis {
    if (reading.analysis !== null) return reading.analysis;
    const current = this.readings.get(path);
    const mine =
      current !== undefined && current.generation === reading.generation ? current : null;
    // The same installation, analysed while this call was suspended: a second call of one
    // revision reads the first's work rather than repeating it.
    if (mine !== null && mine.analysis !== null) return mine.analysis;
    const analysis = analyse(document, held.library, {
      ...(options.assignment === undefined ? {} : { assignment: options.assignment }),
    });
    // Kept whether or not a revision was named: `check` reads the analysis the session holds, and
    // a reading with no revision is simply never *reused* — `kept` refuses it.
    if (mine !== null) this.readings.set(path, { ...reading, analysis });
    return analysis;
  }

  /** The reading held for this document, where it is the same document read the same way. */
  private kept(path: string, options: DocumentOptions, held: HeldLibrary): HeldReading | null {
    if (options.revision === undefined) return null;
    const reading = this.readings.get(path);
    if (reading === undefined) return null;
    if (reading.revision !== options.revision) return null;
    if (reading.library !== held.handle.id) return null;
    if (reading.assignment !== assignmentKey(options.assignment)) return null;
    return reading;
  }
}

/** Raise what the control says, where it says anything. */
function watch(control: Control): void {
  const cancelled = control.cancelled();
  if (cancelled !== null) throw cancelled;
}

/** The document as the evaluators read it: `toPython` of the tree a caller handed over. */
function recordOf(tree: JsonValue): PyRecord {
  return toPython(tree) as PyRecord;
}

/** The assignment as one comparable key: `pyRepr` is deterministic and an assignment is small. */
function assignmentKey(assignment?: PyRecord): string {
  return assignment === undefined ? '' : pyRepr(assignment);
}

/**
 * The base a unit's path falls under, or the path's own directory when none of them does.
 *
 * The comparison is on the normalised paths and the answer is the caller's spelling ({@link
 * HeldBase}). The fallback is the file's own directory, which is what makes the refusal read
 * "not under primitives/, axes/ or precision/ of <where the file is>" for a file under no base at
 * all — a statement about the file, not about a base it was never claimed to belong to.
 */
function baseOf(bases: readonly HeldBase[], path: string): string {
  const file = normalise(path);
  const under = bases.filter((base) => contains(base.root, file));
  // The longest match, so that a base inside another is preferred to the one that contains it.
  // The normalised length, because that is the path the containment was decided on.
  under.sort((left, right) => right.root.length - left.root.length);
  return under[0]?.path ?? dirname(path);
}

/**
 * Whether a normalised path lies at or under a normalised directory.
 *
 * A prefix test on the text alone would read `base-other/x.json` as inside `base`, so the
 * separator is part of what is compared. The two directories that are not spelled with a trailing
 * segment are the two `normalise` can answer: `/`, the absolute root, and `.`, the relative one.
 */
function contains(root: string, file: string): boolean {
  if (root === file) return true;
  if (root === '/') return file.startsWith('/');
  if (root === '.') return !file.startsWith('/');
  return file.startsWith(`${root}/`);
}

/**
 * The session's own guard: a base the document declares that the gathered library has not.
 *
 * The tools load the library *from the document* on every run (`load_for(path, document, …)`), so
 * the two can never disagree there. The API gathers once per session so that a keystroke does not
 * pay the load's 261 ms, and the caller is the one that resolved the bases — which is exactly
 * where a document could end up validated against a vocabulary it does not declare. There is no
 * wording in the tools for that, so this states its own, as a **warning**: it changes no verdict,
 * and an override that gathers *more* than the document declares (the tools' own
 * `--primitive-library`) is not reported at all.
 */
function basesMissing(held: HeldLibrary, document: PyRecord, path: string): Problem[] {
  const { bases, problem } = basesOf(path, document);
  if (problem !== null) return [libraryRow(problem)];
  // `basesOf` answers normalised paths, and the held bases carry theirs for exactly this: a
  // document declaring `../primitive-library` from `data/models/` names the base the session
  // gathered as `data/primitive-library`, and the two must be recognised as one.
  const gathered = new Set(held.bases.map((base) => base.root));
  return bases
    .filter((base) => !gathered.has(base))
    .map((base) => ({
      code: '',
      message:
        `${path}: the primitive library gathered for this session does not carry the base ` +
        `'${base}' the document declares`,
      path: '/primitive_libraries',
      rule: 'bases',
      file: path,
      severity: 'warning' as const,
      source: 'library' as const,
    }));
}
