/**
 * What the `Lang` API answers with: the problem the whole editor reads, the verdict of a
 * validation, the facts of a document, and the handles that name what the worker holds.
 *
 * Appendix C of the plan sketches these; this is that sketch made exact against what the core
 * actually carries, with every divergence stated where it is made.
 */
import type { PyRecord, PyValue } from '../expr/value.js';
import type { AssignmentNeeded } from '../validate/quantities.js';

/**
 * Which stage produced a problem (plan Appendix C).
 *
 * The first four are the stages a document crosses, in the order the tools cross them;
 * `checkpoint` is V17 against a file on disk, `derivation` a refusal that only a derivation can
 * raise, and `editor` is the interface's own row — a binding that names a slot an argument change
 * removed, a sidecar key that resolves nowhere — which no call of this API produces and which
 * feature 2.8 will.
 */
export type ProblemSource =
  | 'schema'
  | 'library'
  | 'semantic'
  | 'lint'
  | 'checkpoint'
  | 'derivation'
  | 'editor';

/**
 * Every {@link ProblemSource}, as a value each and as the order a panel lists them in.
 *
 * The order is §4.17's table: the four stages a document crosses, then the checkpoint, then the
 * interface's own rows, then a refusal only a derivation can raise. `satisfies` is what holds the
 * two in step — a member added to the union without a value here, or a value here the union does
 * not carry, fails to compile — so a component that lists the sources reads them rather than
 * writing them out (plan §1 (b): the same discipline the schemas' own vocabulary is held to, one
 * type along).
 */
export const PROBLEM_SOURCE = {
  schema: 'schema',
  library: 'library',
  semantic: 'semantic',
  lint: 'lint',
  checkpoint: 'checkpoint',
  editor: 'editor',
  derivation: 'derivation',
} as const satisfies Readonly<Record<ProblemSource, ProblemSource>>;

/** The sources, in §4.17's own order. */
export const PROBLEM_SOURCES: readonly ProblemSource[] = Object.values(PROBLEM_SOURCE);

/**
 * How much a problem weighs (plan Appendix C).
 *
 * `error` is a refusal: the document is not valid, and nothing downstream of the stage runs.
 * `warning` is a row the tools print without refusing — a lint finding ("an opinion a reasonable
 * author may decline"), the validator's own advisories, a checkpoint tensor nothing names.
 * `notice` has no producer in the core today; it is declared because Appendix C declares it, for
 * the editor's own rows.
 */
export type ProblemSeverity = 'error' | 'warning' | 'notice';

/** Every {@link ProblemSeverity}, heaviest first — §4.17's legend reads in this order. */
export const PROBLEM_SEVERITY = {
  error: 'error',
  warning: 'warning',
  notice: 'notice',
} as const satisfies Readonly<Record<ProblemSeverity, ProblemSeverity>>;

/** The severities, heaviest first. */
export const PROBLEM_SEVERITIES: readonly ProblemSeverity[] = Object.values(PROBLEM_SEVERITY);

/** One line under a problem's head, with the place in the document it names. */
export interface ProblemDetail {
  /** The tools' words for this line, unindented. */
  readonly message: string;
  /** Where, as a JSON pointer (RFC 6901); `''` when the line names no place. */
  readonly path: string;
}

/**
 * A fix the editor can apply to a problem.
 *
 * Plan §3 asks for them — "a Problem with a fix action (`Create axis 'ffn.inner' in this base`,
 * `Add argument 'inner'`)", "`Rebind q → q_gated`" — and the framework is feature 2.8's. No call
 * of this API produces one yet; the field is declared so that the producer arrives without the
 * contract moving.
 */
export interface FixAction {
  /** What distinguishes the fix, for the component that knows how to apply it. */
  readonly kind: string;
  /** What the menu says. */
  readonly title: string;
  /** Whatever the fix needs, as data. */
  readonly detail?: PyValue;
}

/**
 * One problem, whatever produced it (plan Appendix C, §3).
 *
 * "The core emits `{code, message, path, node?, rule?, file?}` natively — the message with the
 * tools' wording for parity, the pointer beside it." So {@link message} is the **line the tools
 * print**, prefix included (`[V8] attn @decoder/attn: '…' does not hold`), because that line is
 * the parity contract the rejection suite matches on; everything else is what the panel reads
 * instead of parsing it.
 */
export interface Problem {
  /**
   * The rule the line names: `V1` … `V20`, `V12` for the JSON layer, `schema` for the grammar,
   * `registry` for a missing schema, `''` where the producing stage names none (most of the
   * loader's refusals, every lint finding).
   */
  readonly code: string;
  /** The tools' line, word for word, as their own printer writes it. */
  readonly message: string;
  /**
   * Where in the document or the unit, as a JSON pointer (RFC 6901); `''` at the root.
   *
   * **The place in the document as it was written.** The semantic stage reads the document
   * `model.normalise` answers — §5.2 rule 7 expands every composition-scoped binding into one
   * top-level rule "before any other rule applies" — so its own pointers name places a file does
   * not have (`/bindings/values/decoder.attn.norm_in` for a rule written at
   * `/compositions/decoder/bindings/values/attn.norm_in`). A panel navigates by this, and a
   * pointer into a document nobody holds is no pointer at all, so the hoist's own record is read
   * backwards here and {@link normalisedPath} keeps what the core's reading named.
   */
  readonly path: string;
  /**
   * The place the core's own reading named, where {@link path} was written back from it.
   *
   * Present only where the two differ, which is only for a hoisted binding. It is what a parity
   * comparison against the tools' normalised reading would use, and what a log line says when a
   * navigation lands somewhere a reader did not expect.
   */
  readonly normalisedPath?: string;
  /**
   * True where {@link path} is the nearest written place rather than the place itself.
   *
   * The hoist invents three things — the generated instance selector, an index selecting itself,
   * an identity §5.2 rule 7 names after the rule — and a pointer into one of them has no written
   * counterpart. The nearest place that has one is answered, and this says so rather than letting
   * a component believe it has the exact node.
   */
  readonly approximate?: boolean;
  /**
   * The identifier of the derived thing the row is about: a D1 node, or the D3 identity instance
   * a checkpoint row names (`wq[layer=3]`).
   *
   * The **validator's** refusals record none: they name a site inside their message and carry a
   * pointer into the document beside it, which is what a panel navigates by. Stated rather than
   * guessed — parsing an identifier back out of a message would be a second reading of the
   * wording the parity job pins (plan §7 F1 is the change set that would fix it at the source).
   * The checkpoint check does record one, because `artifact.check` walks D3 identity by identity.
   */
  readonly node?: string;
  /**
   * The members of {@link path} the row refuses by name, where the producing stage names some.
   *
   * The grammar's `additionalProperties` is the one that does today: it reports the place of the
   * *object* and names the extra members in its prose, because that is where `jsonschema` puts
   * them and the wording is the parity contract. The names the check already computed travel
   * beside the line so that a reader can be taken to them — §4.10's "structural errors at their
   * range" — without anyone reading an identifier back out of a message (§7 F1's own shape).
   */
  readonly members?: readonly string[];
  /** Which rule of the producing stage decided it: a loader kind, a lint rule, a V17 kind. */
  readonly rule?: string;
  /** The file the row is about: a unit, a document, a checkpoint shard. */
  readonly file?: string;
  readonly severity: ProblemSeverity;
  readonly source: ProblemSource;
  /**
   * True where the tools would already have stopped before reaching this row — plan §3's
   * continuation, "extra rows, never fewer". The parity job compares the tools' rows only.
   */
  readonly afterRefusal?: boolean;
  /** The lines under the head, in the tools' order: the loader's refusals carry them. */
  readonly detail?: readonly ProblemDetail[];
  /**
   * The mechanical repairs offered beside the row (§4.17, plan §3).
   *
   * A list and not one, because S10 draws two on a row (`Rebind q → q_gated`, `Remove binding`)
   * and because "a fix action exists only where the fix is *one creation*" is a rule about each of
   * them, not about how many there are. **No stage of the core produces one**: a verdict is what
   * the core answers and a repair is what the editor offers, so the producer is feature 2.8's fix
   * framework and this is where it hangs them.
   */
  readonly fixes?: readonly FixAction[];
}

/** A stage of {@link Lang.validate}, and the unit progress is reported in. */
export type Stage = 'schema' | 'library' | 'assignment' | 'semantic' | 'lint';

/**
 * Every {@link Stage}, in the order `validate` runs them.
 *
 * Held to the union by `satisfies`, as {@link PROBLEM_SOURCE} is: a caller that asks which stages
 * ran — the Problems panel's banner, which says what a refusal stopped — reads the names rather
 * than writing them out.
 */
export const STAGE = {
  schema: 'schema',
  library: 'library',
  assignment: 'assignment',
  semantic: 'semantic',
  lint: 'lint',
} as const satisfies Readonly<Record<Stage, Stage>>;

/** The stages, in the order `validate` runs them. */
export const STAGES: readonly Stage[] = Object.values(STAGE);

/**
 * What a long call reports as it crosses a stage boundary.
 *
 * It is reported where the API *itself* sequences the work, and nowhere else: splitting a call of
 * the core into steps it does not have would be a second implementation of its order (D3), and
 * the derivation's order is feature 1.8e's — D3, D4, D2, D5, then D1 and the agreement, then D6.
 * So a validation reports its four stages and a derivation reports one. The pipeline of §5.4 is
 * four calls, each reporting its own.
 */
export interface Progress {
  /** What the caller asked for. */
  readonly call: string;
  /** What is starting. */
  readonly stage: Stage | 'expansion' | 'products';
  /** How many stages of this call have finished. */
  readonly done: number;
  /** How many it may run at most. */
  readonly total: number;
}

/** What a validation answers (plan §5.3). */
export interface Verdict {
  /**
   * The stages that ran, in order.
   *
   * The tools stop at the first that refuses — "meaning assumes grammar" — and so does this: a
   * document off the grammar answers `['schema']` and nothing else. A document whose external
   * quantities have no value answers without `'semantic'`, with {@link needsAssignment} set,
   * because `--validate` counts such a document as *skipped* and not as failed.
   */
  readonly stagesRun: readonly Stage[];
  /** Every row, in the order the stages produce them. */
  readonly problems: readonly Problem[];
  /** The counters `--validate` prints, by the tools' own names; empty where the stage did not run. */
  readonly stats: ReadonlyMap<string, PyValue>;
  /** Present when an external quantity has no value: what the sheet of §4.6 is generated from. */
  readonly needsAssignment?: AssignmentNeeded;
}

/**
 * A handle on something the worker holds because it cannot cross the boundary.
 *
 * A {@link SchemaRegistry} carries compiled Ajv validators and answers through methods, and a
 * gathered library is read through a {@link LibrarySource}: functions are a `DataCloneError`. So
 * the worker keeps them and the caller names them. The handle is a string so that it survives
 * every transport unchanged and prints in a log.
 */
export interface Handle<Kind extends string> {
  readonly kind: Kind;
  readonly id: string;
}

/** The schemas, loaded and held. */
export type SchemasHandle = Handle<'schemas'>;

/** A primitive library, gathered and held — with the schemas it was gathered against. */
export type LibraryHandle = Handle<'library'>;

/** One base as the caller hands it over: its path, and the bytes of every file under it. */
export interface LibraryBaseFiles {
  /** The base's path, as a document's `primitive_libraries[].base` resolves to it. */
  readonly base: string;
  /**
   * Every file of the base, by the same path space as {@link base}.
   *
   * A directory base is its manifest, its `primitives/`, `axes/` and `precision/` units and the
   * template documents its manifest points at; a monolithic base is the one file. Reading them is
   * the caller's — `Platform.workspace` (§5.2) — because the core opens nothing.
   */
  readonly files: Readonly<Record<string, string>>;
}

/** What {@link Lang.loadSchemas} answers: the handle, and what the interface reads (plan §5.3). */
export interface SchemasLoaded {
  readonly handle: SchemasHandle;
  /** Where the files came from, as the "no schema with $id ending in …" line names it. */
  readonly origin: string;
  /** Every schema indexed, by `$id`, with the role and the file it came from. */
  readonly schemas: readonly LoadedSchemaInfo[];
}

/** One schema of the registry, as the interface reads it. */
export interface LoadedSchemaInfo {
  readonly path: string;
  readonly id: string;
  readonly role: string;
}

/** What {@link Lang.loadLibrary} answers: the handle, the library, and its refusals (plan §5.3). */
export interface LibraryLoaded {
  readonly handle: LibraryHandle;
  /**
   * The gathered library: units, `byId`, axes, roles, template interfaces.
   *
   * It travels because the palette, the argument sheet and the primitive editor read it; it is
   * plain data and it clones. It is typed through the core's own `Library`, since the alternative
   * is a second declaration of thirty members that would have to be kept in step.
   */
  readonly library: LibraryData;
  /** Every refusal the load carries, in the tools' order: the first is the one they would raise. */
  readonly problems: readonly Problem[];
}

/**
 * The gathered library as it travels.
 *
 * `Library` itself is what the core answers; it is re-exported under this name so that a consumer
 * of the API names the thing it receives rather than reaching into the core's modules.
 */
export type LibraryData = import('../library/load.js').Library;

/**
 * What a document says it needs read before it can be judged — {@link Lang.documentBases}.
 *
 * The bases are `primitive_libraries[].base` taken relative to the document, in `bases_of`'s own
 * spelling, and they are what the workspace reads and hands to {@link Lang.loadLibrary}. The
 * problem is the one `bases_of` records where the document cannot be resolved from at all: a
 * `schema` that is not this revision, or no `primitive_libraries` to resolve from. It is reported
 * rather than raised, because a half-typed document is the ordinary case in an editor.
 */
export interface DocumentBases {
  readonly bases: readonly string[];
  readonly problems: readonly Problem[];
}

/** What every call that reads a document needs. */
export interface CallOptions {
  /** The library the document resolves against, gathered by {@link Lang.loadLibrary}. */
  readonly library: LibraryHandle;
  /** The external quantities' values (§4.6); absent for a closed document. */
  readonly assignment?: PyRecord;
  /**
   * The store's revision counter for this tree (§5.4).
   *
   * When it is given, the worker keeps the semantic analysis of `(path, revision)` and the next
   * call on the same pair reads it instead of analysing again — which is what lets the pipeline
   * of §5.4 pay for `analyse` once and `describe`, `check` and `validate` share it (feature
   * 1.6d's `describeAnalysis`). **The caller promises that one revision names one tree**; without
   * a revision nothing is kept and every call analyses afresh, which is always correct.
   */
  readonly revision?: number;
  /** Abort the request: a superseded keystroke, a closed tab. */
  readonly signal?: AbortSignal;
  /** Called as the call crosses a stage boundary. */
  readonly onProgress?: (progress: Progress) => void;
}

/** What {@link Lang.describe} answers (plan §5.3). */
export interface Facts {
  /** Whether the document is on the grammar. Nothing below was computed when it is not. */
  readonly conforms: boolean;
  /** The schema stage's rows, in `--validate`'s own words. */
  readonly structural: readonly Problem[];
  /**
   * One entry per described site, keyed by the identifier a refusal and D1 name it with
   * (`embed`, `decoder/attn[layer=3]`) — `SiteDescription.where`, §5.2 rule 2.
   *
   * The whole document's sites unless {@link DescribeOptions.only} named some: §3 calls
   * `describe` "for the selected instance", and on the largest corpus documents the clone of
   * every site costs more than the round trip is given (`editor/spikes/timings.md`).
   */
  readonly sites: ReadonlyMap<string, import('../describe/site.js').SiteDescription>;
  /**
   * Present where an external quantity has no value, with no facts beside it.
   *
   * A template denotes one graph per admissible assignment (§4.6), and every call site of the
   * tools checks `missing_assignment` before it analyses: "a template with no assignment is
   * skipped, not failed — refusing it would report a defect where there is none", and the skip is
   * printed, never silent (I7). `describe` is called on every keystroke, so it takes that gate
   * rather than letting `index 'layer': stop does not resolve to a value` out; §4.6's assignment
   * sheet is generated from what this carries.
   */
  readonly needsAssignment?: AssignmentNeeded;
  /**
   * Present where an assignment supplies a value the document's own declaration refuses, with no
   * facts beside it: `check_assignment`'s rows, in `--validate`'s wording.
   *
   * The gate beside {@link needsAssignment}, and for the same reason: a value *present* is not a
   * value *admissible*, and `analyse` reads an assigned quantity as the extent of an index range
   * — a `1.5` where a cardinality is declared reaches `range` and raises `'float' object cannot be
   * interpreted as an integer` out of the expansion. `validate` takes this gate before its
   * semantic stage (§4.6); `describe` is called on every keystroke of the assignment sheet, which
   * is precisely where a half-typed value lives, so it takes the same one and reports it rather
   * than raising. The rows are the same rows, from the same `checkAssignment`: the two calls
   * cannot disagree about an assignment.
   *
   * They are **not** {@link structural}: the document is on the grammar, and it is the assignment
   * that is refused.
   */
  readonly assignmentRefused?: readonly Problem[];
  /**
   * What §4.14's sheet asks about one identity, where {@link DescribeOptions.identity} named one.
   *
   * `null` where the document's bindings produced no instance of that rule — a rule every guard
   * removed, or a name no binding carries.
   */
  readonly identity?: import('../describe/identity.js').IdentityFacts | null;
}

/** What {@link Lang.describe} takes beside a document. */
export interface DescribeOptions extends CallOptions {
  /** Describe only these sites, by the identifier {@link Facts.sites} keys them with. */
  readonly only?: readonly string[];
  /**
   * Describe one site per *declared* instance — the folded canvas of §4.7.
   *
   * A composition's thirty-two iterations are one card, drawn over "one representative iteration"
   * (§4.8), and which iteration that is only the analysis knows: a guard removes some (§5.2 rule
   * 3), so a caller naming them in {@link only} would ask for sites that are not there. 9 sites
   * instead of 195 on `llama3-8b`, 35 instead of 552 on `deepseek-v4-pro`.
   */
  readonly folded?: boolean;
  /**
   * Fill the compatibility lists — `tiesWith` and `sharesWith` (V15, V9). Default true.
   *
   * The one knob that narrows what {@link only} and {@link folded} cannot: the partners a slot may
   * join are answered from every identity instance of the graph, so describing one site still pays
   * the document's walk — 3.9–63.7 ms, feature 1.6d. Feature 2.10 turned it off in the pipeline of
   * §5.4, which asks on every keystroke for facts the sheet and the cards show, and left it on for
   * the one call a "Tie to…" or "Share with…" menu makes (§4.11, feature 2.13). It changes nothing
   * else: the identity a slot *belongs to* is filled either way.
   */
  readonly compatibility?: boolean;
  /**
   * Answer {@link Facts.identity} for one binding rule as well — the sheet of §4.14.
   *
   * A location editor, the members it lists and the dtypes its role admits are about **one
   * identity**, where a description is about sites; the two are one call because they are read
   * off one analysis (§5.4's reading per revision), and a sheet that asked twice would analyse
   * twice. Name the rule as the *normalised* document names it (`decoder.attn.q`) and say which
   * map of `bindings` it is written in.
   */
  readonly identity?: import('../describe/identity.js').IdentityRequest;
}

/** What {@link Lang.validate} takes beside a document. */
export interface ValidateOptions extends CallOptions {
  /**
   * The documents the lint stage reads, and with them the lint stage itself.
   *
   * `--lint`'s answer is a **function of the set**: "called by none of the N model(s) linted"
   * names the size of the set, and the set decides what is called, so the same document linted
   * alone and linted with the corpus produces different lines (feature 1.10). The caller
   * therefore names the set — the open document alone, or the whole workspace — and the stage
   * does not run when it names none. Measured: one document 27.5 ms, the corpus 781 ms.
   */
  readonly lint?: LintDocuments;
}

/** The set `--lint` is read over: every document, with the path each was named by. */
export type LintDocuments = readonly import('../lint/lint.js').LintDocument[];

/**
 * What a checkpoint check answers (plan §5.3, `checkCheckpoint`).
 *
 * Two readings of the same answer, because it has two readers and neither is derived from the
 * other without loss: the Problems panel reads {@link Problem}s, and the Weights panel of §4.19
 * reads the physical tensor a row names, which a `Problem` has no member for.
 */
export interface CheckpointReport {
  readonly errors: readonly Problem[];
  readonly warnings: readonly Problem[];
  readonly stats: import('../artifact/check.js').CheckpointStats;
  /** The core's own rows, with the identity, the tensor and the kind each names. */
  readonly rows: readonly import('../artifact/check.js').CheckpointProblem[];
}

/** Why a request did not answer. */
export type CancelReason =
  /** The caller aborted it. */
  | 'requested'
  /** A later request for the same document replaced it: "one in-flight derivation per document". */
  | 'superseded'
  /** The proxy was closed with the request still outstanding. */
  | 'closed';

/** What a cancelled request rejects with. */
export class LangCancelled extends Error {
  readonly reason: CancelReason;
  /** The call that was cancelled. */
  readonly call: string;

  constructor(call: string, reason: CancelReason) {
    super(`${call} was cancelled (${reason})`);
    this.name = 'LangCancelled';
    this.call = call;
    this.reason = reason;
  }
}

/** Raised where a handle names nothing the worker holds: a released registry, a closed session. */
export class LangHandleError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'LangHandleError';
  }
}
