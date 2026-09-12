/**
 * The arguments of one instance: `resolve_arguments`, `_resolve_record`, `_check_argument_domain`
 * and the V8 block of `tools/validate.py`.
 *
 * A model names a primitive and supplies arguments (§3.3); what the primitive declares about them
 * — types, defaults, conditional presence, numeric domains, and the invariants that relate
 * several of them — is what this module reads. Three rules of §6 meet in one walk, in the order
 * the tools apply them:
 *
 * - **V2** — "every required primitive argument is present; every undeclared argument is
 *   rejected; declared defaults are applied before checking". A default is an expression over the
 *   *other* arguments (§4.6: `kv_heads` defaults to `heads`), resolved to a fixpoint in any order,
 *   acyclically; what never resolves is a refusal, never a guess (I7).
 * - **V3** — types, enums, record fields, units and domains conform, records recursively. The type
 *   table is {@link checkType}, shared with the quantity side (feature 1.5); what is this side's
 *   own is the *domain*, whose bound "is a scalar literal or a reference to a required-or-defaulted
 *   argument, checked here after the type" — evaluated in the instance's own resolved arguments,
 *   not in the empty scope a quantity's bound is read in.
 * - **V8** — the primitive's declared invariants hold on the resolved arguments, defaults applied.
 *   "A domain constrains one argument; an invariant relates several." An invariant reading an
 *   argument V3 refused is skipped — the V3 line covers it — and one reading an argument that may
 *   be absent is guarded by a `present` test, which the primitive library loader enforces at load
 *   (feature 1.3), so a decidable invariant is never spuriously false.
 *
 * **An inapplicable argument is refused, not ignored.** `present_when` says when an argument
 * applies; supplying it otherwise is V3, "not a value that is ignored (I2)" — which is why the
 * walk evaluates the condition for every declared name, present or absent, and why the answer is
 * a fact the property sheet shows (§4.12) rather than a silent filter.
 *
 * **The map is filled in place, and that is load-bearing.** A record argument's fields are
 * resolved into the very record held under the instance's argument map, because a condition
 * written as an absolute path (`rope.scaling.kind`) reads it through that map while its siblings
 * are still being resolved. The port keeps the mutation for that reason, and `evaluate` is applied
 * once per *given* value at the top level only — inside a record the values are values already.
 *
 * **What the sheet reads.** Beside the values and the refusals, the walk records one
 * {@link ArgumentFact} per declared argument — applicable, where its value came from, what it
 * resolved to, the verdict on its domain and the problems about it — and one
 * {@link InvariantVerdict} per invariant, with the values it read. That is the per-argument half
 * of `describe` (plan §5.3, §4.12): the property sheet displays those facts and computes none of
 * them itself, which is the rule of the inventory's §7 ("whether an argument applies — `describe`,
 * never a re-read of `present_when`").
 */
import {
  argumentAt,
  argumentReferences,
  primitiveCondition,
  primitiveValue,
} from '../expr/primitive.js';
import { truthy, UNRESOLVED, type PyRecord, type PyValue } from '../expr/value.js';
import { put } from '../json/index.js';
import { demand, entries, get, has, listOf, optional } from '../library/access.js';
import { pyStr } from '../library/repr.js';
import { comparePythonStrings } from '../schema/index.js';
import type { PathSegment } from '../schema/types.js';

import { checkDomain, checkType, type RecordCheck } from './conformance.js';
import { semanticProblem, type SemanticProblem } from './problems.js';

/**
 * How an argument value written in the document becomes a value: `analyse`'s `lambda v: static(v,
 * env)`, which is `static_argument(v, quantities, env)` — a literal, a resolved quantity, an index
 * of the composition, or a record of those.
 *
 * It is the caller's, because the environment is the call site's: feature 1.6b hands one per site.
 */
export type Evaluate = (value: PyValue) => PyValue;

/** Where an argument's effective value came from. */
export type ArgumentSource =
  /** The document supplies it at this site. */
  | 'given'
  /** The declaration's `default` supplies it (§4.6): not written, and not silent either. */
  | 'default'
  /** Nothing supplies it: an optional argument the document leaves out, or a default that failed. */
  | 'absent';

/** What V3 could say about an argument's declared domain, once the type has been decided. */
export type DomainVerdict =
  /** The declaration carries no domain. */
  | 'undeclared'
  /** A domain is declared and was not read: the argument is absent, inapplicable, or refused. */
  | 'unchecked'
  /** Read, and the value lies in it — an undecidable bound being skipped, never read as a limit. */
  | 'ok'
  /** Read, and refused: the problems are on the row. */
  | 'refused';

/**
 * What `describe` answers about one declared argument of one instance (plan §4.12).
 *
 * One per declared name, in declaration order, the fields of a record following their record —
 * flat, keyed by the argument path, as the sheet's rows are ("indented by depth", which the dots
 * of the path give). A record's fields appear only where the walk resolved them: an absent record,
 * an inapplicable one, or a value that is not a record has none, because the tools resolve none.
 */
export interface ArgumentFact {
  /** The argument path: `heads`, `rope.scaling.beta_fast`. */
  readonly path: string;
  /**
   * Where the value is written in the document, or where it would be: the place a sheet edits.
   *
   * The walk knows it — it is the place the V2 and V3 refusals about this argument carry — and a
   * caller that had to rebuild it would have to know that a record argument is written
   * `{"record": {…}}`, which is a tag of the grammar and the shortcut §1 (b) forbids the
   * interface. The last segment is the argument's own name, present or absent, so
   * `[...at.slice(0, -1)]` is the map it belongs to and `at` is the member itself.
   */
  readonly at: readonly PathSegment[];
  /**
   * The `kind` its declared type carries — `argument_type.kind`, as the unit writes it.
   *
   * Read rather than rendered: `--document primitive-schema` maps a kind onto what JSON Schema
   * can say (a cardinality and a whole-number physical are both `integer`, and a *set* domain on a
   * cardinality is an `enum`), so the artifact cannot answer "whose type kind matches the
   * argument's" — which is what §4.12 asks of the quantity mode's select. The interface compares
   * this with the alternative a quantity's own `type` is, and branches on neither.
   */
  readonly kind: string;
  /** Whether `present_when` holds for these arguments; true when the declaration carries none. */
  readonly applicable: boolean;
  /**
   * Whether the declaration requires the argument.
   *
   * As the declaration writes it, not as the generated argument schema does: an argument that is
   * required *under* a `present_when` is kept out of that artifact's `required` (it cannot be
   * unconditional there), and the sheet's `required` badge is about the declaration —
   * `applicable && required && source === 'absent'` is exactly V2's own condition for
   * `required argument missing`.
   */
  readonly required: boolean;
  /**
   * Whether the declaration says the argument is `structural`.
   *
   * "Structural summary | arguments whose declaration is `structural: true`" (§4.7) — and the
   * component inventory's §3 says whose answer it is: **the core says which**. It is read here,
   * inside the walk that has the declaration, because the answer a sheet and a card need is per
   * *argument path* and a record's fields declare it as its own members do.
   */
  readonly structural: boolean;
  /** Whether the effective value is the document's, the declared default's, or nothing. */
  readonly source: ArgumentSource;
  /** The effective value, defaults applied; absent when nothing supplies one, `UNRESOLVED` when refused. */
  readonly value?: PyValue;
  /** The expression the document writes at this site, when it writes one. */
  readonly written?: PyValue;
  /** V3's verdict on the declared domain. */
  readonly domain: DomainVerdict;
  /** The refusals about this argument — not those about its fields, which are on their own rows. */
  readonly problems: readonly SemanticProblem[];
  /**
   * The declaration's own `description` — plan §1's "Help text is the schema's", one level along:
   * the tooltip of an argument row is what the unit says about that argument.
   *
   * Read here, in the walk that holds the declaration, for the reason `structural` is (feature
   * 2.9): the sheet is handed facts and never a declaration to pick members out of. The three
   * documentation fields below are the same answer — what `--document primitive-library` renders
   * for a reader, rendered instead on the row that edits the value.
   */
  readonly description?: string;
  /** One description per admissible value of an enum argument, keyed by the value as a string. */
  readonly valueDescriptions?: Readonly<Record<string, string>>;
  /** A deprecated declaration's own words, and the argument that supersedes it where it names one. */
  readonly deprecation?: ArgumentDeprecation;
}

/** What a `deprecated` declaration says: why, and what to write instead (unit guide §4.1). */
export interface ArgumentDeprecation {
  readonly reason: string;
  readonly supersededBy?: string;
}

/** What V8 said about one declared invariant of one instance (§4.1, unit guide §5). */
export interface InvariantVerdict {
  /** The words the unit gives the relation: what the sheet shows, and what the refusal quotes. */
  readonly description: PyValue;
  /** Whether the relation holds, fails, or was skipped because it reads a refused argument. */
  readonly verdict: 'holds' | 'fails' | 'skipped';
  /** The argument paths the condition reads, sorted as the tools sort them. */
  readonly reads: readonly string[];
  /** `heads = 32, kv_heads = 3`: the values it read, as the refusal shows them. */
  readonly shown: string;
  /** The V8 refusal, when it fails. */
  readonly problem?: SemanticProblem;
}

/** `resolve_arguments`' two answers, with the facts the walk saw on the way. */
export interface ArgumentResolution {
  /** "The complete, typed argument map of one instance": defaults applied, records recursive. */
  readonly values: PyRecord;
  /** The V2 and V3 refusals, in the order the tools append them. */
  readonly problems: readonly SemanticProblem[];
  /** One fact per declared argument, in declaration order (§4.12). */
  readonly facts: readonly ArgumentFact[];
}

/** What the V8 block answers about one instance. */
export interface InvariantResult {
  /** One refusal per invariant that does not hold, in declaration order. */
  readonly problems: readonly SemanticProblem[];
  /** One verdict per declared invariant, holding ones included: the block under the sheet's rows. */
  readonly verdicts: readonly InvariantVerdict[];
}

/** The per-argument half of `describe`: the resolution, the invariants, and the refusals of both. */
export interface ArgumentDescription extends ArgumentResolution {
  /** The V2 and V3 refusals of the resolution, then the V8 refusals — `analyse`'s own order. */
  readonly problems: readonly SemanticProblem[];
  /** One verdict per declared invariant. */
  readonly invariants: readonly InvariantVerdict[];
}

/**
 * `resolve_arguments(definition, given, evaluate)`: the complete, typed argument map of one
 * instance, and the problems found on the way.
 *
 * `at` is where the instance is in the document — feature 1.6b passes `['instances', 'attn']` —
 * and every problem is placed under its `arguments` from there, at the member it is about when the
 * document writes one and at the map it belongs to when it does not (a missing required argument
 * has no node to point at). The tools carry no place at all; the message is the parity contract
 * and the pointer is the plan's addition (§3).
 */
export function resolveArguments(
  definition: PyValue,
  given: PyValue,
  evaluate: Evaluate,
  at: readonly PathSegment[] = [],
): ArgumentResolution {
  const problems: SemanticProblem[] = [];
  const facts: FactUnderWay[] = [];
  const values = resolveRecord({
    declared: demand(definition, 'arguments'),
    given,
    written: given,
    evaluate,
    root: null,
    path: '',
    at: [...at, 'arguments'],
    into: null,
    problems,
    facts,
  });
  return { values, problems, facts };
}

/**
 * The V8 block of `analyse`: the primitive's invariants on the resolved arguments, after the types
 * and the domains.
 *
 * "An invariant reading an argument refused upstream (`UNRESOLVED`) is skipped — the V3 line
 * already covers it; otherwise it must hold." The message the tools write is
 * `'<description>' does not hold (heads = 32, kv_heads = 3)`, the values being those of the paths
 * the condition reads, sorted, and only those that resolve to something; `analyse` prefixes it
 * with the instance and its site, which feature 1.6b adds.
 *
 * The verdicts carry the same reading for an invariant that *holds*, which the tools never
 * compute because they speak only to refuse: the sheet shows every invariant with ✓ or ✗ and the
 * values it read (§4.12).
 */
export function checkInvariants(
  definition: PyValue,
  values: PyRecord,
  at: readonly PathSegment[] = [],
): InvariantResult {
  const problems: SemanticProblem[] = [];
  const verdicts: InvariantVerdict[] = [];
  for (const invariant of listOf(optional(definition, 'invariants', []))) {
    const holds = demand(invariant, 'holds');
    const description = demand(invariant, 'description');
    const references = argumentReferences(holds);
    const reads = [...references].sort(comparePythonStrings);
    // `', '.join(f"{p} = {_resolve_path(p, args)}" for p in sorted(refs) if … is not None)`. A
    // path that resolves to nothing is left out; so is one that resolves to the sentinel, which
    // the tools never reach here — an invariant reading one is skipped before the text is built.
    const shown = reads
      .map((path) => [path, argumentAt(path.split('.'), values)] as const)
      .filter(([, value]) => value !== null && value !== UNRESOLVED)
      .map(([path, value]) => `${path} = ${pyStr(value)}`)
      .join(', ');
    if ([...references].some((path) => argumentAt(path.split('.'), values) === UNRESOLVED)) {
      verdicts.push({ description, verdict: 'skipped', reads, shown });
      continue;
    }
    if (truthy(primitiveCondition(holds, values))) {
      verdicts.push({ description, verdict: 'holds', reads, shown });
      continue;
    }
    const problem = semanticProblem(
      'V8',
      `'${pyStr(description)}' does not hold` + (shown === '' ? '' : ` (${shown})`),
      at,
    );
    problems.push(problem);
    verdicts.push({ description, verdict: 'fails', reads, shown, problem });
  }
  return { problems, verdicts };
}

/**
 * The arguments of one instance as `describe` reports them (plan §5.3): the resolution, the
 * invariant verdicts, and the refusals of both in the order `analyse` emits them — the V2 and V3
 * lines of the resolution first, then V8.
 */
export function describeArguments(
  definition: PyValue,
  given: PyValue,
  evaluate: Evaluate,
  at: readonly PathSegment[] = [],
): ArgumentDescription {
  const resolution = resolveArguments(definition, given, evaluate, at);
  const invariants = checkInvariants(definition, resolution.values, at);
  return {
    values: resolution.values,
    problems: [...resolution.problems, ...invariants.problems],
    facts: resolution.facts,
    invariants: invariants.verdicts,
  };
}

/**
 * `_check_argument_domain`: a **primitive argument's** value against its declared domain (§4.6,
 * "admissibility at the call site").
 *
 * The same table as a quantity's ({@link checkDomain}, feature 1.5) — the same two shapes, the
 * same wording — read in a different scope: "a bound's value is a scalar literal or a reference to
 * another argument (`kv_heads <= heads`), evaluated in the instance's resolved arguments; an
 * undecidable bound (its argument refused upstream) is skipped, not read as a limit". An argument
 * path that resolves to nothing answers `None` on the primitive side, and is skipped too.
 */
export function checkArgumentDomain(
  value: PyValue,
  domain: PyValue,
  label: string,
  problems: SemanticProblem[],
  scope: PyRecord,
): void {
  checkDomain(value, domain, label, problems, (bound) => {
    const limit = primitiveValue(bound, scope);
    return limit === UNRESOLVED || limit === null ? undefined : limit;
  });
}

/** A record being filled: the argument map of an instance, or the fields of a record argument. */
type MutableRecord = Record<string, PyValue>;

/** A fact while the walk is still filling it in; a caller reads it as an {@link ArgumentFact}. */
interface FactUnderWay {
  path: string;
  at: readonly PathSegment[];
  kind: string;
  applicable: boolean;
  required: boolean;
  structural: boolean;
  source: ArgumentSource;
  value?: PyValue;
  written?: PyValue;
  domain: DomainVerdict;
  problems: readonly SemanticProblem[];
  description?: string;
  valueDescriptions?: Readonly<Record<string, string>>;
  deprecation?: ArgumentDeprecation;
}

/** The documentation an `argument_declaration` carries, as a fact rather than as a node. */
function documentationOf(declaration: PyValue, fact: FactUnderWay): void {
  const description = optional(declaration, 'description', null);
  if (description !== null) fact.description = pyStr(description);
  const values = optional(declaration, 'value_descriptions', null);
  if (values !== null) {
    const found: Record<string, string> = {};
    for (const [value, text] of entries(values)) found[value] = pyStr(text);
    fact.valueDescriptions = found;
  }
  const deprecated = optional(declaration, 'deprecated', null);
  if (deprecated !== null) {
    const superseded = optional(deprecated, 'superseded_by', null);
    fact.deprecation = {
      reason: pyStr(demand(deprecated, 'reason')),
      ...(superseded === null ? {} : { supersededBy: pyStr(superseded) }),
    };
  }
}

/** One call of `_resolve_record`: what it resolves, where it writes, and what it reads paths in. */
interface Resolution {
  /** The declarations of this level: `definition['arguments']`, or a record type's `fields`. */
  readonly declared: PyValue;
  /** The values given at this level — expressions at the top, values inside a record. */
  readonly given: PyValue;
  /** The expressions the document writes at this level; `null` where it writes none. */
  readonly written: PyValue | null;
  /** `evaluate` at the top level, `lambda x: x` inside a record. */
  readonly evaluate: Evaluate;
  /** The map an absolute argument path is read in: the instance's own, `null` at the top level. */
  readonly root: MutableRecord | null;
  /** The prefix the labels carry: `''` at the top level, `rope.scaling.` inside a record. */
  readonly path: string;
  /** Where this level is in the document, for the places the problems carry. */
  readonly at: readonly PathSegment[];
  /** The record to fill in place — a record argument's own — or `null` for a fresh map. */
  readonly into: MutableRecord | null;
  readonly problems: SemanticProblem[];
  readonly facts: FactUnderWay[];
}

/**
 * `_resolve_record`: one map of values against one map of declarations — the top level of an
 * instance, or the fields of a record.
 *
 * "Unknown names are V2, everything about a value's type is V3. Defaults are applied before any
 * check (V2), and an inapplicable field is forbidden, not ignored (I2)."
 */
function resolveRecord(scope: Resolution): MutableRecord {
  const values: MutableRecord = scope.into ?? {};
  // `if root is None: root = values` — the top level: the record is the scope.
  const root = scope.root ?? values;
  const declared = scope.declared;
  const supplied = new Set<string>();

  for (const [name, value] of entries(scope.given)) {
    if (!has(declared, name)) {
      scope.problems.push(
        semanticProblem('V2', `unknown argument '${scope.path}${name}'`, [...scope.at, name]),
      );
    } else {
      supplied.add(name);
      put(values, name, scope.evaluate(value));
    }
  }

  // Defaults may read other arguments, in any order, acyclically (§4.6): a fixpoint, and what
  // never resolves is a refusal. Paths are absolute (`rope.scaling.kind`), so the scope is `root`.
  const pending = entries(declared)
    .filter(([name, declaration]) => !Object.hasOwn(values, name) && has(declaration, 'default'))
    .map(([name]) => name);
  while (pending.length > 0) {
    let progress = false;
    for (const name of [...pending]) {
      const value = primitiveValue(demand(demand(declared, name), 'default'), root);
      if (value !== null && value !== UNRESOLVED) {
        put<PyValue>(values, name, value);
        pending.splice(pending.indexOf(name), 1);
        progress = true;
      }
    }
    if (!progress) {
      for (const name of pending) {
        scope.problems.push(
          semanticProblem('V2', `default of '${scope.path}${name}' does not resolve`, scope.at),
        );
      }
      break;
    }
  }

  for (const [name, declaration] of entries(declared)) {
    const label = `${scope.path}${name}`;
    const written =
      scope.written !== null && has(scope.written, name) ? demand(scope.written, name) : undefined;
    // The place: the member the document writes, or the map the argument belongs to when it
    // writes none — a missing required argument and an unresolved default have no node.
    const place = supplied.has(name) ? [...scope.at, name] : scope.at;
    const fact: FactUnderWay = {
      path: label,
      // The member itself, written or not: a sheet that sets a value writes it there, and a sheet
      // that clears one removes it from the map above it. `place` is the *problem's* place, which
      // is the map where nothing is written — a refusal has no node to point at, a row has.
      at: [...scope.at, name],
      kind: pyStr(demand(demand(declaration, 'type'), 'kind')),
      applicable: true,
      required: truthy(demand(declaration, 'required')),
      structural: truthy(optional(declaration, 'structural', false)),
      source: 'absent',
      domain: has(declaration, 'domain') ? 'unchecked' : 'undeclared',
      problems: [],
    };
    documentationOf(declaration, fact);
    if (written !== undefined) fact.written = written;
    scope.facts.push(fact);
    const fromProblem = scope.problems.length;
    const fromFact = scope.facts.length;
    /** The problems appended for this argument, less those its fields took for their own rows. */
    const own = (): void => {
      const claimed = new Set(scope.facts.slice(fromFact).flatMap((one) => one.problems));
      fact.problems = scope.problems
        .slice(fromProblem)
        .filter((problem) => !claimed.has(problem));
    };

    if (has(declaration, 'present_when')) {
      // Python keeps whatever the condition answered and reads its truth; so does the fact.
      fact.applicable = truthy(primitiveCondition(demand(declaration, 'present_when'), root));
    }
    if (!Object.hasOwn(values, name)) {
      if (truthy(demand(declaration, 'required')) && fact.applicable) {
        scope.problems.push(semanticProblem('V2', `required argument missing '${label}'`, place));
      }
      own();
      continue;
    }
    fact.source = supplied.has(name) ? 'given' : 'default';
    fact.value = values[name] as PyValue;
    if (!fact.applicable) {
      scope.problems.push(
        semanticProblem(
          'V3',
          `argument '${label}' is present but inapplicable for these arguments`,
          place,
        ),
      );
      own();
      continue;
    }
    const type = demand(declaration, 'type');
    checkType(
      values[name] as PyValue,
      type,
      label,
      scope.problems,
      recordCheck(scope, {
        root,
        at: [...scope.at, name, 'record'],
        written: written === undefined ? null : get(written, 'record'),
      }),
    );
    if (
      scope.problems.length === fromProblem &&
      has(declaration, 'domain') &&
      values[name] !== UNRESOLVED
    ) {
      const beforeDomain = scope.problems.length;
      checkArgumentDomain(
        values[name] as PyValue,
        demand(declaration, 'domain'),
        label,
        scope.problems,
        root,
      );
      fact.domain = scope.problems.length === beforeDomain ? 'ok' : 'refused';
    }
    if (scope.problems.length > fromProblem && demand(type, 'kind') !== 'record') {
      // Refused once, with its reason; nothing downstream reads it as a value.
      put<PyValue>(values, name, UNRESOLVED);
    }
    // The type table and the domain table write no place; a field's problems have their own.
    locate(scope.problems, fromProblem, place);
    fact.value = values[name] as PyValue;
    own();
  }
  return values;
}

/** Where a record argument's fields are resolved: what they read paths in, and where they are. */
interface FieldScope {
  readonly root: MutableRecord;
  readonly at: readonly PathSegment[];
  readonly written: PyValue | null;
}

/**
 * The recursion `_check_type` makes into a record type, which the shared type table (feature 1.5)
 * takes as a parameter because the quantity side has none.
 *
 * "Records are evaluated already: their fields are values, not expressions" — hence the identity
 * evaluator — "and `v` is the very dict held under `root`, so it is filled in place and a
 * condition written as an absolute path sees it".
 */
function recordCheck(scope: Resolution, field: FieldScope): RecordCheck {
  return (value, declared, label, problems) => {
    // The record the walk filled is the one held under the instance's map: it is cleared and
    // refilled in place, so a condition written as an absolute path reads what is in it.
    const held: MutableRecord = value;
    const given: MutableRecord = {};
    for (const name of Object.keys(held)) put<PyValue>(given, name, held[name] as PyValue);
    for (const name of Object.keys(held)) delete held[name];
    resolveRecord({
      declared: demand(declared, 'fields'),
      given,
      written: field.written,
      evaluate: (one) => one,
      root: field.root,
      path: `${label}.`,
      at: field.at,
      into: held,
      problems,
      facts: scope.facts,
    });
  };
}

/**
 * The problems appended since `from` that carry no place yet, given the place they are about.
 *
 * A problem a deeper level already placed keeps its own: the fields of a record are at their own
 * pointers, and `at` is never empty here — the top level of an instance is at least `arguments`.
 */
function locate(
  problems: SemanticProblem[],
  from: number,
  segments: readonly PathSegment[],
): void {
  for (let index = from; index < problems.length; index += 1) {
    const problem = problems[index] as SemanticProblem;
    if (problem.path !== '') continue;
    problems[index] = semanticProblem(problem.code, problem.message, segments);
  }
}
