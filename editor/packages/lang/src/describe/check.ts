/**
 * `check`: what one candidate edit would be refused with (plan §5.3, §4.7, inventory §7).
 *
 * "Whether an edge may be made — `check` during the drag; and per Q5 the drop happens anyway."
 * The verdict is never a veto: the author wires first and fixes afterwards (§9 Q5), so this
 * answers *what the document would then say*, in the tools' words, and the caller shows it at the
 * handle during the drag and lists it in Problems on the drop.
 *
 * Three candidates, the three gestures of §4.7:
 *
 * - an **edge** — an output handle dragged onto an input handle: V1 (the ends resolve), V4 (the
 *   shapes unify), V7 (the input is fed once), V6 (the value graph stays acyclic) and V5 (the
 *   indexing domains agree);
 * - a **member** — a slot chip dragged onto another chip, or "Tie to…" / "Share state with…":
 *   V1 and V7 about the slot itself, then V15 for a parameter identity and V9 for a state one;
 * - a **location** — a physical tensor dragged from the Weights panel onto a slot chip: V17, the
 *   evaluation of the location against the slot's stored shape and the names it would bind.
 *
 * **Every line is the validator's own.** The V4, V5, V6 and V7 lines are written by the same
 * blocks of `analyse` this file calls into — {@link topologicalOrder} for the cycle,
 * {@link tyingProblems} for V15, {@link compareStateMember} for V9, {@link evaluateLocation},
 * {@link bindPhysicalNames} and {@link physicalNameProblems} for V17 — and the edge's own four
 * are written here because `analyse` writes them inside a loop over the document's bindings,
 * which a candidate is not in. Their wording is the contract (D2, §7 F1) and a test pins each.
 *
 * **What a candidate does not carry.** A binding fires in an index environment; a candidate names
 * concrete sites of the expanded graph, so it is judged as *one* edge and named with the empty
 * environment — what a top-level binding without `for_each` prints. The rule's name is the
 * caller's, defaulting to §4.7's own proposal (`<to>.<port>`).
 *
 * **A member candidate is a slot *joining* an identity, not a second binding of it.** "Every slot
 * present under its `present_when` is bound exactly once" (V7), so in a valid document every slot
 * a gesture can drag is already bound — to its own private identity — and the gesture moves it
 * ("creates or extends the identity", §4.7). Reporting the membership it leaves as "bound twice"
 * would make V7 fire on every candidate the compatibility list offers, which is why the verdict is
 * about the identity it joins: the slot's own V1 and V7 lines (an unknown instance, an unknown
 * slot, a slot absent under these arguments, a multiplicity that does not resolve) and then V15 or
 * V9 over the members it would stand beside.
 *
 * **What it does not decide.** V14 — "a dtype selected for an identity is admissible for the role
 * of every member" — is a consequence of a *member* candidate the compatibility lists of §5.3 do
 * not carry (the plan and the inventory both name V15 and V9 there), so it is left to the
 * validation that follows the drop; a finding, not an omission by accident.
 */
import { pyEqual } from '../expr/arithmetic.js';
import type { Env } from '../expr/model.js';
import { modelValue } from '../expr/model.js';
import { primitiveValue } from '../expr/primitive.js';
import type { PyValue } from '../expr/value.js';
import { demand, entries, has, listOf, optional } from '../library/access.js';
import { pyRepr, pyStr } from '../library/repr.js';
import type { PathSegment } from '../schema/types.js';
import {
  bindPhysicalNames,
  compareStateMember,
  declaredMultiplicity,
  emptyAgreement,
  evaluateLocation,
  indexingToken,
  keyOf,
  physicalNameProblems,
  portKeyOf,
  portShape,
  present,
  reprEnv,
  reprIndexings,
  reprShape,
  segmentsOfSite,
  semanticProblem,
  shapesAgree,
  signatureOf,
  storageShape,
  topologicalOrder,
  tyingProblems,
  whereOfSite,
  wholeCount,
  type Analysis,
  type Indexing,
  type PhysicalSlice,
  type ResolvedSite,
  type SemanticProblem,
  type SiteKey,
  type ValueEdge,
} from '../validate/index.js';

/** A `(site, port)` the editor points at: one handle of the expanded graph. */
export interface PortRef {
  readonly site: SiteKey;
  readonly port: string;
}

/** A `(site, slot)` the editor points at: one parameter, constant or state chip. */
export interface SlotRef {
  readonly site: SiteKey;
  readonly name: string;
}

/** An edge dragged from an output handle to an input handle (§4.7). */
export interface EdgeCandidate {
  readonly from: PortRef;
  readonly to: PortRef;
  /**
   * The name the binding would carry, which every refusal about it prints.
   *
   * Absent, §4.7's own proposal is used — `<to>.<port>`, the target site's name and its port —
   * so that the line a drag shows is the line the drop produces.
   */
  readonly rule?: string;
}

/** A slot or state port added to an identity: "Tie to…", "Share state with…", chip onto chip. */
export interface MemberCandidate {
  readonly kind: 'parameter' | 'state';
  readonly slot: SlotRef;
  /**
   * What it would join: an identity instance by name, or another slot — the chip-onto-chip
   * gesture, which "creates or extends the identity" (§4.7). A slot names the identity it is
   * already a member of; a slot that is a member of none is a new identity, which takes any
   * partner V15 or V9 admits.
   */
  readonly into: { readonly identity: string } | { readonly slot: SlotRef };
}

/** A location written on a parameter identity: a tensor dragged from the Weights panel (§4.19). */
export interface LocationCandidate {
  /** The identity instance the location would be written on: `wq[layer=3]`, `tied_embeddings`. */
  readonly identity: string;
  /** The location, as the document would write it (`{tensor}`, `{stack}`, `{concat}`, `{slice}`). */
  readonly location: PyValue;
}

/** One candidate edit, as `check` takes it. */
export type Candidate =
  | { readonly edge: EdgeCandidate }
  | { readonly member: MemberCandidate }
  | { readonly location: LocationCandidate };

/** What `check` answers about one candidate. */
export interface Verdict {
  /** Whether the candidate introduces no refusal at all. */
  readonly ok: boolean;
  /** What the document would refuse it with, in the validator's own words and order. */
  readonly problems: readonly SemanticProblem[];
  /**
   * What the candidate names and the document does not have, as a sentence of the core's own.
   *
   * The tools have no wording for it — nothing in a document can name an identity instance that
   * was never derived — so this is stated rather than guessed, as feature 1.3 states its own line
   * for a unit file in none of a base's sections. `null` when everything the candidate names
   * resolved.
   */
  readonly unknown: string | null;
}

/** The verdict on one candidate, over an analysed document. */
export function checkCandidate(analysis: Analysis, candidate: Candidate): Verdict {
  if ('edge' in candidate) return checkEdge(analysis, candidate.edge);
  if ('member' in candidate) return checkMember(analysis, candidate.member);
  return checkLocation(analysis, candidate.location);
}

/** A verdict from the problems it collected. */
function verdict(problems: readonly SemanticProblem[], unknown: string | null = null): Verdict {
  return { ok: problems.length === 0 && unknown === null, problems, unknown };
}

// --- an edge ----------------------------------------------------------------

function checkEdge(analysis: Analysis, candidate: EdgeCandidate): Verdict {
  const rule = candidate.rule ?? `${candidate.to.site.name}.${candidate.to.port}`;
  const at: PathSegment[] = ['bindings', 'values', rule];
  const problems: SemanticProblem[] = [];
  const empty: Env = new Map();

  // `for end in ((src, 'from', 'outputs'), (dst, 'to', 'inputs'))`: V1 on each end, in order.
  const ends = [
    { ref: candidate.from, side: 'outputs', label: 'from' },
    { ref: candidate.to, side: 'inputs', label: 'to' },
  ] as const;
  let ok = true;
  const sites: (ResolvedSite | undefined)[] = [];
  for (const end of ends) {
    const site = analysis.resolved.get(keyOf(end.ref.site));
    sites.push(site);
    if (site === undefined) {
      problems.push(
        semanticProblem(
          'V1',
          `${rule}${reprEnv(empty)}: ${end.label} instance does not exist ` +
            `${whereOfSite(end.ref.site)}`,
          at,
        ),
      );
      ok = false;
      continue;
    }
    if (!has(demand(demand(site.definition, 'ports'), end.side), end.ref.port)) {
      problems.push(
        semanticProblem(
          'V1',
          `${rule}: ${pyStr(site.primitive)} has no ${end.side.slice(0, -1)} port ` +
            `'${end.ref.port}'`,
          at,
        ),
      );
      ok = false;
    }
  }
  if (!ok) return verdict(problems);
  const from = sites[0] as ResolvedSite;
  const to = sites[1] as ResolvedSite;

  // V4: shapes unify by axis identity and exact extent.
  const fromShape = portShape(
    demand(demand(demand(from.definition, 'ports'), 'outputs'), candidate.from.port),
    from.args,
  );
  const toShape = portShape(
    demand(demand(demand(to.definition, 'ports'), 'inputs'), candidate.to.port),
    to.args,
  );
  if (fromShape !== null && toShape !== null && !shapesAgree(fromShape, toShape)) {
    problems.push(
      semanticProblem(
        'V4',
        `${rule}: shapes do not unify ${pyStr(from.primitive)}.${candidate.from.port}` +
          `${reprShape(fromShape)} -> ${pyStr(to.primitive)}.${candidate.to.port}` +
          `${reprShape(toShape)}`,
        at,
      ),
    );
  }

  // V7: an input port is fed exactly once. The tools walk the value bindings before the public
  // inputs, so a candidate binding onto a port an *input* feeds is refused by the interface
  // block's own line, which names the input and blames the binding that got there first.
  const fed = portKeyOf(candidate.to.site, candidate.to.port);
  const producer = analysis.producers.get(fed);
  if (producer !== undefined) {
    const input = producer.by.startsWith('input:') ? producer.by.slice('input:'.length) : null;
    problems.push(
      input === null
        ? semanticProblem(
            'V7',
            `input port fed twice: ${whereOfSite(candidate.to.site)}.${candidate.to.port} ` +
              `by ${producer.by} and ${rule}`,
            at,
          )
        : semanticProblem(
            'V7',
            `input ${input}: port ${whereOfSite(candidate.to.site)}.${candidate.to.port} also ` +
              `fed by ${rule}`,
            ['interfaces', 'inputs', input],
          ),
    );
  }

  // V6: the value graph is acyclic within an invocation. Only what the candidate introduces is
  // its own: a document already in a cycle carries that refusal without it.
  if (analysis.order.length === analysis.resolved.size) {
    const edge: ValueEdge = {
      from: candidate.from.site,
      fromPort: candidate.from.port,
      to: candidate.to.site,
      toPort: candidate.to.port,
      binding: rule,
    };
    const order = topologicalOrder(analysis.resolved, [...analysis.edges, edge]);
    if (order.length !== analysis.resolved.size) {
      problems.push(
        semanticProblem(
          'V6',
          `value cycle: ${analysis.resolved.size - order.length} instance(s) in a cycle`,
          [],
        ),
      );
    }
  }

  // V5: "indexing domains agree on every edge (§5.3)".
  problems.push(...domainProblems(analysis, candidate, rule, to));
  return verdict(problems);
}

/** The V5 lines a candidate edge would produce at its target (§5.3). */
function domainProblems(
  analysis: Analysis,
  candidate: EdgeCandidate,
  rule: string,
  to: ResolvedSite,
): SemanticProblem[] {
  const problems: SemanticProblem[] = [];
  const at = segmentsOfSite(to.key);
  const source = analysis.domains.get(portKeyOf(candidate.from.site, candidate.from.port));
  if (source === undefined) {
    problems.push(
      semanticProblem(
        'V5',
        `${rule}: the domain of ${whereOfSite(candidate.from.site)}.${candidate.from.port} ` +
          `is undetermined`,
        ['bindings', 'values', rule],
      ),
    );
    return problems;
  }
  const carried = source.domain;
  const ports = demand(demand(to.definition, 'ports'), 'inputs');
  const port = demand(ports, candidate.to.port);
  const declared = demand(demand(port, 'domain'), 'kind');
  if (!pyEqual(declared, 'inherit') && !pyEqual(declared, carried[0])) {
    problems.push(
      semanticProblem(
        'V5',
        `${pyStr(to.primitive)}@${whereOfSite(to.key)}.${candidate.to.port} expects ` +
          `${pyStr(declared)}, receives ${pyStr(carried[0])} (stream '${pyStr(carried[1])}')`,
        at,
      ),
    );
  }
  // "an instance's own domain is the common domain of its untransformed inputs": the candidate's
  // domain beside the ones the graph already resolved for this instance's other present inputs.
  const transformed = new Set(
    listOf(optional(to.definition, 'domain_transforms', [])).map((transform) =>
      pyStr(demand(transform, 'from_port')),
    ),
  );
  const agree = new Map<string, Indexing>();
  for (const [name, declaredPort] of entries(ports)) {
    if (!present(declaredPort, to.args) || transformed.has(name)) continue;
    const entry =
      name === candidate.to.port
        ? carried
        : analysis.domains.get(portKeyOf(to.key, name))?.domain;
    if (entry === undefined) continue;
    agree.set(indexingToken(entry), entry);
  }
  if (agree.size > 1) {
    problems.push(
      semanticProblem(
        'V5',
        `${pyStr(to.primitive)}@${whereOfSite(to.key)}: inputs in different domains ` +
          `${reprIndexings([...agree.values()])}, and no domain_transform declares it`,
        at,
      ),
    );
  }
  return problems;
}

// --- a member ---------------------------------------------------------------

function checkMember(analysis: Analysis, candidate: MemberCandidate): Verdict {
  const site = analysis.resolved.get(keyOf(candidate.slot.site));
  const parameter = candidate.kind === 'parameter';
  const held = identityOf(analysis, candidate, parameter);
  if (typeof held === 'string') return verdict([], held);
  const rule = held.rule;
  const at: PathSegment[] = ['bindings', parameter ? 'parameters' : 'states', rule];
  const problems: SemanticProblem[] = [];
  const label = parameter ? 'parameter' : 'state';
  if (site === undefined) {
    return verdict([
      semanticProblem(
        'V1',
        `${label} ${rule}: instance does not exist ${whereOfSite(candidate.slot.site)}`,
        at,
      ),
    ]);
  }
  const declared = optional(
    demand(site.definition, parameter ? 'parameters' : 'state_ports'),
    candidate.slot.name,
    null,
  );
  if (declared === null) {
    return verdict([
      parameter
        ? semanticProblem(
            'V7',
            `parameter ${rule}: ${pyStr(site.primitive)} has no parameter ` +
              `'${candidate.slot.name}'`,
            at,
          )
        : semanticProblem(
            'V1',
            `state ${rule}: ${pyStr(site.primitive)} has no state port '${candidate.slot.name}'`,
            at,
          ),
    ]);
  }
  if (!present(declared, site.args)) {
    return verdict([
      semanticProblem(
        'V7',
        parameter
          ? `parameter ${rule}: slot '${candidate.slot.name}' absent for these arguments`
          : `state ${rule}: port '${candidate.slot.name}' absent for these arguments`,
        at,
      ),
    ]);
  }
  if (parameter) {
    // "A present slot's declared multiplicity resolves to a positive integer" (V7, §3.4).
    const multiplicity = declaredMultiplicity(declared);
    if (multiplicity !== null) {
      const count = primitiveValue(multiplicity, site.args);
      if (wholeCount(count) === null) {
        problems.push(
          semanticProblem(
            'V7',
            `parameter ${rule}: slot '${candidate.slot.name}' declares a multiplicity that ` +
              `resolves to ${pyRepr(count)} — a present slot has a positive integer number of ` +
              'copies; none is `present_when` (§3.4)',
            at,
          ),
        );
      }
    }
  }
  problems.push(
    ...(parameter
      ? tyingVerdict(analysis, candidate, site, declared, held, at)
      : sharingVerdict(analysis, candidate, site, declared, held, at)),
  );
  return verdict(problems);
}

/** The identity a member candidate would join: the binding that declares it and its members. */
interface Joined {
  readonly rule: string;
  readonly members: readonly { readonly site: SiteKey; readonly name: PyValue }[];
}

/** The identity a member candidate would join: named, or the one another slot belongs to. */
function identityOf(
  analysis: Analysis,
  candidate: MemberCandidate,
  parameter: boolean,
): Joined | string {
  const instances = parameter
    ? analysis.bindings.tensorInstances
    : analysis.bindings.stateInstances;
  if ('identity' in candidate.into) {
    const named = candidate.into.identity;
    const found = instances.find((one) => one.identity === named);
    if (found === undefined) {
      return (
        `no ${parameter ? 'parameter' : 'state'} identity instance named ` +
        `'${named}' was derived from this document`
      );
    }
    return { rule: found.rule, members: found.members };
  }
  const other = candidate.into.slot;
  const key = portKeyOf(other.site, other.name);
  const found = instances.find((one) =>
    one.members.some((member) => portKeyOf(member.site, member.name) === key),
  );
  // A slot that is a member of no identity is a new identity of its own: the gesture "creates or
  // extends" one (§4.7), and a new identity's only member is the slot dragged onto.
  if (found === undefined) {
    const site = analysis.resolved.get(keyOf(other.site));
    if (site === undefined) {
      return `${whereOfSite(other.site)} is not an instance of this document`;
    }
    return { rule: `${other.site.name}.${other.name}`, members: [{ site: other.site, name: other.name }] };
  }
  return { rule: found.rule, members: found.members };
}

/** V15 over the identity's members and the candidate: "parameter identity compatibility (§3.4)". */
function tyingVerdict(
  analysis: Analysis,
  candidate: MemberCandidate,
  site: ResolvedSite,
  declared: PyValue,
  held: Joined,
  at: readonly PathSegment[],
): SemanticProblem[] {
  const signatures = [];
  for (const member of held.members) {
    if (portKeyOf(member.site, member.name) === portKeyOf(candidate.slot.site, candidate.slot.name)) {
      continue;
    }
    const other = analysis.resolved.get(keyOf(member.site));
    if (other === undefined) continue;
    const slot = optional(demand(other.definition, 'parameters'), pyStr(member.name), null);
    if (slot === null) continue;
    signatures.push(signatureOf(other, slot, member.name));
  }
  if (signatures.length === 0) return []; // one member: `checkTying` does not run at all
  signatures.push(signatureOf(site, declared, candidate.slot.name));
  return tyingProblems(held.rule, signatures, at);
}

/** V9 over the identity's members and the candidate: "a state identity connects only compatible ports". */
function sharingVerdict(
  analysis: Analysis,
  candidate: MemberCandidate,
  site: ResolvedSite,
  declared: PyValue,
  held: Joined,
  at: readonly PathSegment[],
): SemanticProblem[] {
  const agreement = emptyAgreement();
  for (const member of held.members) {
    if (portKeyOf(member.site, member.name) === portKeyOf(candidate.slot.site, candidate.slot.name)) {
      continue;
    }
    const other = analysis.resolved.get(keyOf(member.site));
    if (other === undefined) continue;
    const port = optional(demand(other.definition, 'state_ports'), pyStr(member.name), null);
    if (port === null) continue;
    compareStateMember(analysis, held.rule, other, pyStr(member.name), port, agreement, at);
  }
  return compareStateMember(
    analysis,
    held.rule,
    site,
    candidate.slot.name,
    declared,
    agreement,
    at,
  );
}

// --- a location -------------------------------------------------------------

function checkLocation(analysis: Analysis, candidate: LocationCandidate): Verdict {
  const instance = analysis.bindings.tensorInstances.find(
    (one) => one.identity === candidate.identity,
  );
  if (instance === undefined) {
    return verdict(
      [],
      `no parameter identity instance named '${candidate.identity}' was derived from this document`,
    );
  }
  const at: PathSegment[] = ['bindings', 'parameters', instance.rule, 'location'];
  const first = instance.members[0];
  const site = first === undefined ? undefined : analysis.resolved.get(keyOf(first.site));
  const slot =
    first === undefined || site === undefined
      ? null
      : optional(demand(site.definition, 'parameters'), pyStr(first.name), null);
  if (slot === null || site === undefined) {
    return verdict(
      [],
      `the identity instance '${candidate.identity}' has no resolved member, so a location has ` +
        'no slot to be written against',
    );
  }
  // "The location is written against the first resolved member's slot: one identity, one stored
  // shape (V15 is what makes the others equal to it)."
  const answer = evaluateLocation(
    candidate.location,
    instance.env,
    storageShape(slot),
    site.args,
    (expression, env) => modelValue(expression, analysis.quantities, env),
  );
  const problems: SemanticProblem[] = answer.problems.map((problem) =>
    semanticProblem('V17', `${candidate.identity}: ${problem}`, at),
  );
  if (answer.evaluated === null) return verdict(problems);
  // The names the document binds without this identity's own, and the candidate's beside them.
  const whole = new Map<string, string>();
  for (const [name, identity] of analysis.bindings.physical.whole) {
    if (identity !== candidate.identity) whole.set(name, identity);
  }
  const slices = new Map<string, PhysicalSlice[]>();
  for (const [name, regions] of analysis.bindings.physical.slices) {
    const kept = regions.filter((region) => region.identity !== candidate.identity);
    if (kept.length > 0) slices.set(name, [...kept]);
  }
  const before = new Set(slices.keys());
  problems.push(...bindPhysicalNames(candidate.identity, answer.evaluated, whole, slices, at));
  // Only the names the candidate touches: a refusal about a name it does not name is the
  // document's, not the candidate's.
  const touched = new Map<string, readonly PhysicalSlice[]>();
  for (const [name, regions] of slices) {
    if (!before.has(name) || regions.some((region) => region.identity === candidate.identity)) {
      touched.set(name, regions);
    }
  }
  problems.push(...physicalNameProblems(whole, touched, ['bindings', 'parameters']));
  return verdict(problems);
}
