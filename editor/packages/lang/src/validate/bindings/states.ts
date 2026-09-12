/**
 * The state bindings: `analyse`'s `# --- V7/V9/V14/V16: states ---` block, the V20 block that
 * follows it, and the V18 block that closes the bindings.
 *
 * A state identity is "several members under one identity, nothing else" (§3.4): the primitive
 * derives the descriptor, the graph declares which instances name the same storage. What is
 * checked here:
 *
 * - **V9** — "a state identity connects only compatible ports: the same applicable rule, the same
 *   key axes, and equal payload shapes and indexing domains; a present state port that no rule
 *   matches is rejected". The rules are ordered and "the first matching rule wins" (§4.3).
 * - **V7** — every present state port is bound exactly once, and a port absent by its condition
 *   is bound by nothing.
 * - **V14** — a dtype selected for a state identity is admissible "for the role of every payload
 *   component of every member".
 * - **V16** — "an instance whose state is carried across fragments (its primitive's
 *   `carried_across` condition holds) sits on a fragmented stream".
 * - **V20** — "a state identity instance has exactly one writer among its present members: the
 *   member whose port's `written_when` holds, or every member when the port declares none".
 * - **V18** — "an instance whose primitive reads across positions (§4.1), on a fragmented stream,
 *   carries a state across the fragments of that stream (§5.3)".
 *
 * Beside the refusals it answers the two facts §4.4 calls derived and the document never declares:
 * the **instance key** of each identity instance — "the identity's indices × the primitive's
 * `key_axes`" — and **what is carried**, which "follows from the primitive's carrying condition
 * and the input's fragmentation" and is declared nowhere. The third answer is an *advisory*: "a
 * self-indexed state that is not carried is reset at each fragment of its stream", which is valid
 * and worth a second look, so `--lint` prints it and `--validate` does not.
 */
import { pyEqual } from '../../expr/arithmetic.js';
import type { Env } from '../../expr/model.js';
import { primitiveCondition } from '../../expr/primitive.js';
import { truthy, type PyRecord, type PyValue } from '../../expr/value.js';
import { demand, entries, has, listOf, optional } from '../../library/access.js';
import { pyRepr, pyStr } from '../../library/repr.js';
import { comparePythonStrings } from '../../schema/index.js';
import type { PathSegment } from '../../schema/types.js';

import {
  keyOf,
  portKeyOf,
  present,
  reprTuple,
  segmentsOfSite,
  shapeIdentity,
  shapesAgree,
  whereOfSite,
  type Indexing,
  type PortDomain,
  type ResolvedSite,
  type ShapeIdentity,
  type SiteKey,
} from '../graph.js';
import { semanticProblem, type SemanticProblem } from '../problems.js';
import type { IdentityMember } from './analysis.js';
import type { Bindings } from './context.js';
import { dtypeValues, slotKeyOf } from './parameters.js';

/** One payload component as V9 compares them: `(name, role, shape)`, sorted by name. */
export interface Component {
  readonly name: string;
  readonly role: PyValue;
  readonly shape: ShapeIdentity;
}

/**
 * What the members of one state identity must agree on (V9): the four readings, each open until
 * a member settles it.
 *
 * "One identity, one instance key, one payload, one rule, one stream." The tools hold the four in
 * four locals of `analyse`'s state loop and test each with `is None`, so a member that settles one
 * to `None` leaves it open for the next; this record is those four locals, named, so that the
 * comparison is one function — the walk's, and feature 1.6d's compatibility list's.
 */
export interface StateAgreement {
  /** The port's `key_axes`, as the first member that named them wrote them. */
  keyAxes: PyValue[] | null;
  /** The payload components, sorted by name, with their evaluated shapes. */
  payload: Component[] | null;
  /** The applying rule, encoded as `json.dumps(rule, sort_keys=True)`. */
  ruleText: string | null;
  /** The indexing domain the state grows along. */
  indexing: Indexing | null;
}

/** An agreement no member has settled yet: `analyse`'s four `None`s. */
export function emptyAgreement(): StateAgreement {
  return { keyAxes: null, payload: null, ruleText: null, indexing: null };
}

/** What {@link streamOf} reads: the graph's own domains and its `(site, port)` domains. */
export interface StreamSource {
  readonly own: ReadonlyMap<string, Indexing | null>;
  readonly domains: ReadonlyMap<string, PortDomain>;
}

/**
 * One member of a state identity against what the members before it settled (V9).
 *
 * The four comparisons of `analyse`'s state loop, in its order and with its words, and the
 * agreement grows as the loop's four locals grow. It answers the refusals rather than appending
 * them so that feature 1.6d can ask what a *candidate* member would be refused with — the
 * compatibility list of a state port is this function over the members an identity already has,
 * then over the candidate, and nothing else decides it.
 */
export function compareStateMember(
  graph: StreamSource,
  rule: string,
  site: ResolvedSite,
  portName: string,
  port: PyValue,
  agreement: StateAgreement,
  at: readonly PathSegment[],
): SemanticProblem[] {
  const problems: SemanticProblem[] = [];
  const applying = applicableRule(port, site.args);
  if (applying === null) {
    problems.push(
      semanticProblem(
        'V9',
        `state ${rule}: no rule of ${pyStr(site.primitive)}.${portName} applies to these arguments`,
        at,
      ),
    );
  }
  const axes = listOf(demand(port, 'key_axes'));
  if (agreement.keyAxes === null) {
    agreement.keyAxes = axes;
  } else if (!sameValues(axes, agreement.keyAxes)) {
    problems.push(
      semanticProblem(
        'V9',
        `state ${rule}: members keyed on ${pyRepr(agreement.keyAxes)} and ` +
          `${pyStr(demand(port, 'key_axes'))} cannot share one allocation`,
        at,
      ),
    );
  }
  const shapes = componentsOf(port, site.args);
  if (agreement.payload === null) {
    agreement.payload = shapes;
  } else if (!samePayload(shapes, agreement.payload)) {
    problems.push(
      semanticProblem(
        'V9',
        `state ${rule}: members with different payloads cannot share one allocation`,
        at,
      ),
    );
  }
  // `json.dumps(rule, sort_keys=True) if rule else None`: Python's truthiness, so a rule
  // the grammar could not produce — an empty record — leaves the reading open too.
  const text = applying === null || !truthy(applying) ? null : ruleTokenOf(applying);
  if (agreement.ruleText === null) {
    agreement.ruleText = text;
  } else if (text !== agreement.ruleText) {
    problems.push(
      semanticProblem('V9', `state ${rule}: members under different derivation rules`, at),
    );
  }
  if (applying !== null) {
    const stream = streamOf(graph, site.key, applying);
    if (agreement.indexing === null) {
      agreement.indexing = stream;
    } else if (!sameIndexing(stream, agreement.indexing)) {
      problems.push(
        semanticProblem('V9', `state ${rule}: members indexed by different streams`, at),
      );
    }
  }
  return problems;
}

/** The state half of the bindings stage, in the order `analyse` walks it. */
export function checkStates(bindings: Bindings): void {
  const { stage } = bindings;
  const declared = demand(demand(stage.model, 'bindings'), 'states');
  const policy = stage.library.precision;
  let stateIdentities = 0;

  for (const [rule, binding] of entries(declared)) {
    const at: PathSegment[] = ['bindings', 'states', rule];
    let values = dtypeValues(stage.model, optional(binding, 'dtype', null));
    if (typeof values === 'string') {
      bindings.fail(
        'V14',
        `${rule}: dtype selector is ${values.toLowerCase().replaceAll('_', ' ')}`,
        [...at, 'dtype'],
      );
      values = null;
    }
    for (const env of bindings.loopEnvs(binding, rule, at)) {
      // The position travels with the member (`IdentityMember.at`): the tools read the list and
      // keep the values, and a gesture that takes a slot out of an identity has to name the item
      // it removes. Named `position` here because `at` is already this block's path.
      const members: IdentityMember[] = listOf(demand(binding, 'members')).map((member, position) => ({
        site: bindings.select(demand(member, 'instance'), env),
        name: demand(member, 'state'),
        at: position,
      }));
      if (members.some((member) => stage.absent.has(keyOf(member.site)))) continue; // §5.2 rule 3
      stateIdentities += 1;
      const identity = demand(binding, 'identity');
      bindings.stateInstances.push({
        identity: bindings.instanceName(identity, env),
        // `tid if False else sid`: the tools write the state binding's own name; the dead
        // conditional beside it is the parameter block's `tid`, and it never wins.
        rule,
        members: members.filter((member) => stage.resolved.has(keyOf(member.site))),
        dtype: optional(binding, 'dtype', null),
        indices: has(identity, 'indices')
          ? listOf(demand(identity, 'indices'))
              .map((one) => pyStr(one))
              .sort(comparePythonStrings)
          : [],
        writer: null,
      });
      // "V9: one identity, one instance key, one payload, one rule, one stream." Each of the four
      // is `None` until a member settles it, and a member that settles it to `None` leaves it open
      // — the tools test `is None`, so a state with no applicable rule does not fix the reading.
      const agreement = emptyAgreement();

      for (const member of members) {
        const site = stage.resolved.get(keyOf(member.site));
        if (site === undefined) {
          bindings.fail(
            'V1',
            `state ${rule}: instance does not exist ${whereOfSite(member.site)}`,
            at,
          );
          continue;
        }
        const name = site.primitive;
        const portName = pyStr(member.name);
        const port = optional(demand(site.definition, 'state_ports'), portName, null);
        if (port === null) {
          bindings.fail('V1', `state ${rule}: ${pyStr(name)} has no state port '${portName}'`, at);
          continue;
        }
        if (!truthy(primitiveCondition(demand(port, 'present_when'), site.args))) {
          bindings.fail('V7', `state ${rule}: port '${portName}' absent for these arguments`, at);
          continue;
        }
        const bound = bindings.stateSlots.get(slotKeyOf(member));
        if (bound !== undefined) {
          bindings.fail('V7', `state port ${pyStr(name)}.${portName} bound twice`, at);
        }
        bindings.stateSlots.set(slotKeyOf(member), { ...member, rule });
        for (const problem of compareStateMember(stage, rule, site, portName, port, agreement, at)) {
          stage.problems.push(problem);
        }
        if (values !== null && values.length > 0) {
          for (const [, component] of entries(demand(port, 'payload'))) {
            const admissible = policy.get(pyStr(demand(component, 'role')));
            const bad =
              admissible === undefined
                ? []
                : values.filter((one) => !contains(demand(admissible, 'admissible'), one));
            if (bad.length > 0 && admissible !== undefined) {
              bindings.fail(
                'V14',
                `${rule}: precision ${pyRepr(bad)} outside the admissible set of ` +
                  `role '${pyStr(demand(component, 'role'))}' ` +
                  `${pyStr(demand(admissible, 'admissible'))}`,
                [...at, 'dtype'],
              );
            }
            bindings.checked += 1n;
          }
        }
      }
      if (agreement.keyAxes !== null) {
        // `sid + (f"{env}" if env else "")`, and `tuple(indices) + key_axes`: the identity's own
        // index names in document order, then the port's key axes (§4.4).
        const own = has(identity, 'indices')
          ? listOf(demand(identity, 'indices')).map((one) => pyStr(one))
          : [];
        bindings.instanceKeys.set(rule + (env.size === 0 ? '' : reprEnv(env)), [
          ...own,
          ...agreement.keyAxes,
        ]);
      }
    }
  }
  stage.stats.set('precisions_checked', bindings.checked);
  checkCarrying(bindings);
  checkWriters(bindings);
  stage.stats.set('state_slots', BigInt(bindings.stateSlots.size));
  stage.stats.set('state_identities', BigInt(stateIdentities));
  checkAcrossPositions(bindings);
}

/**
 * What every present state port of every instance carries, and what it does not (§5.3).
 *
 * "A state whose carrying condition holds survives between the invocations that deliver
 * successive fragments of its stream"; "a state indexed by a source stream grows along that
 * stream, across its fragments"; "a self-indexed state that is not carried is reset at each
 * fragment of its stream" — the first is V16's obligation, the second is carried by definition,
 * and the third is the advisory. The unbound-port half of V7 is decided in the same walk.
 */
function checkCarrying(bindings: Bindings): void {
  const { stage } = bindings;
  for (const [id, site] of stage.resolved) {
    const at = segmentsOfSite(site.key);
    const where = whereOfSite(site.key);
    for (const [portName, port] of entries(demand(site.definition, 'state_ports'))) {
      if (!truthy(primitiveCondition(demand(port, 'present_when'), site.args))) continue;
      const member: IdentityMember = { site: site.key, name: portName };
      if (!bindings.stateSlots.has(slotKeyOf(member))) {
        bindings.fail(
          'V7',
          `unbound state port: ${pyStr(site.primitive)}@${where}.${portName}`,
          at,
        );
      }
      const carriedAcross = optional(port, 'carried_across', null);
      const applying = applicableRule(port, site.args);
      // "the state's own stream (§4.3): the instance's, or the port's it is indexed by"
      const mine =
        applying === null || has(demand(applying, 'indexed_by'), 'self')
          ? (stage.own.get(id) ?? null)
          : streamOf(stage, site.key, applying);
      const byPort = applying !== null && has(demand(applying, 'indexed_by'), 'port');
      const onFragment = mine !== null && bindings.stage.fragmented.has(pyStr(mine[1]));
      const held = bindings.stateSlots.get(slotKeyOf(member));
      const under = held === undefined ? where : held.rule;
      if (truthy(carriedAcross) && truthy(primitiveCondition(demand(carriedAcross, 'when'), site.args))) {
        if (!onFragment) {
          bindings.fail(
            'V16',
            `${pyStr(site.primitive)}@${where}.${portName}: carried across fragments, but its ` +
              `stream ${mine === null ? 'None' : reprTuple(mine)} is not a fragmented input`,
            at,
          );
        }
        if (!bindings.carried.has(under)) bindings.carried.set(under, mine);
        if (mine !== null) addCarried(bindings, id, mine);
      } else if (byPort && onFragment) {
        // "indexed by a port whose stream is fragmented: carried by definition (§5.3)"
        if (!bindings.carried.has(under)) bindings.carried.set(under, mine);
        addCarried(bindings, id, mine);
      } else if (
        onFragment &&
        applying !== null &&
        has(demand(applying, 'indexed_by'), 'self')
      ) {
        stage.advisories.push(
          `${pyStr(site.primitive)}@${where}.${portName}: a self-indexed state on the ` +
            `fragmented stream '${pyStr(mine[1])}' that is not carried ` +
            '— reset at every fragment',
        );
      }
    }
  }
}

/** V20: "exactly one member writes an identity, the others read it" (§4.3 writing, §4.4). */
function checkWriters(bindings: Bindings): void {
  const { stage } = bindings;
  for (const instance of bindings.stateInstances) {
    const writers: IdentityMember[] = [];
    for (const member of instance.members) {
      const site = stage.resolved.get(keyOf(member.site));
      if (site === undefined) continue;
      const port = optional(demand(site.definition, 'state_ports'), pyStr(member.name), null);
      if (port === null || !truthy(primitiveCondition(demand(port, 'present_when'), site.args))) {
        continue;
      }
      if (
        !has(port, 'written_when') ||
        truthy(primitiveCondition(demand(port, 'written_when'), site.args))
      ) {
        writers.push(member);
      }
    }
    if (writers.length !== 1) {
      bindings.fail(
        'V20',
        `state ${instance.identity}: ${writers.length} writer(s) among ` +
          `${instance.members.length} member(s) — exactly one member writes an identity, the ` +
          'others read it',
        ['bindings', 'states', instance.rule],
      );
    }
    instance.writer = writers[0] ?? null;
  }
}

/**
 * V18: "reading across positions of a fragmented stream needs a state carried across it".
 *
 * "The fragments would not compute what the whole stream does": carrying is necessary, and the
 * sufficient condition is the primitive's own invariant (§5.3, V8).
 */
function checkAcrossPositions(bindings: Bindings): void {
  const { stage } = bindings;
  for (const [id, site] of stage.resolved) {
    const across = optional(optional(site.definition, 'effects', {}), 'across_positions', null);
    if (!truthy(across) || !truthy(primitiveCondition(demand(across, 'when'), site.args))) continue;
    for (const [portName, port] of entries(demand(demand(site.definition, 'ports'), 'inputs'))) {
      if (!present(port, site.args)) continue;
      const domain = stage.domains.get(portKeyOf(site.key, portName))?.domain;
      if (domain === undefined) continue;
      const stream = pyStr(domain[1]);
      if (!stage.fragmented.has(stream)) continue;
      if (!(bindings.carriedOn.get(id) ?? new Set<string>()).has(stream)) {
        bindings.fail(
          'V18',
          `${pyStr(site.primitive)}@${whereOfSite(site.key)} reads across positions of the ` +
            `fragmented stream '${stream}' (port '${portName}') and carries no state across its ` +
            'fragments: the fragments would not compute what the whole stream does',
          segmentsOfSite(site.key),
        );
      }
    }
  }
}

/** `applicable[0] if applicable else None`: "rules are ordered; the first matching rule wins". */
export function applicableRule(port: PyValue, args: PyRecord): PyValue | null {
  for (const rule of listOf(demand(port, 'rules'))) {
    if (truthy(primitiveCondition(demand(rule, 'when'), args))) return rule;
  }
  return null;
}

/** The stream a rule indexes its state by: the instance's own, or one of its input ports'. */
export function streamOf(graph: StreamSource, site: SiteKey, rule: PyValue): Indexing | null {
  const indexedBy = demand(rule, 'indexed_by');
  if (has(indexedBy, 'self')) return graph.own.get(keyOf(site)) ?? null;
  return graph.domains.get(portKeyOf(site, demand(indexedBy, 'port')))?.domain ?? null;
}

/** `carried_on[key].add(mine[1])`: the streams an instance's states carry across fragments. */
function addCarried(bindings: Bindings, id: string, mine: Indexing): void {
  const held = bindings.carriedOn.get(id);
  if (held === undefined) bindings.carriedOn.set(id, new Set([pyStr(mine[1])]));
  else held.add(pyStr(mine[1]));
}

/** `tuple(sorted((c, comp['role'], _shape_identity(comp['shape'], args)) …))` over a payload. */
export function componentsOf(port: PyValue, args: PyRecord): Component[] {
  return entries(demand(port, 'payload'))
    .map((entry): Component => ({
      name: entry[0],
      role: demand(entry[1], 'role'),
      shape: shapeIdentity(demand(entry[1], 'shape'), args),
    }))
    .sort((one, other) => comparePythonStrings(one.name, other.name));
}

/** Two payloads as Python's `==` reads the two tuples. */
function samePayload(one: readonly Component[], other: readonly Component[]): boolean {
  if (one.length !== other.length) return false;
  return one.every((component, index) => {
    const theirs = other[index] as Component;
    return (
      component.name === theirs.name &&
      pyEqual(component.role, theirs.role) &&
      shapesAgree(component.shape, theirs.shape)
    );
  });
}

/** Two key-axis lists as Python's `==` reads the two tuples. */
function sameValues(one: readonly PyValue[], other: readonly PyValue[]): boolean {
  return one.length === other.length && one.every((value, index) => pyEqual(value, other[index] as PyValue));
}

/** Two indexing domains, either of which may be `None`, as Python's `!=` reads them. */
function sameIndexing(one: Indexing | null, other: Indexing | null): boolean {
  if (one === null || other === null) return one === other;
  return pyEqual(one[0], other[0]) && pyEqual(one[1], other[1]);
}

/** `v in container`, by Python's `==`. */
function contains(container: PyValue, value: PyValue): boolean {
  return listOf(container).some((one) => pyEqual(one, value));
}

/** `f"{env}"`: the index environment written as Python writes a dictionary. */
function reprEnv(env: Env): string {
  return `{${[...env].map(([name, one]) => `${pyRepr(name)}: ${pyRepr(one)}`).join(', ')}}`;
}

/**
 * `json.dumps(rule, sort_keys=True)`: the text two members' rules are compared as.
 *
 * Only equality matters, but the encoding is the tools' own so that no two rules the tools tell
 * apart are conflated here: `true` is not `1`, `1.0` is not `1` — the distinction feature 1.2
 * keeps — and a name is written as CPython's encoder writes it.
 */
function ruleTokenOf(rule: PyValue): string {
  return sortedJson(rule);
}

/** `json.dumps(value, sort_keys=True)` over the values a unit file can hold. */
function sortedJson(value: PyValue): string {
  if (value === null) return 'null';
  if (value === true) return 'true';
  if (value === false) return 'false';
  if (typeof value === 'bigint') return value.toString();
  if (typeof value === 'number') return pythonJsonNumber(value);
  if (typeof value === 'string') return JSON.stringify(value);
  if (Array.isArray(value)) {
    return `[${(value as readonly PyValue[]).map((one) => sortedJson(one)).join(', ')}]`;
  }
  if (typeof value === 'object') {
    const record = value as Record<string, PyValue>;
    const names = Object.keys(record).sort(comparePythonStrings);
    return `{${names
      .map((name) => `${JSON.stringify(name)}: ${sortedJson(record[name] as PyValue)}`)
      .join(', ')}}`;
  }
  // `json.dumps` of the sentinel raises; a rule read out of a loaded unit never holds one.
  throw new TypeError('Object of type object is not JSON serializable');
}

/** `float.__repr__` as `json.dumps` writes it, which is what feature 0.3's formatter answers. */
function pythonJsonNumber(value: number): string {
  if (Number.isNaN(value)) return 'NaN';
  if (value === Infinity) return 'Infinity';
  if (value === -Infinity) return '-Infinity';
  return pyRepr(value);
}
