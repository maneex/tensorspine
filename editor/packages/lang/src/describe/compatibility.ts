/**
 * "Compatible partners for tying and sharing" (plan §5.3): which identities a slot or a state
 * port may join.
 *
 * The inventory's §7: "**Whether an identity may take a member** — the core's compatibility list
 * (V15, V9), with the reason a candidate is excluded." The list is here; the reason is `check`'s
 * answer on that one candidate, in the tools' words. Both call the same two functions the
 * validator calls — {@link tyingProblems} for a parameter identity and {@link compareStateMember}
 * for a state identity — so a partner offered here is a partner the document would accept, and a
 * partner refused is refused with the line V15 or V9 would print.
 *
 * **Why a table of tokens.** A document declares up to 552 sites and as many identity instances,
 * and asking the two functions about every (slot, identity) pair is quadratic — well past the
 * 20 ms `describe` is given (§5.6). But neither verdict reads anything about a member beyond a
 * handful of facts: V15 reads the role, the stored shape and the sharing record, V9 the applying
 * rule, the key axes, the payload and the indexing domain. So the facts are encoded as a token,
 * one representative is kept per token, and the verdict is computed **by the validator's own
 * function over the representatives**, once per pair of tokens. The tables are small (a document
 * declares a dozen distinct slot signatures), the answer is exact, and there is still one
 * implementation of each rule.
 *
 * A token is Python's own reading of equality: {@link valueToken} writes `1`, `1.0` and `True`
 * alike because `==` and `hash` do, which is what `shapesAgree` and `sameValues` compare with.
 */
import { UNRESOLVED, type PyRecord, type PyValue } from '../expr/value.js';
import { demand, optional } from '../library/access.js';
import { pyStr } from '../library/repr.js';
import { comparePythonStrings } from '../schema/index.js';
import {
  compareStateMember,
  emptyAgreement,
  keyOf,
  portKeyOf,
  present,
  signatureOf,
  tyingProblems,
  valueToken,
  type Analysis,
  type IdentityMember,
  type ResolvedSite,
  type Signature,
  type StateAgreement,
} from '../validate/index.js';

import type {
  CompatibleIdentity,
  Mutable,
  SiteDescription,
  SlotDescription,
  StateDescription,
} from './site.js';

/**
 * The compatibility lists of every present slot and state port, filled into the descriptions.
 *
 * The identity a slot already belongs to is filled here too: `analyse` keys its identity
 * instances by their members, and a description is keyed by its site — this is the one walk that
 * relates the two.
 *
 * **`partners` is what a caller that does not need the lists turns off.** The identity a slot
 * *belongs to* is the cheap half — one pass over the identity instances, which a card's chip and
 * a sheet's row both show — and the lists are the walk feature 1.6d measured at 3.9–63.7 ms,
 * because the partners a slot may join are answered from every identity instance of the graph and
 * narrowing the sites does not narrow it. Feature 2.10 is where that mattered: the sheet needs
 * the facts on every keystroke (§5.4) and the lists only when "Tie to…" or "Share with…" opens a
 * menu (§4.11, feature 2.13), so the pipeline asks without them and the menu asks for the one
 * site with them. Nothing else changes: with `partners` false every other member of every
 * description is what it was, which `test/describe/describe.test.ts` asserts document by document.
 */
export function attachCompatibility(
  analysis: Analysis,
  sites: ReadonlyMap<string, Mutable<SiteDescription>>,
  partners = true,
): void {
  attachParameters(analysis, sites, partners);
  attachStates(analysis, sites, partners);
}

/** What both kinds of identity instance carry, of what the cheap half reads. */
interface HeldIdentity {
  readonly identity: string;
  readonly rule: string;
  readonly members: readonly IdentityMember[];
}

/** The identity each member of each instance belongs to, by `(site, slot)`: the cheap half. */
function membersOf(instances: readonly HeldIdentity[]): Map<string, CompatibleIdentity> {
  const found = new Map<string, CompatibleIdentity>();
  for (const instance of instances) {
    const identity: CompatibleIdentity = { identity: instance.identity, rule: instance.rule };
    for (const member of instance.members) found.set(portKeyOf(member.site, member.name), identity);
  }
  return found;
}

// --- parameters: V15 --------------------------------------------------------

/** The identity instances whose members carry one set of signatures: V15 answers them together. */
interface Grouped {
  /** The signature tokens of the members whose slot resolved, sorted and made unique. */
  readonly members: readonly string[];
  /** The same set as one token, which is half of the verdict memo's key. */
  readonly token: string;
  /** Every identity instance of the group, in the order the bindings emitted them. */
  readonly identities: CompatibleIdentity[];
}

function attachParameters(
  analysis: Analysis,
  sites: ReadonlyMap<string, Mutable<SiteDescription>>,
  partners: boolean,
): void {
  if (!partners) {
    const held = membersOf(analysis.bindings.tensorInstances);
    for (const [, description] of sites) {
      description.parameters = description.parameters.map((slot) => ({
        ...slot,
        identity: held.get(portKeyOf(description.key, slot.name))?.identity ?? null,
      }));
    }
    return;
  }
  const representative = new Map<string, Signature>();
  // One signature per (site, slot), whether the slot is reached as a member of an identity or as
  // a candidate for one: the token costs a stored shape's evaluation, and a document has as many
  // slots as members.
  const tokens = new Map<string, string | null>();
  /** The signature token of the slot one member names, or `null` where it names none. */
  const tokenOfMember = (site: ResolvedSite, name: PyValue): string | null => {
    const key = portKeyOf(site.key, name);
    const cached = tokens.get(key);
    if (cached !== undefined) return cached;
    const answer = signatureTokenOf(site, name);
    tokens.set(key, answer);
    return answer;
  };
  const signatureTokenOf = (site: ResolvedSite, name: PyValue): string | null => {
    const slot = optional(demand(site.definition, 'parameters'), pyStr(name), null);
    if (slot === null) return null;
    const signature = signatureOf(site, slot, name);
    const token = [
      valueToken(signature.role),
      shapeToken(signature.shape),
      factToken(signature.sharing),
    ].join('\u0002');
    if (!representative.has(token)) representative.set(token, signature);
    return token;
  };

  // Every identity instance, **grouped** by the set of signatures its members carry: the verdict
  // is the same for every identity of one group, so a slot answers a group at a time and the
  // walk is proportional to the answer rather than to the product of slots and identities.
  const groups = new Map<string, Grouped>();
  const memberOf = new Map<string, CompatibleIdentity>();
  for (const instance of analysis.bindings.tensorInstances) {
    const identity: CompatibleIdentity = { identity: instance.identity, rule: instance.rule };
    const tokens: string[] = [];
    for (const member of instance.members) {
      memberOf.set(portKeyOf(member.site, member.name), identity);
      const site = analysis.resolved.get(keyOf(member.site));
      if (site === undefined) continue;
      // An absent slot is no part of the identity's signature: the validator skips it before V15
      // reads it (`check_parameters`), so a group taken over it would answer for a shape the
      // identity does not have.
      const slot = optional(demand(site.definition, 'parameters'), pyStr(member.name), null);
      if (slot === null || !present(slot, site.args)) continue;
      const token = tokenOfMember(site, member.name);
      if (token !== null) tokens.push(token);
    }
    const distinct = [...new Set(tokens)].sort();
    const token = distinct.join('\u0001');
    const held = groups.get(token);
    if (held === undefined) groups.set(token, { members: distinct, token, identities: [identity] });
    else held.identities.push(identity);
  }

  const pairs = new Map<string, boolean>();
  /** `tyingProblems` over the two representatives: V15's own verdict on one pair of signatures. */
  const pairOk = (one: string, other: string): boolean => {
    const key = one <= other ? `${one}\u0003${other}` : `${other}\u0003${one}`;
    const held = pairs.get(key);
    if (held !== undefined) return held;
    const signatures = [representative.get(one) as Signature, representative.get(other) as Signature];
    const ok = tyingProblems('', signatures, []).length === 0;
    pairs.set(key, ok);
    return ok;
  };

  const verdicts = new Map<string, boolean>();
  const accepts = (slot: string, group: Grouped): boolean => {
    // An identity with no member runs no tying check at all: the tools call `checkTying` only
    // where a second signature joined the first.
    if (group.members.length === 0) return true;
    const key = `${slot}\u0004${group.token}`;
    const held = verdicts.get(key);
    if (held !== undefined) return held;
    const ok = group.members.every((member) => pairOk(slot, member));
    verdicts.set(key, ok);
    return ok;
  };

  for (const [, description] of sites) {
    const site = analysis.resolved.get(keyOf(description.key));
    if (site === undefined) continue;
    description.parameters = description.parameters.map((slot): SlotDescription => {
      const held = memberOf.get(portKeyOf(description.key, slot.name)) ?? null;
      if (!slot.present) return { ...slot, identity: held?.identity ?? null };
      const token = tokenOfMember(site, slot.name) as string;
      const partners: CompatibleIdentity[] = [];
      for (const [, group] of groups) {
        if (!accepts(token, group)) continue;
        for (const identity of group.identities) if (identity !== held) partners.push(identity);
      }
      return { ...slot, identity: held?.identity ?? null, tiesWith: partners };
    });
  }
}

// --- states: V9 -------------------------------------------------------------

function attachStates(
  analysis: Analysis,
  sites: ReadonlyMap<string, Mutable<SiteDescription>>,
  partners: boolean,
): void {
  if (!partners) {
    const held = membersOf(analysis.bindings.stateInstances);
    for (const [, description] of sites) {
      description.states = description.states.map((state) => ({
        ...state,
        identity: held.get(portKeyOf(description.key, state.name))?.identity ?? null,
      }));
    }
    return;
  }
  /** One state port, as a candidate member: what {@link compareStateMember} would read of it. */
  interface Port {
    readonly site: ResolvedSite;
    readonly name: string;
    readonly port: PyValue;
  }
  const candidates = new Map<string, Port>();
  const tokens = new Map<string, string | null>();
  /** The V9 facts of one state port, computed once per `(site, port)`. */
  const tokenOfPort = (site: ResolvedSite, name: PyValue): string | null => {
    const key = portKeyOf(site.key, name);
    const cached = tokens.get(key);
    if (cached !== undefined) return cached;
    const answer = portTokenOf(site, name);
    tokens.set(key, answer);
    return answer;
  };
  const portTokenOf = (site: ResolvedSite, name: PyValue): string | null => {
    const port = optional(demand(site.definition, 'state_ports'), pyStr(name), null);
    if (port === null) return null;
    // The facts V9 compares, taken through the validator's own comparison: run it against a
    // fresh agreement, which *settles* the four readings without refusing anything.
    const agreement = emptyAgreement();
    compareStateMember(analysis, '', site, pyStr(name), port, agreement, []);
    const token = agreementToken(agreement);
    if (!candidates.has(token)) candidates.set(token, { site, name: pyStr(name), port });
    return token;
  };

  // Every state identity instance, with the agreement its members settled — `analyse`'s own four
  // locals after its member loop, which is what the next member is compared against — **grouped**
  // by that agreement, since it is all the next member is judged against.
  const groups = new Map<string, { agreement: StateAgreement; identities: CompatibleIdentity[] }>();
  const memberOf = new Map<string, CompatibleIdentity>();
  for (const instance of analysis.bindings.stateInstances) {
    const identity: CompatibleIdentity = { identity: instance.identity, rule: instance.rule };
    const agreement = emptyAgreement();
    for (const member of instance.members) {
      memberOf.set(portKeyOf(member.site, member.name), identity);
      const site = analysis.resolved.get(keyOf(member.site));
      if (site === undefined) continue;
      const port = optional(demand(site.definition, 'state_ports'), pyStr(member.name), null);
      if (port === null) continue;
      // And the absent state port, which `check_states` skips before V9 settles anything from it.
      if (!present(port, site.args)) continue;
      compareStateMember(analysis, '', site, pyStr(member.name), port, agreement, []);
    }
    const token = agreementToken(agreement);
    const held = groups.get(token);
    if (held === undefined) groups.set(token, { agreement, identities: [identity] });
    else held.identities.push(identity);
  }

  const verdicts = new Map<string, boolean>();
  const accepts = (candidate: string, group: string): boolean => {
    const key = `${candidate}\u0004${group}`;
    const held = verdicts.get(key);
    if (held !== undefined) return held;
    const one = candidates.get(candidate) as Port;
    const reference = (groups.get(group) as { agreement: StateAgreement }).agreement;
    const ok =
      compareStateMember(
        analysis,
        '',
        one.site,
        one.name,
        one.port,
        { ...reference },
        [],
      ).length === 0;
    verdicts.set(key, ok);
    return ok;
  };

  for (const [, description] of sites) {
    const site = analysis.resolved.get(keyOf(description.key));
    if (site === undefined) continue;
    description.states = description.states.map((state): StateDescription => {
      const held = memberOf.get(portKeyOf(description.key, state.name)) ?? null;
      if (!state.present) return { ...state, identity: held?.identity ?? null };
      const token = tokenOfPort(site, state.name) as string;
      const partners: CompatibleIdentity[] = [];
      for (const [group, entry] of groups) {
        if (!accepts(token, group)) continue;
        for (const identity of entry.identities) if (identity !== held) partners.push(identity);
      }
      return { ...state, identity: held?.identity ?? null, sharesWith: partners };
    });
  }
}

// --- tokens -----------------------------------------------------------------

/** The four readings V9 compares, as one token: `None` where a member left one open. */
function agreementToken(agreement: StateAgreement): string {
  return [
    agreement.keyAxes === null ? 'None' : factToken(agreement.keyAxes),
    agreement.payload === null
      ? 'None'
      : agreement.payload
          .map((one) => [one.name, valueToken(one.role), shapeToken(one.shape)].join('\u0002'))
          .join('\u0001'),
    agreement.ruleText === null ? 'None' : JSON.stringify(agreement.ruleText),
    agreement.indexing === null
      ? 'None'
      : `${valueToken(agreement.indexing[0])}\u0002${valueToken(agreement.indexing[1])}`,
  ].join('\u0003');
}

/** A shape identity as one token: the pairs V4 and V15 compare, in order. */
function shapeToken(shape: readonly (readonly [PyValue, PyValue])[]): string {
  return shape.map((axis) => `${factToken(axis[0])}\u0002${factToken(axis[1])}`).join('\u0001');
}

/**
 * A value as a token, containers included.
 *
 * {@link valueToken} is Python's own hash-and-equality reading of a scalar and raises on a list or
 * a record, as Python raises on an unhashable key; the comparisons these tables stand for read
 * containers element by element with `==`, so this walks them instead of raising.
 */
function factToken(value: PyValue): string {
  if (Array.isArray(value)) {
    return `[${(value as readonly PyValue[]).map((one) => factToken(one)).join(',')}]`;
  }
  if (value !== null && value !== UNRESOLVED && typeof value === 'object') {
    const record = value as PyRecord;
    const names = Object.keys(record).sort(comparePythonStrings);
    return `{${names
      .map((name) => `${JSON.stringify(name)}:${factToken(record[name] as PyValue)}`)
      .join(',')}}`;
  }
  return valueToken(value);
}
