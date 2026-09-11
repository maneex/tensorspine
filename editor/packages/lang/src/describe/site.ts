/**
 * What one site of a document is, as the editor asks it (plan §5.3 `describe`, §4.7, §4.11).
 *
 * The inventory's §7 is a list of questions a component must ask rather than compute — "which
 * ports, slots and states exist", "whether an argument applies", "what a shape evaluates to",
 * "whether an identity may take a member" — and this module answers the first three of them for
 * one site, from the analysis the validator already made.
 *
 * **Nothing here is a second reading of a rule.** Presence is `_present` ({@link present}), a
 * shape is `_shape` over the declaration ({@link describeShape}), the arguments are the facts
 * feature 1.6a records *inside* `resolve_arguments`' walk (`ArgumentDescription`, composed here
 * and never copied — "those facts cannot be read off `resolve_arguments`' answer", which is why
 * they travel with the resolved site), the applying state rule is `applicable[0] if applicable`
 * ({@link applicableRule}), the applying partition and cost entries are the same filter
 * `derive.d6` and `derive.d5` apply, and the indexing domains, the producers and the consumers
 * are the maps the graph half of `analyse` filled.
 *
 * **What is evaluated, and what is not.** A shape is evaluated where the element it belongs to is
 * *present*: an absent port's extents are expressions over arguments that may not exist, and the
 * tools evaluate none of them, so neither does this — an absent element is answered with
 * `present: false` and no shape, which is what §4.7 draws ("a port absent by its condition is not
 * drawn") and what the fix action of plan §3 needs ("the bindings that name absent slots or ports
 * as Problems"). Every declared element is listed all the same, present or not: a component that
 * had to ask "does this slot still exist?" by re-reading `present_when` is the shortcut §7 forbids.
 */
import { primitiveCondition, primitiveValue } from '../expr/primitive.js';
import { truthy, type PyValue } from '../expr/value.js';
import { demand, entries, has, listOf, optional } from '../library/access.js';
import { pyStr } from '../library/repr.js';
import { comparePythonStrings } from '../schema/index.js';
import type { PathSegment } from '../schema/types.js';
import type { ArgumentDescription } from '../validate/arguments.js';
import {
  keyOf,
  portKeyOf,
  present,
  segmentsOfSite,
  whereOfSite,
  type Analysis,
  type Indexing,
  type InterfacePorts,
  type ResolvedSite,
  type SiteKey,
} from '../validate/index.js';
import { applicableRule, storageShape } from '../validate/index.js';

import { describeShape, type DescribedShape } from './shape.js';

/** One value port of an instance, as a handle and as a sheet row show it (§4.7, §4.11). */
export interface PortDescription {
  /** The port's name, as the primitive declares it. */
  readonly name: string;
  /** Which side it is on; the two are answered in separate lists, and this says which. */
  readonly side: 'inputs' | 'outputs';
  /** `_present`: whether the port exists under these arguments. */
  readonly present: boolean;
  /** The precision role the port declares. */
  readonly role: PyValue;
  /** The indexing kind the port declares — a kind of §5.3, or `inherit`. */
  readonly kind: PyValue;
  /** The declared `from` of the port's domain (`self`, or another port); `null` when it declares none. */
  readonly from: PyValue | null;
  /** The domain the graph resolved for this port: the pair (kind, stream), or `null` where V5 left it open. */
  readonly domain: Indexing | null;
  /** The evaluated shape; `null` when the port declares none, and when the port is absent. */
  readonly shape: DescribedShape | null;
  /**
   * What feeds this input port — a value binding's name, or `input:<name>` for a public input —
   * and `null` when nothing does (V7's unfed handle) or when the port is an output.
   */
  readonly fedBy: string | null;
  /** Whether an edge or a public output consumes this output port (V13); false on an input. */
  readonly consumed: boolean;
  /** The declaration itself, for a sheet that shows its `description` and its own fields. */
  readonly declared: PyValue;
}

/** One parameter or constant slot of an instance (§3.4, §4.11). */
export interface SlotDescription {
  readonly name: string;
  /** Which map of the primitive declares it. */
  readonly kind: 'parameter' | 'constant';
  /** `_present`: whether the slot exists under these arguments. */
  readonly present: boolean;
  readonly role: PyValue;
  /** A parameter's `sharing` record (`kind`, `roles`); `null` for a constant, which declares none. */
  readonly sharing: PyValue | null;
  /** The declared multiplicity, evaluated; `null` when the slot declares none. */
  readonly multiplicity: PyValue | null;
  /**
   * The shape **as stored**: the storage axis of a declared multiplicity before the slot's own
   * axes (§3.4). `null` while the slot is absent — nothing addresses an absent slot.
   */
  readonly shape: DescribedShape | null;
  /** The binding that bound this slot, or `null` when nothing did (V7's unbound chip). */
  readonly boundBy: string | null;
  /** The identity instance the slot is a member of: `wq[layer=3]`, or `null` while it is unbound. */
  readonly identity: string | null;
  /**
   * The parameter identity instances this slot may join (V15) — empty on a constant slot.
   *
   * "Compatible partners for tying" (§5.3): what the sheet's "Tie to…" lists and what a chip
   * dragged onto another chip is proposed from (§4.7). The *reason* a candidate is absent from
   * the list is `check`'s answer on that candidate, in the tools' words.
   */
  readonly tiesWith: readonly CompatibleIdentity[];
  readonly declared: PyValue;
}

/** One payload component of a state port, with its shape evaluated. */
export interface StateComponent {
  readonly name: string;
  readonly role: PyValue;
  readonly shape: DescribedShape;
}

/** One state port of an instance, with the rule that applies to its arguments (§4.3, §4.11). */
export interface StateDescription {
  readonly name: string;
  /** `_present`: whether the port exists under these arguments. */
  readonly present: boolean;
  /**
   * The rule that applies — "rules are ordered; the first matching rule wins" — as the unit
   * declares it, and `null` when no rule matches (which V9 refuses).
   */
  readonly rule: PyValue | null;
  /** Its position in the port's `rules`, so a sheet can point at the declaration; `null` with no rule. */
  readonly ruleIndex: number | null;
  /** The rule's evolution, access geometry and sharing granularity; `null` with no rule. */
  readonly evolution: PyValue | null;
  readonly access: PyValue | null;
  readonly sharing: PyValue | null;
  /** Whether the rule indexes the state by one of the instance's input ports rather than by itself. */
  readonly indexedBySource: boolean;
  /** The port it is indexed by, or `null` for `self` and for no rule. */
  readonly indexedByPort: PyValue | null;
  /** The rule's `span` and `stride`, evaluated; `null` where the rule declares none. */
  readonly span: PyValue | null;
  readonly stride: PyValue | null;
  /** The stream the state grows along: the instance's own, or the port's it is indexed by. */
  readonly stream: Indexing | null;
  /** The instance axes the port is keyed on; with the identity's indices they form the instance key. */
  readonly keyAxes: readonly PyValue[];
  /** The payload per cached position, component by component, with the shapes evaluated. */
  readonly payload: readonly StateComponent[];
  /** The effects the port admits, as D4 lists them: the set of `effect`s, sorted. */
  readonly operations: readonly PyValue[];
  /** Whether this instance writes the port — `written_when` holds, or the port declares none (V20). */
  readonly written: boolean;
  /** Whether the primitive's own carrying condition holds for these arguments (§5.3, V16). */
  readonly carriedAcross: boolean;
  readonly boundBy: string | null;
  readonly identity: string | null;
  /** The state identity instances this port may share (V9): "compatible partners for sharing". */
  readonly sharesWith: readonly CompatibleIdentity[];
  readonly declared: PyValue;
}

/** One identity instance a slot or a state port may join. */
export interface CompatibleIdentity {
  /** The identity instance's name: `tied_embeddings`, `wq[layer=3]`. */
  readonly identity: string;
  /** The binding that declares it, which is the member list a gesture would extend. */
  readonly rule: string;
}

/** One applying partition option of an instance (§4.4, D6). */
export interface PartitionDescription {
  /** Its position in the primitive's `partition_options`. */
  readonly index: number;
  readonly target: PyValue;
  /** The communications it admits, always as a list — `derive.d6`'s own reading of a bare one. */
  readonly communication: readonly PyValue[];
  /** The granularity a shard keeps whole, evaluated; the integer 1 where the option declares none. */
  readonly granularity: PyValue;
  readonly declared: PyValue;
}

/** One applying cost entry of an instance (§4.5, D5). */
export interface CostDescription {
  /** Its position in the primitive's `logical_cost`. */
  readonly index: number;
  readonly per: PyValue;
  readonly status: PyValue;
  /** The entry's expression, evaluated in the instance's arguments. */
  readonly value: PyValue;
  readonly declared: PyValue;
}

/** Everything `describe` answers about one site (plan §5.3). */
export interface SiteDescription {
  readonly key: SiteKey;
  /** How a refusal names the site, and how D1 identifies it: `embed`, `decoder/attn[layer=3]`. */
  readonly where: string;
  /** Where the site is written in the document, for the pointer a sheet row needs. */
  readonly segments: readonly PathSegment[];
  /** The primitive's name, as the instance writes it. */
  readonly primitive: PyValue;
  /** The version the resolved definition carries. */
  readonly version: PyValue;
  /** The site as the document writes it: its `families`, its `when`, its arguments as expressions. */
  readonly instance: PyValue;
  /**
   * A template instance's interface — the public inputs and outputs its expansion resolved —
   * and `null` for every other instance (§4.6, §4.11 "Template instance").
   */
  readonly interface: InterfacePorts | null;
  /** The per-argument facts of §4.12, as feature 1.6a records them inside the walk. */
  readonly arguments: ArgumentDescription;
  readonly inputs: readonly PortDescription[];
  readonly outputs: readonly PortDescription[];
  readonly parameters: readonly SlotDescription[];
  readonly constants: readonly SlotDescription[];
  readonly states: readonly StateDescription[];
  /** The partition options whose condition holds. */
  readonly partitions: readonly PartitionDescription[];
  /** The cost entries whose condition holds. */
  readonly costs: readonly CostDescription[];
}

/** A description while `describe` is still filling its compatibility lists in. */
export type Mutable<T> = { -readonly [K in keyof T]: T[K] };

/** One site of the analysed graph, described. The compatibility lists are filled by the caller. */
export function describeSite(analysis: Analysis, site: ResolvedSite): SiteDescription {
  const definition = site.definition;
  const args = site.args;
  const bindings = analysis.bindings;
  /** The `(site, port)` key `producers`, `domains`, `consumed` and the two slot maps share. */
  const id = (name: PyValue): string => portKeyOf(site.key, name);

  const ports = (side: 'inputs' | 'outputs'): PortDescription[] =>
    entries(demand(demand(definition, 'ports'), side)).map(([name, port]) => {
      const here = present(port, args);
      const domain = demand(port, 'domain');
      const producer = analysis.producers.get(id(name));
      return {
        name,
        side,
        present: here,
        role: demand(port, 'role'),
        kind: demand(domain, 'kind'),
        from: has(domain, 'from') ? demand(domain, 'from') : null,
        domain: analysis.domains.get(id(name))?.domain ?? null,
        shape: here && has(port, 'shape') ? describeShape(demand(port, 'shape'), args) : null,
        fedBy: side === 'inputs' && producer !== undefined ? producer.by : null,
        consumed: side === 'outputs' && analysis.consumed.has(id(name)),
        declared: port,
      };
    });

  const slots = (kind: 'parameter' | 'constant'): SlotDescription[] =>
    entries(optional(definition, kind === 'parameter' ? 'parameters' : 'constants', {})).map(
      ([name, slot]) => {
        const here = present(slot, args);
        // "A slot that declares a multiplicity is m tensors of its shape … stored with one axis
        // before its shape axes" (§3.4): the stored shape is what a location addresses, what V15
        // compares and what D3 writes.
        const bound =
          kind === 'parameter' ? (bindings.slots.get(id(name)) ?? null) : null;
        return {
          name,
          kind,
          present: here,
          role: demand(slot, 'role'),
          sharing: kind === 'parameter' ? demand(slot, 'sharing') : null,
          multiplicity: has(slot, 'multiplicity')
            ? primitiveValue(demand(slot, 'multiplicity'), args)
            : null,
          shape: here ? describeShape(storageShape(slot), args) : null,
          boundBy: bound === null ? null : bound.rule,
          identity: null,
          tiesWith: [],
          declared: slot,
        };
      },
    );

  const states: StateDescription[] = entries(optional(definition, 'state_ports', {})).map(
    ([name, port]) => {
      const here = truthy(primitiveCondition(demand(port, 'present_when'), args));
      const rules = listOf(demand(port, 'rules'));
      const applying = here ? applicableRule(port, args) : null;
      const index = applying === null ? null : rules.indexOf(applying);
      const indexedBy = applying === null ? null : demand(applying, 'indexed_by');
      const bySource = indexedBy !== null && has(indexedBy, 'port');
      const bound = bindings.stateSlots.get(id(name)) ?? null;
      const written =
        !has(port, 'written_when') ||
        truthy(primitiveCondition(demand(port, 'written_when'), args));
      const carried = optional(port, 'carried_across', null);
      return {
        name,
        present: here,
        rule: applying,
        ruleIndex: index === null || index < 0 ? null : index,
        evolution: applying === null ? null : demand(applying, 'evolution'),
        access: applying === null ? null : demand(applying, 'access'),
        sharing: applying === null ? null : demand(applying, 'sharing'),
        indexedBySource: bySource,
        indexedByPort: bySource && indexedBy !== null ? demand(indexedBy, 'port') : null,
        span:
          applying !== null && has(applying, 'span')
            ? primitiveValue(demand(applying, 'span'), args)
            : null,
        stride:
          applying !== null && has(applying, 'stride')
            ? primitiveValue(demand(applying, 'stride'), args)
            : null,
        stream: applying === null ? null : stateStream(analysis, site.key, bySource, indexedBy),
        keyAxes: listOf(demand(port, 'key_axes')),
        payload: here
          ? entries(demand(port, 'payload')).map(([component, declared]) => ({
              name: component,
              role: demand(declared, 'role'),
              shape: describeShape(demand(declared, 'shape'), args),
            }))
          : [],
        // `sorted({o['effect'] for o in port['operations'].values()})`
        operations: sortedEffects(demand(port, 'operations')),
        written,
        carriedAcross:
          truthy(carried) && truthy(primitiveCondition(demand(carried, 'when'), args)),
        boundBy: bound === null ? null : bound.rule,
        identity: null,
        sharesWith: [],
        declared: port,
      };
    },
  );

  // "the partition_options every instance's primitive declares where their condition holds — the
  // communications each admits and the granularity a shard keeps whole" (D6).
  const partitions: PartitionDescription[] = [];
  listOf(optional(definition, 'partition_options', [])).forEach((option, index) => {
    if (has(option, 'when') && !truthy(primitiveCondition(demand(option, 'when'), args))) return;
    const communication = demand(option, 'communication');
    partitions.push({
      index,
      target: demand(option, 'target'),
      communication: Array.isArray(communication)
        ? (communication as readonly PyValue[])
        : [communication],
      granularity: has(option, 'granularity')
        ? primitiveValue(demand(option, 'granularity'), args)
        : 1n,
      declared: option,
    });
  });

  // "every applying correction" (D5): the same `when` and the same evaluation.
  const costs: CostDescription[] = [];
  listOf(optional(definition, 'logical_cost', [])).forEach((entry, index) => {
    if (has(entry, 'when') && !truthy(primitiveCondition(demand(entry, 'when'), args))) return;
    costs.push({
      index,
      per: demand(entry, 'per'),
      status: demand(entry, 'status'),
      value: primitiveValue(demand(entry, 'expression'), args),
      declared: entry,
    });
  });

  const expansion = analysis.subResults.get(keyOf(site.key));
  return {
    key: site.key,
    where: whereOfSite(site.key),
    segments: segmentsOfSite(site.key),
    primitive: site.primitive,
    version: demand(definition, 'version'),
    instance: site.instance,
    interface: expansion === undefined ? null : expansion.ports,
    arguments: site.arguments,
    inputs: ports('inputs'),
    outputs: ports('outputs'),
    parameters: slots('parameter'),
    constants: slots('constant'),
    states,
    partitions,
    costs,
  };
}

/** The stream a state grows along: the port's it is indexed by, or the instance's own (§4.3). */
function stateStream(
  analysis: Analysis,
  site: SiteKey,
  bySource: boolean,
  indexedBy: PyValue | null,
): Indexing | null {
  if (!bySource || indexedBy === null) return analysis.own.get(keyOf(site)) ?? null;
  return analysis.domains.get(portKeyOf(site, demand(indexedBy, 'port')))?.domain ?? null;
}

/** `sorted({o['effect'] for o in port['operations'].values()})`: a Python set of names, sorted. */
function sortedEffects(operations: PyValue): PyValue[] {
  const seen = new Map<string, PyValue>();
  for (const [, operation] of entries(operations)) {
    const effect = demand(operation, 'effect');
    seen.set(pyStr(effect), effect);
  }
  return [...seen.values()].sort((one, other) => comparePythonStrings(pyStr(one), pyStr(other)));
}
