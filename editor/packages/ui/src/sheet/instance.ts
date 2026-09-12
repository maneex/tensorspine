/**
 * The instance (site) sheet — plan §4.11's first row, artboard S6.
 *
 * > **Instance / site** | Identity (name; primitive with version select; families as a chip editor
 * > …; guard `when` …) · **Arguments** (§4.12) · Ports (inputs: fed by … / *unfed*; outputs:
 * > consumed by … / exposed as … / *unconsumed*) · Parameters (per present slot: role, evaluated
 * > shape, identity name or *unbound* …) · Constants · States (per present state port: the rule
 * > that applies with its evolution/access/sharing, identity or *unbound*, …, the writer verdict)
 * > · Derived (D3 rows of its slots, D4 rows of its states, D5 corrections, D6 partition options;
 * > the D1 `across_positions` flag)
 *
 * Every section below is a projection of one answer the core already gave — `describe`'s
 * `SiteDescription` for the identity, the ports, the slots, the states, the partitions and the
 * costs, and `siteDerived` over the derived document for the rest. The inventory's §7 is the whole
 * design: *a component that shows a fact never computes it*, and no figure is added, converted or
 * rounded here.
 *
 * **What is shown and what is not.** §4.11 lists the *actions* beside these rows — "Bind
 * privately", "Tie to…", "Share with…", the dtype select within the role's admissible set, "Edit
 * location…" — and they are feature 2.13's, whose own block names every one of them. What this
 * feature owes is the row and its state: which slot is bound and to which identity, which is
 * unbound (V7's own row), which is absent under these arguments and why, and what the products say
 * about each. The gestures land on rows that already exist.
 */
import {
  pyStr,
  type DescribedShape,
  type PortDescription,
  type PyValue,
  type SiteDerived,
  type SiteDescription,
  type SlotDescription,
  type StateDescription,
} from '@tensorspine/lang';

/** One port row of the sheet: a declared port, present or not, with what it is joined to. */
export interface PortRow {
  readonly name: string;
  readonly side: 'inputs' | 'outputs';
  readonly present: boolean;
  /** The precision role the port declares. */
  readonly role: string;
  /** The evaluated shape as `axis=extent` rows; empty where the port declares none or is absent. */
  readonly shape: string;
  /** What feeds an input — the binding's name — or `null` where nothing does (V7's row). */
  readonly fedBy: string | null;
  /** Whether an edge or a public output consumes an output port (V13). */
  readonly consumed: boolean;
  /** Why an absent port is absent: the arguments it is conditioned on are the row's own tooltip. */
  readonly absent: boolean;
}

/** One parameter or constant slot row. */
export interface SlotRow {
  readonly name: string;
  readonly kind: 'parameter' | 'constant';
  readonly present: boolean;
  readonly role: string;
  /** The shape **as stored** — a declared multiplicity's axis first (§3.4). */
  readonly shape: string;
  /** The identity instance the slot belongs to, or `null` while nothing binds it. */
  readonly identity: string | null;
  /** The binding rule that bound it, as the normalised document names it. */
  readonly boundBy: string | null;
  /** Where the member is written in that rule's list: what a gesture that moves the slot removes. */
  readonly boundAt: number | null;
  /**
   * Whether a physical tensor names it — S2's located tick, and §4.11's "location summary".
   *
   * D3's own `located`, taken from the row of this slot; `null` while nothing has been derived.
   */
  readonly located: boolean | null;
  /**
   * How many slots the identity holds — D3's own count; `null` while nothing has been derived.
   *
   * What tells a private identity from a tie, which is what says whether "Bind privately" has
   * anything to do: a slot that is already alone in its identity is already bound privately.
   */
  readonly members: number | null;
  /** The declared multiplicity, evaluated; `null` where the slot declares none. */
  readonly multiplicity: string | null;
  /** How many identities the slot may join, when the description carries the list (feature 2.13). */
  readonly partners: number;
}

/** One state-port row: the applying rule, and what the binding made of it. */
export interface StateRow {
  readonly name: string;
  readonly present: boolean;
  /** Which of the port's ordered rules applies, and how many there are (S6's `4 of 4`). */
  readonly rule: number | null;
  readonly rules: number;
  readonly evolution: string | null;
  readonly access: string | null;
  readonly sharing: string | null;
  /** What the state is indexed by: `self`, or the input port it follows. */
  readonly indexedBy: string;
  /** The instance axes the port is keyed on. */
  readonly keyAxes: readonly string[];
  /** The payload per cached position, component by component, with its evaluated shape. */
  readonly payload: readonly { readonly name: string; readonly shape: string }[];
  /** Whether this instance writes the port — `written_when`, or none declared (V20). */
  readonly written: boolean;
  /** Whether the primitive's own carrying condition holds (V16). */
  readonly carried: boolean;
  readonly identity: string | null;
  readonly boundBy: string | null;
  /** Where the member is written in that rule's list; `null` while nothing bound the port. */
  readonly boundAt: number | null;
  /** How many ports share the identity — D4's own count; `null` before a derivation. */
  readonly members: number | null;
  readonly partners: number;
}

/** One applying partition option — D6's own reading of the primitive's declaration. */
export interface PartitionRow {
  readonly target: string;
  readonly communication: readonly string[];
  readonly granularity: string;
}

/** One applying cost entry — D5's, before the products (§4.5). */
export interface CostRow {
  readonly per: string;
  readonly status: string;
  readonly value: string;
}

/** The instance sheet: every section of §4.11's first row, as data. */
export interface InstanceSheet {
  /** How a refusal and D1 name the site: `embed`, `decoder/attn[layer=0]`. */
  readonly where: string;
  /** The site's own name — the map key, which is what the Identity row edits. */
  readonly name: string;
  /** The composition it sits in, or `null` at the top level. */
  readonly composition: string | null;
  readonly primitive: string;
  readonly version: string;
  readonly inputs: readonly PortRow[];
  readonly outputs: readonly PortRow[];
  readonly parameters: readonly SlotRow[];
  readonly constants: readonly SlotRow[];
  readonly states: readonly StateRow[];
  readonly partitions: readonly PartitionRow[];
  readonly costs: readonly CostRow[];
  /** What the products say about this site; empty before anything has been derived. */
  readonly derived: SiteDerived;
}

/** The sheet of one described site. */
export function instanceSheet(site: SiteDescription, derived: SiteDerived): InstanceSheet {
  return {
    where: site.where,
    name: site.key.name,
    composition: site.key.composition === '' ? null : site.key.composition,
    primitive: pyStr(site.primitive),
    version: pyStr(site.version),
    inputs: site.inputs.map(portRow),
    outputs: site.outputs.map(portRow),
    parameters: site.parameters.map((slot) => slotRow(slot, derived)),
    constants: site.constants.map((slot) => slotRow(slot, derived)),
    states: site.states.map((state) => stateRow(state, derived)),
    partitions: site.partitions.map((one) => ({
      target: targetText(one.target),
      communication: one.communication.map((each) => pyStr(each)),
      granularity: pyStr(one.granularity),
    })),
    costs: site.costs.map((one) => ({
      per: pyStr(one.per),
      status: pyStr(one.status),
      value: pyStr(one.value),
    })),
    derived,
  };
}

/**
 * A partition target as one line — S6's `attention.heads`, `payload kv.k, kv.v by head`.
 *
 * A `partition_target` is a tagged object whose payload is a name or a small record of names
 * (`{"payload_axis": {"state": "kv", "component": "k", "axis": "attention.kv_heads"}}`), and
 * `pyStr` of it is Python's dict repr — true, and unreadable on a row. What is shown instead is the
 * tag the schema gives it and the names under it, in the order they are written: generic over any
 * shape the grammar admits, with no member of it named here.
 */
export function targetText(target: PyValue): string {
  const leaves = (value: PyValue): string[] => {
    if (Array.isArray(value)) return (value as readonly PyValue[]).flatMap((one) => leaves(one));
    if (value !== null && typeof value === 'object') {
      return Object.values(value as Record<string, PyValue>).flatMap((one) => leaves(one));
    }
    return [pyStr(value)];
  };
  if (target === null || typeof target !== 'object' || Array.isArray(target)) return pyStr(target);
  return Object.entries(target as Record<string, PyValue>)
    .map(([tag, held]) => `${tag} ${leaves(held).join('.')}`)
    .join(' · ');
}

/** A shape as one line: the same `axis=extent` rendering the products and the handles use. */
export function shapeLine(shape: DescribedShape | null): string {
  if (shape === null) return '';
  return shape.map((axis) => `${pyStr(axis.axis)}=${pyStr(axis.extent)}`).join(', ');
}

function portRow(port: PortDescription): PortRow {
  return {
    name: port.name,
    side: port.side,
    present: port.present,
    role: pyStr(port.role),
    shape: shapeLine(port.shape),
    fedBy: port.fedBy,
    consumed: port.consumed,
    absent: !port.present,
  };
}

function slotRow(slot: SlotDescription, derived: SiteDerived): SlotRow {
  // D3's row for this slot, where the document has been derived: its `located` is §4.11's
  // "location summary", and no component decides what located means.
  const tensor = derived.tensors.find((row) => row.slot === slot.name);
  return {
    name: slot.name,
    kind: slot.kind,
    present: slot.present,
    role: pyStr(slot.role),
    shape: shapeLine(slot.shape),
    identity: slot.identity,
    boundBy: slot.boundBy,
    boundAt: slot.boundAt,
    located: tensor === undefined ? null : tensor.located,
    members: tensor === undefined ? null : tensor.members,
    multiplicity: slot.multiplicity === null ? null : pyStr(slot.multiplicity),
    partners: slot.tiesWith.length,
  };
}

function stateRow(state: StateDescription, derived: SiteDerived): StateRow {
  const row = derived.states.find((each) => each.port === state.name);
  return {
    name: state.name,
    present: state.present,
    rule: state.ruleIndex === null ? null : state.ruleIndex + 1,
    rules: state.rules,
    evolution: state.evolution === null ? null : pyStr(state.evolution),
    access: state.access === null ? null : pyStr(state.access),
    sharing: state.sharing === null ? null : pyStr(state.sharing),
    indexedBy: state.indexedByPort === null ? SELF : pyStr(state.indexedByPort),
    keyAxes: state.keyAxes.map((one) => pyStr(one)),
    payload: state.payload.map((one) => ({ name: one.name, shape: shapeLine(one.shape) })),
    written: state.written,
    carried: state.carriedAcross,
    identity: state.identity,
    boundBy: state.boundBy,
    boundAt: state.boundAt,
    members: row === undefined ? null : row.members,
    partners: state.sharesWith.length,
  };
}

/**
 * What a state indexed by itself is written as.
 *
 * Not a value of any enumeration the schemas carry: `indexed_by` is a union of `{"self": true}` and
 * `{"port": …}`, so the word is the *member's* name and the row shows the port's name in the other
 * case. The core answers which of the two it is (`indexedBySource`), and this is the label.
 */
const SELF = 'self';
