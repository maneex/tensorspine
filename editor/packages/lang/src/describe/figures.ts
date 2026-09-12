/**
 * The derived facts §4.7 draws **on** the diagram: a node's own figures, and the identity links.
 *
 * > Derived line | D3 bytes over the node's slots, D4 bytes per cached position, D5 share when it
 * > adds up | View ▸ Show Derived Figures
 *
 * > Identity links (View ▸ Show Identities, or when a member is selected): dashed lines between
 * > the slot chips of the members of one identity, labelled with the identity name.
 *
 * Both are readings of a derived document — D3's tensors, D4's states, D2's values — and both are
 * here rather than in a component, for the reason the component inventory's §7 states in one
 * line: **"No component adds, converts or rounds a byte count."** A card's `80.0 MiB params` is
 * the sum of four of D3's rows, and a sum is an addition: it belongs where D3's own `totals` are
 * computed, with a test that says the two agree.
 *
 * **What is attributed to what.** D3 and D4 name their members by the identifier §5.2 rule 2
 * gives a site and the slot it binds — `decoder/attn[layer=0].q`, `embed.weight`,
 * `text/decoder/attn[layer=0].kv` inside a template instance. The folded canvas draws one box per
 * *declared* site, so every iteration of `decoder/attn` lands on one card and every site of an
 * expanded template lands on the template instance's. The reading is by longest prefix against
 * the folded graph's own boxes, so nothing here has to know how deep an expansion goes.
 *
 * **A tied identity counts once in the document and shows on every member's card.**
 * `embed.weight` and `lm_head.weight` are one identity of `qwen3.5-4b-text`, 1.18 GiB once; a
 * card says the tensors that instance uses, so each of the two says 1.18 GiB, and the sum over
 * the *identities* — not over the cards — is what equals D3's `totals.bytes`. The suite asserts
 * both halves.
 */
import type { PyValue } from '../expr/value.js';
import { entries, get, listOf } from '../library/access.js';
import { pyStr } from '../library/repr.js';

import { shapeText, valueGeometry } from '../derive/labels.js';
import type { FoldedGraph } from './folded.js';

// The members of a derived document this reading walks: the derived schema's own, read in the
// core as every other product is (plan §1 keeps them out of the interface's source).
const D2 = 'd2';
const D3 = 'd3';
const D4 = 'd4';
const TENSORS = 'tensors';
const STATES = 'states';
const VALUES = 'values';
const IDENTITY = 'identity';
const MEMBERS = 'members';
const VALUE = 'value';
const BYTES = 'bytes';
const PER_POSITION = 'bytes_per_cached_position';
const TIED = 'tied';
const LOCATION = 'location';

/** What a box of the folded canvas shows on its derived line (§4.7, artboard S2). */
export interface SiteFigures {
  /** D3 bytes over the identities this box's slots are members of; `null` where it has none. */
  readonly bytes: bigint | null;
  /** D4 bytes per cached position over this box's state ports; `null` where it has none. */
  readonly bytesPerCachedPosition: bigint | null;
  /** How many D3 identities name this box. */
  readonly tensors: number;
  /** How many D4 state identities name it. */
  readonly states: number;
  /** How many of its D3 identities carry a `location` — the located mark on a slot chip. */
  readonly located: number;
}

/** One identity a slot chip belongs to, as the canvas marks and links it. */
export interface IdentityMembership {
  /** The identity instance's name: `tied_embeddings`, `wq[layer=3]`, `embed.weight`. */
  readonly identity: string;
  /** Whether the identity has more than one member — S2's shared chip, and the dashed link. */
  readonly shared: boolean;
  /** Whether a `location` names the tensor — S2's located tick. */
  readonly located: boolean;
  /** Whether it is a state identity (D4) rather than a parameter one (D3). */
  readonly state: boolean;
}

/** One member of an identity, placed on the canvas. */
export interface IdentityPlace {
  /** The pointer of the folded box whose card carries the chip. */
  readonly box: string;
  /** The slot or state port the chip is for. */
  readonly slot: string;
}

/** One identity with several members: the dashed link of §4.7. */
export interface IdentityLink {
  readonly identity: string;
  readonly state: boolean;
  readonly places: readonly IdentityPlace[];
}

/** What a derived document says about the boxes of one folded graph. */
export interface DerivedFacts {
  /** The figures of each box, by its pointer. */
  readonly figures: ReadonlyMap<string, SiteFigures>;
  /** The identity each `(box, slot)` belongs to, keyed by {@link slotKey}. */
  readonly memberships: ReadonlyMap<string, IdentityMembership>;
  /** Every identity with more than one member, for View ▸ Show Identities. */
  readonly links: readonly IdentityLink[];
  /** The value type of each D2 value, by the identifier `<node>.<port>` D2 keys it with. */
  readonly types: ReadonlyMap<string, string>;
  /** The sum of every D3 identity's bytes, each counted once — D3's own `totals.bytes`. */
  readonly bytes: bigint | null;
}

/** The key a membership is held under. */
export function slotKey(box: string, slot: string): string {
  return `${box} ${slot}`;
}

/** A member identifier split into the site it names and the slot on it: `a/b[i=0].q`. */
export function splitMember(identifier: string): { site: string; slot: string } | null {
  const at = identifier.lastIndexOf('.');
  if (at < 0) return null;
  const slot = identifier.slice(at + 1);
  // A slot is an identifier (§5.2), so nothing after the last dot can be part of the site — and a
  // site's own indices sit inside brackets, which carry no dot in any corpus document. A name
  // with a bracket or a slash after the dot is no member identifier at all, and is left
  // unattributed rather than guessed at.
  if (slot === '' || slot.includes('[') || slot.includes('/')) return null;
  return { site: identifier.slice(0, at), slot };
}

/**
 * The box a site identifier belongs to: the longest prefix the folded graph has a box for.
 *
 * `decoder/attn[layer=0]` is `decoder`'s site `attn`; `embed` is the root instance; and
 * `text/decoder/attn[layer=0]`, which a template instance's expansion writes, is the root
 * instance `text` — the card the author edits.
 */
export function boxOfSite(graph: FoldedGraph, site: string): string | null {
  const parts = site.split('/').map((part) => part.replace(/\[.*$/, ''));
  for (let width = Math.min(parts.length, 2); width >= 1; width -= 1) {
    const pointer =
      width === 2
        ? `/compositions/${escapePointer(parts[0] as string)}/instances/${escapePointer(parts[1] as string)}`
        : `/instances/${escapePointer(parts[0] as string)}`;
    if (graph.byPointer.has(pointer)) return pointer;
  }
  return null;
}

/**
 * Every box a site's rows count towards, and how each of them counts it.
 *
 * Two readings, because the artboards draw two and they are not the same figure:
 *
 *  - a **card** stands for *one* iteration — S2's `decoder/attn` says `80.0 MiB params · 4 KiB /
 *    cached position`, which is one layer's four tensors and one layer's `kv`. A card is drawn
 *    over "one representative iteration" (§4.8) and its figures are that iteration's, which is
 *    what makes the card readable beside a sheet whose rows are the same site's.
 *  - a **group box** stands for the whole family — S3's `decoder` says `13.00 GiB params · 128
 *    KiB / cached position`, which is thirty-two of each — and so does a **root instance**, whose
 *    "family" is itself and, for a template instance, its whole expansion (S2's `▣ text` says
 *    `2.30 GiB`).
 *
 * So a row of D3 counts once towards its composition's box and once towards the card of the
 * representative iteration, and towards nothing else.
 */
function boxesOfSite(
  graph: FoldedGraph,
  representatives: ReadonlyMap<string, string>,
  site: string,
): { readonly group: string | null; readonly card: string | null } {
  const declared = boxOfSite(graph, site);
  if (declared === null) return { group: null, card: null };
  const node = graph.byPointer.get(declared);
  const group = node?.kind === 'site' ? node.parent : declared;
  const card = representatives.get(site) === declared ? declared : null;
  return { group, card };
}

/** The card each representative site identifier belongs to — `describe(folded)`'s own choice. */
function representativesOf(graph: FoldedGraph): Map<string, string> {
  const found = new Map<string, string>();
  for (const node of graph.nodes) {
    if (node.where !== null) found.set(node.where, node.pointer);
    for (const child of node.children) if (child.where !== null) found.set(child.where, child.pointer);
  }
  return found;
}

/** RFC 6901's escaping, as the pointers of a folded graph carry it. */
function escapePointer(name: string): string {
  return name.replace(/~/g, '~0').replace(/\//g, '~1');
}

/** A figure read off a product: an integer, or `null` where the product left it open. */
function integer(value: PyValue): bigint | null {
  if (typeof value === 'bigint') return value;
  if (typeof value === 'number' && Number.isInteger(value)) return BigInt(value);
  return null;
}

/** Nothing at all: the answer for a document nothing has been derived for. */
export function noDerivedFacts(): DerivedFacts {
  return { figures: new Map(), memberships: new Map(), links: [], types: new Map(), bytes: null };
}

/** One box's running figures while the walk fills them in. */
interface Running {
  bytes: bigint | null;
  per: bigint | null;
  tensors: number;
  states: number;
  located: number;
}

/** What a derived document says about the boxes of a folded graph (§4.7's derived line). */
export function derivedFacts(derived: PyValue, graph: FoldedGraph): DerivedFacts {
  const running = new Map<string, Running>();
  const memberships = new Map<string, IdentityMembership>();
  const links: IdentityLink[] = [];
  const types = new Map<string, string>();
  const representatives = representativesOf(graph);
  let total: bigint | null = null;

  const figureOf = (box: string): Running => {
    let found = running.get(box);
    if (found === undefined) {
      found = { bytes: null, per: null, tensors: 0, states: 0, located: 0 };
      running.set(box, found);
    }
    return found;
  };

  const walk = (rows: readonly PyValue[], figure: string, state: boolean): void => {
    for (const row of rows) {
      const identity = pyStr(get(row, IDENTITY) ?? '');
      const listed = listOf(get(row, MEMBERS) ?? []).map((one) => pyStr(one));
      const amount = integer(get(row, figure) ?? null);
      const located = !state && get(row, LOCATION) !== null;
      if (!state && amount !== null) total = (total ?? 0n) + amount;
      const places: IdentityPlace[] = [];
      const boxes = new Set<string>();
      for (const identifier of listed) {
        const split = splitMember(identifier);
        if (split === null) continue;
        const { group, card } = boxesOfSite(graph, representatives, split.site);
        // The chip is drawn on the card, which is the representative iteration's; a member of
        // another iteration has no chip of its own and only adds to the group box's figure.
        if (card !== null) {
          places.push({ box: card, slot: split.slot });
          memberships.set(slotKey(card, split.slot), {
            identity,
            shared: listed.length > 1,
            located,
            state,
          });
          boxes.add(card);
        }
        // An identity naming the same group box thirty-two times — one per iteration — is one
        // contribution to that box's figure and not thirty-two.
        if (group !== null) boxes.add(group);
      }
      for (const box of boxes) {
        const one = figureOf(box);
        if (state) {
          one.states += 1;
          if (amount !== null) one.per = (one.per ?? 0n) + amount;
        } else {
          one.tensors += 1;
          if (amount !== null) one.bytes = (one.bytes ?? 0n) + amount;
          if (located) one.located += 1;
        }
      }
      // A link joins two *cards*. An identity whose members are every iteration of one site —
      // `shared.sliding.kv` over nine layers of `decoder/attn` — has one card on the folded
      // canvas, and a dashed line from a chip to itself says nothing: the chip's own mark is
      // what says it is shared (S2), and §4.8's drill-in is where the nine are drawn apart.
      if (new Set(places.map((place) => place.box)).size > 1) links.push({ identity, state, places });
    }
  };

  walk(listOf(get(get(derived, D3) ?? null, TENSORS) ?? []), BYTES, false);
  walk(listOf(get(get(derived, D4) ?? null, STATES) ?? []), PER_POSITION, true);

  for (const value of listOf(get(get(derived, D2) ?? null, VALUES) ?? [])) {
    types.set(pyStr(get(value, VALUE) ?? ''), valueGeometry(value));
  }

  const figures = new Map<string, SiteFigures>();
  for (const [box, one] of running) {
    figures.set(box, {
      bytes: one.bytes,
      bytesPerCachedPosition: one.per,
      tensors: one.tensors,
      states: one.states,
      located: one.located,
    });
  }
  return { figures, memberships, links, types, bytes: total };
}

/** How many D3 identities a derived document records as tied — what a suite reads its sum against. */
export function tiedCount(derived: PyValue): number {
  let tied = 0;
  for (const row of listOf(get(get(derived, D3) ?? null, TENSORS) ?? [])) {
    if (get(row, TIED) === true) tied += 1;
  }
  return tied;
}

// --- one site's own rows of the products (plan §4.11's Derived section) ---------------------

// The members of a derived document the per-site reading walks, named here for the reason the
// four above are: the derived schema's vocabulary is the core's to read (plan §1).
const D1 = 'd1';
const D5 = 'd5';
const NODES = 'nodes';
const ACROSS_POSITIONS = 'across_positions';
const CORRECTIONS = 'corrections';
const NODE = 'node';
const ENTRY = 'entry';
const PER = 'per';
const STATUS = 'status';
const ROLE = 'role';
const DTYPE = 'dtype';
const SHAPE = 'shape';
const ELEMENTS = 'elements';
const MULTIPLICITY = 'multiplicity';
const SENSITIVITY = 'sensitivity';
const EVOLUTION = 'evolution';
const ACCESS = 'access';
const SHARING = 'sharing';
const STREAM = 'stream';
const INSTANCE_KEY = 'instance_key';
const PAYLOAD = 'payload';
const COMPONENT = 'component';
const BOUNDED = 'bytes_bounded';
const OPERATIONS = 'operations';
const WRITER = 'writer';
const CARRIED = 'carried_across_fragments';

/** One D3 tensor of a site, as the sheet's Derived section shows it (§4.11, artboard S6). */
export interface DerivedTensorRow {
  /** The identity instance: `decoder.attn.q[layer=0]`. */
  readonly identity: string;
  /** The slot on this site the row is about. */
  readonly slot: string;
  readonly role: string;
  readonly dtype: string;
  /** The shape as `[axis=extent, …]` — the products' own rendering (`shapeText`). */
  readonly shape: string;
  readonly elements: bigint | null;
  readonly bytes: bigint | null;
  /** How many slots are members of the identity: more than one is a tie. */
  readonly members: number;
  readonly tied: boolean;
  /** Whether the document locates the tensor — D3's own `location`. */
  readonly located: boolean;
  readonly multiplicity: bigint | null;
  readonly sensitivity: string | null;
}

/** One payload component of a D4 state row. */
export interface DerivedPayloadRow {
  readonly component: string;
  readonly dtype: string;
  readonly shape: string;
  readonly bytes: bigint | null;
}

/** One D4 state of a site. */
export interface DerivedStateRow {
  readonly identity: string;
  /** The state port on this site. */
  readonly port: string;
  readonly evolution: string;
  readonly access: string;
  readonly sharing: string;
  readonly stream: string | null;
  /** The axes the identity's instances are keyed on, in D4's own order. */
  readonly instanceKey: readonly string[];
  readonly payload: readonly DerivedPayloadRow[];
  readonly bytesPerCachedPosition: bigint | null;
  readonly bytesBounded: bigint | null;
  readonly operations: readonly string[];
  /** The node that writes it, as D4 names it; `null` where D4 records none. */
  readonly writer: string | null;
  readonly carriedAcrossFragments: boolean;
  readonly members: number;
}

/** One applying cost correction of a node — D5's own row (§4.5). */
export interface DerivedCostRow {
  /** Its position in the primitive's `logical_cost`. */
  readonly entry: number;
  readonly per: string;
  readonly status: string;
  readonly value: number | null;
}

/** What the products say about one site: §4.11's "Derived" section, read and never computed. */
export interface SiteDerived {
  readonly tensors: readonly DerivedTensorRow[];
  readonly states: readonly DerivedStateRow[];
  readonly corrections: readonly DerivedCostRow[];
  /** How many D1 nodes the *declared* site expands to — S6's `32 nodes`. */
  readonly nodes: number;
  /** This node's D1 `across_positions`; `null` where D1 does not carry the node. */
  readonly acrossPositions: boolean | null;
}

/** Nothing at all: what a site of a document nothing has been derived for shows. */
export function noSiteDerived(): SiteDerived {
  return { tensors: [], states: [], corrections: [], nodes: 0, acrossPositions: null };
}

/**
 * The declared site behind a node identifier: `decoder/attn[layer=0]` is `decoder/attn`.
 *
 * §5.2 rule 2's own reading, and the one place it is written. Exported because §4.18's selection
 * filter asks the same question of the products' identifiers — a reader who selected `attn` on
 * the canvas selected the *declaration*, and D1, D3, D4, D5 and D6 name its iterations — and the
 * inventory's §7 forbids a component to take a node identifier apart for itself.
 */
export function declaredSite(site: string): string {
  return site
    .split('/')
    .map((part) => part.replace(/\[.*$/, ''))
    .join('/');
}

/**
 * The rows of D3, D4, D5 and D1 that name one site (plan §4.11's Derived section).
 *
 * The site is named as D1 and every refusal name it — `SiteDescription.where`, §5.2 rule 2 — which
 * is one iteration of a declared site. A card and a sheet both stand for that one iteration
 * (§4.8, feature 2.9), so its rows are that iteration's, and `nodes` says how many iterations the
 * declared site has: S6's `32 nodes decoder/attn[layer=0…31]` printed for the one the sheet is of.
 *
 * Nothing is summed, converted or rounded here: every figure is the product's own number, and a
 * shape is the products' own rendering of a shape (`shapeText`).
 */
export function siteDerived(derived: PyValue, where: string): SiteDerived {
  const tensors: DerivedTensorRow[] = [];
  const states: DerivedStateRow[] = [];
  const corrections: DerivedCostRow[] = [];

  /** The slot of a member identifier that names this site, or `null` where it names another. */
  const slotHere = (identifier: string): string | null => {
    const split = splitMember(identifier);
    return split !== null && split.site === where ? split.slot : null;
  };

  for (const row of listOf(get(get(derived, D3) ?? null, TENSORS) ?? [])) {
    const listed = listOf(get(row, MEMBERS) ?? []).map((one) => pyStr(one));
    const slot = listed.map((one) => slotHere(one)).find((one) => one !== null);
    if (slot === undefined || slot === null) continue;
    tensors.push(tensorRow(row, slot));
  }

  for (const row of listOf(get(get(derived, D4) ?? null, STATES) ?? [])) {
    const listed = listOf(get(row, MEMBERS) ?? []).map((one) => pyStr(one));
    const port = listed.map((one) => slotHere(one)).find((one) => one !== null);
    if (port === undefined || port === null) continue;
    states.push(stateRow(row, port));
  }

  for (const row of listOf(get(get(derived, D5) ?? null, CORRECTIONS) ?? [])) {
    if (pyStr(get(row, NODE) ?? '') !== where) continue;
    const value = get(row, VALUE) ?? null;
    corrections.push({
      entry: Number(get(row, ENTRY) ?? 0n),
      per: pyStr(get(row, PER) ?? ''),
      status: pyStr(get(row, STATUS) ?? ''),
      value: value === null ? null : Number(value),
    });
  }

  const nodes = get(get(derived, D1) ?? null, NODES) ?? null;
  const declared = declaredSite(where);
  let count = 0;
  let across: boolean | null = null;
  for (const [identifier, node] of entries(nodes ?? {})) {
    if (declaredSite(identifier) !== declared) continue;
    count += 1;
    if (identifier === where) across = get(node, ACROSS_POSITIONS) === true;
  }
  return { tensors, states, corrections, nodes: count, acrossPositions: across };
}

/** One D3 row, as a sheet shows it — the slot is the caller's, since it says why the row is there. */
function tensorRow(row: PyValue, slot: string): DerivedTensorRow {
  const multiplicity = get(row, MULTIPLICITY) ?? null;
  const sensitivity = get(row, SENSITIVITY) ?? null;
  return {
    identity: pyStr(get(row, IDENTITY) ?? ''),
    slot,
    role: pyStr(get(row, ROLE) ?? ''),
    dtype: pyStr(get(row, DTYPE) ?? ''),
    shape: shapeText(get(row, SHAPE) ?? []),
    elements: integer(get(row, ELEMENTS) ?? null),
    bytes: integer(get(row, BYTES) ?? null),
    members: listOf(get(row, MEMBERS) ?? []).length,
    tied: get(row, TIED) === true,
    located: get(row, LOCATION) !== null && get(row, LOCATION) !== undefined,
    multiplicity: integer(multiplicity),
    sensitivity: sensitivity === null ? null : pyStr(sensitivity),
  };
}

/** One D4 row, likewise. */
function stateRow(row: PyValue, port: string): DerivedStateRow {
  const stream = get(row, STREAM) ?? null;
  const writer = get(row, WRITER) ?? null;
  return {
    identity: pyStr(get(row, IDENTITY) ?? ''),
    port,
    evolution: pyStr(get(row, EVOLUTION) ?? ''),
    access: pyStr(get(row, ACCESS) ?? ''),
    sharing: pyStr(get(row, SHARING) ?? ''),
    stream: stream === null ? null : pyStr(stream),
    instanceKey: listOf(get(row, INSTANCE_KEY) ?? []).map((one) => pyStr(one)),
    payload: listOf(get(row, PAYLOAD) ?? []).map((one) => ({
      component: pyStr(get(one, COMPONENT) ?? ''),
      dtype: pyStr(get(one, DTYPE) ?? ''),
      shape: shapeText(get(one, SHAPE) ?? []),
      bytes: integer(get(one, BYTES) ?? null),
    })),
    bytesPerCachedPosition: integer(get(row, PER_POSITION) ?? null),
    bytesBounded: integer(get(row, BOUNDED) ?? null),
    operations: listOf(get(row, OPERATIONS) ?? []).map((one) => pyStr(one)),
    writer: writer === null ? null : pyStr(writer),
    carriedAcrossFragments: get(row, CARRIED) === true,
    members: listOf(get(row, MEMBERS) ?? []).length,
  };
}

// --- one identity's own rows of the products (plan §4.11's Identity sheet) ------------------

/** The members of D3 and D4 a row is named by; the slot a row is about is the product's own. */
const SLOT = 'slot';
const STATE = 'state';

/** What the products say about one identity — §4.11's Identity row, and its totals. */
export interface IdentityDerived {
  /** The D3 rows of the identity's instances, in D3's own order; empty for a state identity. */
  readonly tensors: readonly DerivedTensorRow[];
  /** Its D4 rows, likewise. */
  readonly states: readonly DerivedStateRow[];
  /** The sum of the instances' bytes — D3's own numbers, added and not converted. */
  readonly bytes: bigint | null;
  /** The sum of the instances' bytes per cached position (D4). */
  readonly bytesPerCachedPosition: bigint | null;
  /** How many instances the identity has: one per index point the rule fired at. */
  readonly instances: number;
}

/** Nothing at all: what an identity of a document nothing has been derived for shows. */
export function noIdentityDerived(): IdentityDerived {
  return { tensors: [], states: [], bytes: null, bytesPerCachedPosition: null, instances: 0 };
}

/**
 * The D3 or D4 rows of one identity, by the name its instances carry.
 *
 * An identity with indices has one instance per index point — `decoder.attn.q[layer=0]` …
 * `[layer=31]` — and one without has exactly itself, which is the same rule read twice: a row
 * belongs to the identity when its own name is the identity's, or the identity's followed by the
 * bracket the instances are written with. Nothing is summed that the products do not carry, and a
 * total is the products' own numbers added — the same licence `derivedFacts` takes for a card.
 */
export function identityDerived(derived: PyValue, identity: string, state: boolean): IdentityDerived {
  const tensors: DerivedTensorRow[] = [];
  const states: DerivedStateRow[] = [];
  let bytes: bigint | null = null;
  let per: bigint | null = null;
  const mine = (row: PyValue): boolean =>
    identityInstanceOf(pyStr(get(row, IDENTITY) ?? ''), identity);

  if (!state) {
    for (const row of listOf(get(get(derived, D3) ?? null, TENSORS) ?? [])) {
      if (!mine(row)) continue;
      const made = tensorRow(row, pyStr(get(row, SLOT) ?? ''));
      tensors.push(made);
      if (made.bytes !== null) bytes = (bytes ?? 0n) + made.bytes;
    }
  } else {
    for (const row of listOf(get(get(derived, D4) ?? null, STATES) ?? [])) {
      if (!mine(row)) continue;
      const made = stateRow(row, pyStr(get(row, STATE) ?? ''));
      states.push(made);
      if (made.bytesPerCachedPosition !== null) per = (per ?? 0n) + made.bytesPerCachedPosition;
    }
  }
  return {
    tensors,
    states,
    bytes,
    bytesPerCachedPosition: per,
    instances: tensors.length + states.length,
  };
}

/**
 * Whether an identity instance name is an instance of that identity.
 *
 * "An identity with indices has one instance per index point — `decoder.attn.q[layer=0]` …
 * `[layer=31]` — and one without has exactly itself", which is the same rule read twice. Exported
 * for §4.18's selection filter, which holds D3's and D4's rows to the identity a sheet is open
 * on: one reading of what an instance name is, never a second.
 */
export function identityInstanceOf(instance: string, identity: string): boolean {
  return instance === identity || instance.startsWith(`${identity}[`);
}

// --- D2's own rows (plan §4.11's Edge, Input and Output sheets) -----------------------------

// The members of D2 the readings below walk, named here for the reason every other member of a
// product is named in this file and not in a component (plan §1).
const STREAMS = 'streams';
const KIND = 'kind';
const COUNT = 'count';
const FRAGMENT_ALIGNMENT = 'fragment_alignment';
const TO = 'to';
const INPUT = 'input';
const EXPOSED = 'exposed';
const REQUIRED = 'required';
const REQUIRED_FOR = 'required_for';
const DOMAIN = 'domain';
const PER_ELEMENT = 'bytes_per_element';

/** One value of D2, as §4.11's Edge, Input and Output sheets show it. */
export interface DerivedValueRow {
  /** The producing `node.port`, or the name of the public input that delivers the value. */
  readonly value: string;
  /** The ports it feeds, as D2 names them. */
  readonly to: readonly string[];
  /** The shape as `[axis=extent, …]` — the products' own rendering. */
  readonly shape: string;
  /** The value type `--view` writes: `bf16[tokens, model.width=4096]` (`valueGeometry`). */
  readonly geometry: string;
  readonly role: string;
  readonly dtype: string;
  readonly elements: number | null;
  readonly bytesPerElement: number | null;
  /** Its domain, as `kind per stream` — the two members D2 writes, in D2's order. */
  readonly domain: string;
  /** The stream its domain names; `null` where D2 writes none, and never a reading of the text. */
  readonly stream: string | null;
  /** Its count per invocation, member by member, as D2 writes it. */
  readonly count: string;
  /** The public input that delivers it, or `null`. */
  readonly input: string | null;
  /** The public outputs that expose it. */
  readonly exposed: readonly string[];
  /** Whether some public output is not evaluated without this input (§7); `null` where D2 is silent. */
  readonly required: boolean | null;
  /** The public outputs not evaluated without it on a first delivery. */
  readonly requiredFor: readonly string[];
}

/** One stream of D2 — what a public input's own sheet shows beside its value (§4.11). */
export interface DerivedStreamRow {
  readonly name: string;
  readonly kind: string;
  /** The elements it delivers per invocation, as D2 writes the count. */
  readonly count: string;
  /** The multiple every fragment delivers, for a fragmented stream; `null` where D2 writes none. */
  readonly fragmentAlignment: bigint | null;
}

/** Every value D2 carries, in D2's own order. */
export function valueRows(derived: PyValue): DerivedValueRow[] {
  return listOf(get(get(derived, D2) ?? null, VALUES) ?? []).map((row) => {
    const input = get(row, INPUT) ?? null;
    const required = get(row, REQUIRED) ?? null;
    const elements = get(row, ELEMENTS) ?? null;
    const perElement = get(row, PER_ELEMENT) ?? null;
    const domain = get(row, DOMAIN) ?? null;
    const stream = domain === null ? null : (get(domain, STREAM) ?? null);
    return {
      value: pyStr(get(row, VALUE) ?? ''),
      to: listOf(get(row, TO) ?? []).map((one) => pyStr(one)),
      shape: shapeText(get(row, SHAPE) ?? []),
      geometry: valueGeometry(row),
      role: pyStr(get(row, ROLE) ?? ''),
      dtype: pyStr(get(row, DTYPE) ?? ''),
      elements: elements === null ? null : Number(elements),
      bytesPerElement: perElement === null ? null : Number(perElement),
      domain: recordText(domain),
      stream: stream === null ? null : pyStr(stream),
      count: recordText(get(row, COUNT) ?? null),
      input: input === null ? null : pyStr(input),
      exposed: listOf(get(row, EXPOSED) ?? []).map((one) => pyStr(one)),
      required: typeof required === 'boolean' ? required : null,
      requiredFor: listOf(get(row, REQUIRED_FOR) ?? []).map((one) => pyStr(one)),
    };
  });
}

/** Every stream D2 reports, in D2's own order. */
export function streamRows(derived: PyValue): DerivedStreamRow[] {
  const streams = get(get(derived, D2) ?? null, STREAMS) ?? null;
  return entries(streams ?? {}).map(([name, row]) => ({
    name,
    kind: pyStr(get(row, KIND) ?? ''),
    count: recordText(get(row, COUNT) ?? null),
    fragmentAlignment: integer(get(row, FRAGMENT_ALIGNMENT) ?? null),
  }));
}

/** The D2 value one end of an edge carries, by the identifier D2 keys it with; `null` for none. */
export function valueOf(rows: readonly DerivedValueRow[], value: string): DerivedValueRow | null {
  return rows.find((row) => row.value === value) ?? null;
}

/** The D2 value a public input delivers; `null` where nothing has been derived for it. */
export function inputValue(rows: readonly DerivedValueRow[], name: string): DerivedValueRow | null {
  return rows.find((row) => row.input === name) ?? null;
}

/** The D2 value a public output exposes. */
export function outputValue(rows: readonly DerivedValueRow[], name: string): DerivedValueRow | null {
  return rows.find((row) => row.exposed.includes(name)) ?? null;
}

/**
 * A small record of the products written on one line: `token per tokens`, `tokens 1.0`.
 *
 * A domain and a count are records of a member or two whose names are the products' and whose
 * values are names and numbers; a sheet shows them as the product wrote them, in the product's
 * own order, with nothing converted or rounded.
 */
function recordText(value: PyValue): string {
  if (value === null) return '';
  if (!isPlainRecord(value)) return pyStr(value);
  return entries(value)
    .map(([name, held]) => `${name} ${pyStr(held)}`)
    .join(' · ');
}

/** Whether a value of a product is one of its records. */
function isPlainRecord(value: PyValue): boolean {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
