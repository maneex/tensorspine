/**
 * The **drill-in** reading of one composition — plan §4.8 and D8, artboards S4 and S5.
 *
 * > Compositions are edited in a drill-in canvas over one representative iteration. Inside
 * > `decoder`, the sites are nodes, the scoped edges are edges, a carry edge with an index
 * > override (`ffn_r[layer−1] → attn_n`) is drawn from a ghost column on the left, a guard is a
 * > badge, and a scrubber sets concrete index values to preview which sites and edges exist —
 * > from D1, the expanded graph the core computes, filtered by index.
 *
 * Feature 2.9's {@link foldedGraph} reads the document's *top level*: the root instances, the
 * compositions as boxes, and `bindings/values`. What it does not read is the inside of a
 * composition — its own `bindings`, the index overrides on their endpoints, the guards on their
 * rules — and feature 2.9 said so: "its scoped edges are not drawn at all: both are §4.8's
 * reading and feature 2.14's". This is that reading, and it is in the core for the reason the
 * folded one is: a drill-in names `site`, `indices`, `when` and `port`, and plan §1 keeps the
 * grammar's members out of the interface.
 *
 * **Three things it answers, and where each comes from.**
 *
 *  - **What is drawn** — the sites, the scoped edges, the ghost columns, the pinned terminals —
 *    is read off the store's own tree, tolerantly, exactly as {@link foldedGraph} reads the top
 *    level: a document between two keystrokes is drawn as far as it goes, and an index the author
 *    wrote as an expression is carried as the author wrote it.
 *  - **What exists at an index** — the scrubber and the alternation strip — is read off **D1**:
 *    "from D1's expanded graph (the nodes `decoder/*[layer=v]`), the core's expansion", and "read
 *    from D1, the core's expansion". {@link drillPresence} takes the expanded graph and asks it
 *    whether a node is there, which is the one reading the plan names and the one the derivation
 *    already computed (`derive`'s own `d1`, §5.4's "on success ──► core.expand"). No second
 *    expansion is run for the strip.
 *  - **Why a site or an edge is absent at an index** — S4's `when layer ≥ 1 — false`, S5's "guard
 *    fails" — is the *condition's truth*, which D1 does not carry: an absent node says nothing
 *    about the reason. So the guard is evaluated, by `expr/model.ts`'s `modelCondition` under the
 *    same environment `d1.emit` evaluates it under, and the suite holds the two readings to each
 *    other on the corpus: a guarded site is present in D1 exactly where its guard is true.
 *
 * **The writers.** A gesture of §4.8 and §4.20 writes a scoped endpoint, a carry override, the
 * guard that carry proposes, the complement of a guard, a composition's own definition: each is a
 * shape of the grammar and each is written here, beside {@link valueEndpoint} and
 * {@link instanceSkeleton}, so that the canvas writes none of it (feature 2.9's rule).
 */
import { indexGrid, modelCondition, modelValue, type Env, type Quantities } from '../expr/model.js';
import { toPython, truthy, UNRESOLVED, type PyRecord, type PyValue } from '../expr/value.js';
import { get, has } from '../library/access.js';
import { pyStr } from '../library/repr.js';
import {
  isJsonArray,
  isJsonObject,
  jsonObject,
  type JsonObject,
  type JsonValue,
} from '../json/index.js';
import { pointerSegment } from '../schema/index.js';
import type { PathSegment } from '../schema/types.js';
import { generatedSite, rootSite, whereOfSite, type IndexBinding } from '../validate/index.js';

import {
  foldedGraph,
  literalIndex,
  type FoldedGraph,
  type FoldedHandle,
  type FoldedHeld,
  type FoldedNode,
  type FoldedRange,
} from './folded.js';

// ---------------------------------------------------------------------------------------------
// The members of the grammar this reading walks. They are the schema's, read in the core: plan
// §1 keeps them out of `packages/ui`, which is the whole reason this module exists.
// ---------------------------------------------------------------------------------------------

const COMPOSITIONS = 'compositions';
const INSTANCES = 'instances';
const BINDINGS = 'bindings';
const VALUES = 'values';
const INTERFACES = 'interfaces';
const INPUTS = 'inputs';
const OUTPUTS = 'outputs';
const INDICES = 'indices';
const INDEX = 'index';
const SITE = 'site';
const PORT = 'port';
const INSTANCE = 'instance';
const COMPOSITION = 'composition';
const KIND = 'kind';
const GENERATED = 'generated';
const WHEN = 'when';
const FROM = 'from';
const TO = 'to';
const FAMILIES = 'families';
const START = 'start';
const STOP = 'stop';
const STEP = 'step';
const NOT = 'not';
const OP = 'op';
const ARGS = 'args';
const SUBTRACT = 'subtract';
const COMPARE = 'compare';
const OPERATOR = 'operator';
const LEFT = 'left';
const RIGHT = 'right';
const GREATER_OR_EQUAL = 'greater_or_equal';
const D1 = 'd1';
const NODES = 'nodes';

/**
 * The composition a place of the document *is*, or `null` where the place is not one.
 *
 * What a drill-in can be opened on is a fact about the document, and the interface may not decide
 * it by naming the map (plan §1): a row of the Model explorer, a box of the canvas and a command
 * all ask here. The folded reading already knows — a composition is the box that holds sites — so
 * this is one line over its answer rather than a second walk.
 */
export function compositionAt(folded: FoldedGraph, pointer: string): string | null {
  const node = folded.byPointer.get(pointer);
  return node !== undefined && node.kind === 'composition' ? node.name : null;
}

/** How §5.2 rule 7 names a composition-scoped rule at the top level: `<composition>.<rule>`. */
export function hoistedRule(composition: string, rule: string): string {
  return `${composition}.${rule}`;
}

/** One index of the composition, as the strip prints it and the scrubber runs over it. */
export interface DrillRange extends FoldedRange {
  /**
   * The values the index takes, in the grid's own order; `null` where a bound does not resolve.
   *
   * `indexGrid` is the validator's and the emitter's own reading of a `finite_range`, so the
   * scrubber's track is the very set of points D1 emitted a site for — not a range the interface
   * counted out of `start`, `stop` and `step`.
   */
  readonly values: readonly bigint[] | null;
}

/** One point of the composition's grid: what the scrubber stands on and what a column stands for. */
export interface DrillPoint {
  /** The index values, sorted by name as §5.2 rule 2 sorts them. */
  readonly indices: readonly IndexBinding[];
  /** `layer=0` — the column's key, and what a label prints. */
  readonly label: string;
}

/** An index an endpoint writes for itself: the expression as the document holds it. */
export interface DrillIndex {
  readonly name: string;
  readonly written: JsonValue;
}

/** Where one end of a drill-in edge hangs. */
export interface DrillEnd {
  /** The site's or the instance's own name. */
  readonly name: string;
  readonly port: string;
  /** The pointer of the site's declaration inside this composition; `null` when it is outside. */
  readonly site: string | null;
  /**
   * The composition the endpoint's selector names, or `null` for a root instance.
   *
   * The drill-in's own composition for a `site_endpoint`, whatever a generated selector writes,
   * and `null` where the endpoint names a root instance — which is what says how the node this
   * end stands for is identified (§5.2 rule 2).
   */
  readonly composition: string | null;
  /** The indices the endpoint writes, in the document's order. */
  readonly indices: readonly DrillIndex[];
  /**
   * Whether the endpoint writes an index of *this* composition — the ghost column's own reason.
   *
   * §4.8: "an edge whose endpoints have no index override is drawn plainly; an edge with an
   * override on its source is drawn from a ghost column on the left". A site endpoint writing no
   * `indices` is the current iteration; one writing any is a ghost.
   */
  readonly overridden: boolean;
}

/** One edge of the drill-in (§4.8). */
export interface DrillEdge {
  /** The rule's own name, as the document writes it. */
  readonly rule: string;
  /** The name §5.2 rule 7 gives it, which is what D1 lists it under. */
  readonly hoisted: string;
  /** The place the rule is written at — the *written* one, never the hoisted one. */
  readonly pointer: string;
  readonly segments: readonly PathSegment[];
  readonly from: DrillEnd | null;
  readonly to: DrillEnd | null;
  /** The rule's guard, as written; `null` where it has none (§4.8's "guards on rules"). */
  readonly guard: JsonValue | null;
  /** Whether the rule is the composition's own, rather than a top-level one crossing its boundary. */
  readonly scoped: boolean;
  /** Whether it is an interface's own wire rather than a value binding. */
  readonly interface: boolean;
}

/** A ghost column: a translucent copy of a site at an overridden index (§4.8, inventory §3). */
export interface DrillGhost {
  /** The column's own identity on the drawing — per (site, side, override), as §4.8 states. */
  readonly id: string;
  readonly name: string;
  /** The pointer of the site it copies; `null` where the endpoint names something outside. */
  readonly site: string | null;
  /** `left` for an override on an edge's source, `right` for one on its destination. */
  readonly side: 'left' | 'right';
  readonly indices: readonly DrillIndex[];
  /** The edges drawn through this column, by the place each rule is written at. */
  readonly edges: readonly string[];
}

/** One link of a pinned terminal: which rule, and which site of this composition it touches. */
export interface DrillLink {
  readonly rule: string;
  readonly pointer: string;
  readonly name: string;
  readonly site: string | null;
  readonly port: string;
}

/** A pinned terminal: an edge from the top level into the composition, or out of it (§4.8). */
export interface DrillTerminal {
  readonly id: string;
  /** `left` for an edge entering the composition, `right` for one leaving it. */
  readonly side: 'left' | 'right';
  /** The outside end, as it is named: `embed.output`, `tokens`. */
  readonly outside: string;
  /** The indices the inside end names — S4's `at layer = 0`. */
  readonly indices: readonly DrillIndex[];
  /** The rules pinned on it, in the document's order. */
  readonly links: readonly DrillLink[];
}

/** The drill-in of one composition: what §4.8's tab draws. */
export interface DrillGraph {
  readonly pointer: string;
  readonly segments: readonly PathSegment[];
  readonly name: string;
  readonly families: readonly string[];
  readonly ranges: readonly DrillRange[];
  /** The composition's sites, in the document's own order — the cards of §4.8. */
  readonly sites: readonly FoldedNode[];
  /** The scoped edges and the boundary edges, the composition's own first. */
  readonly edges: readonly DrillEdge[];
  readonly ghosts: readonly DrillGhost[];
  readonly terminals: readonly DrillTerminal[];
  /** The grid's points, in `indexGrid`'s order; empty where a bound does not resolve. */
  readonly points: readonly DrillPoint[];
  /** How many iterations the ranges resolve to; `null` where a bound does not. */
  readonly count: bigint | null;
  /** What the composition holds, container by container (S3's summary line). */
  readonly held: readonly FoldedHeld[];
  readonly quantities: Quantities;
}

/** What {@link drillGraph} is given beside the document. */
export interface DrillOptions {
  /** The composition, by its name — the map key, which is what a tab is opened on. */
  readonly composition: string;
  /** The assignment the external quantities are read under (§4.6). */
  readonly assignment?: PyRecord;
  /** A folded reading already computed for this tree, so the drill-in does not build a second. */
  readonly folded?: FoldedGraph;
}

/** A pointer from path segments, RFC 6901. */
function pointerOf(segments: readonly PathSegment[]): string {
  return segments.map((step) => `/${pointerSegment(String(step))}`).join('');
}

/** The member of an object node, or `null` where there is none — and where it is not one. */
function member(value: JsonValue | null, name: string): JsonValue | null {
  if (value === null || !isJsonObject(value)) return null;
  const found = value.members.find((one) => one.name === name);
  return found === undefined ? null : found.value;
}

/** Whether an object node writes that member at all (`null` is a value, absence is not). */
function writes(value: JsonValue | null, name: string): boolean {
  return value !== null && isJsonObject(value) && value.members.some((one) => one.name === name);
}

/** The members of a map, in the document's own order; nothing where the place is not a map. */
function entriesOf(value: JsonValue | null): { name: string; value: JsonValue }[] {
  if (value === null || !isJsonObject(value)) return [];
  return value.members.map((one) => ({ name: one.name, value: one.value }));
}

/** A member that must be a name, or `null`. */
function nameOf(value: JsonValue | null, name: string): string | null {
  const found = member(value, name);
  return typeof found === 'string' ? found : null;
}

/** The strings of a list member, ignoring whatever is not one. */
function stringsOf(value: JsonValue | null): string[] {
  if (value === null || !isJsonArray(value)) return [];
  return value.filter((one): one is string => typeof one === 'string');
}

/** The indices an endpoint writes, in the document's order. */
function indicesOf(endpoint: JsonValue | null): DrillIndex[] {
  return entriesOf(member(endpoint, INDICES)).map((one) => ({ name: one.name, written: one.value }));
}

/** The label of a grid point: `layer=0`, which is how §5.2 rule 2 writes one. */
export function pointLabel(indices: readonly IndexBinding[]): string {
  return indices.map((one) => `${one.name}=${pyStr(one.value)}`).join(',');
}

/**
 * The drill-in of one composition, or `null` where the document declares no such composition.
 *
 * It reads the tree, not the analysis — the reasons are feature 2.9's three, unchanged: a
 * document off the grammar, a template with no assignment and a document whose bases were not
 * gathered must all still be drawn.
 */
export function drillGraph(tree: JsonValue, options: DrillOptions): DrillGraph | null {
  const folded = options.folded ?? foldedGraph(tree, options.assignment === undefined ? {} : { assignment: options.assignment });
  const segments: PathSegment[] = [COMPOSITIONS, options.composition];
  const pointer = pointerOf(segments);
  const box = folded.byPointer.get(pointer);
  const declared = member(member(tree, COMPOSITIONS), options.composition);
  if (box === undefined || declared === null) return null;
  const quantities = folded.quantities;

  const grid = gridOf(member(declared, INDICES), quantities);
  const ranges: DrillRange[] = box.ranges.map((range) => ({
    ...range,
    values: grid === null ? null : (grid.ranges[grid.names.indexOf(range.name)] ?? null),
  }));
  const points = pointsOf(grid);

  const sitesByName = new Map(box.children.map((child) => [child.name, child]));
  const edges: DrillEdge[] = [];

  /** One end of a scoped rule: a `site_endpoint`, or a top-level `value_endpoint`. */
  const scopedEnd = (endpoint: JsonValue | null): DrillEnd | null => {
    const port = nameOf(endpoint, PORT);
    if (port === null) return null;
    const site = nameOf(endpoint, SITE);
    if (site !== null) {
      const indices = indicesOf(endpoint);
      return {
        name: site,
        port,
        site: sitesByName.has(site) ? pointerOf([...segments, INSTANCES, site]) : null,
        composition: options.composition,
        indices,
        overridden: indices.length > 0,
      };
    }
    return outsideEnd(endpoint, options.composition, sitesByName, segments);
  };

  for (const one of entriesOf(member(member(declared, BINDINGS), VALUES))) {
    const at: PathSegment[] = [...segments, BINDINGS, VALUES, one.name];
    edges.push({
      rule: one.name,
      hoisted: hoistedRule(options.composition, one.name),
      pointer: pointerOf(at),
      segments: at,
      from: scopedEnd(member(one.value, FROM)),
      to: scopedEnd(member(one.value, TO)),
      guard: writes(one.value, WHEN) ? member(one.value, WHEN) : null,
      scoped: true,
      interface: false,
    });
  }

  // The boundary: a top-level rule or an interface's wire naming a site of this composition.
  // `foldedGraph` has already read every one of them, with the box each end hangs on, so the
  // reading is one `boundary` flag and not a second walk of `bindings/values`.
  for (const edge of folded.edges) {
    const touches = [edge.from, edge.to].some((end) => end !== null && end.boundary && end.box === pointer);
    if (!touches) continue;
    edges.push({
      rule: edge.rule,
      hoisted: edge.rule,
      pointer: edge.pointer,
      segments: edge.segments,
      from: edge.from === null ? null : boundaryEnd(edge.from, pointer, options.composition),
      to: edge.to === null ? null : boundaryEnd(edge.to, pointer, options.composition),
      guard: edge.guard,
      scoped: false,
      interface: edge.interface,
    });
  }

  return {
    pointer,
    segments,
    name: box.name,
    families: stringsOf(member(declared, FAMILIES)),
    ranges,
    sites: box.children,
    edges,
    ghosts: ghostsOf(edges),
    terminals: terminalsOf(edges),
    points,
    count: box.count,
    held: box.held,
    quantities,
  };
}

/** The grid of a composition's indices, or `null` where a bound does not resolve. */
function gridOf(indices: JsonValue | null, quantities: Quantities): { names: string[]; ranges: bigint[][] } | null {
  if (indices === null || !isJsonObject(indices)) return null;
  try {
    return indexGrid(toPython(indices), quantities);
  } catch {
    return null;
  }
}

/** Every point of the grid, in the order `indexGrid` and `itertools.product` give them. */
function pointsOf(grid: { names: string[]; ranges: bigint[][] } | null): DrillPoint[] {
  if (grid === null) return [];
  let points: IndexBinding[][] = [[]];
  for (const [at, name] of grid.names.entries()) {
    const values = grid.ranges[at] ?? [];
    const next: IndexBinding[][] = [];
    for (const point of points) for (const value of values) next.push([...point, { name, value }]);
    points = next;
  }
  return points.map((indices) => ({ indices, label: pointLabel(indices) }));
}

/** An end of a scoped rule written as a top-level `value_endpoint`: it names something outside. */
function outsideEnd(
  endpoint: JsonValue | null,
  composition: string,
  sites: ReadonlyMap<string, FoldedNode>,
  segments: readonly PathSegment[],
): DrillEnd | null {
  const port = nameOf(endpoint, PORT);
  const selector = member(endpoint, INSTANCE);
  if (port === null || selector === null || !isJsonObject(selector)) return null;
  const name = nameOf(selector, INSTANCE);
  if (name === null) return null;
  const named = nameOf(selector, COMPOSITION);
  const inside = named === composition && sites.has(name);
  const indices = indicesOf(selector);
  return {
    name,
    port,
    site: inside ? pointerOf([...segments, INSTANCES, name]) : null,
    composition: named,
    indices,
    overridden: inside && indices.length > 0,
  };
}

/** One end of a boundary edge, read off the folded graph's own handle. */
function boundaryEnd(end: FoldedHandle, box: string, composition: string): DrillEnd {
  const inside = end.boundary && end.box === box;
  return {
    name: end.name,
    port: end.port,
    site: inside ? end.site : null,
    composition: inside ? composition : null,
    indices: end.indices.map((one) => ({ name: one.name, written: one.written })),
    // A boundary edge names a concrete iteration from outside; it is not an override of the
    // current one, so it never draws a ghost — it draws the pinned terminal of §4.8 instead.
    overridden: false,
  };
}

/** The ghost columns: one per (site, side, override expression), generated from the rules (§4.8). */
function ghostsOf(edges: readonly DrillEdge[]): DrillGhost[] {
  const found = new Map<string, { ghost: DrillGhost; edges: string[] }>();
  for (const edge of edges) {
    for (const [end, side] of [
      [edge.from, 'left'],
      [edge.to, 'right'],
    ] as const) {
      if (end === null || !end.overridden || end.site === null) continue;
      const id = `ghost:${side}:${end.name}:${indexKey(end.indices)}`;
      const held = found.get(id);
      if (held === undefined) {
        found.set(id, {
          ghost: { id, name: end.name, site: end.site, side, indices: end.indices, edges: [] },
          edges: [edge.pointer],
        });
      } else {
        held.edges.push(edge.pointer);
      }
    }
  }
  return [...found.values()].map(({ ghost, edges: through }) => ({ ...ghost, edges: through }));
}

/** The pinned terminals: one per (outside end, side, indices), carrying the rules pinned on it. */
function terminalsOf(edges: readonly DrillEdge[]): DrillTerminal[] {
  const found = new Map<string, { terminal: DrillTerminal; links: DrillLink[] }>();
  for (const edge of edges) {
    // A rule with both ends inside is an edge between two cards and no terminal at all; one with
    // neither is not this composition's. A **scoped** rule may name a root instance — the grammar
    // admits it (`scoped_value_endpoint`'s second alternative) and three corpus documents write
    // one (`gemma3n-kvshare`'s `aux`, `whisper-large-v3`'s `cross_source`,
    // `voxtral-realtime`'s `time_scale.condition`) — so it is pinned like any other crossing.
    if (edge.from?.site != null && edge.to?.site != null) continue;
    if (edge.from?.site == null && edge.to?.site == null) continue;
    const entering = edge.to !== null && edge.to.site !== null;
    const inside = entering ? edge.to : edge.from;
    const outside = entering ? edge.from : edge.to;
    if (inside === null || inside.site === null) continue;
    const side = entering ? 'left' : 'right';
    const named = outside === null ? '' : outside.port === '' ? outside.name : `${outside.name}.${outside.port}`;
    const id = `terminal:${side}:${named}:${indexKey(inside.indices)}`;
    const link: DrillLink = {
      rule: edge.rule,
      pointer: edge.pointer,
      name: inside.name,
      site: inside.site,
      port: inside.port,
    };
    const held = found.get(id);
    if (held === undefined) {
      found.set(id, {
        terminal: { id, side, outside: named, indices: inside.indices, links: [] },
        links: [link],
      });
    } else {
      held.links.push(link);
    }
  }
  return [...found.values()].map(({ terminal, links }) => ({ ...terminal, links }));
}

/** A key for a set of written index expressions: the serialized tree would do, the shape is enough. */
function indexKey(indices: readonly DrillIndex[]): string {
  return indices.map((one) => `${one.name}=${JSON.stringify(one.written)}`).join(',');
}

// ---------------------------------------------------------------------------------------------
// What exists at an index: D1, and the guard beside it.
// ---------------------------------------------------------------------------------------------

/** What D1 says exists at each point of the grid, and what each guard says about itself. */
export interface DrillPresence {
  /** The points, in the grid's order — the columns of the alternation strip. */
  readonly points: readonly DrillPoint[];
  /** Per site name, the labels of the points D1 emitted a node for. */
  readonly sites: ReadonlyMap<string, ReadonlySet<string>>;
  /** Per edge (by the place its rule is written at), the labels of the points it fires at. */
  readonly edges: ReadonlyMap<string, ReadonlySet<string>>;
  /**
   * The truth of each guard at each point: `true`, `false`, or `null` where it does not resolve.
   *
   * Keyed by the site's name for a site's guard and by the rule's written place for an edge's, so
   * that S4's `when layer ≥ 1 — false` and S5's "guard fails" read one answer.
   */
  readonly guards: ReadonlyMap<string, ReadonlyMap<string, boolean | null>>;
}

/** A presence nothing was read for: what the strip shows before a derivation (§5.4's freshness). */
export const NO_PRESENCE: DrillPresence = {
  points: [],
  sites: new Map(),
  edges: new Map(),
  guards: new Map(),
};

/**
 * What exists at each point of the grid — the scrubber's and the alternation strip's own reading.
 *
 * The **nodes** are D1's: `expanded` is a derived document or a `--d1` document, and a site is
 * present at a point exactly when `d1.nodes` holds the identifier §5.2 rule 2 gives it. Nothing
 * is counted or inferred: `attn 24 of 30` is twenty-four keys of that map.
 *
 * An **edge** is present at a point when its own guard fires there and both of its ends name a
 * node D1 emitted — which is `d1.emit`'s own rule (`loop_envs` keeps the environments where the
 * guard is true, and §5.2 rule 3 drops a binding whose site is absent). A boundary rule belongs
 * to the iteration its inside end names, and it is present at that one point.
 *
 * A **guard**'s truth is evaluated by `expr/model.ts` under the point's environment — the same
 * evaluator, the same environment. `test/describe/drill.test.ts` holds the two readings to each
 * other over the corpus: every guarded site is present in D1 exactly where its guard is true.
 */
export function drillPresence(drill: DrillGraph, expanded: PyValue | null): DrillPresence {
  if (expanded === null || drill.points.length === 0) return NO_PRESENCE;
  const nodes = nodesOf(expanded);
  if (nodes === null) return NO_PRESENCE;

  const guards = new Map<string, Map<string, boolean | null>>();
  const sites = new Map<string, Set<string>>();
  for (const site of drill.sites) {
    const present = new Set<string>();
    const truths = new Map<string, boolean | null>();
    const guard = site.guard === null ? null : toPython(site.guard);
    for (const point of drill.points) {
      const where = whereOfSite(generatedSite(drill.name, site.name, point.indices));
      if (nodes.has(where)) present.add(point.label);
      if (guard !== null) truths.set(point.label, truthOf(guard, drill.quantities, envOf(point)));
    }
    sites.set(site.name, present);
    if (guard !== null) guards.set(site.name, truths);
  }

  const edges = new Map<string, Set<string>>();
  for (const edge of drill.edges) {
    const present = new Set<string>();
    const truths = new Map<string, boolean | null>();
    const guard = edge.guard === null ? null : toPython(edge.guard);
    if (edge.scoped) {
      for (const point of drill.points) {
        const env = envOf(point);
        const truth = guard === null ? true : truthOf(guard, drill.quantities, env);
        if (guard !== null) truths.set(point.label, truth);
        if (truth !== true) continue;
        if (endsExist(edge, drill, nodes, env)) present.add(point.label);
      }
    } else {
      // A boundary rule is emitted once, by the top level; it belongs to the iteration its inside
      // end names, which is what S4 pins the terminal at (`at layer = 0`).
      const at = boundaryPoint(edge, drill);
      if (at !== null && sites.get(insideOf(edge)?.name ?? '')?.has(at) === true) present.add(at);
      if (guard !== null) for (const point of drill.points) truths.set(point.label, truthOf(guard, drill.quantities, envOf(point)));
    }
    edges.set(edge.pointer, present);
    if (guard !== null) guards.set(edge.pointer, truths);
  }

  return { points: drill.points, sites, edges, guards };
}

/** The identifiers D1 emitted a node for, or `null` where the value is no expanded graph. */
function nodesOf(expanded: PyValue): ReadonlySet<string> | null {
  const graph = has(expanded, D1) ? get(expanded, D1) : expanded;
  if (graph === undefined || !has(graph, NODES)) return null;
  const nodes = get(graph, NODES);
  if (nodes === undefined || typeof nodes !== 'object' || nodes === null) return null;
  return new Set(Object.keys(nodes));
}

/** The environment of one point, as the emitter binds the composition's indices. */
function envOf(point: DrillPoint): Env {
  return new Map(point.indices.map((one) => [one.name, one.value]));
}

/** A condition's truth, or `null` where it does not resolve (§5.2 rule 6's own answer). */
function truthOf(condition: PyValue, quantities: Quantities, env: Env): boolean | null {
  try {
    const truth = modelCondition(condition, quantities, env);
    return truth === UNRESOLVED ? null : truthy(truth);
  } catch {
    return null;
  }
}

/** Whether both ends of a scoped edge name a node D1 emitted, under this environment. */
function endsExist(
  edge: DrillEdge,
  drill: DrillGraph,
  nodes: ReadonlySet<string>,
  env: Env,
): boolean {
  for (const end of [edge.from, edge.to]) {
    if (end === null) return false;
    const where = whereOfEnd(end, drill, env);
    if (where === null || !nodes.has(where)) return false;
  }
  return true;
}

/** The identifier of the node one end names under an environment, or `null` where it does not resolve. */
function whereOfEnd(end: DrillEnd, drill: DrillGraph, env: Env): string | null {
  const written = new Map(end.indices.map((one) => [one.name, one.written]));
  if (end.composition === null) {
    // A root instance: emitted under its own name, with no indices at all (§5.2 rule 2).
    return whereOfSite(rootSite(end.name));
  }
  const indices: IndexBinding[] = [];
  if (end.composition === drill.name) {
    // A site of this composition: every index the composition declares, at the current value
    // unless the endpoint overrides it — which is `selectSite`'s own reading of a site endpoint.
    for (const range of drill.ranges) {
      const expression = written.get(range.name);
      const value =
        expression === undefined
          ? (env.get(range.name) ?? null)
          : valueOf(expression, drill.quantities, env);
      if (value === null) return null;
      indices.push({ name: range.name, value });
    }
  } else {
    for (const one of end.indices) {
      const value = valueOf(one.written, drill.quantities, env);
      if (value === null) return null;
      indices.push({ name: one.name, value });
    }
  }
  return whereOfSite(generatedSite(end.composition, end.name, indices));
}

/** An expression under the quantities and an environment, or `null` where it does not resolve. */
function valueOf(expression: JsonValue, quantities: Quantities, env: Env): PyValue | null {
  try {
    const value = modelValue(toPython(expression), quantities, env);
    return value === UNRESOLVED ? null : value;
  } catch {
    return null;
  }
}

/** The end of a boundary edge that is inside the composition. */
function insideOf(edge: DrillEdge): DrillEnd | null {
  if (edge.to !== null && edge.to.site !== null) return edge.to;
  if (edge.from !== null && edge.from.site !== null) return edge.from;
  return null;
}

/** The point a boundary edge belongs to: the iteration its inside end names. */
function boundaryPoint(edge: DrillEdge, drill: DrillGraph): string | null {
  const inside = insideOf(edge);
  if (inside === null) return null;
  const indices: IndexBinding[] = [];
  const written = new Map(inside.indices.map((one) => [one.name, one.written]));
  for (const range of drill.ranges) {
    const expression = written.get(range.name);
    if (expression === undefined) return null;
    const value = valueOf(expression, drill.quantities, new Map());
    if (value === null) return null;
    indices.push({ name: range.name, value });
  }
  return pointLabel(generatedSite(drill.name, inside.name, indices).indices);
}

/** How many points a site is present at, and how many there are — S5's `24 of 30`. */
export function alternationCount(presence: DrillPresence, site: string): { at: number; of: number } {
  return { at: presence.sites.get(site)?.size ?? 0, of: presence.points.length };
}

// ---------------------------------------------------------------------------------------------
// The writers: what a gesture of §4.8 and §4.20 puts in the document.
// ---------------------------------------------------------------------------------------------

/** A whole number as the grammar writes one, with the lexeme Python would print. */
function wholeNumber(value: bigint): JsonValue {
  return { kind: 'number', value: Number(value), real: false, lexeme: value.toString() };
}

/** A `site_endpoint`: a site of the enclosing composition, at the current indices unless told. */
export function scopedEndpoint(
  site: string,
  port: string,
  indices: readonly DrillIndex[] = [],
): JsonObject {
  const members = [{ name: SITE, value: site as JsonValue }];
  if (indices.length > 0) {
    members.push({
      name: INDICES,
      value: jsonObject(indices.map((one) => ({ name: one.name, value: one.written }))),
    });
  }
  members.push({ name: PORT, value: port });
  return jsonObject(members);
}

/**
 * The override §4.8 names for "Connect from previous iteration…":
 * `{"op": "subtract", "args": [{"index": "layer"}, {"literal": 1}]}`.
 */
export function previousIteration(index: string): JsonObject {
  return jsonObject([
    { name: OP, value: SUBTRACT },
    { name: ARGS, value: [jsonObject([{ name: INDEX, value: index }]), literalIndex(1n)] },
  ]);
}

/**
 * The guard that override proposes: `layer ≥ 1` — "a proposal the user confirms, since the guard
 * states a fact of its own" (§4.8).
 */
export function reachedIteration(index: string, value: bigint = 1n): JsonObject {
  return jsonObject([
    {
      name: COMPARE,
      value: jsonObject([
        { name: OPERATOR, value: GREATER_OR_EQUAL },
        { name: LEFT, value: jsonObject([{ name: INDEX, value: index }]) },
        { name: RIGHT, value: jsonObject([{ name: 'literal', value: wholeNumber(value) }]) },
      ]),
    },
  ]);
}

/**
 * The complement of a guard: `{"not": …}` — §4.20's "Duplicate with complementary guard".
 *
 * It is the negation and not a rewriting of the comparison. S5 draws `ffn` with `layer ≥ 10`
 * beside `ffn_sparse`'s `layer < 10`, which is the corpus's own pair; flipping an operator to
 * produce it would be a logical transformation the editor invented, which feature 2.10 already
 * refused for the same reason (`mask ≠ chunked` against the declaration's `mask = chunked`).
 */
export function complementOf(condition: JsonValue): JsonObject {
  return jsonObject([{ name: NOT, value: condition }]);
}

/** A `finite_range` over whole numbers: what a new composition's index runs over. */
export function finiteRange(start: bigint, stop: bigint, step: bigint): JsonObject {
  return jsonObject([
    { name: START, value: literalIndex(start) },
    { name: STOP, value: literalIndex(stop) },
    { name: STEP, value: literalIndex(step) },
  ]);
}

/** The `index_expression` that names an index: `{"index": "layer"}`. */
export function indexExpression(name: string): JsonObject {
  return jsonObject([{ name: INDEX, value: name }]);
}

/** The expression `<stop> − 1`: the last iteration of a range, as an exit boundary edge names it. */
export function lastIteration(stop: JsonValue): JsonObject {
  return jsonObject([
    { name: OP, value: SUBTRACT },
    { name: ARGS, value: [stop, literalIndex(1n)] },
  ]);
}

/** A `generated_instance_selector`: how a top-level endpoint names a site at an index. */
export function generatedSelector(
  composition: string,
  site: string,
  indices: readonly DrillIndex[],
): JsonObject {
  return jsonObject([
    { name: KIND, value: GENERATED },
    { name: COMPOSITION, value: composition },
    { name: INSTANCE, value: site },
    {
      name: INDICES,
      value: jsonObject(indices.map((one) => ({ name: one.name, value: one.written }))),
    },
  ]);
}

/** A top-level `value_endpoint` naming a site at an index. */
export function generatedEndpoint(
  composition: string,
  site: string,
  port: string,
  indices: readonly DrillIndex[],
): JsonObject {
  return jsonObject([
    { name: INSTANCE, value: generatedSelector(composition, site, indices) },
    { name: PORT, value: port },
  ]);
}

/**
 * The members a new composition is written with — the schema's required three, and no more.
 *
 * The family is the composition's **own name**, which is the corpus's own idiom (`decoder`'s
 * families are `["decoder"]` in every document that has one) and the only proposal the grammar
 * leaves room for: `family_list` requires at least one, so a composition written with none is off
 * the schema. It is a proposal like §9 Q4's, changed in the sheet like any other name.
 */
export function compositionSkeleton(
  name: string,
  index: string,
  stop: bigint,
): Record<string, JsonValue> {
  return {
    [INDICES]: jsonObject([{ name: index, value: finiteRange(0n, stop, 1n) }]),
    [FAMILIES]: [name],
    [INSTANCES]: jsonObject([]),
  };
}

/**
 * What §4.20's "Extract to Composition" proposes to call the composition and its index.
 *
 * §4.20 says the move makes "a new composition with a fresh index" and does not say what either is
 * called — the document is the author's. So the editor proposes, exactly as §9 Q4 decided for an
 * instance's name and its family ({@link proposedName}, {@link proposedFamily}), and the proposal
 * lives here beside those two: `layer` is a value of the unit schema's axis natures as well as the
 * name every composition of the corpus gives its index, and catching rule (b) keeps a word of a
 * schema's vocabulary out of the interface's own source whatever it is being used for.
 */
export const PROPOSED_COMPOSITION = 'block';

/** The index a created composition declares — what every composition of the corpus calls its own. */
export const PROPOSED_INDEX = 'layer';

/** The map a composition is written into. */
export const COMPOSITION_MAP: readonly PathSegment[] = [COMPOSITIONS];

/** The map a composition's sites are written into, under its own place. */
export const SITE_MAP: readonly PathSegment[] = [INSTANCES];

/** The map a composition's scoped value rules are written into, under its own place. */
export const SCOPED_VALUES: readonly PathSegment[] = [BINDINGS, VALUES];

/** The member of a binding rule that carries its guard. */
export const GUARD = WHEN;

// ---------------------------------------------------------------------------------------------
// §4.20's two moves: Extract to Composition, and Add to Composition.
// ---------------------------------------------------------------------------------------------

/** One rule that moves inside the composition, rewritten as the scoped form of itself. */
export interface AbsorbedRule {
  /** Which map of `bindings` it is written in: the schema's own member name. */
  readonly map: string;
  /** The rule's name, as it will be written inside — uniquified against what is already there. */
  readonly name: string;
  /** Where it is written today. */
  readonly from: readonly PathSegment[];
  /** The rule as it will be written inside. */
  readonly value: JsonValue;
}

/** One place of a rule that stays at the top level, with the value it takes now. */
export interface RewrittenPlace {
  readonly path: readonly PathSegment[];
  readonly value: JsonValue;
}

/** What §4.20's move writes, computed against the document as it stands. */
export interface CompositionMove {
  /** The composition the instances move into. */
  readonly composition: string;
  /** Whether it is created by this move. */
  readonly created: boolean;
  /** The index whose bounds a boundary edge names. */
  readonly index: string;
  /** The definition of the composition, where it is created. */
  readonly definition: Record<string, JsonValue> | null;
  /** The instances that move, in the document's order, with the name each takes inside. */
  readonly moved: readonly { readonly from: string; readonly name: string; readonly value: JsonValue }[];
  /** The rules that move inside with them. */
  readonly absorbed: readonly AbsorbedRule[];
  /** The endpoints of the rules that stay, rewritten to name the site at an index. */
  readonly rewritten: readonly RewrittenPlace[];
  /** What the move could not decide, in the editor's own words — the toast's and the log's line. */
  readonly notes: readonly string[];
}

/** What {@link compositionMove} is asked for. */
export interface MoveRequest {
  /** The root instances to move, by name. */
  readonly instances: readonly string[];
  /** The composition to move them into; a name the document has not got is created. */
  readonly composition: string;
  /** The index a created composition declares (the proposal `layer`). */
  readonly index?: string;
}

/**
 * §4.20's "Extract to Composition" and §4.7's "Add to Composition…", as one reading.
 *
 * > select root instances ▸ Extract to Composition: the instances become sites, edges among them
 * > become scoped rules, edges to the outside become boundary edges naming the first or last
 * > index (the editor asks which end is the entry and the exit when the graph leaves it
 * > ambiguous), private identities become scoped rules with `{index}` inserted into their
 * > locations where the physical names differ only by a number (proposed, confirmed).
 *
 * > Add to Composition… (moves a root instance into a composition as a site, rewriting the edges
 * > that named it into scoped rules where both ends are inside, the others into boundary edges)
 *
 * The two are the same move over a composition that is created or already there, which is why
 * they are one function. Three decisions are written down here rather than in the interface:
 *
 *  - **A created composition runs once** (`start 0, stop 1, step 1`). The selection is one copy of
 *    the pattern, so one iteration is the only bound under which the extracted document denotes
 *    the graph the original denoted — and the author sets the real bound in the index strip, with
 *    every figure following. A bound invented from the selection would change what the document
 *    means without anybody saying so.
 *  - **Which end is the entry is the edge's own direction.** An edge from outside in enters at the
 *    first index; one from inside out leaves at the last. §4.20 anticipates an ambiguity, and a
 *    *value* binding has none: it is directed. (An instance that both receives from outside and
 *    sends outside gets one of each, which is what the corpus's own compositions carry.)
 *  - **A location moves unchanged**, and the move says so. §4.20's proposal inserts `{index}`
 *    "where the physical names differ only by a number" — a comparison between the *iterations*
 *    of a repeated block, of which an extraction of one copy has none. The note names the
 *    location editor (§4.14), which is where an index is put into a name.
 */
export function compositionMove(tree: JsonValue, request: MoveRequest): CompositionMove | null {
  const moving = new Set(request.instances);
  if (moving.size === 0) return null;
  const roots = member(tree, INSTANCES);
  const compositions = member(tree, COMPOSITIONS);
  const existing = member(compositions, request.composition);
  const created = existing === null;
  const index = created
    ? (request.index ?? 'i')
    : (entriesOf(member(existing, INDICES))[0]?.name ?? request.index ?? 'i');
  const taken = new Set(entriesOf(member(existing, INSTANCES)).map((one) => one.name));
  const moved: { from: string; name: string; value: JsonValue }[] = [];
  for (const one of entriesOf(roots)) {
    if (!moving.has(one.name)) continue;
    const name = uniqueName(one.name, taken);
    taken.add(name);
    moved.push({ from: one.name, name, value: one.value });
  }
  if (moved.length === 0) return null;
  const renamed = new Map(moved.map((one) => [one.from, one.name]));

  // Where a boundary edge names the composition: the first index, and the last.
  const ranges = created ? null : entriesOf(member(existing, INDICES));
  const first: JsonValue = created
    ? literalIndex(0n)
    : (member(ranges?.find((one) => one.name === index)?.value ?? null, START) ?? literalIndex(0n));
  const stop: JsonValue = created
    ? literalIndex(1n)
    : (member(ranges?.find((one) => one.name === index)?.value ?? null, STOP) ?? literalIndex(1n));
  const last: JsonValue = lastIteration(stop);

  const absorbed: AbsorbedRule[] = [];
  const rewritten: RewrittenPlace[] = [];
  const notes: string[] = [];
  const scopedTaken = new Set(entriesOf(member(member(existing, BINDINGS), VALUES)).map((one) => one.name));

  for (const rule of entriesOf(member(member(tree, BINDINGS), VALUES))) {
    const at: PathSegment[] = [BINDINGS, VALUES, rule.name];
    const ends = [FROM, TO].map((side) => ({ side, endpoint: member(rule.value, side) }));
    const inside = ends.map(({ endpoint }) => rootNameOf(endpoint)).map((name) => name !== null && moving.has(name));
    if (!inside[0] && !inside[1]) continue;
    if (inside[0] === true && inside[1] === true) {
      const name = uniqueName(rule.name, scopedTaken);
      scopedTaken.add(name);
      absorbed.push({
        map: VALUES,
        name,
        from: at,
        value: scopedRule(rule.value, renamed),
      });
      continue;
    }
    for (const [position, { side, endpoint }] of ends.entries()) {
      if (inside[position] !== true || endpoint === null) continue;
      const name = rootNameOf(endpoint);
      const port = nameOf(endpoint, PORT);
      if (name === null || port === null) continue;
      // `to` is the fed end: an edge whose *destination* moved inside enters the composition.
      const entering = side === TO;
      rewritten.push({
        path: [...at, side],
        value: generatedEndpoint(request.composition, renamed.get(name) ?? name, port, [
          { name: index, written: entering ? first : last },
        ]),
      });
    }
  }

  // The interfaces name endpoints too, and one naming a moved instance becomes a boundary edge of
  // the same kind: §4.15's `to` list and an output's `from`.
  const interfaces = member(tree, INTERFACES);
  for (const one of entriesOf(member(interfaces, INPUTS))) {
    const listed = member(one.value, TO);
    if (listed === null || !isJsonArray(listed)) continue;
    listed.forEach((endpoint, position) => {
      const name = rootNameOf(endpoint);
      const port = nameOf(endpoint, PORT);
      if (name === null || port === null || !moving.has(name)) return;
      rewritten.push({
        path: [INTERFACES, INPUTS, one.name, TO, position],
        value: generatedEndpoint(request.composition, renamed.get(name) ?? name, port, [
          { name: index, written: first },
        ]),
      });
    });
  }
  for (const one of entriesOf(member(interfaces, OUTPUTS))) {
    const endpoint = member(one.value, FROM);
    const name = rootNameOf(endpoint);
    const port = nameOf(endpoint, PORT);
    if (name === null || port === null || !moving.has(name)) continue;
    rewritten.push({
      path: [INTERFACES, OUTPUTS, one.name, FROM],
      value: generatedEndpoint(request.composition, renamed.get(name) ?? name, port, [
        { name: index, written: last },
      ]),
    });
  }

  // The parameter, constant and state rules: one whose every member names a moved instance moves
  // inside with them; one that mixes stays, with the members that moved rewritten as selectors.
  for (const map of entriesOf(member(tree, BINDINGS))) {
    if (map.name === VALUES) continue;
    const held = new Set(entriesOf(member(member(existing, BINDINGS), map.name)).map((one) => one.name));
    for (const rule of entriesOf(map.value)) {
      const at: PathSegment[] = [BINDINGS, map.name, rule.name];
      const members = member(rule.value, 'members');
      if (members === null || !isJsonArray(members)) continue;
      const named = members.map((one) => rootNameOf(one));
      const mine = named.map((name) => name !== null && moving.has(name));
      if (!mine.includes(true)) continue;
      if (mine.every((one) => one)) {
        const name = uniqueName(rule.name, held);
        held.add(name);
        absorbed.push({ map: map.name, name, from: at, value: scopedMembers(rule.value, renamed) });
        if (writes(rule.value, 'location')) {
          notes.push(
            `the location of ${rule.name} moves as it is written: a name carrying a fixed number is the same tensor at every iteration, and the location editor is where an index goes into one`,
          );
        }
        continue;
      }
      members.forEach((one, position) => {
        const name = named[position] ?? null;
        if (mine[position] !== true || name === null) return;
        const slot = slotMemberOf(one);
        if (slot === null) return;
        rewritten.push({
          path: [...at, 'members', position],
          value: jsonObject([
            { name: INSTANCE, value: generatedSelector(request.composition, renamed.get(name) ?? name, [{ name: index, written: first }]) },
            { name: slot.member, value: slot.name },
          ]),
        });
      });
    }
  }

  return {
    composition: request.composition,
    created,
    index,
    definition: created ? compositionSkeleton(request.composition, index, 1n) : null,
    moved,
    absorbed,
    rewritten,
    notes,
  };
}

/** The root instance one endpoint names, or `null` where it names something else. */
function rootNameOf(endpoint: JsonValue | null): string | null {
  const selector = member(endpoint, INSTANCE);
  if (selector === null || !isJsonObject(selector)) return null;
  if (nameOf(selector, COMPOSITION) !== null) return null;
  if (member(selector, KIND) === GENERATED) return null;
  return nameOf(selector, INSTANCE);
}

/** Which member of an endpoint names the slot it binds: `parameter`, `constant` or `state`. */
function slotMemberOf(endpoint: JsonValue): { member: string; name: string } | null {
  if (!isJsonObject(endpoint)) return null;
  for (const one of endpoint.members) {
    if (one.name === INSTANCE) continue;
    if (typeof one.value === 'string') return { member: one.name, name: one.value };
  }
  return null;
}

/** A top-level value rule, written as the composition's own: the endpoints become site endpoints. */
function scopedRule(rule: JsonValue, renamed: ReadonlyMap<string, string>): JsonValue {
  if (!isJsonObject(rule)) return rule;
  return jsonObject(
    rule.members.map((one) => {
      if (one.name !== FROM && one.name !== TO) return one;
      const name = rootNameOf(one.value);
      const port = nameOf(one.value, PORT);
      if (name === null || port === null) return one;
      return { name: one.name, value: scopedEndpoint(renamed.get(name) ?? name, port) };
    }),
  );
}

/** A top-level parameter, constant or state rule, written as the composition's own. */
function scopedMembers(rule: JsonValue, renamed: ReadonlyMap<string, string>): JsonValue {
  if (!isJsonObject(rule)) return rule;
  return jsonObject(
    rule.members.map((one) => {
      if (one.name !== 'members' || !isJsonArray(one.value)) return one;
      return {
        name: one.name,
        value: one.value.map((endpoint) => {
          const name = rootNameOf(endpoint);
          const slot = slotMemberOf(endpoint);
          if (name === null || slot === null) return endpoint;
          return jsonObject([
            { name: SITE, value: renamed.get(name) ?? name },
            { name: slot.member, value: slot.name },
          ]);
        }),
      };
    }),
  );
}

/** A name nothing in the set has, by appending a digit — the store's own `unique`, in the core. */
function uniqueName(name: string, taken: ReadonlySet<string>): string {
  if (!taken.has(name)) return name;
  for (let at = 2; ; at += 1) {
    const next = `${name}_${String(at)}`;
    if (!taken.has(next)) return next;
  }
}
