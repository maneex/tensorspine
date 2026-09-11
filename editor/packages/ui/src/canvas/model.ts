/**
 * What the folded canvas draws — plan §4.7, artboards S1, S2 and S3.
 *
 * One function, {@link canvasModel}, turns four answers into the boxes and wires a component
 * renders, and it computes none of them:
 *
 * | What is drawn | Who answers it |
 * |---|---|
 * | which boxes there are, which box an edge's end hangs on, a composition's count | the core's folded reading (`foldedGraph`) |
 * | present ports, slot chips, the structural summary | the core's `describe` |
 * | the derived line, the located tick, the identity links, an edge's type | the derived document, read by the core (`derivedFacts`) |
 * | the problem dot, the hollow-red handle, the amber one | the Problems of §4.17, projected |
 * | the role each box is drawn in, the symbols a guard prints with | `presentation.json` |
 *
 * The component inventory's §7 is the rule: *a component that shows a fact never computes it*.
 * The only arithmetic here is geometry — how tall a card is, which is the interface's own
 * business and nobody else's.
 *
 * **The handles are projected, never re-checked.** "An unfed present input handle is drawn hollow
 * red, an unconsumed output amber — the core's V7 and V13 problems projected onto handles, not a
 * second check." `describe` answers `present`, `fedBy` and `consumed` off the very maps
 * `analyse`'s V7 and V13 block reads (`producers`, `consumed`), so the handle and the Problems row
 * are one reading in two places; `test/canvas/model.test.ts` holds them to each other on every
 * corpus document, which is what makes that sentence true rather than hopeful.
 */
import {
  derivedFacts,
  noDerivedFacts,
  pyStr,
  slotKey,
  toJsonValue,
  type DerivedFacts,
  type FoldedGraph,
  type SiteFigures,
  type FoldedNode,
  type JsonValue,
  type PyValue,
  type SchemaRegistry,
  type SiteDescription,
} from '@tensorspine/lang';
import type { Facts, Problem } from '@tensorspine/lang/api';
import { pointerOf, type Path, type SchemaShapes } from '@tensorspine/store';

import { sizeText } from '../documents/figures.js';
import { printValue, type PrintContext } from '../expressions/index.js';
import type { Binding, Presentation } from '../presentation/index.js';

/** The roles `presentation.json` gives a construct on the canvas. */
export const ROLE = { node: 'node', group: 'group', edge: 'edge', terminal: 'terminal' } as const;

/** The sides a terminal sits on. */
export const SIDE = { left: 'left', right: 'right' } as const;

/** The state a handle is drawn in — V7 and V13, projected (§4.7). */
export type HandleState =
  /** Fed (an input) or consumed (an output). */
  | 'bound'
  /** A present input nothing feeds: V7's row, drawn hollow red. */
  | 'unfed'
  /** A present output nothing consumes: V13's advisory, drawn amber. */
  | 'unused';

/** How a slot chip is drawn — S2's own six states. */
export interface CanvasSlot {
  readonly name: string;
  /** `parameter`, `constant` or `state`: which map of the primitive declares it. */
  readonly kind: 'parameter' | 'constant' | 'state';
  /** The identity instance it belongs to, or `null` while nothing binds it (V7). */
  readonly identity: string | null;
  /** Whether the identity has several members — the `⇄` of S2. */
  readonly shared: boolean;
  /** Whether a physical tensor names it — S2's located tick, from D3. */
  readonly located: boolean;
}

/** One port handle of a card. */
export interface CanvasPort {
  readonly name: string;
  readonly side: 'inputs' | 'outputs';
  readonly state: HandleState;
  /** What the hover shows: the kind, the evaluated shape and the role (§4.7). */
  readonly title: string;
}

/** A boundary handle of a collapsed composition — `attn_n[layer=0].input` (§4.7, D8). */
export interface CanvasHandle {
  /** The label, printed: the site, its indices, and the port. */
  readonly label: string;
  /** Which side of the box it sits on. */
  readonly side: 'left' | 'right';
  /** The site's own pointer, so a click selects the site the handle names. */
  readonly site: string;
  readonly port: string;
}

/** One box of the canvas. */
export interface CanvasBox {
  /** The place, as an RFC 6901 pointer — the identity, and the sidecar's key (D6). */
  readonly pointer: string;
  readonly path: Path;
  readonly name: string;
  /** The role `presentation.json` binds at the box's own place. */
  readonly role: string;
  /** A terminal's side. */
  readonly side: string | null;
  /** The composition a site sits in; `null` at the top level. */
  readonly parent: string | null;
  /** `primitive@version`, or `null` for a composition and a terminal. */
  readonly primitive: string | null;
  readonly version: string | null;
  readonly families: readonly string[];
  /** The guard, printed in §4.13's text form; `null` where the box writes none. */
  readonly guard: string | null;
  /** The structural arguments, `name=value`, in the primitive's own declaration order. */
  readonly summary: string;
  readonly inputs: readonly CanvasPort[];
  readonly outputs: readonly CanvasPort[];
  readonly slots: readonly CanvasSlot[];
  /** A composition's header: `layer ∈ [0, 32)`, printed. */
  readonly range: string | null;
  /** `×32`, or `×?` where a bound does not resolve (§4.7). */
  readonly count: string | null;
  /** S3's `6 sites · 8 scoped edges · …`. */
  readonly held: string | null;
  /** A collapsed composition's boundary handles. */
  readonly handles: readonly CanvasHandle[];
  /** A terminal's own members: `token`, `generative`. */
  readonly badges: readonly string[];
  /** The derived line, or `null` where nothing has been derived for this box. */
  readonly derived: string | null;
  /** Whether the derived line was computed for an older revision (§5.4's freshness). */
  readonly stale: boolean;
  /** The heaviest problem whose pointer falls at or under this box; `null` where there is none. */
  readonly problem: 'error' | 'warning' | 'notice' | null;
  /** How many problems fall under it, for the dot's own label. */
  readonly problems: number;
  /** Whether a composition is drawn collapsed. */
  readonly collapsed: boolean;
  /** Whether the analysis says nothing about this box: a guard removed it, or its primitive did not resolve. */
  readonly facts: boolean;
  /** How D1 and a refusal name the site this box stands for; `null` for a group and a terminal. */
  readonly where: string | null;
  /**
   * Whether the primitive this box pins is a **template** primitive — §4.7's `▣`.
   *
   * Whether a primitive is a template is a fact about the *library* and not about the document
   * (the core's `template_primitives`), so it is handed in rather than read off the tree: with no
   * library gathered the mark is simply absent, which is the honest reading of a document whose
   * bases have not been read. Feature 2.7 took the same course for the explorer's own `▣`.
   */
  readonly template: boolean;
  /** The size the layout is given, in CSS pixels. */
  readonly width: number;
  readonly height: number;
}

/** One wire of the canvas. */
export interface CanvasWire {
  /** The wire's own identity: the pointer of the binding, or of the interface endpoint. */
  readonly id: string;
  readonly path: Path;
  /** The rule name — what labels the edge (`presentation.json`'s `label: "$key"`). */
  readonly label: string;
  /** The box the wire leaves, and the one it enters. */
  readonly from: string;
  readonly to: string;
  /** The handle each end names, for the hover and for the boundary label. */
  readonly fromPort: string;
  readonly toPort: string;
  /** The value's type in D2's convention, when a derivation is fresh and the view asks for it. */
  readonly type: string | null;
  /** A guard on the binding, printed. */
  readonly guard: string | null;
  /** Whether the wire is an interface's own rather than a value binding. */
  readonly interface: boolean;
}

/** One identity link: a dashed line between the chips of two members (§4.7). */
export interface CanvasLink {
  readonly identity: string;
  readonly from: { readonly box: string; readonly slot: string };
  readonly to: { readonly box: string; readonly slot: string };
  readonly state: boolean;
}

/** What the canvas draws. */
export interface CanvasModel {
  readonly boxes: readonly CanvasBox[];
  readonly wires: readonly CanvasWire[];
  readonly links: readonly CanvasLink[];
  /** S1's own note: `folded document · 3 instances · 1 composition · 2 interfaces`. */
  readonly note: string;
  /** Every box by its pointer. */
  readonly byPointer: ReadonlyMap<string, CanvasBox>;
}

/** Which of §4.7's View toggles are on. */
export interface CanvasView {
  readonly families: boolean;
  readonly derivedFigures: boolean;
  readonly edgeTypes: boolean;
  readonly identities: boolean;
}

/** The toggles as the editor starts: S1 draws Derived figures on and Identities off. */
export const DEFAULT_VIEW: CanvasView = {
  families: true,
  derivedFigures: true,
  edgeTypes: false,
  identities: false,
};

/** What {@link canvasModel} is given. */
export interface CanvasRequest {
  readonly folded: FoldedGraph;
  /** The core's facts for the folded sites, or `null` before the first answer. */
  readonly facts: Facts | null;
  /** The derived document, or `null` where none has been computed. */
  readonly derived: PyValue | null;
  /** Whether the derived document was computed for an older revision (§5.4). */
  readonly stale?: boolean;
  /** Every problem the panel holds for this document, in the order the stages produced them. */
  readonly problems: readonly Problem[];
  /** The compositions the reader has collapsed — the sidecar's own list (D6). */
  readonly collapsed?: ReadonlySet<string>;
  readonly view?: CanvasView;
  readonly registry: SchemaRegistry;
  readonly shapes: SchemaShapes;
  readonly bindings: Presentation;
  /** The role the document is read under; the model's unless a caller says otherwise. */
  readonly role?: string;
  /** The primitives that pin a template, by name — the core's `template_primitives` (§4.7). */
  readonly templates?: ReadonlySet<string>;
}

/** The model's own role, as the store reads a document under it. */
const MODEL = 'model';

// The geometry of a card, read off `plans/graph-editor-design/_ts.css` — `.node` is 214 px wide
// with 10 px of padding above and below and a 5 px gap, `.group` is 300 px collapsed. Nothing
// here is a fact about the language; it is what the layout is told a box measures before the
// browser has measured the rendered one, and the canvas lays out again when it differs.
const CARD = { width: 214, padding: 10, gap: 5 } as const;
const GROUP = { width: 300, padding: 10, gap: 6, side: 12 } as const;
const TERMINAL = { width: 190, height: 34 } as const;
const ROW = { head: 18, meta: 16, args: 16, ports: 16, slots: 16, derived: 17, held: 15 } as const;

/** How tall a stack of rows is, with the gaps between them. */
function stack(rows: readonly number[], padding: number, gap: number): number {
  if (rows.length === 0) return padding * 2;
  return padding * 2 + rows.reduce((total, row) => total + row, 0) + gap * (rows.length - 1);
}

/** The binding of a place, most specific first. */
function bindingAt(bindings: Presentation, shapes: SchemaShapes, path: Path, role: string): Binding | undefined {
  let shape = shapes.root(role);
  for (const step of path) shape = shapes.step(shape, step);
  return bindings.firstOf(shape.all.map((place) => place.anchor));
}

/** The shape of a place, for the printer. */
function shapeAt(shapes: SchemaShapes, path: Path, role: string): ReturnType<SchemaShapes['root']> {
  let shape = shapes.root(role);
  for (const step of path) shape = shapes.step(shape, step);
  return shape;
}

/** Whether a problem's pointer falls at or under a place. */
function under(pointer: string, place: string): boolean {
  return pointer === place || pointer.startsWith(`${place}/`);
}

/** The heaviest severity of a list, in §4.17's own order. */
function heaviest(problems: readonly Problem[]): 'error' | 'warning' | 'notice' | null {
  if (problems.some((one) => one.severity === 'error')) return 'error';
  if (problems.some((one) => one.severity === 'warning')) return 'warning';
  if (problems.some((one) => one.severity === 'notice')) return 'notice';
  return null;
}

/** The description of each folded box, by its pointer — `describe`'s answer, re-keyed. */
export function descriptionsByBox(facts: Facts | null): Map<string, SiteDescription> {
  const found = new Map<string, SiteDescription>();
  if (facts === null) return found;
  for (const site of facts.sites.values()) {
    const pointer = pointerOf(site.segments);
    if (!found.has(pointer)) found.set(pointer, site);
  }
  return found;
}

/** The shapes a slot chip is drawn in, from the description and the derived facts. */
function slotsOf(
  site: SiteDescription | undefined,
  pointer: string,
  derived: DerivedFacts,
): CanvasSlot[] {
  if (site === undefined) return [];
  const slots: CanvasSlot[] = [];
  const add = (name: string, kind: CanvasSlot['kind'], boundBy: string | null): void => {
    const membership = derived.memberships.get(slotKey(pointer, name));
    slots.push({
      name,
      kind,
      identity: membership?.identity ?? (boundBy === null ? null : boundBy),
      shared: membership?.shared ?? false,
      located: membership?.located ?? false,
    });
  };
  for (const slot of site.parameters) if (slot.present) add(slot.name, 'parameter', slot.boundBy);
  for (const slot of site.constants) if (slot.present) add(slot.name, 'constant', slot.boundBy);
  for (const state of site.states) if (state.present) add(state.name, 'state', state.boundBy);
  return slots;
}

/** The port handles of a card, with V7 and V13 projected onto them. */
function portsOf(site: SiteDescription | undefined, side: 'inputs' | 'outputs'): CanvasPort[] {
  if (site === undefined) return [];
  const ports = side === 'inputs' ? site.inputs : site.outputs;
  return ports
    .filter((port) => port.present)
    .map((port) => ({
      name: port.name,
      side,
      // The validator's own two conditions, over the maps it reads them from: a present input with
      // no producer is V7's row, a present output nothing consumes is V13's.
      state:
        side === 'inputs'
          ? port.fedBy === null
            ? ('unfed' as const)
            : ('bound' as const)
          : port.consumed
            ? ('bound' as const)
            : ('unused' as const),
      title: portTitle(port),
    }));
}

/** What hovering a handle shows: "kind, evaluated shape and role" (§4.7). */
function portTitle(port: SiteDescription['inputs'][number]): string {
  const parts: string[] = [pyStr(port.kind)];
  if (port.shape !== null) {
    parts.push(port.shape.map((axis) => `${pyStr(axis.axis)}=${pyStr(axis.extent)}`).join(', '));
  }
  parts.push(pyStr(port.role));
  return parts.filter((part) => part !== '').join(' · ');
}

/**
 * The structural arguments of a card, `name=value` — "the core says which" (inventory §3).
 *
 * `ArgumentFact.structural` is the core's answer, read inside `resolve_arguments`' own walk; the
 * value is the expression the *document* writes, printed in §4.13's text form, which is what S2
 * draws (`width=d`, `mask=causal`) rather than the 4096 it resolves to.
 */
function summaryOf(site: SiteDescription | undefined, print: (value: JsonValue) => string): string {
  if (site === undefined) return '';
  const parts: string[] = [];
  for (const fact of site.arguments.facts) {
    if (!fact.structural || fact.written === undefined) continue;
    parts.push(`${fact.path}=${print(toJsonValue(fact.written))}`);
  }
  return parts.join(' · ');
}

/** The model the canvas draws, from what the core answered about the document. */
export function canvasModel(request: CanvasRequest): CanvasModel {
  const { folded, facts, problems, registry, shapes, bindings } = request;
  const role = request.role ?? MODEL;
  const view = request.view ?? DEFAULT_VIEW;
  const collapsed = request.collapsed ?? new Set<string>();
  const context: PrintContext = { registry, shapes, bindings };
  const derived =
    request.derived === null ? noDerivedFacts() : derivedFacts(request.derived, folded);
  const described = descriptionsByBox(facts);

  const print = (path: Path, value: JsonValue): string =>
    printValue(context, shapeAt(shapes, path, role), value);

  const boxes: CanvasBox[] = [];
  const byPointer = new Map<string, CanvasBox>();

  const build = (node: FoldedNode, inside: boolean): void => {
    const binding = bindingAt(bindings, shapes, node.segments, role);
    const drawnRole = binding?.role ?? '';
    const shut = drawnRole === ROLE.group && collapsed.has(node.pointer);
    const site = described.get(node.pointer);
    const mine = problems.filter((one) => under(one.path, node.pointer));
    const figures = derived.figures.get(node.pointer);
    const summary = summaryOf(site, (value) =>
      print([...node.segments, 'arguments'] as Path, value),
    );
    const inputs = portsOf(site, 'inputs');
    const outputs = portsOf(site, 'outputs');
    const slots = slotsOf(site, node.pointer, derived);
    const handles = shut ? handlesOf(folded, node, print) : [];
    const range =
      node.ranges.length === 0
        ? null
        : node.ranges
            .map((one) => rangeText(one, (value) => print([...node.segments, 'indices'] as Path, value)))
            .join(' · ');
    const derivedLine = view.derivedFigures ? figureLine(figures) : null;
    const box: CanvasBox = {
      pointer: node.pointer,
      path: node.segments,
      name: node.name,
      role: drawnRole,
      side: binding?.side ?? null,
      parent: node.parent,
      primitive: node.primitive,
      version: node.version,
      families: view.families ? node.families : [],
      guard: node.guard === null ? null : print([...node.segments, 'when'] as Path, node.guard),
      summary,
      inputs,
      outputs,
      slots,
      range,
      count: node.ranges.length === 0 ? null : node.count === null ? '×?' : `×${node.count.toString()}`,
      held: node.held.length === 0 ? null : node.held.map((one) => `${one.name} ${String(one.count)}`).join(' · '),
      handles,
      badges: node.badges.map((badge) => (badge.value === null ? badge.name : print([...node.segments, badge.name] as Path, badge.value))),
      derived: derivedLine,
      stale: request.stale === true,
      problem: heaviest(mine),
      problems: mine.length,
      collapsed: shut,
      facts: site !== undefined,
      where: node.where,
      template: node.primitive !== null && (request.templates?.has(node.primitive) ?? false),
      width: 0,
      height: 0,
    };
    const sized = { ...box, ...sizeOf(box) };
    boxes.push(sized);
    byPointer.set(sized.pointer, sized);
    if (!inside && !shut) for (const child of node.children) build(child, true);
  };

  for (const node of folded.nodes) build(node, false);

  const wires: CanvasWire[] = [];
  for (const edge of folded.edges) {
    const from = edge.from === null ? null : boxFor(byPointer, edge.from.box, edge.from.site);
    const to = edge.to === null ? null : boxFor(byPointer, edge.to.box, edge.to.site);
    if (from === null || to === null || from === to) continue;
    const type =
      view.edgeTypes && edge.from?.value != null ? (derived.types.get(edge.from.value) ?? null) : null;
    wires.push({
      id: edge.pointer,
      path: edge.segments,
      label: edge.rule,
      from,
      to,
      fromPort: edge.from?.port ?? '',
      toPort: edge.to?.port ?? '',
      type,
      guard: edge.guard === null ? null : print([...edge.segments, 'when'] as Path, edge.guard),
      interface: edge.interface,
    });
  }

  const links: CanvasLink[] = [];
  if (view.identities) {
    for (const link of derived.links) {
      const places = link.places.filter((place) => byPointer.has(place.box));
      for (let at = 1; at < places.length; at += 1) {
        const before = places[at - 1];
        const here = places[at];
        if (before === undefined || here === undefined) continue;
        links.push({
          identity: link.identity,
          from: { box: before.box, slot: before.slot },
          to: { box: here.box, slot: here.slot },
          state: link.state,
        });
      }
    }
  }

  return { boxes, wires, links, note: noteOf(folded), byPointer };
}

/** The box a wire's end is drawn on: the site's own card when it is on the canvas, else its box. */
function boxFor(byPointer: ReadonlyMap<string, CanvasBox>, box: string, site: string): string | null {
  if (byPointer.has(site)) return site;
  return byPointer.has(box) ? box : null;
}

/** S1's own note about what is drawn. */
function noteOf(folded: FoldedGraph): string {
  const counts = new Map<string, number>();
  for (const node of folded.nodes) counts.set(node.kind, (counts.get(node.kind) ?? 0) + 1);
  const parts: string[] = [];
  for (const [kind, count] of counts) parts.push(`${String(count)} ${kind}`);
  return parts.join(' · ');
}

/** A composition's index range, as §4.7 and S3 print it: `layer ∈ [0, 32)`. */
function rangeText(range: FoldedNode['ranges'][number], print: (value: JsonValue) => string): string {
  const step = range.step === null ? '' : ` by ${print(range.step)}`;
  return `${range.name} ∈ [${print(range.start)}, ${print(range.stop)})${step}`;
}

/** The boundary handles a collapsed composition shows (§4.7, D8). */
function handlesOf(
  folded: FoldedGraph,
  node: FoldedNode,
  print: (path: Path, value: JsonValue) => string,
): CanvasHandle[] {
  const handles: CanvasHandle[] = [];
  const seen = new Set<string>();
  for (const edge of folded.edges) {
    for (const [end, side] of [
      [edge.to, SIDE.left],
      [edge.from, SIDE.right],
    ] as const) {
      if (end === null || !end.boundary || end.box !== node.pointer) continue;
      const indices = end.indices
        .map((one) => `${one.name}=${one.value === null ? print(edge.segments, one.written) : pyStr(one.value)}`)
        .join(', ');
      const label = `${end.name}[${indices}].${end.port}`;
      if (seen.has(label)) continue;
      seen.add(label);
      handles.push({ label, side, site: end.site, port: end.port });
    }
  }
  return handles;
}

/** The derived line of a card, from the figures the core attributed to it. */
function figureLine(figures: SiteFigures | undefined): string | null {
  if (figures === undefined) return null;
  const parts: string[] = [];
  if (figures.bytes !== null) parts.push(`${bytes(figures.bytes)} params`);
  if (figures.bytesPerCachedPosition !== null) {
    parts.push(`${bytes(figures.bytesPerCachedPosition)} / cached position`);
  }
  return parts.length === 0 ? null : parts.join(' · ');
}

/**
 * A byte count as the editor renders one — `sizeText`, which is `view.py`'s own `fmt_bytes`.
 *
 * It is feature 2.6's, imported rather than written again: "no component adds, converts or rounds
 * a byte count" (inventory §7), and a second rendering of the same figure is exactly the drift
 * that rule exists to stop. The artboards round it further in three places — S1 writes `1002 MiB`
 * and `8 KiB` where the convention writes `1002.0 MiB` and `8.0 KiB` — and the convention is what
 * ships, because it is what `--view` prints today and what finding F6 moves here.
 */
function bytes(value: bigint): string {
  return sizeText(Number(value));
}

/** How large the layout is told a box is, before the browser has measured the rendered one. */
export function sizeOf(box: CanvasBox): { width: number; height: number } {
  if (box.role === ROLE.terminal) {
    return {
      width: Math.max(TERMINAL.width, 60 + box.name.length * 8 + box.badges.join(' ').length * 6),
      height: TERMINAL.height,
    };
  }
  if (box.role === ROLE.group) {
    const rows: number[] = [ROW.head];
    if (box.families.length > 0) rows.push(ROW.meta);
    if (box.collapsed) {
      const perSide = Math.max(
        box.handles.filter((one) => one.side === SIDE.left).length,
        box.handles.filter((one) => one.side === SIDE.right).length,
      );
      rows.push(...new Array<number>(Math.max(perSide, 1)).fill(ROW.ports));
    }
    if (box.held !== null) rows.push(ROW.held);
    if (box.derived !== null) rows.push(ROW.derived);
    const width = Math.max(
      GROUP.width,
      ...box.handles.map((one) => one.label.length * 6 + 40),
      (box.range ?? '').length * 6.5 + 80,
    );
    return { width: Math.round(width), height: stack(rows, GROUP.padding, GROUP.gap) };
  }
  const rows: number[] = [ROW.head];
  if (box.families.length > 0 || box.guard !== null) rows.push(ROW.meta);
  if (box.summary !== '') {
    rows.push(ROW.args * Math.max(1, Math.ceil((box.summary.length * 6.3) / (CARD.width - 24))));
  }
  const handles = Math.max(box.inputs.length, box.outputs.length);
  if (handles > 0) rows.push(ROW.ports * handles);
  if (box.slots.length > 0) {
    const perRow = Math.max(1, Math.floor((CARD.width - 24) / 64));
    rows.push(ROW.slots * Math.ceil(box.slots.length / perRow));
  }
  if (box.derived !== null) rows.push(ROW.derived);
  return { width: CARD.width, height: stack(rows, CARD.padding, CARD.gap) };
}
