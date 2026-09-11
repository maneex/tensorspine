/**
 * Placing the folded canvas — plan §4.7's "ELK layered layout, **top-to-bottom**, compositions as
 * compound nodes; overrides kept unless Reset Layout", over feature 0.4's measured module.
 *
 * Three decisions, each with its reason:
 *
 * **ELK places the boxes; the wires are drawn from the boxes.** The layout answers routes too, and
 * they are not read: a box the author *moved* (D6's override) is no longer where ELK put it, and a
 * route computed for the old place would leave the wire hanging in the air. So a wire is drawn
 * from the rectangle of the box it leaves to the rectangle of the box it enters, which is what S1
 * draws — a card's bottom to the next card's top — and what stays true whatever the author drags.
 *
 * **ELK is loaded when a canvas first needs it, and never before.** Feature 0.4 measured the cost
 * of letting it into the first bundle — "re-exporting the layout from `packages/ui/src/index.ts`
 * put 1.6 MB of compiled ELK into the static application's chunk (881 B → 1 432 kB) though
 * nothing imported it" — so the import here is dynamic and the bundler gives it a chunk of its
 * own.
 *
 * **A layout is never on an interaction path.** 0.4's own reservation: the largest expanded graph
 * sits *at* the two-second budget. The folded canvas is 8 to 42 boxes and lays out in 8–35 ms, but
 * it is still asked for only when the *shape* of the drawing changes — a box added, removed,
 * renamed, resized, collapsed or expanded — and never while a pointer is down.
 */
import type { Layout as LayoutSidecar, Position } from '@tensorspine/store';

import type { FoldedGraph } from '@tensorspine/lang';
import type { GraphNode } from '../layout/elk.js';

import type { CanvasBox, CanvasModel } from './model.js';

/** Where one box ended up, in the drawing's own coordinates. */
export interface PlacedBox {
  readonly pointer: string;
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
  /** Whether the place is the sidecar's rather than the layout's (D6's override). */
  readonly manual: boolean;
}

/** The drawing: every box placed, and how large the whole is. */
export interface Placement {
  readonly boxes: ReadonlyMap<string, PlacedBox>;
  readonly width: number;
  readonly height: number;
  /** How long the engine took, for §5.6's budget and for the log. */
  readonly milliseconds: number;
}

/** A placement of nothing at all. */
export const NO_PLACEMENT: Placement = {
  boxes: new Map(),
  width: 0,
  height: 0,
  milliseconds: 0,
};

/** What a placement needs beside the model. */
export interface PlaceRequest {
  readonly model: CanvasModel;
  /** The sidecar's manual moves, by the document path of the box each is about (D6). */
  readonly positions?: Readonly<Record<string, Position>>;
  /** The sizes the browser measured, where it has; the model's estimates are used otherwise. */
  readonly measured?: ReadonlyMap<string, { readonly width: number; readonly height: number }>;
}

/** The sidecar's key for a box: the document path, as `keyOf` writes one (D6). */
export function layoutKey(pointer: string): string {
  return pointer.startsWith('/') ? pointer.slice(1) : pointer;
}

/** The size the layout is given for a box: what the browser measured, else what the model said. */
function sizeOf(box: CanvasBox, measured: PlaceRequest['measured']): { width: number; height: number } {
  const found = measured?.get(box.pointer);
  // jsdom measures every element as zero, and a suite that ran on those numbers would be laying
  // out a drawing no browser will ever show: a measurement of nothing is no measurement.
  if (found !== undefined && found.width > 0 && found.height > 0) return found;
  return { width: box.width, height: box.height };
}

/**
 * Place the model, top to bottom.
 *
 * The engine is `elkjs`, loaded on the first call. A box the sidecar carries a position for is put
 * there afterwards, which is exactly what D6 calls an override: "the default is automatic layout;
 * a manual move is an override in the sidecar".
 */
export async function place(request: PlaceRequest): Promise<Placement> {
  const { model } = request;
  const { layout } = await import('../layout/elk.js');
  const children = new Map<string, CanvasBox[]>();
  const roots: CanvasBox[] = [];
  for (const box of model.boxes) {
    if (box.parent === null || !model.byPointer.has(box.parent)) roots.push(box);
    else children.set(box.parent, [...(children.get(box.parent) ?? []), box]);
  }

  const toNode = (box: CanvasBox): GraphNode => {
    const size = sizeOf(box, request.measured);
    const inside = children.get(box.pointer);
    if (inside === undefined || inside.length === 0) return { id: box.pointer, ...size };
    return {
      id: box.pointer,
      ...size,
      children: inside.map(toNode),
      // A composition's own chrome is the room its sites may not use: its header, its families
      // and its summary line. A compound node cannot be given a minimum size (feature 0.4), so
      // the padding is what keeps the header from being drawn over a card.
      padding: { top: GROUP_CHROME, right: 12, bottom: GROUP_FOOT, left: 12 },
    };
  };

  const nodes = roots.map(toNode);
  const edges = model.wires
    .filter((wire) => model.byPointer.has(wire.from) && model.byPointer.has(wire.to) && wire.from !== wire.to)
    .map((wire) => ({ id: wire.id, source: wire.from, target: wire.to }));

  const laid = await layout({ nodes, edges });
  const boxes = new Map<string, PlacedBox>();
  for (const node of laid.nodes) {
    boxes.set(node.id, {
      pointer: node.id,
      x: node.x,
      y: node.y,
      width: node.width,
      height: node.height,
      manual: false,
    });
  }

  const positions = request.positions ?? {};
  for (const box of model.boxes) {
    const override = positions[layoutKey(box.pointer)];
    const placed = boxes.get(box.pointer);
    if (override === undefined || placed === undefined) continue;
    boxes.set(box.pointer, { ...placed, x: override.x, y: override.y, manual: true });
  }

  let width = 0;
  let height = 0;
  for (const box of boxes.values()) {
    width = Math.max(width, box.x + box.width);
    height = Math.max(height, box.y + box.height);
  }
  return { boxes, width, height, milliseconds: laid.milliseconds };
}

/** The room a composition's header, families and boundary handles need above its sites. */
const GROUP_CHROME = 74;

/** The room its summary and derived lines need below them. */
const GROUP_FOOT = 40;

/** Where a wire runs: from the bottom of the box it leaves to the top of the one it enters. */
export interface WireRoute {
  readonly id: string;
  readonly from: { readonly x: number; readonly y: number };
  readonly to: { readonly x: number; readonly y: number };
  /** Where the label sits — §4.7 puts the rule name beside the middle of the wire. */
  readonly label: { readonly x: number; readonly y: number };
  /** The path a bezier takes between the two, for an `<svg>`'s `d`. */
  readonly d: string;
}

/**
 * The wires of a placement, as S1 draws them: a bezier from a card's bottom to the next's top.
 *
 * Several wires leaving one box are fanned across its width rather than stacked on one point,
 * which is what S1's own drawing does with `decoder.entry` and `decoder.entry.a`.
 */
export function routes(model: CanvasModel, placement: Placement): WireRoute[] {
  const leaving = new Map<string, number>();
  const entering = new Map<string, number>();
  const outs = new Map<string, number>();
  const ins = new Map<string, number>();
  for (const wire of model.wires) {
    outs.set(wire.from, (outs.get(wire.from) ?? 0) + 1);
    ins.set(wire.to, (ins.get(wire.to) ?? 0) + 1);
  }
  const found: WireRoute[] = [];
  for (const wire of model.wires) {
    const from = placement.boxes.get(wire.from);
    const to = placement.boxes.get(wire.to);
    if (from === undefined || to === undefined) continue;
    const out = leaving.get(wire.from) ?? 0;
    leaving.set(wire.from, out + 1);
    const into = entering.get(wire.to) ?? 0;
    entering.set(wire.to, into + 1);
    const start = {
      x: from.x + spread(from.width, out, outs.get(wire.from) ?? 1),
      y: from.y + from.height,
    };
    const end = { x: to.x + spread(to.width, into, ins.get(wire.to) ?? 1), y: to.y };
    const bend = Math.max(12, Math.min(48, (end.y - start.y) / 2));
    found.push({
      id: wire.id,
      from: start,
      to: end,
      label: { x: (start.x + end.x) / 2 + 10, y: (start.y + end.y) / 2 },
      d: `M${round(start.x)} ${round(start.y)} C${round(start.x)} ${round(start.y + bend)} ${round(end.x)} ${round(end.y - bend)} ${round(end.x)} ${round(end.y)}`,
    });
  }
  return found;
}

/** Where the nth of `count` wires attaches across a box's width. */
function spread(width: number, at: number, count: number): number {
  if (count <= 1) return width / 2;
  const margin = Math.min(width / 4, 40);
  return margin + ((width - 2 * margin) * at) / (count - 1);
}

function round(value: number): number {
  return Math.round(value * 10) / 10;
}

/** The zoom that fits a drawing into a viewport, and where the drawing then sits. */
export interface Fit {
  readonly zoom: number;
  readonly x: number;
  readonly y: number;
}

/** `View ▸ Zoom to Fit` (§4.4): the whole drawing in the viewport, centred, never magnified. */
export function fit(
  placement: Placement,
  viewport: { readonly width: number; readonly height: number },
  padding = 24,
): Fit {
  if (placement.width <= 0 || placement.height <= 0 || viewport.width <= 0 || viewport.height <= 0) {
    return { zoom: 1, x: 0, y: 0 };
  }
  const zoom = Math.min(
    1,
    (viewport.width - padding * 2) / placement.width,
    (viewport.height - padding * 2) / placement.height,
  );
  return {
    zoom,
    x: (viewport.width - placement.width * zoom) / 2,
    y: (viewport.height - placement.height * zoom) / 2,
  };
}

/** Whether a sidecar carries a manual move at all — what "Reset Layout" has to undo (D6). */
export function hasManualMoves(layout: LayoutSidecar): boolean {
  return Object.keys(layout.positions).length > 0;
}

/**
 * Which compositions are drawn collapsed — §4.7's "collapsed by default at the top level".
 *
 * The default is the section's, and the sidecar's `collapsed` list is the reader's deviation from
 * it (D6). The two are read together the way feature 2.7 reads the explorer's `toggled`: a
 * composition is open exactly where the reader has opened it, which is why the set handed in is
 * the *expanded* one and not the collapsed one — a document with no sidecar draws every box shut
 * without the sidecar having to list them, and so writes nothing.
 */
export function collapsedGroups(
  graph: FoldedGraph,
  expanded: ReadonlySet<string>,
): Set<string> {
  const shut = new Set<string>();
  for (const node of graph.nodes) {
    if (node.kind === 'composition' && !expanded.has(node.pointer)) shut.add(node.pointer);
  }
  return shut;
}

/** What a sidecar's `collapsed` list says is open, on a document the reader has arranged before. */
export function expandedFromSidecar(graph: FoldedGraph, layout: LayoutSidecar): Set<string> {
  const open = new Set<string>();
  if (layout.collapsed.length === 0) return open;
  const shut = new Set(layout.collapsed.map((key) => `/${key}`));
  for (const node of graph.nodes) {
    if (node.kind === 'composition' && !shut.has(node.pointer)) open.add(node.pointer);
  }
  return open;
}
