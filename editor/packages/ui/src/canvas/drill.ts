/**
 * What the drill-in draws — plan §4.8, artboards S4 and S5.
 *
 * > a tab `llama3-8b › decoder`: the index strip at the top … the sites as cards … the scoped
 * > edges … a ghost column on the left … boundary edges as a pinned terminal … the alternation
 * > strip below the canvas.
 *
 * The same division as the folded canvas (`model.ts`): **one function turns answers into a
 * drawing and computes none of them.** The core's {@link drillGraph} says what the composition
 * holds, {@link drillPresence} says what exists at each index, `describe` says what a card shows
 * and the derived document says what it costs — and the only arithmetic here is geometry.
 *
 * **The cards are the folded canvas's own.** A site of the drill-in is drawn exactly as §4.7 draws
 * a card ("Sites as cards (§4.7)"), so the boxes come from {@link canvasModel} over the same
 * document with this composition open, and are re-parented to the top of the drawing: one
 * implementation of a card, two places it is drawn.
 *
 * **The ghosts are not in the layout.** §4.8 calls them a *column*: "a translucent copy of the
 * source site labelled `ffn_r [layer−1]`", generated from the rules and "not editable except by
 * editing the rule". They are copies of boxes that are already drawn, so putting them into ELK
 * would make the engine route the graph around a node that stands for nothing; they are placed
 * beside the drawing instead, on the side the override is on, and the wires run from them.
 */
import {
  alternationCount,
  drillGraph,
  drillPresence,
  languageAnchors,
  NO_PRESENCE,
  type DrillEdge,
  type DrillEnd,
  type DrillGraph,
  type DrillPoint,
  type DrillPresence,
  type FoldedGraph,
  type JsonValue,
  type PyValue,
  type SchemaRegistry,
} from '@tensorspine/lang';
import type { Facts, Problem } from '@tensorspine/lang/api';
import { pathOfPointer, type Path, type SchemaShapes } from '@tensorspine/store';

import { printAt, printValue, type PrintContext } from '../expressions/index.js';
import type { Presentation } from '../presentation/index.js';

import {
  canvasModel,
  ROLE,
  SIDE,
  sizeOf,
  type CanvasBox,
  type CanvasModel,
  type CanvasView,
  type CanvasWire,
} from './model.js';

/** One index of the strip: its bounds as the document writes them, and where each is edited. */
export interface DrillStripRange {
  readonly name: string;
  /** The place of the index's own declaration, so a click selects it (§4.11's composition sheet). */
  readonly at: Path;
  /** The three bounds, printed in §4.13's text form, with the place each is written at. */
  readonly bounds: readonly { readonly name: string; readonly text: string; readonly at: Path }[];
  /** The values the index runs over; empty where a bound does not resolve. */
  readonly values: readonly bigint[];
}

/** A ghost column, ready to be drawn beside the layout (inventory §3 "Ghost column"). */
export interface DrillGhostBox {
  readonly id: string;
  readonly name: string;
  /** `[layer−1]`, printed — what §4.8 labels the copy with. */
  readonly label: string;
  /** The primitive of the site it copies, which is what S4 writes under the name. */
  readonly primitive: string | null;
  readonly side: 'left' | 'right';
  /** The pointer of the site it copies, so a click selects the real card. */
  readonly site: string;
  /** The boxes its wires reach, so the column can be put beside them. */
  readonly targets: readonly string[];
  readonly width: number;
  readonly height: number;
}

/** A pinned terminal, drawn as S4's `.bterm` (inventory §3 "Boundary terminal"). */
export interface DrillTerminalBox {
  readonly id: string;
  readonly side: 'left' | 'right';
  /** `◁ embed.output at layer = 0` — what the terminal says it is. */
  readonly label: string;
  /** `decoder.entry → attn_n.input · decoder.entry.a → attn_r.a` — the rules pinned on it. */
  readonly links: readonly { readonly rule: string; readonly text: string; readonly at: Path }[];
}

/** One row of the alternation strip: a guarded site, its guard, and where it exists (S5). */
export interface AlternationRow {
  readonly name: string;
  /** The pointer of the site, so a click selects it. */
  readonly pointer: string;
  /** The guard, printed in §4.13's text form. */
  readonly guard: string;
  /** One cell per iteration, in the grid's order: whether D1 emitted the node there. */
  readonly cells: readonly boolean[];
  /** `24 of 30`. */
  readonly at: number;
  readonly of: number;
}

/** What the drill-in of one composition draws. */
export interface DrillModel {
  readonly pointer: string;
  readonly name: string;
  readonly families: readonly string[];
  readonly ranges: readonly DrillStripRange[];
  /** The cards and the pinned terminals, for the layout. */
  readonly canvas: CanvasModel;
  readonly ghosts: readonly DrillGhostBox[];
  readonly terminals: ReadonlyMap<string, DrillTerminalBox>;
  /** The grid's points, in order — the scrubber's own track. */
  readonly points: readonly DrillPoint[];
  /** The alternation strip's rows: one per guarded site (S5). */
  readonly rows: readonly AlternationRow[];
  /** How many sites carry no guard, which the strip does not list but says (S5's note). */
  readonly unguarded: number;
  /** What D1 says exists where; empty before a derivation has answered (§5.4's freshness). */
  readonly presence: DrillPresence;
  /** Whether anything was read for the presence at all. */
  readonly expanded: boolean;
  /** The guard of each edge, printed; keyed by the place the rule is written at. */
  readonly guards: ReadonlyMap<string, string>;
  /** S4's own note under the canvas. */
  readonly note: string;
}

/** What {@link drillModel} is given. */
export interface DrillRequest {
  readonly tree: JsonValue;
  readonly composition: string;
  readonly folded: FoldedGraph;
  readonly facts: Facts | null;
  readonly derived: PyValue | null;
  readonly stale?: boolean;
  readonly problems: readonly Problem[];
  readonly view?: CanvasView;
  readonly registry: SchemaRegistry;
  readonly shapes: SchemaShapes;
  readonly bindings: Presentation;
  readonly role?: string;
  readonly templates?: ReadonlySet<string>;
}

/** The model's own role, as the store reads a document under it. */
const MODEL = 'model';

/** What a ghost column measures, in CSS pixels — `_ts.css`'s `.ghostcol` around a `.node.tiny`. */
const GHOST = { width: 180, height: 62 } as const;

/** The drill-in of one composition, or `null` where the document declares no such composition. */
export function drillModel(request: DrillRequest): DrillModel | null {
  const { tree, folded, registry, shapes, bindings } = request;
  const role = request.role ?? MODEL;
  const drill = drillGraph(tree, { composition: request.composition, folded });
  if (drill === null) return null;
  const context: PrintContext = { registry, shapes, bindings };
  const expressionAnchor = languageAnchors(registry).expression;
  const print = (value: JsonValue): string => printAt(context, expressionAnchor, value);
  const printAtPlace = (path: Path, value: JsonValue): string =>
    printValue(context, shapeAt(shapes, path, role), value);

  // The cards are §4.7's own, over the same document with this composition open.
  const shut = new Set<string>();
  for (const node of folded.nodes) {
    if (node.children.length > 0 && node.pointer !== drill.pointer) shut.add(node.pointer);
  }
  const whole = canvasModel({
    folded,
    facts: request.facts,
    derived: request.derived,
    ...(request.stale === undefined ? {} : { stale: request.stale }),
    problems: request.problems,
    collapsed: shut,
    ...(request.view === undefined ? {} : { view: request.view }),
    registry,
    shapes,
    bindings,
    role,
    ...(request.templates === undefined ? {} : { templates: request.templates }),
  });
  const cards: CanvasBox[] = whole.boxes
    .filter((box) => box.parent === drill.pointer)
    .map((box) => ({ ...box, parent: null }));

  const terminals = new Map<string, DrillTerminalBox>();
  const terminalBoxes: CanvasBox[] = [];
  for (const terminal of drill.terminals) {
    const at = terminal.indices
      .map((one) => `${one.name} = ${print(one.written)}`)
      .join(', ');
    const mark = terminal.side === SIDE.left ? '◁' : '▷';
    const label = at === '' ? `${mark} ${terminal.outside}` : `${mark} ${terminal.outside} at ${at}`;
    terminals.set(terminal.id, {
      id: terminal.id,
      side: terminal.side,
      label,
      links: terminal.links.map((link) => ({
        rule: link.rule,
        text:
          terminal.side === SIDE.left
            ? `${link.rule} → ${link.name}.${link.port}`
            : `${link.rule} ← ${link.name}.${link.port}`,
        at: pathOfPointer(link.pointer),
      })),
    });
    const box: CanvasBox = {
      ...blankBox(terminal.id, pathOfPointer(terminal.links[0]?.pointer ?? terminal.id)),
      name: label,
      role: ROLE.terminal,
      side: terminal.side,
      badges: terminal.links.map((link) => link.rule),
    };
    terminalBoxes.push({ ...box, ...sizeOf(box) });
  }

  const ghosts: DrillGhostBox[] = [];
  const ghostOf = new Map<string, string>();
  for (const ghost of drill.ghosts) {
    const site = folded.byPointer.get(ghost.site ?? '');
    const label = `[${ghost.indices.map((one) => print(one.written)).join(', ')}]`;
    for (const edge of ghost.edges) ghostOf.set(`${edge}:${ghost.side}`, ghost.id);
    ghosts.push({
      id: ghost.id,
      name: ghost.name,
      label,
      primitive: site?.primitive ?? null,
      side: ghost.side,
      site: ghost.site ?? '',
      targets: [],
      width: GHOST.width,
      height: GHOST.height,
    });
  }

  const byPointer = new Map<string, CanvasBox>();
  for (const box of [...cards, ...terminalBoxes]) byPointer.set(box.pointer, box);

  const wires: CanvasWire[] = [];
  const guards = new Map<string, string>();
  const targets = new Map<string, Set<string>>();
  for (const edge of drill.edges) {
    if (edge.guard !== null) {
      guards.set(edge.pointer, printAtPlace([...edge.segments, 'when'] as Path, edge.guard));
    }
    const from = endOf(edge, edge.from, SIDE.left, drill, ghostOf, terminals, byPointer);
    const to = endOf(edge, edge.to, SIDE.right, drill, ghostOf, terminals, byPointer);
    if (from === null || to === null || from === to) continue;
    wires.push({
      id: edge.pointer,
      path: edge.segments,
      label: edge.rule,
      from,
      to,
      fromPort: edge.from?.port ?? '',
      toPort: edge.to?.port ?? '',
      type: null,
      guard: guards.get(edge.pointer) ?? null,
      interface: edge.interface,
    });
    for (const [end, id] of [
      [edge.from, from],
      [edge.to, to],
    ] as const) {
      if (end === null) continue;
      const ghost = ghostOf.get(`${edge.pointer}:${end === edge.from ? SIDE.left : SIDE.right}`);
      if (ghost === undefined) continue;
      const other = end === edge.from ? to : from;
      void id;
      targets.set(ghost, (targets.get(ghost) ?? new Set()).add(other));
    }
  }

  const presence = request.derived === null ? NO_PRESENCE : drillPresence(drill, request.derived);
  const rows: AlternationRow[] = [];
  let unguarded = 0;
  for (const site of drill.sites) {
    if (site.guard === null) {
      unguarded += 1;
      continue;
    }
    const present = presence.sites.get(site.name) ?? new Set<string>();
    rows.push({
      name: site.name,
      pointer: site.pointer,
      guard: printValue(context, shapeAt(shapes, [...site.segments, 'when'] as Path, role), site.guard),
      cells: presence.points.map((point) => present.has(point.label)),
      ...alternationCount(presence, site.name),
    });
  }

  return {
    pointer: drill.pointer,
    name: drill.name,
    families: drill.families,
    ranges: drill.ranges.map((range) => ({
      name: range.name,
      at: [...drill.segments, 'indices', range.name] as Path,
      bounds: (['start', 'stop', 'step'] as const)
        .filter((bound) => range[bound] !== null)
        .map((bound) => ({
          name: bound,
          text: print(range[bound]),
          at: [...drill.segments, 'indices', range.name, bound] as Path,
        })),
      values: range.values ?? [],
    })),
    canvas: {
      boxes: [...cards, ...terminalBoxes],
      wires,
      links: [],
      note: noteOf(drill),
      byPointer,
    },
    ghosts: ghosts.map((ghost) => ({ ...ghost, targets: [...(targets.get(ghost.id) ?? [])] })),
    terminals,
    points: drill.points,
    rows,
    unguarded,
    presence,
    expanded: request.derived !== null && presence.points.length > 0,
    guards,
    note: noteOf(drill),
  };
}

/** Where one end of an edge hangs on the drawing: a card, a ghost column or a pinned terminal. */
function endOf(
  edge: DrillEdge,
  end: DrillEnd | null,
  side: 'left' | 'right',
  drill: DrillGraph,
  ghostOf: ReadonlyMap<string, string>,
  terminals: ReadonlyMap<string, DrillTerminalBox>,
  boxes: ReadonlyMap<string, CanvasBox>,
): string | null {
  if (end === null) return null;
  const ghost = ghostOf.get(`${edge.pointer}:${side}`);
  if (ghost !== undefined) return ghost;
  if (end.site !== null && boxes.has(end.site)) return end.site;
  for (const terminal of drill.terminals) {
    if (!terminals.has(terminal.id)) continue;
    if (terminal.links.some((link) => link.pointer === edge.pointer)) return terminal.id;
  }
  return null;
}

/** S4's own note: what the drawing stands for. */
function noteOf(drill: DrillGraph): string {
  const scoped = drill.edges.filter((edge) => edge.scoped).length;
  return `${String(drill.sites.length)} sites · ${String(scoped)} scoped edges`;
}

/** The shape of a place of the document, for the printer. */
function shapeAt(shapes: SchemaShapes, path: Path, role: string): ReturnType<SchemaShapes['root']> {
  let shape = shapes.root(role);
  for (const step of path) shape = shapes.step(shape, step);
  return shape;
}

/** A box with nothing on it: what a synthetic terminal starts from. */
function blankBox(pointer: string, path: Path): CanvasBox {
  return {
    pointer,
    path,
    name: pointer,
    role: ROLE.terminal,
    side: null,
    parent: null,
    primitive: null,
    version: null,
    families: [],
    guard: null,
    summary: '',
    inputs: [],
    outputs: [],
    slots: [],
    range: null,
    count: null,
    held: null,
    handles: [],
    badges: [],
    derived: null,
    stale: false,
    problem: null,
    problems: 0,
    collapsed: false,
    facts: false,
    where: null,
    template: false,
    width: 0,
    height: 0,
  };
}
