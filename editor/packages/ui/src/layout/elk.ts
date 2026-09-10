import ElkConstructor from 'elkjs/lib/elk.bundled.js';
import type { ElkExtendedEdge, ElkNode, LayoutOptions } from 'elkjs/lib/elk-api.js';

/**
 * Top-to-bottom layered layout for the editor's canvases.
 *
 * The plan's §4.7 gives the folded canvas one automatic layout — "ELK layered layout,
 * **top-to-bottom** (the reading `--view` gives; corrected by the design pass), compositions as
 * compound nodes" — and §4.9 lays the expanded graph over D1 out the same way, in a worker,
 * virtualised. This module is that layout and nothing else.
 *
 * It knows nothing about the language. It takes boxes, an optional containment tree and edges
 * between boxes, and answers where every box goes and how every edge runs, in one coordinate
 * system. What a node stands for, what a group stands for and how large a card is are the
 * caller's business, so no vocabulary of the schemas can reach here — the governing rule of §1:
 * an item of information that can be inferred from a schema is never hard-coded in the interface.
 *
 * The engine is `elkjs`, and it is a parameter. The default is the bundled build, which runs
 * wherever this module runs (the test process, a browser tab); the static application hands
 * {@link LayoutSettings.engine} an instance backed by a Web Worker (`elkjs/lib/elk-api.js` over
 * `elk-worker.min.js`) so that laying out several hundred nodes never blocks the interface
 * (§5.6). Both speak the same interface, {@link LayoutEngine}, and this module speaks only that.
 *
 * Coordinates. ELK answers in a nested coordinate system: a child's position is relative to its
 * parent's corner, and an edge's route is relative to the corner of the node the edge was
 * assigned to. Everything this module returns is absolute — measured from the corner of the
 * whole drawing — because a canvas, an overlap test and an SVG all want one frame of reference.
 */

/** A point of the drawing, in absolute coordinates. */
export interface Point {
  readonly x: number;
  readonly y: number;
}

/** The space a compound node keeps between its own border and its children. */
export interface Insets {
  readonly top: number;
  readonly right: number;
  readonly bottom: number;
  readonly left: number;
}

/**
 * A box to place. A node with `children` is a compound node: the layout sizes it around what it
 * holds, and `padding` is the room its own chrome needs — a composition box's header, in §4.7's
 * canvas. A node without children keeps the `width` and `height` it was given.
 */
export interface GraphNode {
  readonly id: string;
  readonly width: number;
  readonly height: number;
  readonly children?: readonly GraphNode[];
  readonly padding?: Insets;
}

/** A label the layout reserves room for beside an edge — a value binding's rule name (§4.7). */
export interface EdgeLabel {
  readonly text: string;
  readonly width: number;
  readonly height: number;
}

/** An edge between two boxes, named anywhere in the containment tree. */
export interface GraphEdge {
  readonly id: string;
  readonly source: string;
  readonly target: string;
  readonly label?: EdgeLabel;
}

/** What is laid out: boxes, their containment, and the edges between them. */
export interface Graph {
  readonly nodes: readonly GraphNode[];
  readonly edges: readonly GraphEdge[];
}

/** A box, placed. `x` and `y` are absolute; `parent` and `depth` describe the containment. */
export interface PlacedNode {
  readonly id: string;
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
  readonly parent: string | null;
  readonly depth: number;
}

/** An edge, routed. `points` runs from the source's border to the target's, in absolute coordinates. */
export interface RoutedEdge {
  readonly id: string;
  readonly source: string;
  readonly target: string;
  readonly points: readonly Point[];
  readonly label?: EdgeLabel & Point;
}

/** The drawing: its extent, every box placed, every edge routed. */
export interface Layout {
  readonly width: number;
  readonly height: number;
  readonly nodes: readonly PlacedNode[];
  readonly edges: readonly RoutedEdge[];
  readonly byId: ReadonlyMap<string, PlacedNode>;
  /** How long the engine took, wall clock, for the budgets of §5.6. */
  readonly milliseconds: number;
}

/**
 * What this module needs of a layout engine: elkjs's `ELK`, whether it runs here or in a worker.
 * Declaring it structurally is what lets the application choose without this module changing.
 */
export interface LayoutEngine {
  layout(graph: ElkNode, args?: { layoutOptions?: LayoutOptions }): Promise<ElkNode>;
}

/** How the drawing is arranged. Every field has the value §4.7 asks for as its default. */
export interface LayoutSettings {
  /** `DOWN` is the reading of §4.7: a value flows from the top of the canvas to the bottom. */
  readonly direction?: 'DOWN' | 'UP' | 'RIGHT' | 'LEFT';
  /** Between two boxes of one layer. */
  readonly nodeSpacing?: number;
  /** Between two layers — the vertical gap a wire crosses. */
  readonly layerSpacing?: number;
  /** How wires are drawn between the boxes. */
  readonly edgeRouting?: 'ORTHOGONAL' | 'POLYLINE' | 'SPLINES';
  /**
   * How hard the engine works at crossing minimisation, 1 to 7 (elkjs's own default is 7, kept
   * here). Lowering it buys less than it looks: on the corpus's most expensive expanded graph
   * it saves under a tenth of the time — `editor/spikes/layout/NOTE.md` measures where the time
   * actually goes.
   */
  readonly thoroughness?: number;
  /**
   * Whether the order the caller wrote the nodes and edges in breaks ties. D1 records a
   * topological order and §4.9 shows the expanded graph "in D1's order", which is what this
   * carries into the drawing.
   */
  readonly respectOrder?: boolean;
  /** The engine. Absent, the bundled build of `elkjs`, created once and reused. */
  readonly engine?: LayoutEngine;
  /** Raw ELK options, applied last: an escape hatch for measuring, never a substitute above. */
  readonly extra?: Readonly<Record<string, string>>;
}

/** The settings §4.7 asks for, as {@link layout} applies them when a field is absent. */
export const defaultLayoutSettings = {
  direction: 'DOWN',
  nodeSpacing: 28,
  layerSpacing: 36,
  edgeRouting: 'ORTHOGONAL',
  thoroughness: 7,
  respectOrder: true,
} as const satisfies Omit<Required<LayoutSettings>, 'engine' | 'extra'>;

/** A graph the engine cannot be asked to lay out: an id used twice, an edge with no endpoint. */
export class LayoutError extends Error {
  override readonly name = 'LayoutError';
}

let bundled: LayoutEngine | undefined;

/** The bundled engine, created on first use: constructing it is what loads a megabyte of ELK. */
function bundledEngine(): LayoutEngine {
  bundled ??= new ElkConstructor();
  return bundled;
}

function elkOptions(settings: LayoutSettings): LayoutOptions {
  const direction = settings.direction ?? defaultLayoutSettings.direction;
  const nodeSpacing = settings.nodeSpacing ?? defaultLayoutSettings.nodeSpacing;
  const layerSpacing = settings.layerSpacing ?? defaultLayoutSettings.layerSpacing;
  const edgeRouting = settings.edgeRouting ?? defaultLayoutSettings.edgeRouting;
  const thoroughness = settings.thoroughness ?? defaultLayoutSettings.thoroughness;
  const respectOrder = settings.respectOrder ?? defaultLayoutSettings.respectOrder;
  const options: Record<string, string> = {
    'elk.algorithm': 'layered',
    'elk.direction': direction,
    'elk.edgeRouting': edgeRouting,
    'elk.spacing.nodeNode': String(nodeSpacing),
    'elk.layered.spacing.nodeNodeBetweenLayers': String(layerSpacing),
    'elk.layered.thoroughness': String(thoroughness),
    // A composition's sites are laid out with the graph around them, so an edge from a root
    // instance to a site inside a composition is one edge and not two (§4.7's boundary handles
    // are a reading of that edge, not a second graph).
    'elk.hierarchyHandling': 'INCLUDE_CHILDREN',
  };
  if (respectOrder) options['elk.layered.considerModelOrder.strategy'] = 'NODES_AND_EDGES';
  return { ...options, ...settings.extra };
}

function insets(padding: Insets): string {
  return `[top=${padding.top},left=${padding.left},bottom=${padding.bottom},right=${padding.right}]`;
}

/**
 * The ELK graph, built from ours: ids, sizes, containment, padding, edge labels.
 *
 * A compound node is sized by what it holds and cannot be given a floor: `elk.nodeSize.minimumSize`
 * with `MINIMUM_SIZE` in `elk.nodeSize.constraints` is ignored on a node with children (measured
 * against elkjs 0.12, both the bare and the enum-set spelling), so a header wider than every card
 * inside the box is the component's business — the spike's drawing cuts it with an ellipsis, as a
 * card's `text-overflow` does.
 */
function toElk(node: GraphNode): ElkNode {
  const built: ElkNode = { id: node.id, width: node.width, height: node.height };
  if (node.padding !== undefined) built.layoutOptions = { 'elk.padding': insets(node.padding) };
  if (node.children !== undefined) built.children = node.children.map(toElk);
  return built;
}

function collectIds(nodes: readonly GraphNode[], into: Set<string>): void {
  for (const node of nodes) {
    if (into.has(node.id)) throw new LayoutError(`two nodes carry the id ${node.id}`);
    into.add(node.id);
    if (node.children !== undefined) collectIds(node.children, into);
  }
}

/**
 * Lay a graph out, top to bottom.
 *
 * The result is absolute: `x` and `y` of every node and every point of every edge are measured
 * from the drawing's corner, whatever the containment depth. A compound node's box encloses its
 * children, and no two boxes that are not one inside the other overlap — the properties the
 * layout spike (feature 0.4) holds this to, on the corpus's folded and expanded graphs.
 */
export async function layout(graph: Graph, settings: LayoutSettings = {}): Promise<Layout> {
  const ids = new Set<string>();
  collectIds(graph.nodes, ids);
  const edgeIds = new Set<string>();
  for (const edge of graph.edges) {
    if (edgeIds.has(edge.id)) throw new LayoutError(`two edges carry the id ${edge.id}`);
    edgeIds.add(edge.id);
    for (const end of [edge.source, edge.target]) {
      if (!ids.has(end)) throw new LayoutError(`edge ${edge.id} names ${end}, which is no node`);
    }
  }

  const root: ElkNode = {
    id: 'tensorspine.canvas',
    layoutOptions: elkOptions(settings),
    children: graph.nodes.map(toElk),
    edges: graph.edges.map((edge): ElkExtendedEdge => {
      const built: ElkExtendedEdge = { id: edge.id, sources: [edge.source], targets: [edge.target] };
      if (edge.label !== undefined) {
        built.labels = [
          { text: edge.label.text, width: edge.label.width, height: edge.label.height },
        ];
      }
      return built;
    }),
  };

  const engine = settings.engine ?? bundledEngine();
  const started = performance.now();
  const laid = await engine.layout(root);
  const milliseconds = performance.now() - started;

  const nodes: PlacedNode[] = [];
  const edges: RoutedEdge[] = [];
  const origins = new Map<string, Point>([[root.id, { x: 0, y: 0 }]]);
  const routes = new Map<string, ElkExtendedEdge>();

  // The nodes first, absolute, and every node's corner recorded: an edge's route is stated in
  // the coordinates of the node the engine assigned it to, which the edge names as `container`
  // and which is not the node it is listed under (elkjs answers every edge where it was given).
  const walk = (node: ElkNode, originX: number, originY: number, depth: number): void => {
    for (const edge of node.edges ?? []) routes.set(edge.id, edge);
    for (const child of node.children ?? []) {
      const x = originX + (child.x ?? 0);
      const y = originY + (child.y ?? 0);
      origins.set(child.id, { x, y });
      nodes.push({
        id: child.id,
        x,
        y,
        width: child.width ?? 0,
        height: child.height ?? 0,
        // The root of the ELK graph is this module's frame, not a node of the caller's: a node
        // sitting directly in it is at depth 0 and has no parent.
        parent: depth === 0 ? null : node.id,
        depth,
      });
      walk(child, x, y, depth + 1);
    }
  };
  walk(laid, 0, 0, 0);

  // The edges in the order the caller wrote them, whatever order the engine answered in.
  for (const declared of graph.edges) {
    const edge = routes.get(declared.id);
    if (edge === undefined) continue;
    const origin = origins.get(edge.container ?? root.id) ?? { x: 0, y: 0 };
    const points: Point[] = [];
    for (const section of edge.sections ?? []) {
      points.push({ x: origin.x + section.startPoint.x, y: origin.y + section.startPoint.y });
      for (const bend of section.bendPoints ?? []) {
        points.push({ x: origin.x + bend.x, y: origin.y + bend.y });
      }
      points.push({ x: origin.x + section.endPoint.x, y: origin.y + section.endPoint.y });
    }
    const routed: RoutedEdge = {
      id: edge.id,
      source: declared.source,
      target: declared.target,
      points,
    };
    const placedLabel = edge.labels?.[0];
    if (declared.label !== undefined && placedLabel !== undefined) {
      edges.push({
        ...routed,
        label: {
          ...declared.label,
          x: origin.x + (placedLabel.x ?? 0),
          y: origin.y + (placedLabel.y ?? 0),
        },
      });
    } else {
      edges.push(routed);
    }
  }

  return {
    width: laid.width ?? 0,
    height: laid.height ?? 0,
    nodes,
    edges,
    byId: new Map(nodes.map((node) => [node.id, node] as const)),
    milliseconds,
  };
}
