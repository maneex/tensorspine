import { readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import type { Graph, GraphEdge, GraphNode, Insets } from '@tensorspine/ui/layout';

/**
 * The graphs the layout spike (feature 0.4) draws, projected from the repository's own material.
 *
 * Three readings of one model, the ones §4.7 and §4.9 name:
 *
 *   folded     the document as its author edits it — root instances, compositions as **compound
 *              nodes** holding one representative iteration of their sites, interfaces as
 *              terminals (§4.7, artboard S1 with a composition box opened in place);
 *   collapsed  the same, every composition shrunk to one box carrying the boundary handles the
 *              outside edges name (§4.7's default state, artboard S1 as drawn, and the reading
 *              `tools/view.py` gives at the top level);
 *   expanded   D1 — every emitted node, every edge, in D1's own topological order (§4.9,
 *              artboard S12).
 *
 * **Why this lives in the spike and not in `packages/ui`.** Reading a document means naming the
 * language's vocabulary: `kind: "root"`, `kind: "generated"`, the members of a value binding.
 * The governing rule of §1 forbids that in the interface's source — an item of information that
 * can be inferred from a schema is never hard-coded there — and §5.3 says where it belongs
 * instead: the core answers what an instance is, and feature 2.9 hands the canvas the facts it
 * draws. Until that core exists this file is the projection, and it is spike code: outside
 * `packages/`, read by the spike's script and by the tests that hold the layout to its
 * properties, shipped nowhere.
 *
 * What a box carries is what the *document itself* says — the name, the primitive, a guard, a
 * composition's index range and site count. Everything §4.7's card adds (present ports, slot
 * chips, the structural argument summary, the derived line) is the core's answer, so the spike
 * does not invent it: a card grows once `describe` answers, and the layout runs again with the
 * measured sizes.
 */

/** `editor/spikes/layout/`, where the spike's script writes and its SVGs and note live. */
export const spikeRoot = dirname(fileURLToPath(import.meta.url));

/** The repository this editor is part of (plan D14): the corpus is read at the source. */
const repositoryRoot = resolve(spikeRoot, '..', '..', '..');

/** `editor/`, whose `tests/oracle/out/` holds what `pnpm oracle` recorded from the tools. */
const editorRoot = resolve(spikeRoot, '..', '..');

/** The three models the feature names. */
export const spikeModels = ['llama3-8b', 'gemma3n-kvshare', 'deepseek-v4-pro'] as const;

// ---------------------------------------------------------------------------------------------
// What is read: the members of the two documents this spike looks at, written as TypeScript so
// the projection below is checked. They describe nothing the schemas do not, and are partial on
// purpose — a projection reads what it draws.
// ---------------------------------------------------------------------------------------------

interface PrimitiveReference {
  readonly name: string;
  readonly version: string;
}

interface InstanceDefinition {
  readonly primitive: PrimitiveReference;
  readonly families: readonly string[];
  readonly when?: unknown;
}

interface RootSelector {
  readonly kind: 'root';
  readonly instance: string;
}

interface GeneratedSelector {
  readonly kind: 'generated';
  readonly composition: string;
  readonly instance: string;
  readonly indices: Record<string, unknown>;
}

type InstanceSelector = RootSelector | GeneratedSelector;

interface ValueEndpoint {
  readonly instance: InstanceSelector;
  readonly port: string;
}

interface SiteEndpoint {
  readonly site: string;
  readonly indices?: Record<string, unknown>;
  readonly port: string;
}

type ScopedEndpoint = SiteEndpoint | ValueEndpoint;

interface ScopedValueBinding {
  readonly from: ScopedEndpoint;
  readonly to: ScopedEndpoint;
  readonly when?: unknown;
}

interface CompositionDefinition {
  readonly indices: Record<string, { start: unknown; stop: unknown; step?: unknown }>;
  readonly families: readonly string[];
  readonly instances: Record<string, InstanceDefinition>;
  readonly bindings?: { readonly values?: Record<string, ScopedValueBinding> };
}

/** A `tensorspine/2.0` document, in the members this spike reads. */
export interface ModelDocument {
  readonly model: string;
  readonly instances: Record<string, InstanceDefinition>;
  readonly compositions: Record<string, CompositionDefinition>;
  readonly bindings: {
    readonly values: Record<string, { from: ValueEndpoint; to: ValueEndpoint; when?: unknown }>;
  };
  readonly interfaces: {
    readonly inputs: Record<string, { to: readonly ValueEndpoint[]; kind: string }>;
    readonly outputs: Record<string, { from: ValueEndpoint; generative: boolean }>;
  };
}

/** A `--d1` document, in the members this spike reads (the derived schema's `d1`). */
export interface ExpandedDocument {
  readonly model: string;
  readonly d1: {
    readonly nodes: Record<string, { primitive: PrimitiveReference; families: readonly string[] }>;
    readonly edges: readonly {
      readonly rule: string;
      readonly from: { node: string; port: string };
      readonly to: { node: string; port: string };
    }[];
    readonly topological_order: readonly string[];
  };
}

// ---------------------------------------------------------------------------------------------
// What is drawn: a box's caption, beside the geometry the layout works on.
// ---------------------------------------------------------------------------------------------

/** What a box stands for. The SVG gives each kind its own ink, as `_ts.css` does. */
export type BoxKind = 'instance' | 'site' | 'composition' | 'input' | 'output' | 'expanded';

/** One box's caption: what the document says about it, and nothing derived. */
export interface Box {
  readonly id: string;
  readonly kind: BoxKind;
  /** The map key — an instance name, a site name, a composition name, a D1 identifier. */
  readonly title: string;
  /** `primitive@version`, a composition's index range, an interface's kind. */
  readonly subtitle: string;
  /** A guard, a carry badge, a boundary handle, the site count: one line each. */
  readonly notes: readonly string[];
}

/** The three readings of a document this spike lays out. */
export type Reading = 'folded' | 'collapsed' | 'expanded';

/** A graph with its captions and the sentence that says which reading it is. */
export interface Diagram {
  readonly id: string;
  readonly model: string;
  readonly reading: Reading;
  readonly caption: string;
  readonly graph: Graph;
  readonly boxes: ReadonlyMap<string, Box>;
}

// ---------------------------------------------------------------------------------------------
// Metrics. Every measure is `plans/graph-editor-design/_ts.css`'s, read off the component the box
// stands for; the text widths are estimated from the type sizes that file sets, since nothing
// here has a DOM to measure. The real canvas measures the rendered card and lays out again when
// a card's size changes — a spike cannot, and does not pretend to.
// ---------------------------------------------------------------------------------------------

/** `.node`: `width: 214px`, `padding: 10px 12px`, `gap: 5px`. */
const CARD = { width: 214, padTop: 10, padBottom: 10, gap: 5 } as const;
/** `.n-head b` 13.5px mono, `.n-fams em` 9.5px in a chip, `.n-guard` 10px mono. */
const CARD_ROWS = { head: 18, families: 15, note: 14 } as const;
/** `.group`: `width: 300px` collapsed, `padding: 10px 12px`, `gap: 6px`. */
const GROUP = { collapsedWidth: 300, padTop: 10, padBottom: 10, padSide: 12, gap: 6 } as const;
/** `.g-head` 14px mono, `.g-fams` chips, `.bh` handles 9px mono, `.g-sites` 9.5px. */
const GROUP_ROWS = { head: 20, families: 15, note: 18, sites: 14 } as const;
/** `.term`: a dashed pill, `padding: 8px 14px`, `.tname` 12.5px mono beside `.tkind` 9.5px. */
const TERMINAL = { height: 34, padSide: 14, gap: 9 } as const;
/** `.xn2`: one row, `padding: 6px 10px`, `.xn2 b` 11px mono beside a 9.5px primitive. */
const EXPANDED = { height: 28, padSide: 10, gap: 8 } as const;

/** IBM Plex Mono at `size` px: its advance is a constant 0.6 em. */
const mono = (text: string, size: number): number => text.length * size * 0.6;
/** IBM Plex Sans at `size` px: 0.55 em is close enough for a caption's reserved width. */
const sans = (text: string, size: number): number => text.length * size * 0.55;

function sum(values: readonly number[]): number {
  return values.reduce((a, b) => a + b, 0);
}

function cardHeight(box: Box): number {
  const rows = [CARD_ROWS.head, CARD_ROWS.families, ...box.notes.map(() => CARD_ROWS.note)];
  return CARD.padTop + CARD.padBottom + sum(rows) + CARD.gap * (rows.length - 1);
}

function collapsedHeight(box: Box): number {
  const rows = [GROUP_ROWS.head, GROUP_ROWS.families, ...box.notes.map(() => GROUP_ROWS.note)];
  return GROUP.padTop + GROUP.padBottom + sum(rows) + GROUP.gap * (rows.length - 1);
}

/** A composition box's own chrome, which is the room its children may not use. */
function groupPadding(box: Box): Insets {
  const rows = [GROUP_ROWS.head, GROUP_ROWS.families, ...box.notes.map(() => GROUP_ROWS.note)];
  return {
    top: GROUP.padTop + sum(rows) + GROUP.gap * rows.length,
    right: GROUP.padSide,
    bottom: GROUP.padBottom + GROUP_ROWS.sites,
    left: GROUP.padSide,
  };
}

// ---------------------------------------------------------------------------------------------
// Printing an expression and a condition. Generic on purpose: §1 says an operator without a
// presentation binding renders as `name(a, b)`, and `presentation.json` (feature 2.2) is where
// the symbols will come from. Nothing here invents one.
// ---------------------------------------------------------------------------------------------

function printExpression(node: unknown): string {
  if (node === null || typeof node !== 'object') return JSON.stringify(node);
  const value = node as Record<string, unknown>;
  if ('literal' in value) return JSON.stringify(value['literal']);
  if ('quantity' in value) return String(value['quantity']);
  if ('index' in value) return String(value['index']);
  if ('op' in value) {
    const args = Array.isArray(value['args']) ? (value['args'] as unknown[]) : [];
    return `${String(value['op'])}(${args.map(printExpression).join(', ')})`;
  }
  if ('if' in value) {
    return `if(${printCondition(value['if'])}, ${printExpression(value['then'])}, ${printExpression(value['else'])})`;
  }
  return JSON.stringify(node);
}

function printCondition(node: unknown): string {
  if (node === null || typeof node !== 'object') return JSON.stringify(node);
  const value = node as Record<string, unknown>;
  if ('boolean' in value) return String(value['boolean']);
  if ('not' in value) return `not(${printCondition(value['not'])})`;
  for (const key of ['all', 'any'] as const) {
    const members = value[key];
    if (Array.isArray(members)) {
      return `${key}(${(members as unknown[]).map(printCondition).join(', ')})`;
    }
  }
  if ('compare' in value) {
    const compare = value['compare'] as Record<string, unknown>;
    return `${String(compare['operator'])}(${printExpression(compare['left'])}, ${printExpression(compare['right'])})`;
  }
  return JSON.stringify(node);
}

function ellipsis(text: string, characters: number): string {
  return text.length <= characters ? text : `${text.slice(0, characters - 1)}…`;
}

/** An index assignment as §5.2 writes it: `[layer=0]`, `[layer=subtract(layer, 1)]`. */
function printIndices(indices: Record<string, unknown> | undefined): string {
  const parts = Object.entries(indices ?? {}).map(
    ([name, value]) => `${name}=${printExpression(value)}`,
  );
  return parts.length === 0 ? '' : `[${parts.join(', ')}]`;
}

/**
 * A composition's header, as §4.7 words it: `layer ∈ [0, 32) · ×32`, "the count only when the
 * range resolves — the same rule as `--view`". A bound that is not a literal leaves the count
 * off: resolving a quantity is the core's answer, never a projection's — E10 of the inventory's
 * errata is exactly the error of assuming otherwise.
 */
function rangeLabel(composition: CompositionDefinition): string {
  const parts: string[] = [];
  let count: number | null = 1;
  for (const [name, range] of Object.entries(composition.indices)) {
    const start = range.start as Record<string, unknown>;
    const stop = range.stop as Record<string, unknown>;
    const step = (range.step ?? { literal: 1 }) as Record<string, unknown>;
    parts.push(`${name} ∈ [${printExpression(start)}, ${printExpression(stop)})`);
    const bounds = [start['literal'], stop['literal'], step['literal']];
    if (count !== null && bounds.every((bound) => typeof bound === 'number')) {
      const [from, to, by] = bounds as [number, number, number];
      count *= by === 0 ? 0 : Math.max(0, Math.ceil((to - from) / by));
    } else {
      count = null;
    }
  }
  return count === null ? parts.join(' · ') : `${parts.join(' · ')} · ×${count}`;
}

// ---------------------------------------------------------------------------------------------
// The projections.
// ---------------------------------------------------------------------------------------------

const rootId = (name: string): string => `instance:${name}`;
const siteId = (composition: string, site: string): string => `site:${composition}/${site}`;
const compositionId = (name: string): string => `composition:${name}`;
const inputId = (name: string): string => `input:${name}`;
const outputId = (name: string): string => `output:${name}`;

function isSiteEndpoint(endpoint: ScopedEndpoint): endpoint is SiteEndpoint {
  return 'site' in endpoint;
}

function endpointId(endpoint: ValueEndpoint, collapsed: boolean): string {
  const selector = endpoint.instance;
  if (selector.kind === 'root') return rootId(selector.instance);
  return collapsed
    ? compositionId(selector.composition)
    : siteId(selector.composition, selector.instance);
}

/** An edge label, sized for the rule name §4.7 puts on a value binding, in small type. */
function label(text: string): { text: string; width: number; height: number } {
  return { text, width: mono(text, 9.5), height: 12 };
}

/** What fits on a `.node` card: 214 px less its padding, at `.n-guard`'s 10 px mono. */
const CARD_LINE = 33;

function instanceBox(
  id: string,
  kind: BoxKind,
  name: string,
  definition: InstanceDefinition,
  extra: readonly string[] = [],
): Box {
  const notes: string[] = [];
  if (definition.when !== undefined) notes.push(`⚑ when ${printCondition(definition.when)}`);
  return {
    id,
    kind,
    title: name,
    subtitle: `${definition.primitive.name}@${definition.primitive.version}`,
    notes: [...notes, ...extra].map((line) => ellipsis(line, CARD_LINE)),
  };
}

/**
 * The folded document (§4.7). `reading` chooses between the two states of a composition box:
 * `folded` opens it in place as a compound node holding one representative iteration of its
 * sites; `collapsed` shrinks it to one box carrying the boundary handles the outside edges name,
 * which is §4.7's default and what artboard S1 draws.
 */
export function foldedDiagram(
  model: string,
  document: ModelDocument,
  reading: 'folded' | 'collapsed',
): Diagram {
  const collapsed = reading === 'collapsed';
  const boxes = new Map<string, Box>();
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];
  const handles = new Map<string, string[]>();

  const note = (map: Map<string, string[]>, id: string, line: string): void => {
    const lines = map.get(id) ?? [];
    if (!lines.includes(line)) lines.push(line);
    map.set(id, lines);
  };

  // The boundary handles first: they are what an outside edge names, and a collapsed box has to
  // carry them before it can be sized.
  for (const binding of Object.values(document.bindings.values)) {
    for (const endpoint of [binding.from, binding.to]) {
      const selector = endpoint.instance;
      if (selector.kind !== 'generated') continue;
      note(
        handles,
        compositionId(selector.composition),
        `${selector.instance}${printIndices(selector.indices)}.${endpoint.port}`,
      );
    }
  }
  for (const [name, composition] of Object.entries(document.compositions)) {
    // A scoped binding may name an instance outside the composition (§5.2's `scoped_value_endpoint`
    // admits a plain value endpoint): the site side of such a binding is a boundary handle too.
    for (const binding of Object.values(composition.bindings?.values ?? {})) {
      const outside = [binding.from, binding.to].some((endpoint) => !isSiteEndpoint(endpoint));
      if (!outside) continue;
      for (const endpoint of [binding.from, binding.to]) {
        if (!isSiteEndpoint(endpoint)) continue;
        note(
          handles,
          compositionId(name),
          `${endpoint.site}${printIndices(endpoint.indices)}.${endpoint.port}`,
        );
      }
    }
  }

  // Terminals: "an input is a terminal node ◁ tokens · token on the left" (§4.7).
  for (const [name, input] of Object.entries(document.interfaces.inputs)) {
    const box: Box = {
      id: inputId(name),
      kind: 'input',
      title: `◁ ${name}`,
      subtitle: input.kind,
      notes: [],
    };
    boxes.set(box.id, box);
    nodes.push({
      id: box.id,
      width: TERMINAL.padSide * 2 + TERMINAL.gap + mono(box.title, 12.5) + sans(box.subtitle, 9.5),
      height: TERMINAL.height,
    });
  }
  for (const [name, output] of Object.entries(document.interfaces.outputs)) {
    const box: Box = {
      id: outputId(name),
      kind: 'output',
      title: `${name} ▷`,
      subtitle: output.generative ? 'generative' : 'value',
      notes: [],
    };
    boxes.set(box.id, box);
    nodes.push({
      id: box.id,
      width: TERMINAL.padSide * 2 + TERMINAL.gap + mono(box.title, 12.5) + sans(box.subtitle, 9.5),
      height: TERMINAL.height,
    });
  }

  // Root instances.
  for (const [name, definition] of Object.entries(document.instances)) {
    const box = instanceBox(rootId(name), 'instance', name, definition);
    boxes.set(box.id, box);
    nodes.push({ id: box.id, width: CARD.width, height: cardHeight(box) });
  }

  // Compositions, in one of their two states.
  for (const [name, composition] of Object.entries(document.compositions)) {
    const sites = Object.keys(composition.instances);
    const carries = new Map<string, string[]>();
    const scoped: GraphEdge[] = [];

    for (const [rule, binding] of Object.entries(composition.bindings?.values ?? {})) {
      const { from, to } = binding;
      const target = isSiteEndpoint(to) ? siteId(name, to.site) : endpointId(to, collapsed);
      // A site endpoint with an index override names another iteration: drawn from a ghost column
      // (§4.8, D8), never as an edge inside one representative iteration, where it would look
      // like a cycle without being one — `--view` gives it as a badge for the same reason.
      if (isSiteEndpoint(from) && from.indices !== undefined) {
        note(carries, target, `↺ carry from ${from.site}${printIndices(from.indices)}`);
        continue;
      }
      const source = isSiteEndpoint(from) ? siteId(name, from.site) : endpointId(from, collapsed);
      if (source === target) continue;
      if (collapsed) {
        const a = source.startsWith('site:') ? compositionId(name) : source;
        const b = target.startsWith('site:') ? compositionId(name) : target;
        if (a === b) continue;
        scoped.push({ id: `scoped:${name}.${rule}`, source: a, target: b, label: label(rule) });
        continue;
      }
      scoped.push({ id: `scoped:${name}.${rule}`, source, target, label: label(rule) });
    }
    edges.push(...scoped);

    if (collapsed) {
      const box: Box = {
        id: compositionId(name),
        kind: 'composition',
        title: name,
        subtitle: rangeLabel(composition),
        notes: [...(handles.get(compositionId(name)) ?? []), `${sites.length} sites per iteration`],
      };
      boxes.set(box.id, box);
      nodes.push({
        id: box.id,
        width: Math.max(
          GROUP.collapsedWidth,
          GROUP.padSide * 2 + Math.max(...box.notes.map((line) => mono(line, 9))),
        ),
        height: collapsedHeight(box),
      });
      continue;
    }

    const children: GraphNode[] = [];
    for (const [site, definition] of Object.entries(composition.instances)) {
      const id = siteId(name, site);
      const box = instanceBox(id, 'site', site, definition, carries.get(id) ?? []);
      boxes.set(id, box);
      children.push({ id, width: CARD.width, height: cardHeight(box) });
    }
    const box: Box = {
      id: compositionId(name),
      kind: 'composition',
      title: name,
      subtitle: rangeLabel(composition),
      notes: [`${sites.length} sites per iteration · families ${composition.families.join(', ')}`],
    };
    boxes.set(box.id, box);
    nodes.push({ id: box.id, width: 0, height: 0, children, padding: groupPadding(box) });
  }

  // The top-level value bindings, and the edges the interfaces stand at the ends of.
  for (const [rule, binding] of Object.entries(document.bindings.values)) {
    const source = endpointId(binding.from, collapsed);
    const target = endpointId(binding.to, collapsed);
    if (source === target) continue;
    edges.push({ id: `value:${rule}`, source, target, label: label(rule) });
  }
  for (const [name, input] of Object.entries(document.interfaces.inputs)) {
    for (const endpoint of input.to) {
      const target = endpointId(endpoint, collapsed);
      edges.push({ id: `interface:in.${name}.${target}`, source: inputId(name), target });
    }
  }
  for (const [name, output] of Object.entries(document.interfaces.outputs)) {
    edges.push({
      id: `interface:out.${name}`,
      source: endpointId(output.from, collapsed),
      target: outputId(name),
    });
  }

  return {
    id: `${model}.${reading}`,
    model,
    reading,
    caption: collapsed
      ? `${model} · folded document, every composition collapsed to its boundary handles`
      : `${model} · folded document, every composition opened as a compound node`,
    graph: { nodes, edges: collapsed ? deduplicate(edges) : edges },
    boxes,
  };
}

/** One arrow per pair, as `--view` draws the collapsed reading; the count rides on the label. */
function deduplicate(edges: readonly GraphEdge[]): GraphEdge[] {
  const byPair = new Map<string, { edge: GraphEdge; rules: string[] }>();
  for (const edge of edges) {
    const key = `${edge.source} ${edge.target}`;
    const seen = byPair.get(key);
    if (seen === undefined) byPair.set(key, { edge, rules: edge.label === undefined ? [] : [edge.label.text] });
    else if (edge.label !== undefined) seen.rules.push(edge.label.text);
  }
  return [...byPair.values()].map(({ edge, rules }) => {
    if (edge.label === undefined) return edge;
    return { ...edge, label: label(rules.length === 1 ? (rules[0] ?? '') : `${rules.length} bindings`) };
  });
}

/**
 * The expanded graph (§4.9): every node D1 emits, every edge, in D1's topological order — which
 * the layout carries into the drawing as the order that breaks its ties.
 */
export function expandedDiagram(model: string, document: ExpandedDocument): Diagram {
  const d1 = document.d1;
  const boxes = new Map<string, Box>();
  const ordered = new Set([...d1.topological_order, ...Object.keys(d1.nodes)]);
  let width = 0;
  for (const id of ordered) {
    const node = d1.nodes[id];
    if (node === undefined) continue;
    boxes.set(id, { id, kind: 'expanded', title: id, subtitle: node.primitive.name, notes: [] });
    width = Math.max(
      width,
      EXPANDED.padSide * 2 + EXPANDED.gap + mono(id, 11) + sans(node.primitive.name, 9.5),
    );
  }
  return {
    id: `${model}.expanded`,
    model,
    reading: 'expanded',
    caption: `${model} · expanded graph, ${boxes.size} nodes and ${d1.edges.length} edges from D1`,
    graph: {
      nodes: [...boxes.keys()].map((id) => ({ id, width, height: EXPANDED.height })),
      edges: d1.edges.map((edge, index) => ({
        id: `${edge.rule}#${index}`,
        source: edge.from.node,
        target: edge.to.node,
      })),
    },
    boxes,
  };
}

// ---------------------------------------------------------------------------------------------
// Reading the material.
// ---------------------------------------------------------------------------------------------

/** A corpus document, read at the source (`data/models/<name>.json`). */
export function readModel(name: string): ModelDocument {
  return JSON.parse(
    readFileSync(join(repositoryRoot, 'data', 'models', `${name}.json`), 'utf8'),
  ) as ModelDocument;
}

/** Where `pnpm oracle` writes `--d1` for a corpus document (plan §0.5). */
export function expandedPath(name: string): string {
  return join(editorRoot, 'tests', 'oracle', 'out', 'd1', `${name}.d1.json`);
}

/**
 * D1 for a corpus document, as the tools wrote it. The oracle is its only producer here: the
 * editor runs no tool of its own, and the core that will expand a document in the browser is
 * feature 1.7. Without `pnpm oracle` this refuses rather than draw something else.
 */
export function readExpanded(name: string): ExpandedDocument {
  const path = expandedPath(name);
  try {
    return JSON.parse(readFileSync(path, 'utf8')) as ExpandedDocument;
  } catch {
    throw new Error(`${path} is missing — run \`pnpm oracle\` from editor/ first`);
  }
}

/** Every diagram the spike draws for one model: folded, collapsed, expanded. */
export function diagramsFor(name: string): Diagram[] {
  const document = readModel(name);
  return [
    foldedDiagram(name, document, 'folded'),
    foldedDiagram(name, document, 'collapsed'),
    expandedDiagram(name, readExpanded(name)),
  ];
}
