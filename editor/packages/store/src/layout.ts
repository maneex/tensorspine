/**
 * The layout sidecar: where the canvas keeps what the document may not carry (plan D6, §5.5).
 *
 * > The grammar closes every object (`additionalProperties: false`) and the specification's
 * > mutation test says a field no conforming implementation reads is a comment (§10.2) — the
 * > language would refuse it, rightly.
 *
 * So `<model>.layout.json` sits beside the document and holds the positions, the collapsed
 * groups, the viewport, a template's preview assignment and the expanded view's filters, keyed
 * by *the document's own paths* (`instances/embed`, `compositions/decoder/instances/attn`,
 * `interfaces/inputs/tokens`). Two consequences the plan states and this module implements:
 *
 * - **a key that names a path the document no longer has is dropped, with a log line** (D6);
 * - **the sidecar's keys follow a rename** (plan §3), which is what {@link moved} does with the
 *   moves a command answers.
 *
 * The file has its own schema and companion note in `editor/schemas/`, as the project's rule
 * asks of every schema; {@link LAYOUT_SCHEMA} is the `schema` member that schema fixes, and a
 * test holds the two to each other.
 *
 * The sidecar is the editor's own file and no tool of the repository reads or writes it, so it
 * is written with the plain writer rather than the core's (D12 binds the interchange formats —
 * the documents and the units — whose bytes the corpus fixes).
 */
import type { JsonObject } from '@tensorspine/lang';

import { EditLog, type Edit, type LogOptions } from './log.js';
import { keyOf, nodeAt, pathOfKey, type Path } from './path.js';
import type { PathMove } from './commands.js';

/**
 * The `schema` member of a layout sidecar, as plan §5.5 writes it.
 *
 * It is the editor's own format and not the language's: no schema of `schemas/` declares it, and
 * the no-hard-coding rule of plan §1 is about the *language's* vocabulary. `editor/schemas/`
 * holds the schema that fixes it, and `test/layout.test.ts` requires the two to agree, which is
 * the same discipline the core's semantic tables are held to.
 */
export const LAYOUT_SCHEMA = 'tensorspine-editor-layout/1';

/** A place on the canvas. */
export interface Position {
  readonly x: number;
  readonly y: number;
}

/** What the canvas is looking at. */
export interface Viewport {
  readonly x: number;
  readonly y: number;
  readonly zoom: number;
}

/** A range of one index of the expanded view (§4.9: "filtered by index range and family"). */
export interface IndexRange {
  readonly from?: number;
  readonly to?: number;
}

/** What the expanded view shows. */
export interface ExpandedView {
  readonly filters?: {
    readonly families?: readonly string[];
    readonly indices?: Readonly<Record<string, IndexRange>>;
  };
}

/** The sidecar as it is read and written. */
export interface Layout {
  readonly schema: string;
  /** A manual move, by the document path of the node it is about. */
  readonly positions: Readonly<Record<string, Position>>;
  /** The document paths of the groups collapsed on the canvas. */
  readonly collapsed: readonly string[];
  readonly viewport?: Viewport;
  /**
   * The assignment a template is previewed under (§4.6, D6) — quantity names to scalar literals.
   * It is not pruned against the document: an assignment is not part of it, and the tools ignore
   * a name the document does not declare (measured by feature 1.5).
   */
  readonly preview_assignment?: Readonly<Record<string, string | number | boolean>>;
  readonly expanded_view?: ExpandedView;
}

/** The sidecar while a command writes it: the same shape, mutable. */
export interface MutableLayout {
  schema: string;
  positions: Record<string, Position>;
  collapsed: string[];
  viewport?: Viewport;
  preview_assignment?: Record<string, string | number | boolean>;
  expanded_view?: ExpandedView;
}

/** A sidecar for a document nobody has moved anything in yet. */
export function emptyLayout(): Layout {
  return { schema: LAYOUT_SCHEMA, positions: {}, collapsed: [] };
}

/** One key the pruning dropped, with the line the log shows. */
export interface DroppedKey {
  /** `positions` or `collapsed`: which of the two the key was in. */
  readonly where: 'positions' | 'collapsed';
  /** The key itself, in the sidecar's own form. */
  readonly key: string;
  /** The log's line (D6: "dropped with a log line"). */
  readonly message: string;
}

/** What a pruning answers. */
export interface Pruned {
  readonly layout: Layout;
  readonly dropped: readonly DroppedKey[];
}

/**
 * Drop every key that names a place the document no longer has (D6).
 *
 * A key is a document path, so resolving it is a walk of the tree; a key that reads as nothing
 * the document holds is dropped and named in the log, never kept in case it comes back — a
 * position for a node that is not there is a position the user cannot see and cannot correct.
 */
export function prune(layout: Layout, tree: JsonObject): Pruned {
  const dropped: DroppedKey[] = [];
  const positions: Record<string, Position> = {};
  for (const [key, position] of Object.entries(layout.positions)) {
    if (resolves(tree, key)) positions[key] = position;
    else dropped.push(drop('positions', key));
  }
  const collapsed = layout.collapsed.filter((key) => {
    if (resolves(tree, key)) return true;
    dropped.push(drop('collapsed', key));
    return false;
  });
  if (dropped.length === 0) return { layout, dropped };
  return { layout: { ...layout, positions, collapsed }, dropped };
}

/** Whether a sidecar key names a place of the document. */
export function resolves(tree: JsonObject, key: string): boolean {
  const path = pathOfKey(key);
  if (path.length === 0) return false;
  return nodeAt(tree, path) !== undefined;
}

/**
 * The sidecar after a rename: every key at or below a moved place follows it (plan §3).
 *
 * A move of `compositions/decoder/instances/attn` carries the key of the site itself and of
 * everything under it, because a path is a prefix of the paths below it and the sidecar is keyed
 * by paths.
 */
export function moved(layout: Layout, moves: readonly PathMove[]): Layout {
  if (moves.length === 0) return layout;
  let any = false;
  const rewrite = (key: string): string => {
    for (const move of moves) {
      const from = keyOf(move.from);
      if (key === from) {
        any = true;
        return keyOf(move.to);
      }
      if (key.startsWith(`${from}/`)) {
        any = true;
        return `${keyOf(move.to)}${key.slice(from.length)}`;
      }
    }
    return key;
  };
  const positions: Record<string, Position> = {};
  for (const [key, position] of Object.entries(layout.positions)) positions[rewrite(key)] = position;
  const collapsed = layout.collapsed.map(rewrite);
  // A rename of something nothing is positioned at moves no key, and the sidecar that comes back
  // is the one that went in — so following it records no edit and costs no undo step.
  return any ? { ...layout, positions, collapsed } : layout;
}

/** The sidecar as its file holds it: two-space indentation and a trailing newline. */
export function writeLayout(layout: Layout): string {
  return `${JSON.stringify(layout, null, 2)}\n`;
}

/**
 * A sidecar read from its file.
 *
 * What is not the shape above is left out rather than refused: a sidecar is the editor's own
 * convenience, and a file another version wrote must not stop a document from opening. The
 * caller validates against the schema of `editor/schemas/` when it wants the file judged.
 */
export function readLayout(text: string): Layout {
  const one = asRecord(JSON.parse(text));
  if (one === null) return emptyLayout();
  const positions: Record<string, Position> = {};
  const written = asRecord(one['positions']);
  if (written !== null) {
    for (const [key, value] of Object.entries(written)) {
      const position = asPosition(value);
      if (position !== null) positions[key] = position;
    }
  }
  const collapsed = Array.isArray(one['collapsed'])
    ? (one['collapsed'] as unknown[]).filter((key): key is string => typeof key === 'string')
    : [];
  const layout: MutableLayout = {
    schema: typeof one['schema'] === 'string' ? one['schema'] : LAYOUT_SCHEMA,
    positions,
    collapsed,
  };
  const viewport = asViewport(one['viewport']);
  if (viewport !== null) layout.viewport = viewport;
  const assignment = asRecord(one['preview_assignment']);
  if (assignment !== null) {
    const found: Record<string, string | number | boolean> = {};
    for (const [key, value] of Object.entries(assignment)) {
      // The three shapes a `scalar_literal` takes. The two booleans are written out rather than
      // tested with `typeof`, because the word that test would need is a word of the language and
      // the interface's sources may not write one (plan §1 b).
      if (typeof value === 'string' || typeof value === 'number' || value === true || value === false) {
        found[key] = value;
      }
    }
    layout.preview_assignment = found;
  }
  const expanded = asExpandedView(one['expanded_view']);
  if (expanded !== null) layout.expanded_view = expanded;
  return layout;
}

function asPosition(value: unknown): Position | null {
  const one = asRecord(value);
  if (one === null) return null;
  return typeof one['x'] === 'number' && typeof one['y'] === 'number'
    ? { x: one['x'], y: one['y'] }
    : null;
}

function asExpandedView(value: unknown): ExpandedView | null {
  const one = asRecord(value);
  if (one === null) return null;
  const filters = asRecord(one['filters']);
  if (filters === null) return {};
  const found: { families?: readonly string[]; indices?: Record<string, IndexRange> } = {};
  const families = filters['families'];
  if (Array.isArray(families)) {
    found.families = (families as unknown[]).filter((name): name is string => typeof name === 'string');
  }
  const indices = asRecord(filters['indices']);
  if (indices !== null) {
    const ranges: Record<string, IndexRange> = {};
    for (const [name, range] of Object.entries(indices)) {
      const bounds = asRecord(range);
      if (bounds === null) continue;
      const kept: { from?: number; to?: number } = {};
      if (typeof bounds['from'] === 'number') kept.from = bounds['from'];
      if (typeof bounds['to'] === 'number') kept.to = bounds['to'];
      ranges[name] = kept;
    }
    found.indices = ranges;
  }
  return { filters: found };
}

/** A value read from a file, as a record of its members, or `null` where it is not one. */
function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function asViewport(value: unknown): Viewport | null {
  const one = asRecord(value);
  if (one === null) return null;
  return typeof one['x'] === 'number' && typeof one['y'] === 'number' && typeof one['zoom'] === 'number'
    ? { x: one['x'], y: one['y'], zoom: one['zoom'] }
    : null;
}

function drop(where: 'positions' | 'collapsed', key: string): DroppedKey {
  return {
    where,
    key,
    message: `layout: dropped ${where} '${key}' — the document has no such place`,
  };
}

/**
 * The layout's own command log (D13).
 *
 * "Layout moves are recorded in a separate log so an undo of a semantic edit does not shuffle
 * positions": the two logs never see each other, so an undo of "Add instance attn" takes back a
 * JSON edit and leaves every position where the user put it, and an undo of "Move attn" takes
 * back a position and leaves the document alone.
 */
export class LayoutStore {
  private readonly log: EditLog<Layout, MutableLayout>;

  constructor(initial: Layout = emptyLayout(), options: LogOptions = {}) {
    this.log = new EditLog<Layout, MutableLayout>(initial, options);
  }

  get layout(): Layout {
    return this.log.state;
  }

  get revision(): number {
    return this.log.revision;
  }

  get canUndo(): boolean {
    return this.log.canUndo;
  }

  get canRedo(): boolean {
    return this.log.canRedo;
  }

  get undoLabel(): string | null {
    return this.log.undoLabel;
  }

  get redoLabel(): string | null {
    return this.log.redoLabel;
  }

  /** Move a node: the one gesture the canvas records here (D6 — "a manual move is an override"). */
  move(path: Path, position: Position, label?: string): Edit<Layout> {
    const key = keyOf(path);
    return this.log.apply(label ?? `Move ${key}`, (draft) => {
      draft.positions[key] = position;
    });
  }

  /** Drop every override: "Reset layout" (D6). */
  reset(label = 'Reset layout'): Edit<Layout> {
    return this.log.apply(label, (draft) => {
      draft.positions = {};
    });
  }

  /** Collapse or expand a group. */
  collapse(path: Path, collapsed: boolean, label?: string): Edit<Layout> {
    const key = keyOf(path);
    return this.log.apply(label ?? `${collapsed ? 'Collapse' : 'Expand'} ${key}`, (draft) => {
      const found = draft.collapsed.indexOf(key);
      if (collapsed && found < 0) draft.collapsed.push(key);
      if (!collapsed && found >= 0) draft.collapsed.splice(found, 1);
    });
  }

  /** Where the canvas is looking. */
  look(viewport: Viewport, label = 'Move the viewport'): Edit<Layout> {
    return this.log.apply(label, (draft) => {
      draft.viewport = viewport;
    });
  }

  /** Follow a document command's moves (plan §3: "and the sidecar keys follow"). */
  follow(moves: readonly PathMove[], label = 'Follow a rename'): Edit<Layout> {
    const current = this.log.state;
    const next = moved(current, moves);
    return this.log.apply(label, (draft) => {
      if (next === current) return;
      draft.positions = { ...next.positions };
      draft.collapsed = [...next.collapsed];
    });
  }

  /** Drop the keys the document no longer has (D6), answering the log's lines. */
  prune(tree: JsonObject, label = 'Drop stale layout keys'): readonly DroppedKey[] {
    const { dropped } = prune(this.log.state, tree);
    if (dropped.length === 0) return dropped;
    this.log.apply(label, (draft) => {
      for (const one of dropped) {
        if (one.where === 'positions') delete draft.positions[one.key];
        else draft.collapsed = draft.collapsed.filter((key) => key !== one.key);
      }
    });
    return dropped;
  }

  undo(): Edit<Layout> | null {
    return this.log.undo();
  }

  redo(): Edit<Layout> | null {
    return this.log.redo();
  }

  subscribe(listener: (layout: Layout, revision: number) => void): () => void {
    return this.log.subscribe(listener);
  }
}
