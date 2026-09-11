/**
 * The Problems panel's row model — plan §4.17, artboard S10, component inventory §5.
 *
 * > The message log. Columns: severity · source · code · message · location. […] Behaviour: click
 * > navigates (selects the node, opens the sheet on the row, or reveals the source range); filter
 * > by source/severity/text; group by node or by unit; a banner when a stage refused and the rows
 * > below it are "after a refusal" (§3); the count in the status bar; a problem's code links to
 * > its anchor in the specification. **The panel never invents a problem: every row but the
 * > editor's own notices is a verdict of the core.**
 *
 * So nothing here decides anything about a document. Every row arrives as a {@link Problem} the
 * core built — its `severity` and its `source` are the core's, its `message` is the tools' line
 * word for word (the parity contract), its `path` is the place in the document as written — and
 * what this module does is *arrange* them: fold the repetitions, put each in a group, apply the
 * filter, and say where the row goes when it is clicked.
 *
 * **Folding, and why it is not hiding.** `analyse` emits one refusal per *iteration*: a scoped
 * parameter rule that names a slot the primitive has not produces the identical line thirty-two
 * times on `llama3-8b` — same code, same message, same place, no index anywhere in it. Thirty-two
 * identical lines is not a reading of anything, so identical rows are shown once with the count
 * beside them, and the count is the core's own row count: the sum over the panel equals what
 * `validate` answered, which a suite asserts. Nothing is dropped and no row is merged with a row
 * that differs in any field a reader can see.
 *
 * **Grouping by node is grouping by *place*.** §4.17 says "group by node or by unit". A row names
 * a place in the document and, for a checkpoint row, a derived identity; what it does **not**
 * carry is the site identifier its message names — "parsing an identifier back out of a message
 * would be a second reading of the wording the parity job pins" (the core's own note, and finding
 * F1 is the change set that would fix it at the source). So the node a row is grouped under is the
 * deepest thing the *document* declares at or above its pointer, which is exactly what the Model
 * explorer's own outline says (feature 2.7) — an instance, a site, a quantity, a binding rule, a
 * public input — and a row that names no place at all is the document's. S10 groups by the site a
 * message names (`decoder/attn[layer=5]`); this groups by the place, which is the same thing
 * wherever the pointer is a site and finer wherever it is a binding.
 */
import type { Problem, ProblemSeverity, ProblemSource } from '@tensorspine/lang/api';

import type { OutlineRow } from '../explorer/outline.js';

/**
 * How the rows are grouped (§4.17: "group by node or by unit"), or not at all.
 *
 * The third is **`flat`** and not `none`, which is what it reads as: `none` is a value of the
 * schemas' `mask` and `partition_options` enumerations, and catching rule (b) is a whole-literal
 * scan that cannot tell the panel's vocabulary from the language's — the sixth feature running to
 * meet the trap, and the sixth better name for it.
 */
export type Grouping = 'node' | 'unit' | 'flat';

/** The grouping that groups nothing: one list, in the order the core produced the rows. */
export const FLAT = 'flat';

/** What the filter keeps (§4.17: "filter by source/severity/text"). */
export interface ProblemFilter {
  /** The sources kept; every source when it is empty. */
  readonly sources: readonly ProblemSource[];
  /** The severities kept; every severity when it is empty. */
  readonly severities: readonly ProblemSeverity[];
  /** Matched against the message, the code, the place and the file, case-insensitively. */
  readonly text: string;
}

/** The filter that keeps everything — what the panel opens with. */
export const NO_FILTER: ProblemFilter = { sources: [], severities: [], text: '' };

/** One problem as it arrives, with the freshness of the reading it came from (§5.4). */
export interface Sourced {
  readonly problem: Problem;
  /** True where it was computed for an older revision than the document's: shown dimmed. */
  readonly stale?: boolean;
}

/**
 * A place the document declares, as the outline of §4.5 reads it.
 *
 * The panel is handed the places rather than the tree: what it needs of a document is which
 * pointers name something with a name, and the outline already answers that for the tree, the
 * sheet and the red dot. One reading, three readers.
 */
export interface DeclaredPlace {
  /** The place, as an RFC 6901 pointer. */
  readonly pointer: string;
  /** What the document calls it. */
  readonly label: string;
  /** What the map it is declared in declares — `presentation.json`'s word (§4.4). */
  readonly declares?: string;
}

/** The document the panel is about, and what grouping and navigation resolve against. */
export interface AboutDocument {
  /** The workspace path — what a row's `file` is compared with. */
  readonly path: string;
  /** What the tab calls it: the document group's label. */
  readonly title: string;
  /** Every place the document declares, deepest last is not required. */
  readonly places: readonly DeclaredPlace[];
}

/**
 * What the panel is about, read off the document's own outline.
 *
 * The places are the outline's named entries — an instance, a site, a quantity, a binding rule, a
 * public input — because that is what the document *declares*, and grouping by node is grouping
 * by the deepest of them a row's pointer falls in.
 */
export function aboutOf(
  rows: readonly OutlineRow[],
  path: string,
  title: string,
): AboutDocument {
  const places: DeclaredPlace[] = [];
  for (const row of rows) {
    if (row.kind !== 'entry' || !row.named) continue;
    places.push({
      pointer: row.pointer,
      label: row.label,
      ...(row.declares === undefined ? {} : { declares: row.declares }),
    });
  }
  return { path, title, places };
}

/** One row of the panel. */
export interface ProblemRow {
  /** A key stable across renders: the fold's own key. */
  readonly key: string;
  /** What the core said. */
  readonly problem: Problem;
  /** How many identical rows of the core's this stands for. */
  readonly count: number;
  /** Computed for an older revision than the document's (§5.4's freshness). */
  readonly stale: boolean;
  /** The place it navigates to, where the row names one the document has. */
  readonly place?: DeclaredPlace;
}

/** One group of rows (§4.17, S10's `.pgrp`). */
export interface ProblemGroup {
  /** A key stable across renders. */
  readonly key: string;
  /** What the heading says. */
  readonly label: string;
  /** What the heading's title says: the place, or the file. */
  readonly detail?: string;
  readonly rows: readonly ProblemRow[];
  /** How many rows of the core's the group stands for. */
  readonly count: number;
}

/** What the panel draws. */
export interface ProblemsView {
  readonly groups: readonly ProblemGroup[];
  /** How many rows of the core's each severity carries, before the filter. */
  readonly counts: Readonly<Record<ProblemSeverity, number>>;
  /** Which sources are present at all — what the source filter offers. */
  readonly sources: readonly ProblemSource[];
  /** How many rows of the core's the panel is showing. */
  readonly shown: number;
  /** How many it holds in all. */
  readonly total: number;
}

/** What {@link problemsView} is given. */
export interface ProblemsRequest {
  readonly rows: readonly Sourced[];
  readonly about?: AboutDocument;
  readonly group?: Grouping;
  readonly filter?: ProblemFilter;
}

/**
 * The key two rows must share to be one row: everything a reader can see.
 *
 * It is exported because the panel gathers its rows from several places — the page's own Ajv, the
 * verdict, the library the workspace gathered, the lint run, the editor's notices — and two of
 * them can carry the same row: the loader's refusals are the *workspace's* and the verdict carries
 * the ones the document resolved against. A duplicate has to go before the fold rather than be
 * folded into a `× 2` that would say the core answered twice.
 */
export function foldKey(problem: Problem): string {
  return JSON.stringify([
    problem.source,
    problem.severity,
    problem.code,
    problem.message,
    problem.path,
    problem.file ?? '',
    problem.rule ?? '',
    problem.node ?? '',
    problem.afterRefusal === true,
    (problem.detail ?? []).map((one) => [one.message, one.path]),
  ]);
}

/** Whether a row passes the filter. */
function kept(problem: Problem, filter: ProblemFilter): boolean {
  if (filter.sources.length > 0 && !filter.sources.includes(problem.source)) return false;
  if (filter.severities.length > 0 && !filter.severities.includes(problem.severity)) return false;
  const text = filter.text.trim().toLowerCase();
  if (text === '') return true;
  const haystack = [problem.message, problem.code, problem.path, problem.file ?? '', problem.node ?? '']
    .join(' ')
    .toLowerCase();
  return haystack.includes(text);
}

/**
 * The deepest place the document declares at or above a pointer.
 *
 * Only for a row about **this** document: a row that names another file names a place of *that*
 * one, and resolving it here would send a click to whatever the current document happens to have
 * at the same pointer.
 */
function placeOf(
  about: AboutDocument | undefined,
  problem: Problem,
): DeclaredPlace | undefined {
  if (about === undefined) return undefined;
  if (problem.file !== undefined && problem.file !== about.path) return undefined;
  const pointer = problem.path;
  let found: DeclaredPlace | undefined;
  for (const place of about.places) {
    if (place.pointer === '') continue;
    if (pointer !== place.pointer && !pointer.startsWith(`${place.pointer}/`)) continue;
    if (found !== undefined && found.pointer.length >= place.pointer.length) continue;
    found = place;
  }
  return found;
}

/** The group a row falls in, and what its heading says. */
function groupOf(
  row: ProblemRow,
  about: AboutDocument | undefined,
  group: Grouping,
): { key: string; label: string; detail?: string } {
  const file = row.problem.file;
  if (group === FLAT) return { key: '', label: '' };
  if (group === 'unit') {
    const named = file ?? about?.path ?? '';
    return named === '' ? { key: '', label: '' } : { key: named, label: baseName(named), detail: named };
  }
  // By node. A row about another file is that file's, whatever its pointer says: the pointer is a
  // place of *that* document and this panel is not showing it.
  if (file !== undefined && about !== undefined && file !== about.path) {
    return { key: `file:${file}`, label: baseName(file), detail: file };
  }
  if (row.place !== undefined) {
    return {
      key: `place:${row.place.pointer}`,
      label: row.place.label,
      detail: row.place.pointer,
    };
  }
  if (about !== undefined) return { key: 'document', label: about.title, detail: about.path };
  return { key: '', label: '' };
}

/** The text after the last separator — a workspace path is `/`-separated everywhere (§5.2). */
function baseName(path: string): string {
  const cut = path.lastIndexOf('/');
  return cut === -1 ? path : path.slice(cut + 1);
}

/**
 * The panel's rows, folded, grouped and filtered.
 *
 * The order is the core's: the stages produce their rows in the order the tools produce them, and
 * a group appears where its first row does. Nothing is sorted by severity or by code — the reading
 * `--validate` gives is the reading the panel gives.
 */
export function problemsView(request: ProblemsRequest): ProblemsView {
  const group = request.group ?? 'node';
  const filter = request.filter ?? NO_FILTER;
  const about = request.about;

  const counts: Record<ProblemSeverity, number> = { error: 0, warning: 0, notice: 0 };
  const sources: ProblemSource[] = [];
  const folded = new Map<string, { row: ProblemRow; count: number; stale: boolean }>();
  let total = 0;
  let shown = 0;

  for (const one of request.rows) {
    total += 1;
    counts[one.problem.severity] += 1;
    if (!sources.includes(one.problem.source)) sources.push(one.problem.source);
    if (!kept(one.problem, filter)) continue;
    shown += 1;
    const key = foldKey(one.problem);
    const held = folded.get(key);
    if (held !== undefined) {
      held.count += 1;
      // A fold is stale only where every row it stands for is: a fresh reading of one of them is
      // a fresh reading of the line.
      held.stale = held.stale && one.stale === true;
      continue;
    }
    const place = placeOf(about, one.problem);
    folded.set(key, {
      row: {
        key,
        problem: one.problem,
        count: 1,
        stale: one.stale === true,
        ...(place === undefined ? {} : { place }),
      },
      count: 1,
      stale: one.stale === true,
    });
  }

  const groups: { key: string; label: string; detail?: string; rows: ProblemRow[]; count: number }[] = [];
  const byKey = new Map<string, (typeof groups)[number]>();
  for (const held of folded.values()) {
    const row: ProblemRow = { ...held.row, count: held.count, stale: held.stale };
    const named = groupOf(row, about, group);
    let found = byKey.get(named.key);
    if (found === undefined) {
      found = {
        key: named.key,
        label: named.label,
        ...(named.detail === undefined ? {} : { detail: named.detail }),
        rows: [],
        count: 0,
      };
      byKey.set(named.key, found);
      groups.push(found);
    }
    found.rows.push(row);
    found.count += row.count;
  }

  return { groups, counts, sources, shown, total };
}

