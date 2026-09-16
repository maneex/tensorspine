/**
 * The Problems panel — plan §4.17, artboard S10, component inventory §5.
 *
 * > Columns: severity · source · code · message · location. […] click navigates (selects the
 * > node, opens the sheet on the row, or reveals the source range); filter by source/severity/text;
 * > group by node or by unit; a banner when a stage refused and the rows below it are "after a
 * > refusal"; the count in the status bar; a problem's code links to its anchor in the
 * > specification (generated map, §1).
 *
 * Every column is a field of the row the core answered, printed as it stands: the source is
 * `problem.source` and never a word this file chooses (`.src` "**is the source** — schema,
 * library, semantic, lint, checkpoint, editor, derivation — never the severity"), the message is
 * the tools' line, and the location is the place in the document as written. The code is a link
 * because `docs/SPECIFICATION.md`'s anchors are vendored by a build step (`pnpm anchors`), so a
 * `V7` takes the reader to where V7 is stated.
 *
 * **Wire first, fix later (Q5).** This panel *is* that promise: no gesture is refused for a
 * semantic reason anywhere in the editor, and what is left to fix lands here. So the rows are
 * navigable, the fixes are offered rather than applied, and nothing is hidden — the filter is the
 * reader's, and the panel says how many rows it is holding back.
 */
import type { JSX } from 'react';

import type { Problem, ProblemSeverity, ProblemSource } from '@tensorspine/lang/api';
import { PROBLEM_SEVERITIES } from '@tensorspine/lang/api';
import type { Path, ReferenceIndex } from '@tensorspine/store';

import { useDocuments, useDocumentsStore } from '../documents/context.js';
import { useCurrentDocument } from '../documents/Pills.js';
import type { OpenDocument } from '../documents/store.js';
import { rowsOf } from '../documents/pipeline.js';
import { outlineOf, type OutlineRow } from '../explorer/outline.js';
import { presentation } from '../presentation/index.js';
import { useShell, useShellStore } from '../shell/context.js';
import { specificationLink } from '../shell/store.js';
import { text, textWith } from '../shell/strings.js';
import { bannerOf, type ProblemsBanner } from './banner.js';
import { fixesFor } from './fixes.js';
import { slotSites, type AbsentSlotSite } from './notices.js';
import {
  aboutOf,
  FLAT,
  foldKey,
  problemsView,
  type Grouping,
  type ProblemRow,
  type ProblemsView,
} from './rows.js';

/**
 * The glyph each severity is drawn with — S10's legend, and its `.sev` classes.
 *
 * Keyed by the core's own severities, so a severity added there fails to compile here rather than
 * rendering as nothing; the glyph is never the only thing that says what a row is, the source
 * column and the row's own title saying it in words.
 */
const MARK: Readonly<Record<ProblemSeverity, { glyph: string; className: string; label: string }>> = {
  error: { glyph: '●', className: 'e', label: 'error' },
  warning: { glyph: '▲', className: 'w', label: 'warning' },
  notice: { glyph: '○', className: 'n', label: 'notice' },
};

/** What the grouping control offers, and what each says. */
const GROUPINGS: readonly { id: Grouping; label: string }[] = [
  { id: 'node', label: 'Group by node' },
  { id: 'unit', label: 'Group by unit' },
  { id: FLAT, label: 'No grouping' },
];

/**
 * The outline of the current document, every row of it, rebuilt when the document moves.
 *
 * Memoised **across components** and not only across renders, which is a measurement and not a
 * habit: the outline of `deepseek-v4-pro` is 334 rows in 5.7 ms (feature 2.7), the tab strip draws
 * a count for each of the four panels and the panel draws the rows, and a `useMemo` per component
 * would build it five times per keystroke — five times sixteen milliseconds' worth of budget
 * (§5.6) for one answer. It is a *cache* and not state: the tree is immutable and the revision
 * counts its states (2.1), so the key names exactly one answer, the way feature 2.1 memoises the
 * reference tags per registry and role.
 */
let lastOutline: { key: string; rows: readonly OutlineRow[] } | null = null;

function outlineFor(one: OpenDocument | undefined): readonly OutlineRow[] {
  if (one === undefined) return [];
  const key = `${one.id}@${String(one.session.store.revision)}`;
  if (lastOutline !== null && lastOutline.key === key) return lastOutline.rows;
  const rows = outlineOf({
    tree: one.session.store.tree,
    shapes: one.session.store.shapes,
    bindings: presentation(),
    role: one.session.store.role,
    openAll: true,
  });
  lastOutline = { key, rows };
  return rows;
}

/**
 * Every row the panel holds, whatever produced it.
 *
 * Five places answer rows about one document — the page's synchronous Ajv, the core's staged
 * validation, a derivation that refused, a lint run the reader asked for, and the editor's own
 * notices — and two more are the *workspace's*: the library the workspace gathered, and the
 * schemas it carries. A row can come from two of them at once (the loader's refusals are the
 * workspace's, and the verdict carries the ones the document resolved against), so the union is
 * taken by the fold's own key: a duplicate goes rather than becoming a `× 2` that would say the
 * core answered twice.
 */
function useRows(): { rows: { problem: Problem; stale: boolean }[]; banner: ProblemsBanner | null } {
  const one = useCurrentDocument();
  const workspace = useDocuments((state) => state.notices);
  const library = useDocuments((state) => state.library.problems);
  const rows: { problem: Problem; stale: boolean }[] = one === undefined ? [] : rowsOf(one.reading);
  const seen = new Set(rows.map((row) => foldKey(row.problem)));
  for (const problem of [...library, ...workspace]) {
    const key = foldKey(problem);
    if (seen.has(key)) continue;
    seen.add(key);
    rows.push({ problem, stale: false });
  }
  return { rows, banner: one === undefined ? null : bannerOf(one.reading.verdict) };
}

/** The panel's own controls, shown in the tab strip beside the region's move button (S10). */
export function ProblemControls(): JSX.Element | null {
  const store = useShellStore();
  const view = useShell((state) => state.problems);
  const summary = useCounted();
  if (summary === null) return null;

  const toggle = (severity: ProblemSeverity): void => {
    const held = view.severities.length === 0 ? [...PROBLEM_SEVERITIES] : [...view.severities];
    const next = held.includes(severity)
      ? held.filter((one) => one !== severity)
      : [...PROBLEM_SEVERITIES].filter((one) => held.includes(one) || one === severity);
    store.getState().setProblemView({
      severities: next.length === PROBLEM_SEVERITIES.length ? [] : next,
    });
  };

  return (
    <>
      <span className="seg" role="group" aria-label={text('Severities')}>
        {PROBLEM_SEVERITIES.map((severity) => {
          const on = view.severities.length === 0 || view.severities.includes(severity);
          return (
            <button
              key={severity}
              type="button"
              className={on ? 'on' : ''}
              data-severity={severity}
              aria-pressed={on}
              title={textWith('Show {}', text(MARK[severity].label))}
              onClick={() => {
                toggle(severity);
              }}
            >
              <i aria-hidden="true" className={`sev ${MARK[severity].className}`}>
                {MARK[severity].glyph}
              </i>
              <span className="count">{summary.counts[severity]}</span>
              <span className="offscreen">{text(MARK[severity].label)}</span>
            </button>
          );
        })}
      </span>
      <span className="ctl sel">
        <select
          aria-label={text('Sources')}
          data-control="sources"
          value={view.sources[0] ?? ''}
          onChange={(event) => {
            const chosen = event.target.value;
            store.getState().setProblemView({
              sources: chosen === '' ? [] : [chosen as ProblemSource],
            });
          }}
        >
          <option value="">{text('All sources')}</option>
          {summary.sources.map((source) => (
            <option key={source} value={source}>
              {source}
            </option>
          ))}
        </select>
      </span>
      <span className="ctl sel">
        <select
          aria-label={text('Grouping')}
          data-control="grouping"
          value={view.group}
          onChange={(event) => {
            store.getState().setProblemView({ group: event.target.value as Grouping });
          }}
        >
          {GROUPINGS.map((one) => (
            <option key={one.id} value={one.id}>
              {text(one.label)}
            </option>
          ))}
        </select>
      </span>
      <span className="ctl">
        <input
          type="search"
          aria-label={text('Filter problems')}
          placeholder={text('filter…')}
          data-control="filter"
          value={view.text}
          onChange={(event) => {
            store.getState().setProblemView({ text: event.target.value });
          }}
        />
      </span>
    </>
  );
}

/**
 * What the panel holds, counted and filtered — for the tab's chip and the controls.
 *
 * **No grouping and no outline**: the chip and the severity segment are drawn on every render of
 * the tab strip, whatever panel is showing, and what they need is how many rows there are of each
 * weight and which sources are present. Grouping is what costs an outline, and only the body needs
 * it.
 */
function useCounted(): ProblemsView | null {
  const one = useCurrentDocument();
  const { rows } = useRows();
  const view = useShell((state) => state.problems);
  if (rows.length === 0 && one === undefined) return null;
  return problemsView({
    rows,
    group: FLAT,
    filter: { sources: view.sources, severities: view.severities, text: view.text },
  });
}

/** What the body draws: the same rows, grouped, and the places they navigate to. */
function useGrouped(): { view: ProblemsView | null; outline: readonly OutlineRow[] } {
  const one = useCurrentDocument();
  const { rows } = useRows();
  const view = useShell((state) => state.problems);
  const outline = outlineFor(one);
  if (rows.length === 0 && one === undefined) return { view: null, outline };
  const about = one === undefined ? undefined : aboutOf(outline, one.path, one.title);
  return {
    view: problemsView({
      rows,
      ...(about === undefined ? {} : { about }),
      group: view.group,
      filter: { sources: view.sources, severities: view.severities, text: view.text },
    }),
    outline,
  };
}

/** How many rows the panel holds — the tab's own count chip (S10's `.n.bad`). */
export function useProblemCount(): { total: number; errors: number } | null {
  const summary = useCounted();
  if (summary === null) return null;
  return { total: summary.total, errors: summary.counts.error };
}

/** The panel's body: the banner, the groups, the rows. */
export function ProblemsPanel(): JSX.Element {
  const one = useCurrentDocument();
  const store = useDocumentsStore();
  const shell = useShellStore();
  const docsBase = useShell((state) => state.docsBase);
  const { banner } = useRows();
  const { view: summary, outline } = useGrouped();
  // The reading a rebind pill needs, and the notice's own: the slots `describe` answered for this
  // revision, and where the document writes each name (§4.17's fix, feature 2.10).
  const rebind =
    one === undefined
      ? {}
      : { slots: slotSites(one.reading.facts), index: one.session.store.context.index };

  if (summary === null) {
    return (
      <>
        <div className="empty-line">{text('Nothing to check — no document is open.')}</div>
        <p className="empty-sub">
          {text('Open a folder or a model, and the core validates and derives it as you edit.')}
        </p>
      </>
    );
  }

  if (summary.total === 0) {
    return (
      <>
        {banner === null ? null : <Banner banner={banner} stale={staleVerdict(one)} />}
        <div className="empty-line">{text('No problems.')}</div>
        <p className="empty-sub">
          {one === undefined
            ? text('Open a folder or a model, and the core validates and derives it as you edit.')
            : textWith('{} is on the grammar, resolves against its library, and every rule of §6 holds.', one.title)}
        </p>
      </>
    );
  }

  const navigate = (row: ProblemRow): void => {
    if (row.place === undefined) {
      // A row that names no place of this document still names *something*: the file it is about,
      // which is the honest thing to say when there is nowhere to go (feature 2.5's rule for a
      // command that cannot act).
      shell.getState().note(
        textWith('Problems: nothing to open for {}', row.problem.file ?? row.problem.message),
      );
      return;
    }
    store.getState().revealPlace(pathOf(row.place.pointer, outline));
  };

  return (
    <>
      {banner === null ? null : <Banner banner={banner} stale={staleVerdict(one)} />}
      {summary.shown < summary.total ? (
        <p className="empty-sub" data-hidden-rows={summary.total - summary.shown}>
          {textWith('{} rows are hidden by the filter.', String(summary.total - summary.shown))}
        </p>
      ) : null}
      {summary.groups.map((group) => (
        <div key={group.key} className="pgroup">
          {group.key === '' ? null : (
            <div className="pgrp" data-group={group.key} title={group.detail ?? ''}>
              {group.label}
              <span className="n">{group.count}</span>
            </div>
          )}
          {group.rows.map((row) => (
            <Row
              key={row.key}
              row={row}
              outline={outline}
              rebind={rebind}
              docsBase={docsBase}
              onOpen={navigate}
            />
          ))}
        </div>
      ))}
    </>
  );
}

/**
 * Whether the verdict the banner is about was computed for an older revision (§5.4's freshness).
 *
 * The banner reads a *stage* of the verdict, so it is stale exactly when the verdict is — between
 * a keystroke and the debounce expiring, which is three hundred milliseconds of every edit.
 */
function staleVerdict(one: OpenDocument | undefined): boolean {
  return one !== undefined && one.reading.verdict !== null && one.reading.verdictAt !== one.reading.revision;
}

/** The banner over the rows (§4.17, plan §3). */
function Banner({ banner, stale }: { banner: ProblemsBanner; stale: boolean }): JSX.Element {
  return (
    <div
      className={`banner ${banner.tone}${stale ? ' stale' : ''}`}
      data-banner={banner.kind}
      role="status"
    >
      <span className="bi" aria-hidden="true">
        {banner.mark}
      </span>
      <span>
        <b>{text(banner.head)}</b> {text(banner.body)}
        {stale ? <i className="stalebdg">{text('stale')}</i> : null}
      </span>
    </div>
  );
}

/** One row: S10's `.prow`, every column a field of the problem. */
function Row({
  row,
  outline,
  rebind,
  docsBase,
  onOpen,
}: {
  row: ProblemRow;
  outline: readonly OutlineRow[];
  /** What a rebind reads: the described slots of each site, and the document's own names. */
  rebind: { slots?: readonly AbsentSlotSite[]; index?: ReferenceIndex };
  docsBase: string;
  onOpen: (row: ProblemRow) => void;
}): JSX.Element {
  const store = useDocumentsStore();
  const mark = MARK[row.problem.severity];
  const link = specificationLink(row.problem.code, docsBase);
  const fixes = fixesFor(row.problem, { rows: outline, ...rebind });
  return (
    <div
      className={`prow${row.stale ? ' stale' : ''}`}
      data-source={row.problem.source}
      data-severity={row.problem.severity}
      data-code={row.problem.code}
      data-place={row.problem.path}
    >
      <span className={`sev ${mark.className}`} title={text(mark.label)}>
        <span aria-hidden="true">{mark.glyph}</span>
        <span className="offscreen">{text(mark.label)}</span>
      </span>
      <span className="src">{row.problem.source}</span>
      <span className="code">
        {link === null ? (
          row.problem.code === '' ? (
            <span aria-hidden="true">—</span>
          ) : (
            row.problem.code
          )
        ) : (
          <a href={link.href} target="_blank" rel="noreferrer" title={link.title}>
            {row.problem.code}
          </a>
        )}
      </span>
      <button type="button" className="msg" onClick={() => { onOpen(row); }}>
        {row.problem.message}
        {row.count > 1 ? <i className="times">{textWith('× {}', String(row.count))}</i> : null}
        {row.stale ? <i className="stalebdg">{text('stale')}</i> : null}
      </button>
      {fixes.map((fix) => (
        <button
          key={fix.action.kind}
          type="button"
          className="fix"
          data-fix={fix.action.kind}
          onClick={() => {
            // "Every fix is shown before it is applied" (§4.17): the command is *offered*, which
            // puts §4.4's confirmation up with what it would remove, and confirming makes it one
            // entry of the undo log (D13).
            store.getState().offer(fix.make);
          }}
        >
          {fix.action.title}
        </button>
      ))}
      {/* A repair the row **declares** and no provider can make yet (`Problem.fixes`, the member
          feature 1.1 put there for exactly this). It is drawn as what it is — the words of the
          repair, not a button that would do nothing — so a reader learns what is owed instead of
          finding out by clicking. Feature 2.21's `Create primitive … in a base of this model` is
          the first, live when 3.2 and 3.3 land. */}
      {(row.problem.fixes ?? [])
        .filter((action) => !fixes.some((fix) => fix.action.kind === action.kind))
        .map((action) => (
          <span key={action.kind} className="fix awaits" data-fix-awaited={action.kind}>
            {action.title}
          </span>
        ))}
      <span className="loc" title={row.problem.path === '' ? (row.problem.file ?? '') : row.problem.path}>
        {locationOf(row)}
      </span>
      {(row.problem.detail ?? []).map((line) => (
        <span className="pdetail" key={`${line.path}:${line.message}`}>
          {line.message}
        </span>
      ))}
    </div>
  );
}

/** What the location column shows: the place, else the file, else the derived thing it names. */
function locationOf(row: ProblemRow): string {
  if (row.problem.path !== '') return trimmed(row.problem.path);
  if (row.problem.node !== undefined) return row.problem.node;
  return trimmed(row.problem.file ?? '');
}

/** How many segments of a place the column shows before it elides — S10's own `…/`. */
const SEGMENTS = 4;

/** The last segments of a path, with `…/` where more were dropped (S10). */
function trimmed(path: string): string {
  const parts = path.split('/').filter((one) => one !== '');
  if (parts.length <= SEGMENTS) return parts.join('/');
  return `…/${parts.slice(-SEGMENTS).join('/')}`;
}

/** The place of a pointer, as the outline wrote it — the store takes a path, not a pointer. */
function pathOf(pointer: string, outline: readonly OutlineRow[]): Path {
  const row = outline.find((one) => one.pointer === pointer);
  return row?.path ?? [];
}
