/**
 * The drill-in tab — plan §4.8, artboards S4 and S5.
 *
 * Three things around the canvas, and the canvas itself is §4.7's (feature 2.9's `Canvas`, with
 * the drill-in's model): the **index strip** above it, the **scrubber** at the strip's right, and
 * the **alternation strip** below it.
 *
 * **The alternation strip is not a panel tab.** The design pass proposed a fourth tab of the
 * bottom panel for it and the inventory's §6 rejected the proposal: "the panel has three tabs; the
 * strip sits below the canvas in the drill-in". So it is a region of this tab, with its own
 * heading, and the Problems / Derived / Log panel is untouched.
 *
 * **The scrubber is one control per index** (§4.8), over the values `indexGrid` resolved — a real
 * `<input type="range">`, so it is operable from the keyboard by construction, with the point it
 * stands on named beside it and an `Unset` beside that. Unset is not a value: it is the state
 * §4.8 describes as "the canvas shows the representative iteration: every site, every scoped
 * edge", and it is where a drill-in opens.
 */
import { Fragment, useMemo, type JSX } from 'react';

import { foldedGraph, pointLabel, type FoldedGraph, type IndexBinding } from '@tensorspine/lang';
import { pathOfPointer, type Path } from '@tensorspine/store';

import { useDocuments, useDocumentsStore } from '../documents/context.js';
import { documentTab, drillOf, type OpenDocument } from '../documents/store.js';
import { DocumentView } from '../documents/views.js';
import { presentation } from '../presentation/index.js';
import { useShellStore, type Tab } from '../shell/index.js';
import { text, textWith } from '../shell/strings.js';

import { Canvas, type DrillContext } from './Canvas.js';
import { drillModel, type AlternationRow, type DrillModel } from './drill.js';
import { DEFAULT_VIEW } from './model.js';

/** The drill-in of one composition of one open document. */
export function Drill({ one, composition }: { one: OpenDocument; composition: string }): JSX.Element {
  const store = useDocumentsStore();
  const registry = useDocuments((state) => state.registry);
  const templates = useDocuments((state) => state.library.templates);
  const bindings = presentation();
  const shapes = one.session.store.shapes;
  const revision = one.session.store.revision;
  const problems = useMemo(
    () => [
      ...(one.reading.structural ?? []),
      ...(one.reading.verdict?.problems ?? []),
      ...one.reading.notices,
    ],
    [one.reading],
  );

  const model: DrillModel | null = useMemo(() => {
    if (registry === null) return null;
    return drillModel({
      tree: one.session.store.tree,
      composition,
      folded: foldedOf(one),
      facts: one.reading.facts,
      derived: one.reading.derived,
      stale: one.reading.derivedAt !== one.reading.revision,
      problems,
      view: DEFAULT_VIEW,
      registry,
      shapes,
      bindings,
      role: one.session.store.role,
      templates,
    });
    // The tree is immutable (2.1), so the revision is what says the reading still holds.
  }, [one.id, revision, composition, registry, one.reading, problems, shapes, bindings, templates]);

  if (model === null) {
    return (
      <div className="canvas">
        <span className="canvas-note">{text('The schemas are not loaded.')}</span>
      </div>
    );
  }
  const scrub = one.scrub[model.pointer] ?? null;
  const context: DrillContext = { composition, model, scrub };
  return (
    <div className="drill">
      <Strip one={one} model={model} scrub={scrub} />
      <Canvas one={one} drill={context} />
      <Alternation
        model={model}
        scrub={scrub}
        onColumn={(label) => {
          store.getState().setScrub(model.pointer, label, one.id);
        }}
        onSite={(pointer) => {
          store.getState().selectPlace(pathOfPointer(pointer), one.id);
        }}
      />
    </div>
  );
}

/** The index strip and the scrubber — §4.8's own first bullet. */
function Strip({
  one,
  model,
  scrub,
}: {
  one: OpenDocument;
  model: DrillModel;
  scrub: string | null;
}): JSX.Element {
  const store = useDocumentsStore();
  const shell = useShellStore();
  const point = model.points.find((each) => each.label === scrub) ?? null;

  /** Select a place of the document and show the sheet that edits it (§4.4's own rule). */
  const open = (at: Path): void => {
    store.getState().selectPlace(at, one.id);
    shell.getState().revealPanel('panel.properties');
  };

  /** Move one index of the scrubber, keeping the others where they are. */
  const move = (name: string, value: bigint): void => {
    const held: IndexBinding[] = (point?.indices ?? model.points[0]?.indices ?? []).map((each) =>
      each.name === name ? { name, value } : each,
    );
    const label = pointLabel(held);
    const found = model.points.find((each) => each.label === label);
    store.getState().setScrub(model.pointer, found?.label ?? null, one.id);
  };

  return (
    <div className="strip" role="group" aria-label={textWith('The indices of {}', model.name)}>
      <span className="lbl">{text('index')}</span>
      {model.ranges.map((range) => (
        <Fragment key={range.name}>
          <button type="button" className="ctl" data-index={range.name} onClick={() => { open(range.at); }}>
            {range.name}
          </button>
          {range.bounds.map((bound) => (
            <Fragment key={bound.name}>
              <span className="lbl">{text(bound.name)}</span>
              <button
                type="button"
                className="ctl"
                data-bound={`${range.name}.${bound.name}`}
                onClick={() => {
                  open(bound.at);
                }}
              >
                {bound.text}
              </button>
            </Fragment>
          ))}
        </Fragment>
      ))}
      {model.families.length === 0 ? null : (
        <>
          <span className="lbl">{text('families')}</span>
          {model.families.map((family) => (
            <em key={family} className="fchip">
              {family}
            </em>
          ))}
        </>
      )}
      <div className="right">
        {model.ranges.map((range) => {
          const held = point?.indices.find((each) => each.name === range.name)?.value;
          const at = typeof held === 'bigint' ? held : null;
          const values = range.values;
          return (
            <div className="scrub" key={range.name}>
              <span className="lbl">{range.name}</span>
              {/* The leftmost position **is** "unset" — §4.8's own state, where the canvas shows
                  the representative iteration — so moving off it is what starts a preview, and a
                  reader who cannot see the track is told which of the two the control is on. */}
              <input
                type="range"
                className="track"
                min={-1}
                max={Math.max(0, values.length - 1)}
                step={1}
                disabled={values.length === 0}
                value={at === null ? -1 : Math.max(0, values.indexOf(at))}
                aria-label={textWith('Preview {}', range.name)}
                aria-valuetext={at === null ? text('unset') : `${range.name} = ${String(at)}`}
                data-scrub={range.name}
                onChange={(event) => {
                  const held = Number(event.currentTarget.value);
                  if (held < 0) {
                    store.getState().setScrub(model.pointer, null, one.id);
                    return;
                  }
                  const value = values[held];
                  if (value !== undefined) move(range.name, value);
                }}
              />
              <span className={scrub === null ? 'val off' : 'val'} data-scrub-value={range.name}>
                {at === null ? text('unset') : `${range.name} = ${String(at)}`}
              </span>
            </div>
          );
        })}
        <button
          type="button"
          className="tbtn"
          data-scrub-unset="true"
          disabled={scrub === null}
          onClick={() => {
            store.getState().setScrub(model.pointer, null, one.id);
          }}
        >
          {text('Unset')}
        </button>
      </div>
    </div>
  );
}

/**
 * The alternation strip — §4.8's last bullet, drawn from the design pass's S5.
 *
 * > below the canvas, one column per iteration and one row per guarded site, filled where the site
 * > exists, the scrubbed column outlined, with the count (`attn 24 of 30`); read from D1, the
 * > core's expansion.
 *
 * The cells are a picture of one answer, so the row says what it means in words — the site, its
 * guard and its count — and the picture is `aria-hidden`. Clicking a column moves the scrubber,
 * which is an accelerator for the control in the strip above and never the only way to it.
 */
function Alternation({
  model,
  scrub,
  onColumn,
  onSite,
}: {
  model: DrillModel;
  scrub: string | null;
  onColumn: (label: string) => void;
  onSite: (pointer: string) => void;
}): JSX.Element {
  const columns = model.points.length;
  return (
    <div className="ribbon-strip" role="region" aria-label={text('Alternation')}>
      <div className="rib-head">
        <b>{text('Alternation')}</b>
        <span className="note-line">
          {model.expanded
            ? textWith('from D1 — one column per iteration of {}', model.name)
            : text('from D1 — nothing has been derived for this document yet')}
        </span>
        <span className="note-line" data-columns={String(columns)}>
          {textWith('{} iterations', String(columns))}
        </span>
        <span className="note-line" data-unguarded={String(model.unguarded)}>
          {textWith('unguarded, in all of them and not listed: {}', String(model.unguarded))}
        </span>
      </div>
      {model.rows.length === 0 ? (
        <p className="note-line" data-no-rows="true">
          {textWith('No site of {} carries a guard.', model.name)}
        </p>
      ) : (
        <div className="ribbon">
          {model.rows.map((row) => (
            <Row key={row.name} row={row} scrub={scrub} model={model} onColumn={onColumn} onSite={onSite} />
          ))}
          <div className="rscale" aria-hidden="true">
            {model.points.map((point, at) => (
              <span key={point.label}>{scaleLabel(at, columns)}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/** One row of the strip: a guarded site, its guard, its cells and its count. */
function Row({
  row,
  scrub,
  model,
  onColumn,
  onSite,
}: {
  row: AlternationRow;
  scrub: string | null;
  model: DrillModel;
  onColumn: (label: string) => void;
  onSite: (pointer: string) => void;
}): JSX.Element {
  return (
    <div className="rrow" data-row={row.name}>
      <button
        type="button"
        className="rname"
        data-site={row.name}
        onClick={() => {
          onSite(row.pointer);
        }}
      >
        {row.name}
      </button>
      <span className="rguard">{row.guard}</span>
      <div
        className="rcells"
        aria-hidden="true"
        onClick={(event) => {
          const cell = (event.target as HTMLElement).closest('[data-at]');
          const at = cell?.getAttribute('data-at');
          const point = at === null || at === undefined ? undefined : model.points[Number(at)];
          if (point !== undefined) onColumn(point.label);
        }}
      >
        {row.cells.map((on, at) => {
          const point = model.points[at];
          const here = point !== undefined && point.label === scrub;
          return (
            <i
              key={point?.label ?? String(at)}
              data-at={String(at)}
              className={[on ? 'on' : '', here ? 'cur' : ''].filter((one) => one !== '').join(' ')}
            />
          );
        })}
      </div>
      <span className="note-line" data-count={row.name}>
        {`${String(row.at)} of ${String(row.of)}`}
      </span>
    </div>
  );
}

/** The scale under the cells: the first, the last and every fifth column (S5's own). */
function scaleLabel(at: number, columns: number): string {
  if (at === columns - 1) return String(at);
  return at % 5 === 0 ? String(at) : '';
}

/**
 * The folded reading of a document, kept per (document, revision).
 *
 * The same arrangement the Problems panel and the sheets use for the outline (feature 2.8): the
 * drill-in and the canvas below it both read one, and building it twice per keystroke is the
 * measured cost feature 2.9 recorded (1.8–6.9 ms).
 */
let lastFolded: { key: string; graph: FoldedGraph } | null = null;

function foldedOf(one: OpenDocument): FoldedGraph {
  const key = `${one.id}@${String(one.session.store.revision)}`;
  if (lastFolded !== null && lastFolded.key === key) return lastFolded.graph;
  const graph = foldedGraph(one.session.store.tree);
  lastFolded = { key, graph };
  return graph;
}

/** The view the shell draws for a drill-in tab. */
export function DrillView({ tab }: { tab: Tab }): JSX.Element {
  const open = useDocuments((state) => state.open);
  const one = open.find((each) => each.id === documentTab(tab.id));
  const composition = drillOf(tab.id);
  if (one === undefined || composition === null) {
    return (
      <div className="canvas">
        <span className="canvas-note">{text('No document')}</span>
      </div>
    );
  }
  return <DocumentView tab={tab} body={(held) => <Drill one={held} composition={composition} />} />;
}
