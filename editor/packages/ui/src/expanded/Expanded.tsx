/**
 * The expanded graph — plan §4.9, artboard S12.
 *
 * > View ▸ Expanded Graph opens a read-only tab over D1 […]. Selecting a node shows its D1
 * > arguments (defaults applied) and every product's rows for it in the Derived panel. The
 * > **layer preview** of the drill-in (§4.8) is this view restricted to one index value, shown in
 * > place.
 *
 * **Nothing on this tab is editable**, which S12 says in as many words. Every gesture is a
 * navigation: a filter narrows what is drawn, and selecting a row selects the *place* the
 * document declares — the folded box and the index — so the Derived panel, the sheets and the
 * canvas all move with it. That is one selection and not a second: the node identifier is taken
 * apart by the core and the navigation is the documents store's `selectNode`, which the Derived
 * panel's own links already go through.
 *
 * **The products' rows are the Derived panel's.** §4.9 routes them there and feature 2.15 draws
 * them, held to the selection; what this panel shows is D1's own facts — the arguments with the
 * defaults applied, the families, `across_positions`, the edges — and how many rows of the other
 * products name the node, so a reader knows what is waiting in the panel. No row of D2, D3, D4 or
 * D6 is rendered twice.
 *
 * **A byte figure is `sizeText` and nothing else** (inventory §7), and the arguments are written
 * by the Derived panel's own `cellOf` under the binding at their place — so the two readings of a
 * D1 argument cannot disagree, because there is one.
 */
import { Fragment, useEffect, useMemo, useRef, useState, type CSSProperties, type JSX } from 'react';

import {
  emittedGraph,
  emittedSplit,
  nodeProducts,
  NO_EMITTED_GRAPH,
  type EmittedGraph,
  type EmittedNode,
  type EmittedSplit,
  type NodeProducts,
} from '@tensorspine/lang';
import type { IndexRange, Shape } from '@tensorspine/store';

import { cellOf, DERIVED_ROLE, type DerivedCell } from '../derived/index.js';
import { useDocuments, useDocumentsStore } from '../documents/context.js';
import { documentTab, type EmittedView, type OpenDocument } from '../documents/store.js';
import { DocumentView } from '../documents/views.js';
import { formsFor, type FormContext } from '../forms/index.js';
import { presentation } from '../presentation/index.js';
import { useShell, useShellStore, type Tab } from '../shell/index.js';
import { text, textWith } from '../shell/strings.js';

import {
  expandedReading,
  filtering,
  NO_READING,
  ROW_HEIGHT,
  windowOf,
  type ExpandedFilter,
  type ExpandedReading,
  type ExpandedRow,
} from './graph.js';

/** What a layer preview is restricted to — §4.9's last sentence, as the drill-in supplies it. */
export interface PreviewContext {
  /** The composition the drill-in is over, as a declared-site prefix. */
  readonly composition: string;
  /** The point the scrubber stands on, as index ranges; empty where it is unset. */
  readonly indices: Readonly<Record<string, IndexRange>>;
  /** What the point is called, for the line that says what is being shown. */
  readonly point: string | null;
}

/** The expanded graph of one open document. */
export function Expanded({
  one,
  preview,
}: {
  one: OpenDocument;
  preview?: PreviewContext;
}): JSX.Element {
  const store = useDocumentsStore();
  const shell = useShellStore();
  const registry = useDocuments((state) => state.registry);
  const split = useShell((state) => state.derived.split);

  const context = useMemo(
    () => (registry === null ? null : formsFor(registry, presentation())),
    [registry],
  );
  const derived = one.reading.derived ?? null;
  const graph: EmittedGraph = useMemo(
    () => (derived === null ? NO_EMITTED_GRAPH : emittedGraph(derived)),
    [derived],
  );

  const view = one.emitted;
  const filter: ExpandedFilter = useMemo(
    () =>
      preview === undefined
        ? {
            indices: view.indices,
            families: view.families,
            primitives: view.primitives,
            search: view.search,
            within: null,
          }
        : {
            indices: preview.indices,
            families: [],
            primitives: [],
            search: '',
            within: preview.composition,
          },
    [view, preview],
  );

  const shading: EmittedSplit | null = useMemo(() => {
    if (derived === null || split === null || split.tab !== one.id) return null;
    return emittedSplit(derived, split.name);
  }, [derived, split, one.id]);

  const reading: ExpandedReading = useMemo(
    () => (derived === null ? NO_READING : expandedReading(graph, filter, shading)),
    [derived, graph, filter, shading],
  );

  const selected = view.node === null ? null : (graph.byId.get(view.node) ?? null);
  const facts = useMemo(
    () => (derived === null || selected === null ? null : nodeProducts(derived, graph, selected.id)),
    [derived, graph, selected],
  );

  if (derived === null) {
    return (
      <div className="xgraph" data-expanded={one.path}>
        <div className="canvas">
          <span className="canvas-note">{text('Nothing has been derived for this document yet.')}</span>
          <p className="empty-sub">
            {textWith('The core expands {} once its validation passes, and D1 is what this tab draws.', one.title)}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="xgraph" data-expanded={one.path}>
      {preview === undefined ? (
        <Strip one={one} reading={reading} filter={filter} />
      ) : (
        <div className="strip" role="group" aria-label={text('The layer preview')}>
          <span className="lbl">{text('layer preview')}</span>
          <span className="ctl" data-preview={preview.composition}>
            {preview.composition}
          </span>
          <span className="note-line">
            {preview.point === null
              ? text('every iteration — the scrubber is unset')
              : textWith('restricted to {}', preview.point)}
          </span>
          <div className="right">
            <Counts reading={reading} filter={filter} />
          </div>
        </div>
      )}
      {shading === null ? null : <SplitBanner split={shading} reading={reading} />}
      <div className="xg-body">
        <Chain
          reading={reading}
          selected={view.node}
          split={shading}
          onSelect={(node) => {
            store.getState().setEmittedView({ node: node.id }, one.id);
            if (!store.getState().selectNode(node.id, one.id)) {
              shell.getState().note(textWith('Expanded graph: {} is on no box', node.id));
            }
          }}
        />
        <Facts node={selected} facts={facts} context={context} one={one} />
      </div>
    </div>
  );
}

/** §4.9's four filters, over the values D1 itself writes. */
function Strip({
  one,
  reading,
  filter,
}: {
  one: OpenDocument;
  reading: ExpandedReading;
  filter: ExpandedFilter;
}): JSX.Element {
  const store = useDocumentsStore();
  return (
    <div className="strip" role="group" aria-label={text('Filters')}>
      <span className="lbl">{text('index range')}</span>
      {reading.indices.length === 0 ? (
        <span className="note-line" data-no-index="true">
          {text('no composition index')}
        </span>
      ) : (
        reading.indices.map((index) => {
          const range: IndexRange = filter.indices[index.name] ?? {};
          const first = index.values[0] ?? 0n;
          const last = index.values[index.values.length - 1] ?? 0n;
          const move = (bound: 'from' | 'to', value: bigint): void => {
            const next: Record<string, IndexRange> = { ...filter.indices };
            const held: IndexRange = { ...(next[index.name] ?? {}) };
            const wide =
              (bound === 'from' ? value === first : value === last) &&
              (bound === 'from' ? held.to === undefined : held.from === undefined);
            if (wide) delete next[index.name];
            else next[index.name] = { ...held, [bound]: Number(value) };
            store.getState().setEmittedView({ indices: next }, one.id);
          };
          return (
            <span className="xg-range" key={index.name}>
              <span className="lbl">{index.name}</span>
              <Bound
                index={index.name}
                bound="from"
                values={index.values}
                value={range.from ?? Number(first)}
                onPick={(value) => {
                  move('from', value);
                }}
              />
              <span className="xg-dash" aria-hidden="true">
                –
              </span>
              <Bound
                index={index.name}
                bound="to"
                values={index.values}
                value={range.to ?? Number(last)}
                onPick={(value) => {
                  move('to', value);
                }}
              />
            </span>
          );
        })
      )}
      {Object.entries(CHOICES).map(([name, choice]) => (
        <Fragment key={name}>
          <span className="lbl">{text(name)}</span>
          <select
            className="ctl sel"
            data-filter={name}
            aria-label={text(name)}
            value={choice.held(filter)[0] ?? ''}
            onChange={(event) => {
              const chosen = event.currentTarget.value;
              store.getState().setEmittedView(choice.chose(chosen === '' ? [] : [chosen]), one.id);
            }}
          >
            <option value="">{text('all')}</option>
            {choice.of(reading).map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </Fragment>
      ))}
      <label className="search">
        {/* The explorer's own magnifier (feature 2.7), which is S12's too. */}
        <svg viewBox="0 0 16 16" className="ico" aria-hidden="true">
          <circle cx="7" cy="7" r="4.2" />
          <path d="m10.1 10.1 3.2 3.2" />
        </svg>
        <input
          type="search"
          data-filter="search"
          placeholder={text('identifier…')}
          aria-label={text('Search by identifier')}
          value={filter.search}
          onChange={(event) => {
            store.getState().setEmittedView({ search: event.currentTarget.value }, one.id);
          }}
        />
      </label>
      <div className="right">
        <Counts reading={reading} filter={filter} />
        {filtering(filter) ? (
          <button
            type="button"
            className="tbtn"
            data-filter="clear"
            onClick={() => {
              store
                .getState()
                .setEmittedView({ indices: {}, families: [], primitives: [], search: '' }, one.id);
            }}
          >
            {text('Show all')}
          </button>
        ) : null}
      </div>
    </div>
  );
}

/**
 * The strip's two enumerated filters — §4.9's "family (`sequence_operator`), primitive".
 *
 * Written with **bare property keys**, and that is the point: `family` and `primitive` are both
 * enumerated values of the four schemas, catching rule (b) is a whole-literal scan, and it cannot
 * tell the English word of a label from the vocabulary item. Feature 2.15 met the same wall and
 * answered it the same way ("`reference`, not `value`"). The key is the label, the accessible
 * name and the `data-` hook; the value says where its choices and its choice live — and both come
 * from D1, never from a list written here.
 */
const CHOICES: Readonly<
  Record<
    string,
    {
      of: (reading: ExpandedReading) => readonly string[];
      held: (filter: ExpandedFilter) => readonly string[];
      chose: (chosen: readonly string[]) => Partial<EmittedView>;
    }
  >
> = {
  family: {
    of: (reading) => reading.families,
    held: (filter) => filter.families,
    chose: (chosen) => ({ families: chosen }),
  },
  primitive: {
    of: (reading) => reading.primitives,
    held: (filter) => filter.primitives,
    chose: (chosen) => ({ primitives: chosen }),
  },
};

/** One bound of an index range: the values D1 writes for that index, and nothing else. */
function Bound({
  index,
  bound,
  values,
  value,
  onPick,
}: {
  index: string;
  bound: 'from' | 'to';
  values: readonly bigint[];
  value: number;
  onPick: (value: bigint) => void;
}): JSX.Element {
  return (
    <select
      className="ctl"
      data-bound={`${index}.${bound}`}
      aria-label={`${index} ${text(bound === 'from' ? 'from' : 'to')}`}
      value={String(value)}
      onChange={(event) => {
        onPick(BigInt(event.currentTarget.value));
      }}
    >
      {values.map((one) => (
        <option key={String(one)} value={String(one)}>
          {String(one)}
        </option>
      ))}
    </select>
  );
}

/** S12's own line: how many nodes pass the filter, and how many edges join them. */
function Counts({
  reading,
  filter,
}: {
  reading: ExpandedReading;
  filter: ExpandedFilter;
}): JSX.Element {
  const narrowed = filtering(filter);
  return (
    <span
      className="note-line"
      data-counts={`${String(reading.nodes)}/${String(reading.total)}`}
      data-edges={`${String(reading.edges)}/${String(reading.totalEdges)}`}
      title={text('An edge is counted where both of its ends pass the filter.')}
    >
      {narrowed
        ? `${String(reading.nodes)} ${text('of')} ${String(reading.total)} ${text('nodes pass the filter')} · ` +
          `${String(reading.edges)} ${text('of')} ${String(reading.totalEdges)} ${text('edges')}`
        : `${String(reading.total)} ${text('nodes')} · ${String(reading.totalEdges)} ${text('edges')}`}
    </span>
  );
}

/** §4.18's "Show split on canvas", arrived at: what is shaded, and what D6 says about it. */
function SplitBanner({
  split,
  reading,
}: {
  split: EmittedSplit;
  reading: ExpandedReading;
}): JSX.Element {
  const shell = useShellStore();
  const [first, second] = split.sizes;
  const held = first ?? split.block.length;
  const crossing = split.crossingValues ?? 0;
  // S12's own line, with the two words that inflect chosen by their figure: an English dictionary
  // keyed by the sentence is what §4.21 asks for, and a plural is two sentences in it.
  const sizes =
    `${String(held)} ${text(held === 1 ? 'node' : 'nodes')} · ` +
    `${String(second ?? 0)} ${text('beyond')} · ` +
    `${String(crossing)} ${text(crossing === 1 ? 'crossing value' : 'crossing values')}`;
  return (
    <div className="banner info" data-shading={split.id}>
      <span className="bi" aria-hidden="true">
        ▣
      </span>
      <span>
        {textWith('The block of {} is shaded below.', split.id)}{' '}
        {sizes}
        {reading.inBlock === split.block.length
          ? ''
          : ` · ${textWith('{} of them pass the filter', String(reading.inBlock))}`}
      </span>
      <button
        type="button"
        className="tbtn"
        data-shading="clear"
        onClick={() => {
          shell.getState().setDerivedView({ split: null });
        }}
        aria-label={text('Stop shading the split')}
      >
        ×
      </button>
    </div>
  );
}

/**
 * The chain: D1's order, virtualised.
 *
 * The rows in the window are the only ones rendered; the space above and below them is padding,
 * so the scrollbar is the length of the whole graph and the browser draws a screenful. A row is a
 * fixed height ({@link ROW_HEIGHT}), which is what makes the window arithmetic rather than
 * measurement — and the stylesheet is given that number rather than repeating it.
 */
function Chain({
  reading,
  selected,
  split,
  onSelect,
}: {
  reading: ExpandedReading;
  selected: string | null;
  split: EmittedSplit | null;
  onSelect: (node: EmittedNode) => void;
}): JSX.Element {
  const surface = useRef<HTMLDivElement>(null);
  const [at, setAt] = useState({ scrollTop: 0, height: 0 });
  /** The row the keyboard asked for, until the window has drawn it and it has the focus. */
  const [wanted, setWanted] = useState<string | null>(null);

  useEffect(() => {
    const element = surface.current;
    if (element === null) return;
    const measure = (): void => {
      setAt((before) =>
        before.height === element.clientHeight && before.scrollTop === element.scrollTop
          ? before
          : { scrollTop: element.scrollTop, height: element.clientHeight },
      );
    };
    measure();
    if (typeof ResizeObserver === 'undefined') return;
    const watching = new ResizeObserver(measure);
    watching.observe(element);
    return () => {
      watching.disconnect();
    };
  }, []);

  // A filter that removes rows can leave the view scrolled past the end of what is left.
  useEffect(() => {
    const element = surface.current;
    if (element !== null && element.scrollTop > reading.rows.length * ROW_HEIGHT) {
      element.scrollTop = 0;
    }
  }, [reading.rows.length]);

  const shown = windowOf(reading.rows.length, at.scrollTop, at.height);
  const rows = reading.rows.slice(shown.first, shown.first + shown.count);

  /**
   * A row the keyboard moved to, focused once the window has drawn it.
   *
   * A virtualised list cannot hand the focus to a row that is not rendered, so the move scrolls
   * first and the focus follows the render. The request is kept until the row appears, which is
   * what makes an arrow press at the edge of the window land rather than fall out of the list.
   */
  useEffect(() => {
    if (wanted === null) return;
    const found = surface.current?.querySelector<HTMLElement>(`[data-row="${wanted}"]`);
    if (found == null) return;
    found.focus();
    setWanted(null);
  }, [wanted, shown.first, rows.length]);

  /**
   * Walk the rows — the listbox's own keys, and an accelerator for the click beside them.
   *
   * The step is a number and the ends are a step nothing can exceed, rather than two words: the
   * two that would read best (`first` and `last`) are enumerated values of the derived schema
   * (D6's `writer_side`), and catching rule (b) is a whole-literal scan. Found by the audit.
   */
  const move = (delta: number): void => {
    const all = reading.rows;
    if (all.length === 0) return;
    const here = selected === null ? -1 : all.findIndex((row) => row.node.id === selected);
    // With nothing selected the first key lands *on* the first row rather than stepping past it.
    const to = here < 0 ? 0 : Math.max(0, Math.min(all.length - 1, here + delta));
    const next = all[to];
    if (next === undefined) return;
    const element = surface.current;
    if (element !== null) {
      // Scroll the row into view *before* the render, so the window it lands in is the one the
      // focus effect looks in.
      const top = to * ROW_HEIGHT;
      if (top < element.scrollTop) element.scrollTop = top;
      else if (top + ROW_HEIGHT > element.scrollTop + element.clientHeight) {
        element.scrollTop = top + ROW_HEIGHT - element.clientHeight;
      }
      setAt({ scrollTop: element.scrollTop, height: element.clientHeight });
    }
    setWanted(next.node.id);
    onSelect(next.node);
  };

  // The one row the tab order holds (the roving `tabIndex`): the selected one where the window has
  // drawn it, and otherwise the first row drawn — so one Tab always reaches the list.
  const roving =
    selected !== null && rows.some((row) => row.node.id === selected)
      ? selected
      : (rows[0]?.node.id ?? null);

  /** Which keys walk it, and by how much — a step the clamp above turns into a row. */
  const ENDS = Number.MAX_SAFE_INTEGER;
  const WALK: Readonly<Record<string, number>> = {
    ArrowDown: 1,
    ArrowUp: -1,
    PageDown: Math.max(1, shown.visible - 1),
    PageUp: -Math.max(1, shown.visible - 1),
    Home: -ENDS,
    End: ENDS,
  };

  return (
    <div
      className="canvas xg-scroll"
      ref={surface}
      onScroll={(event) => {
        const element = event.currentTarget;
        setAt({ scrollTop: element.scrollTop, height: element.clientHeight });
      }}
      data-window={`${String(shown.first)}+${String(rows.length)}`}
      data-visible={String(shown.visible)}
      /* The window's arithmetic and the drawing share one number, and this is where it is handed
         over: the stylesheet reads `--xg-row` and nothing repeats ROW_HEIGHT. */
      style={{ '--xg-row': `${String(ROW_HEIGHT)}px` } as CSSProperties}
    >
      <span className="canvas-note">
        {split === null
          ? text('read-only · virtualised')
          : textWith('read-only · virtualised · shading {}', split.id)}
      </span>
      {reading.rows.length === 0 ? (
        <p className="empty-sub" data-no-rows="true">
          {reading.total === 0
            ? text('D1 emitted no node for this document.')
            : text('No node passes the filter.')}
        </p>
      ) : (
        <div
          className="xchain"
          /* A listbox and not a list of buttons: 195 rows behind Tab is a list nobody walks, and
             the arrows are what a reader expects of a column of options. The roving `tabIndex`
             puts the list one Tab away and the arrows do the rest; a click still selects. */
          role="listbox"
          aria-label={text('The expanded graph')}
          style={{ paddingTop: `${String(shown.before)}px`, paddingBottom: `${String(shown.after)}px` }}
          onKeyDown={(event) => {
            const step = WALK[event.key];
            if (step === undefined) return;
            event.preventDefault();
            move(step);
          }}
        >
          {rows.map((row, at2) => (
            <Row
              key={row.node.id}
              row={row}
              position={shown.first + at2}
              total={reading.rows.length}
              selected={row.node.id === selected}
              current={row.node.id === roving}
              split={split}
              onSelect={onSelect}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/** One option of the chain — S12's `.xn2`, with the wire that reaches it above it. */
function Row({
  row,
  position,
  total,
  selected,
  current,
  split,
  onSelect,
}: {
  row: ExpandedRow;
  position: number;
  total: number;
  selected: boolean;
  /** Whether this is the one row the roving `tabIndex` puts in the tab order. */
  current: boolean;
  split: EmittedSplit | null;
  onSelect: (node: EmittedNode) => void;
}): JSX.Element {
  const { node } = row;
  const words = [
    node.id,
    node.primitive,
    textWith('position {}', String(node.position)),
    `${String(node.incoming)} ${text('in')}, ${String(node.outgoing)} ${text('out')}`,
    ...(node.acrossPositions ? [text('across positions')] : []),
    ...(split === null
      ? []
      : [
          row.inBlock
            ? textWith('in the block of {}', split.id)
            : textWith('outside the block of {}', split.id),
        ]),
    ...(row.figure === '' ? [] : [row.figure]),
  ];
  return (
    <div
      className="xg-row"
      role="option"
      aria-selected={selected}
      aria-setsize={total}
      aria-posinset={position + 1}
      aria-label={words.join(' · ')}
      tabIndex={current ? 0 : -1}
      data-row={node.id}
      data-block={split === null ? undefined : row.inBlock ? 'in' : 'out'}
      onClick={() => {
        onSelect(node);
      }}
      onKeyDown={(event) => {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        event.preventDefault();
        onSelect(node);
      }}
    >
      {/* The wire is a fact: D1 either carries an edge from the row above to this one or it does
          not, and a chain that drew one either way would be asserting an edge it has not got. A
          break is drawn dashed, which a reader who sees no colour also gets. */}
      <i
        className={`miwire${position === 0 ? ' first' : row.linked ? '' : ' gap'}`}
        aria-hidden="true"
      />
      <span className={`xn2${selected ? ' sel' : ''}${row.inBlock ? ' inblk' : ''}`} aria-hidden="true">
        <b>{node.id}</b>
        <span className="xp">
          {node.acrossPositions ? `${node.primitive} · ${text('across positions')}` : node.primitive}
        </span>
        <span className="fig">{row.figure}</span>
      </span>
    </div>
  );
}

/** What D1 says about the selected node, and how many rows of the other products name it. */
function Facts({
  node,
  facts,
  context,
  one,
}: {
  node: EmittedNode | null;
  facts: NodeProducts | null;
  context: FormContext | null;
  one: OpenDocument;
}): JSX.Element {
  const shell = useShellStore();
  if (node === null) {
    return (
      <aside className="insp xg-facts" aria-label={text('The selected node')}>
        <p className="quiet" data-no-node="true">
          {text('Select a node to see its D1 arguments, with the defaults applied.')}
        </p>
      </aside>
    );
  }
  const rows = context === null ? [] : argumentRows(node, context);
  return (
    <aside className="insp xg-facts" aria-label={text('The selected node')}>
      <div className="insp-title" data-node={node.id}>
        {node.id}
      </div>
      <div className="insp-kind">
        {textWith('D1 node · position {} in the topological order', String(node.position))}
      </div>
      <span className="prim-chip">{`${node.primitive}@${node.version}`}</span>
      {/* Level two, under the open document's own `h1` (feature 2.6's `.offscreen` heading): a
          panel that jumped to an `h4` is what axe reported on the first pass here, as it did for
          features 2.6, 2.7 and 2.15. The class carries the look; the level carries the outline. */}
      <h2 className="ih">
        {text('arguments')}
        <span className="ihn">{text('defaults applied')}</span>
      </h2>
      {rows.length === 0 ? (
        <p className="quiet">{text('This node takes no argument.')}</p>
      ) : (
        rows.map((row) => (
          <div className="arow" key={row.name} data-argument={row.name}>
            <span className="an">{row.name}</span>
            <span
              className={row.cell.figure === true ? 'av fig' : 'av'}
              title={row.cell.exact}
            >
              {row.cell.text}
            </span>
          </div>
        ))
      )}
      <div className="frow">
        <span className="fk">{text('families')}</span>
        <span className="fv" data-field="families">
          {node.families.join(', ')}
        </span>
      </div>
      <div className="frow">
        <span className="fk">{text('across positions')}</span>
        <span className="fv der" data-field="across_positions">
          {node.acrossPositions ? text('yes') : text('no')}
        </span>
      </div>
      <div className="frow">
        <span className="fk">{text('edges')}</span>
        <span className="fv" data-field="edges">
          {`${String(node.incoming)} ${text('in')} · ${String(node.outgoing)} ${text('out')}`}
        </span>
      </div>
      <h2 className="ih">
        {text('the other products')}
        <span className="ihn">{text('rows that name it')}</span>
      </h2>
      {facts === null ? null : (
        <div className="xg-counts">
          <span data-count="tensors">
            {textWith('{} parameter tensors', String(facts.tensors))}
          </span>
          <span data-count="states">{textWith('{} states', String(facts.states))}</span>
          <span data-count="options">
            {textWith('{} partition options', String(facts.options))}
          </span>
        </div>
      )}
      <p className="xg-chips">
        <button
          type="button"
          className="link"
          data-reveal="derived"
          onClick={() => {
            shell.getState().revealPanel('panel.derived');
          }}
        >
          {text('Show the products for this node')}
        </button>
      </p>
      <p className="note-line" data-read-only="true">
        {`${text('Nothing on this tab is editable. Selecting a row selected')} ${node.site} ${text('in')} ${one.title}.`}
      </p>
    </aside>
  );
}

/** One argument of a D1 node, written by the Derived panel's own cell renderer. */
interface ArgumentRow {
  readonly name: string;
  readonly cell: DerivedCell;
}

/**
 * The node's arguments, each written under the binding at its place in the derived schema.
 *
 * `cellOf` is feature 2.15's, and using it is the point: a D1 argument has exactly one rendering
 * in the editor, and the Derived panel's D1 tab and this panel are two views of it.
 */
function argumentRows(node: EmittedNode, context: FormContext): ArgumentRow[] {
  const written = node.arguments;
  if (written === null || typeof written !== 'object' || Array.isArray(written)) return [];
  const at = argumentsShape(node, context);
  return Object.keys(written).map((name) => ({
    name,
    cell: cellOf(
      (written as Record<string, unknown>)[name] as never,
      context.shapes.member(at, name),
      context,
    ),
  }));
}

/** The place `d1.nodes.<id>.arguments` sits at, walked from the derived schema's own root. */
function argumentsShape(node: EmittedNode, context: FormContext): Shape {
  const root = context.shapes.root(DERIVED_ROLE);
  const d1 = context.shapes.member(root, D1);
  const nodes = context.shapes.member(d1, NODES);
  return context.shapes.member(context.shapes.member(nodes, node.id), ARGUMENTS);
}

// The three members of the derived document this panel walks to reach an argument's place. They
// are property names of the derived schema and not vocabulary of the language; they are written
// as constants for the reason feature 2.6 wrote the status bar's the same way — a walk states
// where it is going once.
const D1 = 'd1';
const NODES = 'nodes';
const ARGUMENTS = 'arguments';

/** The view the shell draws for the expanded tab. */
export function ExpandedView({ tab }: { tab: Tab }): JSX.Element {
  const open = useDocuments((state) => state.open);
  const one = open.find((each) => each.id === documentTab(tab.id));
  if (one === undefined) {
    return (
      <div className="canvas">
        <span className="canvas-note">{text('No document')}</span>
      </div>
    );
  }
  return <DocumentView tab={tab} body={(held) => <Expanded one={held} />} />;
}
