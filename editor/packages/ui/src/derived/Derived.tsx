/**
 * The Derived panel — plan §4.18, artboard S11, component inventory §5's product table.
 *
 * > The products, live. A tab per product (D1–D6, titled with the products' names from the
 * > specification), each rendered generically from the derived schema […]. A **selection filter**
 * > restricts every tab to the selected node, identity or split; the D6 splits tab has "Show split
 * > on canvas". The header shows freshness (`fresh` / `stale since <edit>` / `derivation failed:
 * > …`) and the assignment used. **Export Derived Document…** writes the JSON, validated by the
 * > core against the derived schema before writing, as the tools do.
 *
 * Nothing here computes a figure, and nothing here knows a product by name: `products.ts` finds
 * the six by walking the derived document against its schema, `cells.ts` writes each value under
 * the binding at its place, and `naming.ts` decides which rows a subject keeps. What is left for
 * a component is the drawing and the gestures — which tab is showing, what a link does, what the
 * header says.
 *
 * **A link is a navigation and never an edit** (§4.18: "a D1 node id selects the folded node and
 * the index; a value `node.port` the edge"), so every one of them goes through the selection every
 * projection of the document already reads (`selectPlace`, feature 2.7) and the scrubber of §4.8.
 * Where a name resolves to nothing the panel says so in the Log rather than doing nothing, which
 * is feature 2.5's rule for a gesture that cannot act.
 */
import { useMemo, type JSX } from 'react';

import {
  declaredSite,
  identityInstanceOf,
  foldedGraph,
  identityReadings,
  splitMember,
  toPython,
  type FoldedGraph,
  type IdentityReading,
} from '@tensorspine/lang';
import { pathOfPointer, pointerOf, type Path } from '@tensorspine/store';

import { useDocuments, useDocumentsStore } from '../documents/context.js';
import { useCurrentDocument } from '../documents/Pills.js';
import type { OpenDocument } from '../documents/store.js';
import { formsFor, type FormContext } from '../forms/index.js';
import { presentation } from '../presentation/index.js';
import { useShell, useShellStore } from '../shell/context.js';
import { text, textWith } from '../shell/strings.js';
import type { DerivedCell } from './cells.js';
import { derivedPathOf } from './export.js';
import { freshnessOf, type DerivedFreshness } from './freshness.js';
import { keptBy, type DerivedSubject } from './naming.js';
import {
  derivedReading,
  LIST,
  ROW_LIMIT,
  sectionsOf,
  tableRows,
  TABLE,
  TOTALS,
  type DerivedProduct,
  type DerivedSection,
} from './products.js';

/** The panel's body. */
export function DerivedPanel(): JSX.Element {
  const one = useCurrentDocument();
  const registry = useDocuments((state) => state.registry);
  const view = useShell((state) => state.derived);
  const context = useMemo(
    () => (registry === null ? null : formsFor(registry, presentation())),
    [registry],
  );
  const derived = one?.reading.derived ?? null;
  const reading = useMemo(
    () => (derived === null || context === null ? null : derivedReading(derived, context)),
    [derived, context],
  );
  const products = reading?.products ?? [];
  const showing = products.find((one2) => one2.member === view.product) ?? products[0] ?? null;
  const sections = useMemo(
    () => (showing === null || context === null ? [] : sectionsOf(showing, context)),
    [showing, context],
  );
  const subject = useSubject(one, view.split);
  const freshness =
    one === undefined ? null : freshnessOf(one.reading, one.session.store.undoLabel);

  if (one === undefined) {
    return (
      <>
        <div className="empty-line">{text('Nothing to show — no document is open.')}</div>
        <p className="empty-sub">
          {text('Open a folder or a model, and the core validates and derives it as you edit.')}
        </p>
      </>
    );
  }

  return (
    <div className="derived" data-derived={one.path}>
      <Rail products={products} showing={showing} subject={subject} one={one} />
      <div className="der-body">
        <Header
          one={one}
          product={showing}
          freshness={freshness}
          header={reading?.header ?? []}
        />
        {subject === null || subject.kind !== SUBJECT.split ? null : (
          <div className="banner info" data-split={subject.name}>
            <span className="bi" aria-hidden="true">
              ▣
            </span>
            <span>
              {textWith(
                'Every tab is held to {}: its block and the values crossing it are the rows below.',
                subject.name,
              )}{' '}
              {text('Its block is shaded on the expanded graph.')}
            </span>
          </div>
        )}
        {showing === null || context === null ? (
          <Nothing one={one} freshness={freshness} />
        ) : (
          sections.map((section) => (
            <Section
              key={`${showing.member}/${section.name}/${section.kind}`}
              section={section}
              context={context}
              subject={subject}
              stale={freshness?.stale === true}
              one={one}
            />
          ))
        )}
      </div>
    </div>
  );
}

/** What the panel says when it has no products to show, in the core's own words. */
function Nothing({
  one,
  freshness,
}: {
  one: OpenDocument;
  freshness: DerivedFreshness | null;
}): JSX.Element {
  if (freshness?.failure != null) {
    return (
      <>
        <div className="empty-line">{text('The derivation refused this document.')}</div>
        <p className="empty-sub">{freshness.failure.message}</p>
      </>
    );
  }
  return (
    <>
      <div className="empty-line">
        {freshness?.running === true ? text('Deriving…') : text('Nothing has been derived yet.')}
      </div>
      <p className="empty-sub">
        {freshness?.skipped === true
          ? textWith('{} has problems, so the products were not computed.', one.title)
          : textWith('The core derives {} once its validation passes.', one.title)}
      </p>
    </>
  );
}

/** The rail: the products, the filter chip, and the export. */
function Rail({
  products,
  showing,
  subject,
  one,
}: {
  products: readonly DerivedProduct[];
  showing: DerivedProduct | null;
  subject: DerivedSubject | null;
  one: OpenDocument;
}): JSX.Element {
  const store = useShellStore();
  const documents = useDocumentsStore();
  const filter = useShell((state) => state.derived.filter);
  return (
    <nav className="der-rail" aria-label={text('Derived products')}>
      <div className="grp">{text('products')}</div>
      {products.map((product) => (
        <button
          key={product.member}
          type="button"
          className={`row${product.member === showing?.member ? ' sel' : ''}`}
          data-product={product.member}
          title={product.description ?? product.label}
          aria-current={product.member === showing?.member}
          onClick={() => {
            store.getState().setDerivedView({ product: product.member });
          }}
        >
          <span className="pn">{product.label}</span>
          <span className="n">{product.short}</span>
        </button>
      ))}
      <div className="grp">{text('filter')}</div>
      {subject === null ? (
        <p className="quiet">
          {filter
            ? text('Select a node, an identity or a split to hold every tab to it.')
            : text('The filter is off; every tab shows every row.')}
        </p>
      ) : (
        <p className="chips">
          <span className="badge own" data-subject={subject.kind}>
            {subject.label}
            <button
              type="button"
              className="tbtn"
              aria-label={text('Clear the filter')}
              title={text('Clear the filter')}
              onClick={() => {
                store.getState().setDerivedView({ filter: false, split: null });
              }}
            >
              ×
            </button>
          </span>
        </p>
      )}
      {subject === null && !filter ? (
        <p className="chips">
          <button
            type="button"
            className="link"
            data-filter="on"
            onClick={() => {
              store.getState().setDerivedView({ filter: true });
            }}
          >
            {text('Hold to the selection')}
          </button>
        </p>
      ) : null}
      <div className="grp">{text('export')}</div>
      <p className="chips">
        <button
          type="button"
          className="link"
          data-export="derived"
          title={textWith('Writes {}, validated against the derived schema first.', derivedPathOf(one.path))}
          onClick={() => {
            void documents.getState().exportDerived(one.id);
          }}
        >
          {text('Export Derived Document…')}
        </button>
      </p>
    </nav>
  );
}

/** The header: the product's name, the schema's own description of it, and the freshness pill. */
function Header({
  one,
  product,
  freshness,
  header,
}: {
  one: OpenDocument;
  product: DerivedProduct | null;
  freshness: DerivedFreshness | null;
  header: readonly { readonly label: string; readonly cell: DerivedCell }[];
}): JSX.Element {
  const pill = pillOf(freshness);
  return (
    <div className="der-head">
      <h2>{product?.label ?? one.title}</h2>
      {product?.description == null ? null : (
        <span className="note-line">{product.description}</span>
      )}
      <span className={`pill ${pill.tone}`} data-freshness={pill.tone}>
        <i />
        {pill.text}
      </span>
      {header.map((field) => (
        <span key={field.label} className="der-env" data-header={field.label}>
          {`${text(field.label)}: ${field.cell.text}`}
        </span>
      ))}
    </div>
  );
}

/** What the freshness pill says — §4.18's three states, in the words it writes them. */
function pillOf(freshness: DerivedFreshness | null): { text: string; tone: string } {
  if (freshness === null) return { text: text('not derived'), tone: 'stale' };
  if (freshness.failure !== null) {
    return { text: `${text('derivation failed')}: ${freshness.failure.message}`, tone: 'bad' };
  }
  if (freshness.stale) {
    return {
      text:
        freshness.since === null
          ? text('stale')
          : `${text('stale since')} ${freshness.since}`,
      tone: 'stale',
    };
  }
  if (!freshness.derived) {
    return { text: freshness.running ? text('deriving…') : text('not derived'), tone: 'stale' };
  }
  return { text: text('fresh'), tone: 'der' };
}

/** One section: a strip of totals, a table, or a list. */
function Section({
  section,
  context,
  subject,
  stale,
  one,
}: {
  section: DerivedSection;
  context: FormContext;
  subject: DerivedSubject | null;
  stale: boolean;
  one: OpenDocument;
}): JSX.Element | null {
  const store = useDocumentsStore();
  const shell = useShellStore();
  const onFollow = (named: { kind: string; name: string }): void => {
    follow(named, one, store.getState(), shell.getState());
  };
  const rows =
    section.kind === TOTALS
      ? null
      : tableRows(section, context, (entry) => keptBy(entry.named, subject), ROW_LIMIT);
  if (section.kind === TOTALS) {
    return (
      <div className={`totals${stale ? ' stale' : ''}`} data-section={section.name}>
        {section.name === '' ? null : <div className="tname">{section.name}</div>}
        {section.fields.map((field) => (
          <div key={field.name} className="q" data-field={field.name}>
            <span>{field.name}</span>
            <b title={field.cell.exact ?? ''}>{field.cell.text}</b>
            {field.cell.status === undefined ? null : (
              <i className="tstat">{field.cell.status}</i>
            )}
          </div>
        ))}
      </div>
    );
  }
  if (rows === null) return null;
  const columns = section.kind === TABLE ? section.columns : [{ name: section.name }];
  return (
    <div className={`der-sect${stale ? ' stale' : ''}`} data-section={section.name}>
      {/* The design's `.ih` carries the look; the *level* is the document's — an `h2` for the
          product under the tab's own `h1`, an `h3` for a section under it. Feature 2.7 made the
          same correction to the sheet, and 2.6 to its dialogs: axe's `heading-order` is a
          serious violation and a level skipped is a reader lost. */}
      <h3 className="ih">
        {section.name}
        <span className="ihn">{rowCount(rows.kept, rows.total)}</span>
      </h3>
      <div className="ttable">
        <table className="t">
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column.name} scope="col">
                  {column.name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.rows.map((row, index) => (
              <tr key={row.key ?? String(index)} data-row={row.key ?? String(index)}>
                {row.cells.map((cell, at) => (
                  <Cell key={columns[at]?.name ?? String(at)} cell={cell} onFollow={onFollow} />
                ))}
              </tr>
            ))}
            {rows.kept > rows.rows.length ? (
              <tr className="more">
                <td className="dim" colSpan={columns.length}>
                  {textWith('{} more', String(rows.kept - rows.rows.length))}
                </td>
              </tr>
            ) : null}
            {rows.kept === 0 ? (
              <tr className="more">
                <td className="dim" colSpan={columns.length}>
                  {section.kind === LIST || subject === null
                    ? text('no rows')
                    : textWith('no rows name {}', subject.label)}
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/** What a section's own count says: how many rows it kept, of how many there are. */
function rowCount(kept: number, total: number): string {
  return kept === total
    ? String(total)
    : `${String(kept)} ${text('of')} ${String(total)}`;
}

/**
 * One cell: a figure, a chip, a link, or the text the product wrote.
 *
 * It takes the handler rather than reaching for the two stores itself: a table draws up to two
 * hundred rows of fourteen columns, and a context read per cell is twenty-eight hundred of them
 * per redraw.
 */
function Cell({
  cell,
  onFollow,
}: {
  cell: DerivedCell;
  onFollow: (named: { kind: string; name: string }) => void;
}): JSX.Element {
  const className = cell.figure === true ? 'num' : cell.names === undefined ? 'm' : 'id';
  if (cell.names === undefined || cell.name === undefined) {
    return (
      <td className={className} {...(cell.exact === undefined ? {} : { title: cell.exact })}>
        {cell.text}
        {cell.status === undefined ? null : <i className="tstat">{cell.status}</i>}
      </td>
    );
  }
  const named = { kind: cell.names, name: cell.name };
  return (
    <td className={className}>
      <button
        type="button"
        className="link"
        data-names={cell.names}
        data-name={cell.name}
        title={text(FOLLOWS[cell.names] ?? 'Select what this names')}
        onClick={() => {
          onFollow(named);
        }}
      >
        {cell.text}
      </button>
    </td>
  );
}

/** What clicking an identifier does — §4.18's own sentence, one kind at a time. */
function follow(
  named: { kind: string; name: string },
  one: OpenDocument,
  documents: ReturnType<ReturnType<typeof useDocumentsStore>['getState']>,
  shell: ReturnType<ReturnType<typeof useShellStore>['getState']>,
): void {
  const graph = foldedGraph(one.session.store.tree);
  // The node identifier is taken apart by the core and the navigation is the documents store's
  // (`selectNode`), so the expanded graph's rows (feature 2.16) and these links land in one place.
  const selectNode = (identifier: string): boolean => documents.selectNode(identifier, one.id);
  if (named.kind === FOLLOW.node) {
    if (!selectNode(named.name)) shell.note(textWith('Derived: {} is on no box', named.name));
    return;
  }
  if (named.kind === FOLLOW.reference) {
    const edge = graph.edges.find(
      (one2) => one2.from?.value === named.name || one2.to?.value === named.name,
    );
    if (edge !== undefined) {
      documents.selectPlace(pathOfPointer(edge.pointer), one.id);
      return;
    }
    const site = splitMember(named.name)?.site;
    if (site === undefined || !selectNode(site)) {
      shell.note(textWith('Derived: {} is on no edge and no box', named.name));
    }
    return;
  }
  if (named.kind === FOLLOW.identity) {
    const readings: readonly IdentityReading[] = identityReadings(
      toPython(one.session.store.tree),
    );
    const found = readings.find((reading) => identityInstanceOf(named.name, reading.identity));
    if (found === undefined) {
      shell.note(textWith('Derived: no binding rule declares {}', named.name));
      return;
    }
    documents.selectPlace(pathOfPointer(found.pointer), one.id);
    return;
  }
  // A split names nothing a document declares, so §4.18's "Show split on canvas" is two things at
  // once: every tab held to it — the block and the crossing values are then the rows the reader is
  // looking at — and its **block shaded on the expanded graph**, which feature 2.16 draws. Both
  // happen, which is the whole of §4.18's sentence.
  shell.setDerivedView({ split: { tab: one.id, name: named.name }, filter: true });
  documents.openExpanded(one.id);
  shell.note(
    textWith('Derived: every tab is held to {}, and its block is shaded on the expanded graph.', named.name),
  );
}

/**
 * What a link of each kind does, said in the words §4.18 uses for it.
 *
 * `Show split on canvas` is §4.18's own name for the gesture on the D6 splits tab, and this is
 * where it is: a split names nothing a document declares, so what the control can do is hold every
 * tab to it — which is the half of the sentence the panel owns. Bare property keys, as every
 * other table of the editor's own vocabulary is written.
 */
const FOLLOWS: Readonly<Record<string, string>> = {
  node: 'Select this node, and its index',
  reference: 'Select the edge that carries this value',
  identity: 'Open the binding rule that declares this identity',
  split: 'Show split on canvas',
};

/**
 * The kinds of name a link follows.
 *
 * Bare property keys, as every other table of the editor's own vocabulary is written: `node` and
 * `reference` are words of `presentation.json`'s `names` enumeration, and a whole-literal scan
 * (§1 b) cannot tell one of them from a value of the four schemas.
 */
const FOLLOW = { node: 'node', reference: 'reference', identity: 'identity' } as const;

/**
 * What every tab is held to: the split a reader chose, failing that the document's own selection.
 *
 * A **place** is what the editor selects (D1: "the canvas, the sheets, the explorer … are
 * projections of it"), and a place denotes a family of D1 nodes — which is why the subject is the
 * declared site and not one iteration (see `naming.ts`). An identity is the selection of a binding
 * rule, which is the place feature 2.12 reads an identity at.
 */
function useSubject(
  one: OpenDocument | undefined,
  split: { readonly tab: string; readonly name: string } | null,
): DerivedSubject | null {
  const filter = useShell((state) => state.derived.filter);
  const chosen = split !== null && one !== undefined && split.tab === one.id ? split.name : null;
  const selection = one?.selection;
  const tree = one?.session.store.tree;
  const revision = one?.session.store.revision ?? 0;
  return useMemo(() => {
    if (chosen !== null) return { kind: SUBJECT.split, name: chosen, label: chosen };
    if (!filter || one === undefined || selection === undefined || tree === undefined) return null;
    return subjectOf(one, selection);
    // The subject is a reading of the *tree*, which `one` carries: the revision is what says the
    // tree moved, and the selection's pointer what says the place did.
  }, [chosen, filter, one, revision, selection]);
}

/** The three kinds of subject §4.18 names. */
const SUBJECT = { node: 'node', identity: 'identity', split: 'split' } as const;

/** The subject a selected place stands for, or `null` where it stands for none. */
function subjectOf(one: OpenDocument, selection: Path): DerivedSubject | null {
  const pointer = pointerOf(selection);
  const graph: FoldedGraph = foldedGraph(one.session.store.tree);
  const box = graph.byPointer.get(pointer);
  if (box !== undefined) {
    // A site stands for its own declaration; a composition, which has ranges and no node of its
    // own, stands for every site under it.
    if (box.where !== null) {
      return { kind: SUBJECT.node, name: declaredSite(box.where), label: box.name };
    }
    if (box.ranges.length > 0) return { kind: SUBJECT.node, name: box.name, label: box.name };
    return null;
  }
  const reading = identityReadings(toPython(one.session.store.tree)).find(
    (each) => each.pointer === pointer,
  );
  if (reading === undefined) return null;
  return { kind: SUBJECT.identity, name: reading.identity, label: reading.identity };
}

