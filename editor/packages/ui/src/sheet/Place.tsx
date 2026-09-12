/**
 * The sheets of plan §4.11 that are not an instance's, and the tables of §4.16 — drawn.
 *
 * Artboard S15 is the picture: a **Document** region with the model id, the schema tag, the bases
 * and the counts; a **Quantities** table with its used-by and resolved columns and the derivation
 * line under a literal that declares one; a **Constants** table; an **Interfaces** table. Beside
 * them stand the sheets of the declarations those tables hold — a quantity's, a constant's, an
 * input's, an output's — and of the three §4.11 names that no table holds: a composition's, an
 * edge's and an identity's.
 *
 * **All of them are one component.** The rows are {@link placeSheet}'s — the generated form of the
 * place's own `$def` (feature 2.3), drawn by {@link FormRows} — and what this file adds is the
 * sections of *facts*: what the core resolves a quantity to, what names a declaration, what the
 * products say about an identity's instances or about the value an edge carries, what a
 * composition holds. Each section is drawn where the fact exists and nowhere else, so a place
 * with none is its form and its name, which is the honest sheet for it.
 */
import { useEffect, useState, type JSX } from 'react';

import {
  compositionAt,
  foldedGraph,
  locationAxisPlaces,
  pyStr,
  type DerivedStreamRow,
  type DerivedValueRow,
  type FoldedGraph,
  type IdentityFacts,
  type IdentityReading,
  type PyValue,
  type QuantityReading,
} from '@tensorspine/lang';
import { EditError, nodeAt, pointerOf, type Path } from '@tensorspine/store';

import { useDocuments, useDocumentsStore } from '../documents/context.js';
import { sizeText } from '../documents/figures.js';
import type { OpenDocument } from '../documents/store.js';
import { renameCommand } from '../explorer/Explorer.js';
import type { OutlineRow } from '../explorer/outline.js';
import { CHOOSER, editsOneValue, type FormContext, type FormRow } from '../forms/index.js';
import { text, textWith } from '../shell/strings.js';
import { referentNames } from './add.js';
import { locationPreview, type LocationPreview } from './location.js';
import { placeSheet, shapeAt, type PlaceSheet, type PlaceTable, type TableRow } from './places.js';
import { FormRows, ModeSelect, NameRow, ScalarField } from './Rows.js';

/** What the sheet of one place is drawn with. */
export interface PlaceProps {
  readonly one: OpenDocument;
  readonly context: FormContext;
  /** The place selected; the document's root where nothing is. */
  readonly path: Path;
  readonly row: OutlineRow | null;
  readonly rows: ReadonlyMap<string, OutlineRow>;
  /** Everything the core and the products answered about the document, gathered by the caller. */
  readonly facts: PlaceFacts;
  /** What §4.13's pickers offer inside an expression editor at a place. */
  readonly names: (at: Path) => (referent: string) => readonly string[];
}

/** The readings the sheet is built from — computed once per document revision by its caller. */
export interface PlaceFacts {
  readonly quantities: readonly QuantityReading[];
  readonly identities: readonly IdentityReading[];
  readonly values: readonly DerivedValueRow[];
  readonly streams: readonly DerivedStreamRow[];
  readonly folded: FoldedGraph | null;
  readonly derived: PyValue | null;
  readonly problems: readonly { readonly path: string; readonly message: string }[];
  readonly notices: readonly { readonly path?: string }[];
}

/** What nothing at all is written as — `view.py`'s own em dash. */
const EMPTY = '—';

/** The sheet of one place of the document. */
export function PlaceView(props: PlaceProps): JSX.Element {
  const { one, context, path, row, rows, facts } = props;
  const store = useDocumentsStore();
  const sheet = placeSheet({
    tree: one.session.store.tree,
    role: one.session.store.role,
    context,
    path,
    row,
    rows,
    index: one.session.store.context.index,
    ...facts,
  });
  const [name, setName] = useState(sheet.name);
  const revision = one.session.store.revision;
  useEffect(() => {
    setName(sheet.name);
  }, [sheet.name, sheet.pointer, revision]);

  const commit = (): void => {
    const to = name.trim();
    if (to === '' || to === sheet.name || row === null) {
      setName(sheet.name);
      return;
    }
    try {
      store.getState().edit(renameCommand(row, to));
      store.getState().selectPlace([...row.path.slice(0, -1), to]);
    } catch (error) {
      store.getState().setToast({ text: error instanceof EditError ? error.message : String(error) });
      setName(sheet.name);
    }
  };

  // What a member that refers to a declaration offers, read once per tree (feature 2.2's rules).
  const offers = referentNames(context, one.session.store.tree, one.session.store.role);
  // And what §4.14 asks the core for about the identity written here: its members, the dtypes V14
  // admits, the axes a location may name and what its names evaluate to.
  const identity = useIdentityFacts(one, sheet.identity?.reading ?? null);
  // Where the location names an axis of the slot's shape: the core's walk over the **tree** the
  // sheet holds, so the places follow a form the author has just chosen even while the grammar
  // refuses what it wrote (a blank axis is no identifier). The *names* those places offer are the
  // slot's, from the last analysis there was.
  const axisPlaces =
    sheet.identity === null ? [] : locationAxisPlaces(nodeAt(one.session.store.tree, path) ?? null);
  const title = sheet.name === '' ? one.title : sheet.name;
  return (
    <>
      <h2 className="insp-title">{title}</h2>
      <p className="insp-kind">{sheet.declares === '' ? kindOf(sheet) : sheet.declares}</p>
      <h3 className="ih">{text('Identity')}</h3>
      <NameRow
        value={name}
        editable={row !== null && row.named}
        onChange={setName}
        onCommit={commit}
        onRevert={() => {
          setName(sheet.name);
        }}
      />
      <div className="frow">
        <span className="fk">{text('Place')}</span>
        <span className="fv mono" data-place={sheet.pointer === '' ? '/' : sheet.pointer}>
          {sheet.pointer === '' ? '/' : sheet.pointer}
        </span>
      </div>
      {sheet.table !== null ? (
        <Table {...props} table={sheet.table} />
      ) : (
        <FormRows
          one={one}
          base={path}
          form={sheet.form}
          context={context}
          names={props.names(path)}
          // An axis of a location is picked from the slot's own stored shape, at the places the
          // core says name one (`axisPlaces`) — never at a row the interface recognised by name.
          picks={(row) =>
            identity.facts !== null && axisPlaces.includes(row.path)
              ? identity.facts.axes
              : offers(lastStep(row))
          }
          tokens={identity.facts?.tokens ?? []}
        />
      )}
      <Libraries sheet={sheet} />
      <Resolved sheet={sheet} />
      <UsedBy sheet={sheet} />
      <IdentityFactsRows sheet={sheet} identity={identity} />
      <IdentityRows sheet={sheet} />
      <ValueRows sheet={sheet} />
      <Held sheet={sheet} one={props.one} />
    </>
  );
}

/** What the core answered about the identity a place declares, and whether it is still fresh. */
export interface HeldIdentity {
  readonly facts: IdentityFacts | null;
  /** Whether the answer was computed for the revision the document now holds (§5.4's `stale`). */
  readonly fresh: boolean;
}

/**
 * The core's facts for the identity a place declares, asked once per revision.
 *
 * §5.4 puts `describe` on the branch that runs at once and feature 2.10 measured why the
 * compatibility lists are not on it; this is the same arrangement one question along — the sheet
 * that *has an identity open* asks for that identity, off the reading the worker already holds
 * for the revision, and shows the previous answer dimmed until the new one lands.
 */
function useIdentityFacts(one: OpenDocument, reading: IdentityReading | null): HeldIdentity {
  const store = useDocumentsStore();
  const [held, setHeld] = useState<{ rule: string; revision: number; facts: IdentityFacts | null }>({
    rule: '',
    revision: -1,
    facts: null,
  });
  const revision = one.session.store.revision;
  const rule = reading?.rule ?? '';
  const state = reading?.state ?? false;
  useEffect(() => {
    if (rule === '') {
      setHeld({ rule: '', revision, facts: null });
      return;
    }
    let live = true;
    void store
      .getState()
      .identityFacts(rule, state, one.id)
      .then((facts) => {
        if (!live) return;
        // A document *between two keystrokes* is off the grammar — a blank axis is no identifier —
        // and the core answers nothing about it, rightly. What the sheet shows then is the last
        // answer, **dimmed**: §5.4's own arrangement for a figure computed for an older revision,
        // and the axes a picker offers are the slot's, which no edit of the location changes.
        setHeld((before) =>
          facts === null && before.rule === rule && before.facts !== null
            ? { ...before, revision: -1 }
            : { rule, revision, facts },
        );
      });
    return (): void => {
      live = false;
    };
  }, [store, one.id, rule, state, revision]);
  if (held.rule !== rule) return { facts: null, fresh: rule === '' };
  return { facts: held.facts, fresh: held.revision === revision };
}

/** The member a row is: the last step of its place, which is what a reference rule names. */
function lastStep(row: FormRow): string {
  const step = row.steps[row.steps.length - 1];
  return typeof step === 'string' ? step : '';
}

/** What a place with no declaring container is: the document itself, or a rule of the bindings. */
function kindOf(sheet: PlaceSheet): string {
  if (sheet.pointer === '') return text('document');
  if (sheet.identity !== null) return textWith('{} identity', sheet.identity.reading.kind);
  if (sheet.edge !== null) return text('binding rule');
  return text('declaration');
}

/**
 * §4.16's table — S15's Quantities, Constants and Interfaces.
 *
 * The columns are the entry definition's own members in the schema's order, with the name first
 * and the facts the core answers last; a cell edits where it stands wherever the place admits one
 * control (§4.16's "inline editing"), and the name opens the entry's own sheet.
 */
function Table(props: PlaceProps & { table: PlaceTable }): JSX.Element {
  const { table } = props;
  const store = useDocumentsStore();
  const selection = props.one.selection === undefined ? null : pointerOf(props.one.selection);
  return (
    <>
      <h3 className="ih">
        {table.declares === '' ? text('Entries') : table.declares}
        <span className="ihn">{String(table.rows.length)}</span>
      </h3>
      <div className="ttable">
        <table className="t">
          <tbody>
            <tr>
              {table.columns.map((column) => (
                <th key={column} data-column={column}>
                  {column}
                </th>
              ))}
            </tr>
            {table.rows.map((entry) => (
              <Row
                key={entry.pointer}
                {...props}
                table={table}
                entry={entry}
                selected={entry.pointer === selection}
              />
            ))}
          </tbody>
        </table>
      </div>
      {table.rows.length === 0 ? (
        <div className="arow more">{text('Nothing is declared here yet.')}</div>
      ) : null}
      <div className="arow more">
        <button
          type="button"
          className="btn"
          data-add-entry={pointerOf(table.path)}
          onClick={() => {
            store.getState().addDeclaration({ at: table.path });
          }}
        >
          {textWith('Add {}', table.declares === '' ? text('entry') : table.declares)}
        </button>
      </div>
    </>
  );
}

/** One row of a table, with the line S15 draws under a declaration that carries an expression. */
function Row(
  props: PlaceProps & { table: PlaceTable; entry: TableRow; selected: boolean },
): JSX.Element {
  const { entry, table, one, context } = props;
  const store = useDocumentsStore();
  const columns = table.columns;
  return (
    <>
      <tr className={props.selected ? 'on' : undefined} data-row={entry.pointer}>
        <td className="id">
          <button
            type="button"
            className="link mono"
            data-open-row={entry.pointer}
            onClick={() => {
              store.getState().selectPlace(entry.path, one.id);
            }}
          >
            {entry.name}
          </button>
          {entry.unused ? (
            <i className="tstat" data-unused={entry.name} title={text('declared and read nowhere')}>
              {text('unused')}
            </i>
          ) : null}
        </td>
        {columns.slice(1).map((column, index) => (
          <Cell
            key={column}
            {...props}
            column={column}
            cell={entry.cells[index] ?? null}
            entry={entry}
            context={context}
          />
        ))}
      </tr>
      {entry.note === null ? null : (
        <tr className="kid">
          <td className="id" colSpan={columns.length} data-note={entry.pointer}>
            {entry.note}
          </td>
        </tr>
      )}
    </>
  );
}

/** One cell: the control the place admits, or what the value reads as. */
function Cell(
  props: PlaceProps & {
    entry: TableRow;
    column: string;
    cell: { readonly text: string; readonly row: FormRow } | null;
  },
): JSX.Element {
  const { entry, cell, column, one, context } = props;
  if (cell === null) {
    // The two columns the core answers rather than the schema — §4.16's last two.
    if (column === USED_BY) {
      return (
        <td className="num" data-used-by={entry.name}>
          {entry.usedBy === null ? EMPTY : String(entry.usedBy)}
        </td>
      );
    }
    if (column === RESOLVED) {
      return (
        <td className="num" data-resolved={entry.name}>
          {entry.resolved === null || entry.resolved === '' ? EMPTY : entry.resolved}
        </td>
      );
    }
    return <td className="dim">{EMPTY}</td>;
  }
  const at = [...entry.path, ...cell.row.steps] as Path;
  const shape = shapeAt(context.shapes, at, one.session.store.role);
  if (cell.row.widget === CHOOSER) {
    return (
      <td className="m" data-cell={column}>
        <ModeSelect row={cell.row} at={at} shape={shape} context={context} />
        {cell.text.split(' ').slice(1).join(' ')}
      </td>
    );
  }
  if (editsOneValue(cell.row.widget)) {
    return (
      <td className="m" data-cell={column}>
        <ScalarField
          one={one}
          base={entry.path}
          form={{ rows: [], notes: [] }}
          context={context}
          row={cell.row}
          at={at}
          shape={shape}
        />
      </td>
    );
  }
  return (
    <td className="m" data-cell={column}>
      {cell.text === '' ? EMPTY : cell.text}
    </td>
  );
}

/**
 * §4.11's "`primitive_libraries` (repeatable base paths with resolution status)".
 *
 * The paths themselves are rows of the generated form above — a list the author edits — and the
 * **status** is not the document's at all: it is what the loader made of them, which the store
 * holds as the library it gathered. What is shown is what the editor actually knows: how many
 * primitives the library carries, how many files were gathered for it, and the loader's own
 * refusals, each naming its file in the loader's words (the parity contract, feature 1.3).
 *
 * S15 writes `36 primitives · 38 axes · 56 roles` beside the base, and the first of those three is
 * this figure; the axes and the roles are the Library activity's (3.1), which is what keeps a
 * gathered base rather than the two projections a sheet needs.
 */
function Libraries({ sheet }: { sheet: PlaceSheet }): JSX.Element | null {
  const library = useDocuments((state) => state.library);
  if (sheet.pointer !== '') return null;
  return (
    <>
      <h3 className="ih">
        {text('Library')}
        <span className="ihn">
          {library.loading ? text('loading') : textWith('{} files', String(library.files))}
        </span>
      </h3>
      <div className="frow">
        <span className="fk">{text('gathered')}</span>
        <span className="fv mono" data-library="primitives">
          {textWith('{} primitives', String(library.versions.size))}
        </span>
      </div>
      <div className="frow">
        <span className="fk">{text('status')}</span>
        <span className="fv">
          {library.problems.length === 0 ? (
            <i className="tstat" data-library="resolved">
              {text('resolved')}
            </i>
          ) : (
            <i className="tstat" data-library="refused">
              {textWith('{} refused', String(library.problems.length))}
            </i>
          )}
        </span>
      </div>
      {library.problems.map((problem, at) => (
        <div className="rowmsg" key={`${String(at)}`} data-library-problem={problem.code}>
          {problem.message}
        </div>
      ))}
    </>
  );
}

/** §4.16's `resolved`: what the core resolves the quantity declared here to. */
function Resolved({ sheet }: { sheet: PlaceSheet }): JSX.Element | null {
  if (sheet.quantity === null) return null;
  const reading = sheet.quantity;
  return (
    <>
      <h3 className="ih">{text('Resolved value')}</h3>
      <div className="frow">
        <span className="fk">{text('value')}</span>
        <span className="fv mono" data-resolved={reading.name}>
          {reading.value === undefined ? text('does not resolve') : pyStr(reading.value)}
        </span>
      </div>
      <div className="frow">
        <span className="fk">{text('follows from')}</span>
        <span className="fv">
          {reading.computed ? text('the document says how') : text('the document says only what')}
        </span>
      </div>
    </>
  );
}

/** §4.16's `used by`: the occurrences a rename would rewrite, in document order. */
function UsedBy({ sheet }: { sheet: PlaceSheet }): JSX.Element | null {
  const store = useDocumentsStore();
  if (!sheet.referenced) return null;
  return (
    <>
      <h3 className="ih">
        {text('Used by')}
        <span className="ihn">{String(sheet.usedBy.length)}</span>
      </h3>
      {sheet.usedBy.length === 0 ? (
        <div className="arow more" data-unused={sheet.name}>
          {text('Nothing in this document names it.')}
        </div>
      ) : null}
      {sheet.usedBy.map((used) => (
        <div className="frow" key={used.pointer}>
          <span className="fk mono">{used.tag}</span>
          <span className="fv">
            <button
              type="button"
              className="link mono"
              data-used-at={used.pointer}
              onClick={() => {
                store.getState().revealPlace(used.path);
              }}
            >
              {used.pointer}
            </button>
          </span>
        </div>
      ))}
    </>
  );
}

/**
 * What the core says about the identity written here — §4.14's own sections.
 *
 * Three things, each a fact no schema and no component can state: the **members** as the sites
 * they resolve to, the **dtypes** V14 admits for every member's role, and the **evaluated names**
 * the location produces, instance by instance (S8's `.evald` block). The form above edits the
 * document; this says what the document then means.
 *
 * The dtype row is a *statement*, not a limit: the select in the form is the schema's own
 * enumeration, because Q5 forbids the editor to refuse a gesture for a semantic reason and a
 * document may already carry a dtype outside the set. V14's line is what says so, in Problems.
 */
function IdentityFactsRows({
  sheet,
  identity,
}: {
  sheet: PlaceSheet;
  identity: HeldIdentity;
}): JSX.Element | null {
  if (sheet.identity === null) return null;
  const facts = identity.facts;
  if (facts === null) return null;
  const preview = locationPreview(facts);
  return (
    <>
      <h3 className="ih">
        {text('Members')}
        <span className="ihn">{String(facts.members.length)}</span>
      </h3>
      {facts.members.map((member) => (
        <div className="frow" key={`${member.where}.${member.slot}`}>
          <span className="fk mono">{member.slot}</span>
          <span className="fv mono" data-member-site={`${member.where}.${member.slot}`}>
            {member.where}
            {member.present ? '' : ` · ${text('absent under these arguments')}`}
          </span>
        </div>
      ))}
      {facts.dtypes.length === 0 ? null : (
        <>
          <h3 className="ih">{text('Precision')}</h3>
          <div className="frow">
            <span className="fk">{text('roles admit')}</span>
            <span className="fv mono" data-admissible={facts.rule}>
              {facts.dtypes.join(', ')}
            </span>
          </div>
        </>
      )}
      {facts.state ? null : (
        <>
          <h3 className="ih">
            {text('Location')}
            <span className="ihn">
              {textWith('{} located', `${String(preview.located)} / ${String(preview.instances)}`)}
            </span>
          </h3>
          {facts.axes.length === 0 ? null : (
            <div className="frow">
              <span className="fk">{text('axes')}</span>
              <span className="fv mono" data-axes={facts.rule}>
                {facts.axes.join(', ')}
              </span>
            </div>
          )}
          <Preview preview={preview} fresh={identity.fresh} rule={facts.rule} />
        </>
      )}
    </>
  );
}

/** S8's evaluated preview: the head, what is not drawn, and the last. */
function Preview({
  preview,
  fresh,
  rule,
}: {
  preview: LocationPreview;
  fresh: boolean;
  rule: string;
}): JSX.Element {
  if (preview.instances === 0 || preview.located === 0) {
    return (
      <div className="evald" data-preview={rule}>
        <span className="dim">{text('This identity has no location.')}</span>
      </div>
    );
  }
  return (
    <div className={fresh ? 'evald' : 'evald stale'} data-preview={rule}>
      {preview.lines.map((line, at) => (
        <div key={line.identity}>
          {at === 2 && preview.more > 0 ? (
            <div className="more" data-preview-more={rule}>
              {textWith('… {} more', String(preview.more))}
            </div>
          ) : null}
          <span className="ix">{line.indices === '' ? line.identity : line.indices}</span>{' '}
          <span className="nm" data-preview-name={line.identity}>
            {[...line.names, ...line.slices].join(' · ')}
            {line.moreNames === 0 ? '' : ` · ${textWith('{} more', String(line.moreNames))}`}
          </span>
        </div>
      ))}
    </div>
  );
}

/** §4.11's Derived for an identity: the D3 tensor rows or the D4 state rows of its instances. */
function IdentityRows({ sheet }: { sheet: PlaceSheet }): JSX.Element | null {
  if (sheet.identity === null) return null;
  const { reading, derived } = sheet.identity;
  return (
    <>
      <h3 className="ih">
        {text('Derived')}
        <span className="ihn">{textWith('{} instances', String(derived.instances))}</span>
      </h3>
      {derived.instances === 0 ? (
        <div className="arow more">{text('Nothing has been derived for this document yet.')}</div>
      ) : null}
      {derived.tensors.map((tensor) => (
        <div className="drow" key={tensor.identity} data-tensor={tensor.identity}>
          <span className="dn mono">{tensor.identity}</span>
          <span className="dm">
            {`${tensor.role} · ${tensor.dtype} · ${tensor.shape} · `}
            <b>{tensor.bytes === null ? EMPTY : sizeText(Number(tensor.bytes))}</b>
            {tensor.tied ? ` · ${text('tied')}` : ''}
            {tensor.located ? ` · ${text('located')}` : ''}
          </span>
        </div>
      ))}
      {derived.states.map((state) => (
        <div className="drow" key={state.identity} data-state-row={state.identity}>
          <span className="dn mono">{state.identity}</span>
          <span className="dm">
            {`${state.evolution} · ${state.access} · `}
            <b>
              {state.bytesPerCachedPosition === null
                ? EMPTY
                : textWith('{} / cached position', sizeText(Number(state.bytesPerCachedPosition)))}
            </b>
          </span>
        </div>
      ))}
      {derived.instances === 0 ? null : (
        <div className="drow tot" data-identity={reading.identity}>
          <span className="dn mono">{reading.identity}</span>
          <span className="dm">
            <b>
              {derived.bytes !== null
                ? sizeText(Number(derived.bytes))
                : derived.bytesPerCachedPosition !== null
                  ? textWith('{} / cached position', sizeText(Number(derived.bytesPerCachedPosition)))
                  : EMPTY}
            </b>
            {textWith(' · {} instances', String(derived.instances))}
          </span>
        </div>
      )}
    </>
  );
}

/** §4.11's Derived for an edge, an input and an output: the D2 value, and the stream. */
function ValueRows({ sheet }: { sheet: PlaceSheet }): JSX.Element | null {
  if (sheet.value === null && sheet.stream === null) return null;
  const value = sheet.value;
  return (
    <>
      <h3 className="ih">{text('Derived')}</h3>
      {value === null ? null : (
        <div className="drow" data-value={value.value}>
          <span className="dn mono">{value.value}</span>
          <span className="dm">
            <b>{value.geometry}</b>
            {` · ${value.role} · ${value.domain}`}
            {value.bytesPerElement === null
              ? ''
              : ` · ${sizeText(value.bytesPerElement)} / element`}
          </span>
        </div>
      )}
      {value === null || value.requiredFor.length === 0 ? null : (
        <div className="frow">
          <span className="fk">{text('required for')}</span>
          <span className="fv mono" data-required-for={value.value}>
            {value.requiredFor.join(', ')}
          </span>
        </div>
      )}
      {value === null || value.exposed.length === 0 ? null : (
        <div className="frow">
          <span className="fk">{text('exposed as')}</span>
          <span className="fv mono">{value.exposed.join(', ')}</span>
        </div>
      )}
      {sheet.stream === null ? null : (
        <div className="drow tot" data-stream={sheet.stream.name}>
          <span className="dn mono">{sheet.stream.name}</span>
          <span className="dm">
            {`${sheet.stream.kind} · ${sheet.stream.count}`}
            {sheet.stream.fragmentAlignment === null
              ? ''
              : textWith(' · fragments align to {}', String(sheet.stream.fragmentAlignment))}
          </span>
        </div>
      )}
    </>
  );
}

/**
 * §4.11's "scoped bindings summary (counts, with **Show in canvas**)" — S3's own line.
 *
 * The counts are the core's (`foldedGraph`'s `held`); the button is the second half of that row,
 * which feature 2.12 left to this one: what a composition's scoped bindings are *shown in* is the
 * drill-in (§4.8), where they are the edges and the chips, so "Show in canvas" opens it.
 */
function Held({ sheet, one }: { sheet: PlaceSheet; one: OpenDocument }): JSX.Element | null {
  const store = useDocumentsStore();
  if (sheet.held.length === 0) return null;
  const composition = compositionAt(foldedGraph(one.session.store.tree), sheet.pointer);
  return (
    <>
      <h3 className="ih">{text('Holds')}</h3>
      {sheet.held.map((held) => (
        <div className="frow" key={held.name} data-held={held.name}>
          <span className="fk">{held.name}</span>
          <span className="fv mono">{String(held.count)}</span>
        </div>
      ))}
      {composition === null ? null : (
        <div className="frow">
          <span className="fk" />
          <span className="fv">
            <button
              type="button"
              className="btn"
              data-show-in-canvas={composition}
              onClick={() => {
                store.getState().drillInto(composition, one.id);
              }}
            >
              {text('Show in canvas')}
            </button>
          </span>
        </div>
      )}
    </>
  );
}

/** The two columns the core answers rather than the schema — §4.16's last two. */
const USED_BY = 'used by';
const RESOLVED = 'resolved';
