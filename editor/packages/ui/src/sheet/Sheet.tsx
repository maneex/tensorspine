/**
 * The Properties sheet of the current selection — plan §4.11, §4.12, artboard S6.
 *
 * The right panel's body. With a **site** selected it is §4.11's first row in full: Identity,
 * Arguments (§4.12), Ports, Parameters, Constants, States and Derived; with anything else selected
 * it is the row every sheet starts with — the name, edited here — and the sections the feature that
 * can answer them will add (2.12's other sheets, 2.13's bindings).
 *
 * **Everything on it is a fact somebody else answered.** `describe` says which ports, slots and
 * states exist, whether an argument applies, what its default resolved to and what a shape
 * evaluates to; the generated argument schema says what edits a literal; the derived document says
 * what a slot costs. The inventory's §7 is the whole rule, and this component computes no verdict
 * and rounds no figure.
 *
 * **What is editable here** (§4.4: "a value is edited in the sheet's row, or by clicking the
 * element itself"): the site's **name**, the **version** its primitive is pinned to, its
 * **families**, and every **argument** — its source mode, its literal, the quantity or the index it
 * names, `Pin value` and the clearing that puts a row back on its default. What is not: the guard
 * and the location tokens (the expression and token editors of 2.11 and 2.13), and the binding
 * gestures beside the slot rows ("Bind privately", "Tie to…", "Share with…", the dtype select),
 * which feature 2.13's own block names one by one. Each says so where it is drawn rather than
 * looking finished.
 */
import { useEffect, useState, type JSX } from 'react';

import {
  isJsonNumber,
  noSiteDerived,
  pyStr,
  siteDerived,
  type JsonValue,
  type SchemaFacts,
  type SchemaRegistry,
  type SiteDescription,
} from '@tensorspine/lang';
import {
  EditError,
  insertItem,
  nodeAt,
  pointerOf,
  remove,
  setMemberAt,
  setValue,
  type Command,
  type EditContext,
  type Path,
  type SchemaShapes,
  type Shape,
} from '@tensorspine/store';

import { useDocuments, useDocumentsStore } from '../documents/context.js';
import { sizeText } from '../documents/figures.js';
import type { OpenDocument } from '../documents/store.js';
import { NAME_FIELD, renameCommand, selectedRow } from '../explorer/Explorer.js';
import { outlineOf, type OutlineRow } from '../explorer/outline.js';
import {
  editsOneValue,
  formContext,
  formOf,
  LIST,
  MAP,
  SECTION,
  type Form,
  type FormContext,
  type FormMode,
  type FormRow,
} from '../forms/index.js';
import {
  blankAt as blankExpression,
  documentResolver,
  editsExpression,
  ExpressionEditor,
  printValue,
  unionAnchorOf,
  type Resolver,
} from '../expressions/index.js';
import { presentation } from '../presentation/index.js';
import { text, textWith } from '../shell/strings.js';
import { argumentSheet, type ArgumentRow, type ArgumentSheet } from './arguments.js';
import { NO_ARGUMENT_FACTS, primitiveId, type ArgumentSchema } from './artifact.js';
import {
  blankOf,
  clearValue,
  literalMode,
  pinValue,
  referringModes,
  writeLiteral,
  writeMode,
  writeRecord,
  writeValue,
} from './edits.js';
import { instanceSheet, type InstanceSheet, type PortRow, type SlotRow, type StateRow } from './instance.js';
import { namesFor } from './names.js';

/** The form context of one registry, built once — the walker's own caches live in it (2.3). */
let lastContext: { registry: SchemaRegistry; context: FormContext } | null = null;

function contextFor(registry: SchemaRegistry | null): FormContext | null {
  if (registry === null) return null;
  if (lastContext !== null && lastContext.registry === registry) return lastContext.context;
  const context = formContext(registry, presentation());
  lastContext = { registry, context };
  return context;
}

/** The outline of one document, memoised across components as the Problems panel memoises it. */
let lastOutline: { key: string; rows: readonly OutlineRow[] } | null = null;

function outlineFor(one: OpenDocument): readonly OutlineRow[] {
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

/** The site the selection stands for, when it stands for one. */
function siteOf(one: OpenDocument, selection: Path | undefined): SiteDescription | undefined {
  if (selection === undefined) return undefined;
  const pointer = pointerOf(selection);
  for (const site of one.reading.facts?.sites.values() ?? []) {
    if (pointerOf(site.segments) === pointer) return site;
  }
  return undefined;
}

/** The sheet of the current selection, or the line that says there is none. */
export function SelectionSheet(): JSX.Element {
  const store = useDocumentsStore();
  const current = useDocuments((state) => state.current);
  const open = useDocuments((state) => state.open);
  const registry = useDocuments((state) => state.registry);
  const library = useDocuments((state) => state.library);
  const one = open.find((each) => each.id === current);
  const selection = one?.selection;
  const site = one === undefined ? undefined : siteOf(one, selection);
  const identity =
    site === undefined ? undefined : primitiveId(pyStr(site.primitive), pyStr(site.version));

  // The artifact of the primitive the selection pins, read once and kept (F5). Asked for in an
  // effect because it is a *read*: the sheet is drawn on every keystroke and must not start one.
  useEffect(() => {
    if (identity !== undefined) store.getState().loadArgumentSchema(identity);
  }, [identity, store]);

  if (one === undefined || selection === undefined) {
    return <p className="empty-sub">{text('Nothing is selected.')}</p>;
  }
  if (site === undefined) {
    return <PlaceSheet />;
  }
  const context = contextFor(registry);
  if (context === null) {
    return <p className="empty-sub">{text('No schemas are loaded, so no form can be generated.')}</p>;
  }
  const artifact = identity === undefined ? undefined : (library.arguments.get(identity) ?? undefined);
  return <SitePanel one={one} site={site} context={context} artifact={artifact ?? undefined} />;
}

/** The sheet of a place that is not a site: the row every sheet of §4.11 starts with. */
function PlaceSheet(): JSX.Element {
  const store = useDocumentsStore();
  const current = useDocuments((state) => state.current);
  const open = useDocuments((state) => state.open);
  const one = open.find((each) => each.id === current);
  const selection = one?.selection;
  const revision = one?.session.store.revision ?? -1;
  const pointer = selection === undefined ? null : pointerOf(selection);
  const row = selection === undefined ? null : selectedRow(store);
  const [name, setName] = useState(row?.label ?? '');

  // The sheet follows the selection and the document: a rename made in the tree, an undo, a
  // selection that moved — the field shows what the document holds, never what was typed into it
  // for something else.
  useEffect(() => {
    setName(row?.label ?? '');
    // The row is rebuilt on every render; the selection and the revision are what change it.
  }, [pointer, revision]);

  if (one === undefined || row === null || selection === undefined) {
    return <p className="empty-sub">{text('Nothing is selected.')}</p>;
  }

  const commit = (): void => {
    const to = name.trim();
    if (to === '' || to === row.label) {
      setName(row.label);
      return;
    }
    try {
      store.getState().edit(renameCommand(row, to));
      store.getState().selectPlace([...row.path.slice(0, -1), to]);
    } catch (error) {
      store.getState().setToast({ text: error instanceof EditError ? error.message : String(error) });
      setName(row.label);
    }
  };

  return (
    <>
      {/* `h2`, not the design's `h4`: the page's own heading is the open document's name, and a
          heading that skipped two levels is a heading order a reader cannot follow — the same
          correction feature 2.6 made to the dialogs, found the same way. */}
      <h2 className="ih">
        {text('Identity')}
        {row.declares === undefined ? null : <span className="ihn">{row.declares}</span>}
      </h2>
      <NameRow value={name} onChange={setName} onCommit={commit} onRevert={() => { setName(row.label); }} editable={row.named} />
      <div className="frow">
        <span className="fk">{text('Place')}</span>
        <span className="fv mono" data-place={pointer}>
          {pointer}
        </span>
      </div>
      {row.tail === undefined ? null : (
        <div className="frow">
          <span className="fk">{text('Holds')}</span>
          <span className="fv mono">{row.tail}</span>
        </div>
      )}
      <PlaceExpressions one={one} at={selection} />
      <p className="empty-sub">
        {text(
          'The sections of this sheet are the ones the feature that can answer them adds: the other declarations’ sheets and the tables, and the bindings and locations.',
        )}
      </p>
    </>
  );
}

/**
 * Every expression and condition written under the selected place — §4.13, and only that.
 *
 * §4.13's own list is what this section is: "the same editor serves guards, index ranges,
 * derivations, defaults, extents, domain bounds, location offsets, `present_when`, rule `when`s,
 * invariants, cost entries and granularities". A quantity's `derivation` is one of them, and it is
 * editable here before the **quantity sheet** exists, because that sheet — its type select, its
 * source kind, its domain, its `Used by` — is feature 2.12's block and nothing of it is written
 * here.
 *
 * The places are found by the generic walker (feature 2.3) and never by a path: whatever the
 * schema puts an expression at, the section shows, and a grammar that grew another one needs no
 * change here. Only what the document **writes** is listed; creating one is the gesture of the
 * sheet that declares it.
 */
function PlaceExpressions({ one, at }: { one: OpenDocument; at: Path }): JSX.Element | null {
  const context = contextFor(useDocuments((state) => state.registry));
  const value = context === null ? undefined : nodeAt(one.session.store.tree, at);
  if (context === null || value === undefined) return null;
  const rows = expressionRowsOf(one, at, context);
  if (rows.length === 0) return null;
  return (
    <>
      <h2 className="ih">
        {text('Expressions')}
        <span className="ihn">{String(rows.length)}</span>
      </h2>
      {rows.map((row) => (
        <ExpressionField
          key={row.path}
          one={one}
          path={[...at, ...row.steps] as Path}
          widget={row.widget}
          name={row.label}
          context={context}
          label={row.path.slice(1).replace(/\//g, ' · ')}
        />
      ))}
    </>
  );
}

/** The expression rows of one place, memoised on the selection and the revision. */
let lastExpressions: { key: string; rows: readonly FormRow[] } | null = null;

function expressionRowsOf(one: OpenDocument, at: Path, context: FormContext): readonly FormRow[] {
  // Walking the place is what finds them (feature 2.3), and a place can be a whole composition —
  // 2.3 measured a whole document at 15.9 ms — so the walk is paid once per state of the tree and
  // not once per keystroke of the editor it draws.
  const key = `${one.id}@${String(one.session.store.revision)}#${pointerOf(at)}`;
  if (lastExpressions !== null && lastExpressions.key === key) return lastExpressions.rows;
  const value = nodeAt(one.session.store.tree, at);
  const form =
    value === undefined
      ? { rows: [], notes: [] }
      : formOf(context, { shape: shapeAt(context.shapes, at, one.session.store.role), value });
  const rows = form.rows.filter((row) => editsExpression(row.widget) && row.present);
  lastExpressions = { key, rows };
  return rows;
}

/** The name row every sheet starts with — §4.4's "a value is edited in the sheet's row". */
function NameRow({
  value,
  editable,
  onChange,
  onCommit,
  onRevert,
}: {
  value: string;
  editable: boolean;
  onChange: (value: string) => void;
  onCommit: () => void;
  onRevert: () => void;
}): JSX.Element {
  return (
    <div className="frow">
      <span className="fk">{text('Name')}</span>
      <span className="fv">
        {editable ? (
          <input
            className="ctl wide mono"
            value={value}
            {...{ [NAME_FIELD]: 'true' }}
            aria-label={text('Name')}
            onChange={(event) => {
              onChange(event.target.value);
            }}
            onBlur={onCommit}
            onKeyDown={(event) => {
              event.stopPropagation();
              if (event.key === 'Enter') onCommit();
              if (event.key === 'Escape') onRevert();
            }}
          />
        ) : (
          <span className="mono">{value}</span>
        )}
      </span>
    </div>
  );
}

/** The whole sheet of one site. */
function SitePanel({
  one,
  site,
  context,
  artifact,
}: {
  one: OpenDocument;
  site: SiteDescription;
  context: FormContext;
  artifact: ArgumentSchema | null | undefined;
}): JSX.Element {
  const store = useDocumentsStore();
  const versions = useDocuments((state) => state.library.versions);
  const [inapplicable, setInapplicable] = useState(false);
  const [problemsFirst, setProblemsFirst] = useState(false);
  const sheet = instanceSheet(
    site,
    one.reading.derived === null ? noSiteDerived() : siteDerived(one.reading.derived, site.where),
  );
  const args = argumentSheet({
    facts: site.arguments.facts,
    invariants: site.arguments.invariants,
    tree: one.session.store.tree,
    role: one.session.store.role,
    context,
    shapes: context.shapes,
    artifact: artifact ?? undefined,
    showInapplicable: inapplicable,
    problemsFirst,
  });
  const outline = outlineFor(one);
  const stale = one.reading.derivedAt !== one.reading.revision;
  return (
    <>
      <h2 className="insp-title">{sheet.name}</h2>
      <p className="insp-kind">
        {sheet.composition === null ? (
          text('top-level declaration')
        ) : (
          <>
            {`${text('site of')} `}
            <span>{sheet.composition}</span>
          </>
        )}
      </p>
      <p className="prim-chip mono">{`${sheet.primitive}@${sheet.version}`}</p>
      <Identity one={one} sheet={sheet} site={site} versions={versions} context={context} />
      <Arguments
        one={one}
        args={args}
        artifact={artifact ?? undefined}
        outline={outline}
        context={context}
        inapplicable={inapplicable}
        onInapplicable={setInapplicable}
        problemsFirst={problemsFirst}
        onProblemsFirst={setProblemsFirst}
      />
      <Ports sheet={sheet} />
      <Slots
        title="Parameters"
        rows={sheet.parameters}
        note={text('This primitive declares no parameter slot.')}
      />
      <Slots
        title="Constants"
        rows={sheet.constants}
        note={text('This primitive declares no constant slot.')}
      />
      <States rows={sheet.states} />
      <Derived sheet={sheet} stale={stale} />
      <p className="empty-sub">
        {text(
          'Binding a slot, tying it to an identity, sharing a state and editing a location are the bindings feature; the rows they act on are here.',
        )}
      </p>
      <button
        type="button"
        className="btn ghost"
        data-sheet="reveal"
        onClick={() => {
          store.getState().revealPlace([...site.segments] as Path);
        }}
      >
        {text('Show in Explorer')}
      </button>
    </>
  );
}

/**
 * §4.11's Identity: the name, the pinned primitive and its version, the families, the guard.
 *
 * **Generated, like every other form.** The rows below the name are the walker's over
 * `instance_definition` at the site's own place — so the members are the schema's, in the schema's
 * order, with the schema's labels, and a grammar that grew one shows it here without a line being
 * written. Two of them get a list the schema does not enumerate, and `presentation.json` is where
 * that is said (§1): the version is picked from the versions the library carries, a family from the
 * names the workspace uses.
 *
 * The arguments are a member of the same object and are **not** drawn here: §4.12 gives them their
 * own section, and which member they are is the core's own answer — the place its facts are written
 * at — never a member name typed in a component.
 */
function Identity({
  one,
  site,
  sheet,
  context,
  versions,
}: {
  one: OpenDocument;
  site: SiteDescription;
  sheet: InstanceSheet;
  context: FormContext;
  versions: ReadonlyMap<string, readonly string[]>;
}): JSX.Element {
  const store = useDocumentsStore();
  const [name, setName] = useState(sheet.name);
  const revision = one.session.store.revision;
  useEffect(() => {
    setName(sheet.name);
  }, [sheet.name, revision]);
  // The outline this sheet already holds, not a second walk of the document: `selectedRow` builds
  // one per call and the sheet is redrawn on every keystroke (feature 2.7 measured 5.7 ms for the
  // largest document, and the Problems panel memoises it for the same reason).
  const row = outlineFor(one).find((each) => each.pointer === pointerOf(site.segments)) ?? null;
  const form = instanceForm(one, site, context);
  // Which member holds the arguments is the *core's* answer — the place its own facts are written
  // at — and a primitive that declares none has no fact to say so, where the map it would be
  // written in is the one map of the place. §4.12 gives that member its own section either way.
  const held = site.arguments.facts[0]?.at[site.segments.length];
  const members = form.rows.filter(
    (each) =>
      each.depth === 1 &&
      (held === undefined
        ? each.widget !== MAP
        : each.steps[each.steps.length - 1] !== held),
  );

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

  return (
    <>
      <h2 className="ih">{text('Identity')}</h2>
      <NameRow
        value={name}
        editable={row !== null && row.named}
        onChange={setName}
        onCommit={commit}
        onRevert={() => { setName(sheet.name); }}
      />
      {members.map((member) => (
        <MemberRow
          key={member.path}
          one={one}
          row={member}
          base={[...site.segments] as Path}
          form={form}
          sheet={sheet}
          versions={versions}
          context={context}
        />
      ))}
    </>
  );
}

/** One generated row of the Identity section, drawn by what its widget is. */
function MemberRow({
  one,
  row,
  base,
  form,
  sheet,
  versions,
  context,
}: {
  one: OpenDocument;
  row: FormRow;
  /**
   * Where the form the row belongs to starts, in the document.
   *
   * A generated form's `steps` are relative to the value it was given (feature 2.3), and the
   * Identity section's form is the walker over the **site** — so a row's place in the document is
   * the site's place and the row's steps, and a reading that took the steps alone named a member
   * of the document's root. Feature 2.11 found it: the guard row read `/when`, which no document
   * has, so it drew an em dash where its condition stands, and a version written from that row
   * would have been written there too.
   */
  base: Path;
  form: Form;
  sheet: InstanceSheet;
  versions: ReadonlyMap<string, readonly string[]>;
  context: FormContext;
}): JSX.Element {
  const under = form.rows.filter(
    (each) => each.depth === row.depth + 1 && each.path.startsWith(`${row.path}/`),
  );
  // The label is the schema's own (its `title`, or the member name), and §4.21 shows schema text
  // as written: it is not a key of the interface's dictionary.
  const label = row.label;
  if (row.widget === LIST) {
    return row.present ? (
      <ListRow one={one} row={row} base={base} items={under} />
    ) : (
      <div className="frow" data-member={row.label}>
        <span className="fk">{label}</span>
        <span className="fv none">{EMPTY}</span>
      </div>
    );
  }
  if (row.widget === SECTION) {
    return (
      <div className="frow" data-member={row.label}>
        <span className="fk">{label}</span>
        <span className="fv">
          {under.map((each) => (
            <span key={each.path} className="subfield">
              <i className="sublabel">{each.label}</i>
              <ScalarField row={each} base={base} sheet={sheet} versions={versions} />
            </span>
          ))}
        </span>
      </div>
    );
  }
  if (!row.present) {
    return (
      <div className="frow" data-member={row.label}>
        <span className="fk">{label}</span>
        <span className="fv none">{EMPTY}</span>
      </div>
    );
  }
  if (editsExpression(row.widget)) {
    // A place a presentation binding gives an editor to — §4.13's, on the guard's own condition.
    return (
      <ExpressionField
        one={one}
        path={[...base, ...row.steps] as Path}
        widget={row.widget}
        name={row.label}
        context={context}
        label={label}
      />
    );
  }
  if (!editsOneValue(row.widget)) {
    // A place with an editor nobody has written yet: the printed value, and the row says so.
    const value = nodeAt(one.session.store.tree, [...base, ...row.steps] as Path);
    return (
      <div className="frow" data-member={row.label}>
        <span className="fk">{label}</span>
        <span className="fv mono" data-editor={row.widget}>
          {value === undefined
            ? EMPTY
            : printValue(
                { registry: context.registry, shapes: context.shapes, bindings: context.bindings },
                shapeAt(context.shapes, [...base, ...row.steps] as Path, one.session.store.role),
                value,
              )}
        </span>
      </div>
    );
  }
  return (
    <div className="frow" data-member={row.label}>
      <span className="fk">{label}</span>
      <span className="fv">
        <ScalarField row={row} base={base} sheet={sheet} versions={versions} />
      </span>
    </div>
  );
}

/**
 * A field whose value is an expression or a condition — §4.13, artboard S7.
 *
 * The row of §4.11 that holds a guard is the one place of the instance sheet this happens today;
 * the same component serves an index range, a derivation and every other place the plan's list
 * names, since what it needs is a place of the document and the anchor its value stands at.
 *
 * The **anchor is the schema's**, stepped down from the document's root like every other reading
 * of a place (feature 2.3's rule: a caller with a shape hands it over, never an anchor written
 * here), and the union along that chain is what the grammar is read at.
 */
function ExpressionField({
  one,
  path,
  widget,
  name,
  context,
  label,
}: {
  one: OpenDocument;
  path: Path;
  widget: string;
  name: string;
  context: FormContext;
  label: string;
}): JSX.Element {
  const store = useDocumentsStore();
  const shape = shapeAt(context.shapes, path, one.session.store.role);
  const anchor = unionAnchorOf(context, shape);
  const value = nodeAt(one.session.store.tree, path);
  return (
    <div className="frow wide" data-member={name} data-editor={widget}>
      <span className="fk">{label}</span>
      <span className="fv">
        <ExpressionEditor
          anchor={anchor}
          value={value}
          context={context}
          label={label}
          names={namePicker(one, context, path)}
          resolve={resolverFor(one)}
          onChange={(next) => {
            edit(store, (made) =>
              nodeAt(made.tree, path) === undefined
                ? setMember(made, path, next, `Set ${name}`)
                : setValue(made, { path, value: next, label: `Set ${name}` }),
            );
          }}
        />
      </span>
    </div>
  );
}

/** A member written where the document has none: the map gains it, appended (§5.5). */
function setMember(context: EditContext, path: Path, value: JsonValue, label: string): Command {
  const name = path[path.length - 1];
  return setMemberAt(context, {
    path: path.slice(0, -1),
    name: typeof name === 'string' ? name : String(name ?? ''),
    value,
    label,
  });
}

/**
 * What an expression editor's pickers offer — the same answer §4.12's referring modes get.
 *
 * `namesFor` is feature 2.10's, and it reads the *outline* (2.7's, which reads the presentation
 * file's `declares`) rather than the document directly: a quantity is in scope everywhere and an
 * index only under the composition that declares it, which is a fact about the document's shape
 * and not about the expression being edited.
 */
function namePicker(
  one: OpenDocument,
  context: FormContext,
  at: Path,
): (referent: string) => readonly string[] {
  return (referent) =>
    namesFor({
      outline: outlineOf({
        tree: one.session.store.tree,
        role: one.session.store.role,
        shapes: context.shapes,
        bindings: context.bindings,
        openAll: true,
      }),
      referent,
      at,
      tree: one.session.store.tree,
      role: one.session.store.role,
      context,
      shapes: context.shapes,
    }).map((choice) => choice.name);
}

/** The resolver of one document, memoised on its tree as the outline is on its revision. */
let lastResolver: { tree: JsonValue; resolve: Resolver } | null = null;

/**
 * What the core resolves a document's expressions to — S7's `4096` beside a quantity.
 *
 * Without an assignment, as the explorer's own figures are read (feature 2.7): a template's
 * external quantities resolve to nothing until §4.6's assignment sheet exists, and an expression
 * over them says nothing rather than saying something wrong.
 *
 * Kept per tree because building it resolves **every** quantity of the document, and the sheet is
 * drawn on every keystroke: the tree is immutable (2.1), so its identity is what says whether the
 * answer still holds.
 */
function resolverFor(one: OpenDocument): Resolver {
  const tree = one.session.store.tree;
  if (lastResolver !== null && lastResolver.tree === tree) return lastResolver.resolve;
  const resolve = documentResolver(tree);
  lastResolver = { tree, resolve };
  return resolve;
}

/** A repeatable list of names — §4.11's chip editor, with the workspace's names as suggestions. */
function ListRow({
  one,
  row,
  base,
  items,
}: {
  one: OpenDocument;
  row: FormRow;
  /** Where the form starts in the document — see {@link MemberRow}. */
  base: Path;
  items: readonly FormRow[];
}): JSX.Element {
  const store = useDocumentsStore();
  const [adding, setAdding] = useState('');
  const suggestions = suggestionsFor(one, items[0]?.picker);
  const least = row.bounds?.minItems ?? 0;
  const listId = `ts-${row.label}-names`;
  return (
    <div className="frow" data-member={row.label}>
      <span className="fk">{row.label}</span>
      <span className="fv">
        {items.map((item) => (
          <span key={item.path} className="fchip">
            <span className="mono">{String(item.written ?? '')}</span>
            <button
              type="button"
              className="tbtn"
              data-remove={String(item.written ?? '')}
              aria-label={textWith('Remove {}', String(item.written ?? ''))}
              title={textWith('Remove {}', String(item.written ?? ''))}
              disabled={items.length <= least}
              onClick={() => {
                edit(store, (made) =>
                  remove(made, {
                    path: [...base, ...item.steps] as Path,
                    label: `Remove ${String(item.written ?? '')}`,
                  }),
                );
              }}
            >
              ×
            </button>
          </span>
        ))}
        <input
          className="ctl mono"
          data-add={row.label}
          aria-label={textWith('Add to {}', row.label)}
          list={listId}
          value={adding}
          onChange={(event) => {
            setAdding(event.target.value);
          }}
          onKeyDown={(event) => {
            event.stopPropagation();
            if (event.key !== 'Enter') return;
            const written = adding.trim();
            if (written === '') return;
            edit(store, (made) =>
              insertItem(made, {
                path: [...base, ...row.steps] as Path,
                value: written,
                label: `Add ${written}`,
              }),
            );
            setAdding('');
          }}
        />
        <datalist id={listId}>
          {suggestions.map((name) => (
            <option key={name} value={name} />
          ))}
        </datalist>
      </span>
    </div>
  );
}

/** One scalar member: a select where the schema or a picker enumerates, a field otherwise. */
function ScalarField({
  row,
  base,
  sheet,
  versions,
}: {
  row: FormRow;
  /** Where the form starts in the document — see {@link MemberRow}. */
  base: Path;
  sheet: InstanceSheet;
  versions: ReadonlyMap<string, readonly string[]>;
}): JSX.Element {
  const store = useDocumentsStore();
  const picked = row.picker === undefined ? undefined : (versions.get(sheet.primitive) ?? []);
  const options = row.options?.map((one) => String(one.value)) ?? picked;
  const held = String(row.written ?? '');
  if (row.constant !== undefined || options === undefined) {
    return (
      <input
        className="ctl mono"
        data-member-value={row.label}
        aria-label={row.label}
        value={held}
        readOnly={row.constant !== undefined}
        onChange={(event) => {
          edit(store, (made) =>
            setValue(made, {
              path: [...base, ...row.steps] as Path,
              value: event.target.value,
              label: `Set ${row.label}`,
            }),
          );
        }}
      />
    );
  }
  return (
    <select
      className="ctl sel mono"
      data-member-value={row.label}
      aria-label={row.label}
      value={held}
      onChange={(event) => {
        edit(store, (made) =>
          setValue(made, {
            path: [...base, ...row.steps] as Path,
            value: event.target.value,
            label: `Set ${row.label}`,
          }),
        );
      }}
    >
      {options.includes(held) ? null : <option value={held}>{held}</option>}
      {options.map((option) => (
        <option key={option} value={option}>
          {option}
        </option>
      ))}
    </select>
  );
}

/** §4.12's own section: the counts, the rows, the two toggles, the invariant block. */
function Arguments({
  one,
  args,
  artifact,
  outline,
  context,
  inapplicable,
  onInapplicable,
  problemsFirst,
  onProblemsFirst,
}: {
  one: OpenDocument;
  args: ArgumentSheet;
  artifact: ArgumentSchema | undefined;
  outline: readonly OutlineRow[];
  context: FormContext;
  inapplicable: boolean;
  onInapplicable: (on: boolean) => void;
  problemsFirst: boolean;
  onProblemsFirst: (on: boolean) => void;
}): JSX.Element {
  return (
    <>
      <h2 className="ih">
        {text('Arguments')}
        <span className="ihn">
          {textWith('{} declared', `${String(args.set)} set · ${String(args.declared)}`)}
        </span>
      </h2>
      {args.rows.map((row) => (
        <ArgumentRowView
          key={row.path}
          one={one}
          row={row}
          artifact={artifact}
          outline={outline}
          context={context}
        />
      ))}
      <div className="arow more">
        {args.hidden > 0
          ? textWith('{} inapplicable hidden', String(args.hidden))
          : text('no inapplicable rows')}
        <button
          type="button"
          className="link"
          data-sheet="inapplicable"
          aria-pressed={inapplicable}
          onClick={() => {
            onInapplicable(!inapplicable);
          }}
        >
          {text('Show inapplicable')}
        </button>
        <button
          type="button"
          className="link"
          data-sheet="problems-first"
          aria-pressed={problemsFirst}
          onClick={() => {
            onProblemsFirst(!problemsFirst);
          }}
        >
          {text('Sort problems first')}
        </button>
      </div>
      {args.invariants.map((invariant, at) => (
        <div
          key={`${String(at)}`}
          className={invariant.verdict === 'fails' ? 'inv fail' : 'inv'}
          data-invariant={invariant.verdict}
        >
          <span className={invariant.verdict === 'fails' ? 'no' : 'ok'} aria-hidden="true">
            {invariant.verdict === 'fails' ? '✗' : '✓'}
          </span>
          <span>
            {pyStr(invariant.description)}
            {invariant.shown === '' ? null : <em>{` — ${invariant.shown}`}</em>}
          </span>
        </div>
      ))}
    </>
  );
}

/** One argument row — S6's `.arow`, with its mode control, its widget and its message. */
function ArgumentRowView({
  one,
  row,
  artifact,
  outline,
  context,
}: {
  one: OpenDocument;
  row: ArgumentRow;
  artifact: ArgumentSchema | undefined;
  outline: readonly OutlineRow[];
  context: FormContext;
}): JSX.Element {
  const store = useDocumentsStore();
  const [open, setOpen] = useState(false);
  const facts = (artifact?.at(row.path) ?? NO_ARGUMENT_FACTS).facts;
  const shaped = shapedMode(row);
  const classes = ['arow'];
  if (!row.applicable) classes.push('inapp');
  if (row.error !== undefined) classes.push('errrow');
  if (row.record) classes.push('rec');
  if (row.source === 'absent') classes.push('unset');
  if (row.deprecation !== undefined) classes.push('dep');
  const title = [
    row.effective === undefined ? undefined : `${row.path} = ${row.effective}`,
    row.description,
    row.presentWhen === undefined ? undefined : `present_when: ${row.presentWhen}`,
    row.deprecation === undefined ? undefined : `deprecated: ${row.deprecation.reason}`,
    row.named.length === 0
      ? undefined
      : row.named
          .map((bound) => `${bound.edge} ${bound.inclusive ? '≤' : '<'} ${bound.argument}` +
            (bound.value === undefined ? '' : ` (${bound.value})`))
          .join(', '),
    row.error,
  ]
    .filter((one) => one !== undefined)
    .join('\n');
  return (
    <>
      <div
        className={classes.join(' ')}
        data-argument={row.path}
        data-mode={row.mode ?? ''}
        data-source={row.source}
        data-applicable={String(row.applicable)}
        style={{ paddingLeft: `${String(row.depth * 14)}px` }}
        title={title === '' ? undefined : title}
      >
        <ModeControl row={row} facts={facts} context={context} onOpen={() => { setOpen(true); }} />
        <span className="an mono">
          {row.label}
          {row.structural ? <i className="sbadge">{text('structural')}</i> : null}
          {row.required && row.source === 'absent' && row.applicable ? (
            <i className="sbadge req">{text('required')}</i>
          ) : null}
        </span>
        <Value
          one={one}
          row={row}
          facts={facts}
          outline={outline}
          context={context}
          onOpen={() => {
            setOpen((was) => !was);
          }}
        />
        {row.effective === undefined ? null : (
          <span className="ae mono" data-effective={row.effective}>
            {row.effective}
          </span>
        )}
        {row.source === 'default' || row.written ? (
          <button
            type="button"
            className="rowact"
            data-pin={row.path}
            aria-label={textWith('Pin the value of {}', row.path)}
            title={text('Pin value — write the effective value as a literal')}
            onClick={() => {
              if (row.source === 'default') {
                edit(store, (made) => pinValue(made, row, facts));
              } else {
                edit(store, (made) => clearValue(made, row));
              }
            }}
          >
            {row.source === 'default' ? text('Pin value') : text('Clear')}
          </button>
        ) : null}
      </div>
      {row.error === undefined ? null : <div className="rowmsg">{row.error}</div>}
      {row.applicable || row.presentWhen === undefined ? null : (
        <div className="rowmsg faint">{`present_when: ${row.presentWhen}`}</div>
      )}
      {open && shaped !== undefined ? (
        <div className="arow-editor" data-editor-of={row.path}>
          <ExpressionEditor
            anchor={shaped.union}
            value={nodeAt(one.session.store.tree, row.at)}
            context={context}
            label={row.path}
            names={namePicker(one, context, row.at)}
            resolve={resolverFor(one)}
            type={row.kind}
            onChange={(next) => {
              edit(store, (made) => writeValue(made, row, next));
            }}
          />
        </div>
      ) : null}
    </>
  );
}

/** The mode of a row that §4.13's editor owns, when the row is in one. */
function shapedMode(row: ArgumentRow): FormMode | undefined {
  const mode = row.modes.find((one) => one.tag === row.mode);
  return mode !== undefined && editsExpression(mode.widget) ? mode : undefined;
}

/** The source-mode control of §4.12 — one button per alternative the place offers. */
function ModeControl({
  row,
  facts,
  context,
  onOpen,
}: {
  row: ArgumentRow;
  facts: SchemaFacts;
  context: FormContext;
  onOpen: () => void;
}): JSX.Element {
  const store = useDocumentsStore();
  return (
    <select
      className="mode-seg"
      data-modes={row.path}
      aria-label={textWith('Source of {}', row.path)}
      title={textWith('Source of {}', row.path)}
      value={row.mode ?? ''}
      onChange={(event) => {
        const mode = row.modes.find((one) => one.tag === event.target.value);
        if (mode === undefined) return;
        if (mode.member !== undefined && (mode.widget === MAP || mode.widget === SECTION)) {
          // A record is written empty and its fields become rows of their own; what is missing is
          // V2's report on those rows, which is D5's rule rather than a refusal of the gesture.
          edit(store, (made) => writeRecord(made, row, mode));
          return;
        }
        if (editsExpression(mode.widget)) {
          // §4.13's editor. The blank is the first value the *grammar* accepts at the union the
          // mode's alternatives belong to, so the row has something on the schema to edit at once
          // (D5) and the core's verdict is what judges it.
          edit(store, (made) => writeValue(made, row, blankExpression(context, mode.union)));
          onOpen();
          return;
        }
        if (mode.member === undefined || !mode.inline) {
          // A shaped mode with no editor bound to it: the gesture is not refused for a semantic
          // reason (Q5) — there is nothing to open yet, and the log says so.
          store.getState().note(`sheet: the ${mode.tag} editor of ${row.path} has no editor yet`);
          return;
        }
        if (mode.referent !== undefined) {
          store.getState().note(`sheet: choose a ${mode.referent} for ${row.path} from its select`);
          return;
        }
        edit(store, (made) => writeLiteral(made, row, blankOf(facts, row.options), facts));
      }}
    >
      {row.mode === undefined ? <option value="">{EMPTY}</option> : null}
      {row.modes.map((mode) => (
        <option key={mode.tag} value={mode.tag} data-mode={mode.tag}>
          {mode.tag}
        </option>
      ))}
    </select>
  );
}

/** The value column: the literal's widget, or the select of the name the row's mode refers to. */
function Value({
  one,
  row,
  facts,
  outline,
  context,
  onOpen,
}: {
  one: OpenDocument;
  row: ArgumentRow;
  facts: SchemaFacts;
  outline: readonly OutlineRow[];
  context: FormContext;
  onOpen: () => void;
}): JSX.Element {
  const store = useDocumentsStore();
  const literal = literalMode(row.modes);
  const referring = referringModes(row.modes).find((mode) => mode.tag === row.mode);
  const label = textWith('Value of {}', row.path);

  if (referring !== undefined) {
    const choices = namesFor({
      outline,
      referent: referring.referent as string,
      at: row.at,
      kind: row.kind,
      tree: one.session.store.tree,
      role: one.session.store.role,
      context,
      shapes: context.shapes,
    });
    const held = scalarText(row.value);
    return (
      <span className="av">
        <select
          className="ctl sel mono"
          data-value={row.path}
          aria-label={label}
          value={held}
          onChange={(event) => {
            edit(store, (made) => writeMode(made, row, referring, event.target.value));
          }}
        >
          {choices.some((choice) => choice.name === held) ? null : (
            <option value={held}>{held}</option>
          )}
          {choices.map((choice) => (
            <option key={choice.name} value={choice.name}>
              {choice.name}
            </option>
          ))}
        </select>
      </span>
    );
  }

  // A place that holds a shape rather than a scalar — a record argument — has its fields as rows
  // of its own (§4.12's "a section header row; its fields as rows one level deeper"), so the value
  // column says which mode it is in and nothing is typed here.
  if (shapedMode(row) !== undefined) {
    // §4.13's text form, and the way into its editor: "every value editable in the sheet or by
    // clicking the element" (§4.4, Q4), so the printed expression is the control.
    return (
      <span className="av" data-value={row.path}>
        <button
          type="button"
          className="link mono"
          data-open-expression={row.path}
          aria-label={textWith('Edit the expression of {}', row.path)}
          onClick={onOpen}
        >
          {row.expression ?? text('unset')}
        </button>
      </span>
    );
  }

  if ((row.mode !== undefined && literal !== undefined && row.mode !== literal.tag) ||
      !editsOneValue(row.widget)) {
    return (
      <span className="av" data-value={row.path}>
        <span className="dim">{row.expression ?? row.mode ?? text('unset')}</span>
      </span>
    );
  }

  if (row.options !== undefined) {
    return (
      <span className="av">
        <select
          className="ctl sel mono"
          data-value={row.path}
          aria-label={label}
          value={scalarText(row.value)}
          onChange={(event) => {
            edit(store, (made) => writeLiteral(made, row, event.target.value, facts));
          }}
        >
          {row.value === undefined ? <option value="">{EMPTY}</option> : null}
          {row.options.map((option) => (
            <option
              key={String(option.value)}
              value={String(option.value)}
              title={row.valueDescriptions?.[String(option.value)]}
            >
              {option.label}
            </option>
          ))}
        </select>
      </span>
    );
  }

  if (facts.holdsTruth) {
    return (
      <span className="av">
        <input
          type="checkbox"
          className="tog"
          data-value={row.path}
          aria-label={label}
          checked={row.value === true || (row.value === undefined && row.effective === 'True')}
          onChange={(event) => {
            edit(store, (made) => writeLiteral(made, row, event.target.checked, facts));
          }}
        />
      </span>
    );
  }

  if (facts.holdsWholeNumber || facts.holdsNumber) {
    return (
      <span className="av">
        <input
          type="number"
          className="ctl mono"
          data-value={row.path}
          aria-label={label}
          step={facts.holdsWholeNumber && !facts.holdsNumber ? 1 : 'any'}
          value={numberText(row)}
          onChange={(event) => {
            const held = Number(event.target.value);
            if (event.target.value === '' || Number.isNaN(held)) return;
            edit(store, (made) => writeLiteral(made, row, held, facts));
          }}
        />
        {row.unit === undefined ? null : <i className="unit">{row.unit}</i>}
      </span>
    );
  }

  return (
    <span className="av">
      <input
        className="ctl mono"
        data-value={row.path}
        aria-label={label}
        value={scalarText(row.value)}
        placeholder={row.effective ?? ''}
        onChange={(event) => {
          edit(store, (made) => writeLiteral(made, row, event.target.value, facts));
        }}
      />
    </span>
  );
}

/** §4.11's Ports: what feeds an input, what consumes an output, and what is absent. */
function Ports({ sheet }: { sheet: InstanceSheet }): JSX.Element {
  const rows = [...sheet.inputs, ...sheet.outputs];
  const present = rows.filter((row) => row.present).length;
  return (
    <>
      <h2 className="ih">
        {text('Ports')}
        <span className="ihn">
          {textWith('{} present', `${String(rows.length)} declared · ${String(present)}`)}
        </span>
      </h2>
      {rows.map((row) => (
        <PortRowView key={`${row.side}:${row.name}`} row={row} />
      ))}
    </>
  );
}

function PortRowView({ row }: { row: PortRow }): JSX.Element {
  return (
    <div className="frow" data-port={`${row.side}:${row.name}`} data-present={String(row.present)}>
      <span className="fk mono">{row.name}</span>
      <span className={row.present ? 'fv' : 'fv none'}>
        {!row.present ? (
          text('absent under these arguments')
        ) : row.side === 'inputs' ? (
          row.fedBy === null ? (
            <em>{text('unfed')}</em>
          ) : (
            <>
              {text('fed by')} <em className="mono">{row.fedBy}</em>
            </>
          )
        ) : row.consumed ? (
          text('consumed')
        ) : (
          <em>{text('unconsumed')}</em>
        )}
        {row.shape === '' ? null : (
          <>
            <br />
            <span className="mono">{`[${row.shape}]`}</span>
          </>
        )}
      </span>
    </div>
  );
}

/** §4.11's Parameters and Constants: one row per declared slot, present or not. */
function Slots({
  title,
  rows,
  note,
}: {
  title: string;
  rows: readonly SlotRow[];
  note: string;
}): JSX.Element {
  const present = rows.filter((row) => row.present).length;
  return (
    <>
      <h2 className="ih">
        {text(title)}
        <span className="ihn">
          {rows.length === 0
            ? text('none declared')
            : textWith('{} present', `${String(rows.length)} declared · ${String(present)}`)}
        </span>
      </h2>
      {rows.length === 0 ? <div className="arow more">{note}</div> : null}
      {rows.map((row) => (
        <div
          key={row.name}
          className={row.present ? 'arow' : 'arow inapp'}
          data-slot={row.name}
          data-present={String(row.present)}
        >
          <span className="an mono">{row.name}</span>
          <span className="av">
            {!row.present ? (
              <span className="dim">{text('absent under these arguments')}</span>
            ) : row.identity === null ? (
              <i className="slot unbound">{text('unbound')}</i>
            ) : (
              <i className="slot priv">{row.identity}</i>
            )}
          </span>
          {row.present ? (
            <span className="ae mono" title={textWith('role {}', row.role)}>
              {row.shape === '' ? '' : `[${row.shape}]`}
            </span>
          ) : null}
        </div>
      ))}
    </>
  );
}

/** §4.11's States: the applying rule and everything it decided. */
function States({ rows }: { rows: readonly StateRow[] }): JSX.Element {
  const present = rows.filter((row) => row.present);
  return (
    <>
      <h2 className="ih">
        {text('States')}
        <span className="ihn">
          {rows.length === 0
            ? text('none declared')
            : textWith('{} present', String(present.length))}
        </span>
      </h2>
      {rows.length === 0 ? (
        <div className="arow more">{text('This primitive declares no state port.')}</div>
      ) : null}
      {rows.map((row) => (
        <div key={row.name} data-state={row.name} data-present={String(row.present)}>
          <div className={row.present ? 'arow' : 'arow inapp'}>
            <span className="an mono">{row.name}</span>
            <span className="av">
              {!row.present ? (
                <span className="dim">{text('absent under these arguments')}</span>
              ) : row.identity === null ? (
                <i className="slot unbound">{text('unbound')}</i>
              ) : (
                <i className="slot st">{row.identity}</i>
              )}
            </span>
          </div>
          {!row.present ? null : (
            <>
              <div className="frow">
                <span className="fk">{text('rule')}</span>
                <span className="fv mono" data-rule={String(row.rule ?? '')}>
                  {row.rule === null
                    ? text('none applies')
                    : `${String(row.rule)} of ${String(row.rules)}`}
                </span>
              </div>
              <div className="frow">
                <span className="fk">{text('evolution')}</span>
                <span className="fv mono">{`${row.evolution ?? EMPTY} · ${row.access ?? EMPTY} · ${row.sharing ?? EMPTY}`}</span>
              </div>
              <div className="frow">
                <span className="fk">{text('indexed by')}</span>
                <span className="fv mono">{row.indexedBy}</span>
              </div>
              <div className="frow">
                <span className="fk">{text('writer')}</span>
                <span className="fv">
                  {row.written ? text('this instance') : text('another member of the identity')}
                </span>
              </div>
              <div className="frow">
                <span className="fk">{text('payload')}</span>
                <span className="fv mono">
                  {row.payload.map((one) => `${one.name} [${one.shape}]`).join(', ')}
                </span>
              </div>
            </>
          )}
        </div>
      ))}
    </>
  );
}

/** §4.11's Derived: D3 of the slots, D4 of the states, D5's corrections, D6's options, D1's flag. */
function Derived({ sheet, stale }: { sheet: InstanceSheet; stale: boolean }): JSX.Element {
  const derived = sheet.derived;
  const nothing =
    derived.tensors.length === 0 &&
    derived.states.length === 0 &&
    derived.corrections.length === 0 &&
    sheet.partitions.length === 0;
  return (
    <>
      <h2 className="ih">
        {text('Derived')}
        <span className="ihn">
          {stale && !nothing ? text('stale') : textWith('{} nodes', String(derived.nodes))}
        </span>
      </h2>
      {nothing ? (
        <div className="arow more">{text('Nothing has been derived for this document yet.')}</div>
      ) : null}
      {derived.tensors.map((row) => (
        <div key={row.identity} className="drow" data-tensor={row.identity}>
          <span className="dn mono">{row.identity}</span>
          <span className="dm">
            {`${row.role} · ${row.dtype} · ${row.shape} · `}
            <b>{row.bytes === null ? EMPTY : sizeText(Number(row.bytes))}</b>
            {row.tied ? ` · ${text('tied')}` : ''}
            {row.located ? ` · ${text('located')}` : ''}
          </span>
        </div>
      ))}
      {derived.states.map((row) => (
        <div key={row.identity} className="drow" data-state-row={row.identity}>
          <span className="dn mono">{row.identity}</span>
          <span className="dm">
            {`${row.evolution} · ${row.access} · `}
            <b>
              {row.bytesPerCachedPosition === null
                ? EMPTY
                : textWith('{} / cached position', sizeText(Number(row.bytesPerCachedPosition)))}
            </b>
          </span>
        </div>
      ))}
      {derived.corrections.map((row, at) => (
        <div key={`${String(at)}`} className="drow tot" data-correction={row.per}>
          <span className="dn">{textWith('D5 · per {}', row.per)}</span>
          <span className="dm">
            <b>{row.value === null ? EMPTY : String(row.value)}</b>
            {` · ${row.status}`}
          </span>
        </div>
      ))}
      {sheet.partitions.map((row, at) => (
        <div key={`${String(at)}`} className="drow tot" data-partition={row.target}>
          <span className="dn mono">{row.target}</span>
          <span className="dm">
            <b>{row.communication.join(', ')}</b>
            {textWith(' · granularity {}', row.granularity)}
          </span>
        </div>
      ))}
      {derived.nodes === 0 ? null : (
        <div className="drow tot" data-d1={sheet.where}>
          <span className="dn">{text('D1')}</span>
          <span className="dm">
            {textWith('{} nodes', String(derived.nodes))}
            {derived.acrossPositions === null
              ? ''
              : ` · across_positions ${String(derived.acrossPositions)}`}
          </span>
        </div>
      )}
    </>
  );
}

/** What nothing at all is written as — `view.py`'s own em dash, as the figures module writes it. */
const EMPTY = '—';

/** The instance as a form: the walker over `instance_definition` at the site's own place. */
function instanceForm(one: OpenDocument, site: SiteDescription, context: FormContext): Form {
  const at = [...site.segments] as Path;
  const value = nodeAt(one.session.store.tree, at);
  return formOf(context, {
    shape: shapeAt(context.shapes, at, one.session.store.role),
    ...(value === undefined ? {} : { value }),
    label: site.key.name,
    limit: 2,
  });
}

/**
 * What a picker's list holds, for the two the sheet offers.
 *
 * A picker names a list the *editor* fills, which is why it is `presentation.json`'s and not a
 * schema's (§1). §4.11 asks for "the workspace's family names as suggestions"; what is offered is
 * the **open document's** own, which is what the editor can answer without reading every file of
 * the workspace on a keystroke — the families the core described for its sites. A suggestion is
 * not a limit: anything typed is written, as Q5 requires of every gesture.
 */
function suggestionsFor(one: OpenDocument, picker: string | undefined): string[] {
  if (picker === undefined) return [];
  const found = new Set<string>();
  for (const site of one.reading.facts?.sites.values() ?? []) {
    for (const family of site.families) found.add(pyStr(family));
  }
  return [...found].sort((a, b) => a.localeCompare(b));
}

/**
 * What a number field shows: the written value, and nothing where the row writes none.
 *
 * The *effective* value is beside the row in its own column (§4.12) and is not what the field
 * holds: typing into an empty field must write a value, not edit a default the document does not
 * carry ("defaults are not written").
 */
function numberText(row: ArgumentRow): string {
  return scalarText(row.value);
}

/**
 * A scalar of the tree as a field shows it.
 *
 * A number keeps the lexeme the file wrote (feature 0.3, D12), so a value the document holds as
 * `1e-05` reads as `1e-05` in the field it is edited in rather than as `0.00001`.
 */
function scalarText(value: JsonValue | undefined): string {
  if (value === undefined || value === null) return '';
  if (typeof value === 'string') return value;
  if (value === true || value === false) return String(value);
  if (isJsonNumber(value)) return value.lexeme ?? String(value.value);
  return '';
}

/** The shape of a place, stepped down from the document's root. */
function shapeAt(shapes: SchemaShapes, path: Path, role: string): Shape {
  let shape = shapes.root(role);
  for (const step of path) shape = shapes.step(shape, step);
  return shape;
}

/** Run a command against the current document, reporting what it refuses as a toast. */
function edit(
  store: ReturnType<typeof useDocumentsStore>,
  make: (context: EditContext) => Command,
): void {
  try {
    store.getState().edit(make);
  } catch (error) {
    store.getState().setToast({
      text: error instanceof EditError ? error.message : String(error),
    });
  }
}
