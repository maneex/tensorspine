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
  foldedGraph,
  identityReadings,
  isJsonNumber,
  noSiteDerived,
  prefixTokens,
  pyStr,
  quantityReadings,
  siteDerived,
  streamRows,
  toPython,
  valueRows,
  type JsonValue,
  type PyRecord,
  type SchemaFacts,
  type CompatibleIdentity,
  type IdentityReading,
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
import { renameCommand } from '../explorer/Explorer.js';
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
import { bindPrivately, tieTo, type HeldMember, type SlotTarget } from './bindings.js';
import { instanceSheet, type InstanceSheet, type PortRow, type SlotRow, type StateRow } from './instance.js';
import { multiSheet } from './multi.js';
import { TokenList } from './Tokens.js';
import { editsTokens, type TokenOffers } from './tokens.js';
import { PlaceView, type PlaceFacts } from './Place.js';
import { NameRow } from './Rows.js';
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

/** The sheet of the current selection — §4.11's table, row by row. */
export function SelectionSheet(): JSX.Element {
  const store = useDocumentsStore();
  const current = useDocuments((state) => state.current);
  const open = useDocuments((state) => state.open);
  const registry = useDocuments((state) => state.registry);
  const library = useDocuments((state) => state.library);
  const one = open.find((each) => each.id === current);
  const site = one === undefined ? undefined : siteOf(one, one.selection);
  // §4.11's multi-selection: every marked place that is a site, in the selection's own order.
  const marked =
    one === undefined
      ? []
      : one.marked
          .map((path) => siteOf(one, path))
          .filter((each): each is SiteDescription => each !== undefined);
  const identity =
    site === undefined ? undefined : primitiveId(pyStr(site.primitive), pyStr(site.version));

  // The artifact of the primitive the selection pins, read once and kept (F5). Asked for in an
  // effect because it is a *read*: the sheet is drawn on every keystroke and must not start one.
  useEffect(() => {
    if (identity !== undefined) store.getState().loadArgumentSchema(identity);
  }, [identity, store]);

  if (one === undefined) {
    return <p className="empty-sub">{text('No document is open.')}</p>;
  }
  const context = contextFor(registry);
  if (context === null) {
    return <p className="empty-sub">{text('No schemas are loaded, so no form can be generated.')}</p>;
  }
  if (marked.length > 1) {
    return <MultiPanel one={one} sites={marked} context={context} />;
  }
  if (site !== undefined) {
    const artifact = identity === undefined ? undefined : (library.arguments.get(identity) ?? undefined);
    return (
      <SitePanel
        one={one}
        site={site}
        context={context}
        artifact={artifact ?? undefined}
        identities={factsFor(one).identities}
      />
    );
  }
  // "Nothing selected | the Document sheet" (§4.11): the document's own place is the root, and
  // the sheet of a place is the same sheet wherever the place is.
  const path = (one.selection ?? ([]));
  const rows = outlineFor(one);
  const at = pointerOf(path);
  return (
    <PlaceView
      one={one}
      context={context}
      path={path}
      row={rows.find((row) => row.pointer === at) ?? null}
      rows={new Map(rows.map((row) => [row.pointer, row]))}
      facts={factsFor(one)}
      names={(where: Path) => namePicker(one, context, where)}
    />
  );
}

/**
 * Everything the core and the products answered about a document, for the sheets of §4.11.
 *
 * Kept per `(tree, derived)` because the sheet is drawn on every keystroke and each of these is a
 * walk of the whole document: the quantities are resolved, the identities are read through the
 * hoist (§5.2 rule 7), the folded graph is the canvas's own reading, and D2's rows are the derived
 * document's. The tree and the derived document are immutable (D1, 2.1), so their identity is what
 * says whether the answers still hold.
 */
let lastFacts: {
  tree: JsonValue;
  derived: unknown;
  verdict: unknown;
  notices: unknown;
  facts: PlaceFacts;
} | null = null;

function factsFor(one: OpenDocument): PlaceFacts {
  const tree = one.session.store.tree;
  const derived = one.reading.derived;
  const verdict = one.reading.verdict;
  const notices = one.reading.notices;
  if (
    lastFacts !== null &&
    lastFacts.tree === tree &&
    lastFacts.derived === derived &&
    lastFacts.verdict === verdict &&
    lastFacts.notices === notices
  ) {
    return lastFacts.facts;
  }
  const document = toPython(tree);
  const facts: PlaceFacts = {
    quantities: read(() => quantityReadings(document as PyRecord), []),
    identities: read(() => identityReadings(document), []),
    folded: read(() => foldedGraph(tree), null),
    derived,
    values: derived === null ? [] : valueRows(derived),
    streams: derived === null ? [] : streamRows(derived),
    problems: [...(one.reading.structural ?? []), ...(verdict?.problems ?? [])]
      .filter((problem) => problem.path !== undefined)
      .map((problem) => ({ path: problem.path, message: problem.message })),
    // The editor's own rows (§4.17's `editor` source): what §4.16 calls a notice on a table's row.
    notices,
  };
  lastFacts = { tree, derived, verdict, notices, facts };
  return facts;
}

/**
 * A reading of a document that may refuse it.
 *
 * The core raises where the tools raise (feature 1.2's rule), and a document being *edited* is
 * off the grammar between two keystrokes: `normalise` refuses a duplicate binding name, the
 * quantities refuse an unorderable bound. The sheet shows what it can either way — which is the
 * canvas's own arrangement (feature 2.9's folded reading reads the tree, not the analysis) — so a
 * reading that refuses answers nothing rather than taking the panel down with it.
 */
function read<T>(reading: () => T, fallback: T): T {
  try {
    return reading();
  } catch {
    return fallback;
  }
}

/** The whole sheet of one site. */
function SitePanel({
  one,
  site,
  context,
  artifact,
  identities,
}: {
  one: OpenDocument;
  site: SiteDescription;
  context: FormContext;
  artifact: ArgumentSchema | null | undefined;
  identities: readonly IdentityReading[];
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
        one={one}
        site={site}
        identities={identities}
      />
      <Slots
        title="Constants"
        rows={sheet.constants}
        note={text('This primitive declares no constant slot.')}
        one={one}
        site={site}
        identities={identities}
      />
      <States rows={sheet.states} one={one} site={site} identities={identities} />
      <Derived sheet={sheet} stale={stale} />
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
 * §4.11's sheet of a **multi-selection**: the intersection of the selected sites' editable rows.
 *
 * > several selected instances show the intersection of their editable rows (primitive, families,
 * > a common argument) with "multiple values" where they differ.
 *
 * The rows are §4.12's own — the same generated rows, the same widgets, the same modes — because
 * they *are* the same rows: one per site, matched by argument path (`multi.ts`). An edit made on
 * one of them is made on every site it stands for, as one command with one undo (`editRows`).
 *
 * What is **not** here, and why: the name (every site has its own, and a shared one would be a
 * rename of several things at once), the ports, the slots and the states (each is about one
 * instance's bindings and §4.11 lists none of them in the intersection), and the derived rows
 * (a figure of several sites is a sum nobody asked for — §4.18's panel is where totals live).
 */
function MultiPanel({
  one,
  sites,
  context,
}: {
  one: OpenDocument;
  sites: readonly SiteDescription[];
  context: FormContext;
}): JSX.Element {
  const [inapplicable, setInapplicable] = useState(false);
  const [problemsFirst, setProblemsFirst] = useState(false);
  const outline = outlineFor(one);
  const sheets = sites.map((site) => ({
    site,
    sheet: argumentSheet({
      facts: site.arguments.facts,
      invariants: site.arguments.invariants,
      tree: one.session.store.tree,
      role: one.session.store.role,
      context,
      shapes: context.shapes,
      showInapplicable: inapplicable,
      problemsFirst,
    }),
  }));
  const multi = multiSheet(sheets);
  return (
    <>
      <h2 className="insp-title">{textWith('{} selected', String(multi.count))}</h2>
      <p className="insp-kind" data-selected={multi.names.join(' ')}>
        {multi.names.join(', ')}
      </p>
      <p className="prim-chip mono" data-multi-primitive={multi.primitive ?? ''}>
        {multi.primitive === null
          ? text('multiple values')
          : `${multi.primitive}@${multi.version ?? text('multiple values')}`}
      </p>
      <h3 className="ih">{text('Families')}</h3>
      <div className="frow" data-member="families">
        <span className="fk">{text('in common')}</span>
        <span className="fv">
          {multi.families.length === 0 ? (
            <span className="dim">{text('none in common')}</span>
          ) : (
            multi.families.map((family) => (
              <em key={family} className="fchip">
                {family}
              </em>
            ))
          )}
          {multi.familiesDiffer ? <span className="dim"> {text('multiple values')}</span> : null}
        </span>
      </div>
      <h2 className="ih">
        {text('Arguments')}
        <span className="ihn">
          {textWith('{} in common', `${String(multi.rows.length)}`)}
        </span>
      </h2>
      {multi.rows.map((row) => (
        <ArgumentRowView
          key={row.path}
          one={one}
          row={row.row}
          also={row.also}
          differs={row.differs}
          artifact={undefined}
          outline={outline}
          context={context}
        />
      ))}
      <div className="arow more">
        {multi.dropped === 0
          ? text('every argument is in common')
          : textWith('{} arguments are not in common', String(multi.dropped))}
        <button
          type="button"
          className="link"
          data-sheet="inapplicable"
          aria-pressed={inapplicable}
          onClick={() => {
            setInapplicable(!inapplicable);
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
            setProblemsFirst(!problemsFirst);
          }}
        >
          {text('Sort problems first')}
        </button>
      </div>
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
          tokens={prefixTokens(site.key)}
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
  tokens,
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
  /** What §4.14's token editor offers at this site — the indices it sits under (the core's). */
  tokens: TokenOffers;
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
  // §4.14's last sentence: "a `weights_location_prefix` on a template instance is the same token
  // editor". One binding at `physical_name`, one editor, wherever the grammar writes one.
  if (editsTokens(row.widget)) {
    return (
      <div className="frow wide" data-member={row.label} data-editor={row.widget}>
        <span className="fk">{label}</span>
        <span className="fv">
          <TokenList
            at={[...base, ...row.steps] as Path}
            shape={shapeAt(context.shapes, [...base, ...row.steps] as Path, one.session.store.role)}
            context={context}
            value={nodeAt(one.session.store.tree, [...base, ...row.steps] as Path)}
            label={label}
            offers={tokens}
          />
        </span>
      </div>
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
  also,
  differs,
  artifact,
  outline,
  context,
}: {
  one: OpenDocument;
  row: ArgumentRow;
  /** The same row at the other selected sites — §4.11's intersection; empty for one selection. */
  also?: readonly ArgumentRow[];
  /** Whether the selected sites write different values here: §4.11's "multiple values". */
  differs?: boolean;
  artifact: ArgumentSchema | undefined;
  outline: readonly OutlineRow[];
  context: FormContext;
}): JSX.Element {
  const store = useDocumentsStore();
  const rows = [row, ...(also ?? [])];
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
          also={also ?? []}
          differs={differs === true}
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
                editRows(store, rows, (made, each) => pinValue(made, each, facts));
              } else {
                editRows(store, rows, (made, each) => clearValue(made, each));
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
              editRows(store, rows, (made, each) => writeValue(made, each, next));
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
  also,
  differs,
  facts,
  outline,
  context,
  onOpen,
}: {
  one: OpenDocument;
  row: ArgumentRow;
  /** The same row at the other selected sites (§4.11); an edit here reaches every one of them. */
  also: readonly ArgumentRow[];
  /** Whether those sites write different values: §4.11's "multiple values". */
  differs: boolean;
  facts: SchemaFacts;
  outline: readonly OutlineRow[];
  context: FormContext;
  onOpen: () => void;
}): JSX.Element {
  const store = useDocumentsStore();
  const rows = [row, ...also];
  const literal = literalMode(row.modes);
  const referring = referringModes(row.modes).find((mode) => mode.tag === row.mode);
  const label = textWith('Value of {}', row.path);

  // §4.11's own state: the selected sites do not agree, so the column says so rather than showing
  // one of them. What the *edit* does is unchanged — a value typed here is written to all of them
  // — which is why the row stays editable and only the reading of it changes.
  if (differs) {
    return (
      <span className="av" data-value={row.path} data-multiple="true">
        <button
          type="button"
          className="link"
          aria-label={label}
          onClick={() => {
            editRows(store, rows, (made, each) => clearValue(made, each));
          }}
        >
          {text('multiple values')}
        </button>
      </span>
    );
  }

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
            editRows(store, rows, (made, each) => writeMode(made, each, referring, event.target.value));
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
            editRows(store, rows, (made, each) => writeLiteral(made, each, event.target.value, facts));
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
            editRows(store, rows, (made, each) => writeLiteral(made, each, event.target.checked, facts));
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
            editRows(store, rows, (made, each) => writeLiteral(made, each, held, facts));
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
          editRows(store, rows, (made, each) => writeLiteral(made, each, event.target.value, facts));
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
  one,
  site,
  identities,
}: {
  title: string;
  rows: readonly SlotRow[];
  note: string;
  one: OpenDocument;
  site: SiteDescription;
  identities: readonly IdentityReading[];
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
          {row.present && row.kind === 'parameter' ? (
            <SlotActions one={one} site={site} row={row} identities={identities} state={false} />
          ) : null}
        </div>
      ))}
    </>
  );
}

/**
 * The gestures §4.11 puts beside a slot row: **Bind privately**, **Tie to…** / **Share with…**,
 * and **Edit location…** — the link to the identity's own sheet, where §4.14's editor stands.
 *
 * | What it shows | Answered by |
 * |---|---|
 * | which identities the slot may join | `describe`'s compatibility lists — V15 for a parameter, V9 for a state (§5.3) |
 * | which identities the document has at all | the core's readings of its binding rules (`identityReadings`) |
 * | what joining one would be refused with | `check` on that candidate, in the validator's words |
 * | where the identity is written | the reading's own pointer — the place the *author* wrote (§5.2 rule 7) |
 *
 * **Every identity is offered, and the compatible ones are marked.** §4.14 asks for "the list and
 * the reason a candidate is excluded", and Q5 forbids the editor to refuse a gesture for a
 * semantic reason: so the list is the document's own identities, the mark is the core's
 * compatibility list, and picking an unmarked one makes the tie and shows `check`'s line in the
 * toast — where the validation that follows repeats it, with its pointer, in Problems.
 *
 * **The lists cost a call, and it is made when the button is pressed.** Feature 2.10 measured
 * `describe` with the compatibility walk at 48.0 ms against 4.5 without it on the largest corpus
 * document and left them off the per-keystroke branch; this is the one caller that asks for them,
 * for its one site (`partnersOf`).
 */
function SlotActions({
  one,
  site,
  row,
  identities,
  state,
}: {
  one: OpenDocument;
  site: SiteDescription;
  row: SlotRow | StateRow;
  identities: readonly IdentityReading[];
  state: boolean;
}): JSX.Element {
  const store = useDocumentsStore();
  const [open, setOpen] = useState(false);
  const [partners, setPartners] = useState<readonly CompatibleIdentity[] | null>(null);
  const target: SlotTarget = { site: site.key, slot: row.name, state };
  const held: HeldMember | null =
    row.boundBy === null
      ? null
      : { rule: pathOfRule(identities, row.boundBy), at: row.boundAt };
  const kind = state ? 'states' : 'parameters';
  // Every identity of the same kind the document declares, in the order its rules are written.
  const offered = identities.filter((reading) => reading.kind === kind);
  const marked = new Set((partners ?? []).map((one_) => one_.rule));

  const tie = (into: IdentityReading): void => {
    // The verdict is asked **before** the gesture is made and never blocks it (Q5): `check` reads
    // the analysis the session holds, so a question sent after the edit would be about a document
    // that already carries it. The answer lands in the toast; the next validation puts the same
    // line in Problems, with its pointer.
    const asked = store.getState().checkMember(
      {
        kind: state ? 'state' : 'parameter',
        slot: { site: site.key, name: row.name },
        into: { identity: into.identity },
      },
      one.id,
    );
    try {
      store.getState().edit((made) => tieTo(made, { target, held, into: pathOf(into.pointer) }));
    } catch (error) {
      store.getState().setToast({ text: error instanceof EditError ? error.message : String(error) });
      return;
    }
    setOpen(false);
    void asked.then((verdict) => {
      const first = verdict?.problems[0];
      if (first !== undefined) store.getState().setToast({ text: `[${first.code}] ${first.message}` });
    });
  };

  return (
    <span className="slotacts" data-slot-actions={row.name}>
      <button
        type="button"
        className="rowact"
        data-bind-private={row.name}
        // A slot already alone in its identity is already bound privately: D3 and D4 answer how
        // many members the identity holds, and nothing is offered that would do nothing.
        disabled={row.boundBy !== null && row.members === 1}
        title={text('Creates an identity of its own, named after the site and the slot.')}
        onClick={() => {
          try {
            store.getState().edit((made) => bindPrivately(made, { target, held }));
          } catch (error) {
            store.getState().setToast({
              text: error instanceof EditError ? error.message : String(error),
            });
          }
        }}
      >
        {text('Bind privately')}
      </button>
      <button
        type="button"
        className="rowact"
        data-tie-open={row.name}
        aria-expanded={open}
        onClick={() => {
          const next = !open;
          setOpen(next);
          if (!next) return;
          void store.getState().partnersOf(site.where, one.id).then((described) => {
            const found = described === undefined || described === null
              ? []
              : state
                ? (described.states.find((each) => each.name === row.name)?.sharesWith ?? [])
                : (described.parameters.find((each) => each.name === row.name)?.tiesWith ?? []);
            setPartners(found);
          });
        }}
      >
        {state ? text('Share with…') : text('Tie to…')}
      </button>
      {row.boundBy === null ? null : (
        <button
          type="button"
          className="rowact"
          data-edit-identity={row.name}
          onClick={() => {
            store.getState().revealPlace(pathOfRule(identities, row.boundBy as string), one.id);
          }}
        >
          {state ? text('Edit identity…') : text('Edit location…')}
        </button>
      )}
      {!open ? null : (
        <span className="tielist" data-tie-list={row.name}>
          {offered.length === 0 ? (
            <span className="dim">{text('This document declares no identity of that kind.')}</span>
          ) : null}
          {offered.map((reading) => (
            <button
              key={reading.pointer}
              type="button"
              className={marked.has(reading.rule) ? 'rowact fit' : 'rowact'}
              data-tie-into={reading.rule}
              data-compatible={String(marked.has(reading.rule))}
              disabled={reading.rule === row.boundBy}
              onClick={() => {
                tie(reading);
              }}
            >
              {reading.identity}
            </button>
          ))}
          {partners === null ? <span className="dim">{text('asking the core…')}</span> : null}
        </span>
      )}
    </span>
  );
}

/** The place a rule is written at, as a path — the core's own reading of §5.2 rule 7. */
function pathOfRule(identities: readonly IdentityReading[], rule: string): Path {
  const reading = identities.find((one) => one.rule === rule);
  return reading === undefined ? [] : pathOf(reading.pointer);
}

/** A JSON pointer as the path the store's commands take. */
function pathOf(pointer: string): Path {
  return pointer
    .split('/')
    .slice(1)
    .map((step) => step.replace(/~1/g, '/').replace(/~0/g, '~'));
}

/** §4.11's States: the applying rule and everything it decided. */
function States({
  rows,
  one,
  site,
  identities,
}: {
  rows: readonly StateRow[];
  one: OpenDocument;
  site: SiteDescription;
  identities: readonly IdentityReading[];
}): JSX.Element {
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
            {row.present ? (
              <SlotActions one={one} site={site} row={row} identities={identities} state />
            ) : null}
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

/**
 * The same edit on **every** selected site — §4.11's intersection sheet, in one command (D13).
 *
 * §4.11: "several selected instances show the intersection of their editable rows … with
 * *multiple values* where they differ". An edit made on such a row is one gesture, so it is one
 * command with one undo: the row of each site is the same row at its own place (`ArgumentRow.at`),
 * and each command is computed against the tree as it stands before any of them ran — which is
 * sound here because the places are disjoint, one per instance.
 */
function editRows(
  store: ReturnType<typeof useDocumentsStore>,
  rows: readonly ArgumentRow[],
  make: (context: EditContext, row: ArgumentRow) => Command,
): void {
  edit(store, (context) => {
    const commands = rows.map((row) => make(context, row));
    const first = commands[0];
    if (first === undefined) throw new EditError('nothing to edit');
    if (commands.length === 1) return first;
    return {
      label: `${first.label} — ${String(commands.length)} sites`,
      moves: commands.flatMap((command) => [...command.moves]),
      edit(draft) {
        for (const command of commands) command.edit(draft);
      },
    };
  });
}
