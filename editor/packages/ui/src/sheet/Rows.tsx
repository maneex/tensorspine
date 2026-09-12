/**
 * A generated form, drawn — the rows every sheet of plan §4.11 is made of.
 *
 * Feature 2.10 drew the Identity section of an instance this way and stopped at its depth; the
 * other sheets of §4.11 are the same walk one level deeper, over other `$def`s: a composition's
 * `indices` is a **map** of index ranges whose bounds are expressions, an edge's `from` is a
 * **section** holding a **chooser** of selectors, an identity's `members` is a **list** of
 * sections, a quantity's `type`, `domain` and `source` are three choosers. So this module renders
 * a {@link FormRow} of *any* widget, recursively, and the sheets below it choose nothing but the
 * place to start at.
 *
 * **Every gesture writes what the schema admits and nothing else.** A chooser writes the blank of
 * the alternative it was set to ({@link blankValue}); a list appends the blank of its item shape;
 * a map takes a name and writes the blank of its value shape; a scalar writes what its widget
 * yields. D5 is the rule these follow — the document stays on the grammar and what is *missing*
 * is the semantic stage's report — and Q5 is why nothing is refused: a value the core will
 * refuse is written, and its refusal lands on the row and in Problems.
 *
 * **Nothing here names a member of the grammar.** The label of a row is the schema's (`title`,
 * failing that the member's own name), the alternatives of a chooser are the walker's, the
 * options of a select are the schema's `enum`, and the editor of an expression is the one
 * `presentation.json` binds. Catching rule §1 (b) is a scan that says so.
 */
import { useState, type JSX } from 'react';

import {
  isJsonNumber,
  isJsonObject,
  type JsonValue,
  type SchemaFacts,
  type VocabularyValue,
} from '@tensorspine/lang';
import {
  EditError,
  insertItem,
  nodeAt,
  remove,
  setMemberAt,
  setValue,
  type Command,
  type EditContext,
  type Path,
  type Shape,
} from '@tensorspine/store';

import { useDocumentsStore } from '../documents/context.js';
import type { OpenDocument } from '../documents/store.js';
import { NAME_FIELD } from '../explorer/Explorer.js';
import {
  blankAt as blankExpression,
  documentResolver,
  editsExpression,
  ExpressionEditor,
  printAt,
  unionAnchorOf,
  type Resolver,
} from '../expressions/index.js';
import {
  CHOOSER,
  editsOneValue,
  factsOfShape,
  FIXED,
  formOf,
  LIST,
  MAP,
  SCALAR,
  SECTION,
  type Form,
  type FormContext,
  type FormMode,
  type FormRow,
} from '../forms/index.js';
import { text, textWith } from '../shell/strings.js';
import { blankValue, itemShape, memberShape, valueShape } from './skeleton.js';
import { shapeAt } from './places.js';

/** What every row below is drawn with. */
export interface RowsProps {
  readonly one: OpenDocument;
  /** Where the form starts in the document: a row's place is this and the row's own steps. */
  readonly base: Path;
  readonly form: Form;
  readonly context: FormContext;
  /** The names a picker at a place offers — the version list, the family list (feature 2.10). */
  readonly picks?: (row: FormRow) => readonly string[];
  /** What §4.13's pickers offer inside an expression editor: the quantities, the indices in scope. */
  readonly names?: (referent: string) => readonly string[];
  /** Only the rows under this one; the top of the form otherwise. */
  readonly under?: FormRow;
}

/** What nothing at all is written as — `view.py`'s own em dash. */
const EMPTY = '—';

/** The rows of a generated form at one level, each drawing its own children. */
export function FormRows(props: RowsProps): JSX.Element {
  const { form, under } = props;
  const depth = under === undefined ? 1 : under.depth + 1;
  const prefix = under === undefined ? '' : under.path;
  const rows = form.rows.filter(
    (row) => row.depth === depth && row.path.startsWith(`${prefix}/`),
  );
  return (
    <>
      {rows.map((row) => (
        <GeneratedRow key={row.path} {...props} row={row} />
      ))}
    </>
  );
}

/** One row of a generated form, drawn by what its widget is. */
function GeneratedRow(props: RowsProps & { row: FormRow }): JSX.Element {
  const { one, base, row, context } = props;
  const store = useDocumentsStore();
  const at = [...base, ...row.steps] as Path;
  const shape = shapeAt(context.shapes, at, one.session.store.role);
  const label = row.label;
  const title = [row.description, row.error].filter((held) => held !== undefined).join('\n');
  const classes = ['frow', row.error === undefined ? '' : 'errrow'].filter((one_) => one_ !== '');

  // A member the document does not write: the row says so and offers what writing it would be.
  if (!row.present) {
    return (
      <div className={classes.join(' ')} data-member={label} data-present="false" title={title || undefined}>
        <span className="fk">
          {label}
          {row.required ? <i className="sbadge req">{text('required')}</i> : null}
        </span>
        <span className="fv none">
          {editsOneValue(row.widget) || row.widget === CHOOSER || row.widget === MAP || row.widget === LIST ? (
            <>
              {EMPTY}
              <button
                type="button"
                className="link"
                data-add-member={row.path}
                aria-label={textWith('Add {}', label)}
                onClick={() => {
                  edit(store, (made) =>
                    writeMember(made, at, blankValue(context, shape), `Add ${label}`),
                  );
                }}
              >
                {text('Add')}
              </button>
            </>
          ) : (
            EMPTY
          )}
        </span>
      </div>
    );
  }

  if (row.widget === SECTION) {
    return (
      <div className="fsect" data-member={label} data-section={row.path}>
        <div className="fsect-head">
          <span>{label}</span>
          <RemoveMember row={row} at={at} label={label} required={row.required} />
        </div>
        <div className="fsect-body">
          <FormRows {...props} under={row} />
        </div>
      </div>
    );
  }

  if (row.widget === MAP) {
    return <MapRow {...props} at={at} shape={shape} />;
  }

  if (row.widget === LIST) {
    return <ListRows {...props} at={at} shape={shape} />;
  }

  if (editsExpression(row.widget)) {
    return <ExpressionRow {...props} at={at} shape={shape} />;
  }

  if (row.widget === CHOOSER) {
    return <ChooserRow {...props} at={at} shape={shape} />;
  }

  if (!editsOneValue(row.widget)) {
    // A place with an editor nobody has written yet (§1's generic widget): the printed value, and
    // the row says which editor it is waiting for.
    const held = nodeAt(one.session.store.tree, at);
    return (
      <div className={classes.join(' ')} data-member={label} data-editor={row.widget}>
        <span className="fk">{label}</span>
        <span className="fv mono">
          {held === undefined
            ? EMPTY
            : printAt(context, unionAnchorOf(context, shape), held)}
        </span>
      </div>
    );
  }

  return (
    <div className={classes.join(' ')} data-member={label} title={title || undefined}>
      <span className="fk">
        {label}
        {row.required ? <i className="sbadge req">{text('required')}</i> : null}
      </span>
      <span className="fv">
        <ScalarField {...props} at={at} shape={shape} />
        <RemoveMember row={row} at={at} label={label} required={row.required} />
      </span>
      {row.error === undefined ? null : <span className="rowmsg">{row.error}</span>}
    </div>
  );
}

/** The × that takes an optional member out again — "defaults are not written" one level up. */
function RemoveMember({
  row,
  at,
  label,
  required,
}: {
  row: FormRow;
  at: Path;
  label: string;
  required: boolean;
}): JSX.Element | null {
  const store = useDocumentsStore();
  if (required || !row.present) return null;
  return (
    <button
      type="button"
      className="tbtn"
      data-remove-member={row.path}
      aria-label={textWith('Remove {}', label)}
      title={textWith('Remove {}', label)}
      onClick={() => {
        edit(store, (made) => remove(made, { path: at, label: `Remove ${label}` }));
      }}
    >
      ×
    </button>
  );
}

/**
 * How much of a map a sheet draws — the one decision this module makes that the schema cannot.
 *
 * A map of a document's own names is one of three things to a sheet, and the difference matters:
 * a composition's `indices` is **part of the declaration being edited** (§4.11: "indices (a
 * repeatable list of `name: start, stop, step` expression rows)"), a composition's `instances` is
 * a set of declarations with **sheets of their own** (the sites, which the canvas and the tree
 * select), and the document's own maps are what the **Document sheet** shows as counts (§4.11:
 * "`model` id · `version` · `primitive_libraries` … · counts", which is what S15 draws).
 *
 * So: at the document's root a map is its **count**; a map whose entries carry a name-keyed map of
 * their own is its entries' **names**, each opening its own sheet; anything else is drawn in full.
 * The middle rule is the one worth stating — an entry that holds a map of names is a declaration
 * large enough to have a sheet (an instance has arguments, a binding rule has a `for_each`), and
 * an entry that holds none is a value of this one (an index range is three expressions).
 */
type MapMode = 'count' | 'names' | 'entries';

function mapMode(context: FormContext, base: Path, shape: Shape): MapMode {
  if (base.length === 0) return 'count';
  const entry = context.shapes.step(shape, ENTRY);
  for (const name of context.shapes.propertyOrder(entry)) {
    if (factsOfShape(context.shapes.member(entry, name)).keyed) return 'names';
  }
  return 'entries';
}

/** Any name steps into a map: what is wanted is the definition its entries are. */
const ENTRY = 'entry';

/**
 * A map whose names are the author's: a composition's `indices`, a rule's `for_each`.
 *
 * §4.11 asks for "a repeatable list of `name: start, stop, step` expression rows", and that is
 * what a map of `index_range`s renders as without a word about indices: each entry is a row with
 * the entry's own form under it, its name editable, and a field that adds another.
 */
function MapRow(props: RowsProps & { row: FormRow; at: Path; shape: Shape }): JSX.Element {
  const { row, at, shape, context, one } = props;
  const store = useDocumentsStore();
  const [adding, setAdding] = useState('');
  const entries = props.form.rows.filter(
    (each) => each.depth === row.depth + 1 && each.path.startsWith(`${row.path}/`),
  );
  // The count is the document's own, not the walk's: a sheet that stops short of a map's entries
  // still says how many it holds, which is what the Document sheet is (§4.11's "counts").
  const held = nodeAt(one.session.store.tree, at);
  const count = held !== undefined && isJsonObject(held) ? held.members.length : entries.length;
  const mode = mapMode(context, props.base, shape);
  return (
    <div className="fsect" data-member={row.label} data-map={row.path} data-mode={mode}>
      <div className="fsect-head">
        <span>{row.label}</span>
        <span className="ihn">{String(count)}</span>
        <RemoveMember row={row} at={at} label={row.label} required={row.required} />
      </div>
      {mode === 'count' ? null : (
        <div className="fsect-body">
          {entries.map((entry) => (
            <div className="fentry" key={entry.path} data-entry={entry.path}>
              <div className="fentry-head">
                {mode === 'names' ? (
                  <button
                    type="button"
                    className="link mono"
                    data-open-entry={entry.path}
                    onClick={() => {
                      store.getState().selectPlace([...at, entry.label] as Path, one.id);
                    }}
                  >
                    {entry.label}
                  </button>
                ) : (
                  <span className="mono">{entry.label}</span>
                )}
                <button
                  type="button"
                  className="tbtn"
                  data-remove-entry={entry.path}
                  aria-label={textWith('Remove {}', entry.label)}
                  onClick={() => {
                    edit(store, (made) =>
                      remove(made, {
                        path: [...at, entry.label] as Path,
                        label: `Remove ${entry.label}`,
                      }),
                    );
                  }}
                >
                  ×
                </button>
              </div>
              {mode === 'names' ? null : <FormRows {...props} under={entry} />}
            </div>
          ))}
          <div className="frow">
            <span className="fk">{text('Add')}</span>
            <span className="fv">
              <input
                className="ctl mono"
                data-add-key={row.path}
                aria-label={textWith('Add to {}', row.label)}
                value={adding}
                onChange={(event) => {
                  setAdding(event.target.value);
                }}
                onKeyDown={(event) => {
                  event.stopPropagation();
                  if (event.key !== 'Enter') return;
                  const name = adding.trim();
                  if (name === '') return;
                  const value = blankValue(context, valueShape(context, shape, name));
                  edit(store, (made) =>
                    setMemberAt(made, { path: at, name, value, label: `Add ${name}` }),
                  );
                  setAdding('');
                  // The new entry is where the author is now working, and the sheet follows it.
                  store.getState().selectPlace([...at, name] as Path, one.id);
                }}
              />
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

/** A list: `to` endpoints, an identity's members, the bases a document resolves from. */
function ListRows(props: RowsProps & { row: FormRow; at: Path; shape: Shape }): JSX.Element {
  const { row, at, shape, context } = props;
  const store = useDocumentsStore();
  const items = props.form.rows.filter(
    (each) => each.depth === row.depth + 1 && each.path.startsWith(`${row.path}/`),
  );
  const least = row.bounds?.minItems ?? 0;
  return (
    <div className="fsect" data-member={row.label} data-list={row.path}>
      <div className="fsect-head">
        <span>{row.label}</span>
        <span className="ihn">{String(items.length)}</span>
        <button
          type="button"
          className="link"
          data-add-item={row.path}
          aria-label={textWith('Add to {}', row.label)}
          onClick={() => {
            edit(store, (made) =>
              insertItem(made, {
                path: at,
                value: blankValue(context, itemShape(context, shape, items.length)),
                label: `Add to ${row.label}`,
              }),
            );
          }}
        >
          {text('Add')}
        </button>
      </div>
      <div className="fsect-body">
        {items.map((item, index) => (
          <div className="fentry" key={item.path} data-entry={item.path}>
            <div className="fentry-head">
              <span className="mono">{String(index)}</span>
              <button
                type="button"
                className="tbtn"
                data-remove-entry={item.path}
                aria-label={textWith('Remove {}', `${row.label} ${String(index)}`)}
                disabled={items.length <= least}
                onClick={() => {
                  edit(store, (made) =>
                    remove(made, {
                      path: [...at, index] as Path,
                      label: `Remove from ${row.label}`,
                    }),
                  );
                }}
              >
                ×
              </button>
            </div>
            {editsOneValue(item.widget) || editsExpression(item.widget) || item.widget === CHOOSER ? (
              <GeneratedRow {...props} row={item} />
            ) : (
              <FormRows {...props} under={item} />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * A union: the chooser of plan §1, labelled by each alternative's discriminating key or `const`.
 *
 * The select writes the **blank of the alternative** — the first value the grammar admits there
 * (D5) — and the alternative's own members are the rows under it, which is how a `location`'s
 * four forms, a selector's two and a dtype's two are edited without any of them being named here.
 */
function ChooserRow(props: RowsProps & { row: FormRow; at: Path; shape: Shape }): JSX.Element {
  const { row, at, shape, context, one } = props;
  const modes = row.modes ?? [];
  const chosen = modes.find((mode) => mode.tag === row.mode);
  const held = nodeAt(one.session.store.tree, at);
  const inline = chosen !== undefined && chosen.inline;
  return (
    <div className="fsect" data-member={row.label} data-chooser={row.path} data-mode={row.mode ?? ''}>
      <div className="fsect-head">
        <span>{row.label}</span>
        <ModeSelect row={row} at={at} shape={shape} context={context} />
        <RemoveMember row={row} at={at} label={row.label} required={row.required} />
      </div>
      {held === undefined && !inline ? null : (
        <div className="fsect-body">
          <FormRows {...props} under={row} />
        </div>
      )}
    </div>
  );
}

/**
 * The select that says which alternative a value is in — a chooser's own control.
 *
 * Shared by the row and by a cell of §4.16's tables, because the two edit the same thing: the
 * options are the alternatives the walker found, the value written is the blank of the one
 * chosen, and no alternative of any union is named here.
 */
export function ModeSelect({
  row,
  at,
  shape,
  context,
}: {
  row: FormRow;
  at: Path;
  shape: Shape;
  context: FormContext;
}): JSX.Element {
  const store = useDocumentsStore();
  const modes = row.modes ?? [];
  return (
    <select
      className="ctl sel mono"
      data-modes={row.path}
      aria-label={textWith('Kind of {}', row.label)}
      value={row.mode ?? ''}
      onChange={(event) => {
        const mode = modes.find((each) => each.tag === event.target.value);
        if (mode === undefined) return;
        edit(store, (made) =>
          writeMember(made, at, blankOfMode(context, shape, mode), `Set ${row.label}`),
        );
      }}
    >
      {row.mode === undefined ? <option value="">{EMPTY}</option> : null}
      {modes.map((mode) => (
        <option key={mode.tag} value={mode.tag}>
          {mode.tag}
        </option>
      ))}
    </select>
  );
}

/**
 * The name row every sheet of §4.11 starts with — §4.4's "a value is edited in the sheet's row".
 *
 * A rename is a command of its own (it rewrites every reference), so the row hands the typed name
 * to its caller rather than writing it: the sheets pass `renameCommand`, which is the explorer's.
 */
export function NameRow({
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

/** A place §4.13's editor owns — a guard, an index bound, a derivation, a domain bound. */
function ExpressionRow(props: RowsProps & { row: FormRow; at: Path; shape: Shape }): JSX.Element {
  const { row, at, shape, context, one } = props;
  const store = useDocumentsStore();
  const anchor = unionAnchorOf(context, shape);
  const value = nodeAt(one.session.store.tree, at);
  return (
    <div className="frow wide" data-member={row.label} data-editor={row.widget}>
      <span className="fk">{row.label}</span>
      <span className="fv">
        <ExpressionEditor
          anchor={anchor}
          value={value}
          context={context}
          label={row.label}
          names={props.names ?? (() => [])}
          resolve={resolverFor(one)}
          onChange={(next) => {
            edit(store, (made) => writeMember(made, at, next, `Set ${row.label}`));
          }}
        />
      </span>
    </div>
  );
}

/** One scalar member: a select where the place enumerates, a toggle, a number, a field. */
export function ScalarField(props: RowsProps & { row: FormRow; at: Path; shape: Shape }): JSX.Element {
  const { row, at, shape } = props;
  const store = useDocumentsStore();
  const facts = factsOfShape(shape);
  const label = row.label;
  const held = row.written;
  const write = (value: JsonValue, what: string): void => {
    edit(store, (made) => writeMember(made, at, value, `Set ${what}`));
  };

  if (row.constant !== undefined || row.widget === FIXED) {
    return (
      <span className="mono" data-member-value={label}>
        {String(row.constant ?? held ?? '')}
      </span>
    );
  }

  // What the **schema** enumerates is a select: its options are the admissible values, and a value
  // outside them is off the grammar.
  if (row.options !== undefined) {
    const shown = held === undefined ? '' : String(held);
    const offered = row.options.map((option) => String(option.value));
    return (
      <select
        className="ctl sel mono"
        data-member-value={label}
        aria-label={label}
        value={shown}
        onChange={(event) => {
          write(event.target.value, label);
        }}
      >
        {offered.includes(shown) ? null : <option value={shown}>{shown === '' ? EMPTY : shown}</option>}
        {offered.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    );
  }

  // What the **editor** offers — the names of the declarations a member refers to (§4.11's
  // "stream (select of the document's streams)") — is a *suggestion*, so it is a list beside a
  // field and not a select: Q5 forbids a gesture refused for a semantic reason, and a name the
  // list does not carry is exactly the case V1 is there to report (S15: "V1 refuses one that
  // *joins unknown stream 'pixels'*"). Feature 2.10 made the same distinction for the families.
  const suggestions = props.picks?.(row) ?? [];
  if (suggestions.length > 0) {
    const listId = `ts-picks-${row.path.replace(/\W/g, '-')}`;
    return (
      <>
        <input
          className="ctl mono"
          data-member-value={label}
          aria-label={label}
          list={listId}
          value={scalarText(held)}
          onChange={(event) => {
            write(event.target.value, label);
          }}
        />
        <datalist id={listId}>
          {suggestions.map((name) => (
            <option key={name} value={name} />
          ))}
        </datalist>
      </>
    );
  }

  // A place that admits several scalar types — the grammar's own `scalar_literal`, which a
  // literal source and a literal expression both are — is one field, and what is written is the
  // value the **text** is: a number where it parses as one, with the fraction or exponent it was
  // typed with deciding its float-ness (D12, V3's lexical rule); a truth where it is one of the
  // two words; the text otherwise. The truth is not tried first, because a place that admits a
  // truth *and* a number is not a checkbox: it would make every other value unwritable.
  if (row.widget === SCALAR) {
    return (
      <input
        className="ctl mono"
        data-member-value={label}
        aria-label={label}
        value={scalarText(held)}
        onChange={(event) => {
          write(scalarOf(event.target.value), label);
        }}
      />
    );
  }

  if (facts.holdsTruth) {
    return (
      <input
        type="checkbox"
        className="tog"
        data-member-value={label}
        aria-label={label}
        checked={held === true}
        onChange={(event) => {
          write(event.target.checked, label);
        }}
      />
    );
  }

  if (facts.holdsWholeNumber || facts.holdsNumber) {
    return (
      <input
        type="number"
        className="ctl mono"
        data-member-value={label}
        aria-label={label}
        step={facts.holdsWholeNumber && !facts.holdsNumber ? 1 : 'any'}
        value={scalarText(held)}
        onChange={(event) => {
          const number = Number(event.target.value);
          if (event.target.value === '' || Number.isNaN(number)) return;
          write(numberFor(number, facts), label);
        }}
      />
    );
  }

  return (
    <input
      className="ctl mono"
      data-member-value={label}
      aria-label={label}
      value={scalarText(held)}
      onChange={(event) => {
        write(event.target.value, label);
      }}
    />
  );
}

/**
 * What a text typed into a place that admits several scalar types is written as.
 *
 * The number's float-ness is the **lexeme's**, which is V3's own lexical rule read forwards: a
 * token with a fraction or an exponent is a real, one without is a whole number. So `32` stays a
 * cardinality and `32.0` does not, and the document round-trips through the serializer unchanged.
 */
function scalarOf(text: string): JsonValue {
  const trimmed = text.trim();
  if (trimmed === TRUE) return true;
  if (trimmed === FALSE) return false;
  if (trimmed !== '' && !Number.isNaN(Number(trimmed))) {
    return { kind: 'number', value: Number(trimmed), real: /[.eE]/.test(trimmed) };
  }
  return text;
}

/** The two words a truth is written as, which are JSON's own. */
const TRUE = 'true';
const FALSE = 'false';

/** A number as the document writes it: whole or real, from what the schema asserts (D12). */
function numberFor(value: number, facts: SchemaFacts): JsonValue {
  const whole = facts.holdsWholeNumber && !facts.holdsNumber;
  return { kind: 'number', value, real: !whole && !Number.isInteger(value) };
}

/**
 * A scalar of a row as a field shows it.
 *
 * The walker puts what the *document* holds on the row, which for a number is the node carrying
 * the lexeme it was written with (feature 0.3, D12): a value the file wrote as `1e-05` is edited
 * as `1e-05` and not as `0.00001`.
 */
function scalarText(value: VocabularyValue | JsonValue | undefined): string {
  if (value === undefined || value === null) return '';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  if (isJsonNumber(value)) return value.lexeme ?? String(value.value);
  return '';
}

/** The blank of one alternative of a union: the expression editor's own, or the walker's. */
function blankOfMode(context: FormContext, shape: Shape, mode: FormMode): JsonValue {
  if (editsExpression(mode.widget)) return blankExpression(context, mode.union);
  return blankValue(context, memberShape(context, shape, mode));
}

/** Write a value at a place the document may not yet have a member for (§5.5's "appended"). */
function writeMember(context: EditContext, at: Path, value: JsonValue, label: string): Command {
  if (nodeAt(context.tree, at) !== undefined) return setValue(context, { path: at, value, label });
  const name = at[at.length - 1];
  return setMemberAt(context, {
    path: at.slice(0, -1),
    name: typeof name === 'string' ? name : String(name ?? ''),
    value,
    label,
  });
}

/** The resolver of one document, memoised on its tree (feature 2.11's own reading). */
let lastResolver: { tree: unknown; resolve: Resolver } | null = null;

function resolverFor(one: OpenDocument): Resolver {
  const tree = one.session.store.tree;
  if (lastResolver !== null && lastResolver.tree === tree) return lastResolver.resolve;
  const resolve = documentResolver(tree);
  lastResolver = { tree, resolve };
  return resolve;
}

/** Run a command against the current document, reporting what it refuses as a toast (Q5). */
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

/** What a form needs of the document to render: exported so a sheet can build one. */
export function formFor(
  one: OpenDocument,
  context: FormContext,
  at: Path,
  options: { readonly limit?: number } = {},
): Form {
  const value = nodeAt(one.session.store.tree, at);
  return formOf(context, {
    shape: shapeAt(context.shapes, at, one.session.store.role),
    ...(value === undefined ? {} : { value }),
    ...(options.limit === undefined ? {} : { limit: options.limit }),
  });
}
