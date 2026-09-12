/**
 * The expression editor — plan §4.13, artboard S7.
 *
 * Two views of one value, on both sides of the language: a **tree** whose rows are the
 * productions of `language.ts` and whose selects are the schemas' own enumerations, and a
 * **text** field read by the strict parser. Both write the same tagged JSON through one callback,
 * so the caller turns a gesture into a named command (D13) and this component holds no store.
 *
 * **What it shows beside the value** is what the core answers: `resolve` is the caller's binding
 * to the core's evaluator (`modelValue`, feature 1.2), and the effective type is the *declared*
 * type of the place — a quantity's own `type.kind`, an argument's `ArgumentFact.kind` — handed in
 * as data by whoever opened the editor. Nothing here evaluates anything or decides what a type
 * is; §4.13's "the editor merely shows the effective type" is a rule about this file.
 *
 * **No gesture is refused** (Q5). A half-typed text keeps the caret and puts its refusal under the
 * field; a wrap writes the blank the grammar accepts and lets V3 say what it thinks of it. The
 * value is written back only when the text reads, which is what stops a keystroke from destroying
 * the document — and the tree goes on editing the last value that did.
 */
import { useMemo, useState, type JSX } from 'react';

import { serialize, type JsonValue } from '@tensorspine/lang';

import { text, textWith } from '../shell/strings.js';

import { grammarAt, type LanguageContext, type Production } from './language.js';
import { parseAt } from './parse.js';
import { printAt } from './print.js';
import {
  blankFor,
  productionLabel,
  replaceAt,
  replacementsAt,
  rowsOf,
  unwrapTo,
  withMoreOperands,
  withOperator,
  withoutOperand,
  withPayload,
  wrapIn,
  wrappersAt,
  unwrapsTo,
  type TreeRow,
} from './tree.js';

/**
 * What is being typed, and the text it was typed over.
 *
 * The second half is what makes a draft expire: a value the document has moved past — an undo, a
 * gesture on the canvas, an edit in the tree — no longer prints as `from`, and the field follows
 * the document rather than showing a text that belongs to a value nobody holds.
 */
interface Draft {
  readonly text: string;
  readonly from: string;
}

/** What the editor is opened with. */
export interface ExpressionEditorProps {
  /** Which of §4.13's four places the value stands at — the core names them (`languageAnchors`). */
  readonly anchor: string;
  /** The value, as the document's tree holds it; `undefined` where nothing is written. */
  readonly value: JsonValue | undefined;
  /** The schemas, their reading and the bindings. */
  readonly context: LanguageContext;
  /** What the editor is called, for the announcements every control carries. */
  readonly label: string;
  /** The names a picker offers for a referent (`quantity`, `index`, `argument`). */
  readonly names?: (referent: string) => readonly string[];
  /** What the core resolves a value to, printed; `null` where it does not resolve. */
  readonly resolve?: (value: JsonValue, anchor: string) => string | null;
  /** The declared type of the place, as the document or the declaration writes it. */
  readonly type?: string;
  /** Writes the value back. The caller makes it a command; this component never touches a store. */
  readonly onChange: (value: JsonValue) => void;
}

/** The editor of one expression or one condition. */
export function ExpressionEditor(props: ExpressionEditorProps): JSX.Element {
  const { anchor, context, label, onChange } = props;
  const value = props.value ?? blankAtAnchor(context, anchor);
  const printed = printAt(context, anchor, value);
  // The field holds a draft **only while one is being typed**, and only while the value it was
  // typed over still stands: `null` means it follows the value, and a draft whose `from` no longer
  // prints is one the document has moved past — an undo, a canvas gesture, a tree edit. Keeping
  // the two apart this way rather than synchronising them is what stops an edit from leaving the
  // text behind, which this feature's own browser suite caught before it shipped.
  const [draft, setDraft] = useState<Draft | null>(null);
  const [selected, setSelected] = useState<string>('');
  const shown = draft !== null && draft.from === printed ? draft.text : printed;

  const parsed = useMemo(() => parseAt(context, anchor, shown), [context, anchor, shown]);
  const rows = useMemo(() => rowsOf(context, anchor, value), [context, anchor, value]);
  const resolved = props.resolve?.(value, anchor) ?? null;

  const write = (next: JsonValue): void => {
    setDraft(null);
    onChange(next);
  };

  const commit = (): void => {
    if (shown === printed) {
      setDraft(null);
      return;
    }
    if (parsed.value === undefined) return;
    write(parsed.value);
  };

  return (
    <div className="xed" data-expression={anchor}>
      <div className="xtree" role="group" aria-label={textWith('Expression tree of {}', label)}>
        {rows.map((row) => (
          <TreeLine
            key={row.path.join('.')}
            row={row}
            props={props}
            value={value}
            selected={selected === row.path.join('.')}
            onSelect={() => {
              setSelected(row.path.join('.'));
            }}
            onWrite={write}
          />
        ))}
      </div>
      <input
        className="xtext"
        data-text={anchor}
        aria-label={textWith('Expression text of {}', label)}
        value={shown}
        spellCheck={false}
        onChange={(event) => {
          setDraft({ text: event.target.value, from: printed });
        }}
        onBlur={commit}
        onKeyDown={(event) => {
          if (event.key === 'Enter') {
            event.preventDefault();
            commit();
          }
          if (event.key === 'Escape') {
            event.preventDefault();
            setDraft(null);
          }
        }}
      />
      {shown !== printed && parsed.value === undefined ? (
        <div className="rowmsg bad" data-refusal={anchor} role="status">
          {parsed.problems.map((one) => one.message).join('; ')}
        </div>
      ) : null}
      <div className="xfacts" data-facts={anchor}>
        {props.type === undefined ? null : (
          <span data-type={anchor}>{textWith('Effective type {}', props.type)}</span>
        )}
        {resolved === null ? null : (
          <span data-resolved={anchor}>{textWith('resolves to {}', resolved)}</span>
        )}
      </div>
      <details className="xjson-holder">
        <summary>{text('What it stores')}</summary>
        <pre className="xjson" data-json={anchor}>
          {serialize(value)}
        </pre>
      </details>
    </div>
  );
}

/** One row of the tree — S7's `.xn`, with the controls an artboard draws as spans. */
function TreeLine({
  row,
  props,
  value,
  selected,
  onSelect,
  onWrite,
}: {
  row: TreeRow;
  props: ExpressionEditorProps;
  value: JsonValue;
  selected: boolean;
  onSelect: () => void;
  onWrite: (next: JsonValue) => void;
}): JSX.Element {
  const { anchor, context, label } = props;
  const at = row.path.join('.');
  const where = row.path.length === 0 ? label : `${label} ${at}`;
  const resolved = props.resolve?.(row.value, row.anchor) ?? null;
  const replace = (next: JsonValue): void => {
    onWrite(replaceAt(context, anchor, value, row.path, next));
  };

  return (
    <div
      className={`xn${selected ? ' sel' : ''}`}
      style={{ marginLeft: `${String(row.depth * 18)}px` }}
      data-row={at}
      data-kind={row.kind}
      onFocusCapture={onSelect}
    >
      {row.keyword === undefined ? null : <span className="xkw">{row.keyword}</span>}
      <span className={`kind ${chipOf(row)}`}>{row.label}</span>
      {row.operators.length > 0 ? (
        <select
          className="ctl sel mono"
          data-operator={at}
          aria-label={textWith('Operator of {}', where)}
          value={row.operator ?? ''}
          onChange={(event) => {
            replace(withOperator(context, row.anchor, row.value, event.target.value));
          }}
        >
          {row.operators.map((one) => (
            <option key={one} value={one}>
              {one}
            </option>
          ))}
        </select>
      ) : null}
      {row.named ? <NamePicker row={row} props={props} where={where} onReplace={replace} /> : null}
      {row.payload && !row.named ? (
        <Literal row={row} props={props} where={where} onReplace={replace} />
      ) : null}
      {resolved === null ? null : <span className="xval">{resolved}</span>}
      {selected ? <Actions row={row} props={props} where={where} onReplace={replace} /> : null}
    </div>
  );
}

/** The reference picker of S7: "references are picked, never typed". */
function NamePicker({
  row,
  props,
  where,
  onReplace,
}: {
  row: TreeRow;
  props: ExpressionEditorProps;
  where: string;
  onReplace: (next: JsonValue) => void;
}): JSX.Element {
  const held = row.written === undefined || row.written === null ? '' : String(row.written);
  const offered = row.referent === undefined ? [] : (props.names?.(row.referent) ?? []);
  if (offered.length === 0) {
    // Nothing to pick from — a unit's declarations before the primitive editor exists, an index
    // named outside every composition — so the name is typed, and the core's own refusal judges
    // it (Q5). Written on blur or Enter, so that a name is one command and not one per keystroke.
    return <TypedName row={row} props={props} where={where} onReplace={onReplace} held={held} />;
  }
  return (
    <select
      className="ctl sel mono"
      data-name={row.path.join('.')}
      aria-label={textWith('Name of {}', where)}
      value={held}
      onChange={(event) => {
        onReplace(withPayload(props.context, row.anchor, row.value, event.target.value));
      }}
    >
      {offered.includes(held) ? null : <option value={held}>{held}</option>}
      {offered.map((one) => (
        <option key={one} value={one}>
          {one}
        </option>
      ))}
    </select>
  );
}

/** A name where nothing offers a list to pick from: typed, and written when it is finished. */
function TypedName({
  row,
  props,
  where,
  held,
  onReplace,
}: {
  row: TreeRow;
  props: ExpressionEditorProps;
  where: string;
  held: string;
  onReplace: (next: JsonValue) => void;
}): JSX.Element {
  const [draft, setDraft] = useState<Draft | null>(null);
  const shown = draft !== null && draft.from === held ? draft.text : held;
  const commit = (): void => {
    setDraft(null);
    if (shown === held) return;
    onReplace(withPayload(props.context, row.anchor, row.value, shown));
  };
  return (
    <input
      className="ctl mono"
      data-name={row.path.join('.')}
      aria-label={textWith('Name of {}', where)}
      value={shown}
      spellCheck={false}
      onChange={(event) => {
        setDraft({ text: event.target.value, from: held });
      }}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key !== 'Enter') return;
        event.preventDefault();
        commit();
      }}
    />
  );
}

/**
 * A literal's own field — and the row's "replace" as well.
 *
 * What is typed is read by the same parser at the same place, so `4096` is the literal and `d` is
 * the quantity: §4.13's "a node can be wrapped, unwrapped, or **replaced**" costs no second
 * control and no second reading of what a value may be.
 */
function Literal({
  row,
  props,
  where,
  onReplace,
}: {
  row: TreeRow;
  props: ExpressionEditorProps;
  where: string;
  onReplace: (next: JsonValue) => void;
}): JSX.Element {
  const printed = printAt(props.context, row.anchor, row.value);
  const [draft, setDraft] = useState<Draft | null>(null);
  const shown = draft !== null && draft.from === printed ? draft.text : printed;
  const commit = (): void => {
    if (shown === printed) {
      setDraft(null);
      return;
    }
    const parsed = parseAt(props.context, row.anchor, shown);
    if (parsed.value === undefined) return;
    setDraft(null);
    onReplace(parsed.value);
  };
  return (
    <input
      className="ctl mono"
      data-literal={row.path.join('.')}
      aria-label={textWith('Value of {}', where)}
      value={shown}
      spellCheck={false}
      onChange={(event) => {
        setDraft({ text: event.target.value, from: printed });
      }}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key === 'Enter') {
          event.preventDefault();
          commit();
        }
      }}
    />
  );
}

/** What the selected row offers: wrap, unwrap, replace, and the operands the schema admits. */
function Actions({
  row,
  props,
  where,
  onReplace,
}: {
  row: TreeRow;
  props: ExpressionEditorProps;
  where: string;
  onReplace: (next: JsonValue) => void;
}): JSX.Element {
  const { context } = props;
  const offered = replacementsAt(context, row.anchor);
  // Only the productions that can *hold* the node where it stands, so a gesture never leaves the
  // document off the grammar (D5): a conditional takes an expression in its `then`, a comparison
  // takes none at the condition anchor at all.
  const wrappers = wrappersAt(context, row.anchor);
  const unwraps = unwrapsTo(context, row.anchor, row.value) !== null;
  return (
    <span className="xacts">
      <select
        className="ctl sel"
        data-wrap={row.path.join('.')}
        aria-label={textWith('Wrap {} in an operator', where)}
        value=""
        onChange={(event) => {
          const production = wrappers.find((one) => productionLabel(one) === event.target.value);
          if (production === undefined) return;
          onReplace(wrapIn(context, row.anchor, row.value, production));
        }}
      >
        <option value="">{text('wrap in…')}</option>
        {wrappers.map((one) => (
          <option key={productionLabel(one)} value={productionLabel(one)}>
            {productionLabel(one)}
          </option>
        ))}
      </select>
      {unwraps ? (
        <button
          type="button"
          className="btn ghost"
          data-unwrap={row.path.join('.')}
          onClick={() => {
            onReplace(unwrapTo(context, row.anchor, row.value));
          }}
        >
          {text('unwrap')}
        </button>
      ) : null}
      <select
        className="ctl sel"
        data-replace={row.path.join('.')}
        aria-label={textWith('Replace {}', where)}
        value=""
        onChange={(event) => {
          const production = offered.find((one) => productionLabel(one) === event.target.value);
          if (production === undefined) return;
          onReplace(startedFrom(props, context, row.anchor, production));
        }}
      >
        <option value="">{text('replace with…')}</option>
        {offered.map((one) => (
          <option key={productionLabel(one)} value={productionLabel(one)}>
            {productionLabel(one)}
          </option>
        ))}
      </select>
      {row.extensible ? (
        <button
          type="button"
          className="btn ghost"
          data-add={row.path.join('.')}
          onClick={() => {
            onReplace(withMoreOperands(context, row.anchor, row.value));
          }}
        >
          {text('+ arg')}
        </button>
      ) : null}
      {row.reducible ? (
        <button
          type="button"
          className="btn ghost"
          data-drop={row.path.join('.')}
          onClick={() => {
            onReplace(withoutOperand(context, row.anchor, row.value, row.operands - 1));
          }}
        >
          {text('− arg')}
        </button>
      ) : null}
      {row.payload ? null : <span className="xarity">{arityText(row)}</span>}
    </span>
  );
}

/**
 * The value a replacement starts from: the grammar's own blank, with a name the reader has.
 *
 * `blankFor` writes a placeholder where a reference is wanted, because the schema answers no name
 * (its `identifier` is a pattern, and the empty string is not one). Where a picker *does* offer
 * names — the document's quantities, its indices — the first of them is a better start and is
 * still the reader's to change; where it offers none, the placeholder stands and the core reports
 * it for what it is.
 */
function startedFrom(
  props: ExpressionEditorProps,
  context: LanguageContext,
  anchor: string,
  production: Production,
): JsonValue {
  const blank = blankFor(context, anchor, production);
  const referent = production.payload?.referent;
  if (referent === undefined) return blank;
  const first = props.names?.(referent)[0];
  return first === undefined ? blank : withPayload(context, anchor, blank, first);
}

/** The blank value at an anchor, for a place nothing is written at yet. */
function blankAtAnchor(context: LanguageContext, anchor: string): JsonValue {
  const first = grammarAt(context, anchor).productions[0];
  return first === undefined ? null : blankFor(context, anchor, first);
}

/**
 * Which of the design's four `.kind` colours a row carries.
 *
 * Read from the *production*, never from the word it writes: a payload whose place binds a prefix
 * is the kind of name that prefix tells apart (`$layer`), another named payload is a reference,
 * an unnamed one is a literal, and everything else applies an operator. The classes are
 * `_ts.css`'s own and say nothing about the language.
 */
function chipOf(row: TreeRow): string {
  if (row.kind === '') return '';
  if (!row.payload) return 'op';
  if (!row.named) return 'lit';
  return row.prefix === '' ? 'q' : 'idx';
}

/** What the schema admits of a row's arity, as S7 writes it beside the blank operand. */
function arityText(row: TreeRow): string {
  if (row.least === row.most) return textWith('exactly {} — the schema’s bounds', String(row.least));
  if (row.most === Infinity) return textWith('{} or more — the schema’s bounds', String(row.least));
  return textWith('{} — the schema’s bounds', `${String(row.least)} to ${String(row.most)}`);
}
