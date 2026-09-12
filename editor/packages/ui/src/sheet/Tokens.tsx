/**
 * The physical-name token editor, drawn — §4.14, artboard S8, inventory §4.
 *
 * S8 draws a name as a row of chips: `model.layers.` `{index layer}` `.self_attn.q_proj.weight`
 * `+ token`. Each chip is one item of the array, in one of the three forms `physical_name`
 * declares, and what this file decides is the same thing every other drawn row decides — that a
 * control is a `<button>`, a `<select>` or an `<input>` and not a `<span>` (feature 2.5's rule) —
 * while {@link tokenModel} decides what a chip *is*.
 *
 * **The chip's colour is the core's fact.** `TokenNames.declared` says whether the names a form
 * carries are the document's declarations (a composition index) or the location's own product (a
 * `stack`'s coordinate); §4.21 gives the first the structural colour and the second the derived
 * one, which is exactly what S8 draws. The interface never reads the member's name to decide it.
 *
 * **The names are a list, never a limit** (Q5, feature 2.12's reading of §4.15): the field carries
 * a `datalist`, so an index the rule does not fire under can be typed and V17 is what says so.
 */
import type { JSX } from 'react';

import type { JsonValue } from '@tensorspine/lang';
import {
  EditError,
  pointerOf,
  type Command,
  type EditContext,
  type Path,
  type Shape,
} from '@tensorspine/store';

import { useDocumentsStore } from '../documents/context.js';
import type { FormContext } from '../forms/index.js';
import { text, textWith } from '../shell/strings.js';
import { writeMember } from './edits.js';
import {
  tokenModel,
  withForm,
  withText,
  withToken,
  withoutToken,
  type TokenOffers,
  type TokenRow,
} from './tokens.js';

/** What the token editor is drawn with. */
export interface TokensProps {
  /** The place of the physical name, as a path of the document. */
  readonly at: Path;
  readonly shape: Shape;
  readonly context: FormContext;
  /** The value written there, or `undefined` where the document writes none yet. */
  readonly value: JsonValue | undefined;
  /** What the core says each form may name (§4.14); empty where it says nothing. */
  readonly offers?: TokenOffers;
  /** What the Edit menu calls the gesture — the member the name is written under. */
  readonly label: string;
}

/** One physical name as S8's row of chips. */
export function TokenList(props: TokensProps): JSX.Element {
  const { at, shape, context, value, label } = props;
  const store = useDocumentsStore();
  const model = tokenModel(context, shape, value, props.offers ?? []);
  const write = (next: JsonValue, what: string): void => {
    edit(store, (made) => writeAt(made, at, next, what));
  };
  return (
    <div className="toks" data-tokens={pointerOf(at)}>
      {model.tokens.map((row) => (
        <Token
          key={`${String(row.at)}`}
          row={row}
          forms={model.forms.map((one) => one.tag)}
          removable={model.tokens.length > model.least}
          label={label}
          onForm={(tag) => {
            write(withForm(model, value, row.at, tag), `${label} token ${String(row.at)}`);
          }}
          onText={(typed) => {
            write(withText(model, value, row.at, typed), `${label} token ${String(row.at)}`);
          }}
          onRemove={() => {
            write(withoutToken(value, row.at), `${label} token ${String(row.at)}`);
          }}
        />
      ))}
      <button
        type="button"
        className="tok add"
        data-token-add={pointerOf(at)}
        aria-label={textWith('Add a token to {}', label)}
        onClick={() => {
          write(withToken(model, value), `${label} token`);
        }}
      >
        {text('+ token')}
      </button>
    </div>
  );
}

/** One chip: the form it is in, what it carries, and the × that takes it out. */
function Token({
  row,
  forms,
  removable,
  label,
  onForm,
  onText,
  onRemove,
}: {
  row: TokenRow;
  forms: readonly string[];
  removable: boolean;
  label: string;
  onForm: (tag: string) => void;
  onText: (text: string) => void;
  onRemove: () => void;
}): JSX.Element {
  const listId = `ts-token-${row.at}-${row.member ?? 'literal'}`;
  return (
    <span className={tokenClass(row)} data-token={String(row.at)} data-token-form={row.form}>
      <select
        className="ctl sel mono"
        data-token-kind={String(row.at)}
        aria-label={textWith('Form of {}', `${label} token ${String(row.at)}`)}
        value={row.form}
        onChange={(event) => {
          onForm(event.target.value);
        }}
      >
        {forms.includes(row.form) ? null : <option value={row.form}>{EMPTY}</option>}
        {forms.map((form) => (
          <option key={form} value={form}>
            {form}
          </option>
        ))}
      </select>
      <input
        className="ctl mono"
        data-token-text={String(row.at)}
        aria-label={textWith('{}', `${label} token ${String(row.at)}`)}
        value={row.text}
        {...(row.names.length > 0 ? { list: listId } : {})}
        onChange={(event) => {
          onText(event.target.value);
        }}
      />
      {row.names.length > 0 ? (
        <datalist id={listId}>
          {row.names.map((name) => (
            <option key={name} value={name} />
          ))}
        </datalist>
      ) : null}
      <button
        type="button"
        className="tbtn"
        data-token-remove={String(row.at)}
        aria-label={textWith('Remove {}', `${label} token ${String(row.at)}`)}
        disabled={!removable}
        onClick={onRemove}
      >
        ×
      </button>
    </span>
  );
}

/**
 * S8's three chip colours, from what the core says the form names.
 *
 * A form that carries no name is a literal; one whose names the document *declares* is the
 * structural colour; one whose names the location generates is the derived colour (§4.21). Read
 * from the fact, as feature 2.11's expression chip is read from its production.
 */
export function tokenClass(row: TokenRow): string {
  if (row.declared === null) return 'tok lit';
  return row.declared ? 'tok idx' : 'tok coord';
}

/** What nothing at all is written as — `view.py`'s own em dash. */
const EMPTY = '—';

/** Write the whole name back at its place — one command per gesture, as D13 asks. */
function writeAt(context: EditContext, at: Path, value: JsonValue, label: string): Command {
  return writeMember(context, at, value, `Set ${label}`);
}

/** Run a command against the current document, reporting what it refuses as a toast (Q5). */
function edit(
  store: ReturnType<typeof useDocumentsStore>,
  make: (context: EditContext) => Command,
): void {
  try {
    store.getState().edit(make);
  } catch (error) {
    store.getState().setToast({ text: error instanceof EditError ? error.message : String(error) });
  }
}
