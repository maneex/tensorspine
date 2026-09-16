/**
 * The Identity row's one field — plan §4.11, feature 2.21.
 *
 * `primitive_reference` is drawn as **one** field in the library's own `name@version` form, with
 * the gathered catalog behind it as a suggest list. What makes one field over two members total
 * rather than a convention is the loader's own sentence beside `identityKey`: `@` occurs in
 * neither half, so the text splits exactly once and reads back byte for byte.
 *
 * **Nothing is refused, and nothing is guessed** (§9 Q5). A text that names an identity of the
 * catalog is written at once — the two members are the catalog's own spelling, which is what makes
 * choosing one write what the corpus writes. A text that names none is neither refused nor
 * silently written: the field goes back to what the document holds and the confirm asks what was
 * meant — the nearest match, adding the primitive to the project, keeping it as typed for V1 to
 * report, or nothing at all. The field reverting *before* the answer is the point: until one of
 * the four is chosen the document has not changed, and Cancel therefore has nothing to undo.
 *
 * **No member of the grammar and no identity of the library is named here.** Which of the two
 * members is the version is read from the picker `presentation.json` binds to it (feature 2.10);
 * the list is `library.catalog`, which is the core's projection of `by_id`.
 */
import { useEffect, useMemo, useState, type JSX } from 'react';

import type { JsonValue } from '@tensorspine/lang';
import type { Path, Shape } from '@tensorspine/store';

import { useDocuments, useDocumentsStore } from '../documents/context.js';
import type { OpenDocument } from '../documents/store.js';
import type { FormContext, FormRow } from '../forms/index.js';
import {
  identityOf,
  PICKER,
  readIdentity,
  suggestIdentities,
  type IdentityOffers,
} from '../library/primitives.js';
import { text, textWith } from '../shell/strings.js';
import { CreateAction } from './Create.js';
import { referenceMembers, referenceOf } from './reference.js';

/** What nothing at all is written as — `view.py`'s own em dash, as every other row writes it. */
const EMPTY = '—';

/** The offers of the gathered library: the catalog, and the versions of one name (2.10's list). */
export function useIdentityOffers(row: FormRow | undefined): IdentityOffers {
  const catalog = useDocuments((state) => state.library.catalog);
  const versions = useDocuments((state) => state.library.versions);
  // The list is the picker's: a place bound to another one offers nothing here rather than
  // offering the catalog to something that is not a pinned primitive.
  const wanted = row?.picker === PICKER.primitives;
  return useMemo(
    () => ({
      identities: wanted ? catalog : [],
      versions: (name: string) => (wanted ? (versions.get(name) ?? []) : []),
    }),
    [catalog, versions, wanted],
  );
}

/** The one field a pinned primitive is chosen in. */
export function PrimitiveField({
  one,
  at,
  shape,
  context,
  value,
  row,
}: {
  readonly one: OpenDocument;
  /** The `primitive` member's place in the document. */
  readonly at: Path;
  /** Its shape, stepped down from the root like every other reading of a place (feature 2.3). */
  readonly shape: Shape;
  readonly context: FormContext;
  readonly value: JsonValue | undefined;
  /** The row the walker made of it: its label, its picker and its `create`. */
  readonly row: FormRow;
}): JSX.Element {
  const store = useDocumentsStore();
  const offers = useIdentityOffers(row);
  const members = referenceMembers(context, shape);
  const held = members === null ? { name: '', version: '' } : referenceOf(value, members);
  const written = identityOf(held);
  const [typed, setTyped] = useState(written);
  useEffect(() => {
    setTyped(written);
  }, [written]);

  const listId = `ts-identity-${row.path.replace(/\W/g, '-')}`;
  const suggestions = suggestIdentities(offers, typed);
  const standing = offers.identities.find((identity) => identity.id === written) ?? null;

  /** Ask for the identity a text names, once the text is finished. */
  const commit = (): void => {
    const verdict = readIdentity(offers, typed, held);
    // The field goes back to the document either way: a known identity is written and the
    // document is what redraws it, and an unknown one has not been written at all yet.
    setTyped(written);
    if (verdict.kind === 'nothing') return;
    if (identityOf(verdict.reference) === written) return;
    if (verdict.kind === 'known') {
      store
        .getState()
        .setReference(
          at,
          verdict.reference,
          `Set ${row.label} ${identityOf(verdict.reference)}`,
          one.id,
        );
      return;
    }
    store.getState().setDialog({
      kind: 'identity',
      id: one.id,
      at,
      typed: typed.trim(),
      wanted: verdict.reference,
      nearest: verdict.nearest,
      held,
    });
  };

  return (
    <div className="frow" data-member={row.label} data-editor={row.widget}>
      <span className="fk">{row.label}</span>
      <span className="fv">
        <input
          className="ctl mono wide"
          data-identity={row.path}
          aria-label={row.label}
          list={listId}
          value={typed}
          onChange={(event) => {
            setTyped(event.target.value);
          }}
          onBlur={commit}
          onKeyDown={(event) => {
            event.stopPropagation();
            if (event.key !== 'Enter' && event.key !== 'Escape') return;
            // The key is **handled here**, so its default must not also happen. Enter opens the
            // confirm, React commits a discrete event synchronously, the dialog's own frame moves
            // the focus into it — and the browser would then give the *keypress* the Enter
            // generates to the control that has just been focused, pressing a button nobody
            // pressed. Found by this feature's own browser suite; the same rule is why Escape is
            // prevented rather than left to whatever else would take it.
            event.preventDefault();
            if (event.key === 'Enter') commit();
            else setTyped(written);
          }}
        />
        <datalist id={listId}>
          {suggestions.map((id) => {
            const identity = offers.identities.find((one_) => one_.id === id);
            return <option key={id} value={id} label={identity?.base ?? ''} />;
          })}
        </datalist>
        <CreateAction
          label={row.create}
          picker={row.picker}
          onCreate={() => {
            // The action beside the picker is for what the list does **not** carry, so it asks the
            // same question the confirm asks, about the text that is in the field. A text the
            // catalog already carries has nothing to declare, and saying so in the Log is §4.4's
            // own rule for a gesture with nothing to do — better than a dialog claiming an
            // identity of the library is not in it.
            const verdict = readIdentity(offers, typed, held);
            if (verdict.kind !== 'unknown') {
              store
                .getState()
                .note(
                  `${row.create ?? ''}: type the name of the primitive to declare — ` +
                    `${typed.trim() === '' ? 'the field is empty' : `the library already carries ${typed.trim()}`}`,
                );
              return;
            }
            store.getState().setDialog({
              kind: 'identity',
              id: one.id,
              at,
              typed: typed.trim(),
              wanted: verdict.reference,
              nearest: verdict.nearest,
              held,
            });
          }}
        />
        <span className="sub" data-identity-base={row.path}>
          {standing === null ? EMPTY : textWith('from {}', standing.base)}
          {standing?.template === true ? ` · ${text('template')}` : ''}
        </span>
      </span>
    </div>
  );
}
