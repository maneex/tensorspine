import type { ReferenceSelector } from '../src/references.js';

/**
 * Which occurrences of a name refer to what a map declares.
 *
 * This is the one thing about the grammar the schemas do not state and the store therefore does
 * not know: a `$ref` names a *shape*, not a scope, and no keyword says that the member
 * `quantity` holds the key of the map `quantities`. The store's `referenceIndex` finds every
 * occurrence and every tag by reading the schemas; a command that renames or deletes is handed
 * the selectors that say which of them it is about, and here is where the suites state them.
 *
 * They belong in `packages/ui/src/presentation.json` — the one data file plan §1 admits, "keyed
 * by JSON pointers into the schemas" — which feature 2.2 writes; until it exists, every caller
 * states its own, and these are the reading this feature was written against.
 */

/** A quantity: named by `{"quantity": …}`, anywhere in the document. */
export const QUANTITY: readonly ReferenceSelector[] = [{ tag: 'quantity', kind: 'tagged' }];

/** A constant: named by `{"constant": …}` in a constant binding and its endpoints. */
export const CONSTANT: readonly ReferenceSelector[] = [{ tag: 'constant', kind: 'tagged' }];

/**
 * A root instance: named by the `instance` of a *root* selector, which is the one written with no
 * `composition` beside it.
 */
export const ROOT_INSTANCE: readonly ReferenceSelector[] = [
  { tag: 'instance', kind: 'tagged', qualifiers: { composition: null } },
];

/** A composition: named by the `composition` of a generated selector. */
export const COMPOSITION: readonly ReferenceSelector[] = [{ tag: 'composition', kind: 'tagged' }];

/**
 * A site of a composition: named as a `site` inside that composition's own bindings, and as the
 * `instance` of a generated selector that names the composition beside it.
 */
export function site(composition: string): readonly ReferenceSelector[] {
  return [
    { tag: 'site', kind: 'tagged', under: ['compositions', composition] },
    { tag: 'instance', kind: 'tagged', qualifiers: { composition } },
  ];
}

/**
 * An index of a composition: named by `{"index": …}` and by the *keys* of every `indices` map —
 * the composition's own ranges, the overrides on its scoped endpoints, and the assignments a
 * generated selector writes from outside.
 */
export function INDEX(composition: string): readonly ReferenceSelector[] {
  return [
    { tag: 'index', kind: 'tagged', under: ['compositions', composition] },
    { tag: 'indices', kind: 'key', under: ['compositions', composition] },
    { tag: 'indices', kind: 'key', qualifiers: { composition } },
  ];
}

/** A public input: named as the `stream` another input joins. */
export const PUBLIC_INPUT: readonly ReferenceSelector[] = [{ tag: 'stream', kind: 'tagged' }];
