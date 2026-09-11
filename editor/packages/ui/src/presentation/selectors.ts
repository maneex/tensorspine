/**
 * The pairing the schemas do not state: which occurrences of a name refer to which declaration.
 *
 * Feature 2.1 measured the gap and left it here. The schemas say *which* members hold a name —
 * the name definitions are discovered (whatever a `propertyNames` points at), and thirteen
 * members that hold a name and thirteen that hold a map keyed by one follow — but **no keyword
 * says what a name refers to**. Nothing links the member `quantity` to the map `quantities`, or
 * a generated selector's `instance` to a *site* of the composition written beside it: a `$ref`
 * names a shape, not a scope. So the store's reference index answers occurrences, and `rename`
 * and `remove` take `ReferenceSelector`s from their caller.
 *
 * (Feature 2.1's ledger row says *fourteen* maps keyed by a name and lists thirteen;
 * `referenceTags(shapes, 'model').keys` has exactly the thirteen it lists, and
 * `tests/audit/presentation.test.ts` holds a table to that set by set-equality.)
 *
 * This module is the caller's answer, read from `presentation.json` rather than typed into a
 * component: a binding on the *place where a map of declarations is written* says what that map
 * declares and where a name of it is referred to, and {@link referenceSelectors} turns that into
 * the selectors the store takes.
 *
 * **Why the place and not the definition.** `#/$defs/index_ranges` is a composition's index
 * declarations in one place and a *reference* to them in another (`for_each`, and the overrides
 * on a scoped endpoint). One definition, two meanings, so the anchor is the place the map is
 * written — `#/$defs/composition_definition/properties/indices` — and never the shape it refers
 * to.
 *
 * **The scope.** A site and an index are declared inside a composition, and are named from two
 * places: inside that composition, and from outside it with the composition's name written
 * beside them. Neither can be stated without knowing *which* composition, so a scoped rule is
 * read against the enclosing declaration the caller supplies — its path for `under`, its name
 * for the qualifier that carries it.
 */
import type { Path, ReferenceKind, ReferenceSelector } from '@tensorspine/store';

import { SCOPE } from './audit.js';
import { PresentationError } from './load.js';
import type { Binding } from './types.js';

/** The enclosing declaration a scoped rule is read against. */
export interface Scope {
  /** Where it is written in the document: `['compositions', 'decoder']`. */
  readonly path: Path;
  /** Its own name: `decoder`. */
  readonly name: string;
}

/**
 * The kinds of occurrence the store answers.
 *
 * Written as a record over the store's own type, so that a kind added there fails to compile
 * here rather than being read as a selector that matches nothing.
 */
const KINDS: Readonly<Record<ReferenceKind, true>> = { key: true, tagged: true };

function isKind(kind: string): kind is ReferenceKind {
  return Object.hasOwn(KINDS, kind);
}

/**
 * The selectors that find every reference to a declaration this binding declares.
 *
 * A binding with no `refers` answers none — which is a statement and not a gap: nothing in a
 * `tensorspine/2.0` document names a public output, so renaming one rewrites nothing.
 */
export function referenceSelectors(binding: Binding, scope?: Scope): readonly ReferenceSelector[] {
  return (binding.refers ?? []).map((rule) => {
    const selector: {
      tag: string;
      kind?: ReferenceKind;
      under?: Path;
      qualifiers?: Record<string, string | null>;
    } = { tag: rule.tag };
    if (rule.kind !== undefined) {
      if (!isKind(rule.kind)) {
        throw new PresentationError(`'${rule.kind}' is no kind of reference`);
      }
      selector.kind = rule.kind;
    }
    if (rule.under !== undefined) {
      if (rule.under !== SCOPE) {
        throw new PresentationError(`'${rule.under}' names no place a reference is under`);
      }
      selector.under = scopeOf(scope, rule.tag).path;
    }
    const qualifiers: Record<string, string | null> = {};
    if (rule.scopedBy !== undefined) {
      qualifiers[rule.scopedBy] = scopeOf(scope, rule.tag).name;
    }
    for (const absent of rule.without ?? []) qualifiers[absent] = null;
    if (Object.keys(qualifiers).length > 0) selector.qualifiers = qualifiers;
    return selector;
  });
}

function scopeOf(scope: Scope | undefined, tag: string): Scope {
  if (scope === undefined) {
    throw new PresentationError(`'${tag}' is read against an enclosing declaration, and none was given`);
  }
  return scope;
}
