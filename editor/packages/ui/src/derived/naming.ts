/**
 * What a row of a derived product names, and which rows a subject keeps — §4.18's selection
 * filter.
 *
 * > A **selection filter** restricts every tab to the selected node, identity or split.
 *
 * The filter is **schema-driven and not product-driven**, and that is the whole design: the
 * derived schema declares four kinds of identifier — `node_identifier`, `value_reference`,
 * `identity_name`, `graph_split_id` — and `presentation.json` says what a string written at each
 * of those places stands for (`names`). So a row is kept when some string it writes, at a place
 * bound that way, names the subject. Nothing here knows that D3 has `tensors`, that a state's
 * `members` are references, or that a split's `block` is a list of nodes; a product that gained a
 * column would be filtered by it on the day the schema declared its type.
 *
 * **How a string names a subject is the core's.** `declaredSite` is §5.2 rule 2's own reading of a
 * node identifier, `splitMember` the reading of a `<node>.<port or slot>`, and
 * `identityInstanceOf` the reading of an identity instance — all three in
 * `packages/lang/src/describe/figures.ts`, where the canvas and the sheets already read them. The
 * inventory's §7 is explicit that a component may not take an identifier apart for itself.
 *
 * **A node subject is a declared site, because a selection is a place of the document.** The
 * reader selects `attn` — a declaration — and D1, D3, D4, D5 and D6 name its iterations
 * (`decoder/attn[layer=0]` … `[layer=31]`); a folded card stands for the family (§4.8, feature
 * 2.9). So the subject is `decoder/attn`, and a row is kept when its own node's declared site is
 * that one *or* sits under it — which is what makes selecting a template instance keep the rows
 * of everything its expansion writes, and selecting a composition keep its sites'.
 */
import { declaredSite, identityInstanceOf, splitMember, type PyValue } from '@tensorspine/lang';
import type { Shape } from '@tensorspine/store';

import type { FormContext } from '../forms/index.js';
import { isList, isRecord, membersOf } from './cells.js';

/** One identifier a row writes, and what the binding says it names. */
export interface Named {
  /** The binding's own word: `node`, `reference`, `identity`, `split`. */
  readonly kind: string;
  readonly name: string;
}

/** What every tab is held to, when one is chosen. */
export interface DerivedSubject {
  /** Which of the four kinds of name the subject is one of. */
  readonly kind: string;
  /** The name itself — a declared site, an identity, a graph split. */
  readonly name: string;
  /** What the chip says: the thing as the reader selected it. */
  readonly label: string;
}

/** How deep a row is walked for the identifiers it writes. */
const DEPTH = 5;

/**
 * Every identifier one entry of a product writes, with what each one names.
 *
 * The walk follows the value against its own schema, so a name nested in a location, in a split's
 * block or in a state's payload is found the same way a name in a column is.
 */
export function namedIn(value: PyValue, shape: Shape, context: FormContext): readonly Named[] {
  const found: Named[] = [];
  const walk = (one: PyValue, at: Shape, depth: number): void => {
    if (typeof one === 'string') {
      const kind = context.bindings.firstOf(at.all.map((place) => place.anchor))?.names;
      if (kind !== undefined) found.push({ kind, name: one });
      return;
    }
    if (depth >= DEPTH) return;
    if (isList(one)) {
      one.forEach((item, index) => {
        walk(item, context.shapes.item(at, index), depth + 1);
      });
      return;
    }
    if (!isRecord(one)) return;
    for (const [member, inside] of membersOf(one)) {
      walk(inside, context.shapes.member(at, member), depth + 1);
    }
  };
  walk(value, shape, 0);
  return found;
}

/**
 * Whether one identifier names the subject.
 *
 * The table is written with **bare property keys** and no string literal, for the reason feature
 * 2.6 gave the status bar's figure renderings: `node` is a word of the editor's own vocabulary
 * here, `reference` was chosen over `value` precisely because `value` is an enumerated value of
 * the four schemas, and a whole-literal scan (§1 b) cannot tell one from the other. A test holds
 * the key set to the `names` enumeration of the editor's presentation schema.
 */
const NAMING: Readonly<Record<string, Readonly<Record<string, (name: string, subject: string) => boolean>>>> =
  {
    node: {
      node: (name, subject) => underSite(name, subject),
      reference: (name, subject) => {
        const split = splitMember(name);
        return split !== null && underSite(split.site, subject);
      },
    },
    identity: {
      identity: (name, subject) => identityInstanceOf(name, subject),
    },
    split: {
      split: (name, subject) => name === subject,
    },
  };

/** Whether a node identifier's declared site is the subject, or sits under it. */
function underSite(identifier: string, subject: string): boolean {
  const declared = declaredSite(identifier);
  return declared === subject || declared.startsWith(`${subject}/`);
}

/** Whether an entry's identifiers keep it under a subject. */
export function keptBy(named: readonly Named[], subject: DerivedSubject | null): boolean {
  if (subject === null) return true;
  const rules = NAMING[subject.kind];
  if (rules === undefined) return true;
  return named.some((one) => rules[one.kind]?.(one.name, subject.name) === true);
}

/** The kinds of subject the filter can be held to — for the audit that closes the set. */
export function subjectKinds(): string[] {
  return Object.keys(NAMING).sort();
}

/** Every kind of name the filter reads, whichever subject it is held to. */
export function namedKinds(): string[] {
  return [...new Set(Object.values(NAMING).flatMap((rules) => Object.keys(rules)))].sort();
}
