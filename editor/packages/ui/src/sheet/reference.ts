/**
 * `primitive_reference` as one field — plan §4.11's Identity row, feature 2.21.
 *
 * The grammar writes a pinned primitive as two required members, and the library writes the same
 * pair as one word: `identityKey`'s `name@version`, with `@` in neither half. §4.11 asks for "the
 * primitive with version select" and using the editor showed what two bare fields cost — a name
 * typed from memory and a version typed after it. One field over the pair is the library's own
 * form, and this module is its model: which member is which, what the field reads, and what a
 * commit writes.
 *
 * **Which member is which is read, not remembered.** The two members are the schema's, in the
 * schema's order (`propertyOrder`), and the one that is the *version* is the one
 * `presentation.json` gives the version picker to — feature 2.10's `primitive-versions` binding,
 * still the list the version half offers. So no member of the grammar is named here, which is
 * what catching rule §1 (b) asks of every component, and a reference that grew a third member
 * would render generically and say so rather than being written wrongly.
 *
 * **A commit writes each member where it stands.** Rewriting the object whole would put the
 * schema's order back over the file's, and an unedited-looking save would move bytes (D12); so
 * the name and the version are two `setValue`s in one command (D13: one gesture, one undo), and
 * a member the document does not carry is appended at its own place (§5.5).
 */
import { isJsonObject, type JsonValue } from '@tensorspine/lang';
import {
  nodeAt,
  setMemberAt,
  setValue,
  type Command,
  type EditContext,
  type Path,
  type Shape,
} from '@tensorspine/store';

import { chainOf, type FormContext } from '../forms/index.js';
import { boundAlong } from '../presentation/load.js';
import { PICKER, type Reference } from '../library/primitives.js';

/** One member of the reference: where it is written, and what its schema says about it. */
export interface ReferenceMember {
  /** The member's own name, as the schema declares it — never written in this package. */
  readonly member: string;
  readonly shape: Shape;
}

/** The two members of a `primitive_reference`, told apart by the binding at the version's own. */
export interface ReferenceMembers {
  readonly name: ReferenceMember;
  readonly version: ReferenceMember;
}

/**
 * The two members, or `null` where this place is no pair the field can edit.
 *
 * `null` is the honest answer and the caller draws the generic rows instead (§1's "unknown
 * constructs get the generic widget"): a definition with other than two members, or with no
 * version picker to tell them apart, is one nobody has bound this editor to.
 */
export function referenceMembers(context: FormContext, shape: Shape): ReferenceMembers | null {
  const order = context.shapes.propertyOrder(shape);
  if (order.length !== 2) return null;
  const members = order.map((member) => {
    const inner = context.shapes.member(shape, member);
    const anchors = chainOf(inner).map((place) => place.anchor);
    return { member, shape: inner, picker: boundAlong(context.bindings, anchors, 'picker') };
  });
  const version = members.find((one) => one.picker === PICKER.versions);
  const name = members.find((one) => one !== version);
  if (version === undefined || name === undefined) return null;
  return {
    name: { member: name.member, shape: name.shape },
    version: { member: version.member, shape: version.shape },
  };
}

/** What the document holds at a reference's place: the two members, empty where it holds none. */
export function referenceOf(value: JsonValue | undefined, members: ReferenceMembers): Reference {
  return {
    name: scalarText(memberValue(value, members.name.member)),
    version: scalarText(memberValue(value, members.version.member)),
  };
}

/**
 * Write a reference at a place: each member where it stands, in one command.
 *
 * The label is the gesture's, in the Edit menu's words (D13), and the caller supplies it because
 * choosing from the list and keeping a name the catalog lacks are two different gestures with the
 * same edit.
 */
export function writeReference(
  context: EditContext,
  at: Path,
  members: ReferenceMembers,
  reference: Reference,
  label: string,
): Command {
  const made = [
    writeMemberValue(context, at, members.name.member, reference.name, label),
    writeMemberValue(context, at, members.version.member, reference.version, label),
  ];
  return {
    label,
    moves: made.flatMap((one) => one.moves),
    edit(draft) {
      for (const one of made) one.edit(draft);
    },
  };
}

/** One member of the reference, set where the document writes it and appended where it does not. */
function writeMemberValue(
  context: EditContext,
  at: Path,
  member: string,
  value: string,
  label: string,
): Command {
  const path = [...at, member] as Path;
  if (nodeAt(context.tree, path) !== undefined) {
    return setValue(context, { path, value, label });
  }
  return setMemberAt(context, { path: at, name: member, value, label });
}

/** The value of one member of an object of the tree, or `undefined`. */
function memberValue(value: JsonValue | undefined, member: string): JsonValue | undefined {
  if (value === undefined || !isJsonObject(value)) return undefined;
  return value.members.find((one) => one.name === member)?.value;
}

/** A member's text, and the empty text wherever the document holds something that is not one. */
function scalarText(value: JsonValue | undefined): string {
  return typeof value === 'string' ? value : '';
}
