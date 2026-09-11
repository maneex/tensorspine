/**
 * The tree as a command writes it: Immer's draft, and the walk that finds a place in it.
 *
 * Undo is a command log of Immer patches over the document tree (plan D13), so every command is
 * a recipe that mutates a draft and every edit answers its forward and inverse patches. The
 * draft is the same plain data the core parses — feature 0.3 chose plain data for exactly this,
 * "it survives the structured clone to the worker and can be drafted by Immer, where a class
 * would lose its prototype crossing the boundary" — with the `readonly` taken off.
 *
 * **Why the shapes are written again here rather than taken as `Draft<JsonValue>`.** Immer's
 * `Draft<T>` is a conditional type that rewrites a type recursively; over `JsonValue`, which is
 * a union that contains itself twice (an object's members and an array's items), TypeScript
 * gives up — *type instantiation is excessively deep*. The mirror below is that rewriting done
 * by hand, once: the same five shapes, mutable. `produceWithPatches` takes the draft's type as
 * its second type argument for precisely this reason, so nothing is cast to reach it.
 *
 * A value the core built (`jsonObject`, `jsonInteger`, a subtree read out of the document) is
 * frozen and typed `readonly`; {@link asDraftValue} is the one place it is admitted into a draft.
 * It is inserted, never mutated — Immer copies what it has to on the way out — and the cast is
 * confined to that function so that no command carries one.
 */
import type { JsonValue } from '@tensorspine/lang';

import { EditError, pointerOf, type Path, type Step } from './path.js';

/** A number of the draft: the core's `JsonNumber`, mutable. */
export interface MutableNumber {
  kind: 'number';
  value: number;
  real: boolean;
  lexeme?: string;
}

/** One member of an object of the draft. */
export interface MutableMember {
  name: string;
  value: MutableValue;
}

/** An object of the draft: its members in order. */
export interface MutableObject {
  kind: 'object';
  members: MutableMember[];
}

/** An array of the draft. */
export type MutableArray = MutableValue[];

/** A value of the draft: the core's `JsonValue`, mutable. */
export type MutableValue = string | boolean | null | MutableNumber | MutableObject | MutableArray;

/**
 * A value the core built, admitted into a draft.
 *
 * The mirror types are structurally the core's with the `readonly` taken off, so the conversion
 * is a widening the type system will not do on its own. The value is inserted and never mutated.
 */
export function asDraftValue(value: JsonValue): MutableValue {
  return value as MutableValue;
}

/** Whether a draft value is an array node. */
export function isMutableArray(value: MutableValue): value is MutableArray {
  return Array.isArray(value);
}

/** Whether a draft value is an object node. */
export function isMutableObject(value: MutableValue): value is MutableObject {
  return typeof value === 'object' && value !== null && !isMutableArray(value) && value.kind === 'object';
}

/** One step down from a draft value, or `undefined` where the step names nothing. */
export function stepIntoDraft(value: MutableValue, step: Step): MutableValue | undefined {
  if (typeof step === 'number') {
    return isMutableArray(value) ? value[step] : undefined;
  }
  if (!isMutableObject(value)) return undefined;
  return value.members.find((member) => member.name === step)?.value;
}

/** The draft value at a place. Throws when the path names nothing: a command edits what is there. */
export function draftAt(draft: MutableValue, path: Path): MutableValue {
  let current: MutableValue = draft;
  for (const [position, step] of path.entries()) {
    const next = stepIntoDraft(current, step);
    if (next === undefined) {
      throw new EditError(`no value at ${pointerOf(path.slice(0, position + 1))}`);
    }
    current = next;
  }
  return current;
}

/** The object at a place of the draft. Throws when the place is not an object. */
export function objectDraftAt(draft: MutableValue, path: Path): MutableObject {
  const node = draftAt(draft, path);
  if (!isMutableObject(node)) throw new EditError(`${pointerOf(path)} is not an object`);
  return node;
}

/** The array at a place of the draft. Throws when the place is not an array. */
export function arrayDraftAt(draft: MutableValue, path: Path): MutableArray {
  const node = draftAt(draft, path);
  if (!isMutableArray(node)) throw new EditError(`${pointerOf(path)} is not an array`);
  return node;
}

/** The position of a member of a draft object, or `-1`. */
export function memberIndex(node: MutableObject, name: string): number {
  return node.members.findIndex((member) => member.name === name);
}

/**
 * Set a member of a draft object, in place.
 *
 * An existing member keeps its position — an edit never moves a member, which is what makes an
 * edited-and-reverted document identical to the one that was read (the core's `withMember` says
 * the same of the frozen tree). A new member goes where `at` says, and at the end otherwise:
 * "new members appended where the gesture put them" (plan §5.5).
 */
export function setMember(node: MutableObject, name: string, value: MutableValue, at?: number): void {
  const found = memberIndex(node, name);
  if (found >= 0) {
    const member = node.members[found];
    if (member !== undefined) member.value = value;
    return;
  }
  const position = at === undefined ? node.members.length : Math.max(0, Math.min(at, node.members.length));
  node.members.splice(position, 0, { name, value });
}

/** Remove a member of a draft object, in place. Answers whether there was one. */
export function removeMember(node: MutableObject, name: string): boolean {
  const found = memberIndex(node, name);
  if (found < 0) return false;
  node.members.splice(found, 1);
  return true;
}

/**
 * Rename a member of a draft object, keeping its position.
 *
 * Member order is the document's (feature 0.3: a plain object cannot hold it, which is why the
 * tree keeps a list), so a rename writes the new name where the old one stood. A name the object
 * already carries is refused: the two members would collide, and V12 is the core's refusal of a
 * document that writes one name twice.
 */
export function renameMember(node: MutableObject, from: string, to: string): void {
  const found = memberIndex(node, from);
  if (found < 0) throw new EditError(`no member named '${from}'`);
  if (from !== to && memberIndex(node, to) >= 0) {
    throw new EditError(`a member named '${to}' is already there`);
  }
  const member = node.members[found];
  if (member !== undefined) member.name = to;
}
