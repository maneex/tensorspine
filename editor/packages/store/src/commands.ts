/**
 * The named commands: the JSON edit each gesture of plan §4.7 makes.
 *
 * A command is computed against the tree as it stands — it has to be, since a delete has to know
 * what names the thing it removes and a connection has to know what already feeds the port — and
 * answers a recipe the log runs on a draft (D13), a name for the Edit menu, and what the gesture
 * did beyond the patches: the places that moved (so the sidecar's keys can follow, plan §3), the
 * cascade a delete listed, the edge a connection replaced.
 *
 * **What a command does not decide.** Which occurrences of a name refer to the thing being
 * renamed or deleted is stated by the caller as {@link ReferenceSelector}s — `./references.ts`
 * says why the schema cannot say it — and every *place* a command edits is a path the caller
 * hands in. The store never guesses a place from a word.
 *
 * **No gesture is refused for a semantic reason** (plan §9 Q5): a command that leaves an edge
 * the core will refuse is a command that worked, and the refusal is a Problem. What a command
 * does refuse is a tree that is off the *grammar* (D5), and the delete cascade is where that
 * bites: a removal that would empty a container the schema requires to be non-empty is not
 * performed, and the entry is reported as kept so the confirmation can say so.
 */
import {
  isJsonArray,
  isJsonNumber,
  isJsonObject,
  jsonObject,
  type JsonObject,
  type JsonValue,
} from '@tensorspine/lang';

import {
  arrayDraftAt,
  asDraftValue,
  draftAt,
  isMutableArray,
  isMutableObject,
  memberIndex,
  objectDraftAt,
  removeMember,
  renameMember,
  setMember,
  type MutableObject,
} from './draft.js';
import {
  arrayAt,
  child,
  EditError,
  isUnder,
  lastOf,
  nodeAt,
  objectAt,
  parentOf,
  pointerOf,
  type Path,
} from './path.js';
import type { Occurrence, ReferenceIndex, ReferenceSelector } from './references.js';
import type { SchemaShapes, Shape } from './shape.js';

/** One name a rename rewrites: the declaration itself, or an occurrence the index found. */
interface Rewrite {
  readonly kind: 'key' | 'tagged';
  readonly tag: string;
  readonly name: string;
  readonly path: Path;
}

/** A place that moved: what the layout sidecar's keys follow (plan D6, §3). */
export interface PathMove {
  readonly from: Path;
  readonly to: Path;
}

/** What a delete removed, and what it could not remove without leaving the grammar. */
export interface Cascade {
  /** The place the gesture named. */
  readonly target: Path;
  /** Every place removed, deepest first — the confirmation's list. */
  readonly removed: readonly Path[];
  /**
   * Places that name the target and stayed, because removing them would have emptied a container
   * the grammar requires to be non-empty (a public input's `to`, `interfaces/inputs` itself).
   * The document stays on the schema and the core's V1 reports the dangling reference, which is
   * the editor's rule: wire first, fix afterwards.
   */
  readonly kept: readonly Path[];
}

/** A gesture, computed against the tree it will be applied to. */
export interface Command {
  /** The Edit menu's wording. */
  readonly label: string;
  /** The edit itself, run on a draft by the log. */
  readonly edit: (draft: MutableObject) => void;
  /** The places that moved. */
  readonly moves: readonly PathMove[];
  /** A delete's listing. */
  readonly cascade?: Cascade;
  /** The binding a connection replaced, when an input was already fed (V7, Q5). */
  readonly replaced?: Path;
}

/** What every command is computed against. */
export interface EditContext {
  readonly tree: JsonObject;
  readonly shapes: SchemaShapes;
  /** The role the document is read under: `model` for a `tensorspine/2.0` document. */
  readonly role: string;
  readonly index: ReferenceIndex;
}

/** The shape of a place of the document, stepped down from the root. */
export function shapeAt(context: EditContext, path: Path): Shape {
  let shape = context.shapes.root(context.role);
  for (const step of path) shape = context.shapes.step(shape, step);
  return shape;
}

/** Set the value at a place. The sheet's every row, and the canvas's every badge, is this. */
export function setValue(
  context: EditContext,
  request: { readonly path: Path; readonly value: JsonValue; readonly label?: string },
): Command {
  const { path, value } = request;
  if (nodeAt(context.tree, path) === undefined) {
    throw new EditError(`no value at ${pointerOf(path)}`);
  }
  const step = lastOf(path);
  return {
    label: request.label ?? `Set ${String(step)}`,
    moves: [],
    edit(draft) {
      write(draft, path, value);
    },
  };
}

/**
 * Put a member into a map, or replace one.
 *
 * A new member goes where `at` says and at the end otherwise; an existing one keeps its place
 * (feature 0.3's rule, so that an edited-and-reverted document is the one that was read).
 */
export function setMemberAt(
  context: EditContext,
  request: {
    readonly path: Path;
    readonly name: string;
    readonly value: JsonValue;
    readonly at?: number;
    readonly label?: string;
  },
): Command {
  const { path, name, value } = request;
  if (objectAt(context.tree, path) === undefined) {
    throw new EditError(`${pointerOf(path)} is not an object`);
  }
  const known = objectAt(context.tree, path)?.members.some((member) => member.name === name) ?? false;
  return {
    label: request.label ?? `${known ? 'Set' : 'Add'} ${name}`,
    moves: [],
    edit(draft) {
      const node = objectDraftAt(draft, path);
      setMember(node, name, asDraftValue(value), request.at);
    },
  };
}

/** Put an item into an array, at a position or at the end. */
export function insertItem(
  context: EditContext,
  request: {
    readonly path: Path;
    readonly value: JsonValue;
    readonly at?: number;
    readonly label?: string;
  },
): Command {
  const { path, value } = request;
  const items = arrayAt(context.tree, path);
  if (items === undefined) throw new EditError(`${pointerOf(path)} is not an array`);
  const position = Math.max(0, Math.min(request.at ?? items.length, items.length));
  return {
    label: request.label ?? `Add to ${String(lastOf(path))}`,
    moves: [],
    edit(draft) {
      arrayDraftAt(draft, path).splice(position, 0, asDraftValue(value));
    },
  };
}

/**
 * Add an instance or a site: the palette's drop (plan §4.7, D5).
 *
 * D5 fixes the shape — "adding an instance writes `{primitive, arguments: {}, families:
 * [<proposed>]}` — on the grammar, refused by V2 (missing required arguments), which the
 * Problems panel lists with navigation". What is *required* of one is the schema's to say, so
 * the command takes a value per required member, writes them in the schema's own order, and
 * refuses a set that is not the schema's. A placeholder is then on the grammar by construction
 * rather than by memory, and a schema that gained a required member fails here instead of
 * producing a document the loader will refuse.
 */
export function addToMap(
  context: EditContext,
  request: {
    readonly path: Path;
    readonly name: string;
    readonly values: Readonly<Record<string, JsonValue>>;
    readonly label?: string;
  },
): Command {
  const { path, name, values } = request;
  const map = objectAt(context.tree, path);
  if (map === undefined) throw new EditError(`${pointerOf(path)} is not an object`);
  if (map.members.some((member) => member.name === name)) {
    throw new EditError(`${pointerOf(path)} already has a member named '${name}'`);
  }
  const shape = context.shapes.values(shapeAt(context, path));
  const required = context.shapes.constraintsOf(shape).required;
  const supplied = Object.keys(values);
  const missing = [...required].filter((member) => !supplied.includes(member));
  if (missing.length > 0) {
    throw new EditError(`a member of ${pointerOf(path)} requires ${missing.join(', ')}`);
  }
  const order = context.shapes.propertyOrder(shape);
  const members = [...supplied].sort((left, right) => position(order, left) - position(order, right));
  const value = jsonObject(members.map((member) => ({ name: member, value: demand(values, member) })));
  return {
    label: request.label ?? `Add ${name}`,
    moves: [],
    edit(draft) {
      setMember(objectDraftAt(draft, path), name, asDraftValue(value));
    },
  };
}

/**
 * Connect two endpoints: the canvas's drag from an output handle to an input handle (§4.7).
 *
 * The drop is never refused (Q5). An input already fed has its older edge **replaced**, undoable,
 * and the toast names it — so the replacement is part of this one command and comes back with
 * one undo, which is what D13 asks of a gesture that does two things.
 *
 * The endpoints are values the caller built: what a `value_endpoint` looks like is the schema's,
 * and which instance a handle belongs to is the canvas's. What is decided here is the binding's
 * *name* — `<to>.<port>` uniquified, §4.7's rule, with the base name supplied — its position in
 * the map (appended) and the replacement.
 */
export function connect(
  context: EditContext,
  request: {
    /** The map the binding goes into: `bindings/values`, or a composition's. */
    readonly path: Path;
    /** The proposed name; a digit is appended while the map has it. */
    readonly name: string;
    /** The members of the binding, `from` and `to` among them. */
    readonly values: Readonly<Record<string, JsonValue>>;
    /** The member that names the fed end, so that an edge already there can be found. */
    readonly into: string;
    readonly label?: string;
  },
): Command {
  const { path, values, into } = request;
  const map = objectAt(context.tree, path);
  if (map === undefined) throw new EditError(`${pointerOf(path)} is not an object`);
  const target = values[into];
  if (target === undefined) throw new EditError(`the binding has no member '${into}'`);
  const shape = context.shapes.values(shapeAt(context, path));
  const missing = [...context.shapes.constraintsOf(shape).required].filter(
    (member) => values[member] === undefined,
  );
  if (missing.length > 0) {
    throw new EditError(`a member of ${pointerOf(path)} requires ${missing.join(', ')}`);
  }
  const existing = map.members.find((member) => {
    const written = isJsonObject(member.value) ? getMemberValue(member.value, into) : undefined;
    return written !== undefined && sameValue(written, target);
  });
  // The replaced binding's own name is free again, so an edge remade at the same place keeps the
  // name it had rather than growing a digit.
  const name = unique(
    request.name,
    map.members.map((member) => member.name).filter((taken) => taken !== existing?.name),
  );
  const order = context.shapes.propertyOrder(shape);
  const members = Object.keys(values).sort(
    (left, right) => position(order, left) - position(order, right),
  );
  const value = jsonObject(members.map((member) => ({ name: member, value: demand(values, member) })));
  const replaced = existing === undefined ? undefined : child(path, existing.name);
  return {
    label: request.label ?? `Connect ${name}`,
    moves: [],
    ...(replaced === undefined ? {} : { replaced }),
    edit(draft) {
      const node = objectDraftAt(draft, path);
      if (existing !== undefined) removeMember(node, existing.name);
      setMember(node, name, asDraftValue(value));
    },
  };
}

/**
 * Rename a declaration and every reference to it (plan §3: "rename is a command that rewrites
 * every reference … and the sidecar keys follow").
 *
 * `path` names the declaration — the member of the map that declares it — and `references` says
 * which occurrences of its name refer to it. The member keeps its place in the map, because
 * member order is the document's.
 */
export function rename(
  context: EditContext,
  request: {
    readonly path: Path;
    readonly to: string;
    readonly references: readonly ReferenceSelector[];
    readonly label?: string;
  },
): Command {
  const { path, to, references } = request;
  const from = lastOf(path);
  if (typeof from !== 'string') throw new EditError(`${pointerOf(path)} is not a named member`);
  const parent = parentOf(path);
  if (objectAt(context.tree, parent) === undefined) {
    throw new EditError(`${pointerOf(parent)} is not an object`);
  }
  if (nodeAt(context.tree, path) === undefined) throw new EditError(`no member at ${pointerOf(path)}`);
  const written: Rewrite[] = [
    { kind: 'key', tag: String(lastOf(parent)), name: from, path },
    ...context.index
      .select(references, from)
      .filter((occurrence) => pointerOf(occurrence.path) !== pointerOf(path))
      .map((occurrence) => ({
        kind: occurrence.kind,
        tag: occurrence.tag,
        name: occurrence.name,
        path: occurrence.path,
      })),
  ];
  // Deepest first. A key written here is a *step* of the paths below it — an index's own name is
  // the key of the range and the key of every override that assigns it — so renaming it before
  // what it contains would leave the rest of the rewrite naming places that have moved.
  const order = [...written].sort((left, right) => right.path.length - left.path.length);
  const moves: PathMove[] = written
    .filter((occurrence) => occurrence.kind === 'key')
    .map((occurrence) => ({ from: occurrence.path, to: child(parentOf(occurrence.path), to) }));
  return {
    label: request.label ?? `Rename ${from} to ${to}`,
    moves,
    edit(draft) {
      for (const occurrence of order) {
        const holder = objectDraftAt(draft, parentOf(occurrence.path));
        if (occurrence.kind === 'key') renameMember(holder, occurrence.name, to);
        else setMember(holder, occurrence.tag, to);
      }
    },
  };
}

/**
 * Delete a declaration and everything that names it (plan §3, §4.7's Select + Delete).
 *
 * The unit a reference costs is not the name but the rule it stands in: an endpoint inside a
 * value binding takes the binding with it, a member of an identity takes the member alone. Which
 * of the two it is comes from the tree — the nearest array item, failing that the nearest member
 * of a map the schema keys by a name — so a grammar that grew another container needs no list
 * here.
 *
 * Then the collapse: a container left below its own `minItems` or `minProperties` goes with its
 * contents, and a container the schema *requires* cannot, so the removals inside that one are
 * cancelled and reported as kept. {@link Cascade} is the confirmation's listing.
 */
export function remove(
  context: EditContext,
  request: {
    readonly path: Path;
    readonly references?: readonly ReferenceSelector[];
    readonly label?: string;
  },
): Command {
  const { path } = request;
  if (nodeAt(context.tree, path) === undefined) throw new EditError(`no value at ${pointerOf(path)}`);
  const name = lastOf(path);
  const naming =
    request.references === undefined || typeof name !== 'string'
      ? []
      : context.index
          .select(request.references, name)
          .filter((occurrence) => pointerOf(occurrence.path) !== pointerOf(path));
  const units = [path, ...naming.map((occurrence) => unitOf(context, occurrence))];
  const cascade = plan(context, path, units);
  return {
    label: request.label ?? `Delete ${String(name)}`,
    moves: [],
    cascade,
    edit(draft) {
      // Deepest first and, inside one array, the highest index first, so that a removal never
      // moves the place of one still to come.
      for (const place of cascade.removed) {
        const holder = draftAt(draft, parentOf(place));
        const step = lastOf(place);
        if (typeof step === 'number') {
          if (isMutableArray(holder)) holder.splice(step, 1);
        } else if (isMutableObject(holder)) {
          removeMember(holder, step);
        }
      }
    },
  };
}

/**
 * The rule an occurrence stands in: what a delete takes with the name.
 *
 * The walk goes outward from the occurrence and stops at the first *item of an array* — a member
 * of an identity, an endpoint of a public input's `to` — or, failing that, the first member of a
 * map the schema keys by a name: a value binding, a parameter binding, a public output. Both
 * tests are the tree's and the schema's, so a grammar that grew another container is read rather
 * than remembered.
 */
function unitOf(context: EditContext, occurrence: Occurrence): Path {
  const path = occurrence.path;
  for (let length = path.length; length >= 1; length -= 1) {
    const place = path.slice(0, length);
    if (typeof place[length - 1] === 'number') return place;
    if (isNameKeyed(context, parentOf(place))) return place;
  }
  return path;
}

/** Whether the map at that place is keyed by a name the schema declares. */
function isNameKeyed(context: EditContext, path: Path): boolean {
  return context.shapes.keys(shapeAt(context, path)).all.length > 0;
}

/** The removals a delete makes, with the collapses and the cancellations the grammar forces. */
function plan(context: EditContext, target: Path, units: readonly Path[]): Cascade {
  const wanted = new Map<string, Path>();
  for (const unit of units) wanted.set(pointerOf(unit), unit);
  const kept = new Map<string, Path>();

  for (;;) {
    const removed = new Map<string, Path>(wanted);
    const blocked = collapse(context, removed);
    if (blocked === null) {
      const all = [...removed.values()];
      const places = all.filter(
        (place) => !all.some((other) => other !== place && isStrictlyUnder(place, other)),
      );
      return {
        target,
        removed: places.sort(removalOrder),
        kept: [...kept.values()].sort(removalOrder),
      };
    }
    // Nothing inside `blocked` can go without taking it below what the grammar admits, and it
    // cannot go itself: every removal asked for under it is cancelled and reported as kept. The
    // target's own is cancelled with the rest, which is how "delete the only public input"
    // answers that it cannot rather than writing a document off the grammar (D5).
    let cancelled = false;
    for (const [pointer, place] of [...wanted]) {
      if (!isUnder(place, blocked)) continue;
      wanted.delete(pointer);
      kept.set(pointer, place);
      cancelled = true;
    }
    if (!cancelled) {
      // A container the grammar cannot admit, with nothing under it left to cancel: the document
      // was already off the grammar. Nothing is removed, and the caller is told so.
      return { target, removed: [], kept: [...kept.values()].sort(removalOrder) };
    }
  }
}

/**
 * Grow a set of removals until every container it empties is dealt with.
 *
 * A container left below its own `minItems` or `minProperties` goes with its contents; a
 * container the schema *requires* of its parent cannot, so the parent goes instead — which is
 * how removing the one member of a parameter binding removes the binding and not its `members`
 * list. Answers the container that could not be dealt with at all, or `null`.
 */
function collapse(context: EditContext, removed: Map<string, Path>): Path | null {
  for (let round = 0; round < 1024; round += 1) {
    const containers = new Map<string, Path>();
    for (const place of removed.values()) {
      if (place.length === 0) continue;
      const holder = parentOf(place);
      containers.set(pointerOf(holder), holder);
    }
    let changed = false;
    for (const holder of [...containers.values()].sort(removalOrder)) {
      if (goes(removed, holder)) continue;
      const node = nodeAt(context.tree, holder);
      if (node === undefined) continue;
      const children = childrenOf(node, holder);
      const survivors = children.filter((place) => !goes(removed, place));
      const constraints = context.shapes.constraintsOf(shapeAt(context, holder));
      const least = isJsonArray(node) ? constraints.minItems : constraints.minProperties;
      if (survivors.length >= least) continue;
      let victim = holder;
      while (victim.length > 0 && !removable(context, victim)) victim = parentOf(victim);
      if (victim.length === 0) return holder;
      if (removed.has(pointerOf(victim))) continue;
      removed.set(pointerOf(victim), victim);
      changed = true;
    }
    if (!changed) return null;
  }
  return null;
}

/** Whether a place goes: it is removed, or something above it is. */
function goes(removed: ReadonlyMap<string, Path>, place: Path): boolean {
  if (removed.has(pointerOf(place))) return true;
  for (const other of removed.values()) {
    if (isStrictlyUnder(place, other)) return true;
  }
  return false;
}

/** Whether a place may be taken out of its container at all. */
function removable(context: EditContext, place: Path): boolean {
  if (place.length === 0) return false;
  const step = lastOf(place);
  if (typeof step === 'number') return true;
  return !context.shapes.constraintsOf(shapeAt(context, parentOf(place))).required.has(step);
}

/** The places of a container's children. */
function childrenOf(node: JsonValue, path: Path): Path[] {
  if (isJsonArray(node)) return node.map((_, position) => child(path, position));
  if (isJsonObject(node)) return node.members.map((member) => child(path, member.name));
  return [];
}

/** Whether `inner` is strictly below `outer`. */
function isStrictlyUnder(inner: Path, outer: Path): boolean {
  return inner.length > outer.length && outer.every((step, position) => inner[position] === step);
}

/**
 * The order removals are made in: deepest first and, inside one array, the highest index first,
 * so that a removal never moves the place of one still to come.
 */
function removalOrder(left: Path, right: Path): number {
  if (left.length !== right.length) return right.length - left.length;
  for (const [position, step] of left.entries()) {
    const other = right[position];
    if (step === other) continue;
    if (typeof step === 'number' && typeof other === 'number') return other - step;
    return String(other) < String(step) ? -1 : 1;
  }
  return 0;
}

/** A name the map has not: `attn`, then `attn_2`, `attn_3`. */
export function unique(proposed: string, taken: readonly string[]): string {
  if (!taken.includes(proposed)) return proposed;
  for (let suffix = 2; ; suffix += 1) {
    const candidate = `${proposed}_${String(suffix)}`;
    if (!taken.includes(candidate)) return candidate;
  }
}

/**
 * Whether two values denote the same thing.
 *
 * A number's lexeme is a hint and its value is the meaning (feature 0.3), but its float-ness is
 * meaning too — V3 reads `32.0` as a real and `32` as a cardinality — so both are compared and
 * the lexeme is not.
 */
export function sameValue(left: JsonValue, right: JsonValue): boolean {
  if (isJsonNumber(left) || isJsonNumber(right)) {
    if (!isJsonNumber(left) || !isJsonNumber(right)) return false;
    return Object.is(left.value, right.value) && left.real === right.real;
  }
  if (isJsonArray(left) || isJsonArray(right)) {
    if (!isJsonArray(left) || !isJsonArray(right) || left.length !== right.length) return false;
    return left.every((item, position) => sameValue(item, right[position] as JsonValue));
  }
  if (isJsonObject(left) || isJsonObject(right)) {
    if (!isJsonObject(left) || !isJsonObject(right)) return false;
    if (left.members.length !== right.members.length) return false;
    return left.members.every((member, position) => {
      const other = right.members[position];
      return other !== undefined && other.name === member.name && sameValue(member.value, other.value);
    });
  }
  return left === right;
}

/**
 * Replace every member of the document's root, as one named command.
 *
 * The gesture behind it is always the same shape: a whole document arrived from outside the
 * tree — the file a Revert reads back, the draft a Restore puts in, the text feature 2.17's
 * source view parsed — and the tree has to become it *without* the history being thrown away.
 * So it is a command like any other (D13): the patches are the log's, `Undo` names it, and a
 * revert the reader did not want costs one Ctrl+Z.
 *
 * It moves nothing — `moves` is empty — and that is a statement rather than an omission: what
 * moved between two whole documents cannot be read off them. A rename typed into the source
 * view is therefore not a rename to the sidecar, whose keys are pruned of what no longer
 * resolves and a line written for each (D6). The alternative would be to infer a rename from a
 * text diff, which is a second reading of what a rename is.
 */
export function replaceRoot(tree: JsonObject, label: string): Command {
  const members = tree.members.map((member) => ({ name: member.name, value: member.value }));
  return {
    label,
    moves: [],
    edit(draft) {
      draft.members = members.map((member) => ({
        name: member.name,
        value: asDraftValue(member.value),
      }));
    },
  };
}

function getMemberValue(node: JsonObject, name: string): JsonValue | undefined {
  return node.members.find((member) => member.name === name)?.value;
}

function demand(values: Readonly<Record<string, JsonValue>>, name: string): JsonValue {
  const value = values[name];
  if (value === undefined) throw new EditError(`no value for '${name}'`);
  return value;
}

function position(order: readonly string[], name: string): number {
  const found = order.indexOf(name);
  return found < 0 ? order.length : found;
}

/** Write a value at a place of the draft, whether the place is a member or an item. */
function write(draft: MutableObject, path: Path, value: JsonValue): void {
  const step = lastOf(path);
  const holder = draftAt(draft, parentOf(path));
  if (typeof step === 'number') {
    if (!isMutableArray(holder)) throw new EditError(`${pointerOf(parentOf(path))} is not an array`);
    holder[step] = asDraftValue(value);
    return;
  }
  if (!isMutableObject(holder)) throw new EditError(`${pointerOf(parentOf(path))} is not an object`);
  if (memberIndex(holder, step) < 0) throw new EditError(`no member at ${pointerOf(path)}`);
  setMember(holder, step, asDraftValue(value));
}

