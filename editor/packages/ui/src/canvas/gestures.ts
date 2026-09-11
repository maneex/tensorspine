/**
 * Every gesture of §4.7's interactions table, as a command of the store (2.1, D13).
 *
 * | Gesture | Effect (a JSON edit) |
 * |---|---|
 * | Drop a primitive from the palette | adds an instance, D5's placeholders |
 * | Drag output handle → input handle | the edge is added, never refused (Q5); an input already fed has its older edge replaced |
 * | Select + Delete | removes the item and everything that names it, after a confirmation listing the cascade |
 * | Right-click ▸ Rename / Duplicate / Delete | the same three commands |
 *
 * **Nothing here refuses a gesture for a semantic reason.** Q5 is the feature's spine: "the
 * author wires first and fixes afterwards; the core's verdict is shown *during* the drag and
 * lands in Problems *on* the drop, and never blocks it." So a connection is made whatever
 * {@link check} said, and the verdict travels beside the command for the toast to name.
 *
 * **Nothing here writes a member of the grammar either.** What a `value_endpoint` looks like is
 * the schema's, and the core writes it (`valueEndpoint`); what a new instance looks like is D5's,
 * and the core writes that too (`instanceSkeleton`). What is decided here is what §4.7 decides:
 * the *name* a new binding is proposed under, and which occurrences a delete cascades over —
 * and the second is read from `presentation.json`, never remembered.
 */
import {
  FED_END,
  instanceSkeleton,
  PRODUCING_END,
  proposedFamily,
  proposedName,
  ROOT_INSTANCES,
  VALUE_BINDINGS,
  valueEndpoint,
  type FoldedHandle,
} from '@tensorspine/lang';
import type { JsonObject, JsonValue } from '@tensorspine/lang';
import {
  addToMap,
  connect,
  objectAt,
  parentOf,
  pointerOf,
  remove,
  rename,
  unique,
  type Command,
  type EditContext,
  type Path,
  type SchemaShapes,
} from '@tensorspine/store';

import { referenceSelectors, type Presentation, type Scope } from '../presentation/index.js';

/** What every gesture is given beside the edit context. */
export interface GestureContext {
  readonly shapes: SchemaShapes;
  readonly bindings: Presentation;
  /** The role the document is read under; the model's unless a caller says otherwise. */
  readonly role?: string;
}

/** The model's own role. */
const MODEL = 'model';

/**
 * The declaration a place belongs to: the map it is written in, and the scope that map is read
 * against.
 *
 * A rename and a delete need to know which occurrences of a name refer to *this* declaration, and
 * that pairing is `presentation.json`'s (feature 2.2). The map is the place's parent, and the
 * scope — where the binding names one — is the declaration that map is written inside: a
 * composition, for a site and for an index.
 */
export function declarationAt(
  context: GestureContext,
  path: Path,
): { readonly anchor: string; readonly scope?: Scope } | null {
  const role = context.role ?? MODEL;
  const map = parentOf(path);
  let shape = context.shapes.root(role);
  for (const step of map) shape = context.shapes.step(shape, step);
  for (const place of shape.all) {
    const binding = context.bindings.at(place.anchor);
    if (binding === undefined || binding.declares === undefined) continue;
    if (binding.scope === undefined) return { anchor: place.anchor };
    // The scope is the enclosing declaration: the composition a site is written in, which is the
    // map's own parent's last step. The caller supplies it because the document is where a scope
    // is (feature 2.2's rule).
    const owner = parentOf(map);
    const name = owner[owner.length - 1];
    if (typeof name !== 'string') return { anchor: place.anchor };
    return { anchor: place.anchor, scope: { path: owner, name } };
  }
  return null;
}

/** The selectors that find every reference to the declaration at a place. */
export function referencesTo(context: GestureContext, path: Path): ReturnType<typeof referenceSelectors> {
  const found = declarationAt(context, path);
  if (found === null) return [];
  const binding = context.bindings.at(found.anchor);
  return binding === undefined ? [] : referenceSelectors(binding, found.scope);
}

/** What the map at a place declares, in the word `presentation.json` gives it. */
export function declaresAt(context: GestureContext, path: Path): string {
  const found = declarationAt(context, path);
  if (found === null) return '';
  return context.bindings.at(found.anchor)?.declares ?? '';
}

/**
 * Drop a primitive onto the root canvas: §4.7's first gesture, D5's skeleton.
 *
 * > Adding an instance writes `{primitive, arguments: {}, families: [<proposed>]}` — on the
 * > grammar, refused by V2 (missing required arguments), which the Problems panel lists with
 * > navigation.
 *
 * The name and the family are the proposals §9 Q4 decided on — the primitive name's last and
 * first segments — uniquified against what the document already has.
 */
export function addInstance(
  edit: EditContext,
  request: { readonly primitive: string; readonly version: string; readonly name?: string },
): Command & { readonly name: string } {
  const taken = namesUnder(edit, ROOT_INSTANCES);
  const name = unique(request.name ?? proposedName(request.primitive), taken);
  // The skeleton is the core's (D5's own sentence); what reaches `addToMap` is its members, so
  // that the new instance is written in the schema's property order like every other member the
  // store adds — and so that no member of the grammar is named here (§1).
  return {
    ...addToMap(edit, {
      path: ROOT_INSTANCES,
      name,
      values: membersOf(instanceSkeleton(request.primitive, request.version)),
      label: `Add instance ${name}`,
    }),
    name,
  };
}

/** The family a dropped primitive proposes — exported so a sheet can show it beside the name. */
export { proposedFamily, proposedName };

/**
 * Connect an output handle to an input handle: §4.7's second gesture.
 *
 * The binding is named `<to>.<port>`, uniquified — "the new binding is named `<to>.<port>`
 * (uniquified)" — and `connect` replaces the edge an already-fed input had, which is Q5's answer
 * to V7: "an input already fed has its older edge replaced, undoable, the toast naming it".
 */
export function connectHandles(
  edit: EditContext,
  from: FoldedHandle,
  to: FoldedHandle,
): Command {
  return connect(edit, {
    path: VALUE_BINDINGS,
    name: `${to.name}.${to.port}`,
    values: {
      [PRODUCING_END]: valueEndpoint(from),
      [FED_END]: valueEndpoint(to),
    },
    into: FED_END,
    label: `Connect ${from.name}.${from.port} → ${to.name}.${to.port}`,
  });
}

/** Delete the item at a place and everything that names it (§4.7, D13). */
export function removeAt(context: GestureContext, edit: EditContext, path: Path): Command {
  const what = declaresAt(context, path);
  const name = path[path.length - 1];
  return remove(edit, {
    path,
    references: referencesTo(context, path),
    label: `Delete ${what === '' ? String(name) : `${what} ${String(name)}`}`,
  });
}

/** Rename the item at a place, rewriting every reference (§4.7, plan §3). */
export function renameAt(
  context: GestureContext,
  edit: EditContext,
  path: Path,
  to: string,
): Command {
  const what = declaresAt(context, path);
  const name = path[path.length - 1];
  return rename(edit, {
    path,
    to,
    references: referencesTo(context, path),
    label: `Rename ${what === '' ? String(name) : `${what} ${String(name)}`} to ${to}`,
  });
}

/**
 * Duplicate the item at a place: the same declaration under a name nothing else has.
 *
 * The copy is the declaration as written, arguments and all — which is what an author duplicating
 * a site wants — and it is *not* connected: an edge names an instance, and a copy that inherited
 * its original's edges would feed two consumers from one producer without anybody asking.
 */
export function duplicateAt(context: GestureContext, edit: EditContext, path: Path): Command & { readonly name: string } {
  const map = parentOf(path);
  const step = path[path.length - 1];
  if (typeof step !== 'string') throw new Error(`${pointerOf(path)} is not a named member`);
  const written = objectAt(edit.tree, path);
  if (written === undefined) throw new Error(`${pointerOf(path)} is nothing to duplicate`);
  const name = unique(`${step}_copy`, namesUnder(edit, map));
  const what = declaresAt(context, path);
  return {
    ...addToMap(edit, {
      path: map,
      name,
      values: membersOf(written),
      label: `Duplicate ${what === '' ? step : `${what} ${step}`}`,
    }),
    name,
  };
}

/** The names a map already carries. */
function namesUnder(edit: EditContext, path: Path): string[] {
  const map = objectAt(edit.tree, path);
  return map === undefined ? [] : map.members.map((member) => member.name);
}

/** An object node's members, as `addToMap` takes them. */
function membersOf(node: JsonObject): Record<string, JsonValue> {
  const values: Record<string, JsonValue> = {};
  for (const member of node.members) values[member.name] = member.value;
  return values;
}
