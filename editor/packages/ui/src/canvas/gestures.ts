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
  complementOf,
  compositionMove,
  COMPOSITION_MAP,
  FED_END,
  GUARD,
  instanceSkeleton,
  jsonObject,
  previousIteration,
  PRODUCING_END,
  proposedFamily,
  proposedName,
  reachedIteration,
  ROOT_INSTANCES,
  SCOPED_VALUES,
  scopedEndpoint,
  SITE_MAP,
  VALUE_BINDINGS,
  valueEndpoint,
  type DrillIndex,
  type FoldedHandle,
} from '@tensorspine/lang';
import type { JsonObject, JsonValue } from '@tensorspine/lang';
import {
  addToMap,
  asDraftValue,
  connect,
  draftAt,
  EditError,
  isMutableArray,
  isMutableObject,
  lastOf,
  memberIndex,
  objectAt,
  objectDraftAt,
  parentOf,
  pointerOf,
  remove,
  removeMember,
  rename,
  setMember,
  setMemberAt,
  unique,
  type Command,
  type EditContext,
  type Path,
  type PathMove,
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

// ---------------------------------------------------------------------------------------------
// The drill-in's own gestures — plan §4.8 and §4.20, feature 2.14.
// ---------------------------------------------------------------------------------------------

/** The place a composition's sites are written at. */
export function sitesOf(composition: string): Path {
  return [...COMPOSITION_MAP, composition, ...SITE_MAP] as Path;
}

/** The place a composition's scoped value rules are written at. */
export function scopedValuesOf(composition: string): Path {
  return [...COMPOSITION_MAP, composition, ...SCOPED_VALUES] as Path;
}

/**
 * Drop a primitive into a drill-in: §4.7's first gesture, one level down.
 *
 * > Drop a primitive from the palette | adds an instance (root canvas) or a **site** (drill-in)
 *
 * The same skeleton, the same proposals (§9 Q4); what differs is the map it is written into.
 */
export function addSite(
  edit: EditContext,
  request: { readonly composition: string; readonly primitive: string; readonly version: string; readonly name?: string },
): Command & { readonly name: string } {
  const path = sitesOf(request.composition);
  const name = unique(request.name ?? proposedName(request.primitive), namesUnder(edit, path));
  return {
    ...addToMap(edit, {
      path,
      name,
      values: membersOf(instanceSkeleton(request.primitive, request.version)),
      label: `Add site ${name}`,
    }),
    name,
  };
}

/** One end of a scoped connection: a site of the composition, at the current indices unless told. */
export interface ScopedEnd {
  readonly site: string;
  readonly port: string;
  /** The index overrides — what "Connect from previous iteration…" puts on the producing end. */
  readonly indices?: readonly DrillIndex[];
}

/**
 * Connect two sites inside a composition — §4.7's "or is a scoped rule in a drill-in".
 *
 * The rule goes into the composition's own `bindings/values`, which the grammar makes optional: a
 * composition that has none gains the map with this rule in it, written where the schema puts it.
 * The endpoints are the core's (`scopedEndpoint`), as every endpoint the canvas writes is.
 */
export function connectScoped(
  edit: EditContext,
  request: { readonly composition: string; readonly from: ScopedEnd; readonly to: ScopedEnd },
): Command {
  const { composition, from, to } = request;
  const values = {
    [PRODUCING_END]: scopedEndpoint(from.site, from.port, from.indices ?? []),
    [FED_END]: scopedEndpoint(to.site, to.port, to.indices ?? []),
  };
  const label = `Connect ${from.site}.${from.port} → ${to.site}.${to.port}`;
  const name = `${to.site}.${to.port}`;
  const map = scopedValuesOf(composition);
  if (objectAt(edit.tree, map) !== undefined) {
    return connect(edit, { path: map, name, values, into: FED_END, label });
  }
  // No `bindings` at all, or no `values` in it: the map is created holding this one rule. The
  // grammar admits a composition with no bindings (`scoped_bindings` is optional), so this is the
  // first rule of one and not a repair.
  const bindings = [...COMPOSITION_MAP, composition, SCOPED_VALUES[0] as string] as Path;
  const rule = jsonObject(
    Object.entries(values).map(([member, value]) => ({ name: member, value })),
  );
  if (objectAt(edit.tree, bindings) === undefined) {
    return setMemberAt(edit, {
      path: [...COMPOSITION_MAP, composition] as Path,
      name: SCOPED_VALUES[0] as string,
      value: jsonObject([
        { name: SCOPED_VALUES[1] as string, value: jsonObject([{ name, value: rule }]) },
      ]),
      label,
    });
  }
  return setMemberAt(edit, {
    path: bindings,
    name: SCOPED_VALUES[1] as string,
    value: jsonObject([{ name, value: rule }]),
    label,
  });
}

/**
 * "Connect from previous iteration…" (§4.8): the same connection, with the override on its source.
 *
 * > writes the override `{"op": "subtract", "args": [{"index": "layer"}, {"literal": 1}]}` and
 * > proposes the guard `layer ≥ 1` (a proposal the user confirms — the guard states a fact of its
 * > own, as the model guide says).
 *
 * The override is written here; the guard is *proposed* and written by {@link proposeGuard} when
 * the author accepts it, which is what "confirms" means.
 */
export function connectFromPreviousIteration(
  edit: EditContext,
  request: { readonly composition: string; readonly index: string; readonly from: ScopedEnd; readonly to: ScopedEnd },
): Command {
  const command = connectScoped(edit, {
    composition: request.composition,
    from: {
      ...request.from,
      indices: [{ name: request.index, written: previousIteration(request.index) }],
    },
    to: request.to,
  });
  return {
    ...command,
    label: `Connect ${request.from.site}.${request.from.port} from the previous iteration`,
  };
}

/** The guard "Connect from previous iteration…" proposes, written when the author accepts it. */
export function proposeGuard(edit: EditContext, path: Path, index: string): Command {
  return setMemberAt(edit, {
    path,
    name: GUARD,
    value: reachedIteration(index),
    label: `Guard ${String(lastOf(path))} with ${index} ≥ 1`,
  });
}

/**
 * "Duplicate with complementary guard" (§4.20, S5): the periodic pattern, in one gesture.
 *
 * > Periodic patterns (Gemma 3n's `attn` / `attn_full`, Qwen's `attn` / `gdn`): two sites with
 * > complementary guards; a site's context menu offers "Duplicate with complementary guard".
 *
 * The copy is the duplicate of §4.7 with one member replaced: the guard, negated. It is **not**
 * the comparison flipped — `not (layer < 10)` and not `layer ≥ 10` — because rewriting a
 * comparison is a logical transformation the editor would be inventing; the core's
 * {@link complementOf} says so, and the author edits the condition like any other.
 */
export function duplicateWithComplementaryGuard(
  edit: EditContext,
  path: Path,
): Command & { readonly name: string } {
  const written = objectAt(edit.tree, path);
  if (written === undefined) throw new EditError(`${pointerOf(path)} is nothing to duplicate`);
  const guard = written.members.find((one) => one.name === GUARD);
  if (guard === undefined) {
    throw new EditError(`${pointerOf(path)} carries no guard to complement`);
  }
  const copy = jsonObject(
    written.members.map((one) =>
      one.name === GUARD ? { name: one.name, value: complementOf(one.value) } : one,
    ),
  );
  const map = parentOf(path);
  const step = path[path.length - 1];
  const name = unique(`${String(step)}_alt`, namesUnder(edit, map));
  return {
    ...addToMap(edit, {
      path: map,
      name,
      values: membersOf(copy),
      label: `Duplicate ${String(step)} with the complementary guard`,
    }),
    name,
  };
}

/**
 * "Extract to Composition" and "Add to Composition…" — §4.20's move, as one command (D13).
 *
 * The *reading* is the core's ({@link compositionMove}): which rules move inside, which endpoints
 * are rewritten, what the composition is written as. What is here is the one thing a command owes
 * the rest of the editor — that all of it is **one** edit, so that one undo puts the document
 * back, and that the places which moved are declared, so the layout sidecar's keys follow (D6).
 */
export function moveIntoComposition(
  edit: EditContext,
  request: { readonly instances: readonly string[]; readonly composition: string; readonly index?: string },
): (Command & { readonly notes: readonly string[] }) | null {
  const plan = compositionMove(edit.tree, {
    instances: request.instances,
    composition: request.composition,
    ...(request.index === undefined ? {} : { index: request.index }),
  });
  if (plan === null) return null;
  const moves: PathMove[] = plan.moved.map((one) => ({
    from: [...ROOT_INSTANCES, one.from] as Path,
    to: [...sitesOf(plan.composition), one.name] as Path,
  }));
  const label = plan.created
    ? `Extract ${String(plan.moved.length)} instances to composition ${plan.composition}`
    : `Add ${String(plan.moved.length)} instances to composition ${plan.composition}`;
  return {
    label,
    moves,
    notes: plan.notes,
    edit(draft) {
      // The composition first, so that everything below is written into a place that exists.
      const compositions = objectDraftAt(draft, COMPOSITION_MAP);
      if (plan.definition !== null) {
        const order = ['indices', 'families', 'instances'] as const;
        setMember(
          compositions,
          plan.composition,
          asDraftValue(
            jsonObject(
              order.map((name) => ({ name, value: (plan.definition ?? {})[name] as JsonValue })),
            ),
          ),
        );
      }
      const sites = objectDraftAt(draft, sitesOf(plan.composition));
      const roots = objectDraftAt(draft, ROOT_INSTANCES);
      for (const one of plan.moved) {
        setMember(sites, one.name, asDraftValue(one.value));
        removeMember(roots, one.from);
      }
      // The rules that move inside leave their maps and are written under the composition's.
      const composition = objectDraftAt(draft, [...COMPOSITION_MAP, plan.composition] as Path);
      for (const rule of plan.absorbed) {
        removeMember(objectDraftAt(draft, rule.from.slice(0, -1)), rule.from[2] as string);
        if (memberIndex(composition, SCOPED_VALUES[0] as string) < 0) {
          setMember(composition, SCOPED_VALUES[0] as string, asDraftValue(jsonObject([])));
        }
        const bindings = objectDraftAt(
          draft,
          [...COMPOSITION_MAP, plan.composition, SCOPED_VALUES[0] as string] as Path,
        );
        if (memberIndex(bindings, rule.map) < 0) {
          setMember(bindings, rule.map, asDraftValue(jsonObject([])));
        }
        setMember(
          objectDraftAt(draft, [
            ...COMPOSITION_MAP,
            plan.composition,
            SCOPED_VALUES[0] as string,
            rule.map,
          ] as Path),
          rule.name,
          asDraftValue(rule.value),
        );
      }
      // The endpoints that stay outside now name the site at an index.
      for (const place of plan.rewritten) {
        const path = place.path;
        const step = path[path.length - 1];
        const holder = draftAt(draft, path.slice(0, -1));
        if (isMutableArray(holder) && typeof step === 'number') {
          holder[step] = asDraftValue(place.value);
          continue;
        }
        if (isMutableObject(holder) && typeof step === 'string') {
          setMember(holder, step, asDraftValue(place.value));
        }
      }
    },
  };
}
