/**
 * What `Add quantity`, `Add constant`, `Expose as input…` and `Expose as output…` write.
 *
 * §4.4's Model menu carries the four commands; §4.15 says where two of them come from — "right-click
 * the port ▸ *Expose as input…* (name proposed from the port). Creating an output: right-click an
 * output port ▸ *Expose as output…* with the generative toggle" — and §4.16 says the other two are
 * beside the table they add to ("**Add quantity** with the type kinds from the schema").
 *
 * **None of them knows what it is adding.** The map is found by what `presentation.json` says it
 * *declares* (feature 2.2's binding, the same one a rename and a delete read); the value written is
 * the blank of the entry's own definition ({@link blankValue}); the name proposed is that word, or
 * the port's, uniquified against the names the map already carries. A grammar that grew another
 * map of declarations would be addable here with no line changing — and the four commands of §4.4
 * are four *words*, which is what their identities carry.
 *
 * **The endpoint is the schema's, and so is the member it goes in.** A public input holds a list of
 * `value_endpoint`s and a public output holds one, and neither member is named here: the definition
 * an endpoint *is* comes from the core's own constants for a value binding's producing end
 * (`VALUE_BINDINGS`, `PRODUCING_END`), and the member of an interface that takes it is the one
 * whose place has that definition in its chain. The endpoint itself is `valueEndpoint`'s — the core
 * writes every selector of the language (feature 2.9's rule).
 */
import {
  isJsonObject,
  PRODUCING_END,
  valueEndpoint,
  VALUE_BINDINGS,
  type FoldedHandle,
  type JsonObject,
  type JsonValue,
} from '@tensorspine/lang';
import {
  addToMap,
  nodeAt,
  objectAt,
  unique,
  type Command,
  type EditContext,
  type Path,
} from '@tensorspine/store';

import { chainOf, type FormContext } from '../forms/index.js';
import type { ReferenceRule } from '../presentation/index.js';
import { blankValue } from './skeleton.js';
import { shapeAt } from './places.js';

/** One map of the document that declares something, as `presentation.json` says it does. */
export interface DeclaringMap {
  readonly path: Path;
  /** The word the binding gives what it declares: `quantity`, `constant`, `input`, `output`. */
  readonly declares: string;
  /** Where a name declared here is referred to — the binding's own rules (feature 2.2). */
  readonly refers: readonly ReferenceRule[];
}

/**
 * Every map of a document that declares something, at the two depths a document has them.
 *
 * The rule is feature 2.7's own reading of the outline: a member of the root that declares is one,
 * and a member of a *grouping* — an object that carries containers and nothing of its own, which
 * is what `interfaces` and `bindings` are — is one too. Nothing below that: a composition's sites
 * and indices are declarations inside a declaration, and the sheet that adds one is the
 * composition's own.
 */
export function declaringMaps(context: FormContext, tree: JsonObject, role: string): DeclaringMap[] {
  const found: DeclaringMap[] = [];
  const root = context.shapes.root(role);
  const walk = (path: Path, shape: ReturnType<typeof context.shapes.root>, depth: number): void => {
    for (const name of context.shapes.propertyOrder(shape)) {
      const member = context.shapes.member(shape, name);
      const at = [...path, name] as Path;
      if (nodeAt(tree, at) === undefined) continue;
      const binding = context.bindings.firstOf(chainOf(member).map((place) => place.anchor));
      if (binding?.declares !== undefined) {
        found.push({ path: at, declares: binding.declares, refers: binding.refers ?? [] });
        continue;
      }
      if (depth < 1) walk(at, member, depth + 1);
    }
  };
  walk([], root, 0);
  return found;
}

/** The map that declares that word, where the document has one. */
export function mapDeclaring(
  context: FormContext,
  tree: JsonObject,
  role: string,
  declares: string,
): DeclaringMap | null {
  return declaringMaps(context, tree, role).find((one) => one.declares === declares) ?? null;
}

/**
 * The map whose declarations the canvas draws as terminals on one side — §4.15's two.
 *
 * `presentation.json` says that a `public_input` is a terminal on the left and a `public_output`
 * one on the right (feature 2.9 draws them from the same two bindings), and that is the only thing
 * that tells the two maps apart without naming either: the word each *declares* is the author's
 * label, the side is where the editor puts it.
 */
export function interfaceMap(
  context: FormContext,
  tree: JsonObject,
  role: string,
  side: string,
): Path | null {
  for (const map of declaringMaps(context, tree, role)) {
    const shape = shapeAt(context.shapes, map.path, role);
    const entry = context.shapes.step(shape, map.declares);
    const binding = context.bindings.firstOf(chainOf(entry).map((place) => place.anchor));
    if (binding?.side === side) return map.path;
  }
  return null;
}

/**
 * The names a member that **refers** to a declaration can be filled from — §4.11's own row:
 * "stream (select of the document's streams, or *introduces its own*)".
 *
 * Which members those are is `presentation.json`'s `refers` (feature 2.2): a rule says that a
 * name written under the member `stream` refers to what `interfaces/inputs` declares, so the
 * names offered there are that map's own. The same reading serves every other referring member
 * the grammar has — the `constant` of a constant binding, the `instance` of a root selector —
 * without one of them being named here.
 *
 * **A list, never a limit.** The names are what the document *has*; §4.15's own example of a
 * refusal is an input joining a stream nobody declares, which the core reports (V1) and the editor
 * does not prevent (Q5). The interface draws them as suggestions beside a field for that reason.
 *
 * §4.15 says the stream select "lists the streams D2 reports (or the inputs' names before a
 * derivation)". What is offered is the second: every stream **is** an introducing input's name
 * (§2.3 — "absent, the input introduces the stream named after it"), so the inputs' names are the
 * streams plus the names of the inputs that join one, and the difference is a name the core
 * refuses rather than one the editor must hide.
 */
export function referentNames(
  context: FormContext,
  tree: JsonObject,
  role: string,
): (tag: string) => readonly string[] {
  const byTag = new Map<string, Path>();
  for (const map of declaringMaps(context, tree, role)) {
    for (const rule of map.refers) {
      if (rule.kind === KEY) continue;
      if (!byTag.has(rule.tag)) byTag.set(rule.tag, map.path);
    }
  }
  return (tag) => {
    const at = byTag.get(tag);
    if (at === undefined) return [];
    const held = nodeAt(tree, at);
    return held !== undefined && isJsonObject(held) ? held.members.map((one) => one.name) : [];
  };
}

/** The reference kind that is a map's own name rather than a name written under a member. */
const KEY = 'key';

/** What an addition is asked for. */
export interface AddRequest {
  readonly context: FormContext;
  readonly role: string;
  /** The map the declaration goes into. */
  readonly path: Path;
  /** The name proposed; the word the map declares otherwise, uniquified either way. */
  readonly name?: string;
  /** Members to write over the blank: the endpoint of an interface, and nothing else today. */
  readonly members?: Readonly<Record<string, JsonValue>>;
  /** What the command is called in the Edit menu; built from the word otherwise. */
  readonly label?: string;
}

/**
 * Add one declaration to a map of them — D5's skeleton, one level down from a new document.
 *
 * The value is the blank of the entry's own definition: its required members, each the first
 * value its own schema admits. For a quantity that is `{"type": {"kind": "cardinality"},
 * "source": {"kind": "literal", "value": 0}}` — on the grammar, and what V3 then says about the
 * value is the semantic stage's report, which is exactly D5's arrangement.
 */
export function addDeclaration(
  edit: EditContext,
  request: AddRequest,
): Command & { readonly name: string } {
  const { context, path } = request;
  const declares = declaresAt(context, path, request.role);
  const shape = shapeAt(context.shapes, path, request.role);
  // The value shape of a map is the shape of any of its entries: the walker steps into it by a
  // name, and which name it is does not change the definition.
  const entry = context.shapes.step(shape, request.name ?? declares);
  const blank = blankValue(context, entry);
  const name = unique(request.name ?? declares, namesUnder(edit, path));
  const values: Record<string, JsonValue> = {};
  if (blank !== null && typeof blank === 'object' && 'members' in blank) {
    for (const member of (blank).members) values[member.name] = member.value;
  }
  for (const [member, value] of Object.entries(request.members ?? {})) values[member] = value;
  return {
    ...addToMap(edit, {
      path,
      name,
      values,
      label: request.label ?? `Add ${declares} ${name}`,
    }),
    name,
  };
}

/**
 * `Expose as input…` / `Expose as output…` — §4.15, from a port of the canvas.
 *
 * The endpoint is the core's (`valueEndpoint`), the member it goes in is the one whose definition
 * an endpoint is, and the name is proposed from the port. Nothing about the *kind* of interface is
 * decided here: the caller names the map — the one that declares an input, or the one that
 * declares an output — and the rest is that map's own definition.
 */
export function exposeAt(
  edit: EditContext,
  request: AddRequest & { readonly handle: FoldedHandle },
): Command & { readonly name: string } {
  const { context, path, handle } = request;
  const shape = shapeAt(context.shapes, path, request.role);
  const entry = context.shapes.step(shape, handle.port);
  const member = endpointMember(context, entry, request.role);
  const endpoint = valueEndpoint(handle);
  return addDeclaration(edit, {
    ...request,
    name: request.name ?? handle.port,
    members:
      member === null
        ? (request.members ?? {})
        : {
            ...(request.members ?? {}),
            [member.name]: member.listed ? [endpoint] : endpoint,
          },
    label: request.label ?? `Expose ${handle.name}.${handle.port}`,
  });
}

/** The member of an interface that holds a value endpoint, and whether it holds a list of them. */
export function endpointMember(
  context: FormContext,
  entry: ReturnType<typeof context.shapes.root>,
  role: string,
): { readonly name: string; readonly listed: boolean } | null {
  const wanted = new Set(endpointAnchors(context, role));
  if (wanted.size === 0) return null;
  for (const name of context.shapes.propertyOrder(entry)) {
    const member = context.shapes.member(entry, name);
    if (chainOf(member).some((place) => wanted.has(place.anchor))) {
      return { name, listed: false };
    }
    const item = context.shapes.item(member, 0);
    if (chainOf(item).some((place) => wanted.has(place.anchor))) return { name, listed: true };
  }
  return null;
}

/** What a value endpoint *is*, read from the producing end of a value binding (the core's names). */
function endpointAnchors(context: FormContext, role: string): string[] {
  let shape = context.shapes.root(role);
  for (const step of VALUE_BINDINGS) shape = context.shapes.step(shape, step);
  // Any name steps into the map; the rule's own definition is what is wanted, not this rule.
  const rule = context.shapes.step(shape, PRODUCING_END);
  const end = context.shapes.member(rule, PRODUCING_END);
  return chainOf(end).map((place) => place.anchor);
}

/** What the map at a place declares, in `presentation.json`'s word. */
function declaresAt(context: FormContext, path: Path, role: string): string {
  const shape = shapeAt(context.shapes, path, role);
  return (
    context.bindings.firstOf(chainOf(shape).map((place) => place.anchor))?.declares ??
    context.bindings.firstOf(shape.all.map((place) => place.anchor))?.declares ??
    ''
  );
}

/** The names a map already carries. */
function namesUnder(edit: EditContext, path: Path): string[] {
  const map = objectAt(edit.tree, path);
  return map === undefined ? [] : map.members.map((member) => member.name);
}
