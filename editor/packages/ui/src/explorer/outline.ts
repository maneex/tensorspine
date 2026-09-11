/**
 * The outline of an open document — plan §4.5, drawn on S1's left panel.
 *
 * > An outline of the document generated from the top-level schema (the order of `properties` of
 * > the root object).
 *
 * So there is no list of sections anywhere below: the rows are the document's own containers and
 * the declarations in them, walked against the schema, in the order the schema writes its
 * members. Two rules make the whole walk:
 *
 *  - **a container's children are its entries** — the members of a map, the items of a list;
 *  - **an entry's children are the declarations it holds** — the entries of the containers under
 *    it that `presentation.json` says *declare* something (a site, an index), flattened into the
 *    entry, because a composition's sites are what the outline is for.
 *
 * Everything else follows from what the schema asserts at a place (`SchemaFacts`) and from the
 * bindings feature 2.2 keyed by schema anchors. Nothing here knows the word `quantities`, the
 * word `instance` or the word `derived`: the member names come from the schema, the kinds from
 * `declares`, the figures and the place they are written from the core.
 *
 * **What the design draws and this does not, and why.** S1 puts a composition's index in the
 * row's tail (`layer ∈ [0, 32)`) and its sites in the children. An index is a declaration like a
 * site — the same `declares` binding, the same rename that rewrites every `{"index": "layer"}` —
 * and §4.5 asks that the tree support rename and delete of its items, which a tail cannot. So
 * both are rows, and which of two declaring maps to fold away is a question no schema answers.
 */
import {
  isJsonArray,
  isJsonNumber,
  isJsonObject,
  mergeFacts,
  pyStr,
  QUANTITIES,
  quantityReadings,
  toPython,
  type JsonObject,
  type JsonValue,
  type PyRecord,
  type SchemaFacts,
} from '@tensorspine/lang';
import {
  child,
  nameMember,
  pointerOf,
  tagOf,
  versionMember,
  type Path,
  type SchemaShapes,
  type Shape,
} from '@tensorspine/store';

import type { Binding, Presentation } from '../presentation/index.js';

/** What a row of the outline stands for. */
export type RowKind =
  /** The document itself: its name, and the tag its schema fixes. */
  | 'document'
  /** A container the schema declares at a place: `quantities`, `bindings`, `bindings/values`. */
  | 'group'
  /** An entry of a container: a quantity, an instance, a site, a rule name, a list item. */
  | 'entry'
  /** What an open row holds that is not on screen — S1's `.quiet` line under a composition. */
  | 'note';

/** One row of the outline. */
export interface OutlineRow {
  /** The place, as an RFC 6901 pointer: the row's identity, and what expansion is keyed by. */
  readonly pointer: string;
  /** The place in the document. */
  readonly path: Path;
  /** How far in: the root is 0. */
  readonly depth: number;
  readonly kind: RowKind;
  /** What the row is called: a declaration's own name, or the schema's member name. */
  readonly label: string;
  /** Whether the label is a name the document writes — so a rename edits it (§4.5). */
  readonly named: boolean;
  /** How many entries a container holds — S1's `.n` pill. */
  readonly count?: number;
  /** What the row says beside its name: its first face member, or a summary of what it holds. */
  readonly tail?: string;
  /** The figure the core resolves for this declaration — S1's `.fig`. */
  readonly figure?: string;
  /** The marks §4.5 asks for, in the order they are written. */
  readonly marks: readonly string[];
  /** Whether a problem's pointer falls at or under this row (§4.5's red dot). */
  readonly problem: boolean;
  /** Whether the row has children of its own. */
  readonly container: boolean;
  /** Whether those children are on screen. */
  readonly open: boolean;
  /**
   * Whether they would be, with nothing toggled.
   *
   * `toggled` is a *deviation* from the default, so a caller that has to make a place visible —
   * the Problems panel, navigating to the row a problem names — cannot know which way to flip an
   * ancestor without it. {@link revealing} is what reads it.
   */
  readonly openByDefault: boolean;
  /** How the row's swatch is tinted: what it holds decides, never what it is called. */
  readonly tint?: 'group' | 'figure';
  /** What the container this row is an entry of declares — the word a command uses (§4.4). */
  readonly declares?: string;
  /** The anchor whose binding said so: where this entry's reference rules are bound (2.2). */
  readonly declaredAt?: string;
  /** The enclosing declaration a scoped reference rule is read against (a composition). */
  readonly scope?: { readonly path: Path; readonly name: string };
}

/** What {@link outlineOf} is given. */
export interface OutlineRequest {
  readonly tree: JsonObject;
  readonly shapes: SchemaShapes;
  readonly bindings: Presentation;
  /** The role the document is read under; the model's unless a caller says otherwise. */
  readonly role?: string;
  /** The rows whose openness the user has changed, by pointer — a deviation from the default. */
  readonly toggled?: ReadonlySet<string>;
  /** The filter box's text (§4.5); everything matches when it is empty. */
  readonly filter?: string;
  /** The pointers the core's problems name, so a row can carry the dot (§4.5). */
  readonly problems?: readonly string[];
  /**
   * Every row, whatever is open: what a command that has to *find* a place needs.
   *
   * The Edit menu's `Delete` is about the selection, and a selection is a place of the document
   * rather than of the tree's own state — so a command must find its row even under a group the
   * reader has closed.
   */
  readonly openAll?: boolean;
  /** The assignment a template is read under (§4.6), where the caller has one. */
  readonly assignment?: PyRecord;
  /**
   * The primitives that pin a template, by name — `primitive_library.template_primitives`, which
   * the core answers when the library is loaded (§4.5's `▣ on template instances`).
   *
   * Whether a primitive is a template is a fact about the *library* and not about the document,
   * so it is handed in rather than read out of the tree: with no library loaded the mark is
   * simply absent, which is the honest reading of a document whose bases have not been gathered.
   */
  readonly templates?: ReadonlySet<string>;
}

/** The mark a guarded row carries — §4.5's `⚑ on guarded sites`. */
export const GUARDED = '⚑';

/** The mark a shared identity carries — §4.5's `⇄ on tied/shared identities`. */
export const SHARED = '⇄';

/** The mark a template instance carries — §4.5's `▣ on template instances`. */
export const TEMPLATE = '▣';

/** What a figure the document computes rather than writes is marked with (S1's `.drv` chip). */
export const COMPUTED = 'computed';

/** The editor a condition is bound to in `presentation.json`, which is what a guard is written in. */
const CONDITION = 'condition';

/** The model's own role, as the store reads a document under it. */
const MODEL = 'model';

/** How many entries of a container a collapsed tail names before it says `…`. */
const TAIL_ENTRIES = 3;

/** The facts of a shape: what the schema asserts at the place, by definition. */
function factsOf(shape: Shape): SchemaFacts {
  return mergeFacts(shape.direct.map((place) => place.node));
}

/** The facts of every candidate, alternatives included: what the place may hold at all. */
function factsUnder(shape: Shape): SchemaFacts {
  return mergeFacts(shape.all.map((place) => place.node));
}

/** The most specific anchor of a place that carries a binding (2.2's `firstOf`, as an anchor). */
function boundAnchor(bindings: Presentation, shape: Shape): string | undefined {
  return shape.all.map((place) => place.anchor).find((anchor) => bindings.at(anchor) !== undefined);
}

/** The binding of a place. */
function bindingOf(bindings: Presentation, shape: Shape): Binding | undefined {
  return bindings.firstOf(shape.all.map((place) => place.anchor));
}

/** Whether the place holds a map the document keys by names of its own. */
function isNamedMap(facts: SchemaFacts): boolean {
  return facts.keyed && facts.named;
}

/** Whether the place holds a container at all: a map, a list, or an object of members. */
function isContainer(facts: SchemaFacts): boolean {
  return facts.holdsObject || facts.holdsArray;
}

/**
 * Whether an object is a *grouping*: it holds containers and nothing of its own.
 *
 * `bindings` and `interfaces` are such objects — four maps, two maps — and are looked through by
 * the outline and by its summaries. A `value_endpoint` is not: it holds a port name beside its
 * indices, so it is a value and not a place with things in it.
 */
function isGrouping(shapes: SchemaShapes, shape: Shape): boolean {
  const facts = factsOf(shape);
  if (!facts.holdsObject || facts.keyed || facts.holdsArray) return false;
  const members = shapes.propertyOrder(shape);
  if (members.length === 0) return false;
  return members.every((name) => isContainer(factsOf(shapes.member(shape, name))));
}

/** The entries of a container as it is written: a map's members, a list's items. */
function entriesOf(value: JsonValue): { readonly step: string | number; readonly value: JsonValue }[] {
  if (isJsonObject(value)) return value.members.map((one) => ({ step: one.name, value: one.value }));
  if (isJsonArray(value)) return value.map((one, at) => ({ step: at, value: one }));
  return [];
}

/** The member of an object, where it writes one. */
function memberOf(value: JsonValue, name: string): JsonValue | undefined {
  return isJsonObject(value) ? value.members.find((one) => one.name === name)?.value : undefined;
}

/**
 * A scalar as the outline prints it.
 *
 * The document's own numbers keep their lexemes (feature 0.3), so `1e-05` is written as the file
 * writes it and not as `JSON.stringify` would; a number the editor computed and has no lexeme for
 * is written as the core writes it (D12).
 */
function printed(value: JsonValue): string | undefined {
  if (typeof value === 'string') return value;
  if (value === true || value === false) return String(value);
  if (isJsonNumber(value)) return value.lexeme ?? pyStr(toPython(value));
  return undefined;
}

/**
 * What a value says about itself in one word: itself where it is a scalar, its first scalar
 * member where it is an object.
 *
 * `{"base": "../primitive-library/"}` answers the base and `{"name": "norm.rms", "version":
 * "1.0.0"}` answers the name — which is what S1 writes in the tail of a library and of an
 * instance. One rule and not two: the first scalar a value carries is what identifies it among
 * others of its kind.
 */
export function summaryOf(value: JsonValue): string | undefined {
  const direct = printed(value);
  if (direct !== undefined) return direct;
  if (!isJsonObject(value)) return undefined;
  for (const one of value.members) {
    const scalar = printed(one.value);
    if (scalar !== undefined) return scalar;
  }
  return undefined;
}

/** One entry of a summary: a container's member name and how many entries it holds. */
interface Held {
  readonly name: string;
  readonly count: number;
}

/**
 * What a row holds, as `<member> <count>` per named map under it.
 *
 * S1's own line — `values 4 · parameters 3 · constants 0 · states 0` — and the summary a
 * composition carries. The walk looks through groupings, so `bindings` answers its four maps
 * rather than itself, and stops at anything that holds a value of its own.
 */
function heldBy(shapes: SchemaShapes, shape: Shape, value: JsonValue): Held[] {
  const found: Held[] = [];
  const walk = (at: Shape, node: JsonValue): void => {
    for (const name of shapes.propertyOrder(at)) {
      const written = memberOf(node, name);
      if (written === undefined) continue;
      const member = shapes.member(at, name);
      if (isNamedMap(factsOf(member))) found.push({ name, count: entriesOf(written).length });
      else if (isGrouping(shapes, member)) walk(member, written);
    }
  };
  walk(shape, value);
  return found;
}

/** A summary line as the tree writes it. */
function summaryLine(held: readonly Held[]): string | undefined {
  if (held.length === 0) return undefined;
  return held.map((one) => `${one.name} ${String(one.count)}`).join(' · ');
}

/**
 * A container's entries as the tail of a collapsed group — §4.5's own sketch.
 *
 * > `▾ Quantities (9)   d 4096 · ffn 14336 · heads 32 · …`
 * > `▾ Instances (3)    embed · final_n · lm_head`
 *
 * A named entry is its name, with the figure the core resolves for it where it has one; an item
 * of a list has no name, so it is what the item says about itself.
 */
function entriesLine(
  path: Path,
  entries: readonly { step: string | number; value: JsonValue }[],
  figures: Figures,
): string | undefined {
  if (entries.length === 0) return undefined;
  const shown = entries.slice(0, TAIL_ENTRIES).map((one) => {
    if (typeof one.step !== 'string') return summaryOf(one.value) ?? String(one.step);
    const figure = figures.get(pointerOf(child(path, one.step)))?.text;
    return figure === undefined || figure === '' ? one.step : `${one.step} ${figure}`;
  });
  return entries.length > TAIL_ENTRIES ? `${shown.join(' · ')} · …` : shown.join(' · ');
}

/**
 * The label of a container: the schema's own `title` where it has one, its member name otherwise.
 *
 * The walker of feature 2.3 reads a row's label the same way ("the schema's `title`, failing that
 * the member name"), so a `title` the schemas gain — the plan's finding F3, one commit per schema
 * — becomes the outline's label without a line changing here. Until then `primitive_libraries` is
 * two words in the schema's own spelling, and an underscore is a word separator: that is the only
 * thing this adds to a name, and S1's shorter *Libraries* is an abbreviation nothing can infer.
 */
function labelOf(name: string, facts: SchemaFacts): string {
  if (facts.title !== null) return facts.title;
  const spaced = name.replace(/_/g, ' ');
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

/**
 * The marks an entry carries, read from what its declaration writes.
 *
 * - **`⚑`**: a member bound to the condition editor is a guard, and a site that writes one is
 *   guarded (§4.5). The binding is `presentation.json`'s, so the interface never names `when`.
 * - **`⇄`**: a list of *objects* with more than one item is several members sharing one thing —
 *   an identity's `members`, a public input's `to`. A list of one shares nothing, and a list of
 *   text (`families`) is not a membership.
 */
function marksOf(
  shapes: SchemaShapes,
  bindings: Presentation,
  shape: Shape,
  value: JsonValue,
): string[] {
  const marks: string[] = [];
  for (const name of shapes.propertyOrder(shape)) {
    const written = memberOf(value, name);
    if (written === undefined) continue;
    const member = shapes.member(shape, name);
    if (bindingOf(bindings, member)?.widget === CONDITION && !marks.includes(GUARDED)) {
      marks.push(GUARDED);
    }
    if (!isJsonArray(written) || written.length < 2) continue;
    if (factsUnder(shapes.item(member, 0)).holdsObject && !marks.includes(SHARED)) marks.push(SHARED);
  }
  return marks;
}

/** What the core resolved for a declaration: the figure it is shown with, and where it came from. */
type Figures = ReadonlyMap<string, { readonly text: string; readonly computed: boolean }>;

/** What the core resolves for each quantity, by the pointer of its own declaration. */
function figuresOf(
  tree: JsonObject,
  assignment?: PyRecord,
): Figures {
  const found = new Map<string, { text: string; computed: boolean }>();
  const written = memberOf(tree, QUANTITIES);
  if (written === undefined || !isJsonObject(written)) return found;
  // `view.py` resolves a document's quantities exactly this way — `resolve_quantities({
  // 'quantities': quantities})` — and prints what it answers with `str`. Both are asked of the
  // core, so a derivation is evaluated by the implementation the parity job holds to the tools
  // and the interface evaluates nothing (the component inventory's §7).
  const model = { [QUANTITIES]: toPython(written) } as PyRecord;
  for (const reading of quantityReadings(model, assignment)) {
    found.set(reading.pointer, {
      text: reading.value === undefined ? '' : pyStr(reading.value),
      computed: reading.computed,
    });
  }
  return found;
}

/** Whether a problem falls at or under a place. */
function problemUnder(problems: readonly string[], pointer: string): boolean {
  return problems.some((one) => one === pointer || one.startsWith(`${pointer}/`));
}

/** The key of the line that says what an open row holds; never a place of the document. */
function heldKey(pointer: string): string {
  return `${pointer}::held`;
}

/** The outline of a document, as {@link OutlineRow}s in the order they are drawn. */
export function outlineOf(request: OutlineRequest): OutlineRow[] {
  const { tree, shapes, bindings } = request;
  const role = request.role ?? MODEL;
  const problems = request.problems ?? [];
  const toggled = request.toggled ?? new Set<string>();
  const filter = (request.filter ?? '').trim().toLowerCase();
  const figures = figuresOf(tree, request.assignment);
  const rows: OutlineRow[] = [];
  const root = shapes.root(role);

  /** Open by default, unless the user said otherwise — and everything is open under a filter. */
  const isOpen = (pointer: string, byDefault: boolean): boolean =>
    filter.length > 0 || request.openAll === true
      ? true
      : toggled.has(pointer)
        ? !byDefault
        : byDefault;

  /**
   * What an entry holds that the outline goes into: the maps that *declare* things — a
   * composition's indices and its sites, flattened into the row, since those declarations are
   * what the outline is for — and the groupings, which are drawn as groups of their own.
   *
   * A composition's scoped bindings are such a grouping, so the eight edges and the nine
   * parameter identities `decoder` carries are rows like the document's own, and nothing else
   * under an instance is: `arguments` is a map the argument sheet owns (§4.12) and `families` is
   * a list of words, and neither declares anything the outline can select.
   */
  const inside = (
    shape: Shape,
    value: JsonValue,
  ): { name: string; value: JsonValue; declaring: boolean }[] => {
    const found: { name: string; value: JsonValue; declaring: boolean }[] = [];
    for (const name of shapes.propertyOrder(shape)) {
      const written = memberOf(value, name);
      if (written === undefined) continue;
      const member = shapes.member(shape, name);
      const declaring =
        bindingOf(bindings, member)?.declares !== undefined && isNamedMap(factsOf(member));
      if (declaring || isGrouping(shapes, member)) found.push({ name, value: written, declaring });
    }
    return found;
  };

  /** One entry of a container: a declaration, or an item of a list. */
  const entryRow = (
    path: Path,
    shape: Shape,
    value: JsonValue,
    depth: number,
    holder: Shape,
    scope?: OutlineRow['scope'],
  ): void => {
    const pointer = pointerOf(path);
    const step = path[path.length - 1];
    const named = typeof step === 'string';
    const label = named ? step : (summaryOf(value) ?? String(step));
    const declaring = inside(shape, value);
    const openByDefault = declaring.length > 0;
    const open = isOpen(pointer, openByDefault);
    const figure = figures.get(pointer);
    const held = heldBy(shapes, shape, value);
    // The first member of the node's face is what identifies the declaration — an instance's
    // primitive — and the first scalar in it is its name, which is what the tail writes and what
    // a template instance is recognised by.
    const face = bindingOf(bindings, shape)?.face?.[0];
    const faceValue = face === undefined ? undefined : memberOf(value, face);
    const identifies = faceValue === undefined ? undefined : summaryOf(faceValue);
    const tail = identifies ?? (open || !named ? undefined : summaryLine(held));
    const declaredAt = boundAnchor(bindings, holder);
    const declares = bindingOf(bindings, holder)?.declares;
    const inner = scope ?? (named && declares !== undefined ? { path, name: label } : undefined);
    rows.push({
      pointer,
      path,
      depth,
      kind: 'entry',
      label,
      named,
      ...(tail === undefined ? {} : { tail }),
      ...(figure === undefined || figure.text === '' ? {} : { figure: figure.text }),
      marks: [
        ...(identifies !== undefined && request.templates?.has(identifies) === true ? [TEMPLATE] : []),
        ...(figure?.computed === true ? [COMPUTED] : []),
        ...(isJsonObject(value) ? marksOf(shapes, bindings, shape, value) : []),
      ],
      problem: problemUnder(problems, pointer),
      container: declaring.length > 0,
      open,
      openByDefault,
      ...(declaring.length > 0
        ? { tint: 'group' as const }
        : figure === undefined
          ? {}
          : { tint: 'figure' as const }),
      ...(declares === undefined ? {} : { declares }),
      ...(declaredAt === undefined ? {} : { declaredAt }),
      ...(scope === undefined ? {} : { scope }),
    });
    if (!open) return;
    const line = summaryLine(held);
    if (line !== undefined) {
      rows.push({
        pointer: heldKey(pointer),
        path,
        depth: depth + 1,
        kind: 'note',
        label: line,
        named: false,
        marks: [],
        problem: false,
        container: false,
        open: false,
        openByDefault: false,
      });
    }
    for (const one of declaring) {
      const at = child(path, one.name);
      const member = shapes.member(shape, one.name);
      if (!one.declaring) {
        groupRow(at, member, one.value, depth + 1, inner);
        continue;
      }
      for (const entry of entriesOf(one.value)) {
        entryRow(
          child(at, entry.step),
          shapes.member(member, String(entry.step)),
          entry.value,
          depth + 1,
          member,
          inner,
        );
      }
    }
  };

  /** A group: a container the schema declares at a place of the document. */
  const groupRow = (
    path: Path,
    shape: Shape,
    value: JsonValue,
    depth: number,
    scope?: OutlineRow['scope'],
  ): void => {
    const pointer = pointerOf(path);
    const name = String(path[path.length - 1]);
    const facts = factsOf(shape);
    const held = entriesOf(value);
    const counted = isNamedMap(facts) || facts.holdsArray;
    // Open by default when it is a non-empty map of the document's own names — which is the
    // state S1 draws: the quantities, the instances and the compositions open, the libraries (a
    // list), the empty constants, the bindings and the interfaces closed.
    const openByDefault = isNamedMap(facts) && held.length > 0;
    const open = isOpen(pointer, openByDefault);
    const tail = open
      ? undefined
      : counted
        ? entriesLine(path, held, figures)
        : summaryLine(heldBy(shapes, shape, value));
    rows.push({
      pointer,
      path,
      depth,
      kind: 'group',
      label: labelOf(name, facts),
      named: false,
      ...(counted ? { count: held.length } : {}),
      ...(tail === undefined ? {} : { tail }),
      marks: [],
      problem: problemUnder(problems, pointer),
      container: true,
      open,
      openByDefault,
    });
    if (!open) return;
    if (counted) {
      for (const entry of held) {
        entryRow(
          child(path, entry.step),
          typeof entry.step === 'number'
            ? shapes.item(shape, entry.step)
            : shapes.member(shape, entry.step),
          entry.value,
          depth + 1,
          shape,
          scope,
        );
      }
      return;
    }
    for (const member of shapes.propertyOrder(shape)) {
      const written = memberOf(value, member);
      if (written === undefined) continue;
      const inner = shapes.member(shape, member);
      if (!isContainer(factsOf(inner))) continue;
      groupRow(child(path, member), inner, written, depth + 1, scope);
    }
  };

  // The root row: the document's name and the tag its schema fixes, read at the two members the
  // *schema* names (feature 2.6's `nameMember` and `tagOf`), with a template's version beside it.
  const name = nameMember(shapes, role);
  const version = versionMember(shapes, role);
  const tag = tagOf(shapes, tree, role);
  const held = version === null ? undefined : memberOf(tree, version);
  const tail = [tag, held === undefined ? undefined : printed(held)].filter((one) => one != null);
  const rootOpen = isOpen('', true);
  rows.push({
    pointer: '',
    path: [],
    depth: 0,
    kind: 'document',
    label: (name === null ? undefined : printed(memberOf(tree, name) ?? '')) ?? '',
    named: false,
    ...(tail.length === 0 ? {} : { tail: tail.join(' · ') }),
    marks: [],
    problem: problemUnder(problems, ''),
    container: true,
    open: rootOpen,
    openByDefault: true,
  });
  if (rootOpen) {
    for (const member of shapes.propertyOrder(root)) {
      const written = memberOf(tree, member);
      if (written === undefined) continue;
      const shape = shapes.member(root, member);
      if (!isContainer(factsOf(shape))) continue;
      groupRow(child([], member), shape, written, 1);
    }
  }
  return filter.length === 0 ? rows : filtered(rows, filter);
}

/**
 * The rows a filter keeps: those whose name matches, and every row above them.
 *
 * The tail is not matched: it summarises what is *inside* a row, and a filter that kept a row for
 * what it holds would keep the row without the thing that matched.
 */
function filtered(rows: readonly OutlineRow[], filter: string): OutlineRow[] {
  const keep = new Set<number>();
  const above: number[] = [];
  for (const [at, row] of rows.entries()) {
    while (above.length > 0 && (rows[above[above.length - 1] as number] as OutlineRow).depth >= row.depth) {
      above.pop();
    }
    if (row.kind !== 'note' && row.label.toLowerCase().includes(filter)) {
      keep.add(at);
      for (const one of above) keep.add(one);
    }
    above.push(at);
  }
  return rows.filter((_, at) => keep.has(at));
}

/**
 * The toggles that have to change so that every ancestor of a place is on screen.
 *
 * `toggled` is a deviation from the outline's own default (`isOpen` reads it that way), so making
 * a place visible is not "add every ancestor": an ancestor that opens by default must be *out* of
 * the set and one that does not must be *in* it. The rows are the whole outline — computed with
 * `openAll`, so that an ancestor under a closed one is there to be read.
 *
 * It answers the next set, and the set it was given where nothing has to change.
 */
export function revealing(
  rows: readonly OutlineRow[],
  toggled: readonly string[],
  pointer: string,
): readonly string[] {
  const wanted = new Set(toggled);
  let moved = false;
  for (const row of rows) {
    if (!row.container) continue;
    if (!pointer.startsWith(`${row.pointer}/`) && row.pointer !== '') continue;
    if (row.pointer !== '' && row.pointer === pointer) continue;
    const held = wanted.has(row.pointer);
    // Open is `toggled ? !byDefault : byDefault`, so the set has to hold exactly the ancestors
    // the default would have left shut.
    const shouldHold = !row.openByDefault;
    if (held === shouldHold) continue;
    if (shouldHold) wanted.add(row.pointer);
    else wanted.delete(row.pointer);
    moved = true;
  }
  return moved ? [...wanted] : toggled;
}
