/**
 * The sheets of plan §4.11 that are not an instance's, and the tables of §4.16 — as data.
 *
 * | Selection | Sections |
 * |---|---|
 * | **Composition** | Name · indices · families · scoped bindings summary (counts) |
 * | **Edge** | Rule name · `from` / `to` endpoints · guard · `for_each` · the D2 value when fresh |
 * | **Identity** | Name · members · dtype · location · `for_each` / guard · Derived: D3 or D4 rows |
 * | **Input** | Name · `to` · kind · stream · fragmented · Derived: D2 stream count, alignment … |
 * | **Output** | Name · `from` · generative · Derived: the value's domain and shape |
 * | **Quantity** | Name · type · source · domain · Used by · Resolved value |
 * | **Constant** | Name · identity · shape · dtype · multiplicity · Bound to |
 * | **Document** | `model` · `version` · `primitive_libraries` · schema tag · counts |
 *
 * **Every row of the left column above is the generated form of the place's own `$def`** — the
 * walker of feature 2.3 over `quantity_definition`, `value_binding`, `public_input` and the rest,
 * which already yields exactly the members those sections name, in the schema's own order, with
 * the choosers, the selects, the toggles, the lists and the expression editors bound to them. So
 * this module writes no section list: it takes the place, walks it, and attaches beside the form
 * the **facts** the place has and a schema cannot state:
 *
 * | Fact | Attached where | Answered by |
 * |---|---|---|
 * | the resolved value of a quantity, and whether the document says how it follows | the core has a reading whose pointer is this place | `quantityReadings` (feature 2.7) |
 * | the occurrences that name this declaration | the place is an entry of a map `presentation.json` says declares something | the reference index, through the same selectors a rename rewrites |
 * | the identity a binding rule declares, and the D3 or D4 rows of its instances | the core has an identity reading whose pointer is this place | `identityReadings` / `identityDerived` |
 * | the D2 value an edge carries, a public input delivers or a public output exposes | the folded graph has an edge or a terminal at this place | `valueRows` / `streamRows` |
 * | what a composition holds | the folded graph has a box at this place | `foldedGraph` |
 *
 * Nothing here decides what kind of thing a place is by its name: the core's readings are keyed
 * by the pointer of the place the author wrote, and a place with no reading gets no section. That
 * is the whole dispatch, and it is why a grammar that grew another declaration would show its
 * form here without a line changing.
 */
import {
  identityAt,
  identityDerived,
  inputValue,
  isJsonObject,
  noIdentityDerived,
  outputValue,
  pyStr,
  valueOf,
  type DerivedStreamRow,
  type DerivedValueRow,
  type FoldedEdge,
  type FoldedGraph,
  type FoldedHeld,
  type FoldedNode,
  type IdentityDerived,
  type IdentityReading,
  type JsonObject,
  type PyValue,
  type QuantityReading,
} from '@tensorspine/lang';
import {
  nodeAt,
  pointerOf,
  type Path,
  type ReferenceIndex,
  type SchemaShapes,
  type Shape,
} from '@tensorspine/store';

// The two leaf modules of §4.13, not its entry point: this module is read by the document store,
// and a sheet's *data* has no business dragging an editor's component into that import graph.
import { editsExpression, unionAnchorOf } from '../expressions/language.js';
import { printAt } from '../expressions/print.js';
import { formOf, MAP, type Form, type FormContext, type FormRow } from '../forms/index.js';
import { referenceSelectors } from '../presentation/index.js';
import type { OutlineRow } from '../explorer/outline.js';

/** What the sheet of a place is built from. */
export interface PlaceRequest {
  readonly tree: JsonObject;
  readonly role: string;
  readonly context: FormContext;
  /** The place selected, as a path of the document; the root when nothing is selected. */
  readonly path: Path;
  /** The outline's row for the place, which carries what its container declares (feature 2.7). */
  readonly row?: OutlineRow | null;
  /** Every row of the outline by pointer, so a table can read what each of its entries declares. */
  readonly rows?: ReadonlyMap<string, OutlineRow> | null;
  /** The document's reference index, for the occurrences that name the declaration. */
  readonly index?: ReferenceIndex | null;
  /** The core's readings of the document's quantities (feature 2.7). */
  readonly quantities?: readonly QuantityReading[];
  /** The core's readings of the identities the document's binding rules declare. */
  readonly identities?: readonly IdentityReading[];
  /** The derived document, or `null` where none has been computed for this revision. */
  readonly derived?: PyValue | null;
  /** The D2 values and streams of that derived document, read once by the caller. */
  readonly values?: readonly DerivedValueRow[];
  readonly streams?: readonly DerivedStreamRow[];
  /** The folded reading of the document (feature 2.9), for an edge's ends and a box's counts. */
  readonly folded?: FoldedGraph | null;
  /** The problems whose pointer falls at or under the place, as the form takes them. */
  readonly problems?: readonly { readonly path: string; readonly message: string }[];
  /**
   * The editor's own notices for this document — §4.17's `editor` source, feature 2.8's.
   *
   * §4.16 gives one of them to this feature's own table: "a quantity used nowhere is flagged by
   * the editor as a notice (not a rule: a hint)". The mark a row carries is **that** notice and
   * not a second rule of the table's: the panel and the table say the same thing about the same
   * declaration, and what counts as read is decided once (feature 2.8 states why it is a question
   * about quantities and not about every declaration — a public input nothing joins is not
   * unused, it introduces its own stream).
   */
  readonly notices?: readonly { readonly path?: string }[];
  /** How deep the generated form goes; the sheet's own default otherwise. */
  readonly limit?: number;
}

/** One occurrence of a declaration's name, as §4.16's "used by" shows it. */
export interface UsedBy {
  /** Where it is written, as a pointer — what clicking the row reveals. */
  readonly pointer: string;
  readonly path: Path;
  /** The member it is written under: `quantity`, `stream`, `site`. */
  readonly tag: string;
}

/** What the products say about the identity a place declares. */
export interface IdentityFacts {
  readonly reading: IdentityReading;
  readonly derived: IdentityDerived;
}

/** The sheet of a place that is not a site: §4.11's other rows. */
export interface PlaceSheet {
  readonly pointer: string;
  readonly path: Path;
  /** The name the document writes at this place, where it is a named entry. */
  readonly name: string;
  /** What the container declares, in `presentation.json`'s own word; `''` where it declares none. */
  readonly declares: string;
  /** The generated form of the place — §4.11's rows. */
  readonly form: Form;
  /** The table of a map of declarations (§4.16), or `null` where the place is not one. */
  readonly table: PlaceTable | null;
  /** The occurrences that name this declaration, in document order. */
  readonly usedBy: readonly UsedBy[];
  /** Whether the place is a declaration whose references are stated at all (§4.16's notice). */
  readonly referenced: boolean;
  /** What the core resolves the quantity declared here to; `null` where the place is not one. */
  readonly quantity: QuantityReading | null;
  /** The identity declared here, with the products' rows for its instances. */
  readonly identity: IdentityFacts | null;
  /** The D2 value this place carries — an edge's producing end, an input's, an output's. */
  readonly value: DerivedValueRow | null;
  /** The stream a public input joins or introduces, as D2 reports it. */
  readonly stream: DerivedStreamRow | null;
  /** What a composition holds, container by container (S3's summary line). */
  readonly held: readonly FoldedHeld[];
  /** The folded box at this place, where there is one: a composition, a terminal, a card. */
  readonly box: FoldedNode | null;
  /** The folded edge at this place, where there is one. */
  readonly edge: FoldedEdge | null;
}

/** One row of a table of declarations (§4.16, artboard S15). */
export interface TableRow {
  readonly name: string;
  readonly pointer: string;
  readonly path: Path;
  /** One cell per column, in the columns' order; `null` where the entry writes nothing there. */
  readonly cells: readonly (TableCell | null)[];
  /** The line drawn under the row — S15's `derivation …` and its notice. */
  readonly note: string | null;
  /** How many occurrences name it, where anything can; `null` where nothing refers to this kind. */
  readonly usedBy: number | null;
  /** What the core resolves it to, printed as the core prints it; `null` where it resolves none. */
  readonly resolved: string | null;
  /** Whether the editor has a notice about this declaration — §4.16's hint, and 2.8's own row. */
  readonly unused: boolean;
}

/** One cell: the value as a line, and the place it is edited at. */
export interface TableCell {
  readonly text: string;
  /** The row of the generated form the cell shows, so a sheet can edit it where it stands. */
  readonly row: FormRow;
}

/** A table of the declarations a map holds — §4.16's Quantities, Constants and Interfaces. */
export interface PlaceTable {
  /** The map's own place. */
  readonly path: Path;
  /** What one entry is called, in `presentation.json`'s word. */
  readonly declares: string;
  /**
   * The columns: the name, then the entry definition's own members in the schema's order, then
   * the columns the core answers for this kind of declaration.
   */
  readonly columns: readonly string[];
  readonly rows: readonly TableRow[];
}

/**
 * How deep a place's form is walked.
 *
 * Deep enough for a declaration and everything written inside it — an identity's members are
 * endpoints whose selectors are choosers of their own, which is four levels — and one less at the
 * document's root, whose maps are drawn as counts (§4.11) and whose one list is the bases it
 * resolves from, whose own rows are three levels down.
 */
const DEPTH = 4;
const ROOT_DEPTH = 3;

/** The sheet of one place. */
export function placeSheet(request: PlaceRequest): PlaceSheet {
  const { context, tree, path } = request;
  const pointer = pointerOf(path);
  const shape = shapeAt(context.shapes, path, request.role);
  const value = nodeAt(tree, path);
  const form = formOf(context, {
    shape,
    ...(value === undefined ? {} : { value }),
    label: nameOf(path),
    limit: request.limit ?? (path.length === 0 ? ROOT_DEPTH : DEPTH),
    ...(request.problems === undefined ? {} : { problems: relative(request.problems, pointer) }),
  });
  const quantity = request.quantities?.find((one) => one.pointer === pointer) ?? null;
  const reading = identityAt(request.identities ?? [], pointer);
  const box = request.folded?.byPointer.get(pointer) ?? null;
  const edge = request.folded?.edges.find((one) => one.pointer === pointer) ?? null;
  const references = occurrences(request);
  return {
    pointer,
    path,
    name: nameOf(path),
    declares: request.row?.declares ?? '',
    form,
    table: mapTable(request),
    usedBy: references.found,
    referenced: references.stated,
    quantity,
    identity:
      reading === undefined
        ? null
        : {
            reading,
            derived:
              request.derived === null || request.derived === undefined
                ? noIdentityDerived()
                : identityDerived(request.derived, reading.identity, reading.state),
          },
    value: valueFor(request, box, edge),
    stream: streamFor(request, box),
    held: box?.held ?? [],
    box,
    edge,
  };
}

/**
 * The occurrences that name the declaration at a place, and whether any rule states them.
 *
 * The selectors are the ones a **rename** rewrites — `presentation.json`'s `refers` at the
 * anchor the outline recorded, scoped by the declaration the row sits in — so the count a sheet
 * shows is the count a rename would change, and the two cannot drift. A map whose names nothing
 * refers to (a binding rule's, whose name is a label) has no rule at all, which is an answer and
 * not an absence: `referenced` is what says which of the two a reader is looking at.
 */
function occurrences(request: PlaceRequest): { found: UsedBy[]; stated: boolean } {
  const row = request.row ?? null;
  const index = request.index ?? null;
  if (row === null || index === null || !row.named || row.declaredAt === undefined) {
    return { found: [], stated: false };
  }
  const binding = request.context.bindings.at(row.declaredAt);
  // A map whose names nothing refers to states an **empty** rule list — `interfaces/outputs`'
  // own `"refers": []`, which feature 2.2 calls "an answer and not a gap" — so a place with no
  // rule and a place with no reference are told apart here rather than read as the same thing.
  if (binding?.refers === undefined || binding.refers.length === 0) {
    return { found: [], stated: false };
  }
  const selectors = referenceSelectors(binding, row.scope);
  const found = index
    .select(selectors, row.label)
    // The declaration's own place is not a use of it. A map whose names are referred to under the
    // *map's* own member — a composition's `indices`, named again in every selector's assignment —
    // would otherwise count its own declaration; feature 2.8's notice reads it the same way.
    .filter((one) => pointerOf(one.path) !== row.pointer)
    .map((one) => ({ pointer: pointerOf(one.path), path: one.path, tag: one.tag }));
  return { found, stated: true };
}

/** The D2 value a place carries: an edge's producing end, an input's own, an output's. */
function valueFor(
  request: PlaceRequest,
  box: FoldedNode | null,
  edge: FoldedEdge | null,
): DerivedValueRow | null {
  const values = request.values ?? [];
  if (values.length === 0) return null;
  if (edge !== null) {
    const carried = edge.from?.value ?? null;
    return carried === null ? null : valueOf(values, carried);
  }
  if (box === null) return null;
  // A terminal is a public input or a public output; which of the two it is, is what D2 says
  // about the name — the input that *delivers* a value, or the output that *exposes* one.
  return inputValue(values, box.name) ?? outputValue(values, box.name);
}

/** The stream a public input introduces or joins, as D2 reports it. */
function streamFor(request: PlaceRequest, box: FoldedNode | null): DerivedStreamRow | null {
  const streams = request.streams ?? [];
  if (box === null || streams.length === 0) return null;
  const value = inputValue(request.values ?? [], box.name);
  // The stream the value's own domain names — the product's own member, not a reading of the
  // line the domain is shown as.
  if (value === null || value.stream === null) return null;
  return streams.find((one) => one.name === value.stream) ?? null;
}

/**
 * The table of a map of declarations — §4.16's Quantities, Constants and Interfaces.
 *
 * > The Quantities panel … is a table: name · type · source · value / expression · domain · used
 * > by · resolved. Inline editing … Constants: a table with identity, shape, dtype, multiplicity,
 * > bound slots.
 *
 * The columns are **the entry definition's own members, in the schema's order** — the component
 * inventory's rule for a product table read one schema along ("a table whose columns are the
 * schema's properties in order") — with the name first and the facts the core answers last. So a
 * table exists for every map of declarations the grammar has, and the three §4.16 names are the
 * three the explorer can select.
 */
export function mapTable(request: PlaceRequest): PlaceTable | null {
  const { context, tree, path } = request;
  const shape = shapeAt(context.shapes, path, request.role);
  const value = nodeAt(tree, path);
  if (value === undefined || !isJsonObject(value)) return null;
  const outer = formOf(context, { shape, value, limit: 1 });
  if (outer.rows[0]?.widget !== MAP) return null;
  const entryShape = entryOf(context.shapes, shape, value);
  if (entryShape === null) return null;
  const members = context.shapes.propertyOrder(entryShape);
  const rows: TableRow[] = [];
  let resolves = false;
  let referenced = false;
  for (const member of value.members) {
    const at = [...path, member.name] as Path;
    const form = formOf(context, {
      shape: entryShape,
      value: member.value,
      label: member.name,
      limit: 2,
    });
    const pointer = pointerOf(at);
    const quantity = request.quantities?.find((one) => one.pointer === pointer) ?? null;
    const used = occurrences({ ...request, path: at, row: rowAt(request, pointer) });
    if (quantity !== null) resolves = true;
    if (used.stated) referenced = true;
    rows.push({
      name: member.name,
      pointer,
      path: at,
      cells: members.map((name) => cellOf(form, name)),
      note: noteOf(request, form, at),
      usedBy: used.stated ? used.found.length : null,
      resolved: quantity === null ? null : resolvedText(quantity),
      unused: (request.notices ?? []).some((one) => one.path === pointer),
    });
  }
  return {
    path,
    declares: declaresOf(request),
    // The facts the core answers come after the schema's own members, and only where there is
    // one to answer: a map whose entries nothing resolves has no `resolved` column at all.
    columns: [NAME, ...members, ...(referenced ? [USED_BY] : []), ...(resolves ? [RESOLVED] : [])],
    rows,
  };
}

/** What a quantity resolves to, as the core prints every value (`pyStr`); `—` where nothing does. */
function resolvedText(reading: QuantityReading): string {
  return reading.value === undefined ? '' : pyStr(reading.value);
}

/** The outline's row for a pointer, where the caller supplied the outline's answer for one. */
function rowAt(request: PlaceRequest, pointer: string): OutlineRow | null {
  const row = request.row ?? null;
  if (row !== null && row.pointer === pointer) return row;
  return request.rows?.get(pointer) ?? null;
}

/**
 * The line drawn under a row of the table — S15's `derivation  div(d, heads)`.
 *
 * Every expression the entry writes, printed in §4.13's text form and labelled by the member it
 * is written under. A literal quantity that declares a `derivation` is what S15 draws it for, and
 * an external's `default` and a derived source's `expression` get the same line from the same
 * rule: the *places* are found by the grammar (feature 2.11's own reading), never by a name.
 */
function noteOf(request: PlaceRequest, form: Form, at: Path): string | null {
  const { context } = request;
  const written = form.rows.filter((row) => editsExpression(row.widget) && row.present);
  if (written.length === 0) return null;
  return written
    .map((row) => {
      const place = [...at, ...row.steps] as Path;
      const value = nodeAt(request.tree, place);
      if (value === undefined) return '';
      const anchor = unionAnchorOf(context, shapeAt(context.shapes, place, request.role));
      return `${row.label} ${printAt(context, anchor, value)}`;
    })
    .filter((one) => one !== '')
    .join(' · ');
}

/** The cell of one column: the row of the entry's form at that member, with its value as a line. */
function cellOf(form: Form, member: string): TableCell | null {
  const row = form.rows.find((one) => one.depth === 1 && lastStep(one) === member);
  if (row === undefined || !row.present) return null;
  return { text: lineOf(form, row), row };
}

/** The last step of a row's place, which is the member it is. */
function lastStep(row: FormRow): string {
  const step = row.steps[row.steps.length - 1];
  return typeof step === 'string' ? step : String(step ?? '');
}

/**
 * One value as a table cell reads it: the alternative it is in, and the scalars it holds.
 *
 * A cell is one line of a table and a value of the grammar is a tree, so what is shown is the
 * **chooser's own answer** — which alternative the value is, as the walker labelled it — followed
 * by the scalars written under it, in the schema's order. `{"kind": "literal", "value": 4096}`
 * reads `literal 4096`; `{"kind": "cardinality"}` reads `cardinality`; a list of names reads as
 * its names. Nothing is interpreted: every part comes from a row the walker already made.
 */
function lineOf(form: Form, row: FormRow): string {
  const parts: string[] = [];
  if (row.mode !== undefined) parts.push(row.mode);
  // An alternative told apart by a `const` on its one member is *inline*: the walker puts that
  // constant on the row as its written value, and the mode has already said it.
  if (row.written !== undefined && scalar(row.written) !== row.mode) parts.push(scalar(row.written));
  const under = form.rows.filter(
    (one) => one.depth === row.depth + 1 && one.path.startsWith(`${row.path}/`) && one.present,
  );
  for (const child of under) {
    // The member that *names* the alternative is what the mode already said — a `const` the
    // walker put on the row, or the same constant read back as the value written there — and
    // anything else is what the alternative holds. A nested shape is left to the sheet's own rows.
    if (child.constant !== undefined) continue;
    if (child.written === undefined) continue;
    if (row.mode !== undefined && scalar(child.written) === row.mode) continue;
    parts.push(scalar(child.written));
  }
  return parts.join(' ');
}

/** A scalar as a cell writes it: the lexeme a number was written with (D12), the text otherwise. */
function scalar(value: unknown): string {
  if (typeof value === 'string') return value;
  return String(value);
}

/**
 * The definition one entry of a map is.
 *
 * Any name steps into a map, so an **empty** map has its entry definition like every other: S15
 * draws the Constants table empty, with its columns, and the corpus declares no constant at all
 * (finding F8) — a table that vanished with its last row would be a table nobody could add to.
 */
function entryOf(shapes: SchemaShapes, shape: Shape, value: JsonObject): Shape | null {
  const first = value.members[0];
  return shapes.step(shape, first?.name ?? ENTRY);
}

/** The name a step into an empty map is made with; the definition it reaches is the entry's. */
const ENTRY = 'entry';

/** What the entries of the map at a place are called, in `presentation.json`'s word. */
function declaresOf(request: PlaceRequest): string {
  const shape = shapeAt(request.context.shapes, request.path, request.role);
  for (const place of shape.all) {
    const binding = request.context.bindings.at(place.anchor);
    if (binding?.declares !== undefined) return binding.declares;
  }
  return '';
}

/** The problems of a document, as the form takes them: pointers relative to the place. */
function relative(
  problems: readonly { readonly path: string; readonly message: string }[],
  pointer: string,
): { path: string; message: string }[] {
  const here: { path: string; message: string }[] = [];
  for (const problem of problems) {
    if (problem.path === pointer) here.push({ path: '', message: problem.message });
    else if (problem.path.startsWith(`${pointer}/`)) {
      here.push({ path: problem.path.slice(pointer.length), message: problem.message });
    }
  }
  return here;
}

/** The name a place is written under, or the empty text at the root. */
function nameOf(path: Path): string {
  const step = path[path.length - 1];
  return typeof step === 'string' ? step : step === undefined ? '' : String(step);
}

/** The shape of a place, stepped down from the document's root. */
export function shapeAt(shapes: SchemaShapes, path: Path, role: string): Shape {
  let shape = shapes.root(role);
  for (const step of path) shape = shapes.step(shape, step);
  return shape;
}

/** The first column of every table: the declaration's own name. */
const NAME = 'name';

/** The two columns the core answers rather than the schema — §4.16's last two. */
const USED_BY = 'used by';
const RESOLVED = 'resolved';
