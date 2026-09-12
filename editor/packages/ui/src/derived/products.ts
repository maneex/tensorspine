/**
 * The derived document, read against the derived schema — plan §4.18's six tabs.
 *
 * > A tab per product (D1–D6, titled with the products' names from the specification: *Derived
 * > Computation Graph* … *Derived Decomposition Options*), each rendered generically from the
 * > derived schema — an array of objects becomes a table whose columns are the schema's
 * > properties in order, a map a keyed table, a `qualified_value` a value with its status chip.
 *
 * So this module **finds** the products rather than listing them: it walks the derived document's
 * root against the schema, exactly as the status bar's four figures are found (feature 2.6), and
 * a member whose place carries a `product` binding is a tab. Nothing below names `d3`, `tensors`
 * or `bytes`; the schema says what a place holds, `presentation.json` says how a figure is shown
 * and what a name stands for, and the core says what a byte count reads as. The component
 * inventory's §7 in one line: *no component adds, converts or rounds a figure*.
 *
 * **Three shapes of section, from what the schema asserts.** A member that lists objects is a
 * table whose columns are the item's own properties, in the schema's order; a map is the same
 * table with the map's names as its first column; an object of scalars is the strip of totals the
 * artboards draw above a table. An object that holds *both* — D2's `peak_live` carries two
 * figures, a list of values and a map of byte counts — answers a totals strip for its scalars and
 * a section of its own for each container, which is one rule and not a special case.
 *
 * **The rows are found before they are rendered.** A corpus document's D3 has up to 1 771 rows and
 * §4.18's selection filter throws most of them away; rendering fourteen cells of each before
 * deciding would be fourteen times the work for the same answer. So a section carries its
 * *entries* — the value, its place, and the identifiers it writes — and {@link tableRows} renders
 * the cells of the ones a subject keeps, as many as a panel can show.
 */
import { factsOfShape, type FormContext } from '../forms/index.js';
import type { Shape } from '@tensorspine/store';
import type { PyRecord, PyValue } from '@tensorspine/lang';

import { cellOf, figureIn, isList, isRecord, membersOf, type DerivedCell } from './cells.js';
import { namedIn, type Named } from './naming.js';

/** A strip of figures above a table — the artboards' own `.totals`. */
export const TOTALS = 'totals';
/** An array of objects, or a map: the columns are the schema's properties in order. */
export const TABLE = 'table';
/** An array of scalars — D1's `topological_order`, D4's `carried`. */
export const LIST = 'list';

/** One product of the derived document: a tab of §4.18. */
export interface DerivedProduct {
  /** The member the derived document writes it under — `d3`. */
  readonly member: string;
  /** What a tab is called in one glyph pair: the member's own name, in capitals. */
  readonly short: string;
  /** The name the specification gives it, from the binding (`presentation.json`). */
  readonly label: string;
  /** The schema's own `description` — §1's "help text is the schema's". */
  readonly description: string | null;
  /** The product itself, and its place: {@link sectionsOf} builds its sections from these. */
  readonly value: PyValue;
  readonly shape: Shape;
}

/** One field of a totals strip: a scalar, or a qualified value with its status. */
export interface DerivedField {
  readonly name: string;
  readonly cell: DerivedCell;
}

/** One column of a table. */
export interface DerivedColumn {
  /** The property's own name, which is the header. */
  readonly name: string;
  /** Whether it is the map's key rather than a property of the value. */
  readonly key?: true;
  /** What a key column's names stand for, from the map's `propertyNames`. */
  readonly names?: string;
  readonly shape: Shape;
}

/** One entry of a table, before its cells are rendered. */
export interface DerivedEntry {
  /** The map's own name for it, or `null` in an array. */
  readonly key: string | null;
  readonly value: PyValue;
  readonly shape: Shape;
  /** Every identifier it writes, with what each one names — §4.18's filter reads these. */
  readonly named: readonly Named[];
}

/** One rendered row. */
export interface DerivedRow {
  readonly key: string | null;
  readonly cells: readonly DerivedCell[];
}

/** A section of a product's tab. */
export type DerivedSection =
  | {
      readonly kind: typeof TOTALS;
      readonly name: string;
      readonly fields: readonly DerivedField[];
    }
  | {
      readonly kind: typeof TABLE;
      readonly name: string;
      readonly columns: readonly DerivedColumn[];
      readonly entries: readonly DerivedEntry[];
    }
  | {
      readonly kind: typeof LIST;
      readonly name: string;
      readonly entries: readonly DerivedEntry[];
    };

/** What the panel's header names beside the freshness — §4.18's "the assignment used". */
export interface HeaderField {
  /** The word the binding gives it. */
  readonly label: string;
  readonly cell: DerivedCell;
}

/** A derived document read as the panel shows it. */
export interface DerivedReading {
  readonly products: readonly DerivedProduct[];
  readonly header: readonly HeaderField[];
}

/** The role the registry indexes the derived schema under — its file's own name. */
export const DERIVED_ROLE = 'derived';

/** Nothing at all: what a document nothing has been derived for reads as. */
export function noDerivedReading(): DerivedReading {
  return { products: [], header: [] };
}

/**
 * The products of a derived document, and the envelope its header names.
 *
 * The walk is over the document's own root: every member is looked up in the schema, and the
 * binding at that place decides whether it is a product, a header field, or neither — which is
 * how `schema`, `model` and `primitive_libraries` stay out of the panel without being named.
 */
export function derivedReading(derived: PyValue, context: FormContext): DerivedReading {
  const root = context.shapes.root(DERIVED_ROLE);
  if (!isRecord(derived)) return noDerivedReading();
  const products: DerivedProduct[] = [];
  const header: HeaderField[] = [];
  for (const [member, value] of membersOf(derived)) {
    const shape = context.shapes.member(root, member);
    const binding = context.bindings.firstOf(anchorsOf(shape));
    if (binding?.product !== undefined) {
      products.push(productOf(member, binding.product, value, shape));
      continue;
    }
    if (binding?.header !== undefined) {
      header.push({ label: binding.header, cell: cellOf(value, shape, context) });
    }
  }
  return { products, header };
}

/** The anchors of a place, most specific first — what a binding is looked up by. */
function anchorsOf(shape: Shape): readonly string[] {
  return shape.all.map((place) => place.anchor);
}

/** One product: its name, the schema's own description of it, and where its sections come from. */
function productOf(member: string, label: string, value: PyValue, shape: Shape): DerivedProduct {
  return {
    member,
    short: member.toUpperCase(),
    label,
    description: factsOfShape(shape).description,
    value,
    shape,
  };
}

/**
 * The sections of one product — built for the tab that is showing and for no other.
 *
 * `deepseek-v4-pro`'s D3 lists 1 771 tensors and its D6 8 082 partition options; building all six
 * products' entries to draw one of them would be five products' work nobody looks at, on every
 * derivation. So the reading answers the products and this answers one product's sections.
 */
export function sectionsOf(product: DerivedProduct, context: FormContext): DerivedSection[] {
  return sectionsAt(product.value, product.shape, context, '');
}

/**
 * The sections one place answers.
 *
 * An object's scalars are one strip of totals and each of its containers is a section of its own,
 * named by the path that reaches it — which is what makes D1's `interfaces` two keyed tables and
 * D2's `peak_live` a strip beside a list and a map, with no rule written for either of them.
 */
function sectionsAt(
  value: PyValue,
  shape: Shape,
  context: FormContext,
  prefix: string,
): DerivedSection[] {
  const facts = factsOfShape(shape);
  if (isList(value)) return [listSection(prefix, value, shape, context)];
  if (!isRecord(value)) return [];
  if (facts.keyed) return [keyedSection(prefix, value, shape, context)];
  const fields: DerivedField[] = [];
  const under: DerivedSection[] = [];
  for (const [member, one] of membersOf(value)) {
    const at = context.shapes.member(shape, member);
    const name = prefix === '' ? member : `${prefix} · ${member}`;
    if (isScalar(one) || figureIn(one, at, context) !== null) {
      fields.push({ name: member, cell: cellOf(one, at, context) });
      continue;
    }
    under.push(...sectionsAt(one, at, context, name));
  }
  const totals: DerivedSection[] =
    fields.length === 0 ? [] : [{ kind: TOTALS, name: prefix, fields }];
  return [...totals, ...under];
}

/**
 * A map.
 *
 * A map of *objects* is the keyed table §4.18 asks for: the map's own names as the first column,
 * then the value's properties in the schema's order. A map of *scalars* is a strip of totals
 * instead — `{"append": 32}`, `{"tokens": 521216}` — because a two-column table whose second
 * column the schema gives no name to is a table with a blank header, and the artboards write
 * those figures as a strip (S11's `evolution append 32`).
 */
function keyedSection(
  name: string,
  value: PyRecord,
  shape: Shape,
  context: FormContext,
): DerivedSection {
  const members = membersOf(value);
  if (!members.some(([, one]) => isRecord(one) || isList(one))) {
    return {
      kind: TOTALS,
      name,
      fields: members.map(([key, one]) => ({
        name: key,
        cell: cellOf(one, context.shapes.member(shape, key), context),
      })),
    };
  }
  const naming = keyNaming(shape, context);
  // Two things about the list below. The type is annotated because an object literal widens the
  // core's `unique symbol` sentinel to `symbol`, which is not a `PyValue` (feature 1.2's decision,
  // met by the type checker); and the map's own **name** is part of what the row names, without
  // which holding D1's `nodes` to a node would keep nothing at all — the identifier there is the
  // key and not a member of the value (found by this feature's own suite).
  const entries: DerivedEntry[] = members.map(([key, one]) => {
    const at = context.shapes.member(shape, key);
    const named = namedIn(one, at, context);
    return {
      key,
      value: one,
      shape: at,
      named: naming === undefined ? named : [{ kind: naming, name: key }, ...named],
    };
  });
  const first = entries[0];
  const columns: DerivedColumn[] = [
    { name, key: true, shape, ...(naming === undefined ? {} : { names: naming }) },
  ];
  if (first !== undefined) {
    // The value's own place — every value of a map shares it — so the columns are the schema's
    // properties in order, and a member one entry omits still has a column.
    for (const member of columnNames(
      entries.map((entry) => entry.value),
      first.shape,
    )) {
      columns.push({ name: member, shape: context.shapes.member(first.shape, member) });
    }
  }
  return { kind: TABLE, name, columns, entries };
}

/**
 * What a map's own names stand for, from the `propertyNames` the schema writes on it.
 *
 * D1's `nodes` and `instances` are keyed by a `node_identifier`, which `presentation.json` binds
 * as naming a node — so the key column is a link and the selection filter reads it, without this
 * module knowing that either map exists.
 */
function keyNaming(shape: Shape, context: FormContext): string | undefined {
  for (const place of shape.all) {
    const node = place.node as Record<string, unknown>;
    const names: unknown = node['propertyNames'];
    const reference =
      typeof names === 'object' && names !== null && !Array.isArray(names)
        ? (names as Record<string, unknown>)['$ref']
        : undefined;
    if (typeof reference !== 'string') continue;
    const anchor = reference.startsWith('#') ? `${place.schema}${reference}` : reference;
    const binding = context.bindings.firstOf(
      context.shapes.at(anchor).all.map((one) => one.anchor),
    );
    if (binding?.names !== undefined) return binding.names;
  }
  return undefined;
}

/** An array: a table of the item's own properties, or a plain list where the items are scalars. */
function listSection(
  name: string,
  value: readonly PyValue[],
  shape: Shape,
  context: FormContext,
): DerivedSection {
  const item = context.shapes.item(shape, 0);
  const entries: DerivedEntry[] = value.map((one, index) => {
    const at = context.shapes.item(shape, index);
    return { key: null, value: one, shape: at, named: namedIn(one, at, context) };
  });
  // The columns are the *schema's* properties, so an array the document left empty still draws
  // its headers and an array whose one entry omits an optional member still has a column for it.
  // A list of scalars has no properties to read and stays a list.
  const declared = factsOfShape(item).members;
  const names = declared.length > 0 || value.some((one) => isRecord(one)) ? columnNames(value, item) : [];
  if (names.length === 0) return { kind: LIST, name, entries };
  const columns = names.map((member) => ({
    name: member,
    shape: context.shapes.member(item, member),
  }));
  return { kind: TABLE, name, columns, entries };
}

/**
 * The columns of a table: the schema's properties, in the order it writes them.
 *
 * A member the schema does not declare — which nothing in a derived document writes, the products
 * being closed objects — is appended in the order the document writes it, so a table shows what it
 * was given rather than silently dropping a column.
 */
function columnNames(values: readonly PyValue[], item?: Shape): readonly string[] {
  const declared = item === undefined ? [] : factsOfShape(item).members;
  const found = [...declared];
  for (const one of values) {
    if (!isRecord(one)) continue;
    for (const member of Object.keys(one)) if (!found.includes(member)) found.push(member);
  }
  return found;
}

/** Whether a value is a scalar a totals strip can show on one line. */
function isScalar(value: PyValue): boolean {
  return !isList(value) && !isRecord(value);
}

/** How many rows a table draws before it says how many more there are. */
export const ROW_LIMIT = 200;

/** What a table answers a reader: the rows it drew, how many it kept, how many there are. */
export interface DerivedRows {
  readonly rows: readonly DerivedRow[];
  readonly kept: number;
  readonly total: number;
}

/**
 * The rows of a section, filtered to a subject and capped.
 *
 * The cap is the artboards' own reading — S11 writes "27 more — one per layer", "32 more", "318
 * more" — and it is what keeps a table of `deepseek-v4-pro`'s 1 771 tensors inside §5.6's budget
 * for a keystroke. The cells of a row nobody sees are never rendered.
 */
export function tableRows(
  section: DerivedSection,
  context: FormContext,
  keeps: (entry: DerivedEntry) => boolean,
  limit: number = ROW_LIMIT,
): DerivedRows {
  if (section.kind === TOTALS) return { rows: [], kept: 0, total: 0 };
  const kept = section.entries.filter((entry) => keeps(entry));
  const columns = section.kind === TABLE ? section.columns : [];
  const rows = kept.slice(0, limit).map((entry) => ({
    key: entry.key,
    cells:
      section.kind === LIST
        ? [cellOf(entry.value, entry.shape, context)]
        : columns.map((column) =>
            column.key === true
              ? keyCell(entry.key ?? '', column.names)
              : cellOf(memberOf(entry.value, column.name), column.shape, context),
          ),
  }));
  return { rows, kept: kept.length, total: section.entries.length };
}

/** A map's own name, as the first cell of its row. */
function keyCell(key: string, names: string | undefined): DerivedCell {
  return names === undefined ? { text: key } : { text: key, name: key, names };
}

/** One member of a record, or `null` where the record does not write it. */
function memberOf(value: PyValue, name: string): PyValue {
  if (!isRecord(value)) return null;
  return Object.prototype.hasOwnProperty.call(value, name) ? (value[name] ?? null) : null;
}
