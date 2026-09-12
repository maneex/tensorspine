/**
 * What a row's quantity and index selects are filled from — plan §4.12's two referring modes.
 *
 * > **quantity** | `{"quantity": name}` | a select of the document's quantities whose type kind
 * > matches the argument's … **index** | `{"index": name}` | a select of the indices in scope
 * > (inside a composition only)
 *
 * Three questions, and none of them is answered by writing a word of the language down:
 *
 * - **which declarations are quantities, and which are indices** — `presentation.json` says so.
 *   A binding on the place a map is written at carries `declares`, and the Model explorer's own
 *   outline puts that word on every entry of it (`OutlineRow.declares`, feature 2.7). The mode's
 *   `referent` is the same word, from the same file (`references` on `scalar_expression`), so the
 *   two are compared and neither is typed.
 * - **which of them are in scope** — the declaration's own. A quantity is declared once and read
 *   everywhere ("quantities form one flat namespace", O0.4) and the outline gives it no scope; an
 *   index is declared by a composition and the outline records that composition as its scope. So a
 *   name is offered where it has no scope at all, or where the place being edited is under it.
 * - **which of them match the argument's type** — the alternative the declaration's own type is.
 *   `describe` answers the argument's `kind` (the declaration's word, which the generated argument
 *   schema cannot carry: a cardinality and a whole-number physical are both `integer` there, and a
 *   set domain is an `enum`), and the *quantity*'s kind is which alternative of its own union Ajv
 *   says its value is — the walker's `mode`, which is the grammar's verdict and not a resemblance.
 *   A declaration matches when it has a member offering that alternative and standing in it.
 */
import { isJsonObject, type JsonValue } from '@tensorspine/lang';
import { isUnder, nodeAt, type Path, type SchemaShapes, type Shape } from '@tensorspine/store';

import { formOf, type FormContext } from '../forms/index.js';
import type { OutlineRow } from '../explorer/outline.js';

/** What {@link namesFor} reads. */
export interface NameRequest {
  /** Every row of the document's outline (2.7's `openAll`). */
  readonly outline: readonly OutlineRow[];
  /** The word the mode refers to, as `presentation.json` writes it on both sides. */
  readonly referent: string;
  /** The place being edited, for the scope test. */
  readonly at: Path;
  /** The kind the declaration's type must be, when the row has one to match. */
  readonly kind?: string | undefined;
  readonly tree: JsonValue;
  readonly role: string;
  readonly context: FormContext;
  readonly shapes: SchemaShapes;
}

/** One name a select offers, with where it is declared so a row can navigate to it. */
export interface NameChoice {
  readonly name: string;
  readonly pointer: string;
  /** The scope it was declared in, when it has one — a composition's name. */
  readonly scope?: string;
}

/** The names a referring mode may be set to, in the document's own order. */
export function namesFor(request: NameRequest): NameChoice[] {
  const found: NameChoice[] = [];
  for (const row of request.outline) {
    if (row.declares !== request.referent || !row.named) continue;
    if (row.scope !== undefined && !isUnder(request.at, row.scope.path)) continue;
    if (request.kind !== undefined && !isOfKind(row, request)) continue;
    found.push({
      name: row.label,
      pointer: row.pointer,
      ...(row.scope === undefined ? {} : { scope: row.scope.name }),
    });
  }
  return found;
}

/**
 * Whether a declaration's own type is the kind an argument declares.
 *
 * The declaration is walked one level — its members, each with the alternatives its place offers
 * and the one its value is — and it matches when some member offers the kind as an alternative and
 * stands in it. Only a *type* member can offer `cardinality`, so nothing has to say which member is
 * the type: the union's own labels do.
 */
function isOfKind(row: OutlineRow, request: NameRequest): boolean {
  const value = nodeAt(request.tree, row.path);
  if (value === undefined || !isJsonObject(value)) return false;
  const form = formOf(request.context, {
    shape: shapeAt(request.shapes, row.path, request.role),
    value,
    limit: 1,
  });
  return form.rows.some(
    (one) =>
      one.depth === 1 &&
      one.modes?.some((mode) => mode.tag === request.kind) === true &&
      one.mode === request.kind,
  );
}

/** The shape of a place of the document, stepped down from the root. */
function shapeAt(shapes: SchemaShapes, path: Path, role: string): Shape {
  let shape = shapes.root(role);
  for (const step of path) shape = shapes.step(shape, step);
  return shape;
}
