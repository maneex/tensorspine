/**
 * The figures every product is written from: widths, counts, shapes as stored, selected dtypes.
 *
 * `tools/derive.py` opens with six of them above its first product, because D2, D3, D4 and D5 all
 * write the same four kinds of number and must write them identically — an element count is the
 * product of a shape's extents wherever it is taken, and a byte size is that count times the
 * dtype's width wherever it is taken. They are here for the same reason, and nothing below
 * re-derives one of them.
 *
 * **`BYTES` is a semantic table** in the sense of the plan's §1: the core "names a vocabulary item
 * only to attach semantics to it", and the width of a dtype is exactly such a semantics — the
 * schema declares the sixteen names and says nothing about what they cost. The audit of §1 (d)
 * (`editor/tests/audit/semantic-tables.test.ts`) proves the table's key set equal to the schema's
 * `dtype` enumeration, in both directions: a dtype the language gains without a width, or a width
 * for a name the language does not have, fails the build.
 *
 * **The arithmetic is Python's** (feature 1.2): a count is a `bigint` while every extent is one,
 * and a `fp4`, `u4` or `i4` tensor's byte size is a `float` because the width is `0.5` — which is
 * what the derived documents record and what a byte comparison of them decides. `pyMultiply` is
 * the promotion, written down once there rather than by each product.
 */
import { pyMultiply } from '../expr/arithmetic.js';
import { PyKeyError, PyOverflowError, PyValueError } from '../expr/errors.js';
import type { Quantities } from '../expr/model.js';
import { primitiveValue } from '../expr/primitive.js';
import type { PyRecord, PyValue } from '../expr/value.js';
import { describeShape, type DescribedShape } from '../describe/shape.js';
import { demand, listOf } from '../library/access.js';
import type { Library } from '../library/load.js';
import { pyRepr } from '../library/repr.js';

/** A derived count or byte figure: a number, or `null` where an expression did not resolve to one. */
export type Figure = bigint | number | null;

/**
 * `BYTES`: the width of one element of each dtype, in bytes.
 *
 * The sub-byte kinds are `0.5` and therefore `float`s, which is what makes a `fp4` tensor's
 * `bytes` a float in the derived document while a `bf16` tensor's is an integer. The whole widths
 * are `bigint`s for that reason: Python's `int * int` is an `int` and its `int * float` a `float`,
 * and the distinction travels into the products' bytes.
 */
export const BYTES: Readonly<Record<string, bigint | number>> = {
  bool: 1n,
  u4: 0.5,
  i4: 0.5,
  u8: 1n,
  i8: 1n,
  i16: 2n,
  i32: 4n,
  i64: 8n,
  fp4: 0.5,
  f8e4m3: 1n,
  f8e4m3fn: 1n,
  f8e5m2: 1n,
  bf16: 2n,
  f16: 2n,
  f32: 4n,
  f64: 8n,
};

/** `BYTES[dtype]`: the width, or the `KeyError` the tools raise for a name the table has not. */
export function widthOf(dtype: PyValue): bigint | number {
  const width = typeof dtype === 'string' ? BYTES[dtype] : undefined;
  if (width === undefined) throw new PyKeyError(pyRepr(dtype));
  return width;
}

/**
 * `_num(v)`: the value when it is a number, `None` otherwise.
 *
 * A boolean is excluded although Python's `bool` is an `int` — the tools write
 * `not isinstance(v, bool)` — and so is the sentinel, a string, a list and a record: an extent
 * that did not resolve to a number is a blank in the product, not a refusal.
 */
export function numberOf(value: PyValue): Figure {
  if (typeof value === 'boolean') return null;
  return typeof value === 'bigint' || typeof value === 'number' ? value : null;
}

/**
 * `_sound(value, kind, where)`: R11's guard on a derived count or byte figure.
 *
 * "After the argument domains (V3) and invariants (V8) it cannot be otherwise, so a violation here
 * is an internal inconsistency — a domain the validator is missing — named and raised, never
 * clamped." `math.isfinite` converts its argument to a float first, so an integer past the double
 * range raises there before the sign is ever looked at, and here too.
 */
export function sound(value: Figure, kind: string, where: string): Figure {
  if (value === null) return value;
  const asFloat = Number(value);
  if (typeof value === 'bigint' && !Number.isFinite(asFloat)) {
    throw new PyOverflowError('int too large to convert to float');
  }
  if (!Number.isFinite(asFloat) || value < 0) {
    throw new PyValueError(
      `${where}: derived ${kind} is ${pyRepr(value)} — negative or non-finite, which the ` +
        `validator's domains should have refused (admitted upstream, a domain is missing)`,
    );
  }
  return value;
}

/**
 * `_shape(shape, args)`: a shape as a product writes it — `{axis, extent}` per position, with the
 * declared `factors` where there are any.
 *
 * It is handed the shape **as stored**, `storageShape(param)`: "a slot that declares a
 * multiplicity leads with the storage axis `storage.multiplicity` (§3.4, finding 30) — the copies
 * have an address without being an axis of the computation". The evaluation is
 * {@link describeShape}'s, which is the same walk feature 1.6d lifted for the property sheet;
 * only `_num`'s reading of an extent is added, because a product writes a blank where a sheet
 * shows the sentinel. One walk, two readings — and feature 1.6d's parity suite already requires
 * the sheet's rows to equal D3's, extent for extent, on every corpus node.
 */
export function productShape(stored: PyValue, args: PyRecord): PyValue[] {
  return rowsOf(describeShape(stored, args));
}

/** {@link productShape}'s mapping, recursive over a flattened axis's factors. */
function rowsOf(shape: DescribedShape): PyValue[] {
  return shape.map((axis) => {
    const row: Record<string, PyValue> = { axis: axis.axis, extent: numberOf(axis.extent) };
    if (axis.factors !== undefined) row['factors'] = rowsOf(axis.factors);
    return row;
  });
}

/**
 * `_elements(shape, args, multiplicity)`: the product of a shape's extents, the declared count in
 * it once.
 *
 * The shape is the **declared** one and the multiplicity is applied after it, as the tools apply
 * it: the order of a product of floats is not the product of the same floats in another order, so
 * the multiplier stays where `_elements` puts it rather than being read off the stored shape.
 * `None` as soon as one extent does not resolve to a number — a count is a fact or it is absent.
 */
export function elementsOf(shape: PyValue, args: PyRecord, multiplicity: PyValue | null): Figure {
  let count: bigint | number = 1n;
  for (const axis of listOf(demand(shape, 'axes'))) {
    const extent = numberOf(primitiveValue(demand(axis, 'extent'), args));
    if (extent === null) return null;
    count = pyMultiply(count, extent) as bigint | number;
  }
  if (multiplicity !== null) {
    const copies = numberOf(primitiveValue(multiplicity, args));
    if (copies === null) return null;
    count = pyMultiply(count, copies) as bigint | number;
  }
  return count;
}

/**
 * `_dtype(graph, cat, selector, role)`: "the dtype a binding selects, else the role's default".
 *
 * Three readings, the tools' own: no selector is the precision role's `default` (V14), a literal
 * is itself, and `{"quantity": …}` is the quantity's value when it resolved to a string and the
 * role's default otherwise.
 *
 * **The quantities are the derivation's, not the template's.** `_expand` answers
 * `dict(graph, …)`, so `graph['quantities']` stays the *calling* document's at every level, and a
 * dtype selector written inside a template instance is looked up among the caller's quantities —
 * missing there, it falls back to the role's default. Reproduced, not corrected; the one template
 * of the repository declares `precision` external and the composite that instantiates it declares
 * a `precision` of its own with the same value, so no document of the corpus can tell.
 */
export function selectedDtype(
  quantities: Quantities,
  library: Library,
  selector: PyValue | null,
  role: PyValue,
): PyValue {
  if (selector === null) return defaultDtype(library, role);
  if (typeof selector === 'string') return selector;
  const name = demand(selector, 'quantity');
  const value = typeof name === 'string' ? quantities.get(name) : undefined;
  return typeof value === 'string' ? value : defaultDtype(library, role);
}

/** `cat['precision'][role]['default']`, and the `KeyError` an undeclared role raises. */
export function defaultDtype(library: Library, role: PyValue): PyValue {
  return demand(precisionRole(library, role), 'default');
}

/** `cat['precision'][role]['sensitivity']`: the role's quantisation advice, carried through (§7). */
export function sensitivityOf(library: Library, role: PyValue): PyValue {
  return demand(precisionRole(library, role), 'sensitivity');
}

/** `cat['precision'][role]`: the declared precision role, or Python's own `KeyError`. */
function precisionRole(library: Library, role: PyValue): PyValue {
  const found = typeof role === 'string' ? library.precision.get(role) : undefined;
  if (found === undefined) throw new PyKeyError(pyRepr(role));
  return found;
}
