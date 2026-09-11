/**
 * A shape as the editor shows it: the declaration's axis rows with their extents evaluated.
 *
 * The validator compares shapes as {@link ShapeIdentity} — "axis identity and extent, position by
 * position; local names and natures are not compared" (V4, V15) — which is all a *verdict* needs.
 * A reader needs more: §4.7's handle tooltip prints `bf16[tokens, model.width=4096]`, §4.11's
 * Parameters row prints the evaluated shape, §4.22's axis-row editor shows the local name, the
 * axis, the nature, the extent and the factors of every row. So this is the same walk with the
 * declaration's own fields kept beside the two the comparison reads.
 *
 * It is `derive._shape` without the products' arithmetic: the same `primitive_value` over the same
 * `args`, in the same order, with a slot's storage axis already in the shape it is handed
 * ({@link storageShape}) — which is why feature 1.6d's parity test can require the rows it answers
 * to equal D2's and D3's, extent for extent. `_num`'s reading of an extent (a number, or `None`
 * where the expression did not resolve to one) belongs to the products, not here: an extent that
 * resolves to the sentinel is a fact the sheet shows, not a blank.
 */
import { primitiveValue } from '../expr/primitive.js';
import type { PyRecord, PyValue } from '../expr/value.js';
import { demand, has, listOf, optional } from '../library/access.js';

/** One row of a shape: the declaration's four fields, the extent evaluated. */
export interface DescribedAxis {
  /** The local name the declaration gives this position — `feature`, `multiplicity`. */
  readonly name: PyValue;
  /** The axis identity of the primitive library: `model.width`, `storage.multiplicity`. */
  readonly axis: PyValue;
  /** `feature`, `structural`, `storage` … — the axis's nature, as the declaration writes it. */
  readonly nature: PyValue;
  /** The extent, evaluated in the instance's arguments. */
  readonly extent: PyValue;
  /** The declared decomposition of a flattened axis (O5.10); absent when it declares none. */
  readonly factors?: readonly DescribedAxis[];
}

/** A shape as the editor shows it: one {@link DescribedAxis} per position, in declaration order. */
export type DescribedShape = readonly DescribedAxis[];

/** `_shape(shape, args)`: every axis of a declared shape with its extent evaluated. */
export function describeShape(shape: PyValue, args: PyRecord): DescribedShape {
  return listOf(demand(shape, 'axes')).map((axis) => describeAxis(axis, args));
}

/** One axis row, with its factors where the declaration carries them. */
function describeAxis(axis: PyValue, args: PyRecord): DescribedAxis {
  const row: DescribedAxis = {
    name: demand(axis, 'name'),
    axis: demand(axis, 'axis'),
    nature: demand(axis, 'nature'),
    extent: primitiveValue(demand(axis, 'extent'), args),
  };
  if (!has(axis, 'factors')) return row;
  return {
    ...row,
    factors: listOf(optional(axis, 'factors', [])).map((factor) => describeAxis(factor, args)),
  };
}
