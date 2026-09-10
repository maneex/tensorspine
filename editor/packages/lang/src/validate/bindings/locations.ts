/**
 * Where a parameter identity's tensor is stored: `_storage_shape`, `evaluate_location` and
 * `location_names` of `tools/validate.py` (§3.4, V17).
 *
 * A location is written against the slot's shape **as stored**, which is not the shape the
 * primitive declares: "a slot that declares a multiplicity is m tensors of its shape, applied
 * independently (one per auxiliary stream, one per shared expert); it is stored with one axis
 * before its shape axes — local name `multiplicity`, the primitive library's
 * `storage.multiplicity`, extent m — which every form addresses as it addresses a shape axis".
 * {@link storageShape} is that shape; it is what a location addresses, what V15 compares and what
 * D3 writes, and it is absent from the computation's shape.
 *
 * {@link evaluateLocation} answers the *evaluated* form: names substituted, axes resolved to their
 * position, a stack expanded over its axis, a slice with its offset and the slot's extent — and
 * the problems it met, which V17 prefixes with the identity instance. {@link locationNames} then
 * reads back the whole physical names it uses and the regions it slices, which is what "a physical
 * name is bound by one identity; the slices of one physical tensor do not overlap and do not
 * coexist with a whole binding of it" is checked over.
 *
 * The names themselves are `_physical_name`'s, ported by feature 1.6b for the one place a name is
 * evaluated before the slots are — an instance's `weights_location_prefix` — and reused here.
 */
import type { Env } from '../../expr/model.js';
import { primitiveValue } from '../../expr/primitive.js';
import { UNRESOLVED, type PyRecord, type PyValue } from '../../expr/value.js';
import { PyTypeError } from '../../expr/errors.js';
import { demand, has, listOf, optional } from '../../library/access.js';
import { pyRepr, pyStr } from '../../library/repr.js';
import { pyInt } from '../conformance.js';
import { physicalName } from '../graph.js';

import type { EvaluatedLocation } from './analysis.js';

/** `STORAGE_AXIS`: the axis identity a declared multiplicity is stored along (§3.4, finding 30). */
export const STORAGE_AXIS = 'storage.multiplicity';

/**
 * `_storage_shape(param)`: "a parameter slot's shape as stored".
 *
 * "A declared multiplicity is a leading axis — local name `multiplicity`, the primitive_library's
 * `storage.multiplicity`, extent the count — before the shape axes. What a location addresses
 * (V17), what V15 compares, what D3 writes; absent from the computation's shape, which the
 * primitive declares."
 */
export function storageShape(param: PyValue): PyValue {
  if (!has(param, 'multiplicity')) return demand(param, 'shape');
  const lead: PyRecord = {
    name: 'multiplicity',
    axis: STORAGE_AXIS,
    nature: 'storage',
    extent: demand(param, 'multiplicity'),
  };
  return { axes: [lead, ...listOf(demand(demand(param, 'shape'), 'axes'))] };
}

/** How a location is read: the model-level evaluator, which a `{index}` in a name goes through. */
export type LocationValue = (expression: PyValue, env: Env) => PyValue;

/** {@link evaluateLocation}'s answer: the evaluated form, or `null`, and what it refused with. */
export interface LocationResult {
  readonly evaluated: EvaluatedLocation | null;
  readonly problems: readonly string[];
}

/**
 * `evaluate_location(loc, env, shape, args, value, coordinates, in_concat)`.
 *
 * "The evaluated form of a location (D3): names substituted, axes resolved to their position, a
 * stack expanded over its axis, a slice with its offset and the slot's extent."
 *
 * The extents of *every* axis are evaluated first, as the tools evaluate them, so an expression
 * that raises does so whether or not the form names its axis.
 */
export function evaluateLocation(
  location: PyValue,
  env: Env,
  shape: PyValue,
  args: PyRecord,
  value: LocationValue,
  coordinates: ReadonlyMap<string, PyValue> = new Map(),
  inConcat = false,
): LocationResult {
  const declared = listOf(demand(shape, 'axes'));
  const axes = declared.map((axis) => demand(axis, 'name'));
  const extents = declared.map((axis) => primitiveValue(demand(axis, 'extent'), args));
  const problems: string[] = [];

  /** `dim_of(axis, what)`: the axis's position in the stored shape, or a refusal naming it. */
  const dimOf = (axis: PyValue, what: string): bigint | null => {
    const at = axes.findIndex((one) => pythonEqual(one, axis));
    if (at < 0) {
      problems.push(
        `${what}: '${pyStr(axis)}' is not an axis of the slot (axes: ${joinNames(axes)})`,
      );
      return null;
    }
    return BigInt(at);
  };

  if (has(location, 'tensor')) {
    const written = physicalName(demand(location, 'tensor'), env, value, coordinates);
    if (written.problem !== null) problems.push(written.problem);
    // `{"tensor": name} if name else None`: an empty name is no location, and no refusal either.
    const evaluated = written.name === null || written.name === '' ? null : { tensor: written.name };
    return { evaluated, problems };
  }
  if (has(location, 'stack')) {
    const stack = demand(location, 'stack');
    const axis = demand(stack, 'axis');
    const dim = dimOf(axis, 'stack');
    if (dim === null) return { evaluated: null, problems };
    const extent = extents[Number(dim)] as PyValue;
    const count = wholeCount(extent);
    if (count === null) {
      problems.push(
        `stack: axis '${pyStr(axis)}' has no coordinates — its extent resolves to ${pyRepr(extent)}`,
      );
      return { evaluated: null, problems };
    }
    const parts: EvaluatedLocation[] = [];
    for (let at = 0n; at < count; at += 1n) {
      const deeper = new Map(coordinates);
      deeper.set(pyStr(axis), at);
      const inner = evaluateLocation(
        demand(stack, 'part'),
        env,
        shape,
        args,
        value,
        deeper,
        inConcat,
      );
      problems.push(...inner.problems);
      if (inner.evaluated === null) return { evaluated: null, problems };
      parts.push(inner.evaluated);
    }
    return { evaluated: { stack: { axis, dim, parts } }, problems };
  }
  if (has(location, 'concat')) {
    const concat = demand(location, 'concat');
    const axis = demand(concat, 'axis');
    const dim = dimOf(axis, 'concat');
    if (dim === null) return { evaluated: null, problems };
    const parts: EvaluatedLocation[] = [];
    for (const part of listOf(demand(concat, 'parts'))) {
      const inner = evaluateLocation(part, env, shape, args, value, coordinates, true);
      problems.push(...inner.problems);
      if (inner.evaluated === null) return { evaluated: null, problems };
      parts.push(inner.evaluated);
    }
    return { evaluated: { concat: { axis, dim, parts } }, problems };
  }
  if (has(location, 'slice')) {
    if (inConcat) {
      problems.push('slice inside a concat: its extent along the axis would be unknown');
      return { evaluated: null, problems };
    }
    const slice = demand(location, 'slice');
    const axis = demand(slice, 'axis');
    const dim = dimOf(axis, 'slice');
    if (dim === null) return { evaluated: null, problems };
    const written = physicalName(demand(slice, 'tensor'), env, value, coordinates);
    if (written.problem !== null) {
      problems.push(written.problem);
      return { evaluated: null, problems };
    }
    const offset = value(demand(slice, 'offset'), env);
    if (
      offset === UNRESOLVED ||
      typeof offset === 'boolean' ||
      typeof offset !== 'bigint' ||
      offset < 0n
    ) {
      problems.push('slice: the offset does not resolve to a non-negative integer');
      return { evaluated: null, problems };
    }
    return {
      evaluated: {
        slice: {
          tensor: written.name as string,
          axis,
          dim,
          offset,
          extent: wholeOf(extents[Number(dim)] as PyValue),
        },
      },
      problems,
    };
  }
  problems.push('unknown location form');
  return { evaluated: null, problems };
}

/** One region a location slices: `(name, offset, extent)`, before an identity is attached. */
export interface SlicedRegion {
  readonly name: string;
  readonly offset: bigint;
  readonly extent: bigint;
}

/** {@link locationNames}' answer: the names used whole, and the regions sliced. */
export interface LocationUse {
  readonly whole: readonly string[];
  readonly slices: readonly SlicedRegion[];
}

/**
 * `location_names(ev)`: "the whole physical names a location uses, and its slices as
 * (name, offset, extent)".
 */
export function locationNames(evaluated: EvaluatedLocation): LocationUse {
  const whole: string[] = [];
  const slices: SlicedRegion[] = [];
  const walk = (one: EvaluatedLocation): void => {
    if ('tensor' in one) {
      whole.push(one.tensor);
      return;
    }
    if ('stack' in one) {
      for (const part of one.stack.parts) walk(part);
      return;
    }
    if ('concat' in one) {
      for (const part of one.concat.parts) walk(part);
      return;
    }
    slices.push({ name: one.slice.tensor, offset: one.slice.offset, extent: one.slice.extent });
  };
  walk(evaluated);
  return { whole, slices };
}

/**
 * `not isinstance(n, (int, float)) or isinstance(n, bool) or n != int(n) or n < 1`, negated.
 *
 * The stack's guard, and the multiplicity's: "a present slot's count resolves to a positive
 * integer: a slot with no copies is absent by its condition, never by arithmetic" (§3.4, V7). A
 * float that is a whole number passes, a boolean never does, and the count is answered as the
 * integer the caller iterates with. `int(n)` raises on a non-finite float where CPython raises:
 * the chain reaches it before the comparison with 1.
 */
export function wholeCount(value: PyValue): bigint | null {
  if (typeof value === 'boolean') return null;
  if (typeof value !== 'bigint' && typeof value !== 'number') return null;
  const whole = pyInt(value);
  if (typeof value === 'number' && value !== Number(whole)) return null;
  return whole >= 1n ? whole : null;
}

/** `int(extents[dim])` for a slice's extent, which the tools take without checking it first. */
function wholeOf(value: PyValue): bigint {
  if (typeof value === 'bigint' || typeof value === 'number') return pyInt(value);
  // `int()` of anything else is CPython's refusal, and the tools raise it here as everywhere.
  throw new PyTypeError(
    `int() argument must be a string, a bytes-like object or a real number, not '${typeName(value)}'`,
  );
}

/** Python's `==` on the two things an axis name can be here: the shape's, and the form's. */
function pythonEqual(one: PyValue, other: PyValue): boolean {
  return typeof one === 'string' || typeof other === 'string'
    ? one === other
    : Object.is(one, other);
}

/**
 * `', '.join(axes)`: the axis names a refusal lists.
 *
 * Every one of them is an `identifier` of the unit schema, so the join always succeeds on a base
 * the loader accepted; a name that is not a string raises where CPython raises rather than being
 * written as something it is not.
 */
function joinNames(axes: readonly PyValue[]): string {
  return axes
    .map((name, at) => {
      if (typeof name !== 'string') {
        throw new PyTypeError(
          `sequence item ${at}: expected str instance, ${typeName(name)} found`,
        );
      }
      return name;
    })
    .join(', ');
}

/** The name CPython gives a value's type in those two messages. */
function typeName(value: PyValue): string {
  if (value === UNRESOLVED) return 'object';
  if (value === null) return 'NoneType';
  if (typeof value === 'boolean') return 'bool';
  if (typeof value === 'bigint') return 'int';
  if (typeof value === 'number') return 'float';
  if (typeof value === 'string') return 'str';
  return Array.isArray(value) ? 'list' : 'dict';
}

/** `param.get('multiplicity')` — read once, so the callers that need it agree on the name. */
export function declaredMultiplicity(param: PyValue): PyValue | null {
  return optional(param, 'multiplicity', null);
}
