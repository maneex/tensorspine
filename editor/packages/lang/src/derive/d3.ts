/**
 * D3, the Derived Parameter Tensor Inventory: the port of `derive.d3` (§7, derived guide §5).
 *
 * "Shapes as stored (a declared multiplicity leading, §3.4), sharing, and total count; the role,
 * selected dtype and sensitivity of every tensor; when the document locates its weights, the
 * evaluated location of every tensor." One entry per parameter identity instance — **a tied tensor
 * once**, which is what makes D3 the inventory a loader reads rather than a list of slots: the
 * identity is the tensor, its members are the slots it satisfies, and `tied` says whether there is
 * more than one.
 *
 * Everything an entry says about a slot is read from its **first** member's definition — "of the
 * first member; V15 makes the others compatible" — so the product never has to reconcile two
 * declarations, and an identity whose members all name unresolved sites has no entry at all.
 *
 * Three figures and their absence: `elements` is the product of the stored shape's extents with
 * the declared count in it once, `bytes` is that times the dtype's width, and either is `null`
 * where an extent did not resolve to a number. `_sound` guards both against a negative or
 * non-finite value, which after V3's domains and V8's invariants can only mean the validator is
 * missing a domain (R11) — named and raised, never clamped.
 */
import { pyDivide, pyEqual, pyMultiply } from '../expr/arithmetic.js';
import { primitiveValue } from '../expr/primitive.js';
import { truthy, type PyRecord, type PyValue } from '../expr/value.js';
import { put } from '../json/tree.js';
import { demand, has, listOf, optional } from '../library/access.js';
import type { Library } from '../library/load.js';
import { pyStr } from '../library/repr.js';
import { storageShape } from '../validate/bindings/locations.js';
import {
  elementsOf,
  numberOf,
  orZero,
  productShape,
  pySum,
  selectedDtype,
  sensitivityOf,
  sound,
  widthOf,
  type Figure,
} from './figures.js';
import {
  identOf,
  locatedValue,
  nodeAt,
  type ExpandedGraph,
  type ExpandedNode,
  type ExpandedTensorInstance,
} from './expand.js';

/** `d3(graph, cat)`: the parameter tensor inventory of one expanded graph. */
export function d3(graph: ExpandedGraph, library: Library): PyRecord {
  const tensors: PyRecord[] = [];
  for (const instance of graph.tensorInstances) {
    const entry = tensorOf(graph, library, instance);
    if (entry !== null) tensors.push(entry);
  }
  const product: PyRecord = {};
  put(product, 'tensors', tensors);
  put(product, 'totals', totalsOf(tensors));
  return product;
}

/** One entry, or `null` for an identity instance no member of which resolved. */
function tensorOf(
  graph: ExpandedGraph,
  library: Library,
  instance: ExpandedTensorInstance,
): PyRecord | null {
  const first = instance.members[0];
  if (first === undefined) return null;
  const node = nodeAt(graph, first);
  const slot = pyStr(first.name);
  const declared = demand(demand(node.definition, 'parameters'), slot);
  const role = demand(declared, 'role');
  const dtype = selectedDtype(graph.quantities, library, instance.dtype, role);
  const multiplicity = has(declared, 'multiplicity') ? demand(declared, 'multiplicity') : null;
  const count = elementsOf(demand(declared, 'shape'), node.args, multiplicity);

  const entry: PyRecord = {};
  put(entry, 'identity', instance.identity);
  put(
    entry,
    'members',
    instance.members.map((member) => `${identOf(member.site)}.${pyStr(member.name)}`),
  );
  put(entry, 'primitive', node.primitive);
  put(entry, 'slot', first.name);
  put(entry, 'role', role);
  put(entry, 'sensitivity', sensitivityOf(library, role));
  put(entry, 'dtype', dtype);
  put(entry, 'shape', productShape(storageShape(declared), node.args));
  put(
    entry,
    'multiplicity',
    multiplicity === null ? 1n : numberOf(primitiveValue(multiplicity, node.args)),
  );
  put(entry, 'elements', sound(count, 'element count', instance.identity));
  put(
    entry,
    'bytes',
    sound(count === null ? null : bytesOf(count, dtype), 'byte size', instance.identity),
  );
  put(entry, 'tied', instance.members.length > 1);

  const unit = sparsityOf(node, declared, first.name);
  if (unit !== null) put(entry, 'sparsity', unit);
  if (instance.location !== undefined) put(entry, 'location', locatedValue(instance.location));
  return entry;
}

/** `n * BYTES[dtype]`, in Python's arithmetic: a whole width keeps an integer, `0.5` makes a float. */
function bytesOf(count: bigint | number, dtype: PyValue): Figure {
  return pyMultiply(count, widthOf(dtype)) as Figure;
}

/**
 * The sparsity unit a slot belongs to (§4.5), or `null`.
 *
 * "Cost derivation counts a unit's weights at the activated fraction per element, and in full for
 * residency and worst-case transfer"; D3 records what that fraction is. The walk is the tools'
 * own: **every** declared unit is looked at and the last one naming the slot wins, as does the
 * last axis of the slot's shape carrying the unit's axis — neither loop breaks, and no unit of the
 * reference base names a slot twice. The extent is read from the slot's **declared** shape, not
 * its stored one: a unit is laid out along an axis of the computation, and the storage axis is in
 * no primitive shape (§3.4).
 */
function sparsityOf(node: ExpandedNode, declared: PyValue, slot: PyValue): PyRecord | null {
  let unit: PyRecord | null = null;
  const units = listOf(optional(node.definition, 'sparsity', []));
  for (let index = 0; index < units.length; index += 1) {
    const declaredUnit = demand(units[index] as PyValue, 'unit');
    if (!listOf(demand(declaredUnit, 'parameters')).some((name) => pyEqual(name, slot))) continue;
    const axis = demand(declaredUnit, 'axis');
    const activated = numberOf(
      primitiveValue(demand(units[index] as PyValue, 'activated_per_element'), node.args),
    );
    let extent: Figure = null;
    for (const shapeAxis of listOf(demand(demand(declared, 'shape'), 'axes'))) {
      if (pyEqual(demand(shapeAxis, 'axis'), axis)) {
        extent = numberOf(primitiveValue(demand(shapeAxis, 'extent'), node.args));
      }
    }
    const found: PyRecord = {};
    put(found, 'unit', BigInt(index));
    put(found, 'axis', axis);
    put(found, 'activated_per_element', activated);
    put(found, 'units', extent);
    put(
      found,
      'activated_fraction',
      activated !== null && extent !== null && truthy(extent) ? pyDivide(activated, extent) : null,
    );
    unit = found;
  }
  return unit;
}

/** `totals`: the four sums, `or 0` over a figure the shape left undetermined. */
function totalsOf(tensors: readonly PyRecord[]): PyRecord {
  const totals: PyRecord = {};
  put(totals, 'tensors', BigInt(tensors.length));
  put(totals, 'elements', summed(tensors, 'elements'));
  put(totals, 'bytes', summed(tensors, 'bytes'));
  put(totals, 'tied', BigInt(tensors.filter((entry) => entry['tied'] === true).length));
  return totals;
}

/** `sum(t[name] or 0 for t in tensors)`: {@link pySum} over {@link orZero} of each figure. */
function summed(tensors: readonly PyRecord[], name: string): bigint | number {
  return pySum(tensors.map((entry) => orZero(entry[name] as Figure | undefined)));
}
