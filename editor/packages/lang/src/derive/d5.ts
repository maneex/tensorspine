/**
 * D5, the Derived Logical Resource Requirements and Costs: the port of `derive.d5` (§7, derived
 * guide §7).
 *
 * D5 is the one product that computes almost nothing of its own. §4.1 fixes the **inventory rule**
 * — "for every parameter slot, two operations per weight element per element of the instance's
 * output domain, scaled by the activated fraction of a sparse unit (§4.5); the bytes of the
 * inventory, in full for residency" — and the validator applies it, because the same walk that
 * checks a parameter binding is the walk that knows which slots are present and what they cost:
 * `validate.analyse`'s own "D5, first derivation" block is where `parameter_elements` and the four
 * `ops_per_*` counters come from (ported in `validate/bindings/parameters.ts`, feature 1.6c).
 * D3, D4 and D2 supply the rest. What is left here is the *inventory the rule could not see*:
 *
 * - **corrections**, "an ordered list of entries, each guarded by a condition over the arguments,
 *   each stating an expression, a status (O0.5) and what it is counted per … every entry whose
 *   condition holds contributes" (§4.1) — one row per applying entry, with the index of the entry
 *   in the primitive's list, so that a reader can go back to the declaration;
 * - **sparsity bounds** (§4.5): per declared unit, how many units an element activates, how many
 *   there are, the fraction that is, and the declared upper bound on "the union of units activated
 *   per invocation" — "a per-element count is insufficient because an invocation's union may
 *   include every unit";
 * - the **totals**, each with the status §2.2's algebra gives it: the parameter inventory and the
 *   state bytes are `exact` ("everything derived from the inventory is exact", derived guide §2),
 *   and each `operations` figure carries the sum of `exact` with the status of every correction
 *   counted per that unit — `qualified.ts`'s {@link sumStatus}, the tools' own `_status`;
 * - the **payload of every graph split**, which is D2's, restated per element and per invocation
 *   because a split's cost is a cost and belongs where the costs are read.
 *
 * **The walk is the resolved nodes in their own order**, and a node inside a template instance is
 * one of them under its prefix (§5.2 rule 2) — so a composite's corrections are its template's,
 * renamed, and the flat document and the composite answer the same figures (`tests/run_costs.py`).
 *
 * **Three places where the tools read one fact two ways**, each reproduced rather than reconciled:
 * `derive.d5` reads a correction's value through `_num`, which answers a blank for a boolean and
 * for the sentinel, while the validator's counter adds whatever is not the sentinel and not
 * `None`; `derive.d5` subscripts `definition['parameters'][pname]` for a sparsity unit's slot
 * where the validator's counter reads `.get(pname)` and skips a slot the primitive has not; and a
 * unit's `units` extent is found by *the last* slot and *the last* axis that match, neither loop
 * stopping. The loader refuses a unit naming a slot its primitive has not, so the subscript raises
 * for no gathered primitive; the readings still differ, and the port keeps each where its tool
 * writes it.
 */
import { pyDivide, pyEqual } from '../expr/arithmetic.js';
import { PyKeyError } from '../expr/errors.js';
import { primitiveCondition, primitiveValue } from '../expr/primitive.js';
import { truthy, type PyRecord, type PyValue } from '../expr/value.js';
import { put } from '../json/tree.js';
import { demand, has, listOf, optional } from '../library/access.js';
import { pyRepr, pyStr } from '../library/repr.js';
import { numberOf, type Figure } from './figures.js';
import { identOf, type ExpandedGraph, type ExpandedNode } from './expand.js';
import { sumStatus } from './qualified.js';

/**
 * What a `per` of a correction is counted against, and the counter the validator's first
 * derivation keeps it in.
 *
 * The tools write the pairs as a literal tuple, in the order `operations` is written in; the names
 * on the left are the `cost_entry.per` enumeration and the audit of §1 (d) holds them to it, in
 * both directions. The names on the right are `--validate`'s own counters.
 */
export const OPERATION_COUNTERS: readonly (readonly [string, string])[] = [
  ['element', 'ops_per_element'],
  ['cached_position', 'ops_per_cached_position'],
  ['sequence', 'ops_per_sequence'],
  ['invocation', 'ops_per_invocation'],
];

/**
 * `d5(graph, cat, products3, products4, products2, stats)`: the logical costs of one expanded
 * graph.
 *
 * The tools' `cat` is dead — the body never reads it, `_dtype` and the precision roles belonging
 * to the products that write bytes — so it is not a parameter here, as `_visits`' first parameter
 * was not (feature 1.8b) and `_check_type`'s two were not (feature 1.5).
 */
export function d5(
  graph: ExpandedGraph,
  products3: PyValue,
  products4: PyValue,
  products2: PyValue,
  stats: ReadonlyMap<string, PyValue>,
): PyRecord {
  const corrections: PyRecord[] = [];
  const sparsity: PyRecord[] = [];
  for (const [, node] of graph.resolved) {
    corrections.push(...correctionsOf(node));
    sparsity.push(...sparsityOf(node));
  }

  const product: PyRecord = {};
  put(product, 'parameters', totals(demand(products3, 'totals'), ['elements', 'bytes']));
  put(product, 'operations', operationsOf(corrections, stats));
  put(product, 'corrections', corrections);
  put(product, 'sparsity', sparsity);
  put(
    product,
    'state',
    totals(demand(products4, 'totals'), [
      'append_bytes_per_cached_position',
      'bounded_bytes',
      'fixed_bytes',
    ]),
  );
  put(product, 'graph_splits', payloadsOf(products2));
  return product;
}

/**
 * "A primitive declares only **corrections**: an ordered list of entries, each guarded by a
 * condition over the arguments … Every entry whose condition holds contributes" (§4.1).
 *
 * The index is the entry's place in the primitive's own list and is written whether or not earlier
 * entries applied, so a reader lands on the declaration that produced the row. The value is `_num`
 * of the expression: a blank where it did not resolve to a number, never a refusal — a correction
 * the arguments leave undetermined is stated as undetermined, which is what the derived schema's
 * nullable `value` is for.
 */
function correctionsOf(node: ExpandedNode): PyRecord[] {
  const rows: PyRecord[] = [];
  const declared = listOf(optional(node.definition, 'logical_cost', []));
  for (let index = 0; index < declared.length; index += 1) {
    const entry = declared[index] as PyValue;
    if (has(entry, 'when') && !truthy(primitiveCondition(demand(entry, 'when'), node.args))) {
      continue;
    }
    const row: PyRecord = {};
    put(row, 'node', identOf(node.site));
    put(row, 'primitive', node.primitive);
    put(row, 'entry', BigInt(index));
    put(row, 'value', numberOf(primitiveValue(demand(entry, 'expression'), node.args)));
    put(row, 'status', demand(entry, 'status'));
    put(row, 'per', demand(entry, 'per'));
    rows.push(row);
  }
  return rows;
}

/**
 * One row per sparsity unit a primitive declares (§4.5), whether or not the instance's arguments
 * make it bite: the unit is a property of the primitive, and a fraction of one is a fraction.
 *
 * `units` is the extent of the unit's axis in the slots that form the unit — "the parameter slots
 * that form one unit and the axis along which units are laid out" — and it is searched the tools'
 * way: **every** named slot and **every** axis of each are walked, and the last match wins. The
 * loader requires each named slot to exist and to carry the axis, so a unit naming two slots finds
 * the same extent twice; the walk is kept as written because nothing guarantees that a primitive
 * outside the reference base declares two slots of one extent.
 *
 * `activated_fraction` is Python's conditional: a blank when the count did not resolve **or** when
 * the extent is absent or zero — the truth of the extent, not its presence, so a unit laid out
 * along an axis of extent zero has no fraction rather than a division by zero.
 */
function sparsityOf(node: ExpandedNode): PyRecord[] {
  const rows: PyRecord[] = [];
  const units = listOf(optional(node.definition, 'sparsity', []));
  for (let index = 0; index < units.length; index += 1) {
    const declared = demand(units[index] as PyValue, 'unit');
    const activated = numberOf(
      primitiveValue(demand(units[index] as PyValue, 'activated_per_element'), node.args),
    );
    let extent: Figure = null;
    for (const slotName of listOf(demand(declared, 'parameters'))) {
      // `definition['parameters'][pname]`, subscripted: the validator's own counter reads it with
      // `.get(pname)` and skips what is absent, and this does not. The loader's
      // "unit parameter '…' is not a slot of this primitive" refuses the difference away.
      const slot = demand(demand(node.definition, 'parameters'), pyStr(slotName));
      for (const axis of listOf(demand(demand(slot, 'shape'), 'axes'))) {
        if (pyEqual(demand(axis, 'axis'), demand(declared, 'axis'))) {
          extent = numberOf(primitiveValue(demand(axis, 'extent'), node.args));
        }
      }
    }
    const bound = demand(units[index] as PyValue, 'union_per_invocation');
    const union: PyRecord = {};
    put(union, 'value', numberOf(primitiveValue(demand(bound, 'expression'), node.args)));
    put(union, 'status', demand(bound, 'status'));

    const row: PyRecord = {};
    put(row, 'node', identOf(node.site));
    put(row, 'primitive', node.primitive);
    put(row, 'unit', BigInt(index));
    put(row, 'activated_per_element', activated);
    put(row, 'units', extent);
    put(
      row,
      'activated_fraction',
      activated !== null && truthy(extent) ? pyDivide(activated, extent) : null,
    );
    put(row, 'union_per_invocation', union);
    rows.push(row);
  }
  return rows;
}

/**
 * `operations`: the inventory rule's four counters, each with the status of its own sum.
 *
 * The value is the validator's counter — the inventory rule of §4.1 applied while the parameter
 * bindings were checked — and the status is `_status(['exact'] + [the status of every correction
 * counted per this unit])`. `exact` leads the list because the inventory figure itself is exact
 * (derived guide §2: "everything derived from the inventory is exact; a status other than `exact`
 * always comes from a primitive's declared correction or sparsity bound"), so a unit no correction
 * touches is exact and an estimate anywhere in a unit's corrections makes that unit's total an
 * estimate. The lead changes no answer — `_status` reads a **set**, and an empty one answers
 * `exact` already — so it is kept because it is what the tools write and because it states, in the
 * call itself, where the exactness of a unit's own figure comes from.
 */
function operationsOf(
  corrections: readonly PyRecord[],
  stats: ReadonlyMap<string, PyValue>,
): PyRecord {
  const byPer = new Map<string, string[]>();
  for (const row of corrections) {
    const per = pyStr(row['per'] as PyValue);
    const held = byPer.get(per);
    if (held === undefined) byPer.set(per, [pyStr(row['status'] as PyValue)]);
    else held.push(pyStr(row['status'] as PyValue));
  }
  const operations: PyRecord = {};
  for (const [per, counter] of OPERATION_COUNTERS) {
    const entry: PyRecord = {};
    put(entry, 'value', counterOf(stats, counter));
    put(entry, 'status', sumStatus(['exact', ...(byPer.get(per) ?? [])]));
    put(operations, per, entry);
  }
  return operations;
}

/** `stats[name]`, subscripted: a counter the analysis did not keep is Python's own `KeyError`. */
function counterOf(stats: ReadonlyMap<string, PyValue>, name: string): PyValue {
  const held = stats.get(name);
  if (held === undefined) throw new PyKeyError(pyRepr(name));
  return held;
}

/** One `{…figures, status: 'exact'}` block over another product's `totals`. */
function totals(source: PyValue, names: readonly string[]): PyRecord {
  const block: PyRecord = {};
  for (const name of names) put(block, name, demand(source, name));
  put(block, 'status', 'exact');
  return block;
}

/**
 * `graph_splits`: D2's splits, keeping the two figures a cost reader wants.
 *
 * The payload itself — which values cross, and what each weighs — stays in D2 (derived guide §8:
 * "the payload itself and its bytes stay in D2 and D5"); what is restated here is the split's
 * identifier and its two totals, per element of the graph's own domain and per invocation by
 * public input. The `bytes_per_invocation` record is D2's own object, as the tools hand it on.
 */
function payloadsOf(products2: PyValue): PyValue[] {
  return listOf(demand(products2, 'graph_splits')).map((split) => {
    const written: PyRecord = {};
    put(written, 'graph_split', demand(split, 'graph_split'));
    put(written, 'bytes_per_element', demand(split, 'bytes_per_element'));
    put(written, 'bytes_per_invocation', demand(split, 'bytes_per_invocation'));
    return written;
  });
}
