/**
 * The model side of `tools/expr.py`: expressions and conditions over a document's *quantities*
 * and the composition indices currently in scope.
 *
 * Derivations, declared defaults, domain bounds, index ranges, guards, argument values, selector
 * indices and location offsets are all written in this language (§2.2). Its two references are a
 * quantity name and an index name, and neither is ever guessed: what the document does not
 * supply is `UNRESOLVED`, and the caller decides whether that is acceptable — a guard that does
 * not resolve is a rejection (V10), never a false (I7).
 *
 * The difference from the primitive side, and it is the point of the module:
 *
 * - `modelCondition` answers `UNRESOLVED` where `primitiveCondition` answers false. A guard on a
 *   site or a binding is refused rather than silently dropped.
 * - `all` and `any` evaluate **every** part before looking for an unresolved one, so a
 *   conjunction with one undecidable part is undecidable even when another part is false. The
 *   primitive side short-circuits; this side does not, and the tools' answers differ accordingly.
 * - A comparison of two values Python cannot order is `UNRESOLVED` here — `model_condition`
 *   catches the `TypeError` — and raises on the primitive side.
 */
import { PyTypeError, PyValueError } from './errors.js';
import { apply, compare } from './arithmetic.js';
import { comparePythonStrings } from '../schema/index.js';
import { put } from '../json/tree.js';

import {
  asRecord,
  conditionNode,
  demandKey as demand,
  expressionNode,
  hasKey,
  isRecord,
  items,
  member,
  pyIterate,
  pythonTypeName,
  truthy,
  UNRESOLVED,
  type PyRecord,
  type PyValue,
} from './value.js';

/** The statically known quantities of a document, by name. */
export type Quantities = ReadonlyMap<string, PyValue>;

/** The composition indices in scope, by name: what a `for_each` grid binds at one point. */
export type Env = ReadonlyMap<string, PyValue>;

/**
 * A bound that cannot be evaluated because an external quantity has no value.
 *
 * "Not a defect of the document: a template denotes one graph per admissible assignment (§4.6),
 * so reading it alone requires one." It is a `ValueError` in the tools, which is what `--d1` and
 * `--derive` catch and print as the document's refusal.
 */
export class Unassigned extends Error {}

/** Value of a model expression against quantities and loop indices. */
export function modelValue(expression: PyValue, quantities: Quantities, env?: Env): PyValue {
  const scope = env ?? new Map<string, PyValue>();
  const e = expressionNode(expression);
  if (e === undefined) return UNRESOLVED;
  if (hasKey(e, 'literal')) return member(e, 'literal') as PyValue;
  if (hasKey(e, 'quantity')) return lookup(quantities, member(e, 'quantity') as PyValue);
  if (hasKey(e, 'index')) return lookup(scope, member(e, 'index') as PyValue);
  if (hasKey(e, 'op')) {
    const operands = pyIterate(demand(e, 'args')).map((one) => modelValue(one, quantities, scope));
    if (operands.includes(UNRESOLVED)) return UNRESOLVED;
    return apply(member(e, 'op') as PyValue, operands);
  }
  if (hasKey(e, 'if')) {
    const truth = modelCondition(member(e, 'if') as PyValue, quantities, scope);
    if (truth === UNRESOLVED) return UNRESOLVED;
    return modelValue(demand(e, truthy(truth) ? 'then' : 'else'), quantities, scope);
  }
  return UNRESOLVED;
}

/** `map.get(name, UNRESOLVED)`, with a name that is not a string finding nothing. */
function lookup(map: ReadonlyMap<string, PyValue>, name: PyValue): PyValue {
  if (typeof name !== 'string') return UNRESOLVED;
  const found = map.get(name);
  return found === undefined ? UNRESOLVED : found;
}

/**
 * Truth of a model condition — a `when` guard on a site or a binding, or the test of a
 * conditional expression. `UNRESOLVED` when it cannot be decided.
 */
export function modelCondition(condition: PyValue, quantities: Quantities, env?: Env): PyValue {
  const scope = env ?? new Map<string, PyValue>();
  const c = conditionNode(condition);
  if (hasKey(c, 'boolean')) return member(c, 'boolean') as PyValue;
  if (hasKey(c, 'not')) {
    const truth = modelCondition(member(c, 'not') as PyValue, quantities, scope);
    return truth === UNRESOLVED ? UNRESOLVED : !truthy(truth);
  }
  if (hasKey(c, 'all') || hasKey(c, 'any')) {
    const all = hasKey(c, 'all');
    const listed = all ? (member(c, 'all') as PyValue) : (member(c, 'any') as PyValue);
    // Every part is evaluated before the undecidable ones are looked for: `all` with a false
    // part and an undecidable one is undecidable, where the primitive side would stop at false.
    const parts = pyIterate(listed).map((one) => modelCondition(one, quantities, scope));
    if (parts.includes(UNRESOLVED)) return UNRESOLVED;
    return all ? parts.every((one) => truthy(one)) : parts.some((one) => truthy(one));
  }
  const comparison = asRecord(demand(c, 'compare'));
  const left = modelValue(demand(comparison, 'left'), quantities, scope);
  const right = modelValue(demand(comparison, 'right'), quantities, scope);
  if (left === UNRESOLVED || right === UNRESOLVED) return UNRESOLVED;
  try {
    return compare(member(comparison, 'operator') as PyValue, left, right);
  } catch (error) {
    // "comparison across types is undecidable, not an exception" (`tests/run_expressions.py`).
    if (error instanceof PyTypeError) return UNRESOLVED;
    throw error;
  }
}

/**
 * Statically known quantities: the literal ones, the external ones the assignment or a declared
 * default supplies, and the derived ones, evaluated to a fixpoint.
 *
 * What is left absent does not resolve: an unassigned external, or a derivation that is cyclic or
 * reads nothing. Derivations and declared defaults may read other quantities, in any order,
 * acyclically (§2.2, §4.6) — hence the fixpoint — and what never resolves is left out of the map
 * for the validator to refuse (V10).
 */
export function resolveQuantities(model: PyRecord, assignment?: PyRecord): Map<string, PyValue> {
  const given = assignment ?? {};
  const resolved = new Map<string, PyValue>();
  const pending: { name: string; expression: PyValue }[] = [];
  for (const [name, declaration] of items(demand(model, 'quantities'))) {
    const source = asRecord(demand(asRecord(declaration), 'source'));
    const kind = member(source, 'kind');
    if (kind === 'literal') {
      resolved.set(name, demand(source, 'value'));
    } else if (kind === 'external' && Object.hasOwn(given, name)) {
      resolved.set(name, given[name] as PyValue);
    } else if (kind === 'external' && hasKey(source, 'default')) {
      pending.push({ name, expression: member(source, 'default') as PyValue });
    } else if (kind === 'derived') {
      pending.push({ name, expression: demand(source, 'expression') });
    }
  }
  while (pending.length > 0) {
    let progress = false;
    for (const one of [...pending]) {
      const value = modelValue(one.expression, resolved);
      if (value === UNRESOLVED) continue;
      resolved.set(one.name, value);
      pending.splice(pending.indexOf(one), 1);
      progress = true;
    }
    if (!progress) break;
  }
  return resolved;
}

/** Static value of an instance argument: a literal, a resolved quantity, or a record of those. */
export function staticArgument(value: PyValue, quantities: Quantities, env?: Env): PyValue {
  if (isRecord(value) && hasKey(value, 'record')) {
    const out: Record<string, PyValue> = {};
    for (const [name, one] of items(member(value, 'record') as PyValue)) {
      put(out, name, staticArgument(one, quantities, env));
    }
    return out;
  }
  return modelValue(value, quantities, env);
}

/**
 * External quantity names the document expects an assignment to supply — the quantity's own
 * name; `withDefaults: false` leaves out those a declared default can stand for.
 */
export function externalNames(model: PyRecord, withDefaults = true): Set<string> {
  const out = new Set<string>();
  for (const [name, declaration] of items(demand(model, 'quantities'))) {
    const source = asRecord(demand(asRecord(declaration), 'source'));
    if (member(source, 'kind') !== 'external') continue;
    if (withDefaults || !hasKey(source, 'default')) out.add(name);
  }
  return out;
}

/** External names the assignment leaves unset, in order. */
export function missingAssignment(model: PyRecord, assignment?: PyRecord): string[] {
  const given = assignment ?? {};
  const missing = [...externalNames(model, false)].filter((name) => !Object.hasOwn(given, name));
  return missing.sort(comparePythonStrings);
}

/**
 * The (name, values) pairs a set of index declarations unrolls to, in lexicographic order of the
 * index names (§5.2).
 *
 * The ranges are Python's `range(start, stop, step)`, so every bound is an integer — a real bound
 * is a `TypeError` there and here, as a step of zero is a `ValueError` — and a bound that does
 * not resolve is {@link Unassigned}, which is the template's missing assignment reaching the
 * expansion.
 *
 * The values are unrolled at once, where Python's `range` is lazy; every caller of the tools'
 * `index_grid` walks the whole product anyway (`itertools.product`), so the work is the same and
 * only the memory differs. A document declaring a range of a billion positions is as slow there
 * as here, which is what the worker's cancellation is for (plan §5.3), not a refusal this side
 * would invent.
 */
export function indexGrid(
  indices: PyValue,
  quantities: Quantities,
): { names: string[]; ranges: bigint[][] } {
  const declared = asRecord(indices);
  const names = Object.keys(declared).sort(comparePythonStrings);
  const ranges: bigint[][] = [];
  for (const name of names) {
    const range = asRecord(member(declared, name) as PyValue);
    const bounds: bigint[] = [];
    for (const edge of ['start', 'stop', 'step'] as const) {
      const value = modelValue(demand(range, edge), quantities);
      if (value === UNRESOLVED) {
        throw new Unassigned(`index '${name}': ${edge} does not resolve to a value`);
      }
      bounds.push(asRangeBound(value));
    }
    ranges.push(unroll(bounds[0] as bigint, bounds[1] as bigint, bounds[2] as bigint));
  }
  return { names, ranges };
}

/** `range()` takes integers, and a `bool` is one; everything else is Python's own refusal. */
function asRangeBound(value: PyValue): bigint {
  if (typeof value === 'bigint') return value;
  if (typeof value === 'boolean') return value ? 1n : 0n;
  throw new PyTypeError(
    `'${pythonTypeName(value)}' object cannot be interpreted as an integer`,
  );
}

/** The values of `range(start, stop, step)`. */
function unroll(start: bigint, stop: bigint, step: bigint): bigint[] {
  if (step === 0n) throw new PyValueError('range() arg 3 must not be zero');
  const values: bigint[] = [];
  if (step > 0n) for (let at = start; at < stop; at += step) values.push(at);
  else for (let at = start; at > stop; at += step) values.push(at);
  return values;
}

/** Names of the quantities an expression reads. */
export function quantityReferences(expression: PyValue): Set<string> {
  const out = new Set<string>();
  if (!isRecord(expression)) return out;
  if (hasKey(expression, 'quantity')) out.add(asName(member(expression, 'quantity')));
  const args = member(expression, 'args');
  if (args !== undefined) {
    for (const one of pyIterate(args)) {
      for (const name of quantityReferences(one)) out.add(name);
    }
  }
  for (const branch of ['then', 'else'] as const) {
    if (!hasKey(expression, branch)) continue;
    for (const name of quantityReferences(member(expression, branch) as PyValue)) out.add(name);
  }
  if (hasKey(expression, 'if')) {
    for (const name of conditionReferences(member(expression, 'if') as PyValue)) out.add(name);
  }
  return out;
}

/** Names of the quantities a condition reads, through `not`, `all`, `any` and both operands. */
export function conditionReferences(condition: PyValue): Set<string> {
  const c = asRecord(condition);
  const out = new Set<string>();
  if (hasKey(c, 'not')) {
    for (const name of conditionReferences(member(c, 'not') as PyValue)) out.add(name);
  }
  for (const part of concat(member(c, 'all'), member(c, 'any'))) {
    for (const name of conditionReferences(part)) out.add(name);
  }
  if (hasKey(c, 'compare')) {
    const comparison = asRecord(member(c, 'compare') as PyValue);
    for (const name of quantityReferences(demand(comparison, 'left'))) out.add(name);
    for (const name of quantityReferences(demand(comparison, 'right'))) out.add(name);
  }
  return out;
}

/** A quantity name, as a string; anything else is a value no reference can be made of. */
function asName(value: PyValue | undefined): string {
  if (typeof value !== 'string') {
    throw new PyTypeError(
      `'${pythonTypeName(value === undefined ? null : value)}' is not a name`,
    );
  }
  return value;
}

/** `c.get('all', []) + c.get('any', [])`, list concatenation and its refusal. */
function concat(left: PyValue | undefined, right: PyValue | undefined): PyValue[] {
  const list = (value: PyValue | undefined): PyValue[] => {
    if (value === undefined) return [];
    if (!Array.isArray(value)) {
      throw new PyTypeError(`can only concatenate list (not "${pythonTypeName(value)}") to list`);
    }
    return [...(value as readonly PyValue[])];
  };
  return [...list(left), ...list(right)];
}
