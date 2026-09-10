/**
 * The primitive side of `tools/expr.py`: expressions and conditions over *resolved arguments*.
 *
 * "Declared defaults, `present_when` guards and shape extents are written in that language" — so
 * are a state rule's `when`, `written_when` and `carried_across`, the modulators, the invariants
 * and the cost entries (§4.1, §4.3). Every reference in one is an argument path: `heads`,
 * `rope.scaling.kind`, dotted through the fields of a record argument.
 *
 * Three answers this side gives are not the model side's, and all three are the tools' own:
 *
 * - **An absent argument is `None`, not `UNRESOLVED`.** `primitiveValue` answers `null` for a
 *   path that resolves to nothing, and an operator whose operand is `null` answers `UNRESOLVED`
 *   — the same refusal, reached from a different fact. The two are told apart because
 *   `resolve_arguments` writes `UNRESOLVED` for an argument it *refused* and leaves an argument
 *   that was never given out of the map entirely.
 * - **An undecidable comparison is false, not an exception.** "The guard it protects simply does
 *   not fire." That is right for a guard and wrong for a derived fact, which is why
 *   {@link argumentReferences} exists: a fact evaluated from a condition refuses when any path
 *   the condition reads is unresolved, instead of taking the false.
 * - **`all` and `any` stop at the first part that decides them**, where `model_condition`
 *   evaluates every part before looking for an undecidable one; and a comparison of two values
 *   Python cannot order raises here, where the model side catches it. Both follow from the code
 *   as it is written — a generator expression inside `all`, no `try` around the comparison — and
 *   are reproduced rather than harmonised.
 *
 * The loader is what makes the false safe: an argument that may be absent is compared or
 * computed with only under a `present` test of it (§4.3), and a primitive that does otherwise is
 * refused when the primitive library is loaded. So a decidable guard is never false by accident.
 */
import { PyTypeError } from './errors.js';
import { apply, compare } from './arithmetic.js';
import {
  conditionNode,
  demandKey as demand,
  expressionNode,
  hasKey,
  isRecord,
  member,
  pyIterate,
  pythonTypeName,
  truthy,
  UNRESOLVED,
  type PyRecord,
  type PyValue,
} from './value.js';

/** The path a node names, as a string; anything else is the refusal `split` raises. */
function asPath(value: PyValue | undefined): string {
  if (typeof value !== 'string') {
    throw new PyTypeError(
      `'${pythonTypeName(value === undefined ? null : value)}' object has no attribute 'split'`,
    );
  }
  return value;
}

/**
 * The value an argument path resolves to, `null` when it resolves to nothing.
 *
 * The walk stops at the first step that is not a record or that the record has no member for —
 * which is also `validate._resolve_path`, the same walk, used to show an invariant's operands.
 */
export function argumentAt(path: readonly string[], args: PyValue): PyValue {
  let current = args;
  for (const part of path) {
    if (!isRecord(current) || !hasKey(current, part)) return null;
    current = member(current, part) as PyValue;
  }
  return current;
}

/** Whether an argument path resolves to something at all — the `present` test of §4.3. */
export function argumentPresent(path: readonly string[], args: PyValue): boolean {
  let current = args;
  for (const part of path) {
    if (!isRecord(current) || !hasKey(current, part)) return false;
    current = member(current, part) as PyValue;
  }
  return true;
}

/** Value of a primitive expression against resolved arguments. */
export function primitiveValue(expression: PyValue, args: PyRecord): PyValue {
  const node = expressionNode(expression);
  if (node === undefined) return UNRESOLVED;
  if (hasKey(node, 'literal')) return member(node, 'literal') as PyValue;
  if (hasKey(node, 'argument')) {
    return argumentAt(asPath(member(node, 'argument')).split('.'), args);
  }
  if (hasKey(node, 'op')) {
    const operands = pyIterate(demand(node, 'args')).map((one) => primitiveValue(one, args));
    if (operands.some((one) => one === null || one === UNRESOLVED)) return UNRESOLVED;
    return apply(member(node, 'op') as PyValue, operands);
  }
  if (hasKey(node, 'if')) {
    const branch = truthy(primitiveCondition(member(node, 'if') as PyValue, args))
      ? 'then'
      : 'else';
    return primitiveValue(demand(node, branch), args);
  }
  return UNRESOLVED;
}

/**
 * Truth of a primitive condition.
 *
 * The answer is Python's own: the `boolean` form answers the member as it stands, `not`, `all`
 * and `any` answer real booleans over the truth of their parts, and a comparison whose operands
 * do not both resolve answers false. A comparison of two values Python cannot order raises —
 * this side does not catch what the model side catches, which is stated as it stands rather than
 * corrected here.
 */
export function primitiveCondition(condition: PyValue, args: PyRecord): PyValue {
  const node = conditionNode(condition);
  if (hasKey(node, 'boolean')) return member(node, 'boolean') as PyValue;
  if (hasKey(node, 'not')) return !truthy(primitiveCondition(member(node, 'not') as PyValue, args));
  if (hasKey(node, 'all')) {
    return pyIterate(member(node, 'all') as PyValue).every((one) =>
      truthy(primitiveCondition(one, args)),
    );
  }
  if (hasKey(node, 'any')) {
    return pyIterate(member(node, 'any') as PyValue).some((one) =>
      truthy(primitiveCondition(one, args)),
    );
  }
  if (hasKey(node, 'present')) {
    return argumentPresent(asPath(member(node, 'present')).split('.'), args);
  }
  const comparison = conditionNode(demand(node, 'compare'));
  const left = primitiveValue(demand(comparison, 'left'), args);
  const right = primitiveValue(demand(comparison, 'right'), args);
  if (left === UNRESOLVED || right === UNRESOLVED || left === null || right === null) return false;
  return compare(member(comparison, 'operator') as PyValue, left, right);
}

/**
 * The argument paths a primitive condition reads: those of every `compare` — through the
 * expressions compared, the tests of their conditionals included — and of every `present` test,
 * through `not`, `all` and `any`.
 *
 * What `conditionReferences` is for the model side. A derived fact evaluated from a condition
 * needs it: `primitiveCondition` answers false to what it cannot decide, which is right for a
 * guard and wrong for a fact, so the emitter refuses when any path listed here is unresolved
 * (D1's `across_positions`).
 *
 * The set is in the order the walk meets the paths; the tools' callers sort it (`sorted(refs)`),
 * since a Python set has no order to promise.
 */
export function argumentReferences(condition: PyValue): Set<string> {
  const node = conditionNode(condition);
  const out = new Set<string>();
  if (hasKey(node, 'not')) {
    for (const path of argumentReferences(member(node, 'not') as PyValue)) out.add(path);
  }
  for (const part of concat(member(node, 'all'), member(node, 'any'))) {
    for (const path of argumentReferences(part)) out.add(path);
  }
  if (hasKey(node, 'present')) out.add(asPath(member(node, 'present')));
  if (hasKey(node, 'compare')) {
    const comparison = conditionNode(member(node, 'compare') as PyValue);
    for (const path of expressionReferences(demand(comparison, 'left'))) out.add(path);
    for (const path of expressionReferences(demand(comparison, 'right'))) out.add(path);
  }
  return out;
}

/** `c.get('all', []) + c.get('any', [])`: two lists, and a refusal for anything else. */
function concat(left: PyValue | undefined, right: PyValue | undefined): PyValue[] {
  const list = (value: PyValue | undefined): PyValue[] => {
    if (value === undefined) return [];
    if (!Array.isArray(value)) {
      throw new PyTypeError(
        `can only concatenate list (not "${pythonTypeName(value)}") to list`,
      );
    }
    return [...(value as readonly PyValue[])];
  };
  return [...list(left), ...list(right)];
}

/** The argument paths an expression reads, `expr.py`'s `_argument_refs`. */
export function expressionReferences(expression: PyValue): Set<string> {
  const out = new Set<string>();
  if (!isRecord(expression)) return out;
  if (hasKey(expression, 'argument')) out.add(asPath(member(expression, 'argument')));
  const args = member(expression, 'args');
  if (args !== undefined) {
    for (const one of pyIterate(args)) {
      for (const path of expressionReferences(one)) out.add(path);
    }
  }
  for (const branch of ['then', 'else'] as const) {
    if (!hasKey(expression, branch)) continue;
    for (const path of expressionReferences(member(expression, branch) as PyValue)) out.add(path);
  }
  if (hasKey(expression, 'if')) {
    for (const path of argumentReferences(member(expression, 'if') as PyValue)) out.add(path);
  }
  return out;
}
