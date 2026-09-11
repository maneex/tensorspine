/**
 * Expressions and conditions: the port of `tools/expr.py` (plan §5.3, `evaluate`).
 *
 * "There are two evaluators here, and they do not read the same thing": the **primitive** side
 * evaluates a declaration's expressions against an instance's resolved arguments — defaults,
 * `present_when` guards, shape extents, state rules, invariants — and the **model** side
 * evaluates a document's expressions against its quantities and the composition indices in
 * scope — derivations, index ranges, guards, argument values.
 *
 * Everything the two need to agree with the tools on is here: the value model that keeps
 * Python's `int` and `float` apart (`value.ts`), the operator and comparison tables the audit
 * holds to the schemas' enumerations (`arithmetic.ts`), and the two evaluators with the
 * reference walkers beside them.
 */
export {
  asRecord,
  conditionNode,
  demandKey,
  expressionNode,
  hasKey,
  isInteger,
  isRecord,
  isResolved,
  items,
  member,
  pyIterate,
  pythonTypeName,
  toJsonValue,
  toPython,
  truthy,
  UNRESOLVED,
  type PyRecord,
  type PyValue,
  type Resolved,
  type Unresolved,
} from './value.js';
export {
  PyError,
  PyIndexError,
  PyKeyError,
  PyOverflowError,
  PyTypeError,
  PyValueError,
  PyZeroDivisionError,
} from './errors.js';
export {
  apply,
  compare,
  COMPARISONS,
  OPERATORS,
  pyAbsolute,
  pyAdd,
  pyDivide,
  pyEqual,
  pyFloorDivide,
  pyModulo,
  pyMultiply,
  pyNegate,
  pyOrder,
  pySubtract,
} from './arithmetic.js';
export {
  argumentAt,
  argumentPresent,
  argumentReferences,
  expressionReferences,
  primitiveCondition,
  primitiveValue,
} from './primitive.js';
export {
  conditionReferences,
  externalNames,
  indexGrid,
  missingAssignment,
  modelCondition,
  modelValue,
  QUANTITIES,
  quantityReadings,
  quantityReferences,
  resolveQuantities,
  staticArgument,
  Unassigned,
  type Env,
  type Quantities,
  type QuantityReading,
} from './model.js';
