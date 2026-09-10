/**
 * Validation: the port of `tools/validate.py`, stage by stage (plan §5.3, `validate`).
 *
 * "Grammar first, then meaning." The grammar is `schema/` (feature 1.1, `structural`) and the
 * library is `library/` (feature 1.3); what is left is the semantic stage — V1–V20 over a
 * normalised document — and it is built here in the order the tools compute it: the quantities
 * first, because every later rule is written over the map they resolve to.
 *
 * What each problem carries is stated once, in `problems.ts`: the tools' line, and the place in
 * the document it is about.
 */
export {
  formatSemanticProblem,
  formatSemanticProblems,
  semanticProblem,
  withMessage,
  type RuleCode,
  type SemanticProblem,
} from './problems.js';
export { checkDomain, checkType, type RecordCheck } from './conformance.js';
export {
  assignmentNeeded,
  checkAssignment,
  checkQuantities,
  variableQuantities,
  type AssignmentNeeded,
} from './quantities.js';
