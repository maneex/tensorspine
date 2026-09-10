/**
 * What the semantic stage refuses with.
 *
 * `tools/validate.py` collects strings: `analyse`'s `fail(code, message)` appends
 * `f"[{code}] {message}"` to one list, and every caller — the rejection runner, the status page,
 * `--validate`'s printer — reads that text. Those strings are the parity contract (the plan's D2,
 * and §7 F1: `tests/rejections/models.json` matches substrings of them), so the core prints them
 * character for character.
 *
 * The Problems panel wants more than the text: plan §3 asks the core to emit
 * `{code, message, path, node?, rule?, file?}` *natively* — "the message with the tools' wording
 * for parity, the pointer beside it" — so a row has somewhere to click. Hence the shape here: the
 * tools' two fields, and the place in the document beside them.
 *
 * {@link formatSemanticProblem} is the bridge: it writes the line `fail` would have appended, so a
 * parity suite compares lines and the interface reads fields. Feature 1.3's
 * `library/problems.ts` is the same arrangement one stage earlier, for the loader's refusals.
 */
import { pointerOf, type PathSegment } from '../schema/types.js';

/**
 * A rule of §6 of the specification, as its identifier.
 *
 * The identifiers are the specification's — `V1` … `V20` — not a schema's vocabulary, so naming
 * one here is not the hard-coding §1 forbids; and they are not enumerated, because the set is the
 * specification's to grow and a table of twenty names with nothing attached to them would be a
 * list to keep in step rather than a rule.
 */
export type RuleCode = `V${number}`;

/** One refusal of the semantic stage: the tools' line, and the place in the document it is about. */
export interface SemanticProblem {
  /** The rule the tools name in the line's `[…]` prefix. */
  readonly code: RuleCode;
  /** The tools' words, exactly, without that prefix. */
  readonly message: string;
  /** Where in the document, as a JSON pointer (RFC 6901); `''` when the line names no place. */
  readonly path: string;
  /** The same place as its segments, for a store that walks the tree. */
  readonly segments: readonly PathSegment[];
}

/** One problem, from the tools' `(code, message)` pair and the place the message is about. */
export function semanticProblem(
  code: RuleCode,
  message: string,
  segments: readonly PathSegment[] = [],
): SemanticProblem {
  return { code, message, path: pointerOf(segments), segments };
}

/** The same problem with its message rewritten: `check_quantities`' final `str.replace` needs it. */
export function withMessage(problem: SemanticProblem, message: string): SemanticProblem {
  return { ...problem, message };
}

/** `f"[{code}] {message}"`: the line `analyse`'s `fail` appends, and what the suites match on. */
export function formatSemanticProblem(problem: SemanticProblem): string {
  return `[${problem.code}] ${problem.message}`;
}

/** {@link formatSemanticProblem} over each, in order: the `errors` list the tools return. */
export function formatSemanticProblems(problems: readonly SemanticProblem[]): string[] {
  return problems.map((problem) => formatSemanticProblem(problem));
}
