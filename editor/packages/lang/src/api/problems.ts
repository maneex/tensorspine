/**
 * The five refusal shapes of the core, read as one {@link Problem}.
 *
 * Each stage of the core carries the refusal its tool carries — a `StructuralProblem` is
 * `schema.format_error`'s, a `LibraryProblem` is what a `PrimitiveLibraryError` raises, a
 * `SemanticProblem` is `analyse`'s `fail`, a `LintFinding` is `lint.py`'s row, a
 * `CheckpointProblem` is `artifact.check`'s — and each keeps the tools' wording, which is the
 * parity contract (D2, §7 F1). The Problems panel reads one shape: plan Appendix C's, whose
 * `source` says which stage a row came from and whose `message` is the line that stage prints.
 *
 * Nothing is decided here. Every field is copied from the row the core already built, and the
 * only composition is the one each stage's own printer does — `formatProblem`,
 * `formatSemanticProblem`, `formatCheckpointProblem`, and `<scope>: <message>` for a lint
 * finding, which is what `lint.run` prints after its `  W  `.
 */
import type { CheckpointProblem } from '../artifact/check.js';
import type { LibraryProblem } from '../library/problems.js';
import type { LintFinding } from '../lint/lint.js';
import { writtenPlace, type Hoisting } from '../model/hoisting.js';
import { formatProblem, type StructuralProblem } from '../schema/registry.js';
import { formatSemanticProblem, type SemanticProblem } from '../validate/problems.js';
import { formatCheckpointProblem } from '../artifact/check.js';

import type { LangFailure } from './codec.js';
import type { Problem } from './types.js';

/** The grammar stage's row: the line `validate.structural` returns, and the place it is about. */
export function schemaProblem(problem: StructuralProblem, file?: string): Problem {
  return {
    code: problem.code,
    message: formatProblem(problem),
    path: problem.path,
    ...(problem.keyword === null ? {} : { rule: problem.keyword }),
    ...(file === undefined ? {} : { file }),
    severity: 'error',
    source: 'schema',
  };
}

/**
 * The loader's row: the head the tools raise, with the lines under it beside rather than inside.
 *
 * `formatLibraryProblem` is what reassembles the raised text (the head, then each line indented
 * by four spaces); the panel shows the head as the row and the lines under it, each with its own
 * pointer into the unit.
 */
export function libraryRow(problem: LibraryProblem): Problem {
  return {
    code: problem.code ?? '',
    message: problem.message,
    // The head names the *file*; the places the refusal knows are on its lines.
    path: '',
    rule: problem.kind,
    file: problem.file,
    severity: 'error',
    source: 'library',
    ...(problem.afterRefusal ? { afterRefusal: true } : {}),
    ...(problem.detail.length === 0
      ? {}
      : { detail: problem.detail.map((one) => ({ message: one.message, path: one.path })) }),
  };
}

/**
 * The semantic stage's row: `[V8] …`, and the pointer beside it — in the document as written.
 *
 * The stage reads the document `model.normalise` answers, so a refusal about a
 * composition-scoped binding names the hoisted rule (`/bindings/values/decoder.attn.norm_in`) and
 * not the place the author typed. Given the hoist's own record the place is written back here, in
 * the one function that turns a `SemanticProblem` into the row every panel reads, and the core's
 * own pointer is kept beside it. Without the record — a caller that did not ask for one — the
 * pointer is answered as the stage named it, which is what it has always been.
 */
export function semanticRow(
  problem: SemanticProblem,
  options: {
    readonly file?: string;
    readonly afterRefusal?: boolean;
    readonly hoisting?: Hoisting;
  } = {},
): Problem {
  const written = options.hoisting === undefined ? null : writtenPlace(options.hoisting, problem.path);
  return {
    code: problem.code,
    message: formatSemanticProblem(problem),
    path: written === null ? problem.path : written.path,
    ...(written === null ? {} : { normalisedPath: problem.path }),
    ...(written === null || written.exact ? {} : { approximate: true }),
    ...(options.file === undefined ? {} : { file: options.file }),
    severity: 'error',
    source: 'semantic',
    ...(options.afterRefusal === true ? { afterRefusal: true } : {}),
  };
}

/** A lint finding: the scope and the message `lint.run` prints after its `  W  `. */
export function lintRow(finding: LintFinding): Problem {
  return {
    code: '',
    message: `${finding.scope}: ${finding.message}`,
    path: '',
    rule: finding.rule,
    ...(finding.file === undefined ? {} : { file: finding.file }),
    severity: 'warning',
    source: 'lint',
  };
}

/** A V17 row against a checkpoint: `[V17] …`, with the identity and the shard it names. */
export function checkpointRow(problem: CheckpointProblem): Problem {
  return {
    code: problem.code,
    message: formatCheckpointProblem(problem),
    path: '',
    ...(problem.identity === undefined ? {} : { node: problem.identity }),
    rule: problem.kind,
    ...(problem.file === undefined ? {} : { file: problem.file }),
    severity: problem.severity,
    source: 'checkpoint',
  };
}

/**
 * A refusal the derivation raised, as a row.
 *
 * `derive` refuses three ways and none of them is a list of problems: `not valid, no products:
 * <the first line>` when the document does not validate, `_sound`'s R11 line and `_consistent`'s
 * four when a figure or an expansion disagrees, and `DerivedSchemaError` when the emitted
 * document is off the derived schema. The tools *print* them; the core raises them, because "the
 * core's writing is its returning: a document it cannot vouch for is not returned" (feature
 * 1.8e). This is how the panel of §4.18 shows one without inventing a wording.
 */
export function derivationRow(failure: LangFailure | Error, file?: string): Problem {
  const raised = 'raised' in failure ? failure.raised : failure.name;
  return {
    code: '',
    message: failure.message,
    path: '',
    rule: raised,
    ...(file === undefined ? {} : { file }),
    severity: 'error',
    source: 'derivation',
  };
}
