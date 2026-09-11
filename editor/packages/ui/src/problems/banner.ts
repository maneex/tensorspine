/**
 * The banner over the rows — §4.17 ("a banner when a stage refused and the rows below it are
 * 'after a refusal'"), plan §3, artboard S10.
 *
 * > The tools' `validate.analyse` returns early on some failures; a faithful port would too, and
 * > the Problems panel wants everything. […] The port keeps the tools' staging (grammar, then
 * > library, then meaning) because parity requires it, and adds *continuation* where the tools
 * > stop: after the stage the tools would abort in, the core keeps checking what remains decidable
 * > and marks those problems "after a refusal" — extra rows, never fewer; the parity job compares
 * > the tools' rows only.
 *
 * A reader who does not know that would read the rows below a library refusal as if the tools had
 * printed them, and a reader who does not know why the semantic rows are *missing* would read an
 * off-grammar document as a valid one. So the banner says which of the four things happened, and
 * every one of them is read off the verdict the core answered: which stages ran, whether an
 * assignment was wanted, and whether the core marked its own rows as continuation. Nothing is
 * inferred from a message.
 *
 * **S10's own banner says "Semantic checking stopped at V1".** It does not: `[V1] primitive absent
 * from primitive library` is a refusal of the semantic stage, which collects every row it can and
 * stops at none of them. What does stop a reading is the grammar (D5's "meaning assumes grammar"),
 * a library that would not gather, and an assignment that is missing or refused — the four below.
 * The board's wording is its own; the state is the core's.
 */
import type { Verdict } from '@tensorspine/lang/api';
import { STAGE } from '@tensorspine/lang/api';

/** Which of the four the panel is showing, if any. */
export type BannerKind =
  /** The document is off the grammar, so nothing after the schema stage ran (D5). */
  | 'off-grammar'
  /** The library would not gather; every row after it is the core's continuation (plan §3). */
  | 'after-refusal'
  /** A template with no assignment: skipped, not failed (§4.6, I7). */
  | 'needs-assignment'
  /** An assignment the document's own declarations refuse, so nothing was analysed. */
  | 'assignment-refused';

/** What the banner says. */
export interface ProblemsBanner {
  readonly kind: BannerKind;
  /** The class S10 gives it: a refusal, a warning, or a statement. */
  readonly tone: 'stop' | 'warn' | 'info';
  /** The glyph beside it (S10's `.bi`). */
  readonly mark: string;
  /** The sentence in bold. */
  readonly head: string;
  /** The rest. */
  readonly body: string;
}

/** What each kind says — the editor's own words, since no stage of the core writes one. */
const SAID: Readonly<Record<BannerKind, Omit<ProblemsBanner, 'kind'>>> = {
  'off-grammar': {
    tone: 'stop',
    mark: '■',
    head: 'The document is off the grammar.',
    body:
      'Meaning assumes grammar, so nothing after the schema stage ran. The rows below are Ajv’s, ' +
      'at the places they are about; the semantic checks resume as soon as the document is back ' +
      'on the schema.',
  },
  'after-refusal': {
    tone: 'stop',
    mark: '■',
    head: 'The primitive library refused.',
    body:
      'The tools stop here. The rows below the refusal are what the core could still decide ' +
      'after it — extra rows, never fewer — and the parity job compares only what the tools print.',
  },
  'needs-assignment': {
    tone: 'info',
    mark: '▲',
    head: 'This template has no assignment.',
    body:
      'A template denotes one graph per admissible assignment, so it is skipped rather than ' +
      'failed: give its external quantities values and the semantic stage runs on the graph they ' +
      'name.',
  },
  'assignment-refused': {
    tone: 'warn',
    mark: '▲',
    head: 'The assignment is refused.',
    body:
      'A value present is not a value admissible: the rows below are the document’s own ' +
      'declarations refusing what was assigned, and nothing was analysed under it.',
  },
};

/**
 * The banner a verdict calls for, or `null`.
 *
 * The order is the order the stages run in: a document off the grammar has not reached the
 * library, and one whose assignment is missing has not reached the semantic stage, so the first
 * thing that stopped is the thing to say.
 */
export function bannerOf(verdict: Verdict | null): ProblemsBanner | null {
  if (verdict === null) return null;
  const ran = (stage: (typeof STAGE)[keyof typeof STAGE]): boolean => verdict.stagesRun.includes(stage);
  if (!ran(STAGE.library)) {
    return verdict.problems.length === 0 ? null : named('off-grammar');
  }
  if (verdict.needsAssignment !== undefined) return named('needs-assignment');
  if (!ran(STAGE.semantic)) {
    return verdict.problems.length === 0 ? null : named('assignment-refused');
  }
  if (verdict.problems.some((one) => one.afterRefusal === true)) return named('after-refusal');
  return null;
}

function named(kind: BannerKind): ProblemsBanner {
  return { kind, ...SAID[kind] };
}

/** Every banner the panel can show — what a suite walks, and what a translator sees. */
export const BANNERS: Readonly<Record<BannerKind, Omit<ProblemsBanner, 'kind'>>> = SAID;
