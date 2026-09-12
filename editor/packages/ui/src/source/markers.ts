/**
 * A problem's **range** in the source text — plan §4.10, §4.17, artboard S16.
 *
 * > structural errors at their range … "Show in JSON" from any selection reveals its range
 *
 * Every problem the editor holds carries a JSON pointer (`Problem.path`), which is what the
 * Problems panel navigates by and what the store keys a place by. The core answers where a
 * pointer stands in a text (`spansOf`, feature 2.17's addition to the parser), and this turns the
 * two into what a text editor draws: a squiggle under a stretch of characters, with the row's own
 * message on it.
 *
 * **Nothing here decides whether a document is valid.** The rows are the core's — Ajv's for the
 * grammar, `analyse`'s for the meaning, the loader's for a unit — and what this module adds is
 * the place, which the core answered too. That is why the source view attaches the schema to
 * Monaco for *completion and hover* and switches Monaco's own JSON validation **off**: a third
 * reading of the schemas beside `jsonschema` and Ajv would have its own wording, outside the
 * parity contract (D2, D4) and outside the audit of §1.
 *
 * **Which characters are marked.** A member's name where there is one and the value spans more
 * than a line — a squiggle under thirty lines of object says nothing a reader can use — and the
 * value itself otherwise, so `"eps": 1e-05` is marked whole and `"final_n"` is marked at its
 * name. A place with no name that runs past its first line (the root) is marked at its first
 * line. Revealing is the other rule and the wider one: S16 highlights `"final_n"` through its
 * closing brace, so a reveal is the name through the end of the value.
 */
import type { Problem, ProblemSeverity } from '@tensorspine/lang/api';
import { pointerSegment, type JsonSpan } from '@tensorspine/lang';

/** A stretch of the source text, as offsets into it. */
export interface TextRange {
  readonly start: number;
  readonly end: number;
}

/** One marker: where it is, what it says, and which stage said it. */
export interface SourceMarker extends TextRange {
  readonly severity: ProblemSeverity;
  /** The row's own line, word for word, as the Problems panel prints it. */
  readonly message: string;
  /** The stage the row came from — §4.17's `.src` column, never the severity. */
  readonly source: string;
  /** The rule the line names, where it names one. */
  readonly code: string;
}

/** The spans of a text, as the core answers them. */
export type SourceSpans = ReadonlyMap<string, JsonSpan>;

/**
 * Where a pointer stands, as a reveal shows it: the member's name through the end of its value.
 *
 * `null` where the text has no such place — a problem about a hoisted binding (`model.normalise`
 * writes places no file holds, feature 2.8), a row about another file, or a pointer into a
 * document the pane has moved past.
 */
export function revealRange(spans: SourceSpans, pointer: string): TextRange | null {
  const span = spans.get(pointer);
  if (span === undefined) return null;
  return { start: span.nameStart ?? span.start, end: span.end };
}

/** Where a pointer is marked: the name where the value is long, the value otherwise. */
export function markRange(spans: SourceSpans, pointer: string, text: string): TextRange | null {
  const span = spans.get(pointer);
  if (span === undefined) return null;
  const from = span.nameStart ?? span.start;
  if (span.nameEnd !== undefined && text.lastIndexOf('\n', span.end - 1) >= span.start) {
    // A value that runs past its own first line: the name is what a reader can read.
    return { start: from, end: span.nameEnd };
  }
  const line = text.indexOf('\n', from);
  return { start: from, end: line >= 0 && line < span.end ? line : span.end };
}

/**
 * The markers for a document's rows, over the text the pane holds.
 *
 * A row whose pointer names members by name (the grammar's `additionalProperties`, which reports
 * the *object* and names the extras in its prose — the core carries the names beside the wording
 * rather than making anyone read them back out of it) becomes one marker per member. A row whose
 * pointer the text has not is left out: a marker is a place, and a row with no place is the
 * panel's to show.
 */
export function markersFor(
  problems: readonly Problem[],
  spans: SourceSpans,
  text: string,
): SourceMarker[] {
  const markers: SourceMarker[] = [];
  for (const problem of problems) {
    const places =
      problem.members === undefined || problem.members.length === 0
        ? [problem.path]
        : problem.members.map((name) => `${problem.path}/${pointerSegment(name)}`);
    for (const place of places) {
      const range = markRange(spans, place, text) ?? (place === problem.path ? null : markRange(spans, problem.path, text));
      if (range === null) continue;
      markers.push({
        ...range,
        severity: problem.severity,
        message: problem.message,
        source: problem.source,
        code: problem.code,
      });
    }
  }
  return markers;
}
