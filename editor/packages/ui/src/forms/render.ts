/**
 * A form as text: one line per row, one per note.
 *
 * The rows are data and a renderer draws them; this is the *other* reading, the one a log line
 * and a snapshot want. Catching rule §1 (c) is what it is for — "the forms are exercised on every
 * `$def` of every schema in a snapshot test, so a schema change shows up as a diff, not as a
 * crash" — and a diff is only a diff if a row reads as one line.
 *
 * The shape of a line is the shape of the sheet: the indent is the row's depth, then the widget
 * and the label, then everything the row carries about its place — what it requires, the branch a
 * conditional requirement hangs on, the options of a select, the modes of a chooser with the
 * widget each opens and a star on the one in force, the bounds, the name definition a map's keys
 * are, the figure format, the problem, and the reason the row renders generically.
 */
import type { VocabularyValue } from '@tensorspine/lang';

import type { Form, FormBounds, FormMode, FormRow } from './types.js';

/** One line per row, for the log and for a snapshot: the shape of the form as text. */
export function formLines(form: Form): readonly string[] {
  return form.rows.map((row) => lineOf(row));
}

/** One line per thing the walker had no reading for (plan §1 — "never fails, and is listed"). */
export function noteLines(form: Form): readonly string[] {
  return form.notes.map(
    (note) => `form: ${note.path === '' ? '<root>' : note.path}: ${note.message}`,
  );
}

/** One row as a line: the indent is the depth, then the widget, the label and what it carries. */
export function lineOf(row: FormRow): string {
  const marks: string[] = [];
  if (row.required) marks.push('required');
  for (const condition of row.conditions ?? []) {
    marks.push(
      `${condition.requires ? 'required-when' : 'declared-when'} ${short(condition.test)}` +
        (condition.holds === null ? '' : condition.holds ? ' (holds)' : ' (does not hold)'),
    );
  }
  if (row.constant !== undefined) marks.push(`= ${printed(row.constant)}`);
  if (row.options !== undefined) marks.push(`[${row.options.map((one) => one.label).join('|')}]`);
  if (row.modes !== undefined) {
    marks.push(`{${row.modes.map((one) => modeMark(one, row.mode)).join(' | ')}}`);
  }
  if (row.written !== undefined) marks.push(`is ${printed(row.written)}`);
  if (row.referent !== undefined) marks.push(`names a ${row.referent}`);
  if (row.picker !== undefined) marks.push(`from ${row.picker}`);
  if (row.keys !== undefined) marks.push(`keys ${short(row.keys.anchor)}`);
  const bounds = boundLine(row.bounds);
  if (bounds !== '') marks.push(bounds);
  if (row.format !== undefined) marks.push(`shown as ${row.format}`);
  if (row.error !== undefined) marks.push(`! ${row.error}`);
  if (row.generic !== undefined) marks.push(`generic: ${row.generic}`);
  const head = `${'  '.repeat(row.depth)}${row.widget} ${row.label}`;
  return marks.length === 0 ? head : `${head}  ${marks.join('  ')}`;
}

/** One mode inside a chooser's mark: its tag, its widget, and a star on the one in force. */
function modeMark(mode: FormMode, chosen: string | undefined): string {
  const tag = mode.tag === '' ? '?' : mode.tag;
  const referent = mode.referent === undefined ? '' : `\u2192${mode.referent}`;
  return `${mode.tag === chosen ? '*' : ''}${tag}:${mode.widget}${referent}`;
}

/** The bounds of a row as one mark, or `''` when it has none. */
function boundLine(bounds: FormBounds | undefined): string {
  if (bounds === undefined) return '';
  const parts: string[] = [];
  if (bounds.minimum !== undefined) {
    parts.push(`${bounds.minimumExcluded === true ? '>' : '>='}${String(bounds.minimum)}`);
  }
  if (bounds.maximum !== undefined) {
    parts.push(`${bounds.maximumExcluded === true ? '<' : '<='}${String(bounds.maximum)}`);
  }
  if (bounds.minLength !== undefined) parts.push(`length>=${String(bounds.minLength)}`);
  if (bounds.maxLength !== undefined) parts.push(`length<=${String(bounds.maxLength)}`);
  if (bounds.pattern !== undefined) parts.push(`matches ${bounds.pattern}`);
  if (bounds.minItems !== undefined) parts.push(`items>=${String(bounds.minItems)}`);
  if (bounds.maxItems !== undefined) parts.push(`items<=${String(bounds.maxItems)}`);
  if (bounds.uniqueItems === true) parts.push('unique');
  if (bounds.minProperties !== undefined) parts.push(`members>=${String(bounds.minProperties)}`);
  if (bounds.maxProperties !== undefined) parts.push(`members<=${String(bounds.maxProperties)}`);
  return parts.join(' ');
}

/** An anchor without its schema identity: what a line shows of a place. */
function short(anchor: string): string {
  const hash = anchor.indexOf('#');
  return hash < 0 ? anchor : anchor.slice(hash);
}

/** A scalar as a line prints it. */
function printed(value: VocabularyValue): string {
  return typeof value === 'string' ? value : JSON.stringify(value);
}
