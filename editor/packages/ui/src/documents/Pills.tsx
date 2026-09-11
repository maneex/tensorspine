/**
 * The bar's two pills and the status bar's eight document fields — feature 2.6.
 *
 * Feature 2.5 left both empty and said why: "every one of them is a figure D1–D6 computes", and
 * writing `tensorspine/2.0` into the interface is the `const` feature 2.1 refused to write. With a
 * document open they are answerable, and every one of them is answered by asking:
 *
 * | §4.2's field | Where it comes from |
 * |---|---|
 * | model id | the member the schema says is the document's name (`nameMember`), read off the tree |
 * | `tensorspine/2.0` | the member the schema **fixes**, read off the same tree — never a literal |
 * | validation state | `Lang.validate`'s verdict: checking, or its rows counted |
 * | derivation state | whether a derivation ran for *this* revision (§5.4's freshness) |
 * | D3 total bytes | the derived document, at the place `presentation.json` marks |
 * | D5 operations per element | the same |
 * | D4 append bytes per cached position | the same |
 * | D2 peak live bytes per element | the same |
 *
 * The two pills are the first two of those states, which is what the component inventory says
 * they are: "the pills are the core's validation and derivation state".
 */
import type { JSX } from 'react';

import { useShellStore } from '../shell/context.js';
import { useDocuments } from './context.js';
import { countsOf, type DerivationState, type Reading } from './pipeline.js';
import type { OpenDocument } from './store.js';
import { text, textWith } from '../shell/strings.js';

/** The document the bar and the status bar are about: the tab that is current. */
export function useCurrentDocument(): OpenDocument | undefined {
  return useDocuments((state) => state.open.find((one) => one.id === state.current));
}

/**
 * Whether the status bar has a document's own eight fields to show.
 *
 * §4.2 gives the bar ten fields and no more; the two feature 2.5 put there in their place — what
 * a read-only workspace does with a Save, and which platform this is — belong to an editor with
 * nothing open, and the banner and Help ▸ About say both once there is something.
 */
export function useOpenDocument(): boolean {
  return useDocuments((state) => state.open.some((one) => one.id === state.current));
}

/** What the first pill says, and the class that colours it. */
export function validationPill(reading: Reading): { text: string; tone: string } {
  if (reading.checking) return { text: text('checking…'), tone: 'stale' };
  if (reading.verdict === null && (reading.structural ?? []).length === 0) {
    return { text: text('not checked'), tone: 'stale' };
  }
  const { errors, warnings } = countsOf(reading);
  if (errors > 0) return { text: textWith('{} problems', String(errors + warnings)), tone: 'bad' };
  if (warnings > 0) return { text: textWith('{} warnings', String(warnings)), tone: 'der' };
  return { text: text('no problems'), tone: 'ok' };
}

/** What the second pill says about the derivation (§4.18's `fresh` / `stale` / `failed`). */
export function derivationPill(reading: Reading): { text: string; tone: string } {
  const named: Readonly<Record<DerivationState, string>> = {
    waiting: 'not derived',
    running: 'deriving…',
    fresh: 'derived · fresh',
    stale: 'derived · stale',
    failed: 'derivation failed',
    skipped: 'not derived',
  };
  const state = reading.derivedAt === reading.revision ? reading.derivation : staleOf(reading);
  const tone = state === 'fresh' ? 'der' : state === 'failed' ? 'bad' : 'stale';
  return { text: text(named[state]), tone };
}

/**
 * What a derivation that is not for the document's own revision reads as.
 *
 * Stale, where there *is* a derived document — the figures beside it are about an older reading
 * and the bar says so. Where there is none, "stale" would be a word about a figure that does not
 * exist: the bar says the document has not been derived, or that it is being derived now. A
 * refusal stays a refusal whichever revision it was for.
 */
function staleOf(reading: Reading): DerivationState {
  if (reading.derivation === 'failed') return 'failed';
  if (reading.derived !== null) return 'stale';
  return reading.derivation === 'running' ? 'running' : 'waiting';
}

/**
 * What the bottom panel says about the open document until its own rows are built.
 *
 * `null` with nothing open, which is what the shell already had a line for. With a document, the
 * three lines are the core's own answers: how many problems it found, how the derivation stands,
 * and which document all of that is about.
 */
export function useDocumentState(): { problems: string; derivation: string; where: string } | null {
  const one = useCurrentDocument();
  if (one === undefined) return null;
  const validation = validationPill(one.reading);
  const derivation = derivationPill(one.reading);
  return {
    problems: validation.text,
    derivation: derivation.text,
    where: `${one.title} — ${one.path}`,
  };
}

/**
 * The bar's two pills, shown only with a document open (inventory §2).
 *
 * The validation pill is a **button**: §4.17 puts the count in the status bar and a count nobody
 * can act on is a count that sends the reader hunting for the panel. It reveals Problems, which is
 * the same thing `View ▸ Problems` does — a second way in, never the only one (§4.4).
 */
export function BarPills(): JSX.Element | null {
  const shell = useShellStore();
  const one = useCurrentDocument();
  if (one === undefined) return null;
  const validation = validationPill(one.reading);
  const derivation = derivationPill(one.reading);
  return (
    <>
      <button
        type="button"
        className={`pill ${validation.tone}`}
        data-pill="validation"
        title={text('Show the Problems panel')}
        onClick={() => {
          shell.getState().revealPanel('panel.problems');
        }}
      >
        <i aria-hidden="true" />
        {validation.text}
      </button>
      <span className={`pill ${derivation.tone}`} data-pill="derivation">
        <i aria-hidden="true" />
        {derivation.text}
      </span>
    </>
  );
}

/**
 * The document's own fields of the status bar.
 *
 * The model id and the schema tag are read off the document's tree at the two members the schema
 * itself names — the one it requires as free text, and the one it fixes — so neither is written
 * here. The four figures are whatever the presentation file marked, in its own order.
 */
export function StatusFields(): JSX.Element | null {
  const shell = useShellStore();
  const one = useCurrentDocument();
  if (one === undefined) return null;
  const validation = validationPill(one.reading);
  const derivation = derivationPill(one.reading);
  return (
    <>
      <span className="mono" data-document={one.path}>
        {one.title}
      </span>
      {one.tag === undefined ? null : (
        <span className="mono dimf" data-tag={one.tag}>
          {one.tag}
        </span>
      )}
      <button
        type="button"
        className={`asfield ${validation.tone === 'ok' ? 'ok' : validation.tone === 'bad' ? 'bad' : ''}`}
        data-validation={validation.tone}
        title={text('Show the Problems panel')}
        onClick={() => {
          shell.getState().revealPanel('panel.problems');
        }}
      >
        {validation.text}
      </button>
      <span className="der" data-derivation={derivation.tone}>
        {derivation.text}
      </span>
      {one.figures.map((figure) => (
        <span className="fig d" key={figure.anchor} data-figure={figure.label} title={figure.figure.exact}>
          <b>{figure.figure.text}</b>
          {text(figure.label)}
          {figure.status === undefined ? null : <i className="chip">{figure.status}</i>}
        </span>
      ))}
    </>
  );
}
