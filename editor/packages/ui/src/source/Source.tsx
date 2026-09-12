/**
 * The JSON source view — plan §4.10, artboard S16, component inventory §5's "Source pane".
 *
 * > Two-way: an edit in the source updates the tree after a successful parse (debounced), an edit
 * > on the canvas or in a form updates the source text minimally (the serializer preserves member
 * > order and number lexemes and rewrites the changed subtree). "Show in JSON" from any selection
 * > reveals its range. An off-schema source is allowed to exist (§3).
 *
 * **The two writers, stated.** The tree is the model (D1) and `serialize` is the writer of record
 * (D12). This pane holds no second model: it holds the *text* a reader is typing, and that text
 * is a projection of the tree except while it is being typed, when it is the writer. In the two
 * directions:
 *
 *  - **text → tree.** Three hundred milliseconds after the last keystroke the text is parsed. A
 *    text that parses to an object is written into the tree as **one named command** (`Edit the
 *    JSON source`), so it lands in the command log like every other gesture (D13), one Ctrl+Z
 *    takes it back, and the sidecar is pruned of what no longer resolves (D6). It is written *off
 *    the grammar included* — D5's own compensation and Q5's rule — and Ajv's rows then land at
 *    their ranges. A text that is not a document is not written at all: the tree stays what it
 *    was, the core's own `[V12]` line says why, and the canvas keeps its last drawable state.
 *  - **tree → text.** Whenever the document moves for a reason that is not this pane's own edit —
 *    a canvas gesture, a sheet row, an undo, a revert — the pane takes `serialize(tree)` and
 *    applies it as **one minimal edit** (`./edits.ts`), so the caret, the selection, the folded
 *    regions and Monaco's own undo stack stay where the reader left them. The pane's own edit does
 *    not come back at it, which is what leaves a reader's own spacing alone until something else
 *    writes; from then on the serializer's layout is what stands, because the bytes are its (D12).
 *
 * **Every path that reads the document flushes the pane first.** A Ctrl+S typed one keystroke
 * after an edit must write what the reader is looking at, so the pane registers its flush with the
 * store (`holdSource`) and Save, Save As, the autosave, the archive and a close that asks all go
 * through it. Leaving the editor and closing the tab flush too.
 *
 * **React drives the chrome and nothing else.** A controlled component over a text editor is a
 * fight over the caret: Monaco owns the text, the effects below each own one thing, and the head's
 * figures are the only thing that re-renders while a reader types.
 */
import { useEffect, useMemo, useRef, useState, type JSX } from 'react';

import { spansOf, type JsonSpan } from '@tensorspine/lang';
import type { Problem } from '@tensorspine/lang/api';

import { useDocuments, useDocumentsStore } from '../documents/context.js';
import { rowsOf } from '../documents/pipeline.js';
import { sourceReading, type OpenDocument } from '../documents/store.js';
import { useShell } from '../shell/context.js';
import { text, textWith } from '../shell/strings.js';
import { resolveTheme } from '../shell/theme.js';

import { minimalEdit } from './edits.js';
import { markersFor, revealRange } from './markers.js';
import type { MonacoApi, MonacoEditor, MonacoModel, SchemaAttachment } from './monaco.js';

/** How long a keystroke waits before the text is parsed (§4.10's "debounced", §5.6's own figure). */
export const SOURCE_DEBOUNCE_MS = 300;

/** The scheme a document's model is named under, so a schema can be matched to one document. */
const MODEL_SCHEME = 'tensorspine';

/** The owner the editor's markers are set under, so nothing else's are disturbed. */
const MARKER_OWNER = 'tensorspine';

/** The indentation §5.5 fixes for every document and unit the editor writes: two spaces (D12). */
const INDENT = 2;

/**
 * Where each document's pane was left, by its path — the scroll, the caret, the folded regions.
 *
 * Only one tab of the editor area is mounted at a time (§4.2), so looking at the canvas and
 * coming back re-creates the editor over the same model. A reader who was a thousand lines down
 * and is put back at the top has lost their place for having glanced at the diagram; the model
 * survives the round trip and this makes the view survive it too.
 */
const views = new Map<string, unknown>();

/** What the pane holds once Monaco is up. */
interface Live {
  readonly monaco: MonacoApi;
  readonly editor: MonacoEditor;
  readonly model: MonacoModel;
}

/** The source pane of one document. */
export function SourceEditor({ one }: { one: OpenDocument }): JSX.Element {
  const store = useDocumentsStore();
  const registry = useDocuments((state) => state.registry);
  const choice = useShell((state) => state.theme);
  const scheme = useShell((state) => state.scheme);
  const light = resolveTheme(choice, scheme) === 'light';

  const host = useRef<HTMLDivElement>(null);
  const live = useRef<Live | null>(null);
  /** The text the pane last synchronised with the document; what the effect compares against. */
  const shown = useRef<string>('');
  /** The debounce in flight. */
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [ready, setReady] = useState(false);
  const [caret, setCaret] = useState<{ line: number; column: number }>({ line: 1, column: 1 });

  const source = sourceReading(one);
  /** What the pane should be showing: the reader's own text, or the document's own bytes (D12). */
  const held = source?.text ?? one.session.text;
  /**
   * Every row the pane marks: the document's, and — where the text is not one the tree could take
   * — the core's own refusal about the text itself, which is the only row about it there is.
   */
  const problems: readonly Problem[] = useMemo(
    () => (source?.pending === true ? [...source.problems] : rowsOf(one.reading).map((row) => row.problem)),
    [source?.pending, source?.problems, one.reading],
  );
  // What the markers are recomputed for: the rows as one text, joined on two characters no
  // message carries — the core's own idiom for an injective key (`valueToken`, feature 1.6b).
  const key = problems.map((row) => [row.code, row.path, row.message].join('\u0000')).join('\u0001');

  // ── The editor itself: created once per mounted pane, over a model kept per document. ────────
  useEffect(() => {
    let alive = true;
    const node = host.current;
    if (node === null) return;
    const id = one.id;
    const path = one.path;
    const start = held;

    const flush = (): void => {
      if (timer.current === null) return;
      clearTimeout(timer.current);
      timer.current = null;
      const value = live.current?.model.getValue();
      if (value !== undefined) store.getState().sourceEdit(value, id);
    };

    void (async () => {
      const { loadMonaco, defineTheme, JSON_LANGUAGE, THEME } = await import('./monaco.js');
      const monaco = await loadMonaco();
      if (!alive) return;
      defineTheme(monaco, node, light);
      const uri = monaco.Uri.from({ scheme: MODEL_SCHEME, path: `/${path}` });
      // The model outlives the pane, so that switching to the canvas and back keeps the scroll,
      // the selection and the editor's own undo stack; what it does not outlive is its document,
      // and a document that has been closed is swept here rather than left in the page for ever.
      const open = new Set(store.getState().open.map((document) => document.path));
      for (const held of monaco.editor.getModels()) {
        if (held.uri.scheme !== MODEL_SCHEME || held.uri.path === uri.path) continue;
        if (!open.has(held.uri.path.slice(1))) held.dispose();
      }
      const model = monaco.editor.getModel(uri) ?? monaco.editor.createModel(start, JSON_LANGUAGE, uri);
      const editor = monaco.editor.create(node, {
        model,
        theme: THEME,
        automaticLayout: true,
        fontFamily: getComputedStyle(node).getPropertyValue('--mono').trim(),
        fontSize: 12,
        lineHeight: 19,
        minimap: { enabled: false },
        scrollBeyondLastLine: false,
        tabSize: INDENT,
        insertSpaces: true,
        renderLineHighlight: 'gutter',
        // Monaco colours nested brackets by depth from a palette of its own — four inks the
        // design does not carry, two of which do not reach §4.21's floor on this ground. The
        // design's own rule is that "colour carries meaning and nothing else" (component
        // inventory §1), and a bracket's depth is not one of the four meanings it names.
        bracketPairColorization: { enabled: false },
        padding: { top: 8, bottom: 8 },
        // A widget clipped by the pane's own scrolling is a widget nobody can read.
        fixedOverflowWidgets: true,
        ariaLabel: textWith('The JSON source of {}', path),
      });
      live.current = { monaco, editor, model };
      const left = views.get(path);
      if (left !== undefined) editor.restoreViewState(left as Parameters<typeof editor.restoreViewState>[0]);
      shown.current = model.getValue();
      editor.onDidChangeModelContent(() => {
        // The reader's own text: remembered at once so the effect below does not fight it, and
        // given to the document when the typing settles.
        shown.current = model.getValue();
        if (timer.current !== null) clearTimeout(timer.current);
        timer.current = setTimeout(() => {
          timer.current = null;
          store.getState().sourceEdit(model.getValue(), id);
        }, SOURCE_DEBOUNCE_MS);
      });
      editor.onDidChangeCursorPosition((event) => {
        setCaret({ line: event.position.lineNumber, column: event.position.column });
      });
      // Leaving the editor finishes the edit: whatever the reader clicks on next has to be
      // showing what they typed.
      editor.onDidBlurEditorText(() => {
        flush();
      });
      setReady(true);
    })();

    store.getState().holdSource(id, flush);
    return () => {
      alive = false;
      store.getState().holdSource(id, null);
      flush();
      const state = live.current?.editor.saveViewState();
      if (state !== undefined && state !== null) views.set(path, state);
      live.current?.editor.dispose();
      live.current = null;
      setReady(false);
    };
    // The editor is made once per mounted pane and for one document; the theme, the text, the
    // markers and the schemas are followed by the effects below, which is what keeps this one
    // from tearing it down. `held` and `light` are read once, as the model's starting text and
    // its starting palette, and the two effects that own them take over from there. (There is no
    // `react-hooks` plugin in this workspace's ESLint configuration, so the dependency list is a
    // decision stated here rather than one a rule has to be silenced for.)
  }, [one.id, one.path, store]);

  // ── tree → text: the document moved for a reason that was not this pane's. ───────────────────
  useEffect(() => {
    const current = live.current;
    if (current === null) return;
    if (held === shown.current) return;
    const edit = minimalEdit(current.model.getValue(), held);
    shown.current = held;
    // A text arriving from elsewhere supersedes whatever was half-typed, and the debounce it was
    // waiting for goes with it: the tree is the model (D1).
    if (timer.current !== null) {
      clearTimeout(timer.current);
      timer.current = null;
    }
    if (edit === null) return;
    current.model.pushEditOperations(
      null,
      [{ range: rangeOf(current, edit.start, edit.end), text: edit.text }],
      () => null,
    );
  }, [held, ready]);

  // ── The markers: the core's rows, at the ranges the core's own spans put them (§4.10). ───────
  const refusedAt = source?.refusedAt;
  useEffect(() => {
    const current = live.current;
    if (current === null) return;
    const value = current.model.getValue();
    const marked = markersFor(problems, spansAt(value), value).map((marker) => ({
      ...rangeOf(current, marker.start, marker.end),
      message: marker.message,
      severity: severityOf(current.monaco, marker.severity),
      source: marker.source,
      ...(marker.code === '' ? {} : { code: marker.code }),
    }));
    // A text the core could not read has no places, so its own refusal is placed by the *parser*,
    // at the character CPython's message names. It is the one row whose range is not a span.
    const refusal = refusedAt === undefined ? [] : [problems[0]];
    const markers = [
      ...refusal.flatMap((row) =>
        row === undefined || refusedAt === undefined
          ? []
          : [
              {
                ...rangeOf(current, refusedAt, Math.min(refusedAt + 1, value.length)),
                message: row.message,
                severity: severityOf(current.monaco, row.severity),
                source: row.source,
                ...(row.code === '' ? {} : { code: row.code }),
              },
            ],
      ),
      ...marked,
    ];
    current.monaco.editor.setModelMarkers(current.model, MARKER_OWNER, markers);
  }, [held, key, ready, refusedAt]);

  // ── The schemas: completion and hover are the schema's, and nothing else is (§1, §4.10). ─────
  const role = one.session.store.role;
  useEffect(() => {
    const current = live.current;
    if (current === null || registry === null) return;
    const uri = current.monaco.Uri.from({ scheme: MODEL_SCHEME, path: `/${one.path}` }).toString();
    const attachments: SchemaAttachment[] = registry.schemas.map((schema) => ({
      uri: schema.id,
      schema: schema.document,
      ...(schema.role === role ? { fileMatch: [uri] } : {}),
    }));
    void (async () => {
      const { attachSchemas } = await import('./monaco.js');
      attachSchemas(current.monaco, attachments);
    })();
  }, [registry, one.path, role, ready]);

  // ── The theme: read back off the page, so a switch repaints (§4.21). ─────────────────────────
  useEffect(() => {
    const current = live.current;
    if (current === null) return;
    void (async () => {
      const { defineTheme } = await import('./monaco.js');
      if (host.current !== null) defineTheme(current.monaco, host.current, light);
    })();
  }, [light, ready]);

  // ── "Show in JSON": the place a command or a problem row asked for (§4.10, §4.17). ───────────
  const pointer = one.reveal?.pointer;
  const seq = one.reveal?.seq;
  useEffect(() => {
    const current = live.current;
    if (current === null || pointer === undefined) return;
    const range = revealRange(spansAt(current.model.getValue()), pointer);
    if (range === null) return;
    const where = rangeOf(current, range.start, range.end);
    current.editor.setSelection(where);
    current.editor.revealRangeInCenterIfOutsideViewport(where);
  }, [pointer, seq, ready, held]);

  const errors = problems.filter((problem) => problem.severity === 'error').length;
  const schema = registry?.locate(role);
  return (
    <div className="src-pane" data-path={one.path} data-ready={ready ? 'true' : 'false'}>
      <div className="src-head">
        <span className="src-name">{one.session.name}</span>
        <span className="src-schema">
          {schema === undefined ? text('no schema is loaded') : textWith('JSON · {} attached', schema.id)}
        </span>
        <span className="right">
          {errors === 0 ? null : (
            <span className="src-bad" data-problems={errors}>
              {textWith(errors === 1 ? '{} problem' : '{} problems', String(errors))}
            </span>
          )}
          {/* S16's own head: `ln 7, col 3`, composed the way feature 2.16 composes a pair of
              figures — one key per word, so a dictionary can move them. */}
          <span className="fig" data-caret={`${String(caret.line)}:${String(caret.column)}`}>
            {`${text('ln')} ${String(caret.line)}, ${text('col')} ${String(caret.column)}`}
          </span>
          <span className="fig">{textWith('spaces: {}', String(INDENT))}</span>
        </span>
      </div>
      <div className="src-body" ref={host} />
      {ready ? null : <p className="src-loading">{text('The editor is loading…')}</p>}
    </div>
  );
}

/** The places of a text, or none where it is not one the core can read. */
function spansAt(value: string): ReadonlyMap<string, JsonSpan> {
  try {
    return spansOf(value);
  } catch {
    // A text the core cannot read has no places; the row about the text itself carries its own
    // position, and the Problems panel is where it reads.
    return new Map<string, JsonSpan>();
  }
}

/** Two offsets as a Monaco range, which is where offsets become lines and columns. */
function rangeOf(
  current: Live,
  start: number,
  end: number,
): { startLineNumber: number; startColumn: number; endLineNumber: number; endColumn: number } {
  const from = current.model.getPositionAt(start);
  const to = current.model.getPositionAt(end);
  return {
    startLineNumber: from.lineNumber,
    startColumn: from.column,
    endLineNumber: to.lineNumber,
    endColumn: to.column,
  };
}

/** A problem's weight as Monaco marks one. */
function severityOf(monaco: MonacoApi, weight: string): number {
  if (weight === 'error') return monaco.MarkerSeverity.Error;
  if (weight === 'warning') return monaco.MarkerSeverity.Warning;
  return monaco.MarkerSeverity.Info;
}
