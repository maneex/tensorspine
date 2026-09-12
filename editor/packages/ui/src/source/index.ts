/**
 * The JSON source view — feature 2.17, plan §4.10, §3, artboard S16.
 *
 * Monaco with the loaded schemas attached for completion and hover, the core's own rows marked at
 * the ranges the core's own spans put them, two-way sync with the document tree, "Show in JSON",
 * and the banner an off-grammar source stands behind.
 *
 * **Monaco is not reachable from here.** Four modules name it — `./monaco.ts`, which nothing
 * imports statically, its feature list and the two worker entry points — and
 * `test/source/bundle.test.ts` reads every source of every package and says so, the way feature
 * 2.16 says the expanded graph never reaches the layout.
 */
export { minimalEdit, type TextEdit } from './edits.js';
export {
  markersFor,
  markRange,
  revealRange,
  type SourceMarker,
  type SourceSpans,
  type TextRange,
} from './markers.js';
export { SourceEditor, SOURCE_DEBOUNCE_MS } from './Source.js';
export { SOURCE_VIEWS } from './views.js';
