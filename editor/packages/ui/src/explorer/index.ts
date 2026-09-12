/**
 * The Model explorer — plan §4.5, the first of the activity rail's four panels.
 *
 * `outline.ts` is the model (the document walked against the schema, with the bindings of 2.2 and
 * the figures of the core), `Explorer.tsx` the tree the side bar draws, `Sheet.tsx` the Properties
 * body for whatever is selected. The selection itself lives in the documents store (2.6), because
 * D1 makes it a place of the one tree and the canvas and the sheets read the same place.
 */
export {
  declarationOf,
  DECLARATION_TRANSFER,
  ModelExplorer,
  NAME_FIELD,
  removeCommand,
  renameCommand,
  selectedRow,
  selectionHandlers,
  type DeclarationTransfer,
} from './Explorer.js';
export {
  COMPUTED,
  GUARDED,
  outlineOf,
  SHARED,
  summaryOf,
  TEMPLATE,
  type OutlineRequest,
  type OutlineRow,
  type RowKind,
} from './outline.js';

/** The activity the shell shows this panel under (§4.2's rail, §4.5's own name). */
export const EXPLORER_ACTIVITY = 'activity.explorer';
