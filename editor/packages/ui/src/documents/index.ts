/**
 * Open and save — feature 2.6: plan §4.3 and the File menu of §4.4.
 *
 * The store is `store.ts` and the pipeline of §5.4 is `pipeline.ts`; the rest is what the chrome
 * draws around them. Nothing here computes a verdict or a figure: the core answers and this
 * displays, which is the component inventory's §7 stated as a package boundary.
 */
export {
  DocumentsProvider,
  NO_DOCUMENTS,
  useDocuments,
  useDocumentsStore,
} from './context.js';
export { DocumentDialogs } from './Dialogs.js';
export {
  elementsText,
  figureOf,
  groupedNumber,
  operationsText,
  renderedFormats,
  sizeText,
  statusFigures,
  type Figure,
  type FigureShapes,
  type StatusFigure,
} from './figures.js';
export {
  countsOf,
  noReading,
  Pipeline,
  VALIDATE_DEBOUNCE_MS,
  type DerivationState,
  type PipelineOptions,
  type Reading,
} from './pipeline.js';
export { DERIVED_ROLE, shapesFor } from './shapes.js';
export {
  AUTOSAVE_MS,
  bannerFor,
  createDocuments,
  DOCUMENT_TAB,
  documentTab,
  DRILL_SUFFIX,
  DRILL_TAB,
  drillOf,
  SOURCE_SUFFIX,
  SOURCE_TAB,
  TABS_SETTING,
  WORKSPACE_SETTING,
  type BannerLine,
  type Dialog,
  type Documents,
  type DocumentsOptions,
  type DocumentsState,
  type DocumentsStore,
  type LibraryState,
  type OpenDocument,
  type TabSink,
  type ToastLine,
} from './store.js';
export {
  documentFor,
  DOCUMENT_VIEWS,
  DocumentToast,
  DocumentView,
  SourcePane,
  WorkspaceBanner,
} from './views.js';
