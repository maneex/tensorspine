/**
 * The Problems panel — plan §4.17, §5.4, artboard S10.
 *
 * The one place in the editor where a verdict lands. Q5 decided that no gesture is refused for a
 * semantic reason — "the author wires, binds and names first, and the panel says what is left to
 * fix" — so everything the core refuses, everything it advises and the editor's own few notices
 * arrive here, with the tools' own wording, the place in the document as written, and a link to
 * where the specification states the rule.
 *
 * It is not a separate entry point: the panel is a body of the shell's bottom region, which is
 * where `@tensorspine/ui/shell` renders it (2.5's `Panels.tsx`), and the module is imported from
 * there rather than mounted by the application.
 */
export { bannerOf, BANNERS, type BannerKind, type ProblemsBanner } from './banner.js';
export {
  FIX_PROVIDERS,
  fixesFor,
  type Fix,
  type FixProvider,
  type FixReading,
} from './fixes.js';
export {
  droppedKeys,
  NOTICE,
  schemaMismatches,
  unusedQuantities,
  type DroppedSidecarKey,
  type QuantityReading,
  type SchemaMismatch,
} from './notices.js';
export {
  aboutOf,
  FLAT,
  foldKey,
  NO_FILTER,
  problemsView,
  type AboutDocument,
  type DeclaredPlace,
  type Grouping,
  type ProblemFilter,
  type ProblemGroup,
  type ProblemRow,
  type ProblemsRequest,
  type ProblemsView,
  type Sourced,
} from './rows.js';
export { ProblemControls, ProblemsPanel, useProblemCount } from './Problems.js';
