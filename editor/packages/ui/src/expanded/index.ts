/**
 * The expanded graph — plan §4.9, artboard S12: the read-only, filtered, virtualised tab over D1.
 *
 * The package's sixth entry point, `@tensorspine/ui/expanded`, and it is its own for the opposite
 * of the canvas's reason. The canvas is separate because it *reaches* ELK; this view is separate
 * because it must **not**: feature 0.4 measured a whole expanded layout at the two-second budget
 * rather than under it, so the reading here is D1's own topological order — which is what the
 * artboard draws and what virtualisation needs — and `test/expanded/graph.test.ts` asserts that no
 * module of this directory reaches the layout.
 */
export { Expanded, ExpandedView, type PreviewContext } from './Expanded.js';
export {
  expandedReading,
  filtering,
  keeps,
  NO_FILTER,
  NO_READING,
  OVERSCAN,
  ROW_HEIGHT,
  windowOf,
  type ExpandedFilter,
  type ExpandedReading,
  type ExpandedRow,
  type ExpandedWindow,
} from './graph.js';
export { EXPANDED_VIEWS } from './views.js';
