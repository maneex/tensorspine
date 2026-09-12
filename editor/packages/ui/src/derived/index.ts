/**
 * The Derived panel — plan §4.18: the six products, live, rendered from the derived schema.
 *
 * The rule this module is built on is the component inventory's §7: *every figure — D1–D6. No
 * component adds, converts or rounds a byte count.* So the products are **found** in the derived
 * document (`products.ts`), each value is written under the binding at its place (`cells.ts`),
 * the filter compares identifiers the schema types and the core reads (`naming.ts`), and what is
 * left for the component is which tab is showing and what a link does.
 */
export { cellOf, countText, figureIn, isRecord, NOTHING, valueFormats, type DerivedCell } from './cells.js';
export { DerivedPanel } from './Derived.js';
export { derivedBytes, derivedPathOf, type DerivedExport } from './export.js';
export { freshnessOf, type DerivedFreshness } from './freshness.js';
export {
  keptBy,
  namedIn,
  namedKinds,
  subjectKinds,
  type DerivedSubject,
  type Named,
} from './naming.js';
export {
  derivedReading,
  DERIVED_ROLE,
  LIST,
  noDerivedReading,
  ROW_LIMIT,
  sectionsOf,
  TABLE,
  tableRows,
  TOTALS,
  type DerivedColumn,
  type DerivedEntry,
  type DerivedField,
  type DerivedProduct,
  type DerivedReading,
  type DerivedRow,
  type DerivedRows,
  type DerivedSection,
  type HeaderField,
} from './products.js';
