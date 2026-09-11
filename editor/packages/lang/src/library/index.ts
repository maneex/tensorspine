/**
 * The primitive library loader (plan §5.3, `loadLibrary` and `validateUnit`): the port of
 * `tools/primitive_library.py`, with `validate.template_interface` beside it.
 *
 * "The primitive library is the closed vocabulary a consumer implements in advance (§4.3, O0.6); a
 * unit outside it is a load error naming the file, never a document read with a guess (I7)."
 * Everything here serves that sentence: a base is a set of units whose paths reproduce their
 * identities, every unit is read against the unit schema, and the references one unit makes to
 * another are resolved once the bases are gathered.
 *
 * Two entry points, one body:
 *
 * - `loadLibrary(bases, context)` gathers the bases and answers the primitive library with every
 *   refusal it carries. The tools raise at the first; the port collects them in the tools' order
 *   so that the first one, rendered by `formatLibraryProblem`, is the text they would have raised.
 * - `validateUnit(unit, where, library, context)` runs the same checks on one unit against bases
 *   already gathered — the live judgement of §4.22, D15.
 */
export {
  basesOf,
  identityKey,
  librariesFor,
  libraryUnits,
  loadLibrary,
  primitiveOf,
  semanticVersion,
  templateInterfaces,
  templateOf,
  templatePinOf,
  templatePrimitives,
  SECTIONS,
  UNIT_SCHEMA,
  type BasesResult,
  type Library,
  type LibraryContext,
  type LibrarySection,
  type LibraryUnit,
  type LoadedBase,
  type PrimitiveVersion,
} from './load.js';
export {
  placeOf,
  validateUnit,
  validateUnitText,
  type UnitLocation,
  type UnitPlace,
} from './unit.js';
export {
  absentComparisons,
  conditionPaths,
  declaredPaths,
  expressionPaths,
  optionalPaths,
  primitiveReferences,
  type ReferenceLibrary,
} from './references.js';
export {
  pinnedTemplate,
  templateInterface,
  toPrimitiveExpression,
  type PinResult,
  type TemplatePin,
} from './template.js';
export {
  detailAt,
  formatLibraryProblem,
  formatLibraryProblems,
  libraryProblem,
  PrimitiveLibraryError,
  type LibraryDetail,
  type LibraryProblem,
  type LibraryProblemCode,
  type LibraryProblemKind,
} from './problems.js';
export { LibrarySourceError, memorySource, type LibrarySource } from './source.js';
export { readRefusal, readText, type ReadText, type TextReading } from './read.js';
export { layoutVocabulary, legacyLayout } from './legacy.js';
export { pyRepr, pyStr } from './repr.js';
export { basename, dirname, isAbsolute, join, normalise as normalisePath, relative } from './paths.js';
