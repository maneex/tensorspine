/**
 * The schema stage of the language core: the registry, structural validation, and the vocabulary
 * the interface reads instead of carrying it (plan §1, D4, §5.3 `loadSchemas`).
 *
 * `loadSchemas` is the entry point. Everything else here is what it answers with — the problems
 * of a document that is off the grammar, in `--validate`'s own words, and the enumerations and
 * tagged unions the forms, the audits and the presentation bindings are generated from.
 */
export {
  formatProblem,
  formatProblems,
  loadSchemas,
  SchemaLoadError,
  type LoadedSchema,
  type SchemaFile,
  type SchemaRegistry,
  type StructuralOptions,
  type StructuralProblem,
} from './registry.js';
export {
  absolutePath,
  bestMatch,
  compareRelevance,
  deepest,
  isType,
  schemaError,
  sortByPlace,
  type SchemaError,
} from './errors.js';
export { factsOf, mergeFacts, NO_FACTS, type SchemaFacts } from './facts.js';
export { pythonPattern, pythonRegExp } from './pattern.js';
export {
  anchorOf,
  followAnchor,
  nodeAtPointer,
  parseAnchor,
  pointerSegment,
  pointerSteps,
  resolveAnchor,
  type Anchor,
  type Resolution,
} from './pointer.js';
export { comparePythonStrings, pythonRepr, pythonReprString } from './repr.js';
export {
  comparePaths,
  instanceOf,
  isSchemaObject,
  pointerOf,
  whereOf,
  type Instance,
  type PathSegment,
  type SchemaNode,
  type SchemaObject,
} from './types.js';
export { ASSERTION_KEYWORDS, type AssertionEngine } from './assertions.js';
export { KNOWN_KEYWORDS } from './walk.js';
export {
  alternativeLabel,
  vocabularyOf,
  type Vocabulary,
  type VocabularyAlternative,
  type VocabularyEnum,
  type VocabularyUnion,
  type VocabularyValue,
} from './vocabulary.js';
