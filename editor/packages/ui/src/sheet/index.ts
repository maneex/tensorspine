/**
 * The Properties sheets — plan §4.11 and §4.12, artboard S6.
 *
 * The instance sheet and the argument sheet: every section of §4.11's first row, generated from
 * the schemas and enriched by the core's `describe`, with the literal-mode widgets read from the
 * tools' own generated argument schemas (F5). It is not a separate entry point: the sheet is a
 * body of the shell's right-hand region, which is where `@tensorspine/ui/shell` renders it.
 */
export {
  argumentSheet,
  type ArgumentRow,
  type ArgumentSheet,
  type ArgumentSheetRequest,
  type NamedBound,
} from './arguments.js';
export {
  ArgumentSchema,
  NO_ARGUMENT_FACTS,
  primitiveId,
  type ArgumentBound,
  type ArgumentSchemaFacts,
} from './artifact.js';
export {
  blankOf,
  clearValue,
  literalMode,
  literalOf,
  pinValue,
  referringModes,
  writeLiteral,
  writeMode,
  writeRecord,
} from './edits.js';
export {
  instanceSheet,
  shapeLine,
  targetText,
  type CostRow,
  type InstanceSheet,
  type PartitionRow,
  type PortRow,
  type SlotRow,
  type StateRow,
} from './instance.js';
export { namesFor, type NameChoice, type NameRequest } from './names.js';
export { SelectionSheet } from './Sheet.js';
