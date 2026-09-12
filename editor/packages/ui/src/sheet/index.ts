/**
 * The Properties sheets — plan §4.11 and §4.12, artboard S6.
 *
 * The instance sheet and the argument sheet: every section of §4.11's first row, generated from
 * the schemas and enriched by the core's `describe`, with the literal-mode widgets read from the
 * tools' own generated argument schemas (F5); and (feature 2.12) the other sheets of §4.11 — the
 * composition, the edge, the identity, the input, the output, the quantity, the constant and the
 * document — with §4.16's tables, the generic renderer they are all drawn by, and what `Add
 * quantity`, `Add constant` and §4.15's two `Expose as …` gestures write. It is not a separate
 * entry point: the sheet is a body of the shell's right-hand region, which is where
 * `@tensorspine/ui/shell` renders it.
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
  writeValue,
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
export {
  addDeclaration,
  declaringMaps,
  endpointMember,
  exposeAt,
  interfaceMap,
  mapDeclaring,
  type AddRequest,
  type DeclaringMap,
} from './add.js';
export {
  mapTable,
  placeSheet,
  shapeAt,
  type IdentityFacts,
  type PlaceRequest,
  type PlaceSheet,
  type PlaceTable,
  type TableCell,
  type TableRow,
  type UsedBy,
} from './places.js';
export { PlaceView, type PlaceFacts, type PlaceProps } from './Place.js';
export { FormRows, ModeSelect, NameRow, ScalarField, type RowsProps } from './Rows.js';
export { blankValue, itemShape, memberShape, valueShape } from './skeleton.js';
export { SelectionSheet } from './Sheet.js';
