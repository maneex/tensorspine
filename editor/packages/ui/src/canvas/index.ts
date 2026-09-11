/**
 * The folded canvas — plan §4.7, artboards S1, S2 and S3.
 *
 * The default editor of a model: root instances as cards, compositions as group boxes with their
 * boundary handles, interfaces as terminals, value bindings as wires, laid out top to bottom by
 * ELK with the sidecar's manual moves as overrides, and every gesture of §4.7's interactions
 * table a named command of the store.
 *
 * The package's fifth entry point, `@tensorspine/ui/canvas`. It is its own because it reaches the
 * layout — and through it a megabyte and a half of compiled ELK, which feature 0.4 measured and
 * which is loaded on the first drawing rather than in the shell's first bundle.
 */
export { Canvas, handleFor, primitiveOf, PRIMITIVE_TRANSFER, splitPort, type PrimitiveTransfer } from './Canvas.js';
export { Box, slotClass, slotMark, type BoxProps } from './Box.js';
export { CANVAS_VIEWS } from './views.js';
export {
  addInstance,
  connectHandles,
  declarationAt,
  declaresAt,
  duplicateAt,
  referencesTo,
  removeAt,
  renameAt,
  type GestureContext,
} from './gestures.js';
export {
  collapsedGroups,
  expandedFromSidecar,
  fit,
  hasManualMoves,
  layoutKey,
  NO_PLACEMENT,
  place,
  routes,
  type Fit,
  type PlaceRequest,
  type Placement,
  type PlacedBox,
  type WireRoute,
} from './layout.js';
export { CONTEXT_MENU, entriesFor, type MenuEntry } from './menu.js';
export {
  canvasModel,
  DEFAULT_VIEW,
  descriptionsByBox,
  ROLE,
  SIDE,
  sizeOf,
  type CanvasBox,
  type CanvasHandle,
  type CanvasLink,
  type CanvasModel,
  type CanvasPort,
  type CanvasRequest,
  type CanvasSlot,
  type CanvasView,
  type CanvasWire,
  type HandleState,
} from './model.js';
