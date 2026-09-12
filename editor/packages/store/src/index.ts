/**
 * `@tensorspine/store` — the document store.
 *
 * The document is the model (the plan's D1): this package holds the `tensorspine/2.0` JSON as
 * an ordered tree with its number lexemes, the commands that patch it, undo and redo, the
 * reference index, and the layout sidecar. It carries no rule of the language: what a document
 * means, whether it is valid and what it derives are `@tensorspine/lang`'s answers, and what a
 * name refers to is read from the schemas rather than remembered.
 */

/** The package's own name, as the workspace declares it. */
export const packageName = '@tensorspine/store';

export {
  arrayAt,
  child,
  EditError,
  exists,
  isUnder,
  keyOf,
  lastOf,
  nodeAt,
  objectAt,
  parentOf,
  pathOfKey,
  pathOfPointer,
  pointerOf,
  stepInto,
  type Path,
  type Step,
} from './path.js';
export {
  arrayDraftAt,
  asDraftValue,
  draftAt,
  isMutableArray,
  isMutableObject,
  memberIndex,
  objectDraftAt,
  removeMember,
  renameMember,
  setMember,
  stepIntoDraft,
  type MutableArray,
  type MutableMember,
  type MutableNumber,
  type MutableObject,
  type MutableValue,
} from './draft.js';
export { EditLog, type Edit, type Listener, type LogOptions, type Patch } from './log.js';
export { SchemaShapes, type Constraints, type Place, type Shape } from './shape.js';
export {
  matches,
  referenceIndex,
  referenceIndexWith,
  referenceTags,
  type Occurrence,
  type ReferenceIndex,
  type ReferenceKind,
  type ReferenceSelector,
  type ReferenceTags,
} from './references.js';
export {
  addToMap,
  connect,
  insertItem,
  remove,
  rename,
  sameValue,
  setMemberAt,
  setValue,
  shapeAt,
  unique,
  type Cascade,
  type Command,
  type EditContext,
  type PathMove,
} from './commands.js';
export { DocumentStore, MODEL_ROLE, type Applied, type DocumentStoreOptions } from './document.js';
export {
  crc32,
  DocumentSession,
  dosStamp,
  draftStanding,
  FIRST_VERSION,
  forgetDraft,
  gatherBases,
  gatherSchemas,
  hasOverrides,
  nameMember,
  newDocument,
  readSidecar,
  schemaDifferences,
  sidecarOf,
  fixedTag,
  tagOf,
  versionMember,
  WORKSPACE_SCHEMAS,
  zipOf,
  ZipError,
  type DraftStanding,
  type GatheredBases,
  type GatheredSchemas,
  type NewDocument,
  type Saved,
  type SchemaDifference,
  type SessionOptions,
  type ZipEntry,
} from './documents/index.js';
export {
  emptyLayout,
  LAYOUT_SCHEMA,
  LayoutStore,
  moved,
  prune,
  readLayout,
  resolves,
  writeLayout,
  type DroppedKey,
  type ExpandedView,
  type IndexRange,
  type Layout,
  type MutableLayout,
  type Position,
  type Pruned,
  type Viewport,
} from './layout.js';
export {
  listTree,
  readTree,
  within,
  type TreeOptions,
} from './platform/tree.js';
export type { WorkspacePath } from './platform/paths.js';
