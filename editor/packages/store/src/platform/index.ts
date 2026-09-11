/**
 * `@tensorspine/store/platform` — the interfaces of the implementation plan's §5.2 and the
 * implementations that name no platform (feature 2.4).
 *
 * A second entry point rather than the package root, for the same reason `@tensorspine/ui/layout`
 * and `@tensorspine/lang/api` are: what a module imports says what it depends on. It also keeps
 * two vocabularies apart — a `Path` in the store is a place in a document's tree and a
 * {@link WorkspacePath} is a place in a folder — so nothing has to be renamed to avoid a
 * collision in one flat surface.
 *
 * The static application's implementations are `apps/web/src/platform/`; Electron's and the
 * SaaS's come with their increments (D11). The stub is here, because CI builds the application
 * against it and because every suite in this workspace uses it.
 */
export {
  byName,
  fromPosix,
  isUnder,
  join,
  nameOf,
  normalise,
  parentOf,
  POSIX_ROOT,
  resolveFrom,
  segmentsOf,
  toPosix,
  WORKSPACE_ROOT,
  type WorkspacePath,
} from './paths.js';
export { PlatformError, type PlatformRefusal } from './errors.js';
export {
  ABSENT,
  type AuthProvider,
  type CheckpointSource,
  type Draft,
  type DraftStore,
  type DraftSummary,
  type Entry,
  type MenuCommand,
  type Platform,
  type RecentWorkspace,
  type Session,
  type SettingValue,
  type SettingsStore,
  type Shell,
  type SourceInfo,
  type Unsubscribe,
  type UploadedFile,
  type WatchEvent,
  type Workspace,
  type WorkspaceKind,
  type WorkspaceRef,
  type Workspaces,
} from './types.js';
export { noAuth } from './auth.js';
export { byNewest, draftKey, summaryOf } from './autosave.js';
export {
  byteLength,
  DROPPED_FOLDER,
  ReadOnlyWorkspace,
  snapshotOf,
  stripCommonRoot,
  textsOf,
  type Deliver,
  type Delivered,
  type FileSet,
} from './readonly.js';
export {
  MemoryWorkspace,
  memoryDrafts,
  memorySettings,
  recordingShell,
  stubPlatform,
  type ShellRecord,
  type StubOptions,
} from './memory.js';
export { listTree, readTree, within, type TreeOptions } from './tree.js';
