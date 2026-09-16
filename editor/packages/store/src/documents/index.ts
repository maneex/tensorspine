/**
 * The open document — feature 2.6's half of §4.3, framework-free.
 *
 * What a tab of the editor area is (`session.ts`), what `New Model` and `New Template` make
 * (`skeleton.ts`), what a document needs read before it can be judged (`gather.ts`), and what
 * `Download Workspace as Zip` writes (`zip.ts`). None of it draws anything: the React binding is
 * `@tensorspine/ui/documents`, and the workspace behind it is `Platform`'s (§5.2).
 */
export {
  gatherBases,
  gatherSchemas,
  schemaDifferences,
  WORKSPACE_SCHEMAS,
  type BaseMount,
  type GatheredBases,
  type GatheredSchemas,
  type SchemaDifference,
} from './gather.js';
export {
  DocumentSession,
  draftStanding,
  forgetDraft,
  hasOverrides,
  readSidecar,
  sidecarOf,
  type DraftStanding,
  type Saved,
  type SessionOptions,
} from './session.js';
export {
  FIRST_VERSION,
  nameMember,
  newDocument,
  fixedTag,
  tagOf,
  versionMember,
  type NewDocument,
} from './skeleton.js';
export { crc32, dosStamp, zipOf, ZipError, type ZipEntry } from './zip.js';
