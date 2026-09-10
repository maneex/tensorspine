/**
 * What one engine reports back — feature 0.6.
 *
 * The shape the page produces, the server collects and `engines.json` records. It names no DOM
 * and no Node API, so the page, the server, the runner and the Playwright specification all
 * read it; a case cannot be renamed in one place alone.
 */

/** One case, run or not run, in the engine that ran it. */
export interface CaseReport {
  /** The case's name, as {@link CASES} lists them. */
  readonly name: string;
  /** True when the case did what it claims; false when it refused or failed an expectation. */
  readonly ok: boolean;
  /** Set when the engine or the environment cannot offer what the case needs. */
  readonly skipped?: string;
  /** Milliseconds the case took, as the page measured it. */
  readonly ms: number;
  /** Whatever the case measured, for the note's tables. */
  readonly detail?: Record<string, unknown>;
  /** The refusal, in the words the engine gave it. */
  readonly error?: string;
}

/** One engine's run of the cases it was asked for. */
export interface EngineReport {
  /** The name the runner gave the engine (`chromium`, `firefox`, `webkitgtk`). */
  readonly engine: string;
  readonly userAgent: string;
  readonly startedAt: string;
  /** The base path the page was served under, as the page sees it. */
  readonly base: string;
  /** What the engine offers of the platform the two workspace paths stand on. */
  readonly capabilities: Record<string, boolean>;
  readonly cases: readonly CaseReport[];
}

/**
 * The cases, and what each one proves.
 *
 * The writable path is exercised through an Origin Private File System directory handle: the
 * same `FileSystemDirectoryHandle` API a picked folder gives, without the native picker, which
 * no automated browser can drive (plan §0.6's own instruction).
 */
export const CASES = {
  /** The page was served under a base path, and its own module came from under it. */
  basePath: 'base-path',
  /** A directory handle: `mkdir`, `write`, `list`, `read`, `resolve`, and the concurrency check. */
  directoryRoundTrip: 'directory-round-trip',
  /** `watch` by polling sees a change made behind the workspace's back, and stops when told. */
  directoryWatch: 'directory-watch',
  /** A revision is a modification time and a size: can two writes of one length be told apart? */
  revisionResolution: 'revision-resolution',
  /** A directory handle survives IndexedDB and a reload, and its permission is queried once. */
  handlePersistence: 'handle-persistence',
  /** A folder upload — `webkitdirectory` — read as a read-only snapshot. */
  snapshotUpload: 'snapshot-upload',
  /** Save on a snapshot: a download with the document's own bytes. */
  snapshotSave: 'snapshot-save',
} as const;

/** Every case a page can run on its own, in the order the note reads them. */
export const PAGE_CASES: readonly string[] = [
  CASES.basePath,
  CASES.directoryRoundTrip,
  CASES.directoryWatch,
  CASES.revisionResolution,
  CASES.handlePersistence,
  CASES.snapshotUpload,
  CASES.snapshotSave,
];
