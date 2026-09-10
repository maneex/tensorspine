/**
 * What one engine reports back — the shape the page produces, the runner collects and
 * `engines.json` records. It names no DOM and no Node API, so the page, the server and the
 * runner all read it.
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
  /** Bytes the case caused to be read — of a file, or over the network. */
  readonly bytesRead?: number;
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
  /** What the engine offers of the platform this spike and feature 0.6 depend on. */
  readonly capabilities: Record<string, boolean>;
  readonly cases: readonly CaseReport[];
}

/**
 * The cases, and what each one proves. The runner and the Playwright specification both name
 * them from here, so a case cannot be renamed in one place alone.
 */
export const CASES = {
  /** A small synthetic file: the header is parsed exactly, and only its bytes are read. */
  syntheticBlob: 'synthetic-blob',
  /** A 256 MiB synthetic file: reading the header still costs the header. */
  syntheticLarge: 'synthetic-large',
  /** A file on disk through the Origin Private File System, sliced the same way. */
  opfsFile: 'opfs-file',
  /** A file the user picked, from a file input — a real checkpoint shard when there is one. */
  localFile: 'local-file',
  /** The Hub, one unsharded repository, through `@huggingface/hub` (network). */
  hubSingle: 'hub-single',
  /** The Hub, a sharded repository: the index and every shard's header (network). */
  hubSharded: 'hub-sharded',
  /** The Hub again, with the Xet transport off: plain HTTP range requests (network). */
  hubRange: 'hub-range',
} as const;

/** The cases that need no network. */
export const OFFLINE_CASES: readonly string[] = [CASES.syntheticBlob, CASES.syntheticLarge, CASES.opfsFile];

/** The cases that reach the Hub. */
export const NETWORK_CASES: readonly string[] = [CASES.hubSingle, CASES.hubSharded, CASES.hubRange];
