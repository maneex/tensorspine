/**
 * One open document — feature 2.6, plan §4.3.
 *
 * A tab of the editor area is one of these: the document's tree (the store of feature 2.1, which
 * is the only model there is — D1), its layout sidecar (D6), where it came from in the workspace,
 * and the two revisions that decide whether it is dirty and whether the file moved under it.
 *
 * **What "dirty" is.** The store counts the states a document has had (§5.4's freshness counter);
 * a session remembers the count the file was written at. Equal counts mean saved, and an undo
 * back to the saved count is saved again — which is what an editor's dot has to mean, and what a
 * comparison of texts would give too, at the cost of rendering the document on every keystroke.
 * A document never written has no saved count at all and is dirty from birth, which is exactly
 * what New Model makes (§4.3).
 *
 * **What a save writes.** The document, through the core's serializer (D12) — `DocumentStore.text`
 * and nothing else — and beside it the layout sidecar, *when the layout has overrides* (§4.3:
 * "Save also writes the sidecar `<model>.layout.json` (D6) when the layout has overrides"). A
 * sidecar with nothing in it is not written, so a workspace does not fill with empty files.
 *
 * **Where the bytes go is the workspace's answer, not this one's.** On a folder the browser can
 * write, `write` writes; on a read-only snapshot or the Examples workspace it downloads and
 * reports the revision unmoved (§5.2). Nothing here branches on which: "no gesture is refused for
 * the platform's reason", and the banner and the button are what say what will happen (S18).
 */
import { isJsonObject, parse, type JsonObject } from '@tensorspine/lang';

import { replaceRoot } from '../commands.js';
import { DocumentStore, type Applied, type DocumentStoreOptions } from '../document.js';
import {
  emptyLayout,
  LayoutStore,
  readLayout,
  writeLayout,
  type Layout,
} from '../layout.js';
import { EditError } from '../path.js';
import { byteLength } from '../platform/readonly.js';
import { ABSENT, type Draft, type DraftStore, type Workspace } from '../platform/types.js';
import { nameOf, normalise, type WorkspacePath } from '../platform/paths.js';
import type { SchemaShapes } from '../shape.js';

/** The suffix the layout sidecar of a document carries (§5.5). */
const LAYOUT_SUFFIX = '.layout.json';

/** The suffix a document file carries, which the sidecar's name replaces. */
const JSON_SUFFIX = '.json';

/** `<model>.layout.json` beside `<model>.json` (D6, §5.5). */
export function sidecarOf(path: WorkspacePath): WorkspacePath {
  const at = normalise(path);
  return at.endsWith(JSON_SUFFIX) ? `${at.slice(0, -JSON_SUFFIX.length)}${LAYOUT_SUFFIX}` : `${at}${LAYOUT_SUFFIX}`;
}

/** Whether a layout is worth a file: §4.3's "when the layout has overrides". */
export function hasOverrides(layout: Layout): boolean {
  return (
    Object.keys(layout.positions).length > 0 ||
    layout.collapsed.length > 0 ||
    layout.viewport !== undefined ||
    layout.preview_assignment !== undefined ||
    layout.expanded_view !== undefined
  );
}

/** What a session is opened with. */
export interface SessionOptions extends DocumentStoreOptions {
  /** Which workspace the document belongs to — {@link WorkspaceRef.id}, the draft's key. */
  readonly workspace: string;
  /** Where it is, or where it will be: a New Model has a proposed path from the start. */
  readonly path: WorkspacePath;
  /** The revision the file was read at; {@link ABSENT} for a document not in the workspace yet. */
  readonly revision?: string;
  /** The layout sidecar, where one was read. */
  readonly layout?: Layout;
  /** False for a document the workspace never held — New Model, and a restored draft. */
  readonly saved?: boolean;
}

/** What a save did, for the toast and for the log. */
export interface Saved {
  readonly path: WorkspacePath;
  /** The revision the file is at now; unmoved where the workspace could only download. */
  readonly revision: string;
  readonly bytes: number;
  /** The sidecar written beside it, when the layout had overrides. */
  readonly sidecar?: WorkspacePath;
}

/** One open document: its tree, its layout, its place, and what has been written of it. */
export class DocumentSession {
  readonly store: DocumentStore;
  readonly layout: LayoutStore;
  readonly workspace: string;

  private at: WorkspacePath;
  private fileRevision: string;
  private savedAt: number | null;
  private layoutSavedAt: number | null = null;

  constructor(tree: JsonObject, shapes: SchemaShapes, options: SessionOptions) {
    const { workspace, path, revision, layout, saved, ...store } = options;
    this.store = new DocumentStore(tree, shapes, store);
    this.layout = new LayoutStore(layout ?? emptyLayout());
    this.workspace = workspace;
    this.at = normalise(path);
    this.fileRevision = revision ?? ABSENT;
    this.savedAt = saved === false ? null : this.store.revision;
    this.layoutSavedAt = saved === false ? null : this.layout.revision;
  }

  /** Open a document from the text the workspace answered (D12's parser). */
  static open(text: string, shapes: SchemaShapes, options: SessionOptions): DocumentSession {
    return new DocumentSession(rootOf(text), shapes, options);
  }

  /** Where the document is, or where a New Model proposes to be. */
  get path(): WorkspacePath {
    return this.at;
  }

  /** The file's own name — the download's name, and what a listing shows. */
  get name(): string {
    return nameOf(this.at);
  }

  /** The revision the workspace last answered for this file. */
  get revision(): string {
    return this.fileRevision;
  }

  /** Whether the document has changes the workspace has not been given (§4.3's ●). */
  get dirty(): boolean {
    return this.savedAt !== this.store.revision || this.layoutDirty;
  }

  /** Whether the sidecar has changes the workspace has not been given. */
  get layoutDirty(): boolean {
    return hasOverrides(this.layout.layout) && this.layoutSavedAt !== this.layout.revision;
  }

  /** The document as its file holds it: the core's serializer and nothing else (D12). */
  get text(): string {
    return this.store.text;
  }

  /**
   * Save, and the sidecar with it when the layout has overrides.
   *
   * `expect` is the revision the file was last read or written at, which is §5.2's optimistic
   * concurrency: a file changed behind the editor is a `conflict` that changed nothing, and the
   * caller offers the reload. A document the workspace never held is written with {@link ABSENT},
   * so a Save As over an existing file is a conflict rather than a silent overwrite.
   */
  async save(workspace: Workspace, into?: WorkspacePath): Promise<Saved> {
    const path = into === undefined ? this.at : normalise(into);
    const moving = path !== this.at;
    const text = this.text;
    const revision = this.store.revision;
    const expect = moving ? ABSENT : this.fileRevision;
    const written = await workspace.write(path, text, expect);
    this.at = path;
    this.fileRevision = written.revision;
    this.savedAt = revision;
    const saved: Saved = { path, revision: written.revision, bytes: byteLength(text) };
    if (!hasOverrides(this.layout.layout)) return saved;
    const sidecar = sidecarOf(path);
    const at = this.layout.revision;
    await workspace.write(sidecar, writeLayout(this.layout.layout));
    this.layoutSavedAt = at;
    return { ...saved, sidecar };
  }

  /**
   * Take the document back to what the file holds — §4.4's Revert, as an undoable command (D13).
   *
   * A revert that threw the history away would be the one edit the Edit menu could not take back,
   * which is the opposite of what a revert is for. So it is a command like any other: the members
   * of the tree are replaced, the patches are the log's, and `Undo` says "Revert".
   */
  revert(text: string, revision: string): Applied {
    const applied = this.replace(text, 'Revert');
    this.fileRevision = revision;
    this.savedAt = this.store.revision;
    return applied;
  }

  /**
   * Put an autosaved draft back — §4.3's "the editor offers to restore it".
   *
   * The same edit as a revert and the opposite bookkeeping: a restored draft is *unsaved work*,
   * so the document stays dirty and the next Save is what puts it in the file. Undoable, like
   * every other command (D13), so a restore the user did not want costs one Ctrl+Z.
   */
  restore(text: string): Applied {
    return this.replace(text, 'Restore the draft');
  }

  /** Replace every member of the tree, as one named command (`replaceRoot`, `../commands.ts`). */
  private replace(text: string, label: string): Applied {
    return this.store.apply(replaceRoot(rootOf(text), label));
  }

  /** Give the session a new place without writing anything — a Save As that downloaded. */
  moveTo(path: WorkspacePath, revision: string = ABSENT): void {
    this.at = normalise(path);
    this.fileRevision = revision;
  }

  /** The draft this document would be autosaved as (§4.3, every 30 s and on blur). */
  draftOf(savedAt: Date = new Date()): Draft {
    return {
      workspace: this.workspace,
      path: this.at,
      text: this.text,
      revision: this.fileRevision,
      savedAt: savedAt.toISOString(),
    };
  }
}

/** Read a layout sidecar beside a document, or the empty one where there is none. */
export async function readSidecar(
  workspace: Workspace,
  path: WorkspacePath,
): Promise<Layout> {
  try {
    return readLayout((await workspace.read(sidecarOf(path))).text);
  } catch {
    // A document with no sidecar is the ordinary case, and a sidecar that cannot be parsed is
    // the editor's own file: neither may stop a document from opening (D6).
    return emptyLayout();
  }
}

/**
 * What a draft is worth beside the file it was taken from — §4.3's "a document with a newer
 * draft the editor offers to restore".
 *
 * `null` where the file already holds the draft's own text, which is the ordinary case after a
 * save and is the one draft never offered. `unsaved` where the draft was taken against the
 * revision the file is still at: work the file has not got. `behind` where the file has moved
 * since — somebody wrote it outside the editor, or another tab did — which is still offered,
 * because throwing a user's typing away silently is the one thing an autosave must never do, and
 * still distinguished, because the dialog has to say which of the two it is.
 */
export type DraftStanding = 'unsaved' | 'behind';

/** How a draft stands against the file (§4.3). */
export function draftStanding(draft: Draft, text: string, revision: string): DraftStanding | null {
  if (draft.text === text) return null;
  return draft.revision === revision ? 'unsaved' : 'behind';
}

/** Drop the draft a saved document no longer needs. */
export async function forgetDraft(drafts: DraftStore, session: DocumentSession): Promise<void> {
  await drafts.remove(session.workspace, session.path);
}

/** The root of a document's text, refused where the file is not a JSON object at all. */
function rootOf(text: string): JsonObject {
  const tree = parse(text);
  if (!isJsonObject(tree)) throw new EditError('the document is not a JSON object');
  return tree;
}
