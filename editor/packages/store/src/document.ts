/**
 * The document store: one tree, one history, one revision counter (plan D1, D13, §5.4).
 *
 * > The editor's store holds the `tensorspine/2.0` JSON (as a tree, with member order and number
 * > lexemes), and the canvas, the sheets, the explorer and the source view are projections of
 * > it. A gesture on the canvas is a JSON edit; the JSON editor is another view of the same
 * > tree.
 *
 * So there is nothing else here: no graph model beside the tree, no copy of an instance's
 * arguments, no second reading of anything the core answers. What the store adds to the tree is
 * the *history* — a command log of Immer patches, `./log.ts` — the *revision* every product's
 * freshness is measured against, the *reference index* a rename and a delete are computed from,
 * and the text the file is saved as, which is the core's serializer and nothing else (D12).
 *
 * The index and the text are built when they are asked for and kept until the next edit: a
 * keystroke in the sheet must not pay for an index nobody reads, and a save must not re-render a
 * document that has not changed.
 */
import { isJsonObject, parse, serialize, type JsonObject } from '@tensorspine/lang';

import type { Cascade, Command, EditContext, PathMove } from './commands.js';
import type { MutableObject } from './draft.js';
import { EditLog, type Listener, type LogOptions } from './log.js';
import { EditError, type Path } from './path.js';
import { referenceIndexWith, referenceTags, type ReferenceIndex, type ReferenceTags } from './references.js';
import { SchemaShapes } from './shape.js';

/** The role a `tensorspine/2.0` document is read under, as the registry indexes the schema. */
export const MODEL_ROLE = 'model';

/** What an applied gesture answers. */
export interface Applied {
  /** The Edit menu's wording for what was done. */
  readonly label: string;
  /** The revision the edit produced; `stale` is measured against it (§5.4). */
  readonly revision: number;
  /** Whether the tree changed at all. */
  readonly changed: boolean;
  /** The places that moved, for the layout sidecar (plan §3, D6). */
  readonly moves: readonly PathMove[];
  /** A delete's listing: what went, and what the grammar would not let go. */
  readonly cascade?: Cascade;
  /** The binding a connection replaced (V7, Q5 — the toast names it). */
  readonly replaced?: Path;
}

/** How a document store is built. */
export interface DocumentStoreOptions extends LogOptions {
  /** The role the document is read under; `model` unless a caller says otherwise. */
  readonly role?: string;
}

/** The store of one open document. */
export class DocumentStore {
  private readonly log: EditLog<JsonObject, MutableObject>;
  private readonly tags: ReferenceTags;
  private indexAt = -1;
  private indexed: ReferenceIndex | null = null;
  private textAt = -1;
  private rendered: string | null = null;

  readonly shapes: SchemaShapes;
  readonly role: string;

  constructor(tree: JsonObject, shapes: SchemaShapes, options: DocumentStoreOptions = {}) {
    this.shapes = shapes;
    this.role = options.role ?? MODEL_ROLE;
    this.log = new EditLog<JsonObject, MutableObject>(tree, options);
    this.tags = referenceTags(shapes, this.role);
  }

  /**
   * Open a document from its text, with the core's parser (feature 0.3).
   *
   * A text that is not an object is refused here rather than later: every gesture of §4.7 edits a
   * member of the document's root, and a root that is not one is a file the editor was handed by
   * mistake. A text that is not JSON at all raises the core's own `JsonParseError`, in CPython's
   * words, which is what the source view shows.
   */
  static open(text: string, shapes: SchemaShapes, options: DocumentStoreOptions = {}): DocumentStore {
    const tree = parse(text);
    if (!isJsonObject(tree)) throw new EditError('the document is not a JSON object');
    return new DocumentStore(tree, shapes, options);
  }

  /** The document as it stands. */
  get tree(): JsonObject {
    return this.log.state;
  }

  /** The number of states this document has had (§5.4's freshness). */
  get revision(): number {
    return this.log.revision;
  }

  /** The document as its file holds it: the core's serializer, the corpus's bytes (D12). */
  get text(): string {
    if (this.textAt !== this.log.revision || this.rendered === null) {
      this.rendered = serialize(this.log.state);
      this.textAt = this.log.revision;
    }
    return this.rendered;
  }

  /** Every place the document writes a name, for this revision. */
  get index(): ReferenceIndex {
    if (this.indexAt !== this.log.revision || this.indexed === null) {
      this.indexed = referenceIndexWith(this.log.state, this.shapes, this.role, this.tags);
      this.indexAt = this.log.revision;
    }
    return this.indexed;
  }

  /** What the schemas say about names — the same for every document of this role. */
  get referenceTags(): ReferenceTags {
    return this.tags;
  }

  /** What a command is computed against. */
  get context(): EditContext {
    return { tree: this.tree, shapes: this.shapes, role: this.role, index: this.index };
  }

  get canUndo(): boolean {
    return this.log.canUndo;
  }

  get canRedo(): boolean {
    return this.log.canRedo;
  }

  /** The name of the edit an undo would take back — the Edit menu's line (D13). */
  get undoLabel(): string | null {
    return this.log.undoLabel;
  }

  /** The name of the edit a redo would make again. */
  get redoLabel(): string | null {
    return this.log.redoLabel;
  }

  /** The edits made, oldest first. */
  get history(): readonly string[] {
    return this.log.history;
  }

  /**
   * Make a gesture.
   *
   * The command was computed against {@link context} — against this tree, at this revision — so
   * a caller that holds one across an edit is holding a plan for a document that has moved. That
   * is what the confirmation of a delete does (it is shown, then applied), and the store checks
   * nothing: an edit computed elsewhere still applies, and Immer's patches describe what it did.
   */
  apply(command: Command): Applied {
    const edit = this.log.apply(command.label, command.edit);
    return {
      label: edit.label,
      revision: edit.revision,
      changed: edit.changed,
      // A gesture that wrote nothing moved nothing: a caller that follows the moves must not be
      // told to move a key for an edit the document did not take.
      moves: edit.changed ? command.moves : [],
      ...(command.cascade === undefined ? {} : { cascade: command.cascade }),
      ...(command.replaced === undefined ? {} : { replaced: command.replaced }),
    };
  }

  /** Build a command against the current tree and make it in one step. */
  run(make: (context: EditContext) => Command): Applied {
    return this.apply(make(this.context));
  }

  /**
   * Take back the last edit.
   *
   * The moves are not undone here: the layout has its own log (D13), and an undo of a semantic
   * edit must not shuffle positions. A caller that wants the sidecar to follow a rename back
   * asks the layout store for its own undo.
   */
  undo(): Applied | null {
    const edit = this.log.undo();
    if (edit === null) return null;
    return { label: edit.label, revision: edit.revision, changed: edit.changed, moves: [] };
  }

  /** Make the last undone edit again. */
  redo(): Applied | null {
    const edit = this.log.redo();
    if (edit === null) return null;
    return { label: edit.label, revision: edit.revision, changed: edit.changed, moves: [] };
  }

  /** Be told after every edit: what the pipeline of §5.4 hangs off. */
  subscribe(listener: Listener<JsonObject>): () => void {
    return this.log.subscribe(listener);
  }
}
