/**
 * The command log: named edits, Immer patches, inverse patches, undo, redo and the revision
 * counter (plan D13, §5.4).
 *
 * > Every gesture is a named command ("Add instance attn", "Connect attn.output → attn_r.b",
 * > "Set heads", "Add state port kv") with forward and inverse patches; the Edit menu names
 * > them. Layout moves are recorded in a separate log so an undo of a semantic edit does not
 * > shuffle positions.
 *
 * So the log is generic in its state: the document store holds one over the tree, the layout
 * store holds one over the sidecar, and neither can undo the other's edits. What a command *is*
 * — which JSON edit each gesture makes — is `./commands.ts`; this module only records.
 *
 * **The revision counter** is the number of states this log has had. Plan §5.4 tracks freshness
 * per product against it: "anything computed for an older revision is shown dimmed with
 * `stale`". An undo is a new state and therefore a new revision — the products of the state it
 * restores were computed against a revision that will never come back, and calling them fresh
 * would be showing a figure for a document that is not on screen.
 *
 * **The redo stack is dropped by a new edit**, which is the only behaviour that keeps a linear
 * history honest: the states it held can no longer be reached from here.
 */
import { applyPatches, enablePatches, produceWithPatches, type Patch } from 'immer';

// Immer's patch machinery is a plugin, and `produceWithPatches` raises without it. Enabling it
// at the module's load is what makes every log in the process able to record; the call is
// idempotent.
enablePatches();

/** What an applied edit answers: its name, the patches it made, and the state it left. */
export interface Edit<T> {
  /** The Edit menu's wording: "Add instance attn", "Rename quantity d to width". */
  readonly label: string;
  /** The state after the edit. */
  readonly state: T;
  /** The revision the edit produced. */
  readonly revision: number;
  /** Immer's forward patches. */
  readonly patches: readonly Patch[];
  /** Immer's inverse patches: what an undo applies. */
  readonly inverse: readonly Patch[];
  /** Whether the state changed at all — a recipe that writes nothing answers `false`. */
  readonly changed: boolean;
}

/** One entry of the history: the two patch sets under the name of the command that made them. */
interface Entry {
  readonly label: string;
  readonly patches: readonly Patch[];
  readonly inverse: readonly Patch[];
}

/** A listener of the log, called after every state change. */
export type Listener<T> = (state: T, revision: number) => void;

/** How far back a log remembers. */
export interface LogOptions {
  /**
   * The number of undoable edits kept. The oldest is dropped beyond it, and the plan sets no
   * figure, so the default is generous and the knob exists for the editor to set per document.
   */
  readonly depth?: number;
}

const DEFAULT_DEPTH = 200;

/**
 * A log of named edits over one immutable state.
 *
 * The state is never mutated: every edit runs its recipe on an Immer draft and answers a new
 * frozen state, so a view holding an older one keeps a consistent reading of it. `D` is the
 * draft's type, which is `T` with the `readonly` taken off — `./draft.ts` says why the document
 * tree's mirror is written by hand rather than taken from Immer's `Draft<T>`.
 */
export class EditLog<T, D = T> {
  private current: T;
  private count = 0;
  private readonly done: Entry[] = [];
  private readonly undone: Entry[] = [];
  private readonly listeners = new Set<Listener<T>>();
  private readonly depth: number;

  constructor(initial: T, options: LogOptions = {}) {
    this.current = initial;
    this.depth = options.depth ?? DEFAULT_DEPTH;
  }

  /** The state as it stands. */
  get state(): T {
    return this.current;
  }

  /** The number of states this log has had: 0 for the one it was built with. */
  get revision(): number {
    return this.count;
  }

  /** Whether there is an edit to undo. */
  get canUndo(): boolean {
    return this.done.length > 0;
  }

  /** Whether there is an edit to redo. */
  get canRedo(): boolean {
    return this.undone.length > 0;
  }

  /** The name of the edit an undo would take back — what the Edit menu shows. */
  get undoLabel(): string | null {
    return this.done[this.done.length - 1]?.label ?? null;
  }

  /** The name of the edit a redo would make again. */
  get redoLabel(): string | null {
    return this.undone[this.undone.length - 1]?.label ?? null;
  }

  /** The names of the edits made, oldest first: the history a panel lists. */
  get history(): readonly string[] {
    return this.done.map((entry) => entry.label);
  }

  /**
   * Run a named recipe over the state.
   *
   * A recipe that changes nothing is not recorded: an undo must take back an edit the user can
   * see, and "Set heads" to the value it already had is not one.
   */
  apply(label: string, recipe: (draft: D) => void): Edit<T> {
    const [next, patches, inverse] = produceWithPatches<T, D>(this.current, recipe);
    if (patches.length === 0) {
      return {
        label,
        state: this.current,
        revision: this.count,
        patches,
        inverse,
        changed: false,
      };
    }
    this.current = next;
    this.count += 1;
    this.done.push({ label, patches, inverse });
    if (this.done.length > this.depth) this.done.shift();
    this.undone.length = 0;
    this.announce();
    return { label, state: next, revision: this.count, patches, inverse, changed: true };
  }

  /** Take back the last edit. Answers what was undone, or `null` when there was nothing. */
  undo(): Edit<T> | null {
    const entry = this.done.pop();
    if (entry === undefined) return null;
    this.current = this.applied(entry.inverse);
    this.count += 1;
    this.undone.push(entry);
    this.announce();
    return {
      label: entry.label,
      state: this.current,
      revision: this.count,
      patches: entry.inverse,
      inverse: entry.patches,
      changed: true,
    };
  }

  /** Make the last undone edit again. */
  redo(): Edit<T> | null {
    const entry = this.undone.pop();
    if (entry === undefined) return null;
    this.current = this.applied(entry.patches);
    this.count += 1;
    this.done.push(entry);
    this.announce();
    return {
      label: entry.label,
      state: this.current,
      revision: this.count,
      patches: entry.patches,
      inverse: entry.inverse,
      changed: true,
    };
  }

  /** Be told after every state change. Answers the way to stop being told. */
  subscribe(listener: Listener<T>): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  private applied(patches: readonly Patch[]): T {
    // `applyPatches` is typed over Immer's own `Objectish`, which an interface with named
    // members is not assignable to (TypeScript gives an implicit index signature to a type
    // literal and not to an interface). The state is a plain object either way; the cast is
    // confined to this line.
    return applyPatches(this.current as never, patches);
  }

  private announce(): void {
    for (const listener of this.listeners) listener(this.current, this.count);
  }
}

export type { Patch };
