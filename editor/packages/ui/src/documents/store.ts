/**
 * Open and save — feature 2.6, plan §4.3 and §4.4's File menu.
 *
 * The shell (2.5) owns the chrome and holds nothing of a document; this owns the documents and
 * holds nothing of the chrome. Between them: a tab. The shell is told to open one, to rename it,
 * to mark it dirty and to close it, and it asks this one before a close whether the document
 * behind it may go (§4.3: "Close with unsaved changes asks").
 *
 * **What is here, and why here.** Everything §4.3 names that needs a workspace, a core and a
 * clock: Open Folder…, a dropped folder, Open Examples, the recents, tabs with their dirty state,
 * Save through the core's serializer (D12), Save As, Save All, Download Workspace as Zip, Revert,
 * the autosave and its restore, the sidecar written with overrides, and the close prompts. The
 * reading of a document — its tree, its history, its text — is the store of 2.1; the reading of
 * the workspace is `Platform` (§5.2); the verdicts and the products are the core's (§5.3). This
 * is the wiring, and it computes nothing of its own.
 *
 * **Nothing is opened at launch but what was open before.** D11: "its handle kept in IndexedDB so
 * the workspace reopens after a reload with one permission prompt" — `reopenGranted()` asks for
 * the folder the browser still holds a grant for, with no gesture and no prompt, and answers
 * nothing where there is none. A page that opened the Examples workspace by itself would be a
 * page that decided for the user; S17's empty state is what an editor with nothing open shows.
 */
import type {
  Draft,
  Platform,
  RecentWorkspace,
  UploadedFile,
  Workspace,
  WorkspaceRef,
} from '@tensorspine/store/platform';
import { ABSENT, PlatformError } from '@tensorspine/store/platform';
import {
  DocumentSession,
  draftStanding,
  FIRST_VERSION,
  fixedTag,
  gatherBases,
  gatherSchemas,
  isUnder,
  listTree,
  nameMember,
  newDocument,
  pointerOf,
  readSidecar,
  readTree,
  schemaDifferences,
  SchemaShapes,
  sidecarOf,
  tagOf,
  writeLayout,
  zipOf,
  type Applied,
  type Command,
  type DraftStanding,
  type EditContext,
  type Path,
  type Shape,
  type WorkspacePath,
  type ZipEntry,
} from '@tensorspine/store';
import {
  BASE_MANIFEST,
  isJsonObject,
  loadSchemas,
  parse,
  templatePrimitives,
  type SchemaRegistry,
} from '@tensorspine/lang';
import type { Lang, LibraryHandle, Problem, SchemasHandle } from '@tensorspine/lang/api';
import { createStore, type StoreApi } from 'zustand/vanilla';

import { outlineOf, revealing } from '../explorer/outline.js';
import { presentation, startPresentation, type Presentation } from '../presentation/index.js';
import {
  droppedKeys,
  schemaMismatches,
  unusedQuantities,
  type DroppedSidecarKey,
} from '../problems/notices.js';
import { statusFigures, type FigureShapes, type StatusFigure } from './figures.js';
import { noReading, Pipeline, type Reading } from './pipeline.js';
import { shapesFor } from './shapes.js';

/** The setting the last workspace is remembered under, so a reload comes back to it (§4.3). */
export const WORKSPACE_SETTING = 'documents.workspace';

/** The setting the open tabs are remembered under — §4.3's "tabs restore on relaunch". */
export const TABS_SETTING = 'documents.tabs';

/** How often a dirty document is autosaved (§4.3: "every 30 s and on blur"). */
export const AUTOSAVE_MS = 30_000;

/** How a document reached the editor, for the log and for what a Save will do. */
export interface OpenDocument {
  /** The tab's identity — the workspace and the path, which is what makes it unique. */
  readonly id: string;
  readonly path: WorkspacePath;
  /** The workspace it belongs to ({@link WorkspaceRef.id}) — the draft's key. */
  readonly workspace: string;
  /** What the tab is called: §4.3's "a tab named after `model`". */
  readonly title: string;
  readonly dirty: boolean;
  /** `template` where the core says the document needs an assignment (§4.3, §4.6). */
  readonly badge?: string;
  /**
   * The revision tag the document declares — §4.2's second status-bar field.
   *
   * Read off the tree at the member the *schema* fixes (`tagOf`), so the bar shows what the
   * document says rather than what a component remembers (feature 2.1's refusal, kept).
   */
  readonly tag?: string;
  /** The session behind it: the tree, the history, the layout. */
  readonly session: DocumentSession;
  /** What the pipeline has said about it, and at which revision. */
  readonly reading: Reading;
  /** The status bar's figures, from the derived document and the presentation bindings. */
  readonly figures: readonly StatusFigure[];
  /** A draft found for it, offered rather than applied (§4.3). */
  readonly draft?: { readonly draft: Draft; readonly standing: DraftStanding };
  /**
   * What is selected *in* the document, as a place of it — §4.5's "selecting an item selects it
   * on the canvas and in Properties".
   *
   * A path and nothing else, because D1 leaves nothing else to hold: the canvas, the sheets and
   * the explorer are projections of one tree, so a selection is a place in that tree and every
   * projection reads it. It lives per open document, since a tab keeps what was selected in it.
   */
  readonly selection?: Path;
  /**
   * The outline rows whose openness the reader changed, by pointer (§4.5).
   *
   * A *deviation* from the default and not the open set: the default is the document's own shape
   * — a non-empty map of its own names is open — so a composition added after the fact opens like
   * the others rather than staying shut because nobody had said anything about it.
   */
  readonly toggled: readonly string[];
  /**
   * The sidecar keys this document's edits dropped (D6), for §4.17's `editor` notices.
   *
   * A drop is an *event* — "a sidecar key that names a path the document no longer has is dropped
   * with a log line" — so it cannot be read back off the tree the way an unused quantity can; it
   * is kept here and the panel shows it until the document is closed.
   */
  readonly dropped: readonly DroppedSidecarKey[];
}

/** How the library and the schemas stand for the open workspace. */
export interface LibraryState {
  readonly loading: boolean;
  /** The handle every call names; `null` before the first document resolved one. */
  readonly handle: LibraryHandle | null;
  /** The loader's refusals, in the tools' order. */
  readonly problems: readonly Problem[];
  /** How many files were gathered, and how long it took — the log's line and X.2's figure. */
  readonly files: number;
  readonly ms: number;
  /**
   * The primitives that pin a template, by name (§4.5's `▣`, §4.6's own badge).
   *
   * The names and not the library: what the outline needs of a gathered library is which
   * primitives are templates, and `primitive_library.template_primitives` is the core's own
   * answer to that. Whether the page keeps the whole library beside the worker's copy is the
   * library activity's decision (3.1), not this one's.
   */
  readonly templates: ReadonlySet<string>;
}

/** A banner the chrome shows (component inventory §5: the dropped-folder snapshot, and its kin). */
export interface BannerLine {
  readonly kind: 'stop' | 'info' | 'warn';
  readonly head: string;
  readonly body: string;
}

/** A toast — "a replaced edge, naming what went, undoable" (inventory §5), and what a Save says. */
export interface ToastLine {
  readonly text: string;
  /** The label of the action beside it, where there is one. */
  readonly action?: string;
  readonly run?: () => void;
}

/**
 * Which dialog is up, if any: the workspace picker, the documents of the open workspace, a Save
 * As, a draft to restore.
 */
export type Dialog =
  | { readonly kind: 'workspaces'; readonly recent: readonly RecentWorkspace[] }
  | { readonly kind: 'documents' }
  | { readonly kind: 'save-as'; readonly id: string; readonly path: WorkspacePath }
  | { readonly kind: 'restore'; readonly id: string }
  | {
      readonly kind: 'remove';
      readonly id: string;
      /** The Edit menu's wording for the command waiting to be made. */
      readonly label: string;
      /** What the cascade would remove and what the grammar keeps (§4.7's confirmation). */
      readonly removed: readonly string[];
      readonly kept: readonly string[];
    };

/** The documents as the chrome reads them. */
export interface DocumentsState {
  readonly workspace: WorkspaceRef;
  readonly open: readonly OpenDocument[];
  readonly current: string | null;
  readonly library: LibraryState;
  readonly banner: BannerLine | null;
  readonly toast: ToastLine | null;
  readonly dialog: Dialog | null;
  /** Every `.json` of the workspace that is not under a library base — what Open Model… offers. */
  readonly documents: readonly WorkspacePath[];
  /**
   * The editor's own rows about the **workspace** — §4.17's `editor` source.
   *
   * A schema the workspace carries that the build's own does not match is one (§1). They are the
   * workspace's and not a document's, so they stand whichever tab is open.
   */
  readonly notices: readonly Problem[];
  /**
   * The Model explorer's filter box (§4.5), kept here rather than in the component.
   *
   * The side bar is unmounted whenever another activity is shown, and a filter that was typed and
   * then lost because the reader looked at the library is a filter they have to type again. One
   * box and one string: the explorer filters whichever document is current.
   */
  readonly filter: string;
}

/** The documents, and the gestures of §4.3 on them. */
export interface Documents extends DocumentsState {
  /** What the editor does with no gesture at all: D11's remembered folder, and the tabs. */
  start(): Promise<void>;
  adopt(workspace: Workspace | null): Promise<void>;
  openFolder(): Promise<void>;
  openExamples(): Promise<void>;
  openRecent(id: string): Promise<void>;
  openUpload(files: readonly UploadedFile[]): Promise<void>;
  /** The second half of a drop: the workspace the handler already asked the platform for. */
  dropped(opening: Promise<Workspace | null>): Promise<void>;
  forget(id?: string): Promise<void>;

  openDocument(path: WorkspacePath): Promise<void>;
  newModel(template?: boolean): Promise<void>;
  select(id: string): void;
  /** Whether a tab may close — the shell asks before it removes one (§4.3). */
  mayClose(id: string): Promise<boolean>;
  closeDocument(id: string): void;

  /**
   * Select a place of the current document, or nothing — §4.5, and the canvas's own gesture (2.9).
   */
  selectPlace(path: Path | null, id?: string): void;
  /** Open or close an outline row, by the pointer of the place it stands for (§4.5). */
  togglePlace(pointer: string, id?: string): void;
  /** What the explorer's filter box holds (§4.5). */
  setFilter(filter: string): void;
  /**
   * Select a place **and open the tree down to it** — what clicking a row of Problems does.
   *
   * `selectPlace` writes the selection and every projection reads it; a place under a group the
   * reader closed is selected and invisible, which is no navigation at all. So this opens the
   * ancestors first, against the outline's own default (2.7's `toggled` is a deviation from it,
   * not the open set), and then selects.
   */
  revealPlace(path: Path, id?: string): void;
  /**
   * `Model ▸ Lint` (§4.4): the advisories, over the workspace's own model documents.
   *
   * The set is this feature's decision and it is the set the repository lints itself with:
   * `--lint`'s answer is a function of it (feature 1.10), a document linted **alone** reports
   * every primitive of the base as called by nobody, and the only set under which the corpus lints
   * clean is the whole of it. It is not part of §5.4's debounced run because linting fourteen
   * documents costs 781 ms — one `analyse` each — against the 300 ms a keystroke's validation is
   * given (§5.6), so it is a command, and its rows stand until it is run again.
   */
  lint(id?: string): void;
  /**
   * Make a gesture on the document — the one way a projection edits the tree (D1, D13).
   *
   * The command is built by the caller against {@link DocumentStore.context}, because what a
   * rename rewrites and what a delete takes with it are read from `presentation.json` and the
   * reference index, which is the interface's half (2.1, 2.2). What is done here is what every
   * gesture owes the rest of the editor: the sidecar's keys follow a rename and are pruned after
   * a delete (D6), the selection follows the place it was on, and the pipeline of §5.4 runs.
   */
  edit(make: (context: EditContext) => Command, id?: string): Applied | null;
  /**
   * Offer a command that has to be confirmed before it is made — §4.4's "Delete (cascades with
   * confirmation)". The dialog says what would go; {@link confirmed} is what makes it.
   */
  offer(make: (context: EditContext) => Command, id?: string): void;
  /** Make the offered command. */
  confirmed(): void;

  save(id?: string): Promise<void>;
  saveAs(id: string, path: WorkspacePath): Promise<void>;
  saveAll(): Promise<void>;
  revert(id?: string): Promise<void>;
  downloadZip(): Promise<void>;

  autosave(): Promise<void>;
  restoreDraft(id: string): void;
  discardDraft(id: string): Promise<void>;

  validateNow(id?: string): void;
  setDialog(dialog: Dialog | null): void;
  setToast(toast: ToastLine | null): void;
  note(line: string): void;
  stop(): void;
}

/** The store, as the React binding and the suites hold it. */
export type DocumentsStore = StoreApi<Documents>;

/** What {@link createDocuments} is given. */
export interface DocumentsOptions {
  readonly platform: Platform;
  readonly lang: Lang;
  /** The schema files the build vendored, read once by whoever knows where they are. */
  readonly vendoredSchemas: () => Promise<Readonly<Record<string, string>>>;
  /** Where a line about what happened goes — the shell's Log tab. */
  readonly log?: (line: string) => void;
  /** Open, close and mark the shell's tabs. */
  readonly tabs?: TabSink;
  /** How long the pipeline waits after an edit; a suite may ask for none. */
  readonly debounceMs?: number;
  /** How often a dirty document is autosaved; zero switches the timer off (a suite's). */
  readonly autosaveMs?: number;
}

/** What the shell gives the documents to open a tab with (§4.2's strip). */
export interface TabSink {
  open(tab: { id: string; title: string; kind: string; badge?: string; dirty?: boolean }): void;
  update(id: string, patch: { title?: string; badge?: string; dirty?: boolean }): void;
  close(id: string): void;
  select(id: string): void;
  /** The tab the strip has current, so a command with no argument knows what it is about. */
  current(): string | null;
}

/** The `kind` a document's tab carries, which the shell draws with the view of the same name. */
export const DOCUMENT_TAB = 'view.document';

/** The same open document with nothing selected: the member goes rather than becoming undefined. */
function withoutSelection(open: OpenDocument): OpenDocument {
  const next: OpenDocument & { selection?: Path } = { ...open };
  delete next.selection;
  return next;
}

/** A tab's identity: one per (workspace, path), so the same file twice is the same tab. */
function tabId(workspace: string, path: WorkspacePath): string {
  return `doc:${workspace}:${path}`;
}

/** What a workspace is remembered as, so that a reload can come back to it. */
interface LastWorkspace {
  readonly id: string;
  readonly kind: string;
}

export function createDocuments(options: DocumentsOptions): {
  store: DocumentsStore;
  dispose: () => void;
} {
  const { platform, lang } = options;
  const settings = platform.settings;
  const bindings: Presentation = presentation();
  const note = (line: string): void => options.log?.(line);
  const tabs: TabSink = options.tabs ?? silentTabs();

  /** Per open document: the pipeline that keeps its reading fresh. */
  const pipelines = new Map<string, Pipeline>();
  /** Per open document: the subscription that follows its edits. */
  const following = new Map<string, () => void>();
  /** The command a confirmation is up for, held out of the state because it is a closure. */
  let pending: { id: string; command: Command } | null = null;
  let registry: SchemaRegistry | null = null;
  let shapes: SchemaShapes | null = null;
  /** The derived schema as the figure walk reads it, built once with the registry it reads. */
  let figureShapes: FigureShapes<Shape> | null = null;
  let schemas: SchemasHandle | null = null;
  let vendored: Readonly<Record<string, string>> | null = null;
  let autosaveTimer: ReturnType<typeof setInterval> | null = null;
  /** Which workspace the page is reading for; a resumption that is not it does nothing (§10). */
  let generation = 0;
  let watching: (() => void) | null = null;

  const store: DocumentsStore = createStore<Documents>()((set, get) => {
    const patch = (id: string, change: (one: OpenDocument) => OpenDocument): void => {
      set((state) => ({ open: state.open.map((one) => (one.id === id ? change(one) : one)) }));
    };

    /** Redraw one document's tab from its session — its name and the dirty dot (§4.3). */
    const refresh = (id: string): void => {
      const one = get().open.find((open) => open.id === id);
      if (one === undefined || shapes === null) return;
      const dirty = one.session.dirty;
      // The tab is named after the document (§4.3), so an edit that changes the document's own
      // name changes the tab — which a restored draft can do, and a rename will.
      const title = titleOf(one.session, shapes);
      if (dirty === one.dirty && title === one.title) return;
      patch(id, (open) => ({ ...open, dirty, title }));
      tabs.update(id, { dirty, title });
    };

    /** What a reading says about the figures, recomputed only when the derivation moved. */
    const figuresOf = (reading: Reading): readonly StatusFigure[] => {
      if (reading.derived === null || figureShapes === null) return [];
      return statusFigures(reading.derived, figureShapes, bindings);
    };

    const publishing =
      (id: string) =>
      (change: (before: Reading) => Reading): void => {
        patch(id, (one) => {
          const reading = change(one.reading);
          const figures = reading.derived === one.reading.derived ? one.figures : figuresOf(reading);
          const badge = reading.verdict?.needsAssignment === undefined ? undefined : TEMPLATE_BADGE;
          if (badge !== one.badge) tabs.update(id, badge === undefined ? {} : { badge });
          return { ...one, reading, figures, ...(badge === undefined ? {} : { badge }) };
        });
      };

    /**
     * Gather the schemas and the bases a document resolves from, once per workspace.
     *
     * Three costs, timed apart because §5.6 budgets one of them ("library load: once per session,
     * ≤ 300 ms, cached") and the other two are the editor's own: reading the schemas and indexing
     * them twice — once in the worker, once here, which is what §5.4's synchronous Ajv and the
     * store's `SchemaShapes` need — and reading the base out of the workspace.
     */
    const library = async (
      workspace: Workspace,
      tree: unknown,
      path: WorkspacePath,
    ): Promise<LibraryHandle | null> => {
      const clock = (): number => Date.now();
      const started = clock();
      set((state) => ({ library: { ...state.library, loading: true } }));
      vendored ??= await options.vendoredSchemas();
      const gathered = await gatherSchemas(workspace, vendored);
      if (schemas === null) {
        const loaded = await lang.loadSchemas(gathered.files, { origin: gathered.origin });
        schemas = loaded.handle;
        registry = loadSchemas(
          Object.entries(gathered.files).map(([file, text]) => ({ path: file, text })),
          { origin: gathered.origin },
        );
        shapes = new SchemaShapes(registry);
        figureShapes = shapesFor(registry);
        // §1's catching rule (a), the startup half: every binding resolved against the schemas
        // that were actually loaded, and everything they leave to the generic widget listed. It
        // *logs* rather than refuses — the repository's own schemas are held to the stricter rule
        // by `tests/audit/presentation.test.ts` — and it belongs here because here is where the
        // registry the bindings resolve against comes into being (feature 2.2 built it and left
        // it unwired).
        startPresentation(registry, note);
        if (gathered.fromWorkspace) {
          const differences = schemaDifferences(gathered.files, vendored);
          for (const difference of differences) note(difference.message);
          // The Log says it happened; the panel says it is still true (§4.17's `editor` rows).
          set({ notices: schemaMismatches(differences) });
        }
      }
      const indexed = clock();
      const bases = await gatherBases(lang, schemas, workspace, tree as never, path);
      for (const problem of bases.problems) note(problem.message);
      const read = clock();
      const loaded = await lang.loadLibrary(bases.bases, schemas, { forDocument: path });
      const ms = clock() - started;
      set({
        library: {
          loading: false,
          handle: loaded.handle,
          problems: loaded.problems,
          files: bases.files,
          ms,
          // `primitive_library.template_primitives`: the core's own reading of which primitives
          // pin a template, which is what the explorer marks an instance of one by (§4.5).
          templates: templatePrimitives(loaded.library),
        },
      });
      note(
        `library: ${String(bases.bases.length)} base(s), ${String(bases.files)} files, ` +
          `${String(loaded.problems.length)} refusal(s), ${String(ms)} ms ` +
          `(schemas ${String(indexed - started)} ms, read ${String(read - indexed)} ms, ` +
          `load ${String(clock() - read)} ms)`,
      );
      return loaded.handle;
    };

    /** The document a gesture is about: the one named, or the tab the strip has current. */
    const current = (id?: string): OpenDocument | undefined => {
      const state = get();
      const wanted = id ?? tabs.current() ?? state.current;
      return state.open.find((one) => one.id === wanted);
    };

    /**
     * Make a command on a document, and everything a made gesture owes the rest of the editor.
     *
     * The sidecar follows what moved and is pruned of what went (D6: "a sidecar key that names a
     * path the document no longer has is dropped with a log line"), and the selection follows the
     * place it was on — a rename moves it, a delete clears it. The pipeline needs no telling: the
     * store's own subscription runs it (§5.4).
     */
    const run = (one: OpenDocument, command: Command): Applied => {
      const applied = one.session.store.apply(command);
      if (!applied.changed) return applied;
      if (applied.moves.length > 0) one.session.layout.follow(applied.moves);
      const dropped = one.session.layout.prune(one.session.store.tree);
      for (const key of dropped) {
        note(`layout: ${key.key} names nothing the document has; dropped`);
      }
      // The Log keeps the line; the panel keeps the standing fact (§4.17's `editor` notices).
      if (dropped.length > 0) {
        patch(one.id, (open) => ({
          ...open,
          dropped: [...open.dropped, ...dropped.map((key) => ({ where: key.where, key: key.key }))],
        }));
      }
      const selection = get().open.find((open) => open.id === one.id)?.selection;
      if (selection !== undefined) {
        const moved = applied.moves.find((move) => isUnder(selection, move.from));
        if (moved !== undefined) {
          patch(one.id, (open) => ({
            ...open,
            selection: [...moved.to, ...selection.slice(moved.from.length)],
          }));
        } else if (applied.cascade?.removed.some((place) => isUnder(selection, place)) === true) {
          patch(one.id, withoutSelection);
        }
      }
      if (applied.cascade !== undefined) {
        const { removed, kept } = applied.cascade;
        note(
          `${applied.label}: ${String(removed.length)} place(s) removed` +
            (kept.length === 0 ? '' : `, ${String(kept.length)} kept on the grammar`),
        );
      }
      return applied;
    };

    /** Put a session behind a tab and start its pipeline. */
    const hold = (session: DocumentSession, title: string): void => {
      const id = tabId(session.workspace, session.path);
      const tag = tagOf(session.store.shapes, session.store.tree, session.store.role);
      const one: OpenDocument = {
        dropped: [],
        id,
        path: session.path,
        workspace: session.workspace,
        title,
        dirty: session.dirty,
        session,
        reading: noReading(session.store.revision),
        figures: [],
        toggled: [],
        ...(tag === null ? {} : { tag }),
      };
      set((state) => ({
        open: [...state.open.filter((open) => open.id !== id), one],
        current: id,
      }));
      tabs.open({ id, title, kind: DOCUMENT_TAB, dirty: one.dirty });
      following.get(id)?.();
      following.set(
        id,
        session.store.subscribe(() => {
          refresh(id);
          pipelines.get(id)?.soon();
        }),
      );
      rememberTabs();
      startPipeline(id);
    };

    /**
     * The editor's own rows about one document — §4.17's `editor` source, at `notice`.
     *
     * Computed where the verdict is and for the same revision, over the outline and the reference
     * index the tree and the sheet already read. They name no rule of §6: an unused quantity is a
     * question about references, and a dropped sidecar key is the editor's own event.
     */
    const noticesOf = (id: string): readonly Problem[] => {
      const one = get().open.find((open) => open.id === id);
      if (one === undefined || shapes === null) return [];
      const rows = outlineOf({
        tree: one.session.store.tree,
        shapes,
        bindings,
        role: one.session.store.role,
        openAll: true,
      });
      return [
        ...unusedQuantities({ rows, index: one.session.store.context.index, file: one.path }),
        ...droppedKeys(one.dropped, one.path),
      ];
    };

    /**
     * The documents `Model ▸ Lint` is read over: every model document of the workspace.
     *
     * Which set is this feature's decision, and the reason is measured rather than argued.
     * `--lint`'s answer is a **function of the set** (feature 1.10): "called by none of the N
     * model(s) linted" names its size, and what the set calls decides which primitives are
     * reported — so one document linted alone reports thirty-two of the reference base's
     * primitives as uncalled, which is a wall of advice about the base rather than about the
     * document. The corpus of fourteen lints **clean**, and that is the set the repository lints
     * itself with.
     *
     * A *model* document and not every `.json`: `uncalled_primitives` indexes `instances` and
     * `compositions` before the grammar is checked, so a library unit handed to it crashes the
     * command (feature 1.10's finding). The filter is the tag the **schema** fixes, read off each
     * file — never a revision written here (feature 2.1's refusal, kept).
     *
     * An open document is read as the editor holds it, unsaved edits and all: a lint run is about
     * what is on the screen, not about what is on disk.
     */
    const lintSet = async (): Promise<{ path: string; text: string }[]> => {
      const held = shapes;
      if (held === null) return [];
      const wanted = fixedTag(held);
      const workspace = platform.workspace;
      const open = new Map(get().open.map((one) => [one.path, one]));
      // At once where the workspace can (feature 2.4 measured 131 files at 167–215 ms one after
      // another and 82–90 ms together), and what the editor holds in place of what is on disk.
      const unopened = get().documents.filter((path) => !open.has(path));
      const bulk =
        workspace.readMany === undefined
          ? await Promise.all(
              unopened.map(async (path) => {
                try {
                  return [path, (await workspace.read(path)).text] as const;
                } catch {
                  return null;
                }
              }),
            ).then((read) => new Map(read.filter((one) => one !== null)))
          : new Map(
              Object.entries(await workspace.readMany(unopened)).map(
                ([path, one]) => [path, one.text] as const,
              ),
            );
      const found: { path: string; text: string }[] = [];
      for (const path of get().documents) {
        const text = open.get(path)?.session.text ?? bulk.get(path);
        if (text === undefined) continue;
        const one = { path, text };
        try {
          const tree = parse(one.text);
          if (!isJsonObject(tree)) continue;
          if (wanted !== null && tagOf(held, tree) !== wanted) continue;
          found.push(one);
        } catch {
          // A text that is not JSON at all is not a document to lint: `--lint` would raise on it
          // (feature 1.10's second hole), and `--validate` is where its refusal belongs.
          continue;
        }
      }
      return found;
    };

    const startPipeline = (id: string): void => {
      const one = get().open.find((open) => open.id === id);
      const handle = get().library.handle;
      if (one === undefined || handle === null) return;
      pipelines.get(id)?.stop();
      const pipeline = new Pipeline({
        lang,
        library: handle,
        path: one.path,
        read: () => ({ tree: one.session.store.tree, revision: one.session.store.revision }),
        publish: publishing(id),
        registry,
        role: one.session.store.role,
        lintSet,
        notices: () => noticesOf(id),
        ...(options.debounceMs === undefined ? {} : { debounceMs: options.debounceMs }),
      });
      pipelines.set(id, pipeline);
      pipeline.now();
    };

    const rememberTabs = (): void => {
      const state = get();
      settings.set(TABS_SETTING, {
        workspace: state.workspace.id,
        paths: state.open.map((one) => one.path),
        // The *path*, not the tab's identity: a tab is made when a document opens, and what a
        // relaunch has to find again is the document.
        current: state.open.find((one) => one.id === state.current)?.path ?? '',
      });
    };

    /**
     * Let a document go: its pipeline, its subscription, and what the core holds for it.
     *
     * The core keeps its reading per **document path** (§5.3's "one in-flight derivation per
     * document"), so that is what it is told to forget — the tab's identity is the editor's and
     * means nothing to it. The path is read before the caller moves it, which is what a Save As
     * does.
     */
    const release = (id: string, path?: WorkspacePath): void => {
      pipelines.get(id)?.stop();
      pipelines.delete(id);
      following.get(id)?.();
      following.delete(id);
      const at = path ?? get().open.find((one) => one.id === id)?.path;
      if (at !== undefined) void lang.forget(at);
    };

    const adopt = async (workspace: Workspace | null): Promise<void> => {
      if (workspace === null) return;
      generation += 1;
      const mine = generation;
      watching?.();
      watching = null;
      for (const id of [...pipelines.keys()]) release(id);
      for (const one of get().open) tabs.close(one.id);
      const reference = workspace.root();
      schemas = null;
      registry = null;
      shapes = null;
      figureShapes = null;
      set({
        workspace: reference,
        open: [],
        current: null,
        library: { loading: false, handle: null, problems: [], files: 0, ms: 0, templates: new Set() },
        banner: bannerFor(reference),
        dialog: null,
        documents: [],
        notices: [],
      });
      settings.set(WORKSPACE_SETTING, { id: reference.id, kind: reference.kind });
      note(`workspace: ${reference.name} · ${reference.kind}${reference.writable ? '' : ' · read-only'}`);
      // A watch is started before the first await and stopped the moment another open wins, which
      // is the defect the review repair `910539b` closed one level down.
      watching = workspace.watch('', (event) => {
        if (generation !== mine) return;
        note(`workspace: ${event.kind} ${event.path}`);
      });
      const found = await listTree(workspace, '', { suffix: '.json' }).catch(() => []);
      if (generation !== mine) return;
      set({ documents: found });
      await reopenTabs(workspace, mine);
    };

    /** §4.3's "tabs restore on relaunch", for the workspace that just opened. */
    const reopenTabs = async (workspace: Workspace, mine: number): Promise<void> => {
      const remembered = settings.peek(TABS_SETTING);
      if (typeof remembered !== 'object' || remembered === null || Array.isArray(remembered)) return;
      const held = remembered as { workspace?: unknown; paths?: unknown; current?: unknown };
      if (held.workspace !== workspace.root().id || !Array.isArray(held.paths)) return;
      for (const path of held.paths) {
        if (typeof path !== 'string') continue;
        if (generation !== mine) return;
        await get().openDocument(path);
      }
      if (generation !== mine) return;
      if (typeof held.current === 'string' && held.current !== '') {
        get().select(tabId(workspace.root().id, held.current));
      }
    };

    return {
      workspace: platform.workspace.root(),
      open: [],
      current: null,
      library: { loading: false, handle: null, problems: [], files: 0, ms: 0, templates: new Set() },
      banner: null,
      toast: null,
      dialog: null,
      documents: [],
      notices: [],
      filter: '',

      async start(): Promise<void> {
        // D11: the folder the browser still holds a grant for, with no gesture and no prompt.
        const granted = await reopenGranted(platform);
        if (granted !== null) {
          await adopt(granted);
          return;
        }
        const last = settings.peek(WORKSPACE_SETTING);
        const kind = (last as LastWorkspace | undefined)?.kind;
        // A vendored workspace needs no permission and no files from anybody, so the one thing a
        // page may reopen by itself is that one. A snapshot cannot be: the files are gone.
        if (kind === EXAMPLES) await adopt(await platform.workspaces.openExamples());
        else set({ banner: null });
      },

      adopt,

      async openFolder(): Promise<void> {
        if (!platform.workspaces.writablePicker) {
          set({ dialog: { kind: 'workspaces', recent: await platform.workspaces.recent() } });
          note('this browser has no writable directory picker: drop a folder, or upload one');
          return;
        }
        await adopt(await platform.workspaces.open());
      },

      async openExamples(): Promise<void> {
        await adopt(await platform.workspaces.openExamples());
      },

      async openRecent(id: string): Promise<void> {
        const workspace = await platform.workspaces.reopen(id);
        if (workspace === null) {
          note(`the folder ${id} could not be reopened`);
          return;
        }
        await adopt(workspace);
      },

      async openUpload(files: readonly UploadedFile[]): Promise<void> {
        await adopt(await platform.workspaces.openUpload(files));
      },

      async dropped(opening: Promise<Workspace | null>): Promise<void> {
        await adopt(await opening);
      },

      async forget(id?: string): Promise<void> {
        await platform.workspaces.forget(id);
        if (get().dialog?.kind !== 'workspaces') return;
        set({ dialog: { kind: 'workspaces', recent: await platform.workspaces.recent() } });
      },

      async openDocument(path: WorkspacePath): Promise<void> {
        const workspace = platform.workspace;
        const reference = workspace.root();
        const id = tabId(reference.id, path);
        if (get().open.some((one) => one.id === id)) {
          get().select(id);
          return;
        }
        const mine = generation;
        let text: string;
        let revision: string;
        /** True where the text came from a draft and not from the workspace. */
        let restored = false;
        try {
          const read = await workspace.read(path);
          text = read.text;
          revision = read.revision;
        } catch (error) {
          // A document the workspace does not hold is not always a mistake: a New Model that was
          // never saved has a path and a draft and no file, and §4.3 asks the editor both to
          // restore its tabs and never to lose a draft. There is nothing to *offer* it against —
          // the choice is the draft or nothing at all — so it is opened, and the Log says so.
          const held = await platform.drafts.get(reference.id, path);
          if (generation !== mine) return;
          if (held === null) {
            note(`${path}: ${error instanceof Error ? error.message : String(error)}`);
            return;
          }
          text = held.text;
          revision = ABSENT;
          restored = true;
        }
        if (generation !== mine) return;
        if (get().library.handle === null) {
          const tree = await lang.parse(text);
          if (generation !== mine) return;
          await library(workspace, tree, path);
          if (generation !== mine) return;
        }
        if (shapes === null) return;
        const layout = await readSidecar(workspace, path);
        if (generation !== mine) return;
        const session = DocumentSession.open(text, shapes, {
          workspace: reference.id,
          path,
          revision,
          layout,
          ...(restored ? { saved: false } : {}),
        });
        hold(session, titleOf(session, shapes));
        if (restored) {
          note(`restored the draft of ${path}, which the workspace does not hold`);
          return;
        }
        const draft = await platform.drafts.get(reference.id, path);
        if (generation !== mine || draft === null) return;
        const standing = draftStanding(draft, text, revision);
        if (standing === null) return;
        patch(id, (one) => ({ ...one, draft: { draft, standing } }));
        set({ dialog: { kind: 'restore', id } });
      },

      async newModel(template = false): Promise<void> {
        const workspace = platform.workspace;
        const reference = workspace.root();
        if (shapes === null || get().library.handle === null) {
          // A document made from nothing still needs the schemas the build vendored, and a base
          // to resolve from if the workspace has one; opening them is what a first document does.
          vendored ??= await options.vendoredSchemas();
          const gathered = await gatherSchemas(workspace, vendored);
          const loaded = await lang.loadSchemas(gathered.files, { origin: gathered.origin });
          schemas = loaded.handle;
          registry = loadSchemas(
            Object.entries(gathered.files).map(([file, text]) => ({ path: file, text })),
            { origin: gathered.origin },
          );
          shapes = new SchemaShapes(registry);
          figureShapes = shapesFor(registry);
        }
        const name = untitled(get().open.map((one) => one.title));
        const path = `${name}.json`;
        const base = await defaultBase(workspace, path);
        const tree = newDocument(shapes, {
          name,
          ...(base === null ? {} : { base }),
          ...(template ? { version: FIRST_VERSION } : {}),
        });
        const session = new DocumentSession(tree, shapes, {
          workspace: reference.id,
          path,
          revision: ABSENT,
          saved: false,
        });
        if (get().library.handle === null) await library(workspace, tree, path);
        hold(session, name);
        note(`new ${template ? 'template' : 'model'}: ${path}`);
      },

      select(id: string): void {
        if (!get().open.some((one) => one.id === id)) return;
        set({ current: id });
        tabs.select(id);
        rememberTabs();
      },

      async mayClose(id: string): Promise<boolean> {
        const one = get().open.find((open) => open.id === id);
        if (one === undefined || !one.dirty) return true;
        const kept = await platform.shell.confirm(
          `${one.title} has unsaved changes. Close it and lose them?`,
        );
        return kept;
      },

      closeDocument(id: string): void {
        release(id);
        set((state) => ({
          open: state.open.filter((one) => one.id !== id),
          current: state.current === id ? null : state.current,
          dialog: state.dialog !== null && 'id' in state.dialog && state.dialog.id === id ? null : state.dialog,
        }));
        rememberTabs();
      },

      selectPlace(path: Path | null, id?: string): void {
        const one = current(id);
        if (one === undefined) return;
        patch(one.id, (open) => (path === null ? withoutSelection(open) : { ...open, selection: path }));
      },

      revealPlace(path: Path, id?: string): void {
        const one = current(id);
        if (one === undefined || shapes === null) return;
        const rows = outlineOf({
          tree: one.session.store.tree,
          shapes,
          bindings,
          role: one.session.store.role,
          openAll: true,
        });
        const toggled = revealing(rows, one.toggled, pointerOf(path));
        patch(one.id, (open) => ({ ...open, toggled: [...toggled], selection: path }));
      },

      lint(id?: string): void {
        const one = documentOf(get(), id ?? tabs.current());
        if (one === undefined) {
          note('there is no document open to lint.');
          return;
        }
        note(`lint: ${String(get().documents.length)} candidate file(s) in the workspace`);
        pipelines.get(one.id)?.lint();
      },

      togglePlace(pointer: string, id?: string): void {
        const one = current(id);
        if (one === undefined) return;
        patch(one.id, (open) => ({
          ...open,
          toggled: open.toggled.includes(pointer)
            ? open.toggled.filter((each) => each !== pointer)
            : [...open.toggled, pointer],
        }));
      },

      setFilter(filter: string): void {
        set({ filter });
      },

      edit(make: (context: EditContext) => Command, id?: string): Applied | null {
        const one = current(id);
        if (one === undefined) return null;
        return run(one, make(one.session.store.context));
      },

      offer(make: (context: EditContext) => Command, id?: string): void {
        const one = current(id);
        if (one === undefined) return;
        const command = make(one.session.store.context);
        // A command with nothing to warn about is made rather than asked about: a confirmation
        // that always says yes teaches the reader to stop reading it.
        if (command.cascade === undefined) {
          run(one, command);
          return;
        }
        pending = { id: one.id, command };
        const { removed, kept } = command.cascade;
        set({
          dialog: {
            kind: 'remove',
            id: one.id,
            label: command.label,
            removed: removed.map((place) => pointerOf(place)),
            kept: kept.map((place) => pointerOf(place)),
          },
        });
      },

      confirmed(): void {
        const held = pending;
        pending = null;
        set({ dialog: null });
        if (held === null) return;
        const one = get().open.find((open) => open.id === held.id);
        if (one !== undefined) run(one, held.command);
      },

      async save(id?: string): Promise<void> {
        const one = documentOf(get(), id ?? tabs.current());
        if (one === undefined) return;
        try {
          const saved = await one.session.save(platform.workspace);
          await platform.drafts.remove(one.workspace, one.path);
          patch(one.id, (open) => ({ ...open, path: saved.path, dirty: open.session.dirty }));
          tabs.update(one.id, { dirty: one.session.dirty });
          const where = platform.workspace.root().writable ? 'saved' : 'downloaded';
          note(
            `${where}: ${saved.path} (${String(saved.bytes)} bytes)` +
              (saved.sidecar === undefined ? '' : ` and ${saved.sidecar}`),
          );
          set({ toast: { text: `${where} ${saved.path}` } });
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          note(`${one.path}: ${message}`);
          set({
            banner: {
              kind: 'stop',
              head: `${one.path} was not written.`,
              body:
                error instanceof PlatformError && error.reason === 'conflict'
                  ? `${message} — reopen it, or Save As to a new file.`
                  : message,
            },
          });
        }
      },

      async saveAs(id: string, path: WorkspacePath): Promise<void> {
        const one = get().open.find((open) => open.id === id);
        if (one === undefined) return;
        const before = one.path;
        try {
          const saved = await one.session.save(platform.workspace, path);
          // A tab's identity is its workspace and its path, so a document that moved is a tab that
          // closes and a tab that opens — in that order, and with the document taken out of the
          // state first so that the close has nothing to prompt about: it was just written.
          release(id, before);
          set((state) => ({ open: state.open.filter((open) => open.id !== id), dialog: null }));
          tabs.close(id);
          hold(one.session, one.title);
          note(`saved as: ${saved.path}`);
        } catch (error) {
          note(`${path}: ${error instanceof Error ? error.message : String(error)}`);
        }
      },

      async saveAll(): Promise<void> {
        for (const one of get().open) if (one.dirty) await get().save(one.id);
      },

      async revert(id?: string): Promise<void> {
        const one = documentOf(get(), id ?? tabs.current());
        if (one === undefined) return;
        try {
          const read = await platform.workspace.read(one.path);
          one.session.revert(read.text, read.revision);
          await platform.drafts.remove(one.workspace, one.path);
          patch(one.id, (open) => withoutDraft({ ...open, dirty: open.session.dirty }));
          tabs.update(one.id, { dirty: one.session.dirty });
          note(`reverted: ${one.path}`);
        } catch (error) {
          note(`${one.path}: ${error instanceof Error ? error.message : String(error)}`);
        }
      },

      async downloadZip(): Promise<void> {
        const workspace = platform.workspace;
        const reference = workspace.root();
        const paths = await listTree(workspace, '');
        const files = await readTree(workspace, '');
        // The editor's own unsaved work goes into the archive beside the folder's files: the
        // command is for a snapshot "edited in the browser, whole" (§4.4), and an archive of the
        // copy the page was handed without the edits would be the one thing it is not for.
        const edited = new Map<WorkspacePath, string>();
        for (const one of get().open) {
          edited.set(one.path, one.session.text);
          if (one.session.layoutDirty) edited.set(sidecarOf(one.path), writeSidecar(one));
        }
        const entries: ZipEntry[] = [];
        for (const path of [...new Set([...paths, ...edited.keys()])].sort((a, b) => a.localeCompare(b))) {
          const text = edited.get(path) ?? files[path];
          if (text === undefined) continue;
          entries.push({ path, content: text });
        }
        const archive = zipOf(entries);
        await platform.shell.download(`${reference.name || 'workspace'}.zip`, archive);
        note(`downloaded: ${String(entries.length)} files, ${String(archive.length)} bytes`);
      },

      async autosave(): Promise<void> {
        for (const one of get().open) {
          if (!one.dirty) continue;
          await platform.drafts.put(one.session.draftOf());
        }
      },

      restoreDraft(id: string): void {
        const one = get().open.find((open) => open.id === id);
        if (one?.draft === undefined) return;
        one.session.restore(one.draft.draft.text);
        patch(id, (open) => withoutDraft({ ...open, dirty: open.session.dirty }));
        tabs.update(id, { dirty: one.session.dirty });
        set({ dialog: null });
        note(`restored the draft of ${one.path}`);
        pipelines.get(id)?.now();
      },

      async discardDraft(id: string): Promise<void> {
        const one = get().open.find((open) => open.id === id);
        if (one === undefined) return;
        await platform.drafts.remove(one.workspace, one.path);
        patch(id, (open) => withoutDraft(open));
        set({ dialog: null });
      },

      validateNow(id?: string): void {
        const one = documentOf(get(), id ?? tabs.current());
        if (one === undefined) {
          note('there is no document open to check.');
          return;
        }
        pipelines.get(one.id)?.now();
      },

      setDialog(dialog: Dialog | null): void {
        set({ dialog });
      },

      setToast(toast: ToastLine | null): void {
        set({ toast });
      },

      note,

      stop(): void {
        watching?.();
        watching = null;
        for (const id of [...pipelines.keys()]) release(id);
        if (autosaveTimer !== null) clearInterval(autosaveTimer);
        autosaveTimer = null;
      },
    };
  });

  const every = options.autosaveMs ?? AUTOSAVE_MS;
  if (every > 0) {
    autosaveTimer = setInterval(() => {
      void store.getState().autosave();
    }, every);
  }

  const stopWatchingWorkspace = platform.workspaces.onChange((workspace) => {
    if (workspace !== platform.workspace) return;
    store.setState({ workspace: workspace.root() });
  });

  return {
    store,
    dispose: () => {
      stopWatchingWorkspace();
      store.getState().stop();
    },
  };
}

/** The `examples` workspace: the one a page may reopen by itself, needing nothing from anybody. */
const EXAMPLES = 'examples';

/** The badge §4.3 gives a document that needs an assignment. */
const TEMPLATE_BADGE = 'template';

/** The banner a workspace's own kind asks for (component inventory §5, S18). */
export function bannerFor(reference: WorkspaceRef): BannerLine | null {
  if (reference.writable || reference.kind === 'empty') return null;
  if (reference.kind === EXAMPLES) {
    return {
      kind: 'info',
      head: 'These are the examples the build carries.',
      body:
        'The corpus and the reference base, read-only. Save As copies a document into a folder of ' +
        'your own; everything else — validation, derivation, the checkpoint check — works exactly the same.',
    };
  }
  return {
    kind: 'warn',
    head: 'This folder is a read-only snapshot.',
    body:
      'Your browser cannot write back to it, so Save downloads the file for you to put back. ' +
      'Everything else — validation, derivation, the checkpoint check — works exactly the same.',
  };
}

/** The document a command with no argument is about: the current tab. */
function documentOf(state: DocumentsState, id: string | null): OpenDocument | undefined {
  if (id === null) return undefined;
  return state.open.find((one) => one.id === id);
}

/** §4.3's "a tab named after `model`" — the member the schema says is the document's name. */
function titleOf(session: DocumentSession, shapes: SchemaShapes): string {
  const member = nameMember(shapes, session.store.role);
  const tree = session.store.tree;
  const found = member === null ? undefined : tree.members.find((one) => one.name === member);
  return typeof found?.value === 'string' && found.value !== '' ? found.value : session.name;
}

/** A name for a document made from nothing, not colliding with one already open. */
function untitled(taken: readonly string[]): string {
  const base = 'untitled';
  if (!taken.includes(base)) return base;
  for (let index = 2; ; index += 1) {
    const name = `${base}-${String(index)}`;
    if (!taken.includes(name)) return name;
  }
}

/** The sidecar of a document that has overrides, as the archive carries it (§5.5's plain writer). */
function writeSidecar(one: OpenDocument): string {
  return writeLayout(one.session.layout.layout);
}

/**
 * The base a new document resolves from — "the workspace's default base" (§4.3).
 *
 * The workspace's own, found by its manifest: a folder that holds `primitive-library.json` is a
 * base, which is the loader's own reading of one. Nothing is invented where there is none: the
 * document is written without a base and the Problems panel says what is missing.
 */
async function defaultBase(workspace: Workspace, from: WorkspacePath): Promise<string | null> {
  const manifests = (await listTree(workspace, '', { suffix: BASE_MANIFEST }).catch(() => [])).sort(
    (a, b) => a.length - b.length,
  );
  const found = manifests[0];
  if (found === undefined) return null;
  const directory = found.slice(0, Math.max(0, found.length - BASE_MANIFEST.length - 1));
  const here = from.includes('/') ? from.slice(0, from.lastIndexOf('/')) : '';
  return relativeTo(here, directory);
}

/** `to`, written relative to `from`, as a document's `primitive_libraries[].base` is. */
function relativeTo(from: WorkspacePath, to: WorkspacePath): string {
  const here = from === '' ? [] : from.split('/');
  const there = to === '' ? [] : to.split('/');
  let shared = 0;
  while (shared < here.length && shared < there.length && here[shared] === there[shared]) shared += 1;
  const up = here.slice(shared).map(() => '..');
  const down = there.slice(shared);
  return `${[...up, ...down].join('/') || '.'}/`;
}

/** The folder the browser still holds a grant for, where the platform can answer one (D11). */
async function reopenGranted(platform: Platform): Promise<Workspace | null> {
  const workspaces = platform.workspaces;
  if (workspaces.reopenGranted === undefined) return null;
  try {
    return await workspaces.reopenGranted();
  } catch {
    // A grant the browser has withdrawn, or a folder that is gone: the editor opens with nothing,
    // which is S17's own state, and the folder is offered by name in File ▸ Open Recent.
    return null;
  }
}

/** The same document with no draft outstanding — the member removed, not set to nothing. */
function withoutDraft(one: OpenDocument): OpenDocument {
  const { draft, ...rest } = one;
  void draft;
  return rest;
}

/** A sink that opens no tab: what the store does when nobody gave it a strip (a suite's). */
function silentTabs(): TabSink {
  let current: string | null = null;
  return {
    open: (tab) => {
      current = tab.id;
    },
    update: () => undefined,
    close: () => {
      current = null;
    },
    select: (id) => {
      current = id;
    },
    current: () => current,
  };
}
