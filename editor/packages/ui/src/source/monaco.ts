/**
 * Monaco, loaded when a JSON source view is first opened — plan §4.10, §5.1.
 *
 * > Monaco, JSON mode, with the model schema (or the unit schema, for a unit) attached
 * > (completion, hover from `description`s, structural errors at their range).
 *
 * **It is its own chunk, and nothing imports it statically.** Feature 0.4 measured what a
 * megabyte and a half of ELK does to the shell's first bundle and answered with a dynamic
 * import; feature 0.5 found the same for the Hub client and feature 2.16 kept the layout out of
 * the expanded view. A text editor is the largest of the four, so `./Source.tsx` reaches this
 * module with `await import(…)` and nothing else in the workspace names `monaco-editor` at all —
 * `test/source/bundle.test.ts` reads the sources and says so.
 *
 * **What is imported, and what is not.** The editor API, every editor feature (`./features.ts`,
 * which is `editor.main.js`'s own list) and the JSON language — and no other language: Monaco's
 * `editor.main.js` registers eighty tokenizers the editor has no use for, the TypeScript language
 * service among them. So the feature set is the editor's whole one and the saving is the
 * languages'.
 *
 * **Monaco's own JSON validation is off.** §4.10 asks for the schema to be attached and for
 * "structural errors at their range"; the errors are the **core's** — Ajv on the schema files, as
 * D4 fixes, held to `jsonschema`'s wording by the parity job — and `vscode-json-languageservice`
 * would be a third reading of the same schemas with a fourth wording, outside the parity contract
 * and outside the audit of §1. So the schema is attached for *completion and hover* (§1: "Help
 * text is the schema's"), diagnostics are the core's, and `markers.ts` puts them at their range.
 */
import type * as Monaco from 'monaco-editor/editor/editor.api.js';

/** What Monaco's JSON language answers when it is registered: its defaults, to attach schemas to. */
type JsonDefaults = (typeof import('monaco-editor/languages/features/json/register.js'))['jsonDefaults'];

let jsonDefaults: JsonDefaults | null = null;

/** What the loaded module answers: Monaco's namespace, once. */
export type MonacoApi = typeof Monaco;

/**
 * The three Monaco types the pane names, re-exported here.
 *
 * `./Source.tsx` may not name `monaco-editor` itself — a specifier there would be a static import
 * of a text editor into the shell's chunk, which is what `test/source/bundle.test.ts` forbids —
 * and a `import type` is invisible to a scan that reads specifiers. One module names the package
 * and everything else names this one.
 */
export type MonacoEditor = Monaco.editor.IStandaloneCodeEditor;
export type MonacoModel = Monaco.editor.ITextModel;
export type MonacoRange = Monaco.IRange;

/** The language Monaco reads a `tensorspine/2.0` document as. Monaco's own id, not a vocabulary. */
export const JSON_LANGUAGE = 'json';

/** The theme this editor defines over the design's tokens (§4.21). */
export const THEME = 'tensorspine';

let loading: Promise<MonacoApi> | null = null;

/**
 * Monaco, loaded once per page.
 *
 * The workers are two, and they are created here because `new Worker(new URL(…), {type: 'module'})`
 * is a *build* instruction the bundler follows — the same form `apps/web/src/lang/connect.ts`
 * uses for the core (§5.3) — and because a language service that ran on the interface's own
 * thread would block it exactly when a reader is typing.
 */
export async function loadMonaco(): Promise<MonacoApi> {
  loading ??= (async (): Promise<MonacoApi> => {
    const monaco = (await import('monaco-editor/editor/editor.api.js')) as unknown as MonacoApi;
    // Every editor feature, **before** an editor is created: Monaco instantiates a code editor's
    // contributions from a registry at creation time, so an editor made ahead of them has no
    // suggest controller, no hover, no find and no context menu (`./features.ts` says how that
    // was measured).
    await import('./features.js');
    // Registers the language, its tokenizer, its completion and its hover — and nothing else.
    jsonDefaults = (await import('monaco-editor/languages/features/json/register.js')).jsonDefaults;
    const environment = globalThis as unknown as {
      MonacoEnvironment?: { getWorker?: (id: string, label: string) => Worker };
      monaco?: MonacoApi;
    };
    environment.MonacoEnvironment = {
      getWorker: (_id: string, label: string): Worker =>
        label === JSON_LANGUAGE
          ? new Worker(new URL('./json.worker.js', import.meta.url), {
              type: 'module',
              name: 'tensorspine-json',
            })
          : new Worker(new URL('./editor.worker.js', import.meta.url), {
              type: 'module',
              name: 'tensorspine-editor',
            }),
    };
    // Monaco's own distribution publishes `window.monaco`; the ESM build leaves that to its host
    // and the convention is kept, so the models a page holds are reachable from a console and
    // from the browser layer — which is where a document's bytes are asserted (feature 2.9's
    // three suites read them through the source view, and a virtualised editor renders only what
    // is on screen).
    environment.monaco = monaco;
    return monaco;
  })();
  return loading;
}

/**
 * The JSON schemas Monaco completes and hovers from — the ones the workspace loaded (§1).
 *
 * Every schema of the registry is handed over, so that a `$ref` between them resolves, and the
 * one whose role the document is read under is matched to the model's own URI. Nothing is
 * fetched: `enableSchemaRequest` is off, because the schemas are the repository's files and a
 * page that went to the network for one would be reading a schema nobody vendored.
 */
export interface SchemaAttachment {
  /** The schema's published identity, which is also the key a `$ref` resolves against. */
  readonly uri: string;
  /** The schema itself, as the file writes it. */
  readonly schema: unknown;
  /** The model URIs this schema is the schema *of*; empty for one that is only referred to. */
  readonly fileMatch?: readonly string[];
}

/** Attach the schemas, with Monaco's own validation switched off (see the module's note). */
export function attachSchemas(monaco: MonacoApi, schemas: readonly SchemaAttachment[]): void {
  void monaco;
  console.log('PROBE attachSchemas', jsonDefaults === null ? 'no defaults' : schemas.length, JSON.stringify(schemas.map((s) => [s.uri, s.fileMatch])));
  jsonDefaults?.setDiagnosticsOptions({
    validate: false,
    enableSchemaRequest: false,
    allowComments: false,
    schemas: schemas.map((one) => ({
      uri: one.uri,
      schema: one.schema,
      ...(one.fileMatch === undefined ? {} : { fileMatch: [...one.fileMatch] }),
    })),
  });
}

/**
 * The editor's theme, from the design's own tokens — read off the page, never copied (§4.21).
 *
 * The token *values* live in one place (`shell/tokens.css`, vendored from the design pass and
 * held to it byte for byte) and a theme is the one thing Monaco cannot take as a CSS variable:
 * it paints its own text from a palette it is given. So the palette is read back out of the page
 * with `getComputedStyle`, which means a theme switch re-reads it and the values are never
 * written twice.
 *
 * **It is read from an element inside the pane, not from the document's root**, because the light
 * theme's block is written under a class the shell puts on its own frame (`.app.theme-light`) —
 * reading the root would answer the dark palette in both themes, which is how the light theme
 * came to paint a 3.88:1 string on a dark ground until the axe pass found it.
 *
 * Two of the design's inks do not reach §4.21's floor on this ground and are stepped up the same
 * ramp feature 2.5 established: `--faint` (3.04 dark, 2.16 light on `--bg-sunk`) carries no text
 * at all, and `--accent` is 4.30:1 in the light theme, where `--accent-dim` is 6.23. Every pair
 * this file writes is in `test/shell/contrast.test.ts`.
 */
export function defineTheme(monaco: MonacoApi, within: Element, light: boolean): void {
  const style = getComputedStyle(within);
  const token = (name: string): string => style.getPropertyValue(name).trim();
  const bare = (name: string): string => token(name).replace('#', '');
  const stringInk = light ? bare('--accent-dim') : bare('--accent');
  monaco.editor.defineTheme(THEME, {
    base: light ? 'vs' : 'vs-dark',
    inherit: true,
    rules: [
      { token: '', foreground: bare('--ink-3') },
      // The design's own JSON colouring (`.jk` `.js` `.jn` `.jb` of `_ts.css`), with the two inks
      // that do not reach the floor stepped up: a key is the primary ink, a text value the
      // accent, a number the "derived" amber, and a truth or a null the tertiary ink.
      { token: 'string.key.json', foreground: bare('--ink') },
      { token: 'string.value.json', foreground: stringInk },
      { token: 'string', foreground: stringInk },
      { token: 'number', foreground: bare('--derived') },
      { token: 'keyword.json', foreground: bare('--ink-3') },
      { token: 'keyword', foreground: bare('--ink-3') },
      { token: 'delimiter', foreground: bare('--ink-3') },
    ],
    colors: {
      'editor.background': token('--bg-sunk'),
      'editor.foreground': token('--ink-3'),
      'editorLineNumber.foreground': token('--ink-3'),
      'editorLineNumber.activeForeground': light ? token('--accent-dim') : token('--accent'),
      'editorCursor.foreground': light ? token('--accent-dim') : token('--accent'),
      'editor.lineHighlightBorder': token('--rule'),
      'editor.selectionBackground': token('--bg-tint'),
      'editor.inactiveSelectionBackground': token('--bg-tint'),
      'editorIndentGuide.background1': token('--rule'),
      'editorIndentGuide.activeBackground1': token('--rule-2'),
      'editorBracketMatch.background': token('--bg-tint'),
      'editorBracketMatch.border': token('--rule-strong'),
      'editorError.foreground': token('--bad'),
      'editorWarning.foreground': token('--warn'),
      'editorInfo.foreground': light ? token('--accent-dim') : token('--accent'),
      'editorWidget.background': token('--bg-panel'),
      'editorWidget.border': token('--rule-2'),
      'editorWidget.foreground': token('--ink'),
      'editorHoverWidget.background': token('--bg-panel'),
      'editorHoverWidget.border': token('--rule-2'),
      'editorSuggestWidget.background': token('--bg-panel'),
      'editorSuggestWidget.border': token('--rule-2'),
      'editorSuggestWidget.foreground': token('--ink'),
      'editorSuggestWidget.selectedBackground': token('--bg-tint'),
      'editorSuggestWidget.selectedForeground': token('--ink'),
      // The part of a suggestion that matches what has been typed. Unbound it is Monaco's own
      // white, which is 1.22:1 on `--bg-tint` — measured by the axe pass over the open widget.
      'editorSuggestWidget.highlightForeground': light ? token('--accent-dim') : token('--accent'),
      'editorSuggestWidget.focusHighlightForeground': light ? token('--accent-dim') : token('--accent'),
      'list.hoverBackground': token('--bg-tint'),
      'list.focusBackground': token('--bg-tint'),
      'list.focusForeground': token('--ink'),
      'list.highlightForeground': light ? token('--accent-dim') : token('--accent'),
      'list.focusHighlightForeground': light ? token('--accent-dim') : token('--accent'),
      'input.background': token('--bg-raised'),
      'input.foreground': token('--ink'),
      'input.border': token('--rule-2'),
      'scrollbarSlider.background': token('--rule-2'),
      'scrollbarSlider.hoverBackground': token('--rule-strong'),
      'scrollbarSlider.activeBackground': token('--rule-strong'),
    },
  });
  monaco.editor.setTheme(THEME);
}
