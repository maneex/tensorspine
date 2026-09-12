/**
 * The editor's own contributions — Monaco's `editor.main.js` without the eighty languages.
 *
 * **Why this module exists, measured.** Monaco instantiates a code editor's contributions from a
 * registry *at the moment the editor is created*, so an editor made before they are registered is
 * a bare text box: no suggest controller, no hover, no find, no bracket matching, no context menu
 * — `getSupportedActions()` answers an empty list. The JSON language pulls most of them in as a
 * side effect of its own worker manager, but it does so **asynchronously** (`onLanguage` imports
 * `jsonMode.js` when the first JSON model appears), which is after the editor is up. So they are
 * imported here, eagerly, before `monaco.editor.create` is called.
 *
 * **What is imported.** `features/register.all.js` is Monaco's own list of every editor feature,
 * and the nine modules after it are what `editor/editor.main.js` adds beyond that list. What is
 * deliberately *not* imported is `monaco-editor` itself and `editor.main.js`: both register the
 * eighty language tokenizers, the TypeScript language service among them, which a JSON source
 * view has no use for.
 *
 * **One of `editor.main.js`'s own imports cannot be written here**, and it is stated rather than
 * worked around: `base/browser/ui/codicons/codicon/codicon-modifiers.css` is a stylesheet, and
 * the package's `exports` map sends every specifier through `./esm/vs/*.js`, so a `.css` one
 * resolves to a `.css.js` that does not exist. `codicon.css` itself arrives with
 * `register.all.js`, which imports it by a relative path from inside the package; what is lost is
 * the two *modifier* classes (`codicon-modifier-spin`, `-disabled`), so an icon that would have
 * spun stands still. Nothing this view draws uses one.
 */
import 'monaco-editor/features/register.all.js';
import 'monaco-editor/editor/browser/coreCommands.js';
import 'monaco-editor/editor/contrib/caretOperations/browser/caretOperations.js';
import 'monaco-editor/editor/contrib/dropOrPasteInto/browser/copyPasteContribution.js';
import 'monaco-editor/editor/contrib/find/browser/findController.js';
import 'monaco-editor/editor/contrib/gotoSymbol/browser/goToCommands.js';
import 'monaco-editor/editor/contrib/gotoError/browser/markerSelectionStatus.js';
import 'monaco-editor/editor/contrib/semanticTokens/browser/documentSemanticTokens.js';
import 'monaco-editor/editor/contrib/suggest/browser/suggestController.js';
import 'monaco-editor/editor/common/standaloneStrings.js';
