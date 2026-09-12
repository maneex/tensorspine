/**
 * Monaco's JSON language service, in its own worker — plan §4.10, §5.6.
 *
 * A worker entry point and nothing else: the bundler follows `new Worker(new URL('./…'))` in
 * `./monaco.ts` to this file and emits it as a chunk of its own, which is the same form
 * `apps/web/src/lang/connect.ts` uses for the core. Completion and hover from the schemas are
 * what runs here, so a keystroke in the source view never waits for them.
 */
import 'monaco-editor/languages/features/json/json.worker.js';
