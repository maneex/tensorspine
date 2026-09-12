/**
 * Monaco's own worker — the one every editor asks for whatever language it is showing.
 *
 * Word-based suggestions, link detection and the diff Monaco computes for itself run here. It is
 * a worker entry point and nothing else; see `./json.worker.ts` for the shape and the reason.
 */
import 'monaco-editor/editor/editor.worker.js';
