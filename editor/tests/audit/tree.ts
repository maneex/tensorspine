import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

/** `editor/`, the workspace root. */
export const editorRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');

const skipped = new Set(['node_modules', 'dist', 'coverage', 'out', 'test-results']);

/** Every file under `directory`, as paths relative to `editor/`, build output left out. */
export function filesUnder(directory: string): string[] {
  const found: string[] = [];
  const walk = (current: string): void => {
    for (const entry of readdirSync(current, { withFileTypes: true })) {
      if (skipped.has(entry.name)) continue;
      const path = join(current, entry.name);
      if (entry.isDirectory()) walk(path);
      else found.push(relative(editorRoot, path));
    }
  };
  walk(resolve(editorRoot, directory));
  return found.sort();
}

/** The text of a file named relative to `editor/`. */
export function readEditorFile(path: string): string {
  return readFileSync(join(editorRoot, path), 'utf8');
}
