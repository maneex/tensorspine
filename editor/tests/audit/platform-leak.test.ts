import { existsSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

import ts from 'typescript';
import { describe, expect, it } from 'vitest';

import { CHECK_MODE, inputsFor } from '../../apps/web/vite.config.js';
import { editorRoot, readEditorFile } from './tree.js';

// The **platform leak** rule of the implementation plan's D11 and §5.2 (feature 2.4):
//
//   > The renderer imports nothing from Node or Electron; everything platform-shaped goes through
//   > `Platform` (§5.2). … CI builds the web app against a stub platform from phase 2 on, so a
//   > platform leak is a build failure (D11).
//
// The build is one half of that and it runs in the browser layer: `apps/web/e2e/platform.spec.ts`
// loads the application built against the stub `Platform` and requires its chunks to carry no
// platform implementation at all. This is the other half, and it is the one that names the rule:
// **no package reaches for a platform itself**. Three reasons it is worth a source audit beside
// the build:
//
//   * a build failure says "it broke", a source rule says *what* broke and where;
//   * the type system catches only part of it — `packages/store` is compiled without the DOM, so
//     `localStorage` there would not compile, but `packages/ui` has the DOM (it renders) and
//     `packages/lang` carries Node's types for its own suites;
//   * the rule is what makes the Electron and SaaS increments "add implementations, not changes"
//     (§0), and a rule nobody states is a rule nobody keeps.
//
// What is scanned is every package's `src/`, never `apps/`: an application *is* a platform, which
// is exactly where these names belong. `packages/lang` is in scope because the core is pure — "the
// core is pure (no I/O): the UI reads files through `Platform` and hands the core texts and trees"
// (§5.3) — and it runs in a worker, in Node and in a test unchanged.
//
// One name is deliberately **not** on the list: `webkitRelativePath`, which `UploadedFile` declares
// so that a browser `File` satisfies it without the interface naming the DOM (§5.2). It is a member
// a platform-free interface mirrors, not a reach for one — and a component that could read it would
// have to hold a `File` first, which it cannot obtain without something else on the list.

/** The globals a component must reach through `Platform` instead. */
const PLATFORM_GLOBALS = [
  // Storage: settings (`localStorage`) and drafts, recents (`indexedDB`) are `Platform`'s.
  'localStorage',
  'sessionStorage',
  'indexedDB',
  // The File System Access API: the workspace's, and only the workspace's.
  'showDirectoryPicker',
  'showOpenFilePicker',
  'showSaveFilePicker',
  'FileSystemDirectoryHandle',
  'FileSystemFileHandle',
  'FileSystemHandle',
  'getAsFileSystemHandle',
  'webkitGetAsEntry',
  'createWritable',
  'queryPermission',
  'requestPermission',
  // Reading and writing bytes outside the workspace: a download, a fetch, a checkpoint.
  'createObjectURL',
  'fetch',
  'XMLHttpRequest',
  // Electron's own, and the desktop's preload bridge, whatever it ends up being called.
  'electron',
  'ipcRenderer',
] as const;

/** A module specifier no package may import: a platform, or an application. */
function forbiddenImport(specifier: string): string | null {
  if (specifier.startsWith('node:')) return 'Node';
  if (specifier === 'electron' || specifier.startsWith('electron/')) return 'Electron';
  if (specifier.startsWith('@tensorspine/web') || specifier.includes('apps/')) return 'an application';
  return null;
}

/** Every file under a package's `src/`, as paths relative to `editor/`. */
function sourcesOf(pkg: string): string[] {
  const root = join(editorRoot, pkg, 'src');
  if (!existsSync(root)) return [];
  const found: string[] = [];
  const walk = (current: string): void => {
    for (const entry of readdirSync(current, { withFileTypes: true })) {
      const path = join(current, entry.name);
      if (entry.isDirectory()) walk(path);
      else if (/\.tsx?$/.test(entry.name)) found.push(`${pkg}/src/${path.slice(root.length + 1).split('\\').join('/')}`);
    }
  };
  walk(root);
  return found.sort();
}

/** The packages this rule covers: every one of them, since none of them is a platform. */
function packages(): string[] {
  const root = join(editorRoot, 'packages');
  return readdirSync(root, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => `packages/${entry.name}`)
    .sort();
}

/** One occurrence of a platform name, with where it was written. */
interface Leak {
  readonly path: string;
  readonly line: number;
  readonly name: string;
}

/**
 * The platform names a source **writes**, as TypeScript's own parser sees them.
 *
 * Parsed and not grepped, for the reason feature 1.12 established: a comment that says a component
 * must not call `localStorage` is documentation, and a check that reported it would be deleted
 * within the month. Only an identifier — a global read, a property named, a type written, a module
 * imported — is an occurrence.
 */
export function leaksIn(path: string, text: string, names: readonly string[]): Leak[] {
  const source = ts.createSourceFile(path, text, ts.ScriptTarget.ESNext, true, ts.ScriptKind.TSX);
  const wanted = new Set(names);
  const found: Leak[] = [];
  const at = (node: ts.Node, name: string): void => {
    found.push({ path, line: source.getLineAndCharacterOfPosition(node.getStart(source)).line + 1, name });
  };
  const walk = (node: ts.Node): void => {
    if ((ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) && node.moduleSpecifier !== undefined) {
      const specifier = node.moduleSpecifier;
      if (ts.isStringLiteral(specifier)) {
        const reason = forbiddenImport(specifier.text);
        if (reason !== null) at(specifier, `import of ${specifier.text} (${reason})`);
      }
    } else if (ts.isIdentifier(node) && wanted.has(node.text)) {
      at(node, node.text);
    }
    ts.forEachChild(node, walk);
  };
  walk(source);
  return found;
}

const scanned = packages().flatMap((pkg) => sourcesOf(pkg));
const leaks = scanned.flatMap((path) => leaksIn(path, readEditorFile(path), PLATFORM_GLOBALS));

describe('the sources this rule covers', () => {
  it('is every package, because none of them is a platform', () => {
    expect(packages()).toEqual(['packages/lang', 'packages/store', 'packages/ui']);
    expect(scanned.length).toBeGreaterThan(50);
    // `apps/` is where a platform belongs, so it is deliberately out of scope.
    expect(scanned.filter((path) => path.startsWith('apps/'))).toEqual([]);
  });
});

describe('no package reaches for a platform', () => {
  it('names no storage, no file handle, no download and no Electron', () => {
    expect(leaks.map((leak) => `${leak.path}:${String(leak.line)}: ${leak.name}`)).toEqual([]);
  });

  it('carries the interfaces every one of them goes through instead', () => {
    // The exemption's other half, feature 1.3's lesson: an exemption ships with a test that names
    // what uses it. The rule above is livable only because `Platform` exists and is complete.
    const surface = readEditorFile('packages/store/src/platform/index.ts');
    for (const member of ['Platform', 'Workspace', 'SettingsStore', 'DraftStore', 'Shell', 'AuthProvider', 'CheckpointSource']) {
      expect(surface, `${member} is not exported from the platform surface`).toContain(member);
    }
    const manifest = JSON.parse(readEditorFile('packages/store/package.json')) as {
      exports: Record<string, string>;
    };
    expect(manifest.exports['./platform']).toBe('./src/platform/index.ts');
  });
});

describe('the scan itself', () => {
  const bite = (text: string): string[] =>
    leaksIn('packages/ui/src/probe.ts', text, PLATFORM_GLOBALS).map((leak) => leak.name);

  it('reports a global, a property and a type', () => {
    expect(bite('export const theme = window.localStorage.getItem("t");')).toEqual(['localStorage']);
    expect(bite('export const db = indexedDB.open("x");')).toEqual(['indexedDB']);
    expect(bite('export function open(handle: FileSystemDirectoryHandle): void {}')).toEqual([
      'FileSystemDirectoryHandle',
    ]);
    expect(bite('export const it = await fetch("/vendor/vendor.json");')).toEqual(['fetch']);
  });

  it('reports an import of Node, of Electron and of an application', () => {
    expect(bite('import { readFileSync } from "node:fs";')).toEqual(['import of node:fs (Node)']);
    expect(bite('import { ipcRenderer } from "electron";')).toEqual([
      'import of electron (Electron)',
      'ipcRenderer',
    ]);
    expect(bite('import { x } from "../../apps/web/src/platform/index.js";')).toEqual([
      'import of ../../apps/web/src/platform/index.js (an application)',
    ]);
  });

  it('reports nothing for prose or for a name of one’s own', () => {
    expect(bite('// the settings of §5.2 live in localStorage, which is the platform’s\nexport const a = 1;')).toEqual(
      [],
    );
    expect(bite('export const message = "localStorage is the platform’s";')).toEqual([]);
  });
});

describe('the pages a build emits', () => {
  it('deploys the application alone', () => {
    // `stub.html` and the browser layer's driver are built for `pnpm check` and for nothing else,
    // so neither can reach a deployment by accident (feature 2.18 inherits the decision).
    expect(Object.keys(inputsFor('production'))).toEqual(['main']);
    expect(Object.keys(inputsFor(CHECK_MODE)).sort()).toEqual(['main', 'platform', 'stub']);
  });

  it('names pages that exist', () => {
    for (const page of Object.values(inputsFor(CHECK_MODE))) {
      expect(existsSync(page), `${page} is not there`).toBe(true);
    }
  });
});
