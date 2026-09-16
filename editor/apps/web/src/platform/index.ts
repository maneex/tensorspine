/**
 * `BrowserPlatform` — the static application's implementation of §5.2 (feature 2.4).
 *
 * D11's first deployment: a static build, no server at all, the workspace a local folder the
 * browser opens. Everything below is assembled here and nothing else in the application may reach
 * for a browser storage API, a file handle or a picker — that is what makes the same renderer run
 * against the stub of `@tensorspine/store/platform` in CI, and what the Electron and SaaS
 * increments replace without touching a component.
 *
 * | §5.2 | Here |
 * |---|---|
 * | `workspace` | `DirectoryWorkspace` (File System Access), `ReadOnlyWorkspace` over an upload, a drop or the vendored Examples |
 * | `checkpoints` | none yet: feature 4.1 adds `LocalCheckpoint` and `HubCheckpoint` |
 * | `auth` | `NoAuth` — the static application has no accounts (Q8 defers the SaaS) |
 * | `settings` | `localStorage`, loaded once, written through |
 * | `drafts` | IndexedDB |
 * | `shell` | the download, the clipboard, an external link, `window.confirm` |
 */
import { noAuth, type Platform, type Workspace } from '@tensorspine/store/platform';

import { browserDrafts } from './drafts.js';
import { browserSettings } from './settings.js';
import { browserShell, type BrowserShell } from './shell.js';
import { BrowserWorkspaces } from './workspaces.js';

/** What {@link createBrowserPlatform} is given. */
export interface BrowserPlatformOptions {
  /** Where the vendored schemas, corpus and reference base are served from. */
  readonly vendor?: string;
  /** How often a writable workspace polls for changes made behind it. */
  readonly pollMs?: number;
  /** `false` builds a download without handing it to the browser — an unattended run's. */
  readonly deliver?: boolean;
}

/** The platform of the static application, with the browser's own objects behind it. */
export interface BrowserPlatform extends Platform {
  readonly workspaces: BrowserWorkspaces;
  readonly shell: BrowserShell;
}

/**
 * Build the platform.
 *
 * Asynchronous because IndexedDB is: the draft store and the remembered folders are probed once,
 * here, so that a browser which refuses them is known before the first autosave rather than at it.
 */
export async function createBrowserPlatform(
  options: BrowserPlatformOptions = {},
): Promise<BrowserPlatform> {
  const shell = browserShell(options.deliver === undefined ? {} : { deliver: options.deliver });
  const workspaces = await BrowserWorkspaces.create({
    deliver: (name, text) => shell.download(name, text),
    ...(options.vendor === undefined ? {} : { vendor: options.vendor }),
    ...(options.pollMs === undefined ? {} : { pollMs: options.pollMs }),
  });
  const settings = browserSettings();
  const drafts = await browserDrafts();
  return {
    get workspace(): Workspace {
      return workspaces.current();
    },
    workspaces,
    checkpoints: [],
    auth: noAuth('this deployment has no accounts: everything the editor computes runs in your browser'),
    settings,
    drafts,
    shell,
    describe: () => 'browser',
  };
}

export { DirectoryWorkspace, DEFAULT_POLL_MS, type DirectoryOptions } from './directory.js';
export {
  Vendor,
  VENDOR,
  type VendorBundle,
  type VendorFile,
  type VendorManifest,
} from './examples.js';
export { Recents, type Permission, type RememberedWorkspace } from './recents.js';
export { browserDrafts, draftsOver } from './drafts.js';
export { browserSettings, PREFIX } from './settings.js';
export { appleConventions, browserShell, modifierOf, type BrowserShell, type LastDownload, type ShellOptions } from './shell.js';
export { BrowserWorkspaces, hasDirectoryPicker, type BrowserWorkspacesOptions } from './workspaces.js';
