import { packageName as lang } from '@tensorspine/lang';
import { packageName as store } from '@tensorspine/store';
import type { Platform } from '@tensorspine/store/platform';
import { packageName as ui } from '@tensorspine/ui';

/**
 * The application, whichever platform it was built against.
 *
 * The shell, the activities and the canvas arrive with the features that build them (2.5 onwards);
 * what is here is the seam D11 asks for — **one renderer, three deployments**. This function
 * names no browser API and no Node module: everything platform-shaped reaches it as a
 * {@link Platform}, so the same code runs against the browser's (`main.ts`) and against the stub
 * CI builds it with (`stub.ts`), and a leak would be a build failure in the second.
 */
export function start(platform: Platform): void {
  const root = document.querySelector('#root');
  if (!(root instanceof HTMLElement)) return;
  const show = (): void => {
    const workspace = platform.workspace.root();
    root.textContent = `TensorSpine Editor — ${lang}, ${store}, ${ui}`;
    root.dataset['platform'] = platform.describe();
    root.dataset['workspace'] = workspace.kind;
    root.dataset['writable'] = String(workspace.writable);
    root.dataset['settings'] = String(platform.settings.persistent);
    root.dataset['drafts'] = String(platform.drafts.persistent);
    root.dataset['state'] = 'ready';
  };
  platform.workspaces.onChange(show);
  show();
}
