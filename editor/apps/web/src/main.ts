import { packageName as lang } from '@tensorspine/lang';
import { packageName as store } from '@tensorspine/store';
import { packageName as ui } from '@tensorspine/ui';

/**
 * The application's entry point. The shell, the workspace and the activities arrive with the
 * features that build them; what it does today is what the skeleton has to prove — that the
 * three packages are linked into the static build and reach the page.
 */
const root = document.querySelector('#root');
if (root instanceof HTMLElement) {
  root.textContent = `TensorSpine Editor — ${lang}, ${store}, ${ui}`;
  root.dataset['state'] = 'ready';
}
