import type { EngineReport } from '../report.ts';
import { capabilities, run } from './probe.ts';
import { Shell } from './shell.ts';

/**
 * The page — feature 0.6's spike.
 *
 * It is the shell of `shell.ts` plus the cases of `probe.ts`, and it is a **static build**: the
 * whole of it is HTML and JavaScript served from under a base path, with no server behind it
 * (D11). Two ways to drive it, as in feature 0.5: the specification calls `window.spike.run`,
 * and an engine nobody automates is given `?cases=…&engine=…` and posts its report back.
 */

declare global {
  interface Window {
    spike: {
      readonly run: (names: readonly string[], engine?: string, attended?: boolean) => Promise<EngineReport>;
      readonly capabilities: () => Record<string, boolean>;
      readonly shell: Shell;
    };
  }
}

const shell = new Shell();
window.spike = { run, capabilities, shell };

const parameters = new URLSearchParams(window.location.search);

void shell.start().then(async () => {
  document.body.dataset['ready'] = 'true';
  const asked = parameters.get('cases');
  if (asked === null) return;
  // Unattended: run on load, show the report, post it to the server that served the page.
  const report = await run(asked.split(','), parameters.get('engine') ?? 'unnamed', false);
  const out = document.getElementById('report');
  if (out !== null) out.textContent = JSON.stringify(report, null, 2);
  document.body.dataset['cases'] = report.cases.every((one) => one.ok || one.skipped !== undefined) ? 'ok' : 'failed';
  await fetch('report', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(report),
  }).catch(() => undefined);
});
