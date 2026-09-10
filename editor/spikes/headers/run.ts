#!/usr/bin/env node
import { spawn, type ChildProcess } from 'node:child_process';
import { existsSync, readdirSync, writeFileSync } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { NETWORK_CASES, OFFLINE_CASES, type EngineReport } from './report.ts';
import { buildPage, startSpikeServer } from './serve.ts';

/**
 * `pnpm spike:headers` — feature 0.5's cross-engine run.
 *
 * The feature's note has to say what works in Chromium, Firefox and Safari. Playwright's suite
 * answers for Chromium on every commit; the other two engines are not automated on this box, so
 * this runner does the only thing that gets a fact out of them: it serves the page, opens it in
 * each engine with `?cases=…`, and waits for the report the page posts back.
 *
 *     pnpm spike:headers               the cases that need no network
 *     pnpm spike:headers --hub         those, and the two Hub cases
 *     pnpm spike:headers --engines=firefox
 *
 * What it writes — `engines.json` beside this file — is the evidence the note quotes. Nothing
 * here runs in CI: the engines are whatever the box has, and the note names their versions.
 */

const here = dirname(fileURLToPath(import.meta.url));

/** The engines this box can be asked, and how to open a URL in each headlessly. */
interface Engine {
  readonly name: string;
  /** Answers the command and its arguments, or null when the engine is not installed. */
  readonly launch: (url: string, profile: string) => { command: string; args: string[] } | null;
}

function firstExisting(candidates: readonly string[]): string | null {
  return candidates.find((path) => existsSync(path)) ?? null;
}

/** WebKitGTK's MiniBrowser, wherever the distribution put it. */
function miniBrowser(): string | null {
  const roots = ['/usr/lib/x86_64-linux-gnu', '/usr/lib', '/usr/lib64'];
  for (const root of roots) {
    if (!existsSync(root)) continue;
    for (const entry of readdirSync(root)) {
      if (!entry.startsWith('webkit2gtk-') && !entry.startsWith('webkitgtk-')) continue;
      const path = join(root, entry, 'MiniBrowser');
      if (existsSync(path)) return path;
    }
  }
  return null;
}

const ENGINES: readonly Engine[] = [
  {
    name: 'chromium',
    launch: (url, profile) => {
      const command = firstExisting(['/usr/bin/chromium', '/usr/bin/chromium-browser', '/usr/bin/google-chrome']);
      return command === null
        ? null
        : {
            command,
            args: [
              '--headless=new',
              '--disable-gpu',
              '--no-first-run',
              '--no-default-browser-check',
              `--user-data-dir=${profile}`,
              url,
            ],
          };
    },
  },
  {
    name: 'firefox',
    launch: (url, profile) => {
      const command = firstExisting(['/usr/bin/firefox', '/usr/bin/firefox-esr']);
      return command === null ? null : { command, args: ['--headless', '--no-remote', '--profile', profile, url] };
    },
  },
  {
    name: 'webkitgtk',
    launch: (url) => {
      const browser = miniBrowser();
      if (browser === null || !existsSync('/usr/bin/xvfb-run')) return null;
      // MiniBrowser has no headless mode; a virtual display is the way to run it on a box
      // without one. `--automatic-servernum` picks a free display.
      return { command: '/usr/bin/xvfb-run', args: ['--auto-servernum', browser, url] };
    },
  },
];

/** How long an engine gets to load the page, run its cases and post the report. */
const TIMEOUT_MS = 300_000;

/** Stop an engine and its children — the browsers here fork — and wait for it to go. */
async function stop(child: ChildProcess): Promise<void> {
  if (child.pid === undefined || child.exitCode !== null) return;
  const gone = new Promise<void>((done) => child.once('exit', () => done()));
  try {
    process.kill(-child.pid, 'SIGTERM');
  } catch {
    child.kill('SIGKILL');
  }
  await Promise.race([gone, new Promise<void>((done) => setTimeout(done, 5_000))]);
}

/**
 * Remove a profile directory, giving a browser that is still writing to it a moment to stop.
 * Chromium flushes its profile after the page is gone, so the first attempt can find the
 * directory refilled.
 */
async function discard(path: string): Promise<void> {
  for (let attempt = 0; ; attempt += 1) {
    try {
      await rm(path, { recursive: true, force: true });
      return;
    } catch (error) {
      if (attempt >= 5) {
        process.stdout.write(`    could not remove ${path}: ${String(error)}\n`);
        return;
      }
      await new Promise((done) => setTimeout(done, 500));
    }
  }
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const hub = argv.includes('--hub');
  const asked = argv.find((argument) => argument.startsWith('--engines='))?.slice('--engines='.length);
  const wanted = asked === undefined ? ENGINES : ENGINES.filter((engine) => asked.split(',').includes(engine.name));
  const cases = [...OFFLINE_CASES, ...(hub ? NETWORK_CASES : [])];

  const bundle = await buildPage();
  const server = await startSpikeServer({ build: false });
  process.stdout.write(`serving ${server.url}\n`);

  const results: { engine: string; available: boolean; report?: EngineReport; failure?: string }[] = [];
  for (const engine of wanted) {
    const profile = await mkdtemp(join(tmpdir(), `spike-headers-${engine.name}-`));
    const launch = engine.launch(`${server.url}?cases=${cases.join(',')}&engine=${engine.name}`, profile);
    if (launch === null) {
      process.stdout.write(`${engine.name}: not installed\n`);
      results.push({ engine: engine.name, available: false });
      await rm(profile, { recursive: true, force: true });
      continue;
    }
    process.stdout.write(`${engine.name}: ${launch.command}\n`);
    const child: ChildProcess = spawn(launch.command, launch.args, { detached: true, stdio: ['ignore', 'pipe', 'pipe'] });
    const noise: string[] = [];
    child.stdout?.on('data', (chunk: Buffer) => noise.push(chunk.toString('utf8')));
    child.stderr?.on('data', (chunk: Buffer) => noise.push(chunk.toString('utf8')));
    try {
      const report = await server.next(TIMEOUT_MS);
      results.push({ engine: engine.name, available: true, report });
      for (const one of report.cases) {
        const verdict = one.skipped !== undefined ? `skipped (${one.skipped})` : one.ok ? 'ok' : `FAILED ${one.error ?? ''}`;
        process.stdout.write(`    ${one.name}: ${verdict}\n`);
      }
    } catch (error) {
      process.stdout.write(`    no report: ${String(error)}\n`);
      results.push({ engine: engine.name, available: true, failure: `${String(error)}\n${noise.join('').slice(-2000)}` });
    } finally {
      await stop(child);
      await discard(profile);
    }
  }

  writeFileSync(
    join(here, 'engines.json'),
    `${JSON.stringify({ measuredAt: new Date().toISOString(), node: process.version, cases, bundle, engines: results }, null, 2)}\n`,
    'utf8',
  );
  process.stdout.write(`wrote ${join(here, 'engines.json')}\n`);
  await server.close();
}

await main();
