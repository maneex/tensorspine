#!/usr/bin/env node
import { spawn, type ChildProcess } from 'node:child_process';
import { existsSync, readdirSync, writeFileSync } from 'node:fs';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { PAGE_CASES, type EngineReport } from './report.ts';
import { buildPage, DEFAULT_BASE, startStaticServer } from './serve.ts';

/**
 * `pnpm spike:static` — feature 0.6's cross-engine run.
 *
 * The feature's note has to say **which browsers have the writable picker**, and the answer is
 * not something to reason about: it is measured here. Playwright's suite answers for Chromium
 * on every commit; the other two engines on this box are not automated, so this runner serves
 * the built page under its base path, opens it in each engine with `?cases=…`, and waits for
 * the report the page posts back.
 *
 *     pnpm spike:static
 *     pnpm spike:static --engines=firefox
 *
 * It also builds the page under three base paths and records what each build named its assets,
 * because that — not the server — is what decides whether the application can be served beside
 * the documentation site. What it writes is `engines.json`, the evidence `NOTE.md` quotes.
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
      // without one. `--auto-servernum` picks a free display.
      return { command: '/usr/bin/xvfb-run', args: ['--auto-servernum', browser, url] };
    },
  },
];

/** How long an engine gets to load the page, run its cases and post the report. */
const TIMEOUT_MS = 300_000;

/**
 * The base paths worth building under, and why. `/editor/` is what the feature names;
 * `/tensorspine/editor/` is what the published site would need, since the documentation site is
 * itself served under `/tensorspine/`; `./` is the prefix-agnostic build, which is how the
 * documentation site's own pages are written (`%ROOT%` in `docs/style/nav.html`).
 */
const BASES: readonly { label: string; base: string }[] = [
  { label: 'editor', base: DEFAULT_BASE },
  { label: 'site', base: '/tensorspine/editor/' },
  { label: 'relative', base: './' },
];

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

/** Remove a profile directory, giving a browser that is still writing to it a moment to stop. */
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

/** What one build under one base named its entry script — the base path, made visible. */
async function measureBases(): Promise<Record<string, { base: string; script: string; files: Record<string, number> }>> {
  const measured: Record<string, { base: string; script: string; files: Record<string, number> }> = {};
  for (const { label, base } of BASES) {
    const outDir = join(here, 'dist-bases', label);
    const files = await buildPage(base, outDir);
    const html = await readFile(join(outDir, 'index.html'), 'utf8');
    measured[label] = { base, script: /<script[^>]*src="([^"]+)"/.exec(html)?.[1] ?? '(none)', files };
  }
  return measured;
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const asked = argv.find((argument) => argument.startsWith('--engines='))?.slice('--engines='.length);
  const wanted = asked === undefined ? ENGINES : ENGINES.filter((engine) => asked.split(',').includes(engine.name));

  const bases = await measureBases();
  const server = await startStaticServer({});
  process.stdout.write(`serving ${server.url}\n`);

  const results: { engine: string; available: boolean; report?: EngineReport; failure?: string }[] = [];
  for (const engine of wanted) {
    const profile = await mkdtemp(join(tmpdir(), `spike-static-${engine.name}-`));
    const launch = engine.launch(`${server.url}?cases=${PAGE_CASES.join(',')}&engine=${engine.name}`, profile);
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
      process.stdout.write(`    writable picker: ${String(report.capabilities['showDirectoryPicker'])}\n`);
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
    `${JSON.stringify(
      { measuredAt: new Date().toISOString(), node: process.version, cases: PAGE_CASES, bases, engines: results },
      null,
      2,
    )}\n`,
    'utf8',
  );
  process.stdout.write(`wrote ${join(here, 'engines.json')}\n`);
  await server.close();
}

await main();
