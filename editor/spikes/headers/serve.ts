import { createReadStream } from 'node:fs';
import { readFile, stat } from 'node:fs/promises';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { dirname, extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { build } from 'vite';

import type { EngineReport } from './report.ts';

/**
 * The spike's page, built and served — feature 0.5.
 *
 * The page is a static build, not a dev server, because that is what the application will be
 * (plan §2 D11): what the engines here load is the same kind of artefact GitHub Pages would
 * serve. Two consumers:
 *
 *   - Playwright starts it as a second `webServer` (`apps/web/playwright.config.ts`) and drives
 *     the page from Node;
 *   - `run.ts` starts it in-process and points Firefox and WebKitGTK at a URL, then waits for
 *     the report each one posts back to `POST /report` — the only channel an unautomated
 *     browser has.
 *
 * It serves the built directory and nothing else: a request that escapes it is refused.
 */

const here = dirname(fileURLToPath(import.meta.url));

/** Where the build lands. Gitignored, like every `dist/` of this workspace. */
export const DIST = join(here, 'dist');

/** The port the Playwright configuration and the runner agree on. */
export const DEFAULT_PORT = 4174;

const TYPES: Readonly<Record<string, string>> = {
  '.css': 'text/css; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.wasm': 'application/wasm',
};

/** A running spike server. */
export interface SpikeServer {
  readonly url: string;
  /** Every report posted so far, in the order they arrived. */
  readonly reports: readonly EngineReport[];
  /** Resolves with the next report an engine posts, or rejects when the wait runs out. */
  next(timeoutMs: number): Promise<EngineReport>;
  close(): Promise<void>;
}

/** Build the page with Vite. Answers what the build wrote, for the note's bundle figures. */
export async function buildPage(): Promise<Record<string, number>> {
  await build({
    root: join(here, 'page'),
    base: './',
    logLevel: 'warn',
    build: { outDir: DIST, emptyOutDir: true, target: 'es2022', reportCompressedSize: false },
  });
  const manifest: Record<string, number> = {};
  const assets = join(DIST, 'assets');
  const { readdir } = await import('node:fs/promises');
  for (const name of await readdir(assets)) {
    manifest[name] = (await stat(join(assets, name))).size;
  }
  manifest['index.html'] = (await stat(join(DIST, 'index.html'))).size;
  return manifest;
}

/** Start the server, building the page first unless `build` says otherwise. */
export async function startSpikeServer(
  options: { port?: number; build?: boolean } = {},
): Promise<SpikeServer> {
  if (options.build !== false) await buildPage();

  const reports: EngineReport[] = [];
  const waiting: { resolve: (report: EngineReport) => void }[] = [];

  const server: Server = createServer((request: IncomingMessage, response: ServerResponse) => {
    void handle(request, response, reports, waiting);
  });
  const port = options.port ?? DEFAULT_PORT;
  await new Promise<void>((done) => server.listen(port, '127.0.0.1', done));

  return {
    url: `http://127.0.0.1:${String(port)}/`,
    reports,
    next: (timeoutMs: number) =>
      new Promise<EngineReport>((resolveNext, reject) => {
        const timer = setTimeout(() => reject(new Error(`no report within ${String(timeoutMs)} ms`)), timeoutMs);
        waiting.push({
          resolve: (report) => {
            clearTimeout(timer);
            resolveNext(report);
          },
        });
      }),
    close: () =>
      new Promise<void>((done, fail) => {
        server.closeAllConnections();
        server.close((error) => {
          if (error) fail(error);
          else done();
        });
      }),
  };
}

async function handle(
  request: IncomingMessage,
  response: ServerResponse,
  reports: EngineReport[],
  waiting: { resolve: (report: EngineReport) => void }[],
): Promise<void> {
  const url = new URL(request.url ?? '/', 'http://127.0.0.1');
  if (request.method === 'POST' && url.pathname === '/report') {
    const chunks: Buffer[] = [];
    for await (const chunk of request) chunks.push(chunk as Buffer);
    try {
      const report = JSON.parse(Buffer.concat(chunks).toString('utf8')) as EngineReport;
      reports.push(report);
      for (const one of waiting.splice(0)) one.resolve(report);
      response.writeHead(204).end();
    } catch (error) {
      response.writeHead(400, { 'content-type': 'text/plain' }).end(String(error));
    }
    return;
  }
  if (request.method === 'GET' && url.pathname === '/reports') {
    response.writeHead(200, { 'content-type': TYPES['.json'] ?? 'application/json' }).end(JSON.stringify(reports));
    return;
  }
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    response.writeHead(405).end();
    return;
  }

  const wanted = url.pathname === '/' ? '/index.html' : url.pathname;
  const path = resolve(join(DIST, normalize(wanted)));
  if (!path.startsWith(DIST)) {
    response.writeHead(403).end();
    return;
  }
  try {
    const size = (await stat(path)).size;
    response.writeHead(200, {
      'content-type': TYPES[extname(path)] ?? 'application/octet-stream',
      'content-length': size,
      // The page reads its own bytes only; nothing here needs a cache.
      'cache-control': 'no-store',
    });
    if (request.method === 'HEAD') response.end();
    else createReadStream(path).pipe(response);
  } catch {
    response.writeHead(404, { 'content-type': 'text/plain' }).end('not found');
  }
}

/** `node --experimental-strip-types serve.ts [--port N] [--no-build]` — Playwright's webServer. */
async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const portArgument = argv.indexOf('--port');
  const port = portArgument < 0 ? DEFAULT_PORT : Number(argv[portArgument + 1]);
  const server = await startSpikeServer({ port, build: !argv.includes('--no-build') });
  process.stdout.write(`spike 0.5 headers: ${server.url}\n`);
  // Read once so that a stale build cannot be served silently.
  await readFile(join(DIST, 'index.html'), 'utf8');
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? '').href) await main();
