import { createReadStream } from 'node:fs';
import { readFile, readdir, stat } from 'node:fs/promises';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { dirname, extname, join, normalize, relative, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { build } from 'vite';

import type { EngineReport } from './report.ts';

/**
 * The spike's page, built **under a base path** and served under it — feature 0.6.
 *
 * D11's first deployment is a static build "deployed on GitHub Pages beside the documentation
 * site", which means the application does not live at the root of its origin: it lives under a
 * path, and every asset it names has to agree. So this server does two things a plain static
 * server does not:
 *
 *   - it serves **only** under the base (`/editor/` unless told otherwise), so a build whose
 *     assets are named from the root fails here rather than passing by accident;
 *   - it accepts the report an unautomated engine posts back to `<base>report`, which is how
 *     `run.ts` measures Firefox and WebKitGTK.
 *
 * Two consumers, as in feature 0.5: Playwright starts it as a `webServer`
 * (`apps/web/playwright.config.ts`), and `run.ts` starts it in process.
 */

const here = dirname(fileURLToPath(import.meta.url));

/** Where the build lands. Gitignored, like every `dist/` of this workspace. */
export const DIST = join(here, 'dist');

/** The base path the feature names: the editor beside the documentation site. */
export const DEFAULT_BASE = '/editor/';

/** The port the Playwright configuration and the runner agree on. */
export const DEFAULT_PORT = 4175;

const TYPES: Readonly<Record<string, string>> = {
  '.css': 'text/css; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
};

/** A base path as this server wants it: one leading slash, one trailing slash. */
export function normaliseBase(base: string): string {
  const trimmed = base.replace(/^\.?\/*/, '').replace(/\/*$/, '');
  return trimmed === '' ? '/' : `/${trimmed}/`;
}

/** A running spike server. */
export interface StaticServer {
  /** Where the page is: the origin and the base path together. */
  readonly url: string;
  readonly origin: string;
  readonly base: string;
  /** Every report posted so far, in the order they arrived. */
  readonly reports: readonly EngineReport[];
  /** Resolves with the next report an engine posts, or rejects when the wait runs out. */
  next(timeoutMs: number): Promise<EngineReport>;
  close(): Promise<void>;
}

/** What one build wrote: the file names under `dist/` and their sizes. */
export type BuildManifest = Record<string, number>;

/** Build the page with Vite, under `base`. Answers what the build wrote. */
export async function buildPage(base = DEFAULT_BASE, outDir = DIST): Promise<BuildManifest> {
  await build({
    root: join(here, 'page'),
    base,
    logLevel: 'warn',
    build: { outDir, emptyOutDir: true, target: 'es2022', reportCompressedSize: false },
  });
  const manifest: BuildManifest = {};
  const walk = async (directory: string): Promise<void> => {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) await walk(path);
      else manifest[relative(outDir, path)] = (await stat(path)).size;
    }
  };
  await walk(outDir);
  return manifest;
}

/** Start the server, building the page first unless `build` says otherwise. */
export async function startStaticServer(
  options: { port?: number; base?: string; build?: boolean } = {},
): Promise<StaticServer> {
  const base = normaliseBase(options.base ?? DEFAULT_BASE);
  if (options.build !== false) await buildPage(base);

  const reports: EngineReport[] = [];
  const waiting: { resolve: (report: EngineReport) => void }[] = [];

  const server: Server = createServer((request: IncomingMessage, response: ServerResponse) => {
    void handle(request, response, base, reports, waiting);
  });
  const port = options.port ?? DEFAULT_PORT;
  await new Promise<void>((done) => server.listen(port, '127.0.0.1', done));
  const origin = `http://127.0.0.1:${String(port)}`;

  return {
    url: `${origin}${base}`,
    origin,
    base,
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
  base: string,
  reports: EngineReport[],
  waiting: { resolve: (report: EngineReport) => void }[],
): Promise<void> {
  const url = new URL(request.url ?? '/', 'http://127.0.0.1');
  if (request.method === 'POST' && url.pathname === `${base}report`) {
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
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    response.writeHead(405).end();
    return;
  }
  // The application lives under the base and nowhere else: a request to the origin's root is
  // not the application, and answering it would hide a build that ignored the base.
  if (!url.pathname.startsWith(base)) {
    response.writeHead(404, { 'content-type': 'text/plain' }).end(`nothing here; the application is at ${base}`);
    return;
  }

  const within = url.pathname.slice(base.length);
  const wanted = within === '' ? 'index.html' : within;
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
      // A page whose whole point is being reloaded must not be answered from a cache.
      'cache-control': 'no-store',
    });
    if (request.method === 'HEAD') response.end();
    else createReadStream(path).pipe(response);
  } catch {
    response.writeHead(404, { 'content-type': 'text/plain' }).end('not found');
  }
}

/** `node --experimental-strip-types serve.ts [--port N] [--base /editor/] [--no-build]`. */
async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const value = (flag: string): string | undefined => {
    const index = argv.indexOf(flag);
    return index < 0 ? undefined : argv[index + 1];
  };
  const port = Number(value('--port') ?? DEFAULT_PORT);
  const server = await startStaticServer({
    port,
    base: value('--base') ?? DEFAULT_BASE,
    build: !argv.includes('--no-build'),
  });
  process.stdout.write(`spike 0.6 static: ${server.url}\n`);
  // Read once so that a stale build cannot be served silently.
  await readFile(join(DIST, 'index.html'), 'utf8');
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? '').href) await main();
