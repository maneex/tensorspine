import { createReadStream } from 'node:fs';
import { readdir, readFile, stat } from 'node:fs/promises';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { dirname, extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { build } from 'vite';

/**
 * The page feature 1.11's browser layer drives, built and served.
 *
 * What it is for: the unit layer exercises the protocol over a `MessageChannel`, which is a true
 * structured-clone boundary but one thread. A **real** `Worker` is another thread, and three
 * claims of §5.3 and §5.6 are only true there — that validation and derivation never block the
 * interface, that a cancel message overtakes the work it cancels, and that the core reaches the
 * worker's chunk and not the page's. This page is where a browser is asked.
 *
 * It is the application's own worker under test: the page imports
 * `apps/web/src/lang/connect.ts`, so what Vite emits is the very `new Worker(new URL(…))` the
 * static build emits.
 *
 * The material — the schemas, the reference base, two corpus documents — is served from the
 * repository at `/material.json` rather than from `apps/web/public/vendor/`, which `pnpm vendor`
 * writes and the CI job does not run (feature 0.2). The page therefore reads the repository's own
 * files, which is what the editor reads in development anyway (D14).
 */

const here = dirname(fileURLToPath(import.meta.url));
const repository = resolve(here, '..', '..', '..');

/** Where the build lands. Gitignored, like every `dist/` of this workspace. */
export const DIST = join(here, 'dist');

/** The port the Playwright configuration agrees on. */
export const DEFAULT_PORT = 4176;

/** The two documents the page works on: the smallest of the corpus, and the largest. */
export const DOCUMENTS = ['llama3-8b', 'deepseek-v4-pro'] as const;

const TYPES: Readonly<Record<string, string>> = {
  '.css': 'text/css; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
};

/** What the page loads the core with: exactly what `Platform.workspace` will hand it (§5.2). */
export interface Material {
  readonly schemas: Record<string, string>;
  readonly base: { base: string; files: Record<string, string> };
  readonly documents: Record<string, string>;
}

/** Every `.json` under a directory of the repository, by repository-relative path. */
async function filesUnder(directory: string): Promise<Record<string, string>> {
  const found: Record<string, string> = {};
  const walk = async (current: string): Promise<void> => {
    for (const entry of await readdir(join(repository, current), { withFileTypes: true })) {
      const path = `${current}/${entry.name}`;
      if (entry.isDirectory()) await walk(path);
      else if (entry.name.endsWith('.json')) found[path] = await readFile(join(repository, path), 'utf8');
    }
  };
  await walk(directory);
  return found;
}

/** The material, read from the repository once per server. */
export async function material(): Promise<Material> {
  const documents: Record<string, string> = {};
  for (const name of DOCUMENTS) {
    documents[name] = await readFile(join(repository, 'data', 'models', `${name}.json`), 'utf8');
  }
  return {
    schemas: await filesUnder('schemas'),
    base: {
      base: 'data/primitive-library',
      files: {
        ...(await filesUnder('data/primitive-library')),
        // The template documents the manifest pins (`"templates": "../models/"`).
        ...(await filesUnder('data/models/decoder-causal-yarn')),
      },
    },
    documents,
  };
}

/** Build the page with Vite, as a static build — which is what will be deployed (D11). */
export async function buildPage(): Promise<void> {
  await build({
    root: join(here, 'page'),
    base: './',
    logLevel: 'warn',
    build: { outDir: DIST, emptyOutDir: true, target: 'es2022', reportCompressedSize: false },
  });
}

/** Start the server, building the page first unless `build` says otherwise. */
export async function startServer(
  options: { port?: number; build?: boolean } = {},
): Promise<{ url: string; close: () => Promise<void> }> {
  if (options.build !== false) await buildPage();
  const held = await material();

  const server: Server = createServer((request: IncomingMessage, response: ServerResponse) => {
    void handle(request, response, held);
  });
  const port = options.port ?? DEFAULT_PORT;
  await new Promise<void>((done) => server.listen(port, '127.0.0.1', done));

  return {
    url: `http://127.0.0.1:${String(port)}/`,
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
  held: Material,
): Promise<void> {
  const url = new URL(request.url ?? '/', 'http://127.0.0.1');
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    response.writeHead(405).end();
    return;
  }
  if (url.pathname === '/material.json') {
    const body = JSON.stringify(held);
    response
      .writeHead(200, { 'content-type': TYPES['.json'] as string, 'cache-control': 'no-store' })
      .end(request.method === 'HEAD' ? undefined : body);
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
  const server = await startServer({ port, build: !argv.includes('--no-build') });
  process.stdout.write(`feature 1.11: the language core in a worker — ${server.url}\n`);
  await readFile(join(DIST, 'index.html'), 'utf8');
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? '').href) await main();
