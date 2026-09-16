#!/usr/bin/env node
/**
 * The fixture server of feature 2.20: **published file sets, served from another origin.**
 *
 * The feature's block asks for "Playwright against a fixture server", and what that server has to
 * be is a host that is not the application's: the editor is served at one port and this at
 * another, which is a different origin to a browser, so every claim the suite makes about CORS,
 * about a manifest and about a digest is a claim about a real cross-origin fetch.
 *
 * Four sets, each **generated** by `pnpm manifest` — never written by hand, which is the rule the
 * feature rests on — into a directory of its own under a temporary root:
 *
 *   `/workspace/`     a workspace: one model, the reference base beside it, and the template
 *                     document that base pins. The model declares a base the set does **not**
 *                     hold (`../lab-base/`), so it opens with the loader's V1 until one is stood
 *                     there — which is what makes "a fetched base is gathered" a visible change
 *                     rather than a claim.
 *   `/lab-base/`      the base that fills it: feature 1.13's acceptance-fixture base, which is
 *                     exactly what a laboratory's own base looks like — its axes and its
 *                     precision roles are the reference base's, and it declares the one primitive
 *                     the reference base has not.
 *   `/corrupt-base/`  the same, published and then **edited**: a file whose sha256 no longer
 *                     matches what the manifest declares.
 *   `/nocors-base/`   the same again, served with no `Access-Control-Allow-Origin` at all.
 *
 * Nothing here is an expectation: every byte is the repository's own, and what the suite compares
 * is what the editor makes of it.
 *
 * Usage:  node --experimental-strip-types apps/web/e2e/remote-server.ts --port 4177
 */
import { createServer } from 'node:http';
import {
  cpSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { publish } from '../../../scripts/publish.ts';

/** `editor/`, this file living in `editor/apps/web/e2e/`. */
const editorRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
const repositoryRoot = resolve(editorRoot, '..');

/** The set a request's path names, and what is under it. */
export const WORKSPACE = 'workspace';
export const BASE = 'lab-base';
export const CORRUPT = 'corrupt-base';
export const NO_CORS = 'nocors-base';

/** The one file the corrupt set serves differently from what its manifest declares. */
export const CORRUPTED = 'primitives/fixture/position_bias/1.0.0.json';

/** The document the published workspace holds, and the base it declares and does not carry. */
export const MODEL = 'models/uses-fixture.json';

/**
 * Build the four sets under one root and publish each of them.
 *
 * The workspace's document is `editor/tests/fixtures/models/declared-constant.json` with its two
 * bases written where this workspace keeps them — an *input*, edited the way a person opening the
 * editor on someone else's workspace would have written it, and not an expectation of anything.
 */
export function buildSets(): string {
  const root = mkdtempSync(join(tmpdir(), 'tensorspine-published-'));

  // The workspace: the reference base, the template document it pins, and one model.
  const workspace = join(root, WORKSPACE);
  mkdirSync(join(workspace, 'models'), { recursive: true });
  cpSync(join(repositoryRoot, 'data', 'primitive-library'), join(workspace, 'primitive-library'), {
    recursive: true,
  });
  cpSync(
    join(repositoryRoot, 'data', 'models', 'decoder-causal-yarn'),
    join(workspace, 'models', 'decoder-causal-yarn'),
    { recursive: true },
  );
  const document = readFileSync(
    join(editorRoot, 'tests', 'fixtures', 'models', 'declared-constant.json'),
    'utf8',
  )
    .replace('../../../../data/primitive-library/', '../primitive-library/')
    .replace('../base/', `../${BASE}/`);
  writeFileSync(join(workspace, MODEL), document);
  publish({ directory: workspace, quiet: true });

  // The base, three times: as it is, corrupted after publication, and behind a host that permits
  // nothing.
  for (const name of [BASE, CORRUPT, NO_CORS]) {
    const at = join(root, name);
    cpSync(join(editorRoot, 'tests', 'fixtures', 'base'), at, { recursive: true });
    publish({ directory: at, quiet: true, bundle: name !== CORRUPT });
  }
  // Edited *after* the manifest was written, which is what a base someone changed without
  // republishing looks like from an address — and what the digest is there to catch.
  const corrupted = join(root, CORRUPT, CORRUPTED);
  writeFileSync(corrupted, readFileSync(corrupted, 'utf8').replace('"positions"', '"positionz"'));

  return root;
}

/** What a path is served as; everything a published set holds is text. */
function typeOf(path: string): string {
  if (path.endsWith('.json')) return 'application/json; charset=utf-8';
  if (path.endsWith('.md')) return 'text/markdown; charset=utf-8';
  return 'text/plain; charset=utf-8';
}

function main(argv: string[]): void {
  let port = 4177;
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === '--port') {
      port = Number(argv[index + 1] ?? port);
      index += 1;
    }
  }
  const root = buildSets();
  // What was built for this run goes with it. Feature 0.6's review repair `0bfc6b3` is the same
  // rule one spike along ("the headers spike left its 64 MiB OPFS file behind"): a fixture server
  // that is killed — which is how Playwright ends one — takes its temporary tree with it.
  const clean = (): void => {
    rmSync(root, { recursive: true, force: true });
  };
  process.on('exit', clean);
  for (const signal of ['SIGINT', 'SIGTERM', 'SIGHUP'] as const) {
    process.on(signal, () => {
      clean();
      process.exit(0);
    });
  }
  const server = createServer((request, response) => {
    const path = decodeURIComponent((request.url ?? '/').split('?')[0] ?? '/').replace(/^\/+/, '');
    // Everything but one set answers the header a page on another origin needs; that one is the
    // measurement of what happens when a host does not (feature 0.5's own reading, one host over).
    const permitted = !path.startsWith(`${NO_CORS}/`);
    if (permitted) {
      response.setHeader('access-control-allow-origin', '*');
      response.setHeader('access-control-allow-methods', 'GET');
    }
    if (request.method === 'OPTIONS') {
      response.writeHead(permitted ? 204 : 403).end();
      return;
    }
    const file = join(root, path);
    if (path === '' || !file.startsWith(root) || !existsSync(file) || !statSync(file).isFile()) {
      response.writeHead(404, { 'content-type': 'text/plain' }).end('not here\n');
      return;
    }
    const bytes = readFileSync(file);
    response.writeHead(200, { 'content-type': typeOf(path), 'content-length': bytes.length });
    response.end(bytes);
  });
  server.listen(port, '127.0.0.1', () => {
    console.log(`published sets on http://127.0.0.1:${String(port)}/ from ${root}`);
  });
}

const invoked = process.argv[1];
if (invoked !== undefined && resolve(invoked) === fileURLToPath(import.meta.url)) {
  main(process.argv.slice(2));
}
