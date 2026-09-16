/**
 * Where the static application is published — the implementation plan's D11, settled here.
 *
 * > The first target is a static build — HTML, JS, the vendored schemas and reference base —
 * > deployed on GitHub Pages **beside the documentation site** (under the site's bar, as the model
 * > views are), with no server at all.
 *
 * Two facts follow from "beside", and this module is the one place that states either.
 *
 * **The base path.** A page served from a subdirectory must name its own assets under that
 * directory, so the build needs the path before it runs. Feature 0.6 measured the three
 * candidates and left the decision here: the documentation site is published at
 * `https://maneex.github.io/tensorspine/` and the editor sits beside its pages, hence
 * `/tensorspine/editor/`. The site root is **read**, not typed — {@link siteRootOf} takes it from
 * the repository's own documents, which name it seventeen times and agree — so a site that moves
 * moves the editor with it, and no host appears in any source of the application. A deployment
 * that knows better says so: `TENSORSPINE_EDITOR_BASE`, which is what the Pages workflow passes
 * from `actions/configure-pages`' own `base_path`.
 *
 * **The directory.** `editor/`, one level below the site root, which is what
 * `documentationBase()` in `@tensorspine/ui/shell` already assumes when it resolves the Help
 * menu's links against the directory *above* the page. The two must agree, and a test holds them
 * to each other.
 *
 * Nothing here is read at runtime: `base` is a build parameter, and what the page itself does is
 * resolve everything — the vendor, the documentation, the site's own bar — relative to its own
 * address.
 */
import { readFileSync, readdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

/** `editor/apps/web/`, this file's own directory. */
const here = dirname(fileURLToPath(import.meta.url));

/** The repository the editor is part of (D14). */
export const repositoryRoot = resolve(here, '..', '..', '..');

/**
 * The directory the editor is published in, below the site root.
 *
 * Not a fact of the language and not a name any schema carries: it is the editor's own place on
 * the site, and the one other place that depends on it is `documentationBase()`, which walks one
 * directory up from the page to find the documentation.
 */
export const EDITOR_DIRECTORY = 'editor';

/** The environment variable a deployment sets when it publishes somewhere else. */
export const BASE_VARIABLE = 'TENSORSPINE_EDITOR_BASE';

/** The documents the site root is read from: the README and every guide. */
export function siteDocuments(root: string = repositoryRoot): string[] {
  const docs = readdirSync(join(root, 'docs'))
    .filter((name) => name.endsWith('.md'))
    .sort()
    .map((name) => join(root, 'docs', name));
  return [join(root, 'README.md'), ...docs];
}

/** Every published site URL a text names, as `https://<host>/<first segment>/`. */
export function siteUrlsIn(text: string): string[] {
  return [...text.matchAll(/https:\/\/[A-Za-z0-9.-]+\.github\.io\/[A-Za-z0-9._-]+\//g)].map(
    (match) => match[0],
  );
}

/**
 * The path the documentation site is published under, read from the repository's own documents.
 *
 * The site is named by absolute URL wherever a document links to a page of it — the status page,
 * the primitive library reference, the overview — and every one of them agrees. Disagreement is a
 * refusal rather than a choice: which of two sites the editor sits beside is not this module's to
 * decide.
 */
export function siteRootOf(texts: readonly string[]): string {
  const found = new Set(texts.flatMap((text) => siteUrlsIn(text)));
  if (found.size === 0) {
    throw new Error(
      'no published site URL in the repository’s documents: the editor is deployed beside the ' +
        `documentation site (D11), so its path is the site's — set ${BASE_VARIABLE} to publish elsewhere`,
    );
  }
  if (found.size > 1) {
    throw new Error(
      `the repository’s documents name ${String(found.size)} different sites (${[...found].sort().join(', ')}) — ` +
        `which one the editor sits beside cannot be read, so set ${BASE_VARIABLE}`,
    );
  }
  return new URL([...found][0] as string).pathname;
}

/** A base path as a browser reads one: a leading and a trailing `/`, no doubled separator. */
export function normaliseBase(path: string): string {
  const segments = path.split('/').filter((segment) => segment !== '');
  return segments.length === 0 ? '/' : `/${segments.join('/')}/`;
}

/**
 * The published base path: the site's own path with the editor's directory below it.
 *
 * Computed from the repository each time rather than written down, so that the audit reading the
 * documents and the build reading them cannot part company.
 */
export function publishedBase(root: string = repositoryRoot): string {
  const texts = siteDocuments(root).map((path) => readFileSync(path, 'utf8'));
  return normaliseBase(`${siteRootOf(texts)}/${EDITOR_DIRECTORY}/`);
}

/**
 * The base this build is for: what the deployment says, else where the repository publishes.
 *
 * The variable wins because the deployment knows its own server — the Pages workflow passes
 * `actions/configure-pages`' `base_path`, so a repository renamed or a site moved needs no edit
 * here — and an empty value is no value, not the origin root.
 */
export function baseFor(
  environment: Readonly<Record<string, string | undefined>> = process.env,
  root: string = repositoryRoot,
): string {
  const said = environment[BASE_VARIABLE];
  if (said !== undefined && said.trim() !== '') return normaliseBase(said.trim());
  return publishedBase(root);
}
