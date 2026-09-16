/**
 * Vendor the documentation site's navigation — component inventory §2, artboard S18.
 *
 * > **Documentation-site bar** · `.docbar` · S18 · answered by `docs/style/nav.html`, the page the
 * > static build is served under.
 *
 * The editor is one page of the site (D11: "deployed on GitHub Pages beside the documentation
 * site … under the site's bar, as the model views are"), so it carries the site's bar, and the
 * bar's entries are the site's own — read out of `docs/style/nav.html`, which is the single
 * source `tools/site.sh` puts on every page. This script lifts them into
 * `packages/ui/src/shell/site-nav.ts` the way `scripts/logo.ts` lifts the monogram, and for the
 * same reason: the editor cannot fetch the site's copy (it is a page that must run from a folder
 * of its own, and from a workspace with no site at all), and a hand-written list would drift the
 * first time a guide is added.
 *
 * What travels is the *entries*, not the markup: each keeps the two things a bar needs — its href
 * below the site root and its text — and nothing else. The site's own pop-up menus are its
 * stylesheet's and the editor's bar draws one row (S18); the site's `data-nav` identity is its own
 * bookkeeping and stays behind (see below). `%ROOT%`, which `tools/site.sh` replaces with each
 * page's own path to the site root, is stripped here: the editor resolves the same hrefs against
 * the directory *above* itself, which is what `documentationBase()` already answers for the Help
 * menu.
 *
 * Run it with `pnpm site-nav`, from `editor/`. `tests/audit/site-bar.test.ts` reads the same
 * source and requires this file to equal what the script would write.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

/** `editor/`, this script's own workspace. */
export const editorRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');

/** The repository the editor is part of (D14). */
export const repositoryRoot = resolve(editorRoot, '..');

/** The site's one navigation source. */
export const NAV = 'docs/style/nav.html';

/** Where the copy lives. */
export const MODULE = 'packages/ui/src/shell/site-nav.ts';

/** The placeholder `tools/site.sh` replaces with a page's own path to the site root. */
const ROOT = '%ROOT%';

/** One entry of the site's navigation. */
export interface SiteEntry {
  /** Where the page is, below the site root. */
  readonly href: string;
  /** What the site calls it. */
  readonly label: string;
}

/** One group of entries, as the site arranges them. */
export interface SiteGroup {
  /** The container's class in `nav.html`, without its `nav-` prefix. */
  readonly name: string;
  /** The `<summary>` the site folds the group behind, or `null` for the entries it shows. */
  readonly label: string | null;
  readonly entries: readonly SiteEntry[];
}

/** The whole navigation: where the lockup points, and the groups. */
export interface SiteNav {
  readonly home: string;
  readonly groups: readonly SiteGroup[];
}

/** An href as the entry states it, below the site root; a refusal for anything else. */
function below(href: string, where: string): string {
  if (!href.startsWith(ROOT)) {
    throw new Error(`${NAV}: ${where} points at ${href}, which is not below ${ROOT}`);
  }
  return href.slice(ROOT.length);
}

/** The site's navigation, as `nav.html` states it. */
export function navOf(nav: string): SiteNav {
  const wordmark = /<a class="wordmark" href="([^"]+)"/.exec(nav);
  if (wordmark === null) throw new Error(`${NAV}: no <a class="wordmark"> to read the site root from`);
  const home = below(wordmark[1] ?? '', 'the lockup');

  const containers = [...nav.matchAll(/<div class="nav-([a-z]+)"/g)];
  if (containers.length === 0) throw new Error(`${NAV}: no <div class="nav-…"> group of entries`);
  const groups: SiteGroup[] = containers.map((container, index) => {
    const from = container.index;
    const to = index + 1 < containers.length ? (containers[index + 1]?.index ?? nav.length) : nav.length;
    const slice = nav.slice(from, to);
    const name = container[1] ?? '';
    const summary = /<summary>([^<]*)<\/summary>/.exec(slice);
    // An anchor of the site's navigation is one that carries a `data-nav`: that attribute is how
    // `tools/site.sh` marks the page being read, and it is what separates an entry from any other
    // link in the markup. The **value** does not travel: it is the site's own bookkeeping, the
    // editor is never one of the site's pages, and one of the values (`specification`) is also an
    // enum value of the documentation schema — which catching rule (b) reports in interface
    // source, rightly, a whole-literal scan having no way to tell one vocabulary from the other.
    const entries: SiteEntry[] = [...slice.matchAll(/<a href="([^"]+)" data-nav="([^"]+)">([^<]*)<\/a>/g)].map(
      (entry) => ({
        href: below(entry[1] ?? '', `the ${name} entry ${entry[2] ?? ''}`),
        label: entry[3] ?? '',
      }),
    );
    if (entries.length === 0) throw new Error(`${NAV}: the nav-${name} group holds no entry`);
    return { name, label: summary === null ? null : (summary[1] ?? ''), entries };
  });
  return { home, groups };
}

/** A string as the workspace writes one: single quotes, the two characters that need escaping. */
function quote(text: string): string {
  return `'${text.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`;
}

/** The module written beside the bar. */
export function moduleFor(nav: SiteNav): string {
  const entry = (one: SiteEntry): string =>
    `      { href: ${quote(one.href)}, label: ${quote(one.label)} },`;
  const group = (one: SiteGroup): string =>
    [
      '  {',
      `    name: ${quote(one.name)},`,
      `    label: ${one.label === null ? 'null' : quote(one.label)},`,
      '    entries: [',
      ...one.entries.map(entry),
      '    ],',
      '  },',
    ].join('\n');
  return `/**
 * The documentation site's navigation, vendored — component inventory §2, artboard S18.
 *
 * **Generated by \`editor/scripts/site-nav.ts\` (\`pnpm site-nav\`) from \`${NAV}\`. Do not
 * edit.** \`tests/audit/site-bar.test.ts\` reads the same source and requires this file to equal
 * it, so a guide added to the site appears in the editor's bar rather than being missing from it.
 *
 * Every href is stated below the site root, as \`nav.html\` states it with its \`%ROOT%\`
 * placeholder: the editor resolves them against the directory above its own page, which is where
 * the site is (D11, and \`documentationBase()\` for the Help menu's links).
 */

/** One entry of the site's navigation. */
export interface SiteEntry {
  /** Where the page is, below the site root. */
  readonly href: string;
  /** What the site calls it. */
  readonly label: string;
}

/** One group of entries, as the site arranges them. */
export interface SiteGroup {
  /** The container's class in \`nav.html\`, without its \`nav-\` prefix. */
  readonly name: string;
  /** The \`<summary>\` the site folds the group behind, or \`null\` for the entries it shows. */
  readonly label: string | null;
  readonly entries: readonly SiteEntry[];
}

/** Where the lockup points: the site's own front page. */
export const SITE_HOME = ${quote(nav.home)};

/** The site's entries, in the order \`nav.html\` writes them. */
export const SITE_NAV: readonly SiteGroup[] = [
${nav.groups.map(group).join('\n')}
];
`;
}

export function vendorSiteNav(): { bytes: number } {
  const nav = readFileSync(join(repositoryRoot, NAV), 'utf8');
  const text = moduleFor(navOf(nav));
  writeFileSync(join(editorRoot, MODULE), text);
  return { bytes: text.length };
}

const invoked = process.argv[1];
if (invoked !== undefined && resolve(invoked) === fileURLToPath(import.meta.url)) {
  const { bytes } = vendorSiteNav();
  process.stdout.write(`${MODULE}  ${(bytes / 1024).toFixed(1)} KiB\n`);
}
