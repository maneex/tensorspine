/**
 * The documentation site's bar — component inventory §2 (`.docbar`), artboard S18.
 *
 * > The editor is one page of the site, under the site's own bar, exactly as the model views are
 * > today — so a reader who opens a model view and wants to change it is already in the right
 * > place. (S18)
 *
 * D11 puts the static build **beside** the documentation site, and this is what that looks like:
 * above the application's own frame, the site's bar, with the site's lockup and the site's
 * entries. The entries are `docs/style/nav.html`'s, vendored by `scripts/site-nav.ts` and never
 * written here (`site-nav.ts`); the lockup is `docs/tensorspine.svg` placed and scaled as the
 * site places it, which is `logo.ts`'s vendored monogram and the same two measures the site uses.
 *
 * Three things this bar is deliberately **not**.
 *
 *  - It is not the application's menu bar. `Bar.tsx` carries §4.4's menus, the core's two pills
 *    and the palette; the two are stacked, and the one lockup on the page is this one — the
 *    application's bar drops its own when a site bar is above it, because a page with two
 *    identical lockups reads as two applications.
 *  - It is not a flat list of links. The site shows its four destinations and folds its seven
 *    guides and its three generated documents behind two menus, and that grouping is not
 *    decoration: fourteen entries in a row do not fit the width S18 itself draws, and a row that
 *    scrolls hides its tail with nothing to say it is there. So the arrangement is the site's —
 *    a `<details>` per group, the site's own element, which needs no script and takes the
 *    keyboard by itself — with the hairline the site puts before each.
 *  - It is not a router. Every href is resolved against `docsBase` — the directory above this
 *    page — so a build served from anywhere finds its own neighbour and no host is written down.
 *    The site's own `data-nav` identities do not travel either: they are how `tools/site.sh` marks
 *    the page being read, and the editor is never one of the site's pages.
 *
 * **The entry for the editor itself** is this component's, because `nav.html` has none: the site
 * does not yet link to the editor (a documentation change set is what would add it, and the
 * editor may not edit `docs/`). It is the current page, so it is a mark and not a link.
 */
import type { JSX } from 'react';

import { MONOGRAM } from './logo.js';
import { EDITOR_ENTRY, EDITOR_GROUP } from './site.js';
import { SITE_HOME, SITE_NAV } from './site-nav.js';
import { text } from './strings.js';

/** What {@link SiteBar} is given. */
export interface SiteBarProps {
  /**
   * Where the documentation site is, relative to this page — `documentationBase()`'s answer.
   *
   * Ends in a separator, so an entry's href is appended to it as written.
   */
  readonly docsBase: string;
}

/** The site's bar, above the application (S18). */
export function SiteBar({ docsBase }: SiteBarProps): JSX.Element {
  return (
    <nav className="docbar" aria-label={text('Documentation site')}>
      {/* The site's own lockup and the site's own link: the monogram, the name beside it, and the
          front page behind both. The drawing is `docs/tensorspine.svg` as `nav.html` arranges it
          and is never redrawn; what is set here is the scale. The anchor is named by the text, so
          the name is not hidden — and the logotype is why WCAG 1.4.3 exempts it from contrast. */}
      <a className="wordmark" href={`${docsBase}${SITE_HOME}`}>
        <span className="mark" aria-hidden="true" dangerouslySetInnerHTML={{ __html: MONOGRAM }} />
        <span className="name">
          <span>Tensor</span>
          <span>Spine</span>
        </span>
      </a>
      <div className="doclinks">
        {SITE_NAV.map((group) => {
          const links = group.entries.map((entry) => (
            <a key={entry.href} href={`${docsBase}${entry.href}`}>
              {entry.label}
            </a>
          ));
          // A group the site shows is shown; a group it folds is folded, behind the site's own
          // `<summary>`. `name` makes the two exclusive where a browser implements it, which is
          // what `nav.html` asks for and what stops two panels from overlapping.
          return (
            <span
              className={group.label === null ? 'docgroup shown' : 'docgroup folded'}
              key={group.name}
              data-group={group.name}
            >
              {group.label === null ? (
                links
              ) : (
                <details className="docmenu" name="site-navigation">
                  <summary>{group.label}</summary>
                  <div className="docmenu-panel">{links}</div>
                </details>
              )}
              {group.name === EDITOR_GROUP ? (
                <span className="cur" aria-current="page">
                  {text(EDITOR_ENTRY)}
                </span>
              ) : null}
            </span>
          );
        })}
      </div>
    </nav>
  );
}
