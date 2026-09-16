/**
 * The editor's own place in the documentation site's bar — feature 2.18, S18.
 *
 * `docs/style/nav.html` is the site's one navigation source and it declares no entry for the
 * editor: the site does not link to it yet, and the editor may not edit `docs/` (the plan's
 * standing rule). So the bar supplies the entry for the page it is on — a mark and not a link,
 * there being nowhere for it to go — and `tests/audit/site-bar.test.ts` asserts that the site
 * still declares none, so that the day a documentation change set adds one this fails rather than
 * showing the entry twice.
 *
 * Beside `SiteBar.tsx` rather than in it because an audit reads them, and the audit layer is
 * compiled without JSX.
 */

/** What the bar calls the page it is on. */
export const EDITOR_ENTRY = 'Editor';

/**
 * Which group of the site's navigation the editor's entry joins.
 *
 * The site's top-level destinations — Overview, Models, Architecture, Glossary — which is what
 * the editor is: `nav.html`'s own `nav-main`, named by the class without its prefix.
 */
export const EDITOR_GROUP = 'main';
