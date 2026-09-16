import { readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

// §4.21's accessibility bullet, measured rather than hoped for:
//
// > **Accessibility**: … focus rings, contrast ≥ 4.5:1 for text …
//
// The axe pass of `apps/web/e2e/shell.spec.ts` checks what the page actually renders, which is
// the real test; this one checks the palette the page renders *from*, in both themes, without a
// browser — so that a rule written in the wrong ink is caught where the ink is chosen, and so
// that the two names the design's own palette cannot carry text in are written down with their
// measurements rather than rediscovered.

const here = dirname(fileURLToPath(import.meta.url));
const tokens = readFileSync(resolve(here, '..', '..', 'src', 'shell', 'tokens.css'), 'utf8');

/** The token values of one block of `tokens.css`. */
function blockOf(selector: string): Record<string, string> {
  const start = tokens.indexOf(`${selector} {`);
  if (start < 0) throw new Error(`tokens.css has no ${selector} block`);
  const end = tokens.indexOf('}', start);
  const values: Record<string, string> = {};
  for (const match of tokens.slice(start, end).matchAll(/(--[a-z0-9-]+)\s*:\s*([^;]+);/g)) {
    values[match[1] ?? ''] = (match[2] ?? '').trim();
  }
  return values;
}

const DARK = blockOf(':root');
const LIGHT = { ...DARK, ...blockOf('.theme-light') };
const THEMES = { dark: DARK, light: LIGHT } as const;

/** WCAG 2.1's relative luminance. */
function luminance(colour: string): number {
  const hex = colour.replace('#', '');
  const parts = [0, 2, 4].map((at) => parseInt(hex.slice(at, at + 2), 16) / 255);
  const linear = parts.map((value) => (value <= 0.03928 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4));
  return 0.2126 * (linear[0] ?? 0) + 0.7152 * (linear[1] ?? 0) + 0.0722 * (linear[2] ?? 0);
}

/** WCAG 2.1's contrast ratio, rounded as a report would print it. */
export function contrast(a: string, b: string): number {
  const [high, low] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return Math.round((((high ?? 0) + 0.05) / ((low ?? 0) + 0.05)) * 100) / 100;
}

/** The colour a token names, in one theme. */
function ink(theme: keyof typeof THEMES, token: string): string {
  const value = THEMES[theme][token];
  if (value === undefined) throw new Error(`no ${token} in the ${theme} theme`);
  return value;
}

/**
 * Every (text, ground) pair `shell.css` writes, named here beside it.
 *
 * A list and not a parse of the stylesheet, because what ground a rule sits on is a fact about
 * the markup and not about the rule; it is short on purpose, which is the same thing as saying
 * that the shell writes text in few places and in few inks.
 */
const PAIRS: readonly [string, string, string][] = [
  // the bar
  ['--ink-3', '--bg-chrome', 'a menu title, a pill, the palette button'],
  ['--muted', '--bg-chrome', 'the kind beside the wordmark'],
  ['--ink', '--bg-tint', 'a menu title with its menu open'],
  ['--ink-3', '--bg-tint', 'the avatar'],
  // the rail and the side bar
  ['--ink-3', '--bg-panel', 'the side bar’s heading and its notes'],
  ['--ink', '--bg-raised', 'what is typed into the filter'],
  ['--ink-3', '--bg-raised', 'the filter’s placeholder'],
  // the tabs and the canvas
  ['--muted', '--bg-chrome', 'a tab that is not current'],
  ['--ink', '--bg-panel', 'the current tab'],
  ['--ink-3', '--bg', 'the canvas note'],
  ['--ink-2', '--bg', 'the empty state’s heading'],
  ['--ink-3', '--bg', 'the empty state’s explanation'],
  ['--ink-2', '--bg-raised', 'a button'],
  // the panels
  ['--ink-3', '--bg-panel', 'a panel tab that is not showing'],
  ['--ink', '--bg-panel', 'the panel tab that is showing'],
  ['--ink-2', '--bg-panel', 'a panel’s own lines'],
  ['--accent', '--bg-panel', 'the tick beside “no problems”'],
  ['--bad', '--bg-panel', 'a refusal'],
  ['--derived', '--bg-panel', 'a derived figure'],
  ['--struct', '--bg-panel', 'a composition, a template, a split'],
  // the status bar
  ['--muted', '--bg-chrome', 'the status bar’s labels'],
  ['--ink-2', '--bg-chrome', 'the status bar’s values'],
  ['--ink', '--bg-chrome', 'a status bar figure'],
  // the palette and the menus
  ['--ink-2', '--bg-panel', 'a command in the palette or a menu'],
  ['--ink', '--bg-tint', 'the command the palette has highlighted'],
  ['--ink-3', '--bg-panel', 'a chord beside a command, and the palette’s foot'],
  ['--ink-3', '--bg-sunk', 'the palette’s foot on its own ground'],
  // banners
  ['--ink-2', '--bg-tint', 'an informing banner'],
  ['--ink-2', '--derived-bg', 'a warning banner'],
  ['--ink-2', '--bad-bg', 'a banner that stops'],
  // the open document (feature 2.6): the bar's two pills, the status bar's eight fields, the
  // toast, and the links the empty state offers a workspace through
  ['--ink-3', '--bg-chrome', 'a pill that is checking, or has nothing to report'],
  ['--bad', '--bg-chrome', 'a pill, or a status field, that counts problems'],
  ['--derived', '--bg-chrome', 'the derivation state and every derived figure in the bar'],
  ['--accent', '--bg-chrome', 'the validation state when there is nothing to report'],
  ['--ink-2', '--bg', 'a document shown as the JSON its file holds'],
  ['--ink-2', '--bg-raised', 'a toast'],
  ['--accent', '--bg-raised', 'the action beside a toast'],
  ['--accent', '--bg', 'a workspace offered as a link in the empty state'],
  ['--ink-3', '--bg-panel', 'the label of a field of a dialog'],
  ['--ink', '--bg-raised', 'what is typed into a dialog'],
  // the folded canvas (feature 2.9): a card and a terminal on `--bg-raised`, a composition box
  // on `--struct-bg`, a wire's label on the canvas ground, a chip on its own
  ['--ink', '--bg-raised', 'a card’s name'],
  ['--ink-2', '--bg-raised', 'the primitive a card pins'],
  ['--ink-3', '--bg-raised', 'its version, the structural summary, a port, a boundary handle'],
  ['--derived', '--bg-raised', 'a card’s guard badge and its derived line'],
  ['--bad', '--bg-raised', 'a slot nothing binds, and a refusal shown during a drag'],
  ['--warn', '--bg-raised', 'a port nothing consumes'],
  ['--derived', '--derived-bg', 'a state port’s chip'],
  ['--ink-3', '--bg-tint', 'a constant slot’s chip'],
  ['--ink-3', '--bg', 'a wire’s rule name and the type beside it'],
  ['--accent', '--bg', 'an identity link’s own name'],
  ['--struct', '--struct-bg', 'a composition’s name and its count'],
  ['--derived', '--struct-bg', 'its index range'],
  ['--ink-3', '--struct-bg', 'its summary, its families and its fold'],
  // the Derived panel (feature 2.15): the products rail, a product's header, a totals strip and
  // the tables of §4.18, all on the panel's own ground and on a selected row's tint.
  ['--ink-2', '--bg-panel', 'a product in the rail'],
  ['--ink', '--bg-tint', 'the product whose tab is showing'],
  ['--ink-3', '--bg-panel', 'the rail’s groups, its notes and the header’s envelope'],
  ['--ink', '--bg-panel', 'a product’s name'],
  ['--derived', '--bg-panel', 'a figure of a totals strip'],
  ['--ink-3', '--bg-panel', 'a totals strip’s labels and a table’s headers'],
  ['--ink-2', '--bg-panel', 'a cell of a product’s table'],
  // the expanded graph (feature 2.16): a card of the chain is the folded canvas's own ink on
  // `--bg-raised` (above), and a card inside a D6 split's block carries the composition's ground.
  // The board dims everything *outside* the block to 38 % opacity instead; that is a dimmed
  // identifier, which §4.21's floor does not admit, so the block is what is marked and the rows
  // beyond it are drawn as they always are (the stylesheet says so beside the rule).
  ['--ink', '--struct-bg', 'the identifier of a card inside a split’s block'],
  ['--ink-3', '--struct-bg', 'the primitive it pins'],
  ['--derived', '--struct-bg', 'its byte figure'],
  // the JSON source view (feature 2.17): the pane's head on the chrome's ground, and the text
  // Monaco paints on `--bg-sunk` from the theme `source/monaco.ts` defines over these very names.
  // The design's `_ts.css` writes a line number in `--faint` (3.04 dark, 2.16 light here) and a
  // truth in `--muted` (4.89 / 3.81); both step up the ramp feature 2.5 established, and the
  // string value steps down it in the light theme (below).
  ['--ink-2', '--bg-chrome', 'the file name in the pane’s head'],
  ['--ink-3', '--bg-chrome', 'the schema and the figures beside it'],
  ['--bad', '--bg-chrome', 'the count of problems in the head'],
  ['--ink-3', '--bg-sunk', 'a punctuation, a line number, a truth and a null'],
  ['--ink', '--bg-sunk', 'a member name'],
  ['--derived', '--bg-sunk', 'a number'],
  ['--bad', '--bg-sunk', 'the ink a marker is drawn in'],
  // …and the status bar's figures when they are stale (feature 2.6's `.status .fig.stale`, which
  // feature 2.17's axe pass is the first to reach with a figure in it): the number takes the
  // label's own ink rather than half its opacity, because no ink of the ramp survives an opacity
  // on this ground — `--muted` is 4.99:1 dark and 4.57:1 light at full strength.
  ['--muted', '--bg-chrome', 'the number of a status-bar figure that is stale'],
];

/**
 * The pairs one theme writes and the other does not — feature 2.11's own, and the only ones.
 *
 * `--accent` on `--bg-tint` reaches 6.76:1 in the dark theme and 4.21:1 in the light one, so the
 * expression tree's operator chip (§4.13, artboard S7) is written in `--accent-dim` there: the
 * darker end of the same ramp, which is what the light theme's own tokens do for every other ink.
 * Stated here, beside the tokens, rather than left to the axe pass that found it on the page.
 */
const THEMED: readonly [keyof typeof THEMES, string, string, string][] = [
  ['dark', '--accent', '--bg-tint', 'the operator chip of an expression tree'],
  ['light', '--accent-dim', '--bg-tint', 'the same chip, stepped down the ramp for a light ground'],
  ['dark', '--struct', '--bg-tint', 'a quantity chip'],
  ['light', '--struct', '--bg-tint', 'a quantity chip'],
  ['dark', '--derived', '--bg-tint', 'a literal and an index chip'],
  ['light', '--derived', '--bg-tint', 'a literal and an index chip'],
  // Feature 2.15's own, and the same pair one panel along: the short name beside a product and
  // the links inside its tables are `--accent`, which a selected row's tint refuses in the light
  // theme. Found by the axe pass over the rendered panel, as 2.11's was.
  ['dark', '--accent', '--bg-tint', 'a product’s short name, and a link in its table'],
  ['light', '--accent-dim', '--bg-tint', 'the same, stepped down the ramp for a light ground'],
  ['dark', '--accent', '--bg-panel', 'the Export link of the rail'],
  ['light', '--accent-dim', '--bg-panel', 'the same, stepped down'],
  // Feature 2.17: a text value and the caret in the JSON source. `--accent` is 8.78:1 on
  // `--bg-sunk` in the dark theme and 4.30:1 in the light one, which is the pair 2.11 and 2.15
  // each met one component along; `--accent-dim` is 6.23:1 there.
  ['dark', '--accent', '--bg-sunk', 'a text value, and the caret'],
  ['light', '--accent-dim', '--bg-sunk', 'the same, stepped down the ramp for a light ground'],
];

describe('every ink the shell writes text in', () => {
  it('reaches §4.21’s 4.5:1 on the ground it is written on, in both themes', () => {
    const failures: string[] = [];
    for (const theme of ['dark', 'light'] as const) {
      for (const [token, ground, where] of PAIRS) {
        const ratio = contrast(ink(theme, token), ink(theme, ground));
        if (ratio < 4.5) failures.push(`${theme}: ${token} on ${ground} is ${String(ratio)}:1 — ${where}`);
      }
    }
    for (const [theme, token, ground, where] of THEMED) {
      const ratio = contrast(ink(theme, token), ink(theme, ground));
      if (ratio < 4.5) failures.push(`${theme}: ${token} on ${ground} is ${String(ratio)}:1 — ${where}`);
    }
    expect(failures).toEqual([]);
  });

  it('reaches 4.5:1 for the one pair the design writes as literals, in both themes', () => {
    // `_ts.css`'s `.slot.priv` and `.slot.tied` are `#6fb8c8` on `#22303a`, which the canvas
    // writes as it writes them: a bound chip carries its own ground, so the two are one pair and
    // the theme does not change either of them.
    expect(contrast('#6fb8c8', '#22303a')).toBeGreaterThanOrEqual(4.5);
  });

  it('reaches 4.5:1 for the one ink the shell changed, which the design’s does not', () => {
    // `.btn.pri` is `#06181a` on `--accent-dim` in `_ts.css`; the shell writes `#eafcfb`, which
    // is the design's own light ink (`.tog.on::after`).
    for (const theme of ['dark', 'light'] as const) {
      expect(contrast('#06181a', ink(theme, '--accent-dim')), `${theme}: the design's ink`).toBeLessThan(4.5);
      expect(contrast('#eafcfb', ink(theme, '--accent-dim')), `${theme}: the shell's`).toBeGreaterThanOrEqual(4.5);
    }
  });
});

describe('the documentation site’s bar above the editor (feature 2.18, S18)', () => {
  /** The `--site-*` values `site-bar.css` declares — the site's own palette, by its own names. */
  const site: Record<string, string> = {};
  for (const match of readFileSync(join(here, '..', '..', 'src', 'shell', 'site-bar.css'), 'utf8').matchAll(
    /(--site-[a-z0-9-]+)\s*:\s*([^;]+);/g,
  )) {
    site[match[1] ?? ''] = (match[2] ?? '').trim();
  }

  it('reaches §4.21’s 4.5:1 on the site’s own ground, which has no theme', () => {
    // The bar is the site's and the site has one palette, so there is one measurement and not
    // two. `tests/audit/site-bar.test.ts` is what holds these values to `docs/style/`.
    const ground = site['--site-bg'] ?? '';
    const failures: string[] = [];
    for (const [token, on, where] of [
      ['--site-ink-3', '--site-bg', 'an entry, a folded group’s summary and its marker'],
      ['--site-ink', '--site-bg', 'the page being read, and an entry under the pointer'],
      ['--site-accent', '--site-bg', 'the underline that marks it, and the focus ring'],
      ['--site-ink-3', '--site-bg', 'an entry inside an open group’s panel, which is that ground'],
      ['--site-ink', '--site-bg-raised', 'that entry under the pointer, which lifts its ground'],
    ] as const) {
      const ratio = contrast(site[token] ?? '', site[on] ?? '');
      if (ratio < 4.5) failures.push(`${token} on ${on} is ${String(ratio)}:1 — ${where}`);
    }
    expect(ground).not.toBe('');
    expect(failures).toEqual([]);
  });

  it('is a logotype where it does not, which is the one exemption (WCAG 1.4.3)', () => {
    // The wordmark's teal half is 1.99:1 on the site's ground, there and on the site itself. It
    // is brand text, which has no contrast requirement, and repainting it would be redrawing the
    // logo — which §4.21 forbids in the same breath as the contrast. The browser suite excludes
    // `.wordmark` from the colour-contrast rule alone, exactly as feature 2.5 did for the
    // application's own bar.
    expect(contrast(site['--site-logo-teal'] ?? '', site['--site-bg'] ?? '')).toBeLessThan(4.5);
    expect(contrast(site['--site-logo-blue'] ?? '', site['--site-bg'] ?? '')).toBeGreaterThanOrEqual(4.5);
  });
});

describe('the two names the design’s palette cannot carry text in', () => {
  it('is `--faint`, on every ground, in both themes', () => {
    const grounds = ['--bg', '--bg-chrome', '--bg-panel', '--bg-raised', '--bg-tint', '--bg-sunk'];
    for (const theme of ['dark', 'light'] as const) {
      for (const ground of grounds) {
        expect(contrast(ink(theme, '--faint'), ink(theme, ground)), `${theme} ${ground}`).toBeLessThan(4.5);
      }
    }
  });

  it('is `--muted` on `--bg-panel` in the light theme and on `--bg-raised` in the dark one', () => {
    expect(contrast(LIGHT['--muted'] ?? '', LIGHT['--bg-panel'] ?? '')).toBe(4.37);
    expect(contrast(DARK['--muted'] ?? '', DARK['--bg-raised'] ?? '')).toBe(4.21);
    // And it does reach it on the chrome, which is the one ground the shell writes it on.
    expect(contrast(LIGHT['--muted'] ?? '', LIGHT['--bg-chrome'] ?? '')).toBeGreaterThanOrEqual(4.5);
    expect(contrast(DARK['--muted'] ?? '', DARK['--bg-chrome'] ?? '')).toBeGreaterThanOrEqual(4.5);
  });

  it('leaves `--faint` and `--muted` out of every text rule but the chrome’s', () => {
    const css = readFileSync(join(here, '..', '..', 'src', 'shell', 'shell.css'), 'utf8');
    // Every `color:` declaration naming one of the two, with the selector it belongs to.
    const offences: string[] = [];
    for (const block of css.split('}')) {
      const selector = (block.split('{')[0] ?? '').trim().split('\n').pop() ?? '';
      const body = block.split('{')[1] ?? '';
      for (const match of body.matchAll(/(?:^|[^-])color:\s*var\((--faint|--muted)\)/g)) {
        const token = match[1] ?? '';
        const chrome = /\.bar|\.sitekind|\.status|\.tab\b|\.tbtn|\.act\b|\.menu\b/.test(selector);
        if (token === '--muted' && chrome) continue;
        offences.push(`${selector}: ${token}`);
      }
    }
    expect(offences).toEqual([]);
  });
});
