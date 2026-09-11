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
    expect(failures).toEqual([]);
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
