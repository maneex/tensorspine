import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

// The design's tokens, held to the design — plan §4.21 ("the token set the design pass chose,
// `plans/graph-editor-design/_ts.css`") and the component inventory's §1 table.
//
// Two claims, and they are different claims:
//
//   * the token file **is** the design's two blocks, byte for byte. `plans/` is a working
//     directory the repository does not carry, so this can only be asked where that directory is
//     on the disk; the condition is registered in `tests/audit/parity-job.test.ts` beside every
//     other fact about a machine that may stop a test running.
//   * the token file says what it says, as a snapshot. That runs everywhere, and it is what makes
//     a change to a colour a reviewed diff rather than a commit nobody looked at — which is the
//     claim that has to hold in CI, where the design is not.

const here = dirname(fileURLToPath(import.meta.url));
const tokensPath = resolve(here, '..', '..', 'src', 'shell', 'tokens.css');
const designPath = resolve(here, '..', '..', '..', '..', '..', 'plans', 'graph-editor-design', '_ts.css');

const tokens = readFileSync(tokensPath, 'utf8');
const designPresent = existsSync(designPath);

/** The text of one `:root`-style block, opening brace to closing brace, as it is written. */
function blockText(css: string, selector: string): string {
  const start = css.indexOf(`${selector} {`);
  if (start < 0) throw new Error(`no ${selector} block`);
  const end = css.indexOf('}', start);
  if (end < 0) throw new Error(`no end to the ${selector} block`);
  return css.slice(start, end + 1);
}

/** The declarations of one block, in the order they are written. */
function declarations(text: string): [string, string][] {
  return [...text.matchAll(/(--[a-z0-9-]+)\s*:\s*([^;]+);/g)].map((match) => [
    match[1] ?? '',
    (match[2] ?? '').trim(),
  ]);
}

const dark = declarations(blockText(tokens, ':root'));
const light = new Map(declarations(blockText(tokens, '.theme-light')));

describe('the token file', () => {
  it('is one `:root` block and one `.theme-light` block that redefines the same names', () => {
    // The inventory's §1, stated as a shape rather than as prose: the light theme redefines, it
    // does not introduce. A name only the light theme carried would be a colour the dark theme
    // has no value for.
    const names = new Set(dark.map(([name]) => name));
    expect([...light.keys()].filter((name) => !names.has(name))).toEqual([]);
    expect(dark.length).toBeGreaterThan(30);
  });

  it('is the palette the snapshot records, in both themes', () => {
    const table = dark.map(([name, value]) => {
      const other = light.get(name);
      return `${name.padEnd(16)} ${value.padEnd(12)} ${other ?? '(same)'}`;
    });
    expect(table.join('\n')).toMatchSnapshot();
  });

  it('carries the four colours the plan gives a meaning to', () => {
    // §4.21: "teal for live, selected and valid, petrol for structure …, amber for anything
    // derived rather than declared …, red for a refusal".
    for (const name of ['--accent', '--struct', '--derived', '--bad']) {
      expect(new Map(dark).get(name), name).toMatch(/^#[0-9a-f]{6}$/);
      expect(light.get(name), name).toMatch(/^#[0-9a-f]{6}$/);
      expect(light.get(name)).not.toBe(new Map(dark).get(name));
    }
  });

  it('names IBM Plex and no serif', () => {
    const sans = new Map(dark).get('--sans') ?? '';
    const mono = new Map(dark).get('--mono') ?? '';
    expect(sans).toContain('IBM Plex Sans');
    expect(mono).toContain('IBM Plex Mono');
    expect(`${sans} ${mono}`).not.toContain('serif,');
    expect(`${sans} ${mono}`).not.toContain('Newsreader');
  });
});

describe.skipIf(!designPresent)('the token file against the design it was vendored from', () => {
  it('is `_ts.css`’s two blocks, byte for byte', () => {
    const design = readFileSync(designPath, 'utf8');
    expect(blockText(tokens, ':root')).toBe(blockText(design, ':root'));
    expect(blockText(tokens, '.theme-light')).toBe(blockText(design, '.theme-light'));
  });

  it('carries nothing else of the design: the components are the shell’s own stylesheet', () => {
    const design = readFileSync(designPath, 'utf8');
    // The specimen chrome the inventory marks "design-pass only … has no place in the
    // application" must not have come across with the tokens.
    for (const selector of ['.sheet', '.spec', '.cap', '.sketch']) {
      expect(design).toContain(selector);
      expect(tokens, selector).not.toContain(`${selector} `);
    }
  });
});
