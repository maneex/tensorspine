import { describe, expect, it } from 'vitest';

import { filesUnder, readEditorFile } from './tree.js';

// A shipped source prints nothing to the console on its own account. The interface has a Log
// panel for what a reader should see and the core raises what a caller should catch, so a
// `console.log` in a source under `src/` is one of two things: a debugging probe that outlived
// its debugging, or a message that belongs in the Log. Feature 2.17 shipped the first kind —
// `console.log('PROBE attachSchemas', …)` in the source view's schema attachment, one line in
// every reader's console when a JSON source opened — and nothing refused it for twelve features.
// This does. Build scripts (`editor/scripts/`), the spikes and the tests are not sources.

const shipped = [...filesUnder('packages'), ...filesUnder('apps')].filter(
  (path) => /(^|\/)src\//.test(path) && /\.(ts|tsx)$/.test(path),
);

/** What a probe looks like: the two console calls that print, and the word a probe is named by. */
const PROBES = [/\bconsole\.log\(/, /\bconsole\.debug\(/, /\bPROBE\b/];

describe('no debugging output in the shipped sources', () => {
  it('reads the sources it judges', () => {
    expect(shipped.length).toBeGreaterThan(100);
    expect(shipped.some((path) => path === 'packages/ui/src/source/monaco.ts')).toBe(true);
  });

  it('finds no console.log, console.debug or PROBE under any src/', () => {
    const offenders: string[] = [];
    for (const path of shipped) {
      const lines = readEditorFile(path).split('\n');
      lines.forEach((line, index) => {
        if (PROBES.some((probe) => probe.test(line))) offenders.push(`${path}:${String(index + 1)}: ${line.trim()}`);
      });
    }
    expect(offenders).toEqual([]);
  });
});
