import { describe, expect, it } from 'vitest';

import { filesUnder, readEditorFile } from './tree.js';

// The standing rule of the implementation plan's §0.2: no Python at runtime. Python runs in CI
// and in build scripts only — as the oracle of §0.5 and, later, as the generator of the
// vendored artifacts. The editor's packages and its application must carry none, and must
// start no interpreter.

const sources = [...filesUnder('packages'), ...filesUnder('apps')];

describe('no Python at runtime', () => {
  it('ships no Python file in the packages or the application', () => {
    expect(sources.filter((path) => path.endsWith('.py'))).toEqual([]);
  });

  it('keeps every Python file of the workspace under the oracle', () => {
    const python = filesUnder('tests').filter((path) => path.endsWith('.py'));
    expect(python.length).toBeGreaterThan(0);
    for (const path of python) expect(path.startsWith('tests/oracle/')).toBe(true);
  });

  // Two refusals of `tools/primitive_library.py` name the conversion command inside their own
  // text — "convert it with python3 tools/migrate.py INPUT -o OUTPUT" — and the core prints the
  // tools' words (D2, the parity contract). That is a *message*, not a call, so the audit removes
  // exactly that sentence before it looks; `spawn('python3', …)` and every other occurrence of the
  // word still trip it, which is what this test was written to catch.
  const TOOLS_ADVICE = 'python3 tools/migrate.py INPUT -o OUTPUT';

  it('states the tools’ own conversion advice, so the exemption cannot go stale', () => {
    const carriers = sources.filter(
      (path) => /(^|\/)src\//.test(path) && readEditorFile(path).includes(TOOLS_ADVICE),
    );
    expect(carriers).toEqual([
      'packages/lang/src/library/load.ts',
      'packages/lang/src/library/read.ts',
    ]);
  });

  it('starts no interpreter and no child process from the shipped sources', () => {
    const forbidden = [/\bchild_process\b/, /\bpython3?\b/, /\bexecSync\b/, /\bspawnSync\b/];
    const offenders: string[] = [];
    for (const path of sources) {
      if (!/(^|\/)src\//.test(path)) continue;
      if (!/\.(ts|tsx|js|jsx|mjs|cjs|html)$/.test(path)) continue;
      const text = readEditorFile(path).split(TOOLS_ADVICE).join('<the tools’ own advice>');
      if (forbidden.some((pattern) => pattern.test(text))) offenders.push(path);
    }
    expect(offenders).toEqual([]);
  });
});
