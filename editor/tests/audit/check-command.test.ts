import { describe, expect, it } from 'vitest';

import vitestConfig from '../../vitest.config.js';
import { readEditorFile } from './tree.js';

// `pnpm check` is the suite of the implementation plan's §0.3: typecheck, lint, unit, parity,
// snapshot, audit, e2e. This audit holds the workspace to it — every layer has its script, the
// command runs them in that order, and no script names a Vitest project the configuration does
// not define (which would pass by running nothing).

interface PackageManifest {
  scripts: Record<string, string>;
}

const manifest = JSON.parse(readEditorFile('package.json')) as PackageManifest;

const layers = [
  'typecheck',
  'lint',
  'test:unit',
  'test:parity',
  'test:snapshot',
  'audit:rules',
  'test:e2e',
] as const;

function projectNames(config: unknown): string[] {
  const projects = (config as { test?: { projects?: unknown } }).test?.projects;
  if (!Array.isArray(projects)) return [];
  return (projects as unknown[]).flatMap((project) => {
    const name = (project as { test?: { name?: unknown } }).test?.name;
    return typeof name === 'string' ? [name] : [];
  });
}

describe('pnpm check', () => {
  it('declares a script for every layer', () => {
    for (const layer of layers) expect(Object.keys(manifest.scripts)).toContain(layer);
  });

  it('runs the layers in the order the plan states', () => {
    const check = manifest.scripts['check'] ?? '';
    const positions = layers.map((layer) => check.indexOf(`pnpm run ${layer}`));
    expect(layers.filter((_, index) => (positions[index] ?? -1) < 0)).toEqual([]);
    expect([...positions].sort((a, b) => a - b)).toEqual(positions);
  });

  it('vendors before the unit layer, because that layer reads what the vendor generates', () => {
    // The sheet's suites read the tools' generated argument schemas from the vendored directory
    // (F5: consumed as built) — an untracked directory `pnpm vendor` produces. A check that ran
    // the unit layer first was red on every fresh checkout and every CI run from feature 2.10 on,
    // and green on every machine that had vendored once: the ten failures of 12–17 Sep 2026.
    const check = manifest.scripts['check'] ?? '';
    const vendor = check.indexOf('pnpm run vendor');
    const unit = check.indexOf('pnpm run test:unit');
    expect(vendor).toBeGreaterThanOrEqual(0);
    expect(vendor).toBeLessThan(unit);
  });

  it('names only Vitest projects the configuration defines', () => {
    const declared = projectNames(vitestConfig);
    expect([...declared].sort()).toEqual(['audit', 'lang', 'parity', 'snapshot', 'store', 'ui']);

    const referenced = new Set<string>();
    for (const script of Object.values(manifest.scripts)) {
      for (const match of script.matchAll(/--project[ =](\S+)/g)) {
        const name = match[1];
        if (name !== undefined) referenced.add(name);
      }
    }
    expect(referenced.size).toBeGreaterThan(0);
    for (const name of referenced) expect(declared).toContain(name);
  });
});
