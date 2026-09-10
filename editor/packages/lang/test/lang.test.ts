import { describe, expect, it } from 'vitest';

import { packageName } from '../src/index.js';

describe('@tensorspine/lang', () => {
  it('names itself as the workspace declares it', () => {
    expect(packageName).toBe('@tensorspine/lang');
  });

  it('runs where there is no DOM', () => {
    // The plan's §5.1: the core has no DOM, so that it runs unchanged in a worker, in Node and
    // in a test. A DOM global here would mean the unit project no longer runs in `node`.
    expect('document' in globalThis).toBe(false);
    expect('window' in globalThis).toBe(false);
  });
});
