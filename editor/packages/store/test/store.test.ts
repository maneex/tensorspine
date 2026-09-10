import { describe, expect, it } from 'vitest';

import { packageName } from '../src/index.js';

describe('@tensorspine/store', () => {
  it('names itself as the workspace declares it', () => {
    expect(packageName).toBe('@tensorspine/store');
  });
});
