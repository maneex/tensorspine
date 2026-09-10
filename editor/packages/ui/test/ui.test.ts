import { describe, expect, it } from 'vitest';

import { packageName } from '../src/index.js';

describe('@tensorspine/ui', () => {
  it('names itself as the workspace declares it', () => {
    expect(packageName).toBe('@tensorspine/ui');
  });
});
