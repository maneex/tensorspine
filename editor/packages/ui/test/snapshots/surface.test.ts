import { describe, expect, it } from 'vitest';

import * as ui from '../../src/index.js';

// The snapshot layer of the plan's §0.3 holds the generated forms and the rendered strings.
// Until they exist, it holds the package's public surface, so that an export appearing or
// disappearing is a reviewed diff rather than a silent change.
describe('@tensorspine/ui', () => {
  it('exports the surface the snapshot records', () => {
    expect(Object.keys(ui).sort()).toMatchSnapshot();
  });
});
