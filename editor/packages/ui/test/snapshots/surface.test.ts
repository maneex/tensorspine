import { describe, expect, it } from 'vitest';

import * as layout from '../../src/layout/elk.js';
import * as ui from '../../src/index.js';

// The snapshot layer of the plan's §0.3 holds the generated forms and the rendered strings.
// Until they exist, it holds the package's public surface, so that an export appearing or
// disappearing is a reviewed diff rather than a silent change.
describe('@tensorspine/ui', () => {
  it('exports the surface the snapshot records', () => {
    expect(Object.keys(ui).sort()).toMatchSnapshot();
  });

  // The layout is the package's second entry point, `@tensorspine/ui/layout`, kept out of the
  // index so that ELK reaches the bundle only where a canvas is laid out (§5.6).
  it('exports the layout surface the snapshot records', () => {
    expect(Object.keys(layout).sort()).toMatchSnapshot();
  });
});
