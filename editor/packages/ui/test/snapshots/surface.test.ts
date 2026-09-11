import { describe, expect, it } from 'vitest';

import * as canvas from '../../src/canvas/index.js';
import * as documents from '../../src/documents/index.js';
import * as explorer from '../../src/explorer/index.js';
import * as layout from '../../src/layout/elk.js';
import * as ui from '../../src/index.js';
import * as shell from '../../src/shell/index.js';

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

  // The shell is the third, `@tensorspine/ui/shell`, kept out of the index for the reason the
  // layout is: the walker, the forms and the presentation bindings are readings of the schemas
  // that a worker, a script or a Node suite can use without React.
  it('exports the shell surface the snapshot records', () => {
    expect(Object.keys(shell).sort()).toMatchSnapshot();
  });

  it('exports the documents surface the snapshot records', () => {
    expect(Object.keys(documents).sort()).toMatchSnapshot();
  });

  it('exports the explorer surface the snapshot records', () => {
    expect(Object.keys(explorer).sort()).toMatchSnapshot();
  });

  // The canvas is the fifth, `@tensorspine/ui/canvas`. It is its own for the reason the layout is:
  // drawing a graph reaches ELK, and ELK is loaded on the first drawing (a dynamic import) rather
  // than in the shell's first bundle — 1.43 MB of its own chunk, measured (feature 0.4).
  it('exports the canvas surface the snapshot records', () => {
    expect(Object.keys(canvas).sort()).toMatchSnapshot();
  });
});
