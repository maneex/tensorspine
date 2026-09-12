import { describe, expect, it } from 'vitest';

import * as canvas from '../../src/canvas/index.js';
import * as documents from '../../src/documents/index.js';
import * as expanded from '../../src/expanded/index.js';
import * as explorer from '../../src/explorer/index.js';
import * as layout from '../../src/layout/elk.js';
import * as ui from '../../src/index.js';
import * as shell from '../../src/shell/index.js';
import * as source from '../../src/source/index.js';

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

  // The expanded graph is the sixth, `@tensorspine/ui/expanded` — its own for the *opposite* of
  // the canvas's reason: §4.9's view must not reach ELK at all, since a whole expanded layout
  // sits at the two-second budget (feature 0.4), and `test/expanded/graph.test.ts` asserts it.
  it('exports the expanded surface the snapshot records', () => {
    expect(Object.keys(expanded).sort()).toMatchSnapshot();
  });

  // The JSON source view is `@tensorspine/ui/source`, its own for the reason the layout is, one
  // order of magnitude along: Monaco is about 3.9 MB of editor and JSON language in chunks
  // nothing but a source view pays for — and `test/source/bundle.test.ts` reads every source of
  // every package to say that nothing else reaches them.
  it('exports the source surface the snapshot records', () => {
    expect(Object.keys(source).sort()).toMatchSnapshot();
  });
});
