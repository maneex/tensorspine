import { describe, expect, it } from 'vitest';

import * as api from '../../src/api/index.js';
import * as engine from '../../src/api/engine.js';
import { CALLS } from '../../src/api/index.js';
import { createLang } from '../../src/api/engine.js';

// `@tensorspine/lang/api` is the one seam every feature of phases 2, 3 and 4 reaches the language
// core through (plan §8, "everything from 2.6 on waits for 1.11"). A snapshot makes a change to
// that contract a reviewed diff rather than a surprise in a feature written six months later.

describe('@tensorspine/lang/api', () => {
  it('exports the surface the snapshot records', () => {
    expect(Object.keys(api).sort()).toMatchSnapshot();
  });

  // The side that *runs* the core is its own entry point, so that a page which talks to a worker
  // does not bundle the engine it is talking to (measured: a hundred kilobytes of Ajv).
  it('exports the engine’s surface the snapshot records', () => {
    expect(Object.keys(engine).sort()).toMatchSnapshot();
  });

  it('carries every call of the interface on the wire, and no other', () => {
    // The two must not drift: a call the interface has and the protocol does not is a call the
    // worker cannot serve, and a name on the wire the interface has not is dead.
    const lang = createLang();
    const own = Object.keys(lang)
      .filter((name) => name !== 'close')
      .sort();
    lang.close();
    expect(own).toEqual([...CALLS].sort());
  });
});
