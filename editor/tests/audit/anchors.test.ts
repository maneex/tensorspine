import { readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

import { SPECIFICATION_ANCHORS } from '../../packages/ui/src/shell/anchors.js';
import { specificationLink, SPECIFICATION_PAGE } from '../../packages/ui/src/shell/store.js';
import { anchorsOf, moduleFor, MODULE, SPECIFICATION } from '../../scripts/anchors.js';
import { editorRoot, readEditorFile } from './tree.js';

// **The map of §1, audited** — "links to the specification are generated from the anchors
// `docs/SPECIFICATION.md` carries (a build step extracts `V1`–`V20`, `O`- and `I`-identifiers to
// their anchors)". The build step is `pnpm anchors`; this is what stops what it wrote from
// drifting from what it would write today, the arrangement `tests/audit/logo.test.ts` already uses
// for the wordmark.
//
// Three things are checked, and each is a different way the link could quietly stop working:
//
//   * the module **is** what the script writes from today's specification;
//   * every anchor it names is an anchor the specification actually carries (a section
//     re-anchored, a rule moved to a section with no `<a id>` of its own);
//   * the map covers what the panel will ask it for — V1 through V20, every rule of §6.

const repositoryRoot = resolve(editorRoot, '..');

/** The specification, read at the source (plan §0.1's "repository material"). */
function specification(): string {
  return readFileSync(join(repositoryRoot, SPECIFICATION), 'utf8');
}

describe('the specification’s anchors', () => {
  it('are what the vendoring script would write today', () => {
    expect(readEditorFile(MODULE)).toBe(moduleFor(anchorsOf(specification())));
  });

  it('carry every rule of §6 — V1 through V20, which is what a problem’s code names', () => {
    const rules = Array.from({ length: 20 }, (_, at) => `V${String(at + 1)}`);
    for (const rule of rules) {
      expect(SPECIFICATION_ANCHORS[rule], rule).toBeDefined();
      expect(SPECIFICATION_ANCHORS[rule]?.anchor, rule).toBe('6--static-semantics');
      expect(SPECIFICATION_ANCHORS[rule]?.section, rule).toContain('Static semantics');
    }
    // And nothing beyond V20 has been invented: the series is the specification's to grow.
    expect(SPECIFICATION_ANCHORS['V21']).toBeUndefined();
  });

  it('carry the O- and I-series the plan names, with the sections that state them', () => {
    // §1 asks for all three. The panel reads the V half; the primitive editor and the argument
    // sheet cite an `O` and an `I` (§4.12, §4.22), and the map is built once for all of them.
    expect(SPECIFICATION_ANCHORS['O9.2']?.anchor).toBe('appendix-a--requirements');
    expect(SPECIFICATION_ANCHORS['O1.3']?.section).toContain('Appendix A');
    expect(SPECIFICATION_ANCHORS['I7']?.anchor).toBe('91--invariants');
    expect(SPECIFICATION_ANCHORS['N4']?.anchor).toBe('92--non-requirements');
    // I10 is withdrawn — "belongs to deployment control and is outside this specification" — and
    // the extraction keeps only what is *stated* in a table, so it is not there.
    expect(SPECIFICATION_ANCHORS['I10']).toBeUndefined();
  });

  it('name places the specification actually carries', () => {
    const text = specification();
    const anchors = new Set([...text.matchAll(/<a id="([^"]+)"><\/a>/g)].map((match) => match[1]));
    for (const [id, one] of Object.entries(SPECIFICATION_ANCHORS)) {
      expect(anchors.has(one.anchor), `${id} -> ${one.anchor}`).toBe(true);
    }
  });

  it('refuses rather than guesses where the extraction has nothing to go on', () => {
    expect(() => anchorsOf('| **V1** | stated before any anchor |\n')).toThrow(/before any anchored section/);
    expect(() =>
      anchorsOf('<a id="a"></a><a id="b"></a>\n\n## Two\n\n| **V1** | x |\n'),
    ).toThrow(/which one to link to is a guess/);
    expect(() => anchorsOf('<a id="a"></a>\n\n## One\n\n| **V1** | x |\n| **V1** | y |\n')).toThrow(
      /stated twice/,
    );
    expect(() => anchorsOf('nothing at all')).toThrow(/no identifier is stated/);
  });
});

describe('the link a problem’s code carries', () => {
  it('is the specification’s page at the section that states the rule', () => {
    const link = specificationLink('V7', '../');
    expect(link?.href).toBe(`../${SPECIFICATION_PAGE}#6--static-semantics`);
    expect(link?.title).toBe('§6 — Static semantics');
  });

  it('is nothing for a code the specification does not state', () => {
    // Every lint finding and most of the loader's refusals carry `''`; `schema` and `registry` are
    // the core's own words for a stage, not rules of §6.
    expect(specificationLink('', '../')).toBeNull();
    expect(specificationLink('schema', '../')).toBeNull();
    expect(specificationLink('registry', '../')).toBeNull();
  });
});
