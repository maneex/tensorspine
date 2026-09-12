import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import {
  alternationCount,
  drillGraph,
  drillPresence,
  foldedGraph,
  parse,
  toPython,
  type DrillGraph,
  type PyRecord,
  type PyValue,
} from '../../src/index.js';
import { oracleGenerated, oracleOut, readOracleManifest, repositoryRoot } from './oracle.js';

// The alternation strip of §4.8 and artboard S5, against the **oracle's** D1 — the expanded graph
// `python3 tools/tensorspine --d1` wrote, not the core's own.
//
// Feature 2.14's block names the numbers: "gemma3n-kvshare › decoder's alternation strip counts
// equal the ones computed from the oracle D1 (`attn` 24 of 30, `attn_full` 6 of 30, …)". The unit
// suite (`test/describe/drill.test.ts`) asserts the same counts over the core's expansion and
// cross-checks them against the guards; this one closes the loop at the other end, so that a
// divergence between the two expansions would show as a count rather than as a deep-equal
// somewhere in D1.
//
// It reads the oracle's file **as it stands**: `drillPresence` looks at `d1.nodes`' keys, which
// are the identifiers §5.2 rule 2 gives, and nothing below them — so the tagged encoding of
// `encoding.ts` is not needed here and a plain parse is the honest reading.

const generated = oracleGenerated();

/** The oracle's `--d1` document for one corpus model, by name. */
function oracleD1(name: string): PyValue {
  const found = readOracleManifest().documents.find((one) => one.name === name);
  if (found === undefined) throw new Error(`the oracle recorded no D1 for ${name}`);
  return JSON.parse(readFileSync(join(oracleOut, found.d1), 'utf8')) as PyValue;
}

/** The drill-in of one composition of one corpus document. */
function drillOf(model: string, composition: string): DrillGraph {
  const tree = parse(readFileSync(join(repositoryRoot, 'data', 'models', `${model}.json`), 'utf8'));
  const graph = drillGraph(tree, { composition });
  if (graph === null) throw new Error(`${model} declares no composition ${composition}`);
  return graph;
}

describe('a composition of several indices, over the oracle’s D1', () => {
  // D8 admits a grid — "multi-index compositions (a grid) — none in the corpus, admitted by the
  // grammar; the scrubber takes one control per index and the ghost columns are per overridden
  // index" — and feature 1.13's acceptance fixtures are where the repository has one (F8). So the
  // one reading that cannot be checked against a corpus document is checked against the tools'
  // own expansion of that fixture.
  it.skipIf(!generated)('reads `grid-composition`’s 2 × 3 grid as the tools expanded it', () => {
    const index = JSON.parse(readFileSync(join(oracleOut, 'fixtures', 'index.json'), 'utf8')) as {
      documents: { name: string; path: string; d1: string }[];
    };
    const one = index.documents.find((each) => each.name === 'grid-composition');
    expect(one).toBeDefined();
    const tree = parse(readFileSync(join(repositoryRoot, one?.path ?? ''), 'utf8'));
    const folded = foldedGraph(tree);
    const composition = folded.nodes.find((node) => node.children.length > 0);
    const drill = drillGraph(tree, { composition: composition?.name ?? '', folded }) as DrillGraph;
    const expanded = JSON.parse(readFileSync(join(oracleOut, one?.d1 ?? ''), 'utf8')) as PyValue;
    const presence = drillPresence(drill, expanded);

    // One control per index, over the values `indexGrid` resolved; the points are the product, in
    // the order §5.2 rule 2 sorts a site's indices.
    expect(drill.ranges.map((range) => `${range.name}:${String(range.values?.length)}`)).toEqual([
      'row:2',
      'col:3',
    ]);
    expect(presence.points).toHaveLength(6);
    expect(presence.points[0]?.label).toBe('col=0,row=0');
    expect(alternationCount(presence, 'cell')).toEqual({ at: 6, of: 6 });

    // A ghost column per **overridden index expression**: one rule overrides `col` alone, the
    // other `row` and `col` together, so there are two and not one.
    expect(drill.ghosts.map((ghost) => `${ghost.side}:${ghost.name}:${ghost.indices.length}`)).toEqual([
      'left:cell:1',
      'left:cell:2',
    ]);

    // The tools emitted four edges for the carry along the columns and one for the carry between
    // rows; the drill-in dims at exactly the other points.
    const fired = new Map(
      drill.edges.map((edge) => [edge.rule, presence.edges.get(edge.pointer)?.size ?? 0]),
    );
    expect(fired.get('cell.prev_col')).toBe(4);
    expect(fired.get('cell.prev_row')).toBe(1);
    expect(fired.get('grid.entry')).toBe(1);
    expect(fired.get('grid.exit')).toBe(1);
  });
});

describe('the alternation strip, over the oracle’s D1', () => {
  it.skipIf(!generated)('counts gemma3n-kvshare’s periodic pattern as the tools expanded it', () => {
    const drill = drillOf('gemma3n-kvshare', 'decoder');
    const presence = drillPresence(drill, oracleD1('gemma3n-kvshare'));
    expect(presence.points).toHaveLength(30);
    expect(alternationCount(presence, 'attn')).toEqual({ at: 24, of: 30 });
    expect(alternationCount(presence, 'attn_full')).toEqual({ at: 6, of: 30 });
    expect(alternationCount(presence, 'ffn_sparse')).toEqual({ at: 10, of: 30 });
    expect(alternationCount(presence, 'ffn')).toEqual({ at: 20, of: 30 });
    // S5's own note: "the 13 unguarded sites of this composition are present in all 30".
    const unguarded = drill.sites.filter((site) => site.guard === null);
    expect(unguarded).toHaveLength(13);
    for (const site of unguarded) {
      expect(alternationCount(presence, site.name)).toEqual({ at: 30, of: 30 });
    }
  });

  it.skipIf(!generated)('dims llama3-8b’s two carry edges at layer 0 and nowhere else', () => {
    const drill = drillOf('llama3-8b', 'decoder');
    const presence = drillPresence(drill, oracleD1('llama3-8b'));
    expect(presence.points).toHaveLength(32);
    for (const rule of ['attn_n.carry', 'attn_r.a_carry']) {
      const at = `/compositions/decoder/bindings/values/${rule}`;
      expect(presence.edges.get(at)?.size).toBe(31);
      expect(presence.edges.get(at)?.has('layer=0')).toBe(false);
      expect(presence.guards.get(at)?.get('layer=0')).toBe(false);
    }
    for (const site of drill.sites) {
      expect(alternationCount(presence, site.name)).toEqual({ at: 32, of: 32 });
    }
  });

  it.skipIf(!generated)('reads every composition of the corpus against the tools’ own expansion', () => {
    // The whole corpus, so that a composition nobody thought about — `colbert-v2`'s two, the
    // grids of `deepseek-v4-pro`, a template's — is read here too. What is asserted is the one
    // thing this layer can assert without the core's expansion: every site the drill-in draws is
    // present at as many points as the tools' D1 holds nodes for it.
    for (const one of readOracleManifest().documents) {
      if (!one.path.startsWith('data/models/')) continue;
      const tree = parse(readFileSync(join(repositoryRoot, one.path), 'utf8'));
      // A template is read under the assignment the oracle recorded — §4.6's own rule, and what
      // makes `decoder-causal-yarn`'s `layer ∈ [0, layers)` a grid at all.
      // The manifest writes the assignment as plain JSON, so it is read the way the editor reads
      // any document's numbers: through the core's own parser, where a token with no fraction is a
      // whole number (V3's lexical rule, finding F7).
      const options =
        one.assignment === null
          ? {}
          : { assignment: toPython(parse(JSON.stringify(one.assignment))) as PyRecord };
      const folded = foldedGraph(tree, options);
      const expansion = oracleD1(one.name);
      const nodes = Object.keys(
        ((expansion as Record<string, PyValue>)['d1'] as Record<string, PyValue>)[
          'nodes'
        ] as Record<string, PyValue>,
      );
      for (const node of folded.nodes) {
        if (node.kind !== 'composition') continue;
        const drill = drillGraph(tree, { composition: node.name, folded, ...options }) as DrillGraph;
        const presence = drillPresence(drill, expansion);
        for (const site of drill.sites) {
          const emitted = nodes.filter((name) =>
            name.startsWith(`${node.name}/${site.name}[`),
          ).length;
          expect({ site: site.name, at: presence.sites.get(site.name)?.size }).toEqual({
            site: site.name,
            at: emitted,
          });
        }
      }
    }
  }, 120_000);
});
