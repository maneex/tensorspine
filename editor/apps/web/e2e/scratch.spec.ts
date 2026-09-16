import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { expect, test, type Download } from '@playwright/test';

import { items, member, parse, toPython, type PyRecord, type PyValue } from '@tensorspine/lang';

import { counted, wlHash } from '../../../packages/lang/test/parity/signature.ts';
import { renameDerived, type Renaming } from './derived-names.js';
import { buildLlama3, command, settled, source, SAVED } from './scratch-build.js';

/**
 * **The phase-2 exit: `llama3-8b` built from scratch** — feature 2.19, plan §0's third
 * done-criterion and §6's phase-2 exit.
 *
 * > `llama3-8b` can be built from an empty document inside the editor, from the library palette,
 * > the property sheet and the canvas alone, and the result validates, derives to the same
 * > products as the corpus document (up to instance names), and locates every tensor […]
 *
 * `scratch-build.ts` is that sentence as a script of gestures — a menu entry, a drop on the
 * canvas, a field of a sheet, a connection between two port handles, an entry of a context menu, a
 * token of the location editor — and nothing else: it writes no JSON and calls nothing of the
 * store. This file is what the build is held to.
 *
 * **"Locates" here is the locations, not a checkpoint.** §0's criterion ends "and locates every
 * tensor from the local `Meta-Llama-3-8B` headers", which is phase 4's (features 4.1–4.5). What is
 * in this feature's reach is what the *document* says — the twelve identities, their dtype and
 * their physical names, written with the token editor of §4.14 — and D3's own `location` member is
 * what the comparison holds them to, tensor for tensor.
 *
 * **Three claims, one build.** The build is four hundred gestures and minutes of wall clock, so
 * the claims are made of one run rather than one each:
 *
 * 1. the document the editor wrote **is** the corpus's document, up to the names of its binding
 *    rules — a comparison of the two trees, which is what makes a failure readable;
 * 2. the document the editor **derives** is deep-equal to the one `tools/tensorspine --derive`
 *    wrote for the corpus document — the same 195 nodes with their resolved arguments and
 *    families, the same 258 edges, the same 291 tensors with their shapes, bytes and physical
 *    names, the same 32 state identities, the same arithmetic, the same 38 graph splits and 323
 *    partition options — read through the renaming of `derived-names.ts`;
 * 3. it **denotes the signature suite's graph**: the figures `tests/signatures/llama3-8b.json`
 *    records, taken off the exported D1 with the parity suite's own `wlHash`. That claim needs no
 *    normalisation at all — a signature is "computed from D1 and the validator's derivations and
 *    *never from identifiers*" — so it holds whatever this script chose to call anything, and it
 *    is the independent half of the pair: a renaming that was wrong could not make it pass.
 */

const repository = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', '..');
const oraclePath = join(repository, 'editor/tests/oracle/out/derive/llama3-8b.derived.json');
const oracleGenerated = existsSync(oraclePath);

/** The corpus document this build transcribes, and the graph the repository records for it. */
const corpus = JSON.parse(readFileSync(join(repository, 'data/models/llama3-8b.json'), 'utf8')) as Record<
  string,
  unknown
>;
const recorded = JSON.parse(
  readFileSync(join(repository, 'tests/signatures/llama3-8b.json'), 'utf8'),
) as { nodes: number; edges: number; per_primitive: Record<string, number>; wl: string };

/**
 * The names this script chose that the corpus's author did not — the whole of the renaming.
 *
 * Everything else is named as the corpus names it, because those names are the author's reading of
 * the checkpoint: the instances after the tensors they hold, the composition after what it is, the
 * quantities after the configuration they come from, the interfaces after what goes in and comes
 * out. What is left is the **binding rules**, and those the editor proposes on every connection
 * (§4.7: "the new binding is named `<to>.<port>`, uniquified") — a label nothing refers to
 * (feature 2.2), which is exactly where an author's document and the editor's differ with no
 * difference of meaning. Nine of the twelve differ; three already agree, and carry no entry.
 *
 * They are keyed by the identity §5.2 rule 7 gives each rule: a rule written inside `decoder`
 * carries the composition's name, one written at the top level carries its own.
 */
const RENAMING: Renaming = {
  rules: {
    'attn_n.input': 'decoder.entry',
    'attn_r.a': 'decoder.entry.a',
    'final_n.input': 'final_n.in',
    'lm_head.input': 'lm_head.in',
    'decoder.attn_n.input': 'decoder.attn_n.carry',
    'decoder.attn_r.a': 'decoder.attn_r.a_carry',
    'decoder.attn.input': 'decoder.attn.norm_in',
    'decoder.ffn_n.input': 'decoder.ffn_n.in',
    'decoder.ffn.input': 'decoder.ffn.norm_in',
  },
};

/** The same renaming over the *document*, where a rule is a member of its own map. */
const RULE_NAMES: Readonly<Record<string, string>> = {
  'attn_n.input': 'decoder.entry',
  'attn_r.a': 'decoder.entry.a',
  'final_n.input': 'final_n.in',
  'lm_head.input': 'lm_head.in',
  'attn.input': 'attn.norm_in',
  'ffn_n.input': 'ffn_n.in',
  'ffn.input': 'ffn.norm_in',
};

/** The two rules of the composition whose proposed name the corpus writes differently. */
const SCOPED_RULE_NAMES: Readonly<Record<string, string>> = {
  ...RULE_NAMES,
  'attn_n.input': 'attn_n.carry',
  'attn_r.a': 'attn_r.a_carry',
};

/** What a download holds, read back through its own stream. */
async function held(download: Download): Promise<string> {
  const stream = await download.createReadStream();
  const chunks: Buffer[] = [];
  for await (const chunk of stream) chunks.push(chunk as Buffer);
  return Buffer.concat(chunks).toString('utf8');
}

/** A binding map with its rules renamed, keeping the order they were written in. */
function renamedRules(
  map: Record<string, unknown>,
  names: Readonly<Record<string, string>>,
): Record<string, unknown> {
  return Object.fromEntries(Object.entries(map).map(([name, rule]) => [names[name] ?? name, rule]));
}

/**
 * The document the editor wrote, with its binding rules under the names the corpus gives them.
 *
 * Only the *keys* of the four binding maps move — a rule's name is a label and nothing refers to
 * it, which feature 2.2 established and which is why renaming one here cannot repair a difference
 * anywhere else in the tree.
 */
function underCorpusNames(built: Record<string, unknown>): Record<string, unknown> {
  const bindings = built['bindings'] as Record<string, Record<string, unknown>>;
  const compositions = built['compositions'] as Record<string, Record<string, unknown>>;
  const decoder = compositions['decoder'] as Record<string, unknown>;
  const scoped = decoder['bindings'] as Record<string, Record<string, unknown>>;
  return {
    ...built,
    bindings: { ...bindings, values: renamedRules(bindings['values'] ?? {}, RULE_NAMES) },
    compositions: {
      ...compositions,
      decoder: {
        ...decoder,
        bindings: { ...scoped, values: renamedRules(scoped['values'] ?? {}, SCOPED_RULE_NAMES) },
      },
    },
  };
}

/** The graph half of `tests/signature.py`, taken off a derived document's own D1. */
function signatureOf(derived: string): {
  nodes: number;
  edges: number;
  per_primitive: Record<string, number>;
  wl: string;
} {
  // Through the core's own parser, because the hash turns on Python's int/float distinction
  // (`repr(float)` inside `json.dumps`) and `JSON.parse` has none.
  const graph = member(toPython(parse(derived)) as PyRecord, 'd1') as PyValue;
  const nodes = items(member(graph as PyRecord, 'nodes') as PyValue);
  return {
    nodes: nodes.length,
    edges: (member(graph as PyRecord, 'edges') as readonly PyValue[]).length,
    per_primitive: counted(
      nodes.map(([, node]) => member(member(node as PyRecord, 'primitive') as PyRecord, 'name') as string),
    ),
    wl: wlHash(graph),
  };
}

// The build is four hundred gestures over one page, each waiting for the core to answer; the run
// is minutes rather than seconds. The explicit budget is the one every other long file of this
// layer carries (`drill.spec.ts`, `documents.spec.ts`, `deploy.spec.ts`, `headers.spec.ts`).
test.setTimeout(900_000);

// Seven boxes are on the canvas at once, laid out top to bottom, and every connection is a drag
// between two handles that must both be on the screen. 1600 × 1200 is the window feature 2.14
// measured for the drill-in, for the same reason.
test.use({ viewport: { width: 1600, height: 1200 } });

test.describe('the phase-2 exit', () => {
  test.skip(!oracleGenerated, 'the oracle has not been generated in this working copy');

  test('builds `llama3-8b` from an empty document, and it derives to the corpus’s own products', async ({
    page,
  }) => {
    await buildLlama3(page);

    // §5.4's two stages, on the whole document: the core refuses nothing and derives it.
    await settled(page);

    await test.step('the document is the corpus’s, up to the names of its binding rules', async () => {
      const built = JSON.parse(await source(page, SAVED)) as Record<string, unknown>;
      expect(underCorpusNames(built)).toEqual(corpus);
    });

    const [download] = await Promise.all([
      page.waitForEvent('download', { timeout: 120_000 }),
      command(page, 'File', 'file.export-derived'),
    ]);
    expect(download.suggestedFilename()).toBe('llama3-8b.derived.json');
    const derived = await held(download);

    await test.step('what it derives is what the tools derive from the corpus document', () => {
      const built = JSON.parse(derived) as unknown;
      expect(renameDerived(built, RENAMING)).toEqual(
        JSON.parse(readFileSync(oraclePath, 'utf8')) as unknown,
      );
    });

    await test.step('and it denotes the graph the signature suite records', () => {
      // No normalisation: a signature is taken over the graph and never over its identifiers.
      expect(signatureOf(derived)).toEqual({
        nodes: recorded.nodes,
        edges: recorded.edges,
        per_primitive: recorded.per_primitive,
        wl: recorded.wl,
      });
    });
  });
});
