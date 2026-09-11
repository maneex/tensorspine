import { readdirSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { PROBLEM_SEVERITY, PROBLEM_SOURCE } from '@tensorspine/lang/api';
import { parse, type JsonObject } from '@tensorspine/lang';

import {
  droppedKeys,
  NOTICE,
  schemaMismatches,
  unusedQuantities,
} from '../../src/problems/notices.js';
import { fixesFor } from '../../src/problems/fixes.js';
import { corpus, index, outline, readRepositoryFile, repositoryRoot } from './source.js';

// §4.17's `editor` source, at `notice` — the only rows of the panel the core does not answer.
//
// Two claims run through the cases: a notice says something **true about the document** (it is
// read off the text and the editor's own events, never a rule of §6 re-derived here), and it is
// normally silent — a corpus document shows none, which is what makes one worth reading.

/** Every document of the corpus, the template included, as the walk finds them. */
function corpusNames(): string[] {
  const found: string[] = [];
  const walk = (at: string): void => {
    for (const entry of readdirSync(join(repositoryRoot, at), { withFileTypes: true })) {
      if (entry.isDirectory()) walk(`${at}/${entry.name}`);
      else if (entry.name.endsWith('.json')) found.push(`${at}/${entry.name}`);
    }
  };
  walk('data/models');
  return found.sort();
}

/** The notices a document carries, from the same two readings the store hands the panel. */
function noticesOf(tree: JsonObject, file: string): ReturnType<typeof unusedQuantities> {
  return unusedQuantities({ rows: outline(tree), index: index(tree), file });
}

describe('an unused quantity', () => {
  it('is silent on thirteen of the corpus’s fourteen documents, and on the template', () => {
    const noisy: string[] = [];
    for (const path of corpusNames()) {
      const tree = parse(readRepositoryFile(path)) as JsonObject;
      const found = noticesOf(tree, path);
      if (found.length > 0) noisy.push(`${path}: ${found.map((one) => one.path).join(', ')}`);
    }
    // **A finding about the corpus, pinned rather than hidden.** `deepseek-v4-pro` declares
    // `kv_heads` (a cardinality, literal 1) and nothing in the document reads it — measured, and
    // neither `--validate` nor `--lint` has a rule that says so. The language is frozen, so it is
    // written down here and in the ledger and not fixed.
    expect(noisy).toEqual(['data/models/deepseek-v4-pro.json: /quantities/kv_heads']);
  });

  it('is a notice of the editor, at the place the document declares it', () => {
    const tree = corpus('deepseek-v4-pro');
    const [found] = noticesOf(tree, 'data/models/deepseek-v4-pro.json');
    expect(found?.source).toBe(PROBLEM_SOURCE.editor);
    expect(found?.severity).toBe(PROBLEM_SEVERITY.notice);
    expect(found?.rule).toBe(NOTICE.unusedQuantity);
    expect(found?.code).toBe('');
    expect(found?.path).toBe('/quantities/kv_heads');
    expect(found?.message).toBe('quantity kv_heads is declared and read nowhere');
  });

  it('falls silent the moment something names it', () => {
    // The same document with one argument reading the quantity: the notice is the *reference*
    // question the rename and the delete cascade already ask, so an occurrence answers it.
    const text = readRepositoryFile('data/models/llama3-8b.json').replace(
      '"quantity": "heads"',
      '"quantity": "kv_heads"',
    );
    const before = noticesOf(parse(readRepositoryFile('data/models/llama3-8b.json')) as JsonObject, 'x');
    expect(before).toEqual([]);
    const after = noticesOf(parse(text) as JsonObject, 'x');
    // `heads` is now read one time fewer and `kv_heads` one time more; llama3-8b names `heads`
    // more than once, so neither goes unread.
    expect(after).toEqual([]);
  });

  it('carries the one fix this feature ships, and the fix removes exactly the declaration', () => {
    const tree = corpus('deepseek-v4-pro');
    const [found] = noticesOf(tree, 'data/models/deepseek-v4-pro.json');
    if (found === undefined) throw new Error('the notice is what this case is about');
    const rows = outline(tree);
    const fixes = fixesFor(found, { rows });
    expect(fixes).toHaveLength(1);
    expect(fixes[0]?.action.kind).toBe('remove-declaration');
    expect(fixes[0]?.action.title).toBe('Remove quantity kv_heads');
  });

  it('offers nothing for a row of the core’s, which is the honest answer (§4.17)', () => {
    const tree = corpus('llama3-8b');
    const rows = outline(tree);
    const problem = {
      code: 'V7',
      message: '[V7] something',
      path: '/instances/embed',
      severity: PROBLEM_SEVERITY.error,
      source: PROBLEM_SOURCE.semantic,
    } as const;
    expect(fixesFor(problem, { rows })).toEqual([]);
  });
});

describe('a dropped sidecar key', () => {
  it('is a notice naming the place that went (D6)', () => {
    const [found] = droppedKeys([{ where: 'positions', key: 'instances/embed' }], 'models/x.json');
    expect(found?.rule).toBe(NOTICE.droppedKey);
    expect(found?.severity).toBe(PROBLEM_SEVERITY.notice);
    expect(found?.path).toBe('instances/embed');
    expect(found?.file).toBe('models/x.json');
    expect(found?.message).toContain("positions 'instances/embed'");
  });
});

describe('a schema mismatch', () => {
  it('is a notice naming the file, in the words the log already used (§1)', () => {
    const [found] = schemaMismatches([
      { path: 'schemas/tensorspine.schema.json', message: 'schemas: the workspace carries its own' },
    ]);
    expect(found?.rule).toBe(NOTICE.schemaMismatch);
    expect(found?.file).toBe('schemas/tensorspine.schema.json');
    expect(found?.message).toBe('schemas: the workspace carries its own');
    expect(found?.path).toBe('');
  });
});
