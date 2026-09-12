import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import type { PyValue } from '@tensorspine/lang';

import { formContext, type FormContext } from '../../src/forms/index.js';
import { presentation } from '../../src/presentation/index.js';
import { reading, registry } from '../canvas/source.js';
import { repositoryRoot } from '../presentation/source.js';

/**
 * What the Derived-panel suites read: the corpus, derived by the core.
 *
 * The feature's whole claim is that the panel shows what `derive` answered and what the schema and
 * the bindings say about it, so the suites feed it the core's real products over the repository's
 * own documents — never a fixture written for them. `../canvas/source.ts` already derives every
 * corpus document once and keeps the answer, so the panel and the canvas are held to one reading.
 */

export { registry };
export { repositoryRoot };

/** The form context the panel is generated in — one per registry, as the interface builds it. */
export const context: FormContext = formContext(registry, presentation());

/** One corpus document's derived document, as the core answers it. */
export function derivedOf(name: string): PyValue {
  return reading(name).derived;
}

/** The corpus, by the names the repository's own files carry. */
export const CORPUS: readonly string[] = [
  'colbert-v2',
  'deepseek-v4-pro',
  'gemma3n-kvshare',
  'llama3-8b',
  'llama4-scout',
  'qwen3.5-35b-a3b',
  'qwen3.5-397b',
  'qwen3.5-4b-text',
  'qwen3.8-27b',
  'qwen3.8-27b-text',
  'shieldstral-3b',
  'shieldstral-3b-composite',
  'voxtral-realtime',
  'whisper-large-v3',
];

/** The editor's own presentation schema, read as the audits read it. */
export function presentationSchema(): {
  $defs: { binding: { properties: Record<string, { enum?: string[] }> } };
} {
  return JSON.parse(
    readFileSync(
      join(repositoryRoot, 'editor', 'schemas', 'tensorspine-editor-presentation.schema.json'),
      'utf8',
    ),
  ) as { $defs: { binding: { properties: Record<string, { enum?: string[] }> } } };
}
