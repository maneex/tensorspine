import { readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { expect, test, type Page } from '@playwright/test';

import { langUrl } from '../playwright.config.js';

// The `Lang` API over a **real** `Worker`, in a real browser — feature 1.11's browser layer.
//
// The unit layer already drives the same host and the same client over a `MessageChannel`, which
// is a true structured-clone boundary and refuses a symbol exactly as a thread does. What only a
// thread can answer is here: that a derivation does not block the page while it runs, that a
// cancel message reaches a worker already at work, and that the core is in the worker's chunk and
// not in the page's. The page under test imports the application's own `startLang`, so the worker
// Vite emitted for it is the worker the static build emits.

/** What a probe of the page answers. */
interface Outcome {
  readonly ok: boolean;
  readonly detail: Record<string, unknown>;
  readonly ms: number;
}

async function probe(page: Page, name: string, argument?: string): Promise<Outcome> {
  return page.evaluate(
    ([call, one]) => {
      const probes = window.langProbe as Record<string, (value?: string) => Promise<Outcome>>;
      return probes[call]?.(one) as Promise<Outcome>;
    },
    [name, argument] as const,
  );
}

declare global {
  interface Window {
    langProbe: unknown;
  }
}

test.beforeEach(async ({ page }) => {
  await page.goto(langUrl);
  await expect(page.locator('#out')).toHaveAttribute('data-state', 'ready');
});

test('a corpus document is opened, validated, expanded and derived in the worker', async ({
  page,
}) => {
  const outcome = await probe(page, 'pipeline', 'llama3-8b');
  expect(outcome.detail['error']).toBeUndefined();
  expect(outcome.ok).toBe(true);

  // The bytes come back as the file holds them (D12), across the boundary and back.
  expect(outcome.detail['roundTrips']).toBe(true);
  // §5.4's pipeline, stage by stage, reported as the worker crosses them.
  expect(outcome.detail['stages']).toEqual(['schema', 'library', 'assignment', 'semantic']);
  expect(outcome.detail['problems']).toBe(0);
  // The counters `--validate` prints for llama3-8b, key for key as the tools print them:
  // `instances=195 | edges=258 | tensors=291 | state_slots=32`.
  const stats = outcome.detail['stats'] as Record<string, string>;
  expect([stats['instances'], stats['edges'], stats['tensors'], stats['state_slots']]).toEqual([
    '195',
    '258',
    '291',
    '32',
  ]);
  // D1's nodes, and the six products of a derived document.
  expect(outcome.detail['nodes']).toBe(195);
  expect(outcome.detail['products']).toEqual([
    'schema',
    'model',
    'primitive_libraries',
    'assignment',
    'd1',
    'd2',
    'd3',
    'd4',
    'd5',
    'd6',
  ]);
  // `describe` for the selected instance (plan §3), which is what the sheet asks for.
  expect(outcome.detail['sites']).toEqual(['embed']);
});

test('a derivation of the largest corpus document does not block the page', async ({ page }) => {
  const outcome = await probe(page, 'nonBlocking', 'deepseek-v4-pro');
  expect(outcome.ok).toBe(true);
  const elapsed = outcome.detail['elapsed'] as number;
  const rate = outcome.detail['framesPerSecond'] as number;
  // The derivation is real work — the same call costs 36–262 ms in Node (`editor/spikes/
  // timings.md`) — and the page kept animating through it. A derivation on the page's own thread
  // would have stopped the loop dead for its whole length.
  expect(elapsed).toBeGreaterThan(20);
  expect(rate).toBeGreaterThan(30);
});

test('a superseded derivation is cancelled and its result dropped', async ({ page }) => {
  const outcome = await probe(page, 'superseded', 'deepseek-v4-pro');
  expect(outcome.ok).toBe(true);
  expect(outcome.detail['first']).toEqual({ cancelled: true, reason: 'superseded' });
  // One in-flight derivation per document, and it is the last one asked for: the newer answers.
  expect(outcome.detail['second']).toContain('d6');
});

test('a derivation the caller aborts is cancelled', async ({ page }) => {
  const outcome = await probe(page, 'aborted', 'deepseek-v4-pro');
  expect(outcome.ok).toBe(true);
  expect(outcome.detail).toEqual({ cancelled: true, reason: 'requested' });
});

test('a refusal comes back as a refusal, and the worker answers the next call', async ({ page }) => {
  const outcome = await probe(page, 'refusal');
  expect(outcome.ok).toBe(true);
  expect(outcome.detail['refused']).toBe(
    'LangFailure: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)',
  );
  expect(outcome.detail['after']).toBe('null\n');
});

test('the engine is in the worker’s chunk and not in the page’s', () => {
  // The claim the two entry points of `@tensorspine/lang/api` exist for. What the *page* may
  // carry is the proxy, the codec and the schema registry — §5.4 runs Ajv synchronously on the
  // interface's own thread, "< 5 ms on a corpus document", so the registry belongs there. What it
  // must never carry is the engine: the validator, the expansion and the derivation are the
  // worker's, and a page that bundled them would pay for them at startup and run them on the
  // thread that draws.
  const assets = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', 'spikes', 'lang', 'dist', 'assets');
  const names = readdirSync(assets);
  const page = names.find((name) => name.startsWith('index-') && name.endsWith('.js'));
  const worker = names.find((name) => name.startsWith('worker-') && name.endsWith('.js'));
  expect([page, worker].every((name) => name !== undefined)).toBe(true);

  const pageText = readFileSync(join(assets, page as string), 'utf8');
  const workerText = readFileSync(join(assets, worker as string), 'utf8');
  // Three sentences only the engine writes: `analyse`'s own name, and two refusals no proxy has.
  for (const engineOnly of ['not valid, no products', 'primitive absent from primitive library']) {
    expect(pageText.includes(engineOnly)).toBe(false);
    expect(workerText.includes(engineOnly)).toBe(true);
  }
  // And the page is the smaller of the two, which is the whole point.
  expect(statSync(join(assets, page as string)).size).toBeLessThan(
    statSync(join(assets, worker as string)).size,
  );
});
