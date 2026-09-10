import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { expect, test } from '@playwright/test';

import { CASES, type EngineReport } from '../../../spikes/headers/report.ts';
import { spikeUrl } from '../playwright.config';

/**
 * Feature 0.5 — safetensors headers in the browser.
 *
 * The spike's page (`editor/spikes/headers/page/`) is served by the second web server of
 * `playwright.config.ts`; these are the cases the plan asks to be held to account on every
 * commit: a synthetic `.safetensors` built in the test and parsed from a `Blob` slice, the same
 * reading on a file large enough that reading the payload would show, the same reading on a file
 * the browser has on disk, and the dtype table held to `tools/artifact.py`'s.
 *
 * Two cases are conditional, and say so when they skip: a real checkpoint shard
 * (`TENSORSPINE_CHECKPOINT`) and the Hub, which is tagged `@network` and left to a run by hand
 * — `editor/spikes/headers/NOTE.md` records what it answered, in three engines.
 */

const repository = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', '..');

/** The window the spike's page installs its entry points on. */
type SpikeWindow = Window & {
  spike: {
    readonly dtypes: Record<string, string>;
    readonly run: (names: readonly string[], engine?: string) => Promise<EngineReport>;
  };
};

/** Load the page and run the named cases in it. */
async function runCases(page: import('@playwright/test').Page, names: readonly string[]): Promise<EngineReport> {
  await page.goto(spikeUrl);
  await expect(page.locator('body')).toHaveAttribute('data-state', 'ready');
  return page.evaluate(
    (asked) => (window as unknown as SpikeWindow).spike.run(asked, 'playwright'),
    names as string[],
  );
}

/** The one case of a report, with its failure as the message when it did not pass. */
function only(report: EngineReport, name: string): EngineReport['cases'][number] {
  const found = report.cases.find((one) => one.name === name);
  expect(found, `the page ran no case named ${name}`).toBeDefined();
  return found as EngineReport['cases'][number];
}

test('a synthetic safetensors header is read from a Blob slice, and nothing else is read', async ({ page }) => {
  const report = await runCases(page, [CASES.syntheticBlob]);
  const result = only(report, CASES.syntheticBlob);
  expect(result.error ?? 'none', 'the case refused').toBe('none');
  expect(result.ok).toBe(true);

  const detail = result.detail ?? {};
  // Twelve dtypes `artifact.py` maps and one it does not, plus the file's `__metadata__`.
  expect(detail['tensors']).toBe(13);
  expect(detail['dtypes']).toEqual([
    'bf16',
    'bool',
    'f16',
    'f32',
    'f64',
    'f8e4m3',
    'f8e5m2',
    'i16',
    'i32',
    'i64',
    'i8',
    'u8',
    'u16',
  ]);
  // The prefix and the header, and not one byte of the payload.
  expect(result.bytesRead).toBe(8 + (detail['headerBytes'] as number));
  expect(result.bytesRead).toBeLessThan(detail['fileBytes'] as number);
  // A length that is zero, that runs past the end, or that is not even there: refused, not read.
  expect(detail['refusals']).toHaveLength(3);
});

test('the header of a 256 MiB file costs the header, not the file', async ({ page }) => {
  const report = await runCases(page, [CASES.syntheticLarge]);
  const result = only(report, CASES.syntheticLarge);
  expect(result.error ?? 'none', 'the case refused').toBe('none');
  expect(result.ok).toBe(true);
  expect(result.detail?.['fileBytes']).toBeGreaterThan(256 * 1024 * 1024);
  expect(result.bytesRead).toBe(8 + (result.detail?.['headerBytes'] as number));
  expect(result.bytesRead).toBeLessThan(1024);
});

test('a file the browser holds on disk is read the same way', async ({ page }) => {
  const report = await runCases(page, [CASES.opfsFile]);
  const result = only(report, CASES.opfsFile);
  // The Origin Private File System is how a page makes a file on disk without a picker; an
  // engine that has none says so rather than failing (WebKitGTK has none — see the note).
  test.skip(result.skipped !== undefined, result.skipped ?? '');
  expect(result.error ?? 'none', 'the case refused').toBe('none');
  expect(result.ok).toBe(true);
  expect(result.detail?.['fileBytes']).toBeGreaterThan(64 * 1024 * 1024);
  expect(result.bytesRead).toBe(8 + (result.detail?.['headerBytes'] as number));
});

test("the spike's dtype table is the one tools/artifact.py carries", async ({ page }) => {
  // The safetensors dtype vocabulary is the file format's, not the schemas' (plan §1, "what the
  // rule does not cover"): the table maps it onto the schema's `dtype` enum, and feature 1.9
  // will carry it in the core under the audit every semantic table gets. Until then, this holds
  // the spike's copy to the tools' own dictionary, character for character.
  const source = readFileSync(join(repository, 'tools', 'artifact.py'), 'utf8');
  const literal = /^DTYPES = (\{[^}]*\})/m.exec(source);
  expect(literal, 'tools/artifact.py no longer opens with a DTYPES dictionary').not.toBeNull();
  const tools = JSON.parse((literal?.[1] ?? '{}').replaceAll("'", '"')) as Record<string, string>;

  await page.goto(spikeUrl);
  const table = await page.evaluate(() => (window as unknown as SpikeWindow).spike.dtypes);
  expect(table).toEqual(tools);

  const schema = JSON.parse(
    readFileSync(join(repository, 'schemas', 'tensorspine.schema.json'), 'utf8'),
  ) as { $defs: { dtype: { enum: string[] } } };
  for (const name of Object.values(table)) expect(schema.$defs.dtype.enum).toContain(name);
});

test('a real checkpoint shard costs its header', async ({ page }) => {
  const shard = checkpointShard();
  test.skip(
    shard === null,
    'set TENSORSPINE_CHECKPOINT to a .safetensors file or a checkpoint directory to measure a real one',
  );
  await page.goto(spikeUrl);
  await page.locator('#local').setInputFiles(shard ?? '');
  const report = await page.evaluate(
    (asked) => (window as unknown as SpikeWindow).spike.run(asked, 'playwright'),
    [CASES.localFile],
  );
  const result = only(report, CASES.localFile);
  expect(result.error ?? 'none', 'the case refused').toBe('none');
  expect(result.ok).toBe(true);
  expect(result.detail?.['fileBytes']).toBe(statSync(shard ?? '').size);
  expect(result.bytesRead).toBe(8 + (result.detail?.['headerBytes'] as number));
  expect(result.bytesRead).toBeLessThan(1024 * 1024);
  // The measurement is the point of the case: the note quotes it, so the run prints it.
  const detail = result.detail ?? {};
  console.log(
    `local-file: ${String(detail['file'])} — ${String(detail['fileBytes'])} bytes on disk, ` +
      `header ${String(detail['headerBytes'])} bytes, ${String(detail['tensors'])} tensors, ` +
      `${String(result.bytesRead)} bytes read in ${String(result.ms)} ms; dtypes ${JSON.stringify(detail['dtypes'])}`,
  );
});

test(
  'the Hub answers a safetensors header over range requests',
  { tag: '@network' },
  async ({ page }) => {
    test.skip(!(await hubReachable()), 'the Hub is not reachable (or TENSORSPINE_HUB=0, or this is CI)');
    test.setTimeout(120_000);
    const report = await runCases(page, [CASES.hubRange, CASES.hubSingle, CASES.hubSharded]);
    for (const name of [CASES.hubRange, CASES.hubSingle, CASES.hubSharded]) {
      const result = only(report, name);
      expect(result.error ?? 'none', `${name} refused`).toBe('none');
      expect(result.ok).toBe(true);
      // A few hundred kilobytes at most, for repositories of 548 MB and 16 GB.
      expect(result.bytesRead).toBeLessThan(1024 * 1024);
      expect(result.detail?.['tensors']).toBeGreaterThan(0);
    }
    // The plain-range transport reads the header itself: the prefix, then the header's bytes.
    const ranges = (only(report, CASES.hubRange).detail?.['requests'] as { range: string | null }[]).map(
      (request) => request.range,
    );
    expect(ranges).toContain('bytes=0-7');
  },
);

/** A `.safetensors` file to measure, from `TENSORSPINE_CHECKPOINT`, or nothing. */
function checkpointShard(): string | null {
  const named = process.env['TENSORSPINE_CHECKPOINT'];
  if (named === undefined || named === '' || !existsSync(named)) return null;
  if (statSync(named).isFile()) return named;
  const shards = readdirSync(named)
    .filter((entry) => entry.endsWith('.safetensors'))
    .sort();
  const first = shards[0];
  return first === undefined ? null : join(named, first);
}

/**
 * Whether the Hub case runs: never in CI — a suite that reaches a third party on every commit is
 * a suite that fails for reasons of its own — and otherwise when huggingface.co answers.
 * `TENSORSPINE_HUB=1` forces it on, `0` off.
 */
async function hubReachable(): Promise<boolean> {
  const asked = process.env['TENSORSPINE_HUB'];
  if (asked === '0') return false;
  if (asked !== '1' && process.env['CI'] !== undefined && process.env['CI'] !== '') return false;
  try {
    const response = await fetch('https://huggingface.co/api/models/openai-community/gpt2', {
      method: 'HEAD',
      signal: AbortSignal.timeout(5_000),
    });
    return response.ok;
  } catch {
    return false;
  }
}
