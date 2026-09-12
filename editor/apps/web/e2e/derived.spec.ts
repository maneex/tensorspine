import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Locator, type Page } from '@playwright/test';

/**
 * The Derived panel — feature 2.15, plan §4.18, artboard S11.
 *
 * The five claims the feature's own block names, each of which needs a real page, a real worker
 * and a real derivation to be worth anything:
 *
 *   - **`llama3-8b`'s D3 totals in the status bar equal the oracle's** — the four figures §4.2
 *     puts there are the derived document's own, and the document the oracle wrote with
 *     `tools/tensorspine --derive` is what they are held to;
 *   - **the D4 table has 32 rows** — one state identity per layer, rendered from the schema;
 *   - **an edit dims the figures with `stale` until the next derivation** — §5.4's freshness, in
 *     the bar and in the panel, with the name of the command that did it;
 *   - **Export Derived Document… deep-equals the core's `derive`** — the file the command writes,
 *     read back and compared with the oracle's own products;
 *   - **the edge label of `embed.output` reads as the inventory's convention** — the value type
 *     `bf16[tokens, model.width=4096]`, which is `--view`'s own string (finding F6).
 *
 * Everything that is a question about *how a product is arranged* rather than about a browser is
 * asked in `packages/ui/test/derived/`, over the whole corpus.
 */

const repository = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', '..');
const MODEL = 'models/llama3-8b.json';

/** The oracle's own derived document for `llama3-8b`, where `pnpm oracle` has been run. */
const oraclePath = join(repository, 'editor/tests/oracle/out/derive/llama3-8b.derived.json');
const oracleGenerated = existsSync(oraclePath);

interface OracleDerived {
  d3: { totals: { bytes: number; elements: number; tensors: number } };
  d4: { states: unknown[]; totals: { append_bytes_per_cached_position: number } };
  d5: { operations: { element: { value: number } } };
  d2: { peak_live: { bytes_per_element: number } };
}

function oracle(): OracleDerived {
  return JSON.parse(readFileSync(oraclePath, 'utf8')) as OracleDerived;
}

async function open(page: Page): Promise<void> {
  await page.goto('/');
  await expect(page.locator('#root')).toHaveAttribute('data-state', 'ready');
  await expect(page.locator('#root')).toHaveAttribute('data-documents', 'ready');
}

/** Run a command of §4.4 by its menu entry, which is the way a person reaches it. */
async function command(page: Page, menu: string, id: string): Promise<void> {
  await page.locator(`nav.menu > div > button:text-is("${menu}")`).click();
  await page.locator(`.menu-pop button[data-command="${id}"]`).click();
}

/** Open the Examples workspace and one document of it, and wait for its products. */
async function openModel(page: Page, path = MODEL): Promise<void> {
  await page.locator('.nothing .link[data-open="examples"]').click();
  await page.locator('.dlg [data-workspace="examples"]').click();
  await expect(page.locator('footer.status [data-workspace]')).toHaveText('Examples');
  await command(page, 'File', 'file.open-model');
  await page.locator(`.dlg [data-document="${path}"]`).click();
  await expect(page.locator(`.gcanvas[data-canvas="${path}"]`)).toBeVisible();
  await derived(page);
}

/** Wait until the core has derived the document on screen. */
async function derived(page: Page): Promise<void> {
  await expect(page.locator('.bar [data-pill="derivation"]')).toHaveText('derived · fresh', {
    timeout: 60_000,
  });
}

/** Show the Derived panel, whichever region it sits in. */
async function showPanel(page: Page): Promise<Locator> {
  await command(page, 'View', 'view.derived');
  const panel = page.locator('.derived');
  await expect(panel).toBeVisible();
  return panel;
}

/** Open one product's tab, by the member the derived document writes it under. */
async function showProduct(page: Page, member: string): Promise<void> {
  await page.locator(`.der-rail [data-product="${member}"]`).click();
  await expect(page.locator(`.der-rail [data-product="${member}"]`)).toHaveClass(/sel/);
}

/** One status-bar figure, by the label `presentation.json` gives it. */
function figure(page: Page, label: string): Locator {
  return page.locator(`footer.status [data-figure="${label}"] b`);
}

/** The rows of the section of that name. */
function rows(page: Page, section: string): Locator {
  return page.locator(`.der-sect[data-section="${section}"] tbody tr:not(.more)`);
}

test.describe('the products, live', () => {
  test('shows the six tabs the specification names, with the schema’s own line under each', async ({
    page,
  }) => {
    await open(page);
    await openModel(page);
    const panel = await showPanel(page);
    await expect(panel.locator('.der-rail .row .n')).toHaveText(['D1', 'D2', 'D3', 'D4', 'D5', 'D6']);
    await expect(panel.locator('.der-rail .row .pn').first()).toHaveText(
      'Derived Computation Graph',
    );
    await showProduct(page, 'd3');
    await expect(panel.locator('.der-head h2')).toHaveText('Derived Parameter Tensor Inventory');
    // §1's "help text is the schema's": the line beside the name is the derived schema's own.
    await expect(panel.locator('.der-head .note-line')).toHaveText(
      'Parameter tensors (§7): one entry per identity instance; a tied tensor once.',
    );
    await expect(panel.locator('.der-head .pill')).toHaveText('fresh');
    await expect(panel.locator('.der-head [data-header="assignment"]')).toHaveText('assignment: —');
  });

  test('has 32 rows in D4 — one state identity per layer', async ({ page }) => {
    await open(page);
    await openModel(page);
    await showPanel(page);
    await showProduct(page, 'd4');
    await expect(rows(page, 'states')).toHaveCount(32);
    await expect(
      page.locator('.der-body [data-section="totals"] [data-field="identities"] b'),
    ).toHaveText('32');
  });

  test('holds every tab to the selected node, and lets the chip clear it', async ({ page }) => {
    await open(page);
    await openModel(page);
    await showPanel(page);
    await showProduct(page, 'd3');
    await expect(rows(page, 'tensors')).toHaveCount(200);
    await expect(page.locator('.der-sect[data-section="tensors"] .ih .ihn')).toHaveText('291');
    // Selecting the site on the canvas is what §4.18 calls the selection: `attn` is a declaration
    // and D3 names its thirty-two iterations, four tensors each. A composition is collapsed by
    // default (§4.7), so it is opened first.
    await page.locator('[data-fold="/compositions/decoder"]').click();
    await page.locator('[data-box="/compositions/decoder/instances/attn"]').click();
    await expect(page.locator('.der-rail .badge.own')).toContainText('attn');
    await expect(page.locator('.der-sect[data-section="tensors"] .ih .ihn')).toHaveText('128 of 291');
    await page.locator('.der-rail .badge.own .tbtn').click();
    await expect(page.locator('.der-sect[data-section="tensors"] .ih .ihn')).toHaveText('291');
  });

  test('holds every tab to a split, which is what §4.18’s own control does', async ({ page }) => {
    await open(page);
    await openModel(page);
    await showPanel(page);
    await showProduct(page, 'd6');
    const split = page.locator('.der-sect[data-section="graph_splits"] .link[data-names="split"]');
    await expect(split.first()).toHaveAttribute('title', 'Show split on canvas');
    await split.first().click();
    await expect(page.locator('.der-body [data-split="decoder[layer<=0]"]')).toBeVisible();
    await expect(page.locator('.der-rail .badge.own')).toHaveText(/decoder\[layer<=0\]/);
    await expect(page.locator('.der-sect[data-section="graph_splits"] .ih .ihn')).toHaveText(
      '1 of 38',
    );
    // And the block it names is what the row carries, which is the other half of §4.18's sentence.
    await expect(
      page.locator('.der-sect[data-section="graph_splits"] tbody tr:not(.more) td').nth(3),
    ).toHaveText('1');
  });

  test('follows an identifier to the place the document writes it', async ({ page }) => {
    await open(page);
    await openModel(page);
    await showPanel(page);
    await showProduct(page, 'd1');
    // A D1 node identifier selects the folded box and sets the composition's scrubber (§4.18).
    await page
      .locator('.der-sect[data-section="nodes"] .link[data-name="decoder/attn[layer=3]"]')
      .click();
    await expect(page.locator('.insp-title')).toHaveText('attn');
    // And the scrubber of the composition stands where the identifier said (§4.8, §4.18's "the
    // folded node **and the index**"): opening the drill-in shows the index the row named.
    await page.locator('[data-box="/compositions/decoder"] .n-name').click({ button: 'right' });
    await page.locator('.ctxmenu button[data-entry="canvas.drill-in"]').click();
    await expect(page.locator('.drill')).toBeVisible();
    await expect(page.locator('.scrub [data-scrub-value="layer"]')).toHaveText('layer = 3');
  });
});

test.describe('the status bar’s four totals', () => {
  test.skip(!oracleGenerated, 'the oracle has not been generated in this working copy');

  test('equal the products `tools/tensorspine --derive` writes', async ({ page }) => {
    await open(page);
    await openModel(page);
    const written = oracle();
    // The four figures of §4.2, each found in the derived document by `presentation.json`'s own
    // `statusBar` marks (feature 2.6) and rendered by `view.py`'s own `fmt_bytes` / `fmt_ops`.
    await expect(figure(page, 'params')).toHaveText('14.96 GiB');
    await expect(figure(page, '/ element')).toHaveText('15.01 Gop');
    await expect(figure(page, '/ cached position')).toHaveText('128 KiB');
    await expect(figure(page, 'peak live')).toHaveText('509 KiB');
    // The exact number behind each, which is the oracle's own figure and not a rounding of it.
    const grouped = (value: number): string =>
      String(value).replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
    await expect(page.locator('footer.status [data-figure="params"]')).toHaveAttribute(
      'title',
      grouped(written.d3.totals.bytes),
    );
    await expect(page.locator('footer.status [data-figure="/ element"]')).toHaveAttribute(
      'title',
      grouped(written.d5.operations.element.value),
    );
    await expect(page.locator('footer.status [data-figure="/ cached position"]')).toHaveAttribute(
      'title',
      grouped(written.d4.totals.append_bytes_per_cached_position),
    );
    await expect(page.locator('footer.status [data-figure="peak live"]')).toHaveAttribute(
      'title',
      grouped(written.d2.peak_live.bytes_per_element),
    );
    // And D3's own totals in the panel say the same thing, from the same document.
    await showPanel(page);
    await showProduct(page, 'd3');
    await expect(page.locator('.der-body [data-section="totals"] [data-field="tensors"] b')).toHaveText(
      String(written.d3.totals.tensors),
    );
    await expect(rows(page, 'states')).toHaveCount(0);
    await showProduct(page, 'd4');
    await expect(rows(page, 'states')).toHaveCount(written.d4.states.length);
  });
});

test.describe('freshness', () => {
  test('dims the figures with `stale` until the next derivation, and names the edit', async ({
    page,
  }) => {
    await open(page);
    await openModel(page);
    await showPanel(page);
    await showProduct(page, 'd3');
    await expect(page.locator('.der-head .pill')).toHaveText('fresh');
    await expect(page.locator('footer.status .fig.stale')).toHaveCount(0);

    // A named command of D13 — the rename the explorer and the sheet both offer — and the figures
    // beside it are now about the document as it was. The panel keeps them and dims them; §5.4:
    // "anything computed for an older revision is shown dimmed with `stale`".
    await page.locator('[data-box="/instances/embed"]').click();
    const name = page.locator('.insp input[data-name-field]');
    await name.fill('embed2');
    await name.press('Enter');
    await expect(page.locator('.der-head .pill')).toContainText('stale since');
    // D13's own name for the gesture: "every gesture is a named command … the Edit menu names
    // them", and the header says which one the figures are behind.
    await expect(page.locator('.der-head [data-freshness="stale"]')).toContainText(
      'Rename instance embed to embed2',
    );
    await expect(page.locator('.der-body .totals.stale')).not.toHaveCount(0);
    await expect(page.locator('footer.status .fig.stale')).not.toHaveCount(0);
    // And the same for the figures the canvas draws — §4.7's derived line on a card and on a
    // composition box, which feature 2.9 built and which this is the other half of.
    await expect(page.locator('.n-der.stale .stalebdg').first()).toBeVisible();

    // And it comes back: the next derivation is for this revision, and nothing is dim.
    await derived(page);
    await expect(page.locator('.der-head .pill')).toHaveText('fresh');
    await expect(page.locator('.der-body .totals.stale')).toHaveCount(0);
    await expect(page.locator('footer.status .fig.stale')).toHaveCount(0);
    await expect(page.locator('.n-der.stale')).toHaveCount(0);
  });
});

test.describe('Export Derived Document…', () => {
  test.skip(!oracleGenerated, 'the oracle has not been generated in this working copy');

  test('writes a document that deep-equals the core’s own derivation', async ({ page }) => {
    await open(page);
    await openModel(page);
    // The Examples workspace is read-only, so the command hands the file to the reader as a
    // download — which is the same `Workspace.write` a folder the browser can write takes in
    // place (feature 2.4). The download is what a suite can read.
    const [download] = await Promise.all([
      page.waitForEvent('download'),
      command(page, 'File', 'file.export-derived'),
    ]);
    expect(download.suggestedFilename()).toBe('llama3-8b.derived.json');
    const stream = await download.createReadStream();
    const chunks: Buffer[] = [];
    for await (const chunk of stream) chunks.push(chunk as Buffer);
    const text = Buffer.concat(chunks).toString('utf8');
    expect(JSON.parse(text)).toEqual(oracle());
    expect(text.endsWith('}\n')).toBe(true);
    // "Validated by the core against the derived schema before writing, as the tools do": a clean
    // export writes no refusal into the Log, and the line it does write names the file — which is
    // also what the toast says, the workspace being one the browser cannot write.
    await command(page, 'View', 'view.log');
    await expect(page.locator('.panel .logline').last()).toContainText('llama3-8b.derived.json');
    await expect(page.locator('.panel .logline')).not.toContainText([/off the derived schema/]);
  });
});

test.describe('the edge labels of §4.18 and the inventory’s convention', () => {
  test('reads `embed.output` as its value type', async ({ page }) => {
    await open(page);
    await openModel(page);
    // View ▸ Show Edge Types (§4.7): the label is D2's own value, printed by the core's
    // `valueGeometry` — the convention `tools/view.py` writes today and which the editor takes
    // over when it is removed (finding F6).
    await command(page, 'View', 'view.show-edge-types');
    // `decoder.entry` is the rule `llama3-8b` writes for `embed.output → decoder/attn_n.input`.
    const wire = page.locator('[data-wire="/bindings/values/decoder.entry"] .wtype').first();
    await expect(wire).toHaveText('bf16[tokens, model.width=4096]');
  });
});

test.describe('accessibility', () => {
  for (const theme of ['light', 'dark'] as const) {
    test(`has no axe violation in the ${theme} theme`, async ({ page }) => {
      await open(page);
      await command(page, 'View', theme === 'light' ? 'view.theme-light' : 'view.theme-dark');
      await openModel(page);
      await showPanel(page);
      await showProduct(page, 'd6');
      const found = await new AxeBuilder({ page }).include('.derived').analyze();
      expect(
        found.violations.flatMap((one) => one.nodes.map((node) => `${one.id}: ${node.html}`)),
      ).toEqual([]);
    });
  }
});
