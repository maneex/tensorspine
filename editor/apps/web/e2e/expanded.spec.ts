import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Locator, type Page } from '@playwright/test';

/**
 * The expanded graph — feature 2.16, plan §4.9, artboard S12.
 *
 * The three claims the feature's own block names, each of which needs a real page, a real worker
 * and a real derivation to be worth anything:
 *
 *   - **`qwen3.5-4b-text` expands to the node and edge counts of the oracle D1** — the strip's own
 *     line is the product's two numbers, and what they are held to is the document
 *     `tools/tensorspine --d1` wrote;
 *   - **the index filter `layer 0–3` shows the nodes D1 names there** — every identifier drawn is
 *     one the oracle's `d1.nodes` holds, and every node of those four iterations is drawn;
 *   - **"Show split on canvas" for `decoder[layer<=0]` shades the block D6 lists** — the gesture
 *     is the Derived panel's own link (§4.18), and what it shades is the seven identifiers D6
 *     wrote, not one more.
 *
 * Beside them: the tab opens on §4.4's own command, the chain is *virtualised* (a graph of 195
 * nodes draws a screenful, and scrolling draws the rows further down), selecting a node moves the
 * document's own selection, S5's `Layer preview` shows this view in place of the drill-in's
 * canvas, and axe finds nothing in either theme.
 *
 * Everything that is a question about *which rows a filter keeps* rather than about a browser is
 * asked in `packages/ui/test/expanded/`, over the corpus.
 */

// The tab is a filter strip over a chain with a facts panel beside it. At the suite's default
// 1280×720 the panel takes a third of the width and the chain is legible but cramped; the window
// is what a reader would give it, and it is stated here rather than worked around case by case.
test.use({ viewport: { width: 1600, height: 1000 } });

const repository = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', '..');
const QWEN = 'models/qwen3.5-4b-text.json';

/** The oracle's own `--d1` document, where `pnpm oracle` has been run. */
const oraclePath = join(repository, 'editor/tests/oracle/out/d1/qwen3.5-4b-text.d1.json');
const oracleGenerated = existsSync(oraclePath);

interface OracleD1 {
  d1: {
    nodes: Record<string, { primitive: { name: string } }>;
    edges: unknown[];
    topological_order: string[];
  };
}

function oracle(): OracleD1['d1'] {
  return (JSON.parse(readFileSync(oraclePath, 'utf8')) as OracleD1).d1;
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
async function openModel(page: Page, path = QWEN): Promise<void> {
  await page.locator('.nothing .link[data-open="examples"]').click();
  await page.locator('.dlg [data-workspace="examples"]').click();
  await expect(page.locator('footer.status [data-workspace]')).toHaveText('Examples');
  await command(page, 'File', 'file.open-model');
  await page.locator(`.dlg [data-document="${path}"]`).click();
  await expect(page.locator(`.gcanvas[data-canvas="${path}"]`)).toBeVisible();
  await expect(page.locator('.bar [data-pill="derivation"]')).toHaveText('derived · fresh', {
    timeout: 60_000,
  });
}

/** `View ▸ Expanded Graph` — §4.4's own entry, which is how a person opens it. */
async function openExpanded(page: Page): Promise<Locator> {
  await command(page, 'View', 'view.expanded-graph');
  const view = page.locator('.xgraph');
  await expect(view).toBeVisible();
  return view;
}

/** The identifiers the chain has drawn, in the order it drew them. */
async function drawn(page: Page): Promise<string[]> {
  return page.locator('.xchain .xg-row').evaluateAll((rows) =>
    rows.map((row) => row.getAttribute('data-row') ?? ''),
  );
}

test.describe('the tab over D1', () => {
  test('opens on View ▸ Expanded Graph, as a tab of its own, read-only', async ({ page }) => {
    await open(page);
    await openModel(page);
    const view = await openExpanded(page);
    await expect(page.locator('.tabs .tab[aria-current="true"]')).toContainText(
      'qwen3.5-4b-text › expanded graph',
    );
    await expect(view.locator('.canvas-note')).toHaveText('read-only · virtualised');
    // Asking again selects the tab it already has, as a drill-in does.
    await command(page, 'View', 'view.expanded-graph');
    await expect(page.locator('.tabs .tab')).toHaveCount(2);
    // S12's own sentence, and the reason the tab has no gesture that writes.
    await expect(view.locator('[data-no-node]')).toHaveText(
      'Select a node to see its D1 arguments, with the defaults applied.',
    );
  });

  test('states the node and edge counts of the oracle’s own D1', async ({ page }) => {
    test.skip(!oracleGenerated, 'the oracle has not been generated in this working copy');
    const d1 = oracle();
    const nodes = Object.keys(d1.nodes).length;
    await open(page);
    await openModel(page);
    const view = await openExpanded(page);
    const counts = view.locator('.strip .right .note-line');
    await expect(counts).toHaveAttribute(
      'data-counts',
      `${String(nodes)}/${String(nodes)}`,
    );
    await expect(counts).toHaveAttribute(
      'data-edges',
      `${String(d1.edges.length)}/${String(d1.edges.length)}`,
    );
    await expect(counts).toHaveText(`${String(nodes)} nodes · ${String(d1.edges.length)} edges`);
    // The figures the design pass drew on S12, reached by the core rather than written down.
    expect(nodes).toBe(195);
    expect(d1.edges.length).toBe(258);
  });

  test('is virtualised: 195 nodes draw a screenful, and scrolling draws the rest', async ({
    page,
  }) => {
    test.skip(!oracleGenerated, 'the oracle has not been generated in this working copy');
    const order = oracle().topological_order;
    await open(page);
    await openModel(page);
    const view = await openExpanded(page);

    const first = await drawn(page);
    // A screenful and its overscan — never 195 rows, which is what §4.9's "virtualised" buys.
    expect(first.length).toBeGreaterThan(8);
    expect(first.length).toBeLessThan(60);
    expect(first[0]).toBe(order[0]);
    // The rows are D1's own topological order, and the chain's first card has no wire above it.
    expect(first).toEqual(order.slice(0, first.length));
    await expect(page.locator('.xchain .xg-row').first().locator('.miwire')).toHaveClass(/first/);

    // Scrolling shows rows further down the order, and the same number of them.
    await view.locator('.xg-scroll').evaluate((element) => {
      element.scrollTop = 100 * 40;
    });
    await expect(page.locator(`.xchain [data-row="${order[100] ?? ''}"]`)).toBeVisible();
    await expect(page.locator(`.xchain [data-row="${order[0] ?? ''}"]`)).toHaveCount(0);
    const later = await drawn(page);
    expect(later.length).toBeLessThan(60);
    expect(later).not.toEqual(first);
    expect(later.every((id) => order.includes(id))).toBe(true);
  });

  test('is walked by the arrows, and the window follows the focus past its own edge', async ({
    page,
  }) => {
    test.skip(!oracleGenerated, 'the oracle has not been generated in this working copy');
    const order = oracle().topological_order;
    await open(page);
    await openModel(page);
    const view = await openExpanded(page);

    // One Tab reaches the list, because the roving `tabIndex` puts exactly one row in the tab
    // order — 195 rows behind Tab is a list nobody walks.
    const chain = view.locator('.xchain');
    await expect(chain).toHaveAttribute('role', 'listbox');
    await expect(chain.locator('[tabindex="0"]')).toHaveCount(1);

    await view.locator('.xg-row').first().focus();
    const drew = (await drawn(page)).length;
    // With nothing selected the first arrow lands *on* the first row rather than stepping past it.
    await page.keyboard.press('ArrowDown');
    await expect(view.locator('.xg-facts [data-node]')).toHaveText(order[0] ?? '');
    // Past the edge of what was first drawn: the move scrolls, the window advances, and the focus
    // lands on a row that was not rendered when the key was pressed.
    const at = drew + 4;
    for (let step = 0; step < at; step += 1) await page.keyboard.press('ArrowDown');
    const focused = await page.evaluate(() =>
      document.activeElement?.closest('[data-row]')?.getAttribute('data-row'),
    );
    expect(focused).toBe(order[at]);
    // Moving the focus is selecting, as a listbox reads it: the facts panel and the document's own
    // selection both followed.
    await expect(view.locator('.xg-facts [data-node]')).toHaveText(order[at] ?? '');
    await expect(view.locator(`.xchain [data-row="${order[at] ?? ''}"]`)).toHaveAttribute(
      'aria-selected',
      'true',
    );
    // And `End` reaches the last row of the graph, whatever the window held.
    await page.keyboard.press('End');
    await expect(view.locator('.xg-facts [data-node]')).toHaveText(order.at(-1) ?? '');
  });

  test('selecting a node shows its D1 arguments and selects the place the document declares', async ({
    page,
  }) => {
    await open(page);
    await openModel(page);
    const view = await openExpanded(page);
    await view.locator('.xchain [data-row="decoder/gdn[layer=0]"] .xn2').click();

    await expect(view.locator('.xg-facts [data-node]')).toHaveText('decoder/gdn[layer=0]');
    await expect(view.locator('.xg-facts .insp-kind')).toContainText('D1 node · position');
    await expect(view.locator('.xg-facts .prim-chip')).toHaveText('sequence.gated_delta@1.0.0');
    // "its D1 arguments (defaults applied)": a value the document does not write, that D1 does.
    // The figure is written by the Derived panel's own `cellOf`, so a whole number is grouped as
    // `view.py`'s `fmt_int` groups one — there is one rendering of a figure and this is it.
    await expect(view.locator('.xg-facts [data-argument="width"] .av')).toHaveText('2 560');
    await expect(view.locator('.xg-facts [data-field="across_positions"]')).toHaveText('yes');

    // And the *place* is selected, which is what makes the Derived panel show its rows (§4.9):
    // the chip names the declaration, since a selection is a place and a place is a family of D1
    // nodes (feature 2.15's own reading of the subject).
    await command(page, 'View', 'view.derived');
    await expect(page.locator('.der-rail .badge.own')).toContainText('gdn');
    await page.locator('.der-rail [data-product="d1"]').click();
    await expect(page.locator('.der-sect[data-section="nodes"] .ih .ihn')).toHaveText('24 of 195');
  });

  test('has nothing axe objects to, in either theme, filtered and shaded', async ({ page }) => {
    await open(page);
    await openModel(page);
    const view = await openExpanded(page);
    await view.locator('.xchain .xn2').first().click();
    for (const theme of ['view.theme-dark', 'view.theme-light']) {
      await command(page, 'View', theme);
      expect(await audit(page), theme).toEqual([]);
      await view.locator('[data-bound="layer.to"]').selectOption('3');
      expect(await audit(page), `${theme}, filtered`).toEqual([]);
      await view.locator('[data-filter="clear"]').click();
    }
  });
});

test.describe('§4.9’s filters', () => {
  test('the index filter `layer 0–3` shows the nodes D1 names there', async ({ page }) => {
    test.skip(!oracleGenerated, 'the oracle has not been generated in this working copy');
    const d1 = oracle();
    await open(page);
    await openModel(page);
    const view = await openExpanded(page);

    await view.locator('[data-bound="layer.from"]').selectOption('0');
    await view.locator('[data-bound="layer.to"]').selectOption('3');

    // What D1 names there: every node whose `layer` is 0…3, and the root instances, which no
    // composition index reaches.
    const expected = d1.topological_order.filter((id) => {
      const match = /\[layer=(\d+)\]/.exec(id);
      return match === null || Number(match[1]) <= 3;
    });
    const counts = view.locator('.strip .right .note-line');
    await expect(counts).toHaveAttribute(
      'data-counts',
      `${String(expected.length)}/${String(d1.topological_order.length)}`,
    );
    await expect(counts).toContainText(
      `${String(expected.length)} of ${String(d1.topological_order.length)} nodes pass the filter`,
    );
    // What is *drawn* is the window, so the rows are the head of that list — and scrolling to the
    // end shows its tail. Both are D1's own identifiers, in D1's own order.
    const head = await drawn(page);
    expect(head.length).toBeGreaterThan(8);
    expect(head).toEqual(expected.slice(0, head.length));
    await view.locator('.xg-scroll').evaluate((element) => {
      element.scrollTop = element.scrollHeight;
    });
    // The window is redrawn on the scroll event, so the last row is what says the redraw landed.
    await expect(page.locator(`.xchain [data-row="${expected.at(-1) ?? ''}"]`)).toBeVisible();
    const tail = await drawn(page);
    expect(tail.at(-1)).toBe(expected.at(-1));
    expect(tail).toEqual(expected.slice(expected.length - tail.length));

    // "Show all" puts every node back.
    await view.locator('[data-filter="clear"]').click();
    await expect(counts).toHaveText(
      `${String(d1.topological_order.length)} nodes · ${String(d1.edges.length)} edges`,
    );
  });

  test('the family and primitive filters are the values D1 writes, and the search is by identifier', async ({
    page,
  }) => {
    await open(page);
    await openModel(page);
    const view = await openExpanded(page);

    await view.locator('[data-filter="primitive"]').selectOption('sequence.gated_delta');
    const ids = await drawn(page);
    expect(ids.length).toBeGreaterThan(0);
    expect(ids.every((id) => id.startsWith('decoder/gdn['))).toBe(true);

    await view.locator('[data-filter="clear"]').click();
    // `qwen3.5-4b-text` alternates its two sequence operators by guard — 24 `gdn` and 8 `attn`
    // over 32 layers — so `gdn[layer=6]` is a node D1 emitted and `gdn[layer=7]` is not.
    await view.locator('[data-filter="search"]').fill('gdn[layer=6]');
    await expect(page.locator('.xchain .xg-row')).toHaveCount(1);
    await expect(page.locator('.xchain .xg-row')).toHaveAttribute(
      'data-row',
      'decoder/gdn[layer=6]',
    );
    await view.locator('[data-filter="search"]').fill('gdn[layer=7]');
    await expect(view.locator('[data-no-rows]')).toHaveText('No node passes the filter.');

    await view.locator('[data-filter="search"]').fill('no such node');
    await expect(view.locator('[data-no-rows]')).toHaveText('No node passes the filter.');
  });
});

test.describe('§4.18’s "Show split on canvas"', () => {
  test('shades the block D6 lists for `decoder[layer<=0]`, and nothing else', async ({ page }) => {
    test.skip(!oracleGenerated, 'the oracle has not been generated in this working copy');
    const derived = JSON.parse(
      readFileSync(
        join(repository, 'editor/tests/oracle/out/derive/qwen3.5-4b-text.derived.json'),
        'utf8',
      ),
    ) as { d6: { graph_splits: { graph_split: string; block: string[]; sizes: number[] }[] } };
    const split = derived.d6.graph_splits.find((one) => one.graph_split === 'decoder[layer<=0]');
    expect(split, 'D6 lists the split the feature names').toBeDefined();

    await open(page);
    await openModel(page);
    // The gesture is §4.18's own: the link on the D6 splits tab, whose title says what it does.
    await command(page, 'View', 'view.derived');
    await page.locator('.der-rail [data-product="d6"]').click();
    const link = page.locator(
      '.der-sect[data-section="graph_splits"] .link[data-name="decoder[layer<=0]"]',
    );
    await expect(link.first()).toHaveAttribute('title', 'Show split on canvas');
    await link.first().click();

    // It opens the expanded graph, which is the half of the sentence feature 2.15 left to this one.
    const view = page.locator('.xgraph');
    await expect(view).toBeVisible();
    await expect(view.locator('[data-shading="decoder[layer<=0]"]')).toBeVisible();
    await expect(view.locator('.canvas-note')).toHaveText(
      'read-only · virtualised · shading decoder[layer<=0]',
    );
    await expect(view.locator('[data-shading="decoder[layer<=0]"]')).toContainText(
      `${String(split?.sizes[0] ?? 0)} nodes · ${String(split?.sizes[1] ?? 0)} beyond · 1 crossing value`,
    );

    // The block is D6's own list: exactly those rows are marked, in D1's order.
    const shaded = await page
      .locator('.xchain .xg-row[data-block="in"]')
      .evaluateAll((rows) => rows.map((row) => row.getAttribute('data-row') ?? ''));
    expect(shaded).toEqual(split?.block);
    expect(shaded).toHaveLength(7);
    // Everything drawn beside it is outside the block and says so.
    const out = await page.locator('.xchain .xg-row[data-block="out"]').count();
    expect(out).toBeGreaterThan(0);

    // And the Derived panel's banner now says the block is shaded, where 2.15 said it was not built.
    await expect(page.locator('.der-body [data-split="decoder[layer<=0]"]')).toContainText(
      'Its block is shaded on the expanded graph.',
    );

    // Clearing it from the view stops the shading.
    await view.locator('[data-shading="clear"]').click();
    await expect(view.locator('[data-shading="decoder[layer<=0]"]')).toHaveCount(0);
    await expect(page.locator('.xchain .xg-row[data-block]')).toHaveCount(0);
  });
});

test.describe('§4.9’s layer preview', () => {
  test('shows this view in place of the drill-in’s canvas, restricted to the scrubbed index', async ({
    page,
  }) => {
    await open(page);
    await openModel(page);
    // §4.8's drill-in, through §4.7's context menu — the visible way in (feature 2.14's own).
    await page.locator('[data-box="/compositions/decoder"] .n-name').click({ button: 'right' });
    await page.locator('.ctxmenu button[data-entry="canvas.drill-in"]').click();
    await expect(page.locator('.drill')).toBeVisible();

    await page.locator('.strip [data-scrub="layer"]').fill('2');
    await expect(page.locator('.scrub [data-scrub-value="layer"]')).toHaveText('layer = 2');

    // S5's own button, beside the scrubber.
    await page.locator('[data-layer-preview="decoder"]').click();
    await expect(page.locator('.drill .xgraph')).toBeVisible();
    await expect(page.locator('.drill .gcanvas')).toHaveCount(0);
    await expect(page.locator('.drill [data-preview="decoder"]')).toHaveText('decoder');
    await expect(page.locator('.drill .strip .note-line').first()).toHaveText(
      'restricted to layer=2',
    );

    // Every row is one node of that composition at that index, and nothing else.
    const ids = await drawn(page);
    expect(ids.length).toBeGreaterThan(0);
    expect(ids.every((id) => id.startsWith('decoder/') && id.endsWith('[layer=2]'))).toBe(true);

    // And it goes back, the button being a toggle and the canvas being what the tab is for.
    await page.locator('[data-layer-preview="decoder"]').click();
    await expect(page.locator('.drill .gcanvas')).toBeVisible();
    await expect(page.locator('.drill .xgraph')).toHaveCount(0);
  });

  test('is offered by View ▸ Layer Preview, and says so where there is no drill-in', async ({
    page,
  }) => {
    await open(page);
    await openModel(page);
    await command(page, 'View', 'view.layer-preview');
    // No drill-in is open, so the command says why in the Log rather than opening something else.
    await command(page, 'View', 'view.log');
    await expect(page.locator('.logline').last()).toContainText('Layer Preview');
  });
});

/**
 * What axe finds on the page, as a list of `rule element` lines.
 *
 * The wordmark is excluded from the colour-contrast rule alone, for the reason feature 2.5
 * recorded and every pass since has repeated: WCAG 1.4.3 exempts "text that is part of a logo or
 * brand name", and repainting it would be redrawing the logo (§4.21).
 */
async function audit(page: Page): Promise<string[]> {
  const result = await new AxeBuilder({ page }).exclude('.wordmark').analyze();
  const wordmark = await new AxeBuilder({ page })
    .include('.wordmark')
    .disableRules(['color-contrast'])
    .analyze();
  return [...result.violations, ...wordmark.violations].flatMap((one) =>
    one.nodes.map((node) => `${one.id} ${node.target.join(' ')}`),
  );
}
