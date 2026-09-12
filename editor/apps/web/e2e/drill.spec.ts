import { readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Locator, type Page } from '@playwright/test';

import { sourceText } from './source-text.js';

/**
 * The composition drill-in — feature 2.14, plan §4.8 and §4.20, artboards S4 and S5.
 *
 * What the browser layer is for here is everything a model cannot be asked: that the tab opens on
 * the gesture §4.4 gives it, that the ghost column is drawn beside the chain rather than in it,
 * that the scrubber dims what D1 says is absent and says why, that the alternation strip counts
 * what the tools' own expansion emitted, and that §4.20's two moves leave a document the core
 * still validates and derives.
 *
 * The alternation counts are the oracle's: `packages/lang/test/parity/drill.test.ts` reads them
 * out of `--d1`'s own output, and these are the same four numbers drawn.
 */

// A drill-in is a tab with three things stacked in it — the index strip, the canvas and the
// alternation strip — over a composition of six to seventeen sites laid out top to bottom. At the
// suite's default 1280×720 the canvas is left about a hundred pixels and `fit` shrinks a card to a
// fraction of a pixel: the drawing is correct and nothing can be aimed at. The window is what a
// reader would give it, and it is stated here rather than worked around case by case.
test.use({ viewport: { width: 1600, height: 1200 } });

const repository = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', '..');
const LLAMA = 'models/llama3-8b.json';
const GEMMA = 'models/gemma3n-kvshare.json';
const llamaText = readFileSync(join(repository, 'data/models/llama3-8b.json'), 'utf8');

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

/** Open the Examples workspace and one document of it, and wait for the canvas to be drawn. */
async function openModel(page: Page, path: string): Promise<void> {
  await page.locator('.nothing .link[data-open="examples"]').click();
  await page.locator('.dlg [data-workspace="examples"]').click();
  await expect(page.locator('footer.status [data-workspace]')).toHaveText('Examples');
  await command(page, 'File', 'file.open-model');
  await page.locator(`.dlg [data-document="${path}"]`).click();
  await expect(page.locator(`.gcanvas[data-canvas="${path}"]`)).toBeVisible();
  await expect(page.locator('[data-box="/compositions/decoder"]')).toBeVisible();
}

/** Drill into a composition through §4.7's context menu, which is the visible way in. */
async function drillIn(page: Page, composition: string): Promise<void> {
  await page.locator(`[data-box="/compositions/${composition}"] .n-name`).click({ button: 'right' });
  await page.locator('.ctxmenu button[data-entry="canvas.drill-in"]').click();
  await expect(page.locator('.drill')).toBeVisible();
}

/** One box of the drill-in, by the place it stands for. */
function box(page: Page, pointer: string): Locator {
  return page.locator(`[data-box="${pointer}"]`);
}

test.describe('llama3-8b › decoder (S4)', () => {
  test('opens as a tab of its own, with the six sites, the strip and the terminals', async ({ page }) => {
    await open(page);
    await openModel(page, LLAMA);
    await drillIn(page, 'decoder');

    // §4.2's "one per drill-in", named as §4.8 writes it.
    await expect(page.locator('.tabs .tab[aria-current="true"]')).toContainText('llama3-8b › decoder');

    for (const site of ['attn_n', 'attn', 'attn_r', 'ffn_n', 'ffn', 'ffn_r']) {
      await expect(box(page, `/compositions/decoder/instances/${site}`)).toBeVisible();
    }
    await expect(page.locator('.gcanvas .node:not(.tiny)')).toHaveCount(6);

    // The index strip: the bounds as the document writes them (erratum E10's literal 32).
    await expect(page.locator('.strip [data-index="layer"]')).toHaveText('layer');
    await expect(page.locator('.strip [data-bound="layer.start"]')).toHaveText('0');
    await expect(page.locator('.strip [data-bound="layer.stop"]')).toHaveText('32');
    await expect(page.locator('.strip [data-bound="layer.step"]')).toHaveText('1');

    // The pinned terminals of §4.8, with the rules on each.
    await expect(page.locator('.bterm b')).toHaveText([
      '◁ embed.output at layer = 0',
      '▷ final_n.input at layer = layers - 1',
    ]);
    await expect(page.locator('.bterm [data-rule="decoder.entry"]')).toHaveText(
      'decoder.entry → attn_n.input',
    );
  });

  test('draws one ghost column, with the two carry edges and their guard', async ({ page }) => {
    await open(page);
    await openModel(page, LLAMA);
    await drillIn(page, 'decoder');

    const ghost = page.locator('.ghostcol');
    await expect(ghost).toHaveCount(1);
    await expect(ghost.locator('b')).toHaveText('ffn_r');
    await expect(ghost.locator('.n-prim')).toHaveText('[$layer - 1]');

    // Two edges through it, each with the guard §4.8 names.
    const carry = page.locator('.elbl[data-wire$="attn_n.carry"]');
    await expect(carry).toBeVisible();
    await expect(carry.locator('.wguard')).toContainText('$layer >= 1');
    await expect(page.locator('.elbl[data-wire$="attn_r.a_carry"] .wguard')).toContainText(
      '$layer >= 1',
    );

    // The column is drawn to the **left** of every card, which is what §4.8 asks of it.
    const at = await ghost.boundingBox();
    const card = await box(page, '/compositions/decoder/instances/attn_n').boundingBox();
    expect(at!.x + at!.width).toBeLessThanOrEqual(card!.x);
  });

  test('the scrubber at layer 0 dims the carry edges and says the guard is false', async ({ page }) => {
    await open(page);
    await openModel(page, LLAMA);
    await drillIn(page, 'decoder');

    // Unset: every site and every scoped edge, as §4.8 describes the representative iteration.
    await expect(page.locator('.scrub .val')).toHaveText('unset');
    await expect(page.locator('.wires .w.dim')).toHaveCount(0);

    await page.locator('.strip [data-scrub="layer"]').fill('0');
    await expect(page.locator('.scrub [data-scrub-value="layer"]')).toHaveText('layer = 0');

    // The two carry edges are absent at layer 0 — D1 emitted neither — so both are dimmed, and so
    // is the ghost column they run from. The third is the boundary edge *out* of the composition,
    // which belongs to the last iteration and not to this one.
    await expect(page.locator('.wires .w.dim')).toHaveCount(3);
    await expect(page.locator('.elbl.dim')).toHaveText([
      'attn_n.carry⚑ $layer >= 1 — false',
      'attn_r.a_carry⚑ $layer >= 1 — false',
      'final_n.in',
    ]);
    await expect(page.locator('.ghostcol.dim')).toHaveCount(1);
    await expect(page.locator('.elbl[data-wire$="attn_n.carry"] .wguard b')).toHaveText('— false');
    await expect(page.locator('.elbl[data-wire$="attn_n.carry"] .wguard b')).toHaveClass(/fails/);

    // The boundary into the composition belongs to layer 0, so it stays.
    await expect(page.locator('.bterm.dim b')).toHaveText('▷ final_n.input at layer = layers - 1');

    await page.locator('.strip [data-scrub="layer"]').fill('1');
    // At layer 1 the carries fire; the three boundary edges belong to the first and the last.
    await expect(page.locator('.elbl.dim')).toHaveText([
      'decoder.entry',
      'decoder.entry.a',
      'final_n.in',
    ]);
    await expect(page.locator('.elbl[data-wire$="attn_n.carry"] .wguard b')).toHaveText('— true');

    // Unset puts the representative iteration back.
    await page.locator('.strip [data-scrub-unset]').click();
    await expect(page.locator('.scrub .val')).toHaveText('unset');
    await expect(page.locator('.ghostcol.dim')).toHaveCount(0);
  });

  test('says that no site of this composition is guarded', async ({ page }) => {
    await open(page);
    await openModel(page, LLAMA);
    await drillIn(page, 'decoder');
    await expect(page.locator('.ribbon-strip [data-no-rows]')).toHaveText(
      'No site of decoder carries a guard.',
    );
    await expect(page.locator('.ribbon-strip [data-unguarded]')).toContainText('6');
    await expect(page.locator('.ribbon-strip [data-columns]')).toHaveText('32 iterations');
  });

  test('has nothing axe objects to, in either theme', async ({ page }) => {
    await open(page);
    await openModel(page, LLAMA);
    await drillIn(page, 'decoder');
    for (const theme of ['view.theme-dark', 'view.theme-light']) {
      await command(page, 'View', theme);
      expect(await audit(page), theme).toEqual([]);
      await page.locator('.strip [data-scrub="layer"]').fill('0');
      expect(await audit(page), `${theme}, scrubbed`).toEqual([]);
      await page.locator('.strip [data-scrub-unset]').click();
    }
  });
});

test.describe('the three ways into a drill-in, and the way back (§4.4)', () => {
  test('Ctrl+Enter drills in, Ctrl+↑ returns to the canvas above it', async ({ page }) => {
    await open(page);
    await openModel(page, LLAMA);

    await page.locator('[data-box="/compositions/decoder"] .n-name').click();
    await page.keyboard.press('Control+Enter');
    await expect(page.locator('.drill')).toBeVisible();
    await expect(page.locator('.tabs .tab[aria-current="true"]')).toContainText('llama3-8b › decoder');

    await page.locator('.canvas.graph').click({ position: { x: 4, y: 4 } });
    await page.keyboard.press('Control+ArrowUp');
    await expect(page.locator('.drill')).toHaveCount(0);
    await expect(page.locator('.tabs .tab[aria-current="true"]')).toHaveText(/^llama3-8b/);
    // The drill-in is still open beside it: Ctrl+↑ goes back, it does not close anything.
    await expect(page.locator('.tabs .tab', { hasText: '› decoder' })).toBeVisible();
  });

  test('a double-click on the explorer’s composition row opens it (feature 2.7’s hand-over)', async ({
    page,
  }) => {
    await open(page);
    await openModel(page, LLAMA);
    await page.locator('.tree [data-row="/compositions/decoder"]').dblclick();
    await expect(page.locator('.drill')).toBeVisible();
  });

  test('“Show in canvas” on the composition’s sheet opens it (§4.11, feature 2.12’s hand-over)', async ({
    page,
  }) => {
    await open(page);
    await openModel(page, LLAMA);
    await page.locator('[data-box="/compositions/decoder"] .n-name').click();
    await page.locator('.insp [data-show-in-canvas="decoder"]').click();
    await expect(page.locator('.drill')).toBeVisible();
  });
});

test.describe('gemma3n-kvshare › decoder (S5)', () => {
  test('the alternation strip counts what D1 emitted, site by site', async ({ page }) => {
    await open(page);
    await openModel(page, GEMMA);
    await drillIn(page, 'decoder');

    // The four guarded sites of the composition, with the counts the oracle's D1 carries.
    await expect(page.locator('.rrow [data-count="attn"]')).toHaveText('24 of 30');
    await expect(page.locator('.rrow [data-count="attn_full"]')).toHaveText('6 of 30');
    await expect(page.locator('.rrow [data-count="ffn_sparse"]')).toHaveText('10 of 30');
    await expect(page.locator('.rrow [data-count="ffn"]')).toHaveText('20 of 30');
    await expect(page.locator('.rrow')).toHaveCount(4);
    // S5's own note: thirteen unguarded sites, present in all thirty and not listed.
    await expect(page.locator('.ribbon-strip [data-unguarded]')).toContainText('13');

    // One column per iteration, filled where D1 emitted the node.
    const attn = page.locator('.rrow[data-row="attn"] .rcells i');
    await expect(attn).toHaveCount(30);
    await expect(page.locator('.rrow[data-row="attn"] .rcells i.on')).toHaveCount(24);
    await expect(page.locator('.rrow[data-row="attn_full"] .rcells i.on')).toHaveCount(6);
  });

  test('has nothing axe objects to with the strip drawn, in either theme', async ({ page }) => {
    await open(page);
    await openModel(page, GEMMA);
    await drillIn(page, 'decoder');
    for (const theme of ['view.theme-dark', 'view.theme-light']) {
      await command(page, 'View', theme);
      expect(await audit(page), theme).toEqual([]);
    }
  });

  test('clicking a column moves the scrubber, and the absent site is dimmed', async ({ page }) => {
    await open(page);
    await openModel(page, GEMMA);
    await drillIn(page, 'decoder');

    // The fifth column is `layer = 4`, where `attn_full` exists and `attn` does not (S5's own
    // drawing).
    await page.locator('.rrow[data-row="attn"] .rcells i[data-at="4"]').click();
    await expect(page.locator('.scrub [data-scrub-value="layer"]')).toHaveText('layer = 4');
    await expect(page.locator('.rrow[data-row="attn"] .rcells i.cur')).toHaveCount(1);

    await expect(box(page, '/compositions/decoder/instances/attn')).toHaveClass(/ghosted/);
    await expect(box(page, '/compositions/decoder/instances/attn_full')).not.toHaveClass(/ghosted/);
    await expect(box(page, '/compositions/decoder/instances/ffn_sparse')).not.toHaveClass(/ghosted/);
    await expect(box(page, '/compositions/decoder/instances/ffn')).toHaveClass(/ghosted/);
  });

  test('duplicates a guarded site with the complementary guard (§4.20)', async ({ page }) => {
    await open(page);
    await openModel(page, GEMMA);
    await drillIn(page, 'decoder');

    await page
      .locator('[data-box="/compositions/decoder/instances/ffn_sparse"] .n-name')
      .click({ button: 'right' });
    await page.locator('.ctxmenu button[data-entry="canvas.duplicate-complementary"]').click();

    const copy = box(page, '/compositions/decoder/instances/ffn_sparse_alt');
    await expect(copy).toBeVisible();
    // The complement is the negation, never the comparison rewritten: `not ($layer < 10)`.
    await expect(copy.locator('.n-guard')).toContainText('not($layer < 10)');
    const after = await sourceText(page);
    expect(after).toContain('"ffn_sparse_alt"');
    expect(after).toContain('"not"');
  });
});

test.describe('§4.20’s moves', () => {
  test('Extract to Composition leaves a document that validates and derives', async ({ page }) => {
    await open(page);
    await openModel(page, LLAMA);

    // §4.4's rubber band over the whole drawing: every box it touches is marked, and the move
    // takes the root instances among them (the composition and the terminals are not instances).
    const canvas = page.locator('.canvas.graph');
    const at = await canvas.boundingBox();
    await page.keyboard.down('Shift');
    await page.mouse.move(at!.x + 3, at!.y + 3);
    await page.mouse.down();
    await page.mouse.move(at!.x + at!.width - 3, at!.y + at!.height - 3, { steps: 6 });
    await page.mouse.up();
    await page.keyboard.up('Shift');
    await expect(page.locator('.node.sel')).toHaveCount(3);

    // §4.11's intersection sheet, on the three of them.
    await expect(page.locator('.insp-title')).toHaveText('3 selected');
    await expect(page.locator('.insp-kind[data-selected]')).toHaveText('embed, final_n, lm_head');
    await expect(page.locator('[data-multi-primitive]')).toHaveText('multiple values');
    // `width` is the one argument all three declare, and all three write it the same way, so the
    // row is drawn rather than marked; what is *not* in common is named beside it.
    await expect(page.locator('.arow[data-argument="width"]')).toBeVisible();
    await expect(page.locator('.arow.more')).toContainText('arguments are not in common');

    // Opening the menu on one of the selected boxes keeps the band: a gesture on a group is
    // reached through one of its members, and clearing it there would make the selection unusable.
    await page.locator('[data-box="/instances/embed"] .n-name').click({ button: 'right' });
    await expect(page.locator('.node.sel')).toHaveCount(3);
    await page.locator('.ctxmenu button[data-entry="canvas.extract-to-composition"]').click();

    // The three instances are now sites of a composition, and the document still derives.
    await expect(box(page, '/compositions/block')).toBeVisible();
    // The status bar's own two fields: the core validated the document and derived it again.
    await expect(page.locator('footer.status [data-validation]')).toHaveText('no problems');
    await expect(page.locator('footer.status [data-derivation]')).toHaveText('derived · fresh');

    const after = await sourceText(page);
    expect(after).toContain('"block"');
    expect(after).toContain('"site": "lm_head"');
    // The boundary edges name the composition at an index: the entry at the first, the exit at
    // the last (`stop − 1`, which the one-iteration bound makes 0).
    expect(after).toContain('"composition": "block"');
    expect(llamaText).not.toContain('"block"');
  });

  test('Add to Composition moves a root instance into the composition it names', async ({ page }) => {
    await open(page);
    await openModel(page, LLAMA);

    await page.locator('[data-box="/instances/final_n"] .n-name').click({ button: 'right' });
    await page.locator('.ctxmenu button[data-entry="canvas.add-to-composition"]').click();
    // The ellipsis asks which composition: the document's own list is the answer.
    await page.locator('.ctxmenu button[data-entry="canvas.add-to-composition:decoder"]').click();

    await expect(page.locator('[data-box="/instances/final_n"]')).toHaveCount(0);
    const after = await sourceText(page);
    expect(after).toContain('"final_n"');
    // Its edge into `lm_head` now names the site at the composition's last index.
    expect(after).toContain('"instance": "final_n"');
  });
});

test.describe('§4.4’s other two canvas gestures', () => {
  test('Alt+drag leaves the original and drops a copy where the drag ended', async ({ page }) => {
    await open(page);
    await openModel(page, LLAMA);
    const card = page.locator('[data-box="/instances/final_n"]');
    const at = await card.boundingBox();
    await page.keyboard.down('Alt');
    await page.mouse.move(at!.x + at!.width / 2, at!.y + 4);
    await page.mouse.down();
    await page.mouse.move(at!.x + at!.width / 2 + 140, at!.y + 60, { steps: 6 });
    await page.mouse.up();
    await page.keyboard.up('Alt');

    // The original stays and the copy is the selection, so the sheet opens on what was made.
    await expect(card).toBeVisible();
    await expect(page.locator('[data-box="/instances/final_n_copy"]')).toBeVisible();
    await expect(page.locator('.insp-title')).toHaveText('final_n_copy');
    const after = await sourceText(page);
    expect(after).toContain('"final_n_copy"');
  });

  test('a band over empty ground clears the selection', async ({ page }) => {
    await open(page);
    await openModel(page, LLAMA);
    await page.locator('[data-box="/instances/embed"] .n-name').click();
    await expect(page.locator('.node.sel')).toHaveCount(1);

    const canvas = page.locator('.canvas.graph');
    const at = await canvas.boundingBox();
    await page.keyboard.down('Shift');
    await page.mouse.move(at!.x + at!.width - 6, at!.y + 6);
    await page.mouse.down();
    await page.mouse.move(at!.x + at!.width - 60, at!.y + 60, { steps: 4 });
    await page.mouse.up();
    await page.keyboard.up('Shift');
    await expect(page.locator('.node.sel')).toHaveCount(0);
  });
});

test.describe('“Connect from previous iteration…” (§4.8)', () => {
  test('writes the override and proposes the guard', async ({ page }) => {
    await open(page);
    await openModel(page, LLAMA);
    await drillIn(page, 'decoder');

    // The gesture is offered on the producing port, because a ghost column is generated from a
    // rule and is no drop target.
    await page
      .locator('[data-port="/compositions/decoder/instances/attn:output"]')
      .click({ button: 'right' });
    await page.locator('.ctxmenu button[data-entry="canvas.connect-previous:layer"]').click();
    await page.locator('[data-port="/compositions/decoder/instances/ffn_n:input"]').click();

    // The override §4.8 names, written on the producing end.
    const after = await sourceText(page);
    expect(after).toContain('"op": "subtract"');
    expect(after).toContain('"index": "layer"');

    // The guard is a **proposal**: the edge is made, and the toast offers the condition that says
    // where it may fire.
    const toast = page.locator('.toast');
    await expect(toast).toBeVisible();
    await expect(toast.locator('.undo')).toHaveText('Add guard layer >= 1');
    await toast.locator('.undo').click();

    const guarded = await sourceText(page);
    expect(guarded).toContain('"greater_or_equal"');
    // A second ghost column stands where the new carry runs from.
    await expect(page.locator('.ghostcol')).toHaveCount(2);
  });
});

/**
 * What axe finds on the page, as a list of `rule element` lines.
 *
 * The wordmark is excluded from the colour-contrast rule alone, for the reason feature 2.5
 * recorded and feature 2.9's own pass repeated: WCAG 1.4.3 exempts "text that is part of a logo
 * or brand name", and repainting it would be redrawing the logo (§4.21).
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
