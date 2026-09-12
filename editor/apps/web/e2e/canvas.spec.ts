import { readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Locator, type Page } from '@playwright/test';

/**
 * The folded canvas — feature 2.9, plan §4.7, artboards S1, S2 and S3.
 *
 * What the browser layer is for here is everything a model cannot be asked about: that the boxes
 * are drawn where a layout put them, that a drag from one handle to another makes the edge, that
 * the drop is never refused, that a manual move writes the sidecar and an automatic layout does
 * not, that a delete asks before it cascades, and that a canvas can be used without a pointer.
 *
 * The document is `llama3-8b` throughout, which is what the feature's own block names.
 */

const repository = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', '..');
const MODEL = 'models/llama3-8b.json';
const modelText = readFileSync(join(repository, 'data/models/llama3-8b.json'), 'utf8');

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
async function openModel(page: Page, path = MODEL): Promise<void> {
  await page.locator('.nothing .link[data-open="examples"]').click();
  await page.locator('.dlg [data-workspace="examples"]').click();
  await expect(page.locator('footer.status [data-workspace]')).toHaveText('Examples');
  await command(page, 'File', 'file.open-model');
  await page.locator(`.dlg [data-document="${path}"]`).click();
  await expect(page.locator(`.gcanvas[data-canvas="${path}"]`)).toBeVisible();
  // The cards carry the core's facts, which arrive from the worker: the ports are what say so.
  await expect(page.locator('[data-port="/instances/embed:tokens"]')).toBeVisible();
}

/** One box of the canvas, by the place it stands for. */
function box(page: Page, pointer: string): Locator {
  return page.locator(`[data-box="${pointer}"]`);
}

/** One port handle, by the box and the port it belongs to. */
function port(page: Page, pointer: string, name: string): Locator {
  return page.locator(`[data-port="${pointer}:${name}"]`);
}

/** The document's own bytes, through `View ▸ JSON Source` (§4.2's own tab). */
async function sourceText(page: Page): Promise<string> {
  await command(page, 'View', 'view.json-source');
  const pane = page.locator('.doc-json');
  await expect(pane).toBeVisible();
  const shown = await pane.innerText();
  await page.keyboard.press('Control+w');
  await expect(page.locator('.doc-json')).toHaveCount(0);
  return shown;
}

/** Drag from one element's middle to another's, as a pointer does it. */
async function drag(page: Page, from: Locator, to: Locator): Promise<void> {
  const a = await from.boundingBox();
  const b = await to.boundingBox();
  expect(a).not.toBeNull();
  expect(b).not.toBeNull();
  await page.mouse.move(a!.x + a!.width / 2, a!.y + a!.height / 2);
  await page.mouse.down();
  await page.mouse.move(a!.x + a!.width / 2 + 8, a!.y + a!.height / 2 + 8, { steps: 4 });
  await page.mouse.move(b!.x + b!.width / 2, b!.y + b!.height / 2, { steps: 8 });
  await page.mouse.up();
}

/** Drop a primitive on the canvas, as the Library palette of §4.6 will (feature 3.1). */
async function dropPrimitive(page: Page, primitive: string, version: string): Promise<void> {
  await page.locator('.canvas.graph').evaluate(
    (element, held: string) => {
      const transfer = new DataTransfer();
      transfer.setData('application/x-tensorspine-primitive', held);
      const at = element.getBoundingClientRect();
      for (const kind of ['dragover', 'drop']) {
        element.dispatchEvent(
          new DragEvent(kind, {
            bubbles: true,
            cancelable: true,
            dataTransfer: transfer,
            clientX: at.left + 60,
            clientY: at.top + 60,
          }),
        );
      }
    },
    JSON.stringify({ primitive, version }),
  );
}

test.describe('the folded document of S1', () => {
  test('draws three cards, a composition with its boundary handles, two terminals and the edges', async ({
    page,
  }) => {
    await open(page);
    await openModel(page);

    // Three root instances, as cards.
    for (const name of ['embed', 'final_n', 'lm_head']) {
      await expect(box(page, `/instances/${name}`)).toBeVisible();
      await expect(box(page, `/instances/${name}`).locator('.n-name')).toHaveText(name);
    }
    await expect(page.locator('.gcanvas .node')).toHaveCount(3);

    // One composition, collapsed, with the range, the count, the summary and the handles S3 draws.
    const decoder = box(page, '/compositions/decoder');
    await expect(decoder.locator('.g-range')).toHaveText('layer ∈ [0, 32) by 1');
    await expect(decoder.locator('.g-count')).toHaveText('×32');
    await expect(decoder.locator('.g-sites')).toHaveText(
      'instances 6 · values 8 · parameters 9 · states 1',
    );
    await expect(decoder.locator('.bh')).toHaveText([
      'attn_n[layer=0].input',
      'attn_r[layer=0].a',
      'ffn_r[layer=31].output',
    ]);

    // Two interface terminals, with what the document says about each beside its name.
    await expect(box(page, '/interfaces/inputs/tokens').locator('.tkind')).toHaveText('token');
    await expect(box(page, '/interfaces/outputs/logits').locator('.tkind')).toHaveText('generative');

    // Every edge of the document, labelled with the rule name.
    await expect(page.locator('.gcanvas .elbl')).toHaveText([
      'decoder.entry',
      'decoder.entry.a',
      'final_n.in',
      'lm_head.in',
      'tokens',
      'logits',
    ]);

    // The card of S2: the primitive, the structural summary, the handles, the chips, the figures.
    const embed = box(page, '/instances/embed');
    await expect(embed.locator('.n-prim')).toHaveText('embed@1.0.0');
    await expect(embed.locator('.n-args')).toHaveText('width=d · vocabulary=vocab');
    await expect(embed.locator('.n-ports .p.in')).toHaveText('tokens');
    await expect(embed.locator('.n-ports .p.out')).toHaveText('output');
    await expect(embed.locator('.slot')).toHaveText('◆ weight');
    await expect(embed.locator('.slot')).toHaveClass(/located/);
    await expect(embed.locator('.n-der')).toHaveText('1002.0 MiB params');

    // Nothing is wrong with it, so no dot and no red handle.
    await expect(page.locator('.n-dot')).toHaveCount(0);
    await expect(page.locator('.n-ports .p.unfed')).toHaveCount(0);
  });

  test('is laid out top to bottom, with the input above and the output below (§4.7)', async ({
    page,
  }) => {
    await open(page);
    await openModel(page);
    const tokens = await box(page, '/interfaces/inputs/tokens').boundingBox();
    const decoder = await box(page, '/compositions/decoder').boundingBox();
    const logits = await box(page, '/interfaces/outputs/logits').boundingBox();
    expect(tokens!.y).toBeLessThan(decoder!.y);
    expect(decoder!.y).toBeLessThan(logits!.y);
  });

  test('opens a composition in place and shows its sites (S3)', async ({ page }) => {
    await open(page);
    await openModel(page);
    await page.locator('[data-fold="/compositions/decoder"]').click();
    for (const site of ['attn_n', 'attn', 'attn_r', 'ffn_n', 'ffn', 'ffn_r']) {
      await expect(box(page, `/compositions/decoder/instances/${site}`)).toBeVisible();
    }
    await expect(box(page, '/compositions/decoder').locator('.bh')).toHaveCount(0);
  });
});

test.describe('connecting two handles', () => {
  test('replaces the edge that fed an input, with a toast that undoes it (Q5, V7)', async ({
    page,
  }) => {
    await open(page);
    await openModel(page);
    const before = await sourceText(page);
    expect(before).toBe(modelText);

    await drag(page, port(page, '/instances/final_n', 'output'), port(page, '/instances/lm_head', 'input'));

    // The older edge went, and the toast names it with an Undo beside it.
    const toast = page.locator('.toast');
    await expect(toast).toBeVisible();
    await expect(toast).toContainText('/bindings/values/lm_head.in');
    await expect(toast.locator('.undo')).toHaveText('Undo');

    const after = await sourceText(page);
    expect(after).not.toContain('"lm_head.in"');
    expect(after).toContain('"lm_head.input"');

    await page.locator('.toast .undo').click();
    expect(await sourceText(page)).toBe(modelText);
  });

  test('is never refused, whatever the core says about it (Q5)', async ({ page }) => {
    await open(page);
    await openModel(page);
    // A generative output into an embedding's token input: nonsense, and the gesture makes it.
    await drag(page, port(page, '/instances/lm_head', 'logits'), port(page, '/instances/embed', 'tokens'));
    const after = await sourceText(page);
    expect(after).toContain('"embed.tokens"');
    // And the refusal is where §4.17 puts it, not in the way of the gesture.
    await page.locator('.bar [data-pill="validation"]').click();
    await expect(page.locator('.panel-body .prow')).not.toHaveCount(0, { timeout: 10_000 });
  });

  test('shows the core’s verdict while the drag is in flight, and never a veto', async ({
    page,
  }) => {
    await open(page);
    await openModel(page);
    const from = await port(page, '/instances/lm_head', 'logits').boundingBox();
    const to = await port(page, '/instances/embed', 'tokens').boundingBox();
    await page.mouse.move(from!.x + from!.width / 2, from!.y + from!.height / 2);
    await page.mouse.down();
    await page.mouse.move(to!.x + to!.width / 2, to!.y + to!.height / 2, { steps: 8 });
    // The ghost of §4.7, with the core's own words in it — and the target handle lit all the same.
    const ghost = page.locator('.dghost');
    await expect(ghost).toBeVisible();
    await expect(ghost.locator('[data-verdict]')).toBeVisible({ timeout: 10_000 });
    await expect(ghost.locator('[data-verdict="bad"]')).toContainText('shapes do not unify');
    await expect(port(page, '/instances/embed', 'tokens')).toHaveClass(/lit/);
    await page.mouse.up();
    expect(await sourceText(page)).toContain('"embed.tokens"');
  });

  test('is made by two clicks as well as by a drag, which is the keyboard’s form of it', async ({
    page,
  }) => {
    await open(page);
    await openModel(page);
    await port(page, '/instances/final_n', 'output').click();
    await expect(page.locator('.dghost')).toBeVisible();
    await port(page, '/instances/lm_head', 'input').click();
    await expect(page.locator('.dghost')).toHaveCount(0);
    expect(await sourceText(page)).toContain('"lm_head.input"');
  });
});

test.describe('dropping a primitive from the palette', () => {
  test('adds an instance with the proposed name and family, and opens its sheet', async ({
    page,
  }) => {
    await open(page);
    await openModel(page);
    await dropPrimitive(page, 'norm.rms', '1.0.0');

    // §9 Q4's proposals: the primitive name's last segment for the name, its first for the family.
    const added = box(page, '/instances/rms');
    await expect(added).toBeVisible();
    await expect(added.locator('.n-name')).toHaveText('rms');
    await expect(added.locator('.fchip')).toHaveText('norm');

    // D5's skeleton, on the grammar and refused by V2 — which is what the Problems panel says.
    const after = await sourceText(page);
    expect(after).toContain('"rms": {');
    expect(after).toContain('"arguments": {}');

    // "the sheet open": the Properties region shows the new instance, which is what is selected.
    // A site's sheet is §4.11's own (feature 2.10) — its title, the primitive it pins and its
    // name row — where before this feature it was the place-and-name row of every selection.
    await expect(page.locator('.insp')).toBeVisible();
    await expect(page.locator('.insp .insp-title')).toHaveText('rms');
    await expect(page.locator('.insp .prim-chip')).toHaveText('norm.rms@1.0.0');
    await expect(page.locator('.insp input[data-name-field]')).toHaveValue('rms');
    await expect(added).toHaveClass(/sel/);

    // It was dropped where it was dropped, which is a manual place (D6) — so the document is
    // dirty in its layout as well as in its tree.
    await expect(page.locator('nav.tabs .tab .dot')).toHaveCount(1);
  });
});

test.describe('deleting a box', () => {
  test('asks first, lists the cascade, and writes the document without it', async ({ page }) => {
    await open(page);
    await openModel(page);
    await box(page, '/instances/final_n').click();
    await command(page, 'Edit', 'edit.delete');

    const dialog = page.locator('.dlg');
    await expect(dialog.locator('.dlg-head h2')).toHaveText('Delete instance final_n?');
    // Everything that names it, in the order the cascade removes them.
    await expect(dialog.locator('[data-removed]')).toHaveText([
      '/bindings/values/lm_head.in',
      '/bindings/values/final_n.in',
      '/bindings/parameters/final_n.weight',
      '/instances/final_n',
    ]);
    // Nothing is written until it is confirmed: the card is still there behind the dialog.
    await expect(box(page, '/instances/final_n')).toHaveCount(1);

    await page.locator('.dlg [data-confirm="remove"]').click();
    const after = await sourceText(page);
    expect(after).not.toContain('"final_n"');
    await expect(box(page, '/instances/final_n')).toHaveCount(0);
  });
});

test.describe('the layout sidecar (D6)', () => {
  test('is written by a manual move and not by an automatic layout', async ({ page }) => {
    await open(page);
    await openModel(page);
    // Nothing has been arranged, so nothing is dirty.
    await expect(page.locator('nav.tabs .tab .dot')).toHaveCount(0);

    await command(page, 'View', 'view.auto-layout');
    await expect(page.locator('nav.tabs .tab .dot')).toHaveCount(0);

    // A manual move: the card is dragged by its own ground — a control inside it belongs to the
    // control, which is what the structural summary is not.
    const card = box(page, '/instances/embed');
    const at = await card.boundingBox();
    const ground = await card.locator('.n-args').boundingBox();
    await page.mouse.move(ground!.x + ground!.width / 2, ground!.y + ground!.height / 2);
    await page.mouse.down();
    await page.mouse.move(ground!.x + ground!.width / 2 + 120, ground!.y + 40, { steps: 6 });
    await page.mouse.up();
    await expect(page.locator('nav.tabs .tab .dot')).toHaveCount(1);
    const moved = await card.boundingBox();
    expect(Math.round(moved!.x)).not.toBe(Math.round(at!.x));

    // The document itself is untouched: a position is not a member of it (D6).
    expect(await sourceText(page)).toBe(modelText);

    // Reset Layout drops the override, and the dirt with it.
    await command(page, 'View', 'view.reset-layout');
    await expect(page.locator('nav.tabs .tab .dot')).toHaveCount(0);
  });
});

test.describe('the handles of §4.7', () => {
  test('is hollow red exactly when Problems has the V7 row for it', async ({ page }) => {
    await open(page);
    await openModel(page);
    await expect(page.locator('.n-ports .p.unfed')).toHaveCount(0);
    await expect(page.locator('.n-dot')).toHaveCount(0);

    // Take the edge that feeds `lm_head.input` away, which is what V7 is about.
    await page.locator('.elbl[data-wire="/bindings/values/lm_head.in"]').click();
    await command(page, 'Edit', 'edit.delete');
    await page.locator('.dlg [data-confirm="remove"]').click();

    const handle = port(page, '/instances/lm_head', 'input');
    await expect(handle).toHaveClass(/unfed/, { timeout: 10_000 });
    await expect(box(page, '/instances/lm_head').locator('.n-dot')).toBeVisible();

    // And the row is there, in the tools' own words.
    await page.locator('.bar [data-pill="validation"]').click();
    const rows = page.locator('.panel-body .prow');
    await expect(rows.filter({ hasText: 'input port with no producer' })).toHaveCount(1, {
      timeout: 10_000,
    });
    await expect(rows.filter({ hasText: 'lm_head.input' }).first()).toBeVisible();
  });
});

test.describe('the context menu of §4.7', () => {
  test('offers the entries of the section, and each acts on the box it was opened on', async ({
    page,
  }) => {
    await open(page);
    await openModel(page);
    await box(page, '/instances/final_n').click({ button: 'right' });
    const menu = page.locator('.ctxmenu');
    await expect(menu).toBeVisible();
    await expect(menu.locator('[data-entry="edit.rename"]')).toHaveText('Rename');
    await expect(menu.locator('[data-entry="edit.duplicate"]')).toHaveText('Duplicate');
    await expect(menu.locator('[data-entry="view.json-source"]')).toHaveText('Show in JSON');
    await menu.locator('[data-entry="edit.duplicate"]').click();
    await expect(box(page, '/instances/final_n_copy')).toBeVisible();
  });
});

test.describe('the canvas without a pointer', () => {
  test('selects with an arrow, nudges what is selected, and renames it (§4.4)', async ({ page }) => {
    await open(page);
    await openModel(page);
    await page.locator('.canvas.graph').focus();

    // An arrow with nothing selected selects the first box, which is the keyboard's way in.
    await page.keyboard.press('ArrowDown');
    await expect(page.locator('[data-box].sel')).toHaveCount(1);
    const selected = await page.locator('[data-box].sel').getAttribute('data-box');
    expect(selected).not.toBeNull();
    const before = await page.locator(`[data-box="${selected!}"]`).boundingBox();

    // §4.4's "arrows nudge": the keyboard's form of a manual move, and the same override (D6).
    await page.keyboard.press('ArrowRight');
    await page.keyboard.press('ArrowRight');
    const after = await page.locator(`[data-box="${selected!}"]`).boundingBox();
    expect(Math.round(after!.x)).toBeGreaterThan(Math.round(before!.x));
    await expect(page.locator('nav.tabs .tab .dot')).toHaveCount(1);

    // §4.7: "Click the name … edits that element in place", and the name is a real button that
    // Enter activates — which is what makes the rename reachable without a pointer at all.
    await page.locator(`.gcanvas [data-name="${selected!}"]`).focus();
    await page.keyboard.press('Enter');
    const field = page.locator('.n-rename');
    await expect(field).toBeVisible();
    await field.fill('renamed_here');
    await field.press('Enter');
    expect(await sourceText(page)).toContain('"renamed_here"');
  });

  test('clears the selection with Escape (§4.4)', async ({ page }) => {
    await open(page);
    await openModel(page);
    await box(page, '/instances/embed').click();
    await expect(page.locator('[data-box].sel')).toHaveCount(1);
    await page.locator('.canvas.graph').focus();
    await page.keyboard.press('Escape');
    await expect(page.locator('[data-box].sel')).toHaveCount(0);
  });
});

test.describe('the accessibility of the canvas (§4.21)', () => {
  for (const theme of ['dark', 'light'] as const) {
    test(`finds no axe violation in the ${theme} theme, with a drag in flight`, async ({ page }) => {
      await open(page);
      await command(page, 'View', theme === 'dark' ? 'view.theme-dark' : 'view.theme-light');
      await openModel(page);
      await page.locator('[data-fold="/compositions/decoder"]').click();
      await expect(box(page, '/compositions/decoder/instances/attn')).toBeVisible();
      expect(await audit(page), 'the canvas').toEqual([]);

      await box(page, '/instances/final_n').click({ button: 'right' });
      await expect(page.locator('.ctxmenu')).toBeVisible();
      expect(await audit(page), 'the context menu').toEqual([]);
    });
  }
});

/**
 * What axe finds on the page, as a list of `rule element` lines.
 *
 * The wordmark is excluded from the colour-contrast rule alone, for the reason feature 2.5
 * recorded: WCAG 1.4.3 exempts "text that is part of a logo or brand name", and repainting it
 * would be redrawing the logo, which §4.21 forbids in the same breath as the contrast.
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
