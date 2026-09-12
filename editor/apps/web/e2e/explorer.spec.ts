import { readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Locator, type Page } from '@playwright/test';

/**
 * Feature 2.7 — the Model explorer (§4.5), in a browser.
 *
 * The three claims the feature's block names, each of which needs a real page to be worth
 * anything:
 *
 *   - **the tree for `llama3-8b` matches S1's structure and counts** — drawn, with the rows the
 *     board draws, at the depths it draws them, opened where it opens them;
 *   - **selecting `attn` selects the canvas node and opens its sheet** — the canvas is feature
 *     2.9's, and what a selection *is* is a place of the one tree (D1), so what is asked here is
 *     that the place reaches the two readers that exist: the tree marks it, and Properties opens
 *     on it;
 *   - **rename by click writes the JSON** — the tab's body is the document through the core's
 *     serializer (D12), so the bytes a Save would write are on the page and the rename is read
 *     back out of them.
 *
 * Everything that is a question about *what the outline says* rather than about a browser is in
 * `packages/ui/test/explorer/outline.test.ts`, over the whole corpus.
 */

const repository = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', '..');
const MODEL = 'models/llama3-8b.json';
const modelText = readFileSync(join(repository, 'data/models/llama3-8b.json'), 'utf8');

/** Open the application and wait for it to have decided what to open (D11's own startup). */
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

/** Open the Examples workspace and one document of it, the way the empty state offers them. */
async function openModel(page: Page, path = MODEL): Promise<void> {
  await page.locator('.nothing .link[data-open="examples"]').click();
  await page.locator('.dlg [data-workspace="examples"]').click();
  await expect(page.locator('footer.status [data-workspace]')).toHaveText('Examples');
  await command(page, 'File', 'file.open-model');
  await page.locator(`.dlg [data-document="${path}"]`).click();
  await expect(page.locator(`.gcanvas[data-canvas="${path}"]`)).toBeVisible();
}

/**
 * The document's own bytes, through `View ▸ JSON Source` (§4.4, §4.2's own tab).
 *
 * Feature 2.9 gave a model's tab its canvas (§4.7's "the default editor of a model") and the
 * source pane the tab of its own §4.2 puts it in; this is how a suite reads what a Save would
 * write, which is the same pane through the same serializer (D12).
 */
async function sourceText(page: Page): Promise<string> {
  await command(page, 'View', 'view.json-source');
  const pane = page.locator('.doc-json');
  await expect(pane).toBeVisible();
  const shown = await pane.innerText();
  // The source is a *view* of the document, in a tab of its own (§4.2): it is closed again so
  // that what a suite counts afterwards is the documents it opened and not the readings of them.
  // `Ctrl+W` rather than the tab's ×, because with fifteen documents open the strip is wider
  // than the editor area and the × of the last one is behind the Properties region.
  await page.keyboard.press('Control+w');
  await expect(page.locator('.doc-json')).toHaveCount(0);
  return shown;
}


/** One row of the tree, by the place it stands for. */
function row(page: Page, pointer: string): Locator {
  return page.locator(`.tree [data-row="${pointer}"]`);
}

/** The tree as lines: the indent, the name, the count and the tail, in the order they are drawn. */
async function lines(page: Page): Promise<string[]> {
  return page.locator('.tree .row').evaluateAll((rows) =>
    rows.map((one) => {
      const level = Number(one.getAttribute('aria-level') ?? '1') - 1;
      const expanded = one.getAttribute('aria-expanded');
      const chevron = expanded === null ? ' ' : expanded === 'true' ? '▾' : '▸';
      const part = (selector: string): string =>
        (one.querySelector(selector)?.textContent ?? '').trim();
      return [
        '  '.repeat(level) + chevron,
        part('.n-name'),
        part('.n') === '' ? '' : `(${part('.n')})`,
        part('.fig'),
        part('.drv'),
        part('.tail') === '' ? '' : `· ${part('.tail')}`,
        part('.held'),
      ]
        .filter((each) => each !== '')
        .join(' ');
    }),
  );
}

test.describe('the outline of §4.5', () => {
  test('draws S1’s rows, counts and figures, and opens what S1 opens', async ({ page }) => {
    await open(page);
    await openModel(page);

    expect(await lines(page)).toEqual([
      '▾ llama3-8b · tensorspine/2.0',
      '  ▸ Primitive libraries (1) · ../primitive-library/',
      '  ▾ Quantities (9)',
      '      d 4096',
      '      ffn 14336',
      '      heads 32',
      '      kv_heads 8',
      '      head_dim 128 computed',
      '      layers 32',
      '      vocab 128256',
      '      eps 1e-05',
      '      precision bf16',
      '  ▸ Constants (0)',
      '  ▾ Instances (3)',
      '      embed · embed',
      '      final_n · norm.rms',
      '      lm_head · lm_head',
      '  ▾ Compositions (1)',
      '    ▾ decoder',
      '        indices 1 · instances 6 · values 8 · parameters 9 · states 1',
      '        layer',
      '        attn_n · norm.rms',
      '        attn · attention.dense',
      '        attn_r · residual.add',
      '        ffn_n · norm.rms',
      '        ffn · ffn.gated',
      '        ffn_r · residual.add',
      '      ▸ Bindings · values 8 · parameters 9 · states 1',
      '  ▸ Bindings · values 4 · parameters 3 · constants 0 · states 0',
      '  ▸ Interfaces · inputs 1 · outputs 1',
    ]);
  });

  test('opens and closes a row, and filters the document to what a name matches', async ({
    page,
  }) => {
    await open(page);
    await openModel(page);

    // The chevron is part of the row a pointer can hit; the row itself answers the keyboard.
    await row(page, '/quantities').locator('.chev').click();
    await expect(row(page, '/quantities')).toHaveAttribute('aria-expanded', 'false');
    await expect(row(page, '/quantities/d')).toHaveCount(0);
    await expect(row(page, '/quantities').locator('.tail')).toHaveText(
      'd 4096 · ffn 14336 · heads 32 · …',
    );
    await row(page, '/quantities').locator('.chev').click();
    await expect(row(page, '/quantities/d')).toBeVisible();

    // The filter box of §4.5: what matches, and everything above it.
    await page.locator('.side .search input').fill('head_dim');
    await expect(page.locator('.tree .row')).toHaveCount(3);
    await page.locator('.side .search input').fill('attn.k');
    // A name inside the composition's own bindings, which the default leaves closed.
    await expect(row(page, '/compositions/decoder/bindings/parameters/attn.k')).toBeVisible();
    await page.locator('.side .search input').fill('');
    await expect(page.locator('.tree .row').first()).toBeVisible();
  });
});

test.describe('selecting an item', () => {
  test('marks the row and opens the sheet of §4.11 on it', async ({ page }) => {
    await open(page);
    await openModel(page);

    const attn = row(page, '/compositions/decoder/instances/attn');
    await attn.click();
    await expect(attn).toHaveClass(/\bsel\b/);
    await expect(attn).toHaveAttribute('aria-selected', 'true');

    // Properties is the selection's (§4.2, §4.11). A **site** gets §4.11's own sheet from feature
    // 2.10 — its name, the composition it sits in and the primitive it pins; anything else still
    // gets the row every sheet starts with, the name and the place it is written at.
    const sheet = page.locator('.insp .panel-body');
    await expect(sheet.locator('.insp-title')).toHaveText('attn');
    await expect(sheet.locator('.insp-kind')).toContainText('decoder');
    await expect(sheet.locator('.prim-chip')).toHaveText('attention.dense@1.0.0');
    await expect(sheet.locator('input[data-name-field]')).toHaveValue('attn');

    // Another row, and the sheet follows it.
    await row(page, '/quantities/head_dim').click();
    await expect(sheet.locator('input[data-name-field]')).toHaveValue('head_dim');
    // What the sheet says the selection *is* — the word the map it lives in declares. It moved
    // from the Identity heading's count to the kind line at feature 2.12, which is where the
    // site's own sheet has always written it (`.insp-kind`).
    await expect(sheet.locator('.insp-kind')).toHaveText('quantity');
    await expect(sheet.locator('[data-place]')).toHaveText('/quantities/head_dim');
  });

  test('moves and opens with the arrow keys, one row of the tree taking the tab stop', async ({
    page,
  }) => {
    await open(page);
    await openModel(page);

    await row(page, '/quantities/d').click();
    await page.keyboard.press('ArrowDown');
    await page.keyboard.press('Enter');
    await expect(row(page, '/quantities/ffn')).toHaveAttribute('aria-selected', 'true');
    // Left closes the group the row is in; right opens it again.
    await row(page, '/compositions').click();
    await page.keyboard.press('ArrowLeft');
    await expect(row(page, '/compositions')).toHaveAttribute('aria-expanded', 'false');
    await page.keyboard.press('ArrowRight');
    await expect(row(page, '/compositions')).toHaveAttribute('aria-expanded', 'true');
  });
});

test.describe('editing from the tree', () => {
  test('renames by clicking the name, and the JSON the tab holds is what was written', async ({
    page,
  }) => {
    await open(page);
    await openModel(page);

    const quantity = row(page, '/quantities/d');
    await quantity.click();
    // The first click selected; the second, on the name of the selected row, opens the editor.
    await quantity.locator('.n-name').click();
    const field = page.locator('.tree input[data-rename]');
    await expect(field).toBeVisible();
    await field.fill('width');
    await field.press('Enter');

    // The document, through the core's serializer: the declaration and every argument that named
    // it, and nothing else — which is the file's own bytes with those eleven places rewritten.
    const shown = await sourceText(page);
    expect(shown).toBe(
      modelText
        .replace('"d": {\n      "type"', '"width": {\n      "type"')
        .replaceAll('"quantity": "d"', '"quantity": "width"'),
    );
    // The tree and the sheet followed the name.
    await expect(row(page, '/quantities/width')).toHaveAttribute('aria-selected', 'true');
    await expect(page.locator('.insp input[data-name-field]')).toHaveValue('width');
    // And the tab is dirty, the document having been edited (§4.3's ●).
    await expect(page.locator('nav.tabs .tab .dot')).toHaveCount(1);
  });

  test('renames from the sheet’s own row, which is where §4.4 puts every value', async ({
    page,
  }) => {
    await open(page);
    await openModel(page);

    await row(page, '/instances/embed').click();
    const field = page.locator('.insp input[data-name-field]');
    await field.fill('tokens_in');
    await field.press('Enter');
    const shown = await sourceText(page);
    expect(shown).toContain('"tokens_in": {');
    expect(shown).toContain('"instance": "tokens_in"');
    expect(shown).not.toContain('"instance": "embed"');
  });

  test('deletes with the confirmation §4.4 asks for, listing what goes with it', async ({
    page,
  }) => {
    await open(page);
    await openModel(page);

    await row(page, '/compositions/decoder/instances/attn').click();
    await page.keyboard.press('Delete');

    // The cascade, listed before anything is written (plan §3, §4.7).
    const dialog = page.locator('.dlg');
    await expect(dialog.locator('.dlg-head h2')).toHaveText('Delete site attn?');
    await expect(dialog.locator('[data-removed]')).toHaveCount(8);
    await dialog.locator('[data-confirm="remove"]').click();
    const shown = await sourceText(page);
    expect(shown).not.toContain('attention.dense');
    expect(shown).not.toContain('"site": "attn"');
    await expect(row(page, '/compositions/decoder/instances/attn')).toHaveCount(0);
    // Nothing is selected any more: what was selected went — and §4.11's own last row says what
    // the sheet then shows ("Nothing selected | the Document sheet"), which feature 2.12 built.
    await expect(page.locator('.insp [data-place]')).toHaveText('/');
    await expect(page.locator('.insp .insp-title')).toHaveText('llama3-8b');
  });

  test('answers §4.4’s Edit menu on the selection, and says so in the Log when there is none', async ({
    page,
  }) => {
    await open(page);
    await openModel(page);

    // With nothing selected the command does nothing and writes the line that says why — feature
    // 2.5's rule for a command that cannot act, kept.
    await command(page, 'Edit', 'edit.rename');
    await page.locator('.panel-tabs .ptab:text-is("Log")').click();
    await expect(page.locator('.panel-body .logline').last()).toContainText(
      'Rename: nothing is selected',
    );

    // With a row selected it puts the caret in the sheet's own name row, which is where §4.4 says
    // a value is edited.
    await row(page, '/instances/final_n').click();
    await command(page, 'Edit', 'edit.rename');
    await expect(page.locator('.insp input[data-name-field]')).toBeFocused();
  });

  test('carries the red dot §4.5 asks for, on the row the core’s own pointer names', async ({
    page,
  }) => {
    await open(page);
    await openModel(page);
    await expect(page.locator('.bar [data-pill="validation"]')).toHaveText('no problems', {
      timeout: 60_000,
    });
    await expect(page.locator('.tree .dotbad')).toHaveCount(0);

    // Deleting the instance the public input feeds leaves the endpoint behind — the grammar
    // requires it — and the core refuses the document: `[V1] input tokens: instance does not
    // exist`, at `/interfaces/inputs/tokens`. The dot follows that pointer and nothing else.
    await row(page, '/instances/embed').click();
    await page.keyboard.press('Delete');
    await page.locator('.dlg [data-confirm="remove"]').click();
    await expect(page.locator('.bar [data-pill="validation"]')).not.toHaveText('no problems', {
      timeout: 60_000,
    });
    await expect(row(page, '/interfaces').locator('.dotbad')).toBeVisible();
    await expect(row(page, '').locator('.dotbad')).toBeVisible();
    await expect(row(page, '/quantities').locator('.dotbad')).toHaveCount(0);
  });

  test('carries the declaration on a drag, for the argument row that will take it (§4.5)', async ({
    page,
  }) => {
    await open(page);
    await openModel(page);

    // The drop target is the argument sheet's row (feature 2.10); what this feature owes it is
    // the transfer, so the transfer is what is read back here.
    const held = await page.evaluate(() => {
      const source = document.querySelector('[data-row="/quantities/heads"]');
      if (source === null) return null;
      const transfer = new DataTransfer();
      source.dispatchEvent(new DragEvent('dragstart', { dataTransfer: transfer, bubbles: true }));
      return transfer.getData('application/x-tensorspine-declaration');
    });
    expect(held === null ? null : (JSON.parse(held) as unknown)).toEqual({
      declares: 'quantity',
      name: 'heads',
      pointer: '/quantities/heads',
    });
  });
});

test.describe('the accessibility audit', () => {
  /**
   * The page, checked by axe — feature 2.5's own pass, with its one exemption (the wordmark is a
   * logotype, and WCAG 1.4.3 exempts one from the contrast rule). What is added here is what this
   * feature draws: the tree, a row being renamed, the sheet of the selection, and the
   * confirmation of a delete.
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

  for (const theme of ['dark', 'light'] as const) {
    test(`finds none on the tree, the sheet and a confirmation in the ${theme} theme`, async ({
      page,
    }) => {
      await open(page);
      await page.locator('nav.menu > div > button:text-is("View")').click();
      await page.locator(`.menu-pop button[data-command="view.theme-${theme}"]`).click();
      await openModel(page);

      await row(page, '/compositions/decoder/instances/attn').click();
      expect(await audit(page), 'the tree and the sheet').toEqual([]);

      await row(page, '/quantities/d').click();
      await row(page, '/quantities/d').locator('.n-name').click();
      await expect(page.locator('.tree input[data-rename]')).toBeVisible();
      expect(await audit(page), 'a row being renamed').toEqual([]);
      await page.locator('.tree input[data-rename]').press('Escape');

      await row(page, '/instances/lm_head').click();
      await page.keyboard.press('Delete');
      await expect(page.locator('.dlg')).toBeVisible();
      expect(await audit(page), 'the confirmation of a delete').toEqual([]);
    });
  }
});
