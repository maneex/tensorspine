import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Locator, type Page } from '@playwright/test';

/**
 * The expression editors — feature 2.11, plan §4.13, artboard S7, in a browser.
 *
 * What a model cannot be asked and a page can: that the two views of one value are on the screen
 * together, that a gesture in either of them **writes the JSON** the document is, that the core's
 * own figures come back beside the value, that a half-typed text is a message and not a lost
 * document (Q5), and that the whole editor can be used without a pointer.
 *
 * The value is `head_dim`'s derivation in `llama3-8b` — `d div heads`, S7's own line and the
 * feature's own test — reached through the model explorer, whose selection is a place of the one
 * tree (D1). What the editor answers about a value in general is in
 * `packages/ui/test/expressions/`, over every expression the repository writes.
 */

const MODEL = 'models/llama3-8b.json';
const GUARDED = 'models/qwen3.5-4b-text.json';
const HEAD_DIM = '/quantities/head_dim';

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

/** Open the Examples workspace and one document of it. */
async function openModel(page: Page, path = MODEL): Promise<void> {
  await page.locator('.nothing .link[data-open="examples"]').click();
  await page.locator('.dlg [data-workspace="examples"]').click();
  await expect(page.locator('footer.status [data-workspace]')).toHaveText('Examples');
  await command(page, 'File', 'file.open-model');
  await page.locator(`.dlg [data-document="${path}"]`).click();
  await expect(page.locator(`.gcanvas[data-canvas="${path}"]`)).toBeVisible();
}

/** The document's own bytes, through `View ▸ JSON Source` — what a Save would write (D12). */
async function sourceText(page: Page): Promise<string> {
  await command(page, 'View', 'view.json-source');
  const pane = page.locator('.doc-json');
  await expect(pane).toBeVisible();
  const shown = await pane.innerText();
  await page.keyboard.press('Control+w');
  await expect(page.locator('.doc-json')).toHaveCount(0);
  return shown;
}

/** The Properties panel's body. */
function sheet(page: Page): Locator {
  return page.locator('.insp .panel-body');
}

/** The editor of the one expression the place holds. */
function editor(page: Page): Locator {
  return sheet(page).locator('.xed');
}

/** Select a declaration in the explorer, which is what opens its sheet (§4.5). */
async function select(page: Page, pointer: string): Promise<void> {
  await page.locator(`.tree [data-row="${pointer}"]`).click();
  await expect(sheet(page).locator('[data-place]')).toHaveText(pointer);
}

test.describe('head_dim’s derivation, in both views', () => {
  test('draws S7’s tree and its text over the same value', async ({ page }) => {
    await open(page);
    await openModel(page);
    await select(page, HEAD_DIM);

    // Where §4.11 puts it: the quantity's own sheet, under the `source` its derivation belongs to
    // ("literal with value and optional derivation"). Feature 2.11 drew it in a section of its own
    // because that sheet did not exist yet; feature 2.12 built the sheet and the row moved into it.
    await expect(sheet(page).locator('[data-chooser="/source"] [data-member="derivation"]')).toBeVisible();
    await expect(editor(page)).toHaveCount(1);

    // The tree: one row per node, with the operator select and the two quantity pickers.
    const rows = editor(page).locator('.xn');
    await expect(rows).toHaveCount(3);
    await expect(rows.nth(0).locator('.kind')).toHaveText('floor_divide');
    await expect(rows.nth(0).locator('[data-operator]')).toHaveValue('floor_divide');
    await expect(rows.nth(1).locator('[data-name]')).toHaveValue('d');
    await expect(rows.nth(2).locator('[data-name]')).toHaveValue('heads');

    // The core's own evaluator beside each of them, and beside the whole (S7: 4096, 32, 128).
    await expect(rows.nth(1).locator('.xval')).toHaveText('4096');
    await expect(rows.nth(2).locator('.xval')).toHaveText('32');
    await expect(editor(page).locator('[data-resolved]')).toHaveText('resolves to 128');

    // The text form, and what it stores.
    await expect(editor(page).locator('.xtext')).toHaveValue('d div heads');
    await expect(editor(page).locator('[data-json]')).toContainText('"floor_divide"');
  });

  test('changes the operator in the tree and the JSON is what changed', async ({ page }) => {
    await open(page);
    await openModel(page);
    await select(page, HEAD_DIM);

    await editor(page).locator('[data-operator=""]').selectOption('divide');

    // The value the document holds, read back through the core's serializer.
    expect(await sourceText(page)).toContain('"op": "divide"');
    await expect(editor(page).locator('.xtext')).toHaveValue('d / heads');
    // And the core answers the new value: `divide` is a real, which is what S7's note is about.
    await expect(editor(page).locator('[data-resolved]')).toHaveText('resolves to 128.0');

    // The edit is one named command in the log (D13) — `Set source · derivation` — and undoing it
    // is §4.4's `edit.undo`, which no feature has given a handler yet: the shell registers the
    // command (feature 2.5) and the store has the inverse patches (feature 2.1). Asserted where
    // the two meet, whenever that is.
  });

  test('picks another quantity from the select, and the pickers offer the document’s own', async ({
    page,
  }) => {
    await open(page);
    await openModel(page);
    await select(page, HEAD_DIM);

    const picker = editor(page).locator('.xn').nth(2).locator('[data-name]');
    expect(await picker.locator('option').allInnerTexts()).toEqual([
      'd',
      'ffn',
      'heads',
      'kv_heads',
      'head_dim',
      'layers',
      'vocab',
      'eps',
      'precision',
    ]);
    await picker.selectOption('kv_heads');

    await expect(editor(page).locator('.xtext')).toHaveValue('d div kv_heads');
    expect(await sourceText(page)).toContain('"quantity": "kv_heads"');
  });

  test('writes the JSON from the text form, and refuses nothing while it is half typed', async ({
    page,
  }) => {
    await open(page);
    await openModel(page);
    await select(page, HEAD_DIM);

    const field = editor(page).locator('.xtext');
    await field.fill('d div');
    // Q5: the keystroke stands, the refusal is a message, and the document is untouched.
    await expect(editor(page).locator('[data-refusal]')).toHaveText(
      'the expression stops before it says anything',
    );
    expect(await sourceText(page)).toContain('"op": "floor_divide"');

    await field.fill('(d div heads) * 2');
    await field.press('Enter');
    const written = await sourceText(page);
    expect(written).toContain('"op": "multiply"');
    expect(written).toContain('"literal": 2');
    // The tree follows the text, both being views of the one value (D1).
    await expect(editor(page).locator('.xn').first().locator('[data-operator]')).toHaveValue(
      'multiply',
    );
  });

  test('wraps a node from the row’s own actions, and the blank is on the grammar', async ({
    page,
  }) => {
    await open(page);
    await openModel(page);
    await select(page, HEAD_DIM);

    // The actions appear on the row in hand — S7's `wrap ▸` on the selected row.
    await editor(page).locator('.xn').nth(1).locator('[data-name]').focus();
    await editor(page).locator('[data-wrap="0"]').selectOption('add');
    await expect(editor(page).locator('.xtext')).toHaveValue('(d + 0) div heads');
    expect(await sourceText(page)).toContain('"op": "add"');

    // The blank is a literal row of its own, and what is typed in it is read by the same parser.
    const blank = editor(page).locator('[data-literal="0.1"]');
    await expect(blank).toHaveValue('0');
    await blank.fill('64');
    await blank.press('Enter');
    await expect(editor(page).locator('.xtext')).toHaveValue('(d + 64) div heads');

    // And the other half of the wrap: the node comes back out of it.
    await editor(page).locator('.xn').nth(1).locator('[data-operator="0"]').focus();
    await editor(page).locator('[data-unwrap="0"]').click();
    await expect(editor(page).locator('.xtext')).toHaveValue('d div heads');
    expect(await sourceText(page)).toContain('"op": "floor_divide"');
  });

  test('replaces a node with another shape, and a reference starts from a name in scope', async ({
    page,
  }) => {
    await open(page);
    await openModel(page);
    await select(page, HEAD_DIM);

    // "Replace with…" offers every production of the place — the schema's alternatives and its
    // operator enumerations — and what it writes is on the grammar (D5).
    await editor(page).locator('.xn').nth(2).locator('[data-name]').focus();
    await editor(page).locator('[data-replace="1"]').selectOption('literal');
    await expect(editor(page).locator('.xtext')).toHaveValue('d div 0');

    // A reference starts from the first name the picker offers rather than from a placeholder.
    await editor(page).locator('.xn').nth(2).locator('[data-literal]').focus();
    await editor(page).locator('[data-replace="1"]').selectOption('quantity');
    await expect(editor(page).locator('.xtext')).toHaveValue('d div d');
    expect(await sourceText(page)).toContain('"quantity": "d"');
  });
});

test.describe('the guard of a site', () => {
  test('draws the condition of qwen3.5-4b-text’s gdn and edits it', async ({ page }) => {
    await open(page);
    await openModel(page, GUARDED);

    // The site is inside the composition, so the box is opened before the card is clicked.
    const fold = page.locator('[data-fold="/compositions/decoder"]');
    if ((await fold.getAttribute('aria-expanded')) !== 'true') await fold.click();
    await page.locator('.gcanvas [data-name="/compositions/decoder/instances/gdn"]').click();
    await expect(sheet(page).locator('.insp-title')).toHaveText('gdn');

    // §4.11's guard row is an expression row of condition type, and it is editable here.
    const guard = sheet(page).locator('[data-editor="condition"] .xed');
    await expect(guard.locator('.xtext')).toHaveValue('$layer mod 4 != 3');
    await guard.locator('.xtext').fill('$layer mod 4 != 2');
    await guard.locator('.xtext').press('Enter');
    expect(await sourceText(page)).toContain('"literal": 2');
  });
});

test.describe('the editor without a pointer', () => {
  test('is reachable and announced', async ({ page }) => {
    await open(page);
    await openModel(page);
    await select(page, HEAD_DIM);

    // Every control is a real one, so Tab walks them and each says what it is for.
    // The label is the row's own — the member the schema declares the expression under — since
    // the editor now sits on that row rather than in a section listing the place's expressions.
    await expect(editor(page).locator('[data-operator=""]')).toHaveAttribute(
      'aria-label',
      'Operator of derivation',
    );
    await expect(editor(page).locator('.xtext')).toHaveAttribute(
      'aria-label',
      'Expression text of derivation',
    );

    // The keyboard's own way through: focus the operator select, change it, and the text follows.
    await editor(page).locator('[data-operator=""]').focus();
    await editor(page).locator('[data-operator=""]').selectOption('modulo');
    await expect(editor(page).locator('.xtext')).toHaveValue('d mod heads');
  });

  // Both themes, because the one contrast §4.21 refuses was the light theme's alone: the operator
  // chip's `--accent` on `--bg-tint` is 4.21:1 there, and the ink steps down the ramp for it.
  for (const theme of ['dark', 'light'] as const) {
    test(`draws no axe violation in the ${theme} theme, with a row in hand`, async ({ page }) => {
      await open(page);
      await command(page, 'View', theme === 'dark' ? 'view.theme-dark' : 'view.theme-light');
      await openModel(page);
      await select(page, HEAD_DIM);
      // With the row selected, so the actions S7 draws on it are on the page too.
      await editor(page).locator('.xn').nth(1).locator('[data-name]').focus();
      await expect(editor(page).locator('[data-wrap="0"]')).toBeVisible();

      const audit = await new AxeBuilder({ page }).include('.insp').analyze();
      expect(audit.violations.map((one) => `${one.id}: ${one.nodes.length}`)).toEqual([]);
    });
  }
});
