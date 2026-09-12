import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Locator, type Page } from '@playwright/test';

/**
 * The instance sheet and the argument sheet — feature 2.10, plan §4.11, §4.12, artboard S6.
 *
 * What the browser layer is for here is everything a model cannot be asked about: that the rows
 * S6 draws are on the screen under those names and in those states, that typing in one writes the
 * document and that the *core's* answer comes back onto the row that was typed in, that a
 * structural change puts a notice and its one-click repair in the Problems panel, and that the
 * whole sheet can be used and read without a pointer.
 *
 * The selection is `decoder/attn` of `llama3-8b`, which the feature's own block names.
 */

const MODEL = 'models/llama3-8b.json';
const ATTN = '/compositions/decoder/instances/attn';
const FINAL = '/instances/final_n';

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

/** Open the Examples workspace and `llama3-8b`, and wait for the core's facts to arrive. */
async function openModel(page: Page): Promise<void> {
  await page.locator('.nothing .link[data-open="examples"]').click();
  await page.locator('.dlg [data-workspace="examples"]').click();
  await expect(page.locator('footer.status [data-workspace]')).toHaveText('Examples');
  await command(page, 'File', 'file.open-model');
  await page.locator(`.dlg [data-document="${MODEL}"]`).click();
  await expect(page.locator(`.gcanvas[data-canvas="${MODEL}"]`)).toBeVisible();
  await expect(page.locator('[data-port="/instances/embed:tokens"]')).toBeVisible();
}

/** Select a site by clicking its name on the card — §4.4's "or by clicking the element itself". */
async function select(page: Page, pointer: string): Promise<void> {
  if (pointer.startsWith('/compositions/')) {
    const group = pointer.slice(0, pointer.indexOf('/instances/'));
    const fold = page.locator(`[data-fold="${group}"]`);
    if ((await fold.getAttribute('aria-expanded')) !== 'true') await fold.click();
  }
  // The card's own name button: the explorer's tree rows carry the same attribute, and §4.4's
  // "or by clicking the element itself on the canvas" is the gesture under test here.
  await page.locator(`.gcanvas [data-name="${pointer}"]`).click();
  await expect(sheet(page).locator('.insp-title')).toBeVisible();
}

/** The Properties panel's body. */
function sheet(page: Page): Locator {
  return page.locator('.insp .panel-body');
}

/** One argument row of the sheet. */
function row(page: Page, path: string): Locator {
  return sheet(page).locator(`[data-argument="${path}"]`);
}

/** The Problems panel's rows. */
function problems(page: Page): Locator {
  return page.locator('.panel .panel-body');
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

test.describe('the sheet of decoder/attn', () => {
  test('draws the rows of S6, by name and by state', async ({ page }) => {
    await open(page);
    await openModel(page);
    await select(page, ATTN);

    // Identity: the name, the primitive it pins and the version, the families.
    await expect(sheet(page).locator('.insp-title')).toHaveText('attn');
    await expect(sheet(page).locator('.prim-chip')).toHaveText('attention.dense@1.0.0');
    await expect(sheet(page).locator('[data-member-value="version"]')).toHaveValue('1.0.0');
    await expect(sheet(page).locator('.fchip .mono')).toHaveText('sequence_operator');

    // Arguments: the count S6 writes, and the rows in the states it draws them in.
    await expect(sheet(page).locator('h2.ih:text-is("Arguments") .ihn')).toHaveText(
      '6 set · 19 declared',
    );
    await expect(row(page, 'width')).toHaveAttribute('data-mode', 'quantity');
    await expect(row(page, 'width').locator('[data-effective]')).toHaveText('4096');
    await expect(row(page, 'width').locator('.sbadge')).toHaveText('structural');
    await expect(row(page, 'mask')).toHaveAttribute('data-mode', 'literal');
    await expect(row(page, 'scale')).toHaveAttribute('data-source', 'absent');
    await expect(row(page, 'cross')).toHaveAttribute('data-source', 'default');
    await expect(row(page, 'rope.theta')).toBeVisible();
    // An inapplicable row is hidden until it is asked for, and then carries its condition.
    await expect(row(page, 'chunk')).toHaveCount(0);
    await sheet(page).locator('[data-sheet="inapplicable"]').click();
    await expect(row(page, 'chunk')).toHaveAttribute('data-applicable', 'false');
    // The literal is quoted since feature 2.11: §4.13's text form writes a string literal in
    // quotes, which is what tells it from an argument named the same way.
    await expect(sheet(page).locator('.rowmsg.faint').first()).toHaveText(
      'present_when: mask = "chunked"',
    );

    // The invariant block: every one the primitive declares, with the values it read.
    const invariants = sheet(page).locator('.inv');
    await expect(invariants).toHaveCount(4);
    await expect(invariants.first()).toContainText('heads is a multiple of kv_heads');
    await expect(invariants.first()).toContainText('heads = 32, kv_heads = 8');

    // Ports, Parameters, States and Derived, with what the core and the products answer.
    await expect(sheet(page).locator('[data-port="inputs:input"]')).toContainText('decoder.attn.norm_in');
    await expect(sheet(page).locator('[data-port="inputs:source_values"]')).toContainText('absent');
    await expect(sheet(page).locator('[data-slot="q"] .slot')).toHaveText('decoder.attn.q[layer=0]');
    await expect(sheet(page).locator('[data-slot="q_gated"]')).toHaveAttribute('data-present', 'false');
    await expect(sheet(page).locator('[data-state="kv"] [data-rule]')).toHaveText('4 of 4');
    await expect(sheet(page).locator('[data-tensor="decoder.attn.q[layer=0]"]')).toContainText('32.0 MiB');
    await expect(sheet(page).locator('[data-d1]')).toContainText('32 nodes');
    await expect(sheet(page).locator('[data-d1]')).toContainText('across_positions true');
  });

  test('lists only the cardinality quantities in the quantity mode of `heads`', async ({ page }) => {
    await open(page);
    await openModel(page);
    await select(page, ATTN);

    const select_ = row(page, 'heads').locator('[data-value="heads"]');
    const offered = await select_.locator('option').allInnerTexts();
    expect(offered).toEqual(['d', 'ffn', 'heads', 'kv_heads', 'head_dim', 'layers', 'vocab']);
    // `eps` is a real and `precision` an enum: the list is the argument's own type kind, which the
    // core answers and the generated schema cannot (a set domain there is an `enum` too).
    expect(offered).not.toContain('eps');
    expect(offered).not.toContain('precision');
  });
});

test.describe('an edit on a row', () => {
  test('sets `kv_heads` to 3 and the core answers with V8, on the row and in Problems', async ({
    page,
  }) => {
    await open(page);
    await openModel(page);
    await select(page, ATTN);

    await row(page, 'kv_heads').locator('[data-modes="kv_heads"]').selectOption('literal');
    await row(page, 'kv_heads').locator('[data-value="kv_heads"]').fill('3');
    await row(page, 'kv_heads').locator('[data-value="kv_heads"]').blur();

    await expect(row(page, 'kv_heads').locator('[data-effective]')).toHaveText('3');
    const failed = sheet(page).locator('.inv[data-invariant="fails"]');
    await expect(failed).toHaveCount(1);
    await expect(failed).toContainText('heads is a multiple of kv_heads');
    await expect(failed).toContainText('heads = 32, kv_heads = 3');

    await expect(
      problems(page).locator('.prow[data-code="V8"]').first(),
    ).toContainText("'heads is a multiple of kv_heads' does not hold", { timeout: 10_000 });
  });

  test('sets `eps` to 0 on `final_n` and the domain error lands on that row', async ({ page }) => {
    await open(page);
    await openModel(page);
    await select(page, FINAL);

    // Switching the mode writes the first value the place admits (D5: the document stays on the
    // grammar), which for a real with an exclusive lower bound of zero is already the refusal.
    await row(page, 'eps').locator('[data-modes="eps"]').selectOption('literal');
    await expect(row(page, 'eps')).toHaveClass(/errrow/);
    await expect(row(page, 'eps').locator('+ .rowmsg')).toHaveText(
      "argument 'eps' = 0.0 is below the domain bound 0 (exclusive)",
    );
  });

  test('pins a defaulted value as a literal, and clears it back to nothing', async ({ page }) => {
    await open(page);
    await openModel(page);
    await select(page, ATTN);

    expect(await sourceText(page)).not.toContain('"cross"');
    await row(page, 'cross').locator('[data-pin="cross"]').click();
    await expect(row(page, 'cross')).toHaveAttribute('data-source', 'given');
    expect(await sourceText(page)).toContain('"cross"');

    await row(page, 'cross').locator('[data-pin="cross"]').click();
    await expect(row(page, 'cross')).toHaveAttribute('data-source', 'default');
    // "A row left at its default stores nothing" (§4.12).
    expect(await sourceText(page)).not.toContain('"cross"');
  });
});

test.describe('a structural change', () => {
  test('enables `output_gate`, notices the stale binding, and rebinds it in one click', async ({
    page,
  }) => {
    await open(page);
    await openModel(page);
    await select(page, ATTN);

    await row(page, 'output_gate').locator('[data-value="output_gate"]').check();

    // The core's new slot set, on the sheet: `q` is gone and `q_gated` is there, unbound.
    await expect(sheet(page).locator('[data-slot="q"]')).toHaveAttribute('data-present', 'false');
    await expect(sheet(page).locator('[data-slot="q_gated"] .slot')).toHaveText('unbound');

    const notice = problems(page).locator('.prow[data-source="editor"]');
    await expect(notice).toContainText(
      'binding attn.q names a slot the arguments no longer create',
      { timeout: 10_000 },
    );
    await notice.locator('.fix[data-fix="rebind-slot"]').click();

    expect(await sourceText(page)).toContain('"parameter": "q_gated"');
    await expect(sheet(page).locator('[data-slot="q_gated"] .slot')).toHaveText(
      'decoder.attn.q[layer=0]',
      { timeout: 10_000 },
    );
    await expect(problems(page).locator('.prow[data-source="editor"]')).toHaveCount(0);
  });
});

test.describe('the sheet without a pointer', () => {
  test('is reachable by the keyboard and clean under axe, in both themes', async ({ page }) => {
    await open(page);
    await openModel(page);
    await select(page, ATTN);

    // Every control is a real one: the name field, the mode select, the value, the actions.
    await sheet(page).locator('input[aria-label="Name"]').focus();
    await expect(sheet(page).locator('input[aria-label="Name"]')).toBeFocused();
    await sheet(page).locator('[data-modes="mask"]').focus();
    await expect(sheet(page).locator('[data-modes="mask"]')).toBeFocused();

    for (const theme of ['view.theme-light', 'view.theme-dark']) {
      await command(page, 'View', theme);
      const results = await new AxeBuilder({ page }).include('.insp').analyze();
      expect(results.violations.map((one) => `${one.id}: ${one.nodes.length}`)).toEqual([]);
    }
  });
});
