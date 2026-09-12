import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Locator, type Page } from '@playwright/test';

import { sourceText } from './source-text.js';

/**
 * Bindings, identities and the location editor — feature 2.13, plan §4.14, artboard S8.
 *
 * The block's own three cases are here, each on the document it names:
 *
 * - **qwen3.5-4b-text**: untie `lm_head.weight` and tie it back through "Tie to…" — the document
 *   is the one that was opened, byte for byte;
 * - **llama3-8b**: edit `decoder.attn.q`'s location tokens — the preview names for layer 0 and
 *   layer 31 follow the edit;
 * - **gemma3n-kvshare**: choose the `stack` form on a slot that declares a multiplicity
 *   (`expand.projection`) — the axis list offers `multiplicity`.
 *
 * Beside them: the chip-onto-chip gesture §4.7 left without a source until this feature, and the
 * whole panel under axe in both themes.
 */

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

/** Open the Examples workspace and one of the corpus documents. */
async function openModel(page: Page, model: string): Promise<void> {
  await page.locator('.nothing .link[data-open="examples"]').click();
  await page.locator('.dlg [data-workspace="examples"]').click();
  await expect(page.locator('footer.status [data-workspace]')).toHaveText('Examples');
  await command(page, 'File', 'file.open-model');
  await page.locator(`.dlg [data-document="models/${model}.json"]`).click();
  await expect(page.locator(`.gcanvas[data-canvas="models/${model}.json"]`)).toBeVisible();
}

/** The Properties panel's body. */
function sheet(page: Page): Locator {
  return page.locator('.insp .panel-body');
}

/** Open a row of the tree, which is how a place under a closed group is reached (§4.5). */
async function expand(page: Page, pointer: string): Promise<void> {
  const chevron = page.locator(`.tree [data-chevron="${pointer}"]`);
  if ((await page.locator(`.tree [data-row="${pointer}"]`).count()) === 0) return;
  if ((await chevron.innerText()) === '▸') await chevron.click();
}

/** Select a place in the explorer, which is what opens its sheet (§4.5). */
async function select(page: Page, pointer: string): Promise<void> {
  const steps = pointer.split('/').filter((step) => step !== '');
  for (let depth = 1; depth < steps.length; depth += 1) {
    await expand(page, `/${steps.slice(0, depth).join('/')}`);
  }
  await page.locator(`.tree [data-row="${pointer}"]`).click();
  await expect(sheet(page).locator('[data-place]')).toHaveText(pointer === '' ? '/' : pointer);
}

/**
 * Select an **instance** in the explorer: its sheet is §4.11's first row, which has no place row.
 *
 * The sheet of a site is the instance panel — its title is the site's own name and its chip is the
 * primitive it pins — so that is what says the selection landed.
 */
async function selectSite(page: Page, pointer: string, name: string): Promise<void> {
  const steps = pointer.split('/').filter((step) => step !== '');
  for (let depth = 1; depth < steps.length; depth += 1) {
    await expand(page, `/${steps.slice(0, depth).join('/')}`);
  }
  await page.locator(`.tree [data-row="${pointer}"]`).click();
  await expect(sheet(page).locator('.insp-title')).toHaveText(name);
}

test.describe('an identity a slot leaves and joins', () => {
  test('unties `lm_head.weight` and ties it back — the document is the one that was opened', async ({
    page,
  }) => {
    await open(page);
    await openModel(page, 'qwen3.5-4b-text');
    const before = await sourceText(page);

    // The tie the corpus writes: one identity, two members (§4.7's `⇄` chip).
    await selectSite(page, '/instances/lm_head', 'lm_head');
    await expect(sheet(page).locator('[data-slot="weight"] .slot')).toHaveText('embed.weight');

    // "Bind privately (creates `<site>.<slot>`)" — §4.11's own row.
    await sheet(page).locator('[data-bind-private="weight"]').click();
    await expect(sheet(page).locator('[data-slot="weight"] .slot')).toHaveText('lm_head.weight');
    const untied = JSON.parse(await sourceText(page)) as {
      bindings: { parameters: Record<string, { members: unknown[]; tensor?: { name: string } }> };
    };
    expect(Object.keys(untied.bindings.parameters)).toEqual([
      'embed.weight',
      'final_n.weight',
      'lm_head.weight',
    ]);
    expect(untied.bindings.parameters['embed.weight']?.members).toHaveLength(1);
    // A top-level rule declares the symbol its identity is named by, which the grammar requires.
    expect(untied.bindings.parameters['lm_head.weight']?.tensor).toEqual({ name: 'lm_head.weight' });

    // "Tie to…": the list is the document's own identities, and the core's compatibility list
    // marks the ones V15 admits. `embed.weight` is one of them; the slot's own is not offered.
    await sheet(page).locator('[data-tie-open="weight"]').click();
    const list = sheet(page).locator('[data-tie-list="weight"]');
    await expect(list.locator('[data-tie-into="embed.weight"]')).toHaveAttribute(
      'data-compatible',
      'true',
    );
    await expect(list.locator('[data-tie-into="lm_head.weight"]')).toBeDisabled();
    await list.locator('[data-tie-into="embed.weight"]').click();

    // The member is back where the corpus writes it, and the rule it left went with its last
    // member — the store's own collapse rule, the grammar's `minItems`.
    await expect(sheet(page).locator('[data-slot="weight"] .slot')).toHaveText('embed.weight');
    expect(await sourceText(page)).toBe(before);
  });

  test('ties two identities by dragging a slot chip onto another (§4.7)', async ({ page }) => {
    await open(page);
    await openModel(page, 'llama3-8b');

    // `llama3-8b` binds `embed.weight` and `lm_head.weight` as two identities that *may* join —
    // the two `embedding.table` slots the reference base declares `shareable`.
    const source = page.locator('[data-slot="/instances/lm_head:weight"]');
    const target = page.locator('[data-slot="/instances/embed:weight"]');
    const from = await source.boundingBox();
    const onto = await target.boundingBox();
    expect(from).not.toBeNull();
    expect(onto).not.toBeNull();
    if (from === null || onto === null) return;

    await page.mouse.move(from.x + from.width / 2, from.y + from.height / 2);
    await page.mouse.down();
    await page.mouse.move(onto.x + onto.width / 2, onto.y + onto.height / 2, { steps: 8 });
    // The verdict is shown *during* the gesture and never blocks it (§4.7, Q5).
    await expect(page.locator('[data-tie-verdict]')).toHaveAttribute('data-tie-verdict', 'ok');
    await page.mouse.up();

    const shown = await sourceText(page);
    expect(shown).not.toContain('"lm_head.weight": {');
    expect(shown).toContain('"instance": "lm_head"');
    // One identity with two members now, which is what `qwen3.5-4b-text` writes by hand.
    await expect(page.locator('[data-slot="/instances/lm_head:weight"]')).toHaveAttribute(
      'title',
      'embed.weight',
    );
    // One gesture, one undo (D13).
    await command(page, 'Edit', 'edit.undo');
    expect(await sourceText(page)).toContain('"lm_head.weight": {');
  });
});

test.describe('the Identities list (the explorer’s Bindings node)', () => {
  test('creates an identity, and its sheet is where its members and its location are written', async ({
    page,
  }) => {
    await open(page);
    await openModel(page, 'llama3-8b');
    await select(page, '/bindings/parameters');

    // §4.16's table over the map of rules: three identities, the entry definition's own members
    // as columns. A binding rule's name is a label — nothing refers to it (feature 2.2) — so the
    // map has no word of its own and the table calls one an entry.
    await expect(sheet(page).locator('table.t tr[data-row]')).toHaveCount(3);
    await sheet(page).locator('[data-add-entry="/bindings/parameters"]').click();

    // D5's skeleton of a `parameter_binding`, and the sheet of the new rule is open on it.
    await expect(sheet(page).locator('[data-place]')).toHaveText('/bindings/parameters/entry');
    const shown = await sourceText(page);
    expect(shown).toContain('"entry": {');
    expect(shown).toContain('"members": []');
  });
});

test.describe('the location editor', () => {
  test('edits `decoder.attn.q`’s tokens — the preview names layer 0 and layer 31', async ({
    page,
  }) => {
    await open(page);
    await openModel(page, 'llama3-8b');
    await select(page, '/compositions/decoder/bindings/parameters/attn.q');

    // S8's own row: the three item forms of `physical_name`, as chips.
    const tokens = sheet(page).locator('[data-tokens="/compositions/decoder/bindings/parameters/attn.q/location/tensor"]');
    await expect(tokens.locator('.tok:not(.add)')).toHaveCount(3);
    await expect(tokens.locator('[data-token="0"]')).toHaveClass(/lit/);
    await expect(tokens.locator('[data-token="1"]')).toHaveClass(/idx/);
    await expect(tokens.locator('select[data-token-kind="1"]')).toHaveValue('index');
    await expect(tokens.locator('input[data-token-text="1"]')).toHaveValue('layer');

    // And the evaluated names beneath: the head, what is not drawn, and the last (S8's `… N more`).
    const preview = sheet(page).locator('[data-preview="decoder.attn.q"]');
    await expect(preview.locator('[data-preview-name="decoder.attn.q[layer=0]"]')).toHaveText(
      'model.layers.0.self_attn.q_proj.weight',
    );
    await expect(preview.locator('[data-preview-name="decoder.attn.q[layer=31]"]')).toHaveText(
      'model.layers.31.self_attn.q_proj.weight',
    );
    await expect(preview.locator('[data-preview-more="decoder.attn.q"]')).toHaveText('… 29 more');

    // Edit the literal token: every instance's name follows, evaluated by the core and not here.
    await tokens.locator('input[data-token-text="0"]').fill('model.decoder.layers.');
    await expect(preview.locator('[data-preview-name="decoder.attn.q[layer=0]"]')).toHaveText(
      'model.decoder.layers.0.self_attn.q_proj.weight',
    );
    await expect(preview.locator('[data-preview-name="decoder.attn.q[layer=31]"]')).toHaveText(
      'model.decoder.layers.31.self_attn.q_proj.weight',
    );
    expect(await sourceText(page)).toContain('"model.decoder.layers."');

    // A token added is a literal, and one removed leaves the name the author wrote.
    await tokens.locator('[data-token-add]').click();
    await expect(tokens.locator('.tok:not(.add)')).toHaveCount(4);
    await tokens.locator('[data-token-remove="3"]').click();
    await expect(tokens.locator('.tok:not(.add)')).toHaveCount(3);

    // The axes the core answers for this slot, which is what a `stack` or a `slice` would name.
    await expect(sheet(page).locator('[data-axes="decoder.attn.q"]')).toHaveText('heads_flat, feature');
    // And V14's own set, for the role the slot declares.
    await expect(sheet(page).locator('[data-admissible="decoder.attn.q"]')).toContainText('bf16');
  });

  test('offers `multiplicity` on the `stack` form of a slot that declares one', async ({ page }) => {
    await open(page);
    await openModel(page, 'gemma3n-kvshare');
    await select(page, '/bindings/parameters/expand.projection');

    const axis = sheet(page).locator('[data-chooser="/location"] [data-member="axis"]');
    const optionsOf = async (): Promise<(string | null)[]> => {
      const list = await axis.locator('input[list]').getAttribute('list');
      return page
        .locator(`datalist#${String(list)} option`)
        .evaluateAll((options) => options.map((option) => option.getAttribute('value')));
    };

    // §4.14: "axis (select of the slot's axis names plus `multiplicity` when the slot declares one
    // — names the core returns)". The storage axis leads them, as §3.4 stores it.
    await expect(sheet(page).locator('[data-modes="/location"]')).toHaveValue('stack');
    await expect(axis.locator('input')).toHaveValue('multiplicity');
    expect(await optionsOf()).toEqual(['multiplicity', 'feature']);

    // The same after *choosing* the form, twice over: the chooser writes the blank of the
    // alternative — a blank `axis`, which the grammar refuses — and the rows that repair it are
    // drawn all the same, with the list still offering the slot's own axes. The places come from
    // the tree the sheet holds and the names from the last analysis there was, which is why
    // neither waits for a document the core can read.
    await sheet(page).locator('[data-modes="/location"]').selectOption('tensor');
    await expect(sheet(page).locator('[data-member="axis"]')).toHaveCount(0);
    await sheet(page).locator('[data-modes="/location"]').selectOption('stack');
    await expect(axis.locator('input')).toHaveValue('');
    expect(await optionsOf()).toEqual(['multiplicity', 'feature']);
    // And typing one of them puts the document back on the grammar, which the preview then reads.
    await axis.locator('input').fill('multiplicity');
    await expect(sheet(page).locator('[data-axes="expand.projection"]')).toHaveText(
      'multiplicity, feature',
    );

    // The stack's part is a location of its own, and its tokens are the same editor (S8's note
    // about the recursion).
    await expect(
      sheet(page).locator('[data-tokens="/bindings/parameters/expand.projection/location/stack/part/tensor"]'),
    ).toBeVisible();
  });

  test('is reachable by the keyboard and clean under axe, in both themes', async ({ page }) => {
    await open(page);
    await openModel(page, 'llama3-8b');
    await select(page, '/compositions/decoder/bindings/parameters/attn.q');

    // Every control of a token is a real control: the form is a select, the text a field, the
    // removal a button — so the tab order walks them (§4.4's rule about accelerators).
    await sheet(page).locator('select[data-token-kind="0"]').focus();
    await page.keyboard.press('Tab');
    await expect(sheet(page).locator('input[data-token-text="0"]')).toBeFocused();

    for (const theme of ['view.theme-light', 'view.theme-dark']) {
      await command(page, 'View', theme);
      const results = await new AxeBuilder({ page }).include('.insp').analyze();
      expect(results.violations.map((one) => `${one.id}: ${one.nodes.length}`)).toEqual([]);
    }
  });
});

test.describe('a template instance’s `weights_location_prefix`', () => {
  test('is the same token editor, on the instance’s own sheet (§4.14)', async ({ page }) => {
    await open(page);
    await openModel(page, 'shieldstral-3b-composite');
    await selectSite(page, '/instances/text', 'text');

    // "A `weights_location_prefix` on a template instance is the same token editor" — one binding
    // at `physical_name`, one editor, wherever the grammar writes one.
    const tokens = sheet(page).locator('[data-tokens="/instances/text/weights_location_prefix"]');
    await expect(tokens.locator('input[data-token-text="0"]')).toHaveValue('language_model.model.');
    await tokens.locator('input[data-token-text="0"]').fill('language_model.decoder.');
    expect(await sourceText(page)).toContain('"language_model.decoder."');
  });
});

test.describe('the gestures beside a slot row', () => {
  test('open the identity’s own sheet, where its dtype and its location are edited', async ({
    page,
  }) => {
    await open(page);
    await openModel(page, 'llama3-8b');
    await selectSite(page, '/compositions/decoder/instances/attn', 'attn');

    // §4.11's Parameters row, with the gestures this feature adds beside it.
    await expect(sheet(page).locator('[data-slot="q"] .slot')).toHaveText('decoder.attn.q[layer=0]');
    await sheet(page).locator('[data-edit-identity="q"]').click();
    // The place the *author* wrote, which §5.2 rule 7 names `decoder.attn.q` (feature 2.12).
    await expect(sheet(page).locator('[data-place]')).toHaveText(
      '/compositions/decoder/bindings/parameters/attn.q',
    );
    await expect(sheet(page).locator('[data-member-site="decoder/attn[layer=0].q"]')).toHaveText(
      'decoder/attn[layer=0]',
    );
  });

  test('share a state port with another identity, from the States row', async ({ page }) => {
    await open(page);
    await openModel(page, 'gemma3n-kvshare');
    await selectSite(page, '/compositions/decoder/instances/attn_full', 'attn_full');

    // The state row carries "Share with…" where a parameter row carries "Tie to…" — V9 is the
    // rule behind the first as V15 is behind the second (§5.3).
    await expect(sheet(page).locator('[data-state="kv"] [data-tie-open="kv"]')).toHaveText(
      'Share with…',
    );
    await sheet(page).locator('[data-state="kv"] [data-tie-open="kv"]').click();
    const list = sheet(page).locator('[data-tie-list="kv"]');
    await expect(list.locator('[data-tie-into]').first()).toBeVisible();
    // The identity the port already belongs to is offered and not takeable: a slot does not join
    // the identity it is in.
    await expect(list.locator('[data-tie-into="decoder.attn_full.kv"]')).toBeDisabled();
  });
});
