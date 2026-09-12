import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Locator, type Page } from '@playwright/test';

import { sourceText } from './source-text.js';

/**
 * The other sheets of §4.11 and the tables of §4.16 — feature 2.12, artboard S15.
 *
 * The block's own four cases are here: **add a quantity and use it from an argument**; **rename
 * `d`**, whose references are rewritten and undone in one step; **expose `final_n.output` as an
 * output**, and the JSON the interface then holds; and **the composition sheet editing `stop` to
 * a quantity**. Beside them, what S15 draws: the Document sheet with nothing selected, the
 * Quantities table with its used-by and resolved columns, an identity's D3 rows, an edge's D2
 * value, and the whole panel under axe in both themes.
 */

const MODEL = 'models/llama3-8b.json';

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

/** Open the Examples workspace and `llama3-8b`. */
async function openModel(page: Page): Promise<void> {
  await page.locator('.nothing .link[data-open="examples"]').click();
  await page.locator('.dlg [data-workspace="examples"]').click();
  await expect(page.locator('footer.status [data-workspace]')).toHaveText('Examples');
  await command(page, 'File', 'file.open-model');
  await page.locator(`.dlg [data-document="${MODEL}"]`).click();
  await expect(page.locator(`.gcanvas[data-canvas="${MODEL}"]`)).toBeVisible();
  await expect(page.locator('[data-port="/instances/embed:tokens"]')).toBeVisible();
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
  // The groups of the outline that are shut by default — `interfaces`, `bindings` and a
  // composition's own bindings — are opened on the way down, which is the reader's own gesture.
  const steps = pointer.split('/').filter((step) => step !== '');
  for (let depth = 1; depth < steps.length; depth += 1) {
    await expand(page, `/${steps.slice(0, depth).join('/')}`);
  }
  await page.locator(`.tree [data-row="${pointer}"]`).click();
  await expect(sheet(page).locator('[data-place]')).toHaveText(pointer === '' ? '/' : pointer);
}

test.describe('the Document sheet', () => {
  test('is what nothing selected shows — S15’s own region', async ({ page }) => {
    await open(page);
    await openModel(page);

    // §4.11's last row: "Nothing selected | the Document sheet".
    await expect(sheet(page).locator('.insp-title')).toHaveText('llama3-8b');
    await expect(sheet(page).locator('.insp-kind')).toHaveText('document');
    // The rows are the root's own members, generated: the tag the schema fixes is read-only, the
    // model id is a field, the bases are a list, and every map says how many it holds.
    await expect(sheet(page).locator('[data-member="schema"] [data-member-value]')).toHaveText(
      'tensorspine/2.0',
    );
    await expect(sheet(page).locator('[data-member="model"] [data-member-value="model"]')).toHaveValue(
      'llama3-8b',
    );
    await expect(sheet(page).locator('[data-member="version"]')).toHaveAttribute(
      'data-present',
      'false',
    );
    await expect(
      sheet(page).locator('[data-list="/primitive_libraries"] [data-member-value="base"]'),
    ).toHaveValue('../primitive-library/');
    // S15's counts, each the map's own: 9 quantities, 3 instances, 1 composition.
    await expect(sheet(page).locator('[data-map="/quantities"] .ihn')).toHaveText('9');
    await expect(sheet(page).locator('[data-map="/instances"] .ihn')).toHaveText('3');
    await expect(sheet(page).locator('[data-map="/compositions"] .ihn')).toHaveText('1');
    // The resolution status §4.11 asks of the bases — the loader's own answer, and S15's own
    // first figure for the reference base.
    await expect(sheet(page).locator('[data-library="primitives"]')).toHaveText('36 primitives');
    await expect(sheet(page).locator('[data-library="resolved"]')).toHaveText('resolved');
  });

  test('is also what the document’s own row and §4.4’s command open', async ({ page }) => {
    await open(page);
    await openModel(page);
    await select(page, '/quantities/d');
    await expect(sheet(page).locator('.insp-title')).toHaveText('d');

    await command(page, 'Model', 'model.document-properties');
    await expect(sheet(page).locator('.insp-title')).toHaveText('llama3-8b');
    // Feature 2.7 left the tree's own root row clearing the selection; it opens the sheet now.
    await select(page, '/quantities/d');
    await page.locator('.tree [data-row=""]').click();
    await expect(sheet(page).locator('[data-place]')).toHaveText('/');
  });
});

test.describe('the Quantities table', () => {
  test('draws S15’s columns, the derivation line and the notice on a name nothing uses', async ({
    page,
  }) => {
    await open(page);
    await openModel(page);
    await select(page, '/quantities');

    // The columns are the entry definition's own members, in the schema's order, with the name
    // first and the two the *core* answers last.
    const columns = await sheet(page)
      .locator('table.t th')
      .evaluateAll((cells) => cells.map((cell) => cell.getAttribute('data-column')));
    // The schema's own property order, which puts the required members first: `type` and
    // `source` before `domain`, where §4.16 and S15 write `domain` last. The order is read, not
    // chosen, and the two the *core* answers come after them.
    expect(columns).toEqual(['name', 'type', 'source', 'domain', 'used by', 'resolved']);
    await expect(sheet(page).locator('table.t tr[data-row]')).toHaveCount(9);

    // `d`: a cardinality, a literal, ten references — E13's own figure for this document — and
    // the value the core resolves it to.
    const row = sheet(page).locator('tr[data-row="/quantities/d"]');
    await expect(row.locator('[data-modes="/type"]')).toHaveValue('cardinality');
    await expect(row.locator('[data-modes="/source"]')).toHaveValue('literal');
    await expect(row.locator('[data-used-by="d"]')).toHaveText('10');
    await expect(row.locator('[data-resolved="d"]')).toHaveText('4096');

    // `head_dim` is a literal that declares a derivation: S15 draws that under the row.
    await expect(sheet(page).locator('[data-note="/quantities/head_dim"]')).toHaveText(
      'derivation d div heads',
    );
    await expect(
      sheet(page).locator('tr[data-row="/quantities/head_dim"] [data-resolved="head_dim"]'),
    ).toHaveText('128');
  });

  test('adds a quantity, and an argument then uses it', async ({ page }) => {
    await open(page);
    await openModel(page);

    // §4.4's `Model ▸ Add Quantity`, which adds to the map `presentation.json` says declares one.
    await command(page, 'Model', 'model.add-quantity');
    // The new declaration is selected, and its sheet is the one the author fills in.
    await expect(sheet(page).locator('.insp-kind')).toHaveText('quantity');
    await expect(sheet(page).locator('[data-place]')).toHaveText('/quantities/quantity');
    // D5's skeleton, from the schema: the first alternative of each required member.
    await expect(sheet(page).locator('[data-chooser="/type"] [data-modes="/type"]')).toHaveValue(
      'cardinality',
    );
    await expect(sheet(page).locator('[data-chooser="/source"] [data-modes="/source"]')).toHaveValue(
      'literal',
    );

    // Rename it and give it a value, both in the sheet's own rows (§4.4).
    await sheet(page).locator('input[data-name-field]').fill('ffn_inner');
    await sheet(page).locator('input[data-name-field]').press('Enter');
    await sheet(page).locator('[data-chooser="/source"] [data-member-value="value"]').fill('11008');
    await sheet(page).locator('[data-chooser="/source"] [data-member-value="value"]').blur();
    await expect(sheet(page).locator('[data-resolved="ffn_inner"]')).toHaveText('11008');

    // And an argument uses it: the quantity mode of §4.12 offers the document's own quantities,
    // which now include this one.
    await page.locator('.tree [data-row="/compositions/decoder/instances/ffn"]').click();
    const inner = sheet(page).locator('[data-argument="inner"]');
    await inner.locator('[data-value="inner"]').selectOption('ffn_inner');
    await expect(inner.locator('[data-effective]')).toHaveText('11008');
    expect(await sourceText(page)).toContain('"quantity": "ffn_inner"');
  });

  test('renames `d` — every reference rewritten, and undone in one step', async ({ page }) => {
    await open(page);
    await openModel(page);
    await select(page, '/quantities/d');

    // What the sheet says a rename would rewrite is what the rename *does* rewrite: the same
    // selectors answer both (feature 2.2's `refers`).
    await expect(sheet(page).locator('h3.ih:text-is("Used by") .ihn')).toHaveText('10');
    const before = await sourceText(page);
    expect(before.match(/"quantity": "d"/g)).toHaveLength(10);

    await sheet(page).locator('input[data-name-field]').fill('model_width');
    await sheet(page).locator('input[data-name-field]').press('Enter');

    const after = await sourceText(page);
    expect(after).not.toContain('"quantity": "d"');
    expect(after.match(/"quantity": "model_width"/g)).toHaveLength(10);
    expect(after).toContain('"model_width": {');

    // One command, so one undo (D13).
    await command(page, 'Edit', 'edit.undo');
    expect(await sourceText(page)).toBe(before);
  });
});

test.describe('a table nothing has been added to yet', () => {
  test('adds a constant from the table’s own button — the one S15 draws empty (F8)', async ({
    page,
  }) => {
    await open(page);
    await openModel(page);
    await select(page, '/constants');

    // No corpus document declares a constant, so the table exists with its columns and no row.
    await expect(sheet(page).locator('table.t tr[data-row]')).toHaveCount(0);
    const columns = await sheet(page)
      .locator('table.t th')
      .evaluateAll((cells) => cells.map((cell) => cell.getAttribute('data-column')));
    expect(columns).toEqual(['name', 'identity', 'shape', 'dtype', 'multiplicity']);

    await sheet(page).locator('[data-add-entry="/constants"]').click();
    await expect(sheet(page).locator('[data-place]')).toHaveText('/constants/constant');
    await expect(sheet(page).locator('.insp-kind')).toHaveText('constant');
    // D5's skeleton of a `constant_definition`, from the schema alone.
    expect(await sourceText(page)).toContain('"digest": ""');
  });
});

test.describe('the Interfaces table and the sheets under it', () => {
  test('exposes `final_n.output` as an output from the port’s own menu (§4.15)', async ({ page }) => {
    await open(page);
    await openModel(page);

    await page.locator('[data-port="/instances/final_n:output"]').click({ button: 'right' });
    await page.locator('.ctxmenu [data-entry="model.add-output"]').click();

    // The interface the gesture wrote: the endpoint is the core's own selector, the name is
    // proposed from the port, and `generative` is the first value its schema admits.
    const shown = await sourceText(page);
    expect(shown).toContain('"output": {');
    expect(shown).toContain('"instance": "final_n"');
    expect(shown).toContain('"port": "output"');
    expect(shown).toContain('"generative": false');

    // And the sheet of the new output is open, with §4.11's own rows on it.
    await expect(sheet(page).locator('[data-place]')).toHaveText('/interfaces/outputs/output');
    await expect(sheet(page).locator('.insp-kind')).toHaveText('output');
    await expect(sheet(page).locator('[data-member="generative"] [data-member-value]')).not.toBeChecked();
  });

  test('exposes an input port as a public input, from the same menu (§4.15)', async ({ page }) => {
    await open(page);
    await openModel(page);

    // Every input port of `llama3-8b` is already fed, and the menu offers the gesture all the
    // same: "the author wires first and fixes afterwards" (Q5), and V7 is what reports the second
    // feeder. The name is proposed from the port and the kind is the first its enumeration admits.
    await page.locator('[data-port="/instances/lm_head:input"]').click({ button: 'right' });
    await page.locator('.ctxmenu [data-entry="model.add-input"]').click();

    await expect(sheet(page).locator('[data-place]')).toHaveText('/interfaces/inputs/input');
    await expect(sheet(page).locator('.insp-kind')).toHaveText('input');
    const shown = await sourceText(page);
    expect(shown).toContain('"instance": "lm_head"');
    expect(shown).toContain('"kind": "sequence"');
  });

  test('shows what D2 says about the input and the output of §4.11’s two rows', async ({ page }) => {
    await open(page);
    await openModel(page);
    await select(page, '/interfaces/inputs/tokens');

    // The generated form: the endpoints, the kind, the stream, `fragmented`.
    await expect(sheet(page).locator('[data-list="/to"] .ihn')).toHaveText('1');
    // The select is the input's own `kind`; the `kind` of the selector inside `to` is a constant
    // row, which is what tells the two apart without a path being written here.
    await expect(sheet(page).locator('select[data-member-value="kind"]')).toHaveValue('token');
    await expect(sheet(page).locator('[data-member="stream"]')).toHaveAttribute(
      'data-present',
      'false',
    );
    // §4.11's "stream (select of the document's streams, or *introduces its own*)": the member
    // refers to what the inputs declare (feature 2.2's own rule), so the field offers their names
    // — as suggestions, since a name nobody declares is V1's refusal and not the editor's (Q5).
    await sheet(page).locator('[data-member="stream"] [data-add-member]').click();
    const list = await sheet(page)
      .locator('[data-member="stream"] input[list]')
      .getAttribute('list');
    expect(
      await page.locator(`datalist#${String(list)} option`).evaluateAll((options) =>
        options.map((option) => option.getAttribute('value')),
      ),
    ).toEqual(['tokens']);
    // And what the products say: the value it delivers, what it is required for, its stream.
    await expect(sheet(page).locator('[data-value="tokens"]')).toContainText('i32[tokens]');
    await expect(sheet(page).locator('[data-required-for="tokens"]')).toHaveText('logits');
    await expect(sheet(page).locator('[data-stream="tokens"]')).toContainText('token');

    await select(page, '/interfaces/outputs/logits');
    await expect(sheet(page).locator('[data-member="generative"] [data-member-value]')).toBeChecked();
    // S15's own figure: 501 KiB an element, which is D2's `bytes_per_element` as `--view`'s own
    // `fmt_bytes` writes it (no decimal past 10 KiB, which is why S15's figure reads as it does).
    await expect(sheet(page).locator('[data-value="lm_head.logits"]')).toContainText('501 KiB');
  });
});

test.describe('the composition, edge and identity sheets', () => {
  test('edits `decoder`’s `stop` to a quantity, in the sheet’s own rows', async ({ page }) => {
    await open(page);
    await openModel(page);
    await select(page, '/compositions/decoder');

    // §4.11's Composition row: the indices as `name: start, stop, step` expression rows, the
    // families, and what the composition holds.
    const stop = sheet(page).locator('[data-entry="/indices/layer"] [data-member="stop"]');
    await expect(stop.locator('.xtext')).toHaveValue('32');
    await expect(sheet(page).locator('[data-held="instances"]')).toContainText('6');

    // §4.13's text form, which is the editor's other half: what it writes is the tagged JSON.
    await stop.locator('.xtext').fill('layers');
    await stop.locator('.xtext').press('Enter');
    await expect(stop.locator('[data-resolved]')).toHaveText('resolves to 32');

    // `layers` is named nowhere else in this document, so the one occurrence is the bound.
    const shown = await sourceText(page);
    expect(shown).toContain('"quantity": "layers"');
    // The document still derives, so the canvas's own count is the quantity's value.
    await expect(page.locator('[data-box="/compositions/decoder"] .g-count')).toContainText('32');
  });

  test('shows an edge’s endpoints and the D2 value it carries', async ({ page }) => {
    await open(page);
    await openModel(page);
    await select(page, '/bindings/values/decoder.entry');

    await expect(sheet(page).locator('.insp-kind')).toHaveText('binding rule');
    await expect(
      sheet(page).locator('[data-section="/from"] [data-member-value="instance"]'),
    ).toHaveValue('embed');
    await expect(sheet(page).locator('[data-section="/from"] [data-member-value="port"]')).toHaveValue(
      'output',
    );
    await expect(sheet(page).locator('[data-value="embed.output"]')).toContainText(
      'bf16[tokens, model.width=4096]',
    );
  });

  test('shows an identity’s members and the D3 rows of its instances', async ({ page }) => {
    await open(page);
    await openModel(page);
    await select(page, '/compositions/decoder/bindings/parameters/attn.q');

    await expect(sheet(page).locator('.insp-kind')).toHaveText('parameters identity');
    // The members are the generated list; the `tensor` the rule declares none of is absent.
    await expect(sheet(page).locator('[data-list="/members"] .ihn')).toHaveText('1');
    // The rule declares no `tensor` at all — §5.2 rule 7 names the identity for it — and the row
    // says so. (`location` holds a `tensor` of its own, further down; this is the first.)
    await expect(sheet(page).locator('[data-member="tensor"]').first()).toHaveAttribute(
      'data-present',
      'false',
    );
    // And the products: one D3 row per iteration, named as §5.2 rule 7 names the identity.
    await expect(sheet(page).locator('h3.ih:text-is("Derived") .ihn')).toHaveText('32 instances');
    await expect(sheet(page).locator('[data-tensor="decoder.attn.q[layer=0]"]')).toContainText(
      '32.0 MiB',
    );
    // Thirty-two instances of 32 MiB, added where D3's own totals are added: 1 GiB exactly.
    await expect(sheet(page).locator('[data-identity="decoder.attn.q"]')).toContainText('1.00 GiB');
  });
});

test.describe('the sheets without a pointer', () => {
  test('are reachable by the keyboard and clean under axe, in both themes', async ({ page }) => {
    await open(page);
    await openModel(page);
    await select(page, '/quantities');

    await sheet(page).locator('[data-open-row="/quantities/d"]').focus();
    await expect(sheet(page).locator('[data-open-row="/quantities/d"]')).toBeFocused();
    await page.keyboard.press('Enter');
    await expect(sheet(page).locator('[data-place]')).toHaveText('/quantities/d');

    for (const theme of ['view.theme-light', 'view.theme-dark']) {
      await command(page, 'View', theme);
      for (const pointer of ['', '/quantities', '/interfaces/inputs/tokens', '/compositions/decoder']) {
        await select(page, pointer);
        const results = await new AxeBuilder({ page }).include('.insp').analyze();
        expect(results.violations.map((one) => `${one.id}: ${one.nodes.length}`)).toEqual([]);
      }
    }
  });
});
