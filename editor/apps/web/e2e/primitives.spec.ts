import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Locator, type Page } from '@playwright/test';

import { sourceText } from './source-text.js';

/**
 * Instantiating a primitive — feature 2.21, plan §4.6, §4.7, §4.11, §9 Q5.
 *
 * What the browser layer is for here is the three things a model cannot be asked about: that the
 * suggest list is on the screen with the library's own identities in it, that one field over the
 * reference's two members writes the corpus's own bytes back, and that a name the catalog does
 * not carry raises a question rather than a refusal — with Cancel leaving the document untouched.
 *
 * **Nothing below is a fixture.** The identities are read from the repository's reference base,
 * the same directory the build vendors, so the suite offers nothing the editor does not gather.
 * What it types is what a *person* types: `att`, `attention.dens`.
 */

const repository = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', '..');
const MODEL = 'models/llama3-8b.json';
const modelText = readFileSync(join(repository, 'data/models/llama3-8b.json'), 'utf8');
const ATTN = '/compositions/decoder/instances/attn';


/** Every identity the repository's reference base carries — its paths are its identities (§8.2). */
function catalog(): string[] {
  const root = join(repository, 'data/primitive-library/primitives');
  const found: string[] = [];
  const walk = (current: string, name: readonly string[]): void => {
    for (const entry of readdirSync(current, { withFileTypes: true })) {
      if (entry.isDirectory()) {
        walk(join(current, entry.name), [...name, entry.name]);
        continue;
      }
      if (entry.name.endsWith('.json')) {
        found.push(`${name.join('.')}@${entry.name.slice(0, -'.json'.length)}`);
      }
    }
  };
  walk(root, []);
  return found.sort();
}

const IDENTITIES = catalog();
/** The identity `decoder/attn` of `llama3-8b` pins, read out of the corpus document itself. */
const PINNED = (() => {
  const held = JSON.parse(modelText) as {
    compositions: Record<string, { instances: Record<string, { primitive: { name: string; version: string } }> }>;
  };
  const one = held.compositions['decoder']?.instances['attn']?.primitive;
  return `${one?.name ?? ''}@${one?.version ?? ''}`;
})();

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

/** Open the vendored Examples workspace — where the reference base is, hence the catalog. */
async function openExamples(page: Page): Promise<void> {
  await page.locator('.nothing .link[data-open="examples"]').click();
  await page.locator('.dlg [data-workspace="examples"]').click();
  await expect(page.locator('footer.status [data-workspace]')).toHaveText('Examples');
}

/** Open the Examples workspace, and `llama3-8b` in it. */
async function openModel(page: Page): Promise<void> {
  await openExamples(page);
  await command(page, 'File', 'file.open-model');
  await page.locator(`.dlg [data-document="${MODEL}"]`).click();
  await expect(page.locator(`.gcanvas[data-canvas="${MODEL}"]`)).toBeVisible();
  await expect(page.locator('[data-port="/instances/embed:tokens"]')).toBeVisible();
}

/** The Properties panel's body. */
function sheet(page: Page): Locator {
  return page.locator('.insp .panel-body');
}

/** The one field of §4.11's Identity row. */
function field(page: Page): Locator {
  return sheet(page).locator('[data-identity="/primitive"]');
}

/** Select `decoder/attn` by clicking its card, the composition opened first. */
async function selectAttn(page: Page): Promise<void> {
  const fold = page.locator('[data-fold="/compositions/decoder"]');
  if ((await fold.getAttribute('aria-expanded')) !== 'true') await fold.click();
  await page.locator(`.gcanvas [data-name="${ATTN}"]`).click();
  await expect(sheet(page).locator('.insp-title')).toHaveText('attn');
}

/** What the suggest list beside the field offers, in its own order. */
async function offered(page: Page): Promise<string[]> {
  const id = await field(page).getAttribute('list');
  return page
    .locator(`datalist#${id ?? ''} option`)
    .evaluateAll((options) => options.map((one) => (one as HTMLOptionElement).value));
}

test.describe('the Identity row’s one field', () => {
  test('reads the pinned primitive as `name@version`, the library’s own form', async ({ page }) => {
    await open(page);
    await openModel(page);
    await selectAttn(page);
    await expect(field(page)).toHaveValue(PINNED);
    // And says which base it came from, since a lab's own stands beside the reference one (Q6).
    await expect(sheet(page).locator('[data-identity-base="/primitive"]')).toContainText('primitive-library');
    // The two bare fields of feature 2.10 are gone: one field, two members.
    await expect(sheet(page).locator('[data-member-value="version"]')).toHaveCount(0);
  });

  test('offers, for `att`, the identities of the library beginning so', async ({ page }) => {
    await open(page);
    await openModel(page);
    await selectAttn(page);
    await field(page).fill('att');
    const wanted = IDENTITIES.filter((id) => id.startsWith('att'));
    expect(wanted.length).toBeGreaterThan(1);
    expect(await offered(page)).toEqual(wanted);
    // Nothing typed offers the whole catalog, which is the palette's own list (§4.6).
    await field(page).fill('');
    expect((await offered(page)).length).toBe(IDENTITIES.length);
    await field(page).press('Escape');
  });

  test('writes the corpus’s own bytes when the identity it holds is chosen again', async ({ page }) => {
    await open(page);
    await openModel(page);
    await selectAttn(page);
    // Retyping the identity the document already pins must move no byte at all: the two members
    // are written where the file wrote them, in the order it wrote them (D12).
    await field(page).fill(PINNED);
    await field(page).press('Enter');
    await expect(field(page)).toHaveValue(PINNED);
    expect(await sourceText(page, MODEL)).toBe(modelText);
  });

  test('writes both members when another identity of the catalog is chosen', async ({ page }) => {
    await open(page);
    await openModel(page);
    await selectAttn(page);
    const other = IDENTITIES.find((id) => id !== PINNED && !id.startsWith('attention.')) ?? '';
    await field(page).fill(other);
    await field(page).press('Enter');
    await expect(field(page)).toHaveValue(other);
    const [name, version] = other.split('@');
    const text = await sourceText(page, MODEL);
    expect(text).toContain(`"name": "${name ?? ''}"`);
    expect(text).toContain(`"version": "${version ?? ''}"`);
    // And it is one command: Undo takes the whole reference back (D13).
    await command(page, 'Edit', 'edit.undo');
    await expect(field(page)).toHaveValue(PINNED);
    expect(await sourceText(page, MODEL)).toBe(modelText);
  });
});

test.describe('the `create` half of a binding, drawn at last', () => {
  test('stands beside the pinned primitive and beside the families, under its own label', async ({
    page,
  }) => {
    await open(page);
    await openModel(page);
    await selectAttn(page);
    // `presentation.json` has carried `create` since feature 2.2 and nothing drew it: the
    // primitive's is new, the families' was bound and invisible, and one drawing serves both.
    await expect(sheet(page).locator('[data-create="primitives"]')).toHaveText('New primitive…');
    await expect(sheet(page).locator('[data-create="families"]')).toHaveText('New family…');
  });

  test('asks the confirm’s own question for the name in the field', async ({ page }) => {
    await open(page);
    await openModel(page);
    await selectAttn(page);
    await field(page).fill('lab.attention@1.0.0');
    await sheet(page).locator('[data-create="primitives"]').click();
    const dialog = page.locator('.dlg[role="dialog"]');
    await expect(dialog).toContainText('lab.attention@1.0.0');
    await expect(dialog.locator('[data-identity-create]')).toBeVisible();
    await dialog.locator('[data-cancel="identity"]').click();
  });

  test('says so in the Log when there is nothing to declare, rather than claiming otherwise', async ({
    page,
  }) => {
    await open(page);
    await openModel(page);
    await selectAttn(page);
    // The field holds an identity the library carries: §4.4's own rule for a gesture with nothing
    // to do is that it says so, never that it opens something that would be untrue.
    await sheet(page).locator('[data-create="primitives"]').click();
    await expect(page.locator('.dlg[role="dialog"]')).toHaveCount(0);
    await command(page, 'View', 'view.log');
    await expect(page.locator('.panel .panel-body')).toContainText('already carries');
  });
});

test.describe('a name the catalog does not carry', () => {
  test('raises the confirm, and Cancel restores the field with nothing written', async ({ page }) => {
    await open(page);
    await openModel(page);
    await selectAttn(page);
    await field(page).fill('not.a.primitive@1.0.0');
    await field(page).press('Enter');
    const dialog = page.locator('.dlg[role="dialog"]');
    await expect(dialog).toBeVisible();
    await expect(dialog).toContainText('not.a.primitive@1.0.0');
    // Q5: it asks what was meant, it does not refuse the gesture — the three answers and Cancel.
    await expect(dialog.locator('[data-identity-keep]')).toBeVisible();
    await expect(dialog.locator('[data-identity-create]')).toBeVisible();
    await dialog.locator('[data-cancel="identity"]').click();
    await expect(dialog).toHaveCount(0);
    await expect(field(page)).toHaveValue(PINNED);
    expect(await sourceText(page, MODEL)).toBe(modelText);
  });

  test('offers the nearest identity for a typo, and writes it when it is taken', async ({ page }) => {
    await open(page);
    await openModel(page);
    await selectAttn(page);
    const mistyped = PINNED.slice(0, PINNED.indexOf('@') - 1);
    await field(page).fill(mistyped);
    await field(page).press('Enter');
    const dialog = page.locator('.dlg[role="dialog"]');
    await expect(dialog.locator(`[data-identity-nearest="${PINNED}"]`)).toBeVisible();
    await dialog.locator(`[data-identity-nearest="${PINNED}"]`).click();
    await expect(dialog).toHaveCount(0);
    await expect(field(page)).toHaveValue(PINNED);
    expect(await sourceText(page, MODEL)).toBe(modelText);
  });

  test('keeps it as typed when that is what was meant, and the core reports it', async ({ page }) => {
    await open(page);
    await openModel(page);
    await selectAttn(page);
    await field(page).fill('not.a.primitive@1.0.0');
    await field(page).press('Enter');
    await page.locator('.dlg[role="dialog"] [data-identity-keep]').click();
    await expect(field(page)).toHaveValue('not.a.primitive@1.0.0');
    // The verdict is the core's, in the tools' own words, and it lands in Problems (Q5).
    await command(page, 'View', 'view.problems');
    await expect(
      page.locator('.panel .panel-body .prow', {
        hasText: 'primitive absent from primitive library: not.a.primitive',
      }),
    ).toBeVisible({ timeout: 30_000 });
  });

  test('records *add to the project* as a row with the repair it is owed', async ({ page }) => {
    await open(page);
    await openModel(page);
    await selectAttn(page);
    await field(page).fill('lab.attention@1.0.0');
    await field(page).press('Enter');
    const dialog = page.locator('.dlg[role="dialog"]');
    // It says what it can do and what it cannot, rather than pretending the base exists.
    await expect(dialog).toContainText('not built yet');
    await dialog.locator('[data-identity-create]').click();
    await command(page, 'View', 'view.problems');
    const row = page.locator('.panel .panel-body .prow[data-source="editor"]', {
      hasText: 'lab.attention@1.0.0',
    });
    await expect(row).toBeVisible({ timeout: 30_000 });
    // The fix is declared and drawn — as words, since nothing can make it until 3.2 and 3.3.
    await expect(row.locator('[data-fix-awaited="create-primitive"]')).toContainText(
      'Create primitive',
    );
    await expect(row.locator('[data-fix]')).toHaveCount(0);
  });
});

test.describe('Model ▸ Add Instance…', () => {
  test('adds the primitive chosen from the catalog, where §4.7 says', async ({ page }) => {
    await open(page);
    // The Examples workspace first: a document with no base beside it gathers no library, so the
    // catalog is empty — which is the honest answer and not this case's subject.
    await openExamples(page);
    await command(page, 'File', 'file.new-model');
    await expect(page.locator('.gcanvas')).toBeVisible();
    await command(page, 'Model', 'model.add-instance');
    const dialog = page.locator('.dlg[role="dialog"]');
    await expect(dialog).toBeVisible();
    // A prefix, and the list does the rest: the version is never typed.
    const wanted = IDENTITIES.find((id) => id.startsWith('embed@')) ?? '';
    await dialog.locator('[data-catalog-filter]').fill('embe');
    await expect(dialog.locator(`[data-identity-choice="${wanted}"]`)).toBeVisible();
    await dialog.locator(`[data-identity-choice="${wanted}"]`).click();
    // §4.7's own table: the drop "adds an instance (root canvas)", named by §9 Q4's proposal.
    await expect(page.locator('[data-box="/instances/embed"]')).toBeVisible();
    await expect(sheet(page).locator('.insp-title')).toHaveText('embed');
    await expect(sheet(page).locator('[data-identity="/primitive"]')).toHaveValue(wanted);
  });

  test('adds a **site** of the composition a drill-in has open (§4.7’s other half)', async ({ page }) => {
    await open(page);
    await openModel(page);
    // §4.7's own table: the gesture "adds an instance (root canvas) or a site (drill-in)", and
    // which of the two is the tab the strip has current — never a choice made in the picker.
    await page.locator('[data-box="/compositions/decoder"] .n-name').click({ button: 'right' });
    await page.locator('.ctxmenu button[data-entry="canvas.drill-in"]').click();
    await expect(page.locator('.drill')).toBeVisible();
    await command(page, 'Model', 'model.add-instance');
    const dialog = page.locator('.dlg[role="dialog"]');
    const wanted = IDENTITIES.find((id) => id.startsWith('norm.rms@')) ?? '';
    await dialog.locator('[data-catalog-filter]').fill('norm');
    await dialog.locator(`[data-identity-choice="${wanted}"]`).click();
    await expect(page.locator('[data-box="/compositions/decoder/instances/rms"]')).toBeVisible();
    await expect(sheet(page).locator('[data-identity="/primitive"]')).toHaveValue(wanted);
  });

  test('is in the Model menu and in the palette, as every command of §4.4 is', async ({ page }) => {
    await open(page);
    await page.locator('nav.menu > div > button:text-is("Model")').click();
    await expect(
      page.locator('.menu-pop button[data-command="model.add-instance"]'),
    ).toBeVisible();
    await page.keyboard.press('Escape');
  });
});

test.describe('the accessibility of the field, its list and the confirm (§4.21)', () => {
  for (const theme of ['dark', 'light'] as const) {
    test(`finds no axe violation in the ${theme} theme`, async ({ page }) => {
      await open(page);
      await command(page, 'View', theme === 'dark' ? 'view.theme-dark' : 'view.theme-light');
      await openModel(page);
      await selectAttn(page);
      await field(page).fill('att');
      expect(await audit(page), 'the field and its list').toEqual([]);

      await field(page).fill('not.a.primitive@1.0.0');
      await field(page).press('Enter');
      await expect(page.locator('.dlg[role="dialog"]')).toBeVisible();
      expect(await audit(page), 'the confirm').toEqual([]);
      await page.locator('.dlg[role="dialog"] [data-cancel="identity"]').click();

      await command(page, 'Model', 'model.add-instance');
      await expect(page.locator('.dlg[role="dialog"]')).toBeVisible();
      expect(await audit(page), 'the picker').toEqual([]);
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
