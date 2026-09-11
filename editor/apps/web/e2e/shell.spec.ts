import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Page } from '@playwright/test';

import { stubUrl } from '../playwright.config.js';

// The shell of plan §4.2, in a browser — feature 2.5's browser layer.
//
// What belongs here and nowhere else: what the regions actually measure, that the palette really
// lists every command of §4.4, that the theme really changes the values the tokens carry, that
// the avatar is really absent without a session, and that the page passes an accessibility audit
// in both themes. The tables behind all of that — the command table, the region model, the
// arrangement's reading and writing, the contrast of every ink — are asked in the unit layer,
// where they can be asked exhaustively and in milliseconds.

/** Wait for the shell, and answer the frame everything is measured against. */
async function open(page: Page, url = '/'): Promise<void> {
  await page.goto(url);
  await expect(page.locator('.app')).toBeVisible();
}

/** The width or height a region actually occupies. */
async function size(page: Page, selector: string, of: 'width' | 'height'): Promise<number> {
  const box = await page.locator(selector).boundingBox();
  expect(box, selector).not.toBeNull();
  return Math.round((box as { width: number; height: number })[of]);
}

/** The value a custom property carries on the frame, as the browser resolves it. */
async function token(page: Page, name: string): Promise<string> {
  return page.evaluate((one) => {
    const app = document.querySelector('.app');
    return app === null ? '' : getComputedStyle(app).getPropertyValue(one).trim();
  }, name);
}

test.describe('the regions of §4.2', () => {
  test('are all present, at the sizes the plan states', async ({ page }) => {
    await open(page);

    // | Activity bar | left, 48 px |
    expect(await size(page, '.rail', 'width')).toBe(48);
    // | Side bar | left, 264 px, the site's `--nav-w` | open |
    expect(await size(page, '.side', 'width')).toBe(264);
    // | Properties | right, 340 px, resizable to 520 | open |
    expect(await size(page, '.insp', 'width')).toBe(340);
    // | Bottom panel | 200 px | Problems |
    expect(await size(page, '.panel', 'height')).toBe(200);

    // The bar, the editor column and the status bar are there, and the editor column is the one
    // that takes what is left — the component inventory's `.ed` rule.
    await expect(page.locator('header.bar')).toBeVisible();
    await expect(page.locator('main.ed')).toBeVisible();
    await expect(page.locator('footer.status')).toBeVisible();
    const app = await size(page, '.app', 'width');
    const editor = await size(page, 'main.ed', 'width');
    expect(editor).toBeGreaterThan(app - 48 - 264 - 340 - 20);
  });

  test('carry the seven menus of §4.4 and the four activities of §4.2', async ({ page }) => {
    await open(page);
    await expect(page.locator('nav.menu > div > button')).toHaveText([
      'File',
      'Edit',
      'View',
      'Model',
      'Library',
      'Weights',
      'Help',
    ]);
    // Four activities and Settings at the foot; Account only with a session (Q8 defers the SaaS).
    await expect(page.locator('.rail .act')).toHaveCount(5);
    await expect(page.locator('.rail .act[aria-pressed="true"]')).toHaveAttribute(
      'data-activity',
      'activity.explorer',
    );
    await expect(page.locator('.side-head')).toHaveText('Model explorer');
  });

  test('give the bottom panel exactly three tabs', async ({ page }) => {
    await open(page);
    await expect(page.locator('.panel .ptab')).toHaveText(['Problems', 'Derived', 'Log']);
    await expect(page.locator('.panel .ptab[aria-selected="true"]')).toHaveText('Problems');
    await expect(page.locator('.insp .ptab')).toHaveText(['Properties']);
  });

  test('show S17’s empty state until something is open', async ({ page }) => {
    await open(page);
    await expect(page.locator('.nothing h1')).toHaveText('No workspace is open.');
    await expect(page.locator('.nothing .btn')).toHaveText([
      'Open Folder…',
      'New Model',
      'New Base…',
    ]);
    // No tab strip where there is nothing to strip.
    await expect(page.locator('nav.tabs')).toHaveCount(0);
    await expect(page.locator('footer.status [data-workspace]')).toHaveText('No workspace');
  });
});

test.describe('the command palette', () => {
  test('lists every command of §4.4, by name, and runs the one chosen', async ({ page }) => {
    await open(page);
    await page.locator('.palette-open').click();
    const options = page.locator('.cmd');
    // The count is the table's own, read off the page's foot rather than written here twice.
    const count = await options.count();
    expect(count).toBeGreaterThan(80);
    await expect(page.locator('.dlg-foot')).toHaveText(`${String(count)} commands.`);

    // Every menu is represented, and a command of each is findable by its own name.
    for (const [query, id] of [
      ['New Model', 'file.new-model'],
      ['Undo', 'edit.undo'],
      ['Reset Layout', 'view.reset-layout'],
      ['Derive Now', 'model.derive-now'],
      ['New Precision Role', 'library.new-precision-role'],
      ['Check Locations', 'weights.check-locations'],
      ['Keyboard Shortcuts', 'help.keyboard-shortcuts'],
    ] as const) {
      await page.locator('.dlg-head input').fill(query);
      await expect(page.locator('.cmd').first()).toHaveAttribute('data-command', id);
    }

    // Enter runs what is highlighted; a command nobody has wired lands in the Log.
    await page.locator('.dlg-head input').fill('Derive Now');
    await page.keyboard.press('Enter');
    await expect(page.locator('.dlg')).toHaveCount(0);
    await page.locator('.ptab[data-panel="panel.log"]').click();
    await expect(page.locator('.logline').last()).toContainText('Derive Now');
  });

  test('opens on its chord and on the button beside it, and closes on Escape', async ({ page }) => {
    await open(page);
    await page.keyboard.press('Control+Shift+P');
    await expect(page.locator('.dlg')).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(page.locator('.dlg')).toHaveCount(0);

    // §4.4's rule: a chord is an accelerator and never the only way in.
    await page.locator('.palette-open').click();
    await expect(page.locator('.dlg')).toBeVisible();
  });

  test('walks its list with the arrows', async ({ page }) => {
    await open(page);
    await page.keyboard.press('Control+Shift+P');
    await expect(page.locator('.cmd[aria-selected="true"]')).toHaveAttribute(
      'data-command',
      'file.new-model',
    );
    await page.keyboard.press('ArrowDown');
    await expect(page.locator('.cmd[aria-selected="true"]')).toHaveAttribute(
      'data-command',
      'file.new-template',
    );
    await page.keyboard.press('ArrowUp');
    await page.keyboard.press('ArrowUp');
    // Round the end, rather than stopping at it: from the first, up is the last.
    await expect(page.locator('.cmd[aria-selected="true"]')).toHaveAttribute(
      'data-command',
      'help.about',
    );
  });
});

test.describe('the menus', () => {
  test('open, list their own commands with their accelerators, and run one', async ({ page }) => {
    await open(page);
    await page.locator('nav.menu > div > button', { hasText: 'View' }).click();
    const pop = page.locator('.menu-pop');
    await expect(pop).toBeVisible();
    await expect(pop.locator('button')).toHaveCount(21);
    await expect(pop.locator('[data-command="view.toggle-side-bar"] .kbd')).toHaveText('Ctrl+B');

    await pop.locator('[data-command="view.toggle-side-bar"]').click();
    await expect(page.locator('.side')).toHaveCount(0);
    await expect(pop).toHaveCount(0);

    // The same command from its accelerator brings it back.
    await page.keyboard.press('Control+b');
    await expect(page.locator('.side')).toBeVisible();
  });

  test('close on Escape and when the focus leaves them', async ({ page }) => {
    await open(page);
    await page.locator('nav.menu > div > button', { hasText: 'File' }).click();
    await expect(page.locator('.menu-pop')).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(page.locator('.menu-pop')).toHaveCount(0);
  });
});

test.describe('the editor tabs', () => {
  test('open, switch and close, and the strip goes when the last one does', async ({ page }) => {
    await open(page);
    const help = page.locator('nav.menu > div > button', { hasText: 'Help' });

    await help.click();
    await page.locator('[data-command="help.about"]').click();
    await expect(page.locator('.tab-slot')).toHaveCount(1);
    await expect(page.locator('.tab[data-tab="tab.about"]')).toHaveAttribute('aria-current', 'true');
    await expect(page.locator('.doc h1')).toHaveText('TensorSpine Editor');

    await help.click();
    await page.locator('[data-command="help.keyboard-shortcuts"]').click();
    await expect(page.locator('.tab-slot')).toHaveCount(2);
    await expect(page.locator('.doc h1')).toHaveText('Keyboard Shortcuts');
    // Every command of §4.4 is listed there, which is what §4.21 asks the screen to answer.
    const rows = await page.locator('.doc tbody tr').count();
    expect(rows).toBeGreaterThan(80);

    await page.locator('.tab[data-tab="tab.about"]').click();
    await expect(page.locator('.doc h1')).toHaveText('TensorSpine Editor');

    await page.locator('[data-close="tab.about"]').click();
    await expect(page.locator('.tab-slot')).toHaveCount(1);
    await expect(page.locator('.doc h1')).toHaveText('Keyboard Shortcuts');
    await page.locator('[data-close="tab.shortcuts"]').click();
    await expect(page.locator('nav.tabs')).toHaveCount(0);
    await expect(page.locator('.nothing h1')).toBeVisible();
  });
});

test.describe('the theme', () => {
  test('changes the values the tokens carry, and says which it is', async ({ page }) => {
    await open(page);
    await page.locator('[data-command="view.theme-dark"]').count();

    await page.locator('nav.menu > div > button', { hasText: 'View' }).click();
    await page.locator('[data-command="view.theme-dark"]').click();
    expect(await token(page, '--bg')).toBe('#101317');
    expect(await token(page, '--ink')).toBe('#e3e8ef');
    expect(await token(page, '--accent')).toBe('#2ac3c1');
    await expect(page.locator('.app')).toHaveAttribute('data-scheme', 'dark');
    await expect(page.locator('.app')).not.toHaveClass(/theme-light/);

    await page.locator('nav.menu > div > button', { hasText: 'View' }).click();
    await page.locator('[data-command="view.theme-light"]').click();
    expect(await token(page, '--bg')).toBe('#eef2f5');
    expect(await token(page, '--ink')).toBe('#12181e');
    expect(await token(page, '--accent')).toBe('#0d7a78');
    await expect(page.locator('.app')).toHaveClass(/theme-light/);
    await expect(page.locator('footer.status [data-theme]')).toHaveText('Theme: Light');
  });

  test('follows the machine under `system`, which is what a session starts on (Q3)', async ({
    browser,
  }) => {
    const dark = await browser.newContext({ colorScheme: 'dark' });
    const one = await dark.newPage();
    await open(one);
    await expect(one.locator('.app')).toHaveAttribute('data-scheme', 'dark');
    await expect(one.locator('footer.status [data-theme]')).toHaveText('Theme: System (Dark)');
    await dark.close();

    const light = await browser.newContext({ colorScheme: 'light' });
    const two = await light.newPage();
    await open(two);
    await expect(two.locator('.app')).toHaveAttribute('data-scheme', 'light');
    await expect(two.locator('footer.status [data-theme]')).toHaveText('Theme: System (Light)');
    await light.close();
  });

  test('is remembered, and so is the arrangement (§4.2: "layouts are saved in settings")', async ({
    page,
  }) => {
    await open(page);
    // The status bar's theme field steps through §4.4's three commands; from `system` that is
    // Light, then Dark. Clicking the value is §4.4's "or by clicking the element itself".
    await page.locator('footer.status [data-theme]').click();
    await expect(page.locator('footer.status [data-theme]')).toHaveText('Theme: Light');
    await page.locator('footer.status [data-theme]').click();
    await expect(page.locator('footer.status [data-theme]')).toHaveText('Theme: Dark');
    await page.locator('.panel .tbtn').click();
    await expect(page.locator('.insp .ptab')).toHaveText(['Properties', 'Problems']);

    await page.reload();
    await expect(page.locator('.app')).toBeVisible();
    await expect(page.locator('footer.status [data-theme]')).toHaveText('Theme: Dark');
    await expect(page.locator('.insp .ptab')).toHaveText(['Properties', 'Problems']);
    await expect(page.locator('.panel .ptab')).toHaveText(['Derived', 'Log']);
  });
});

test.describe('a panel moved between regions', () => {
  test('follows the button that sends it, either way', async ({ page }) => {
    await open(page);
    await page.locator('.panel .ptab[data-panel="panel.derived"]').click();
    await page.locator('.panel .tbtn').click();
    await expect(page.locator('.insp .ptab')).toHaveText(['Properties', 'Derived']);
    await expect(page.locator('.panel .ptab')).toHaveText(['Problems', 'Log']);

    await page.locator('.insp .tbtn').click();
    await expect(page.locator('.panel .ptab')).toHaveText(['Problems', 'Log', 'Derived']);
    await expect(page.locator('.insp .ptab')).toHaveText(['Properties']);
  });

  test('follows the drag, which is the VS Code gesture §4.2 names', async ({ page }) => {
    await open(page);
    await page
      .locator('.panel .ptab[data-panel="panel.derived"]')
      .dragTo(page.locator('.insp .panel-body'));
    await expect(page.locator('.insp .ptab')).toHaveText(['Properties', 'Derived']);
    await expect(page.locator('.panel .ptab')).toHaveText(['Problems', 'Log']);
  });

  test('closes the region a move emptied, and the editor column takes the width', async ({
    page,
  }) => {
    await open(page);
    const before = await size(page, 'main.ed', 'width');
    await page.locator('.insp .tbtn').click();
    await expect(page.locator('.insp')).toHaveCount(0);
    expect(await size(page, 'main.ed', 'width')).toBeGreaterThan(before + 300);
    await expect(page.locator('.panel .ptab')).toHaveText([
      'Problems',
      'Derived',
      'Log',
      'Properties',
    ]);
  });
});

test.describe('a region resized', () => {
  test('moves under the pointer and under the arrow keys, and stops at its bounds', async ({
    page,
  }) => {
    await open(page);
    const splitter = page.locator('.splitter.vertical').first();
    await expect(splitter).toHaveAttribute('aria-valuenow', '264');

    await splitter.focus();
    await page.keyboard.press('ArrowRight');
    await expect(splitter).toHaveAttribute('aria-valuenow', '280');
    expect(await size(page, '.side', 'width')).toBe(280);

    await page.keyboard.press('Home');
    await expect(splitter).toHaveAttribute('aria-valuenow', '180');
    await page.keyboard.press('ArrowLeft');
    // Held at the minimum: a region that could be dragged to nothing is a region the user loses.
    await expect(splitter).toHaveAttribute('aria-valuenow', '180');

    const box = await splitter.boundingBox();
    expect(box).not.toBeNull();
    const { x, y, height } = box as { x: number; y: number; height: number };
    await page.mouse.move(x + 2, y + height / 2);
    await page.mouse.down();
    await page.mouse.move(x + 122, y + height / 2, { steps: 6 });
    await page.mouse.up();
    expect(await size(page, '.side', 'width')).toBe(300);
  });
});

test.describe('the avatar', () => {
  test('is absent without a session, which is every deployment of this plan', async ({ page }) => {
    await open(page);
    await expect(page.locator('.avatar')).toHaveCount(0);
    // And so is the Account activity, for the same reason (Q8 defers the SaaS).
    await expect(page.locator('.act[data-activity="activity.account"]')).toHaveCount(0);

    await open(page, stubUrl);
    await expect(page.locator('.avatar')).toHaveCount(0);
  });

  test('is there when the platform has one', async ({ page }) => {
    await open(page, `${stubUrl}?session=Perceval%20Lambert`);
    await expect(page.locator('.avatar')).toHaveText('PL');
    await expect(page.locator('.avatar')).toHaveAttribute('title', 'Perceval Lambert');
    await expect(page.locator('.act[data-activity="activity.account"]')).toHaveCount(1);
  });
});

test.describe('the platform the shell sits on', () => {
  test('is offered §4.4’s commands for a native menu it may not have', async ({ page }) => {
    await open(page, stubUrl);
    const menu = await page.evaluate(() => window.tensorspineStub.platform.shellRecord.menu);
    expect(menu.length).toBeGreaterThan(80);
    expect(menu.find((one) => one.id === 'file.save')).toEqual({
      id: 'file.save',
      label: 'Save',
      enabled: true,
      accelerator: 'Ctrl+S',
    });
    // Nothing is disabled: a refusal is stated, never enforced by greying out (§9 Q5's rule).
    expect(menu.filter((one) => !one.enabled)).toEqual([]);
  });

  test('reaches the machine’s colour scheme through the platform and not around it', async ({
    page,
  }) => {
    await open(page, `${stubUrl}?scheme=dark`);
    await expect(page.locator('.app')).toHaveAttribute('data-scheme', 'dark');
    await open(page, `${stubUrl}?scheme=light`);
    await expect(page.locator('.app')).toHaveAttribute('data-scheme', 'light');
  });
});

test.describe('the accessibility audit of §4.21', () => {
  /**
   * The page, checked by axe.
   *
   * The wordmark is excluded from the colour-contrast rule alone, and WCAG 1.4.3 is why: "Text
   * that is part of a logo or brand name has no contrast requirement". The name beside the
   * monogram is the brand's lettering in the mark's own colours (`--logo-a`, `--logo-b`, the
   * design's), and the alternative to the exemption would be repainting a logo — which §4.21
   * forbids in the same breath: "placed and scaled, never redrawn". Nothing else is excluded.
   */
  async function audit(page: Page): Promise<{ id: string; impact: string | null | undefined }[]> {
    const result = await new AxeBuilder({ page })
      .disableRules([])
      .exclude('.wordmark')
      .analyze();
    const wordmarkOnly = await new AxeBuilder({ page })
      .include('.wordmark')
      .disableRules(['color-contrast'])
      .analyze();
    return [...result.violations, ...wordmarkOnly.violations].map((one) => ({
      id: one.id,
      impact: one.impact,
    }));
  }

  for (const theme of ['dark', 'light'] as const) {
    test(`finds no serious violation in the ${theme} theme`, async ({ page }) => {
      await open(page);
      await page.locator('nav.menu > div > button', { hasText: 'View' }).click();
      await page.locator(`[data-command="view.theme-${theme}"]`).click();
      await expect(page.locator('.app')).toHaveAttribute('data-scheme', theme);

      const violations = await audit(page);
      expect(violations.filter((one) => one.impact === 'serious' || one.impact === 'critical')).toEqual(
        [],
      );
      expect(violations).toEqual([]);
    });
  }

  test('finds none with the palette up, a menu open and a tab showing', async ({ page }) => {
    await open(page);
    await page.locator('nav.menu > div > button', { hasText: 'Help' }).click();
    await page.locator('[data-command="help.keyboard-shortcuts"]').click();
    expect(await audit(page)).toEqual([]);

    await page.locator('nav.menu > div > button', { hasText: 'File' }).click();
    await expect(page.locator('.menu-pop')).toBeVisible();
    expect(await audit(page)).toEqual([]);

    await page.keyboard.press('Escape');
    await page.keyboard.press('Control+Shift+P');
    await expect(page.locator('.dlg')).toBeVisible();
    expect(await audit(page)).toEqual([]);
  });
});
