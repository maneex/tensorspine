import { expect, test } from '@playwright/test';

test('the built application loads with its three packages linked', async ({ page }) => {
  await page.goto('/');

  await expect(page).toHaveTitle('TensorSpine Editor');

  const root = page.locator('#root');
  await expect(root).toHaveAttribute('data-state', 'ready');
  await expect(root).toContainText('@tensorspine/lang');
  await expect(root).toContainText('@tensorspine/store');
  await expect(root).toContainText('@tensorspine/ui');
});
