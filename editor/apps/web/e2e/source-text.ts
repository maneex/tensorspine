import { expect, type Page } from '@playwright/test';

/**
 * The document's own bytes, read through `View ▸ JSON Source` — the one seam every suite uses.
 *
 * Feature 2.9 gave a model's tab its canvas (§4.7's "the default editor of a model") and the
 * source pane the tab of its own §4.2 puts it in, and from then on nine browser suites read what
 * a Save would write through that pane. Feature 2.17 replaced it with **Monaco**, which renders
 * only the lines that are on screen — so `innerText` is no longer the document, and the text is
 * read from the *model* instead, through the `window.monaco` Monaco's own distribution publishes
 * (`packages/ui/src/source/monaco.ts` keeps that convention for exactly this). It is still the
 * document as its file holds it, through the same serializer (D12).
 *
 * One module because there were nine copies of it and one change had to reach all nine.
 */
export async function sourceText(page: Page, path?: string): Promise<string> {
  await openSourceTab(page);
  const pane = page.locator(
    path === undefined ? '.src-pane[data-ready="true"]' : `.src-pane[data-path="${path}"][data-ready="true"]`,
  );
  await expect(pane).toBeVisible({ timeout: 60_000 });
  const where = path ?? ((await pane.getAttribute('data-path')) ?? '');
  const shown = await modelText(page, where);
  // The source is a *view* of the document, in a tab of its own (§4.2): it is closed again so
  // that what a suite counts afterwards is the documents it opened and not the readings of them.
  // `Ctrl+W` rather than the tab's ×, because with fifteen documents open the strip is wider than
  // the editor area and the × of the last one is behind the Properties region.
  await page.keyboard.press('Control+w');
  await expect(page.locator('.src-pane')).toHaveCount(0);
  return shown;
}

/** `View ▸ JSON Source` through the menu, which is the way a person reaches it (§4.4). */
async function openSourceTab(page: Page): Promise<void> {
  await page.locator('nav.menu > div > button:text-is("View")').click();
  await page.locator('.menu-pop button[data-command="view.json-source"]').click();
}

/** The whole text of one document's Monaco model. */
export async function modelText(page: Page, path: string): Promise<string> {
  return page.evaluate((where) => {
    const held = (
      window as unknown as {
        monaco?: { editor: { getModels(): { uri: { path: string }; getValue(): string }[] } };
      }
    ).monaco;
    return held?.editor.getModels().find((model) => model.uri.path === `/${where}`)?.getValue() ?? '';
  }, path);
}
