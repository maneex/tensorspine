import { readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import AxeBuilder from '@axe-core/playwright';
import { expect, test, type Page } from '@playwright/test';

import { modelText } from './source-text.js';

/**
 * The JSON source view — feature 2.17, plan §4.10 and §3, artboard S16.
 *
 * The five claims the feature's own block names, each of which needs a real editor, a real worker
 * and a real core to be worth anything:
 *
 *   - **editing `heads` in the source updates the sheet** — the two-way sync's first direction,
 *     driven by real keystrokes over a selection, with the sheet's own effective value as the
 *     witness that the *tree* moved and not only the text;
 *   - **typing an unknown member shows the schema problem at its range and the banner** — the
 *     marker is Monaco's, its range is the core's span of the member, and its message is Ajv's
 *     own line (Monaco's own JSON validation is off: the verdicts are the core's, D4);
 *   - **the canvas keeps its last state** — §4.10's "last drawable state", with the note that
 *     says so and the boxes still drawn;
 *   - **Show in JSON from `final_n` reveals its range** — the selection covers the member and its
 *     value, as S16 highlights it;
 *   - **a lexeme `1e-05` survives a source edit elsewhere** — D12 through the source view: the
 *     bytes after the edit are the file's, with the edit and nothing else.
 *
 * Beside them: the schema really is attached (completion offers what the grammar declares), the
 * other direction of the sync (a canvas gesture rewrites the text minimally), §3's confirmed save
 * of an off-schema document, and axe over the pane in both themes.
 *
 * Everything that is a question about *what the editor does* rather than about a browser is asked
 * in `packages/ui/test/source/` and in `packages/ui/test/documents/store.test.ts`.
 */

const repository = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', '..');
const MODEL = 'models/llama3-8b.json';
const modelFile = readFileSync(join(repository, 'data/models/llama3-8b.json'), 'utf8');

/**
 * The pane needs room: S16 draws it at 872 px beside a canvas, in a 1472 px region. The clipboard
 * is granted because one of the gestures below is a paste — see {@link paste}.
 */
test.use({
  viewport: { width: 1500, height: 950 },
  permissions: ['clipboard-read', 'clipboard-write'],
});

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

/** The same, and the products with it: the status bar's figures read as stale until they land. */
async function openDerived(page: Page): Promise<void> {
  await openModel(page);
  await expect(page.locator('.bar [data-pill="derivation"]')).toHaveText('derived · fresh', {
    timeout: 60_000,
  });
}

/** `View ▸ JSON Source` (Ctrl+Shift+J, "opens beside"), waited for until Monaco is up. */
async function openSource(page: Page): Promise<void> {
  await command(page, 'View', 'view.json-source');
  await expect(page.locator(`.src-pane[data-path="${MODEL}"][data-ready="true"]`)).toBeVisible({
    timeout: 60_000,
  });
}

/** Bring the model's own tab back to the front. */
async function showCanvas(page: Page): Promise<void> {
  await page.locator(`.tabs .tab[data-tab$="${MODEL}"]`).first().click();
  await expect(page.locator(`.gcanvas[data-canvas="${MODEL}"]`)).toBeVisible();
}

/** The whole text of the document's model — what a virtualised editor does not render. */
async function sourceText(page: Page): Promise<string> {
  return modelText(page, MODEL);
}

/**
 * Where the reader is: the scroll, and the *line they are on* — which a whole-text write loses.
 *
 * The line's **text** and not its number, because an edit above the caret moves every line below
 * it by one and a reader who stayed where they were is a reader looking at the same words.
 */
async function where(page: Page): Promise<{ top: number; line: number; text: string }> {
  return page.evaluate(() => {
    const held = (
      window as unknown as {
        monaco?: {
          editor: {
            getEditors(): {
              getScrollTop(): number;
              getPosition(): { lineNumber: number } | null;
              getModel(): { getLineContent(line: number): string } | null;
            }[];
          };
        };
      }
    ).monaco;
    const editor = held?.editor.getEditors()[0];
    const line = editor?.getPosition()?.lineNumber ?? 0;
    return {
      top: editor?.getScrollTop() ?? 0,
      line,
      text: line === 0 ? '' : (editor?.getModel()?.getLineContent(line) ?? ''),
    };
  });
}

/** What the editor has selected, as text — which is what a reveal is asserted on. */
async function selectedText(page: Page): Promise<string> {
  return page.evaluate(() => {
    const held = (
      window as unknown as {
        monaco?: {
          editor: {
            getEditors(): {
              getSelection(): unknown;
              getModel(): { getValueInRange(range: unknown): string } | null;
            }[];
          };
        };
      }
    ).monaco;
    const editor = held?.editor.getEditors()[0];
    const selection = editor?.getSelection();
    if (editor === undefined || selection === undefined) return '';
    return editor.getModel()?.getValueInRange(selection) ?? '';
  });
}

/**
 * Put the editor's selection over a stretch of its own text, or the caret after it, and focus it.
 *
 * `after` is an anchor that tells one occurrence from another — the quantity `heads` and the
 * argument of the same name are written at two indentations — and `what` is the stretch. This is
 * a *selection*, made the way a reader makes one; what is typed into it is typed by the keyboard.
 */
async function selectRange(page: Page, after: string, what: string, collapse = false): Promise<void> {
  await page.locator('.src-pane .monaco-editor').first().click();
  const found = await page.evaluate(
    ({ where, anchor, text, collapsed }) => {
      const held = (
        window as unknown as {
          monaco?: {
            editor: {
              getModels(): {
                uri: { path: string };
                getValue(): string;
                getPositionAt(offset: number): { lineNumber: number; column: number };
              }[];
              getEditors(): {
                getModel(): { uri: { path: string } } | null;
                setSelection(range: unknown): void;
                revealRangeInCenterIfOutsideViewport(range: unknown): void;
                focus(): void;
              }[];
            };
          };
        }
      ).monaco;
      const model = held?.editor.getModels().find((one) => one.uri.path === `/${where}`);
      const editor = held?.editor.getEditors().find((one) => one.getModel()?.uri.path === `/${where}`);
      if (model === undefined || editor === undefined) return false;
      const from = model.getValue().indexOf(anchor);
      if (from < 0) return false;
      const at = model.getValue().indexOf(text, from);
      if (at < 0) return false;
      const start = model.getPositionAt(collapsed ? at + text.length : at);
      const end = model.getPositionAt(at + text.length);
      const range = {
        startLineNumber: start.lineNumber,
        startColumn: start.column,
        endLineNumber: end.lineNumber,
        endColumn: end.column,
      };
      editor.setSelection(range);
      editor.revealRangeInCenterIfOutsideViewport(range);
      editor.focus();
      return true;
    },
    { where: MODEL, anchor: after, text: what, collapsed: collapse },
  );
  expect(found, `${after} … ${what}`).toBe(true);
}

/**
 * Paste a text at the caret — the gesture that puts a member into a document.
 *
 * A paste rather than a keystroke for one member and not for the numbers below, and the reason is
 * the editor's own behaviour rather than the suite's convenience: typing `"` with something
 * selected *surrounds* the selection (Monaco's `autoSurround`, on by default and right), and
 * typing one into an object opens the schema's own property completion. Both are what a person
 * gets and neither is what a scripted keystroke can steer. A paste is a real gesture with an
 * exact result, and what is under test is the pane's answer to an edit.
 */
async function paste(page: Page, text: string): Promise<void> {
  await page.evaluate((what) => navigator.clipboard.writeText(what), text);
  await page.keyboard.press('Control+V');
}

test.describe('the two-way sync (§4.10)', () => {
  test('an edit typed in the source reaches the tree, and the sheet shows it', async ({ page }) => {
    await open(page);
    await openModel(page);
    await openSource(page);

    // The quantity `heads`, told from the argument of the same name by the shape above it, and
    // `head_dim`, which declares `floor_divide(d, heads)` and has to follow it (V11). Selected
    // and typed over, which is how a person changes a number.
    await selectRange(page, '"heads": {\n      "type"', '32');
    await page.keyboard.type('16');
    await selectRange(page, '"head_dim": {', '128');
    await page.keyboard.type('256');

    // The text is the reader's; the tree is the document's. The sheet is what says the tree moved.
    await expect(page.locator('.src-pane .src-bad')).toHaveCount(0, { timeout: 30_000 });
    await showCanvas(page);
    await page.locator('[data-fold="/compositions/decoder"]').click();
    await page.locator('.gcanvas [data-name="/compositions/decoder/instances/attn"]').click();
    const sheet = page.locator('.insp .panel-body');
    await expect(sheet.locator('.insp-title')).toHaveText('attn');
    await expect(sheet.locator('[data-argument="heads"] [data-effective]')).toHaveText('16', {
      timeout: 30_000,
    });
    await expect(sheet.locator('[data-argument="head_dim"] [data-effective]')).toHaveText('256');
    // And the document the tree now holds is one the core still validates.
    await expect(page.locator('.bar [data-pill="validation"]')).toHaveText('no problems', {
      timeout: 60_000,
    });
  });

  test('a lexeme the edit did not touch is written exactly as the file writes it', async ({
    page,
  }) => {
    // D12 through the source view: `1e-05` is not `0.00001`, and the bytes after an edit
    // elsewhere are the file's with the edit and nothing else.
    await open(page);
    await openModel(page);
    await openSource(page);
    expect(await sourceText(page)).toBe(modelFile);

    await selectRange(page, '"heads": {\n      "type"', '32');
    await page.keyboard.type('16');
    await expect(page.locator('.bar [data-pill="validation"]')).not.toHaveText('checking…', {
      timeout: 30_000,
    });

    // The pane holds what was typed; the *document* holds the serializer's rendering of it, and
    // the two are the same bytes here because the edit is canonical. A canvas gesture is what
    // makes the document write itself back, so the assertion is made on both.
    expect(await sourceText(page)).toContain('"value": 1e-05');
    await showCanvas(page);
    await openSource(page);
    const after = await sourceText(page);
    expect(after).toContain('"value": 1e-05');
    expect(after).toBe(modelFile.replace('"value": 32,\n', 'KEEP').replace('"value": 32', '"value": 16').replace('KEEP', '"value": 32,\n'));
  });

  test('an edit made elsewhere rewrites the text under the reader’s caret', async ({ page }) => {
    // The other direction: "an edit on the canvas or in a form updates the source text minimally".
    // *Minimally* is what this asserts: the reader is a screenful down the file when the document
    // moves, and a pane that replaced its whole text would put them back at the top with the
    // caret at (1, 1). The edit is made while the pane is showing — `Edit ▸ Undo` of the reader's
    // own source edit — because that is the one gesture that moves the tree without unmounting it.
    await open(page);
    await openModel(page);
    await openSource(page);

    // A source edit at the top of the file, then the reader looks at the bottom of it.
    await selectRange(page, '{\n  "schema"', '"model": "llama3-8b",', true);
    await paste(page, '\n  "kernels": 1,');
    await expect(page.locator('[data-source-banner="off-grammar"]')).toBeVisible({ timeout: 30_000 });
    await selectRange(page, '"interfaces"', '"outputs"');
    const before = await where(page);
    expect(before.top).toBeGreaterThan(0);

    await command(page, 'Edit', 'edit.undo');
    await expect
      .poll(async () => (await sourceText(page)).includes('"kernels"'), { timeout: 30_000 })
      .toBe(false);
    expect(await sourceText(page)).toBe(modelFile);
    const after = await where(page);
    // The reader is on the same line of the document, and within a line of where they were: a
    // pane that had written its whole text would be at the top of the file with the caret at
    // (1, 1). The line *number* moved by one, because the edit took a line out above it.
    expect(after.text).toBe(before.text);
    expect(after.line).toBe(before.line - 1);
    expect(after.top).toBeGreaterThan(1000);
    expect(Math.abs(after.top - before.top)).toBeLessThan(40);
  });

  test('puts the reader back where they were when the tab comes round again', async ({ page }) => {
    await open(page);
    await openModel(page);
    await openSource(page);
    await selectRange(page, '"interfaces"', '"outputs"');
    const before = await where(page);
    expect(before.top).toBeGreaterThan(0);
    await showCanvas(page);
    await openSource(page);
    expect(await where(page)).toEqual(before);
  });
});

test.describe('a source that leaves the grammar (§3, §4.10)', () => {
  /** Put a member the grammar does not admit into the document, after the model's name. */
  async function typeUnknownMember(page: Page): Promise<void> {
    await selectRange(page, '{\n  "schema"', '"model": "llama3-8b",', true);
    await paste(page, '\n  "kernels": 1,');
  }

  test('shows the schema problem at its range, and the banner', async ({ page }) => {
    await open(page);
    await openModel(page);
    await openSource(page);
    await typeUnknownMember(page);

    // The banner §4.10 asks for, in the words S16 draws.
    const banner = page.locator('[data-source-banner="off-grammar"]');
    await expect(banner).toBeVisible({ timeout: 30_000 });
    await expect(banner.locator('b')).toHaveText('The source is off the grammar.');

    // The marker: Ajv's own line, at the member the refusal is about — not at the document, which
    // is where `jsonschema` puts the *place* of an `additionalProperties` refusal.
    const marker = await page.evaluate(() => {
      const held = (
        window as unknown as {
          monaco?: {
            editor: {
              getModels(): { uri: { path: string }; getValueInRange(range: unknown): string }[];
              getModelMarkers(filter: { owner: string }): {
                message: string;
                startLineNumber: number;
                startColumn: number;
                endLineNumber: number;
                endColumn: number;
              }[];
            };
          };
        }
      ).monaco;
      const markers = held?.editor.getModelMarkers({ owner: 'tensorspine' }) ?? [];
      const first = markers[0];
      if (first === undefined || held === undefined) return null;
      const model = held.editor.getModels()[0];
      return { message: first.message, text: model?.getValueInRange(first) ?? '' };
    });
    expect(marker?.message).toContain("'kernels' was unexpected");
    expect(marker?.text).toBe('"kernels": 1');

    // The same line is the Problems panel's, from the same stage: one verdict, two renderings.
    await expect(page.locator('.panel .prow .msg').first()).toContainText("'kernels' was unexpected");
    await expect(page.locator('.src-pane .src-bad')).toHaveText('1 problem');
  });

  test('keeps a text it could not read, and marks the refusal where the parser refused it', async ({
    page,
  }) => {
    // §3's other half, and the one the block's own wording does not reach: a text that is not
    // JSON at all is **not taken into the tree** — the document keeps what it had — and the
    // refusal is the core's own, in CPython's words and at CPython's own character.
    await open(page);
    await openModel(page);
    await openSource(page);
    const before = await sourceText(page);
    await selectRange(page, '{\n  "schema"', '"model": "llama3-8b",', true);
    await paste(page, '\n  "kernels":');

    const banner = page.locator('[data-source-banner="pending"]');
    await expect(banner).toBeVisible({ timeout: 30_000 });
    await expect(banner.locator('b')).toHaveText('The JSON source is not a document yet.');
    const marker = await page.evaluate(() => {
      const held = (
        window as unknown as {
          monaco?: {
            editor: {
              getModelMarkers(filter: { owner: string }): {
                message: string;
                startLineNumber: number;
                startColumn: number;
              }[];
            };
          };
        }
      ).monaco;
      const first = held?.editor.getModelMarkers({ owner: 'tensorspine' })[0];
      return first === undefined
        ? null
        : { message: first.message, line: first.startLineNumber, column: first.startColumn };
    });
    expect(marker?.message).toContain('[V12]');
    expect(marker?.message).toContain(
      `line ${String(marker?.line ?? 0)} column ${String(marker?.column ?? 0)}`,
    );
    // And the canvas is still drawing the document the tree still holds.
    await showCanvas(page);
    await expect(page.locator('.canvas.graph > .canvas-note').first()).toHaveText(
      /^last drawable state · /,
    );
    await openSource(page);
    expect((await sourceText(page)).startsWith('{\n  "schema"')).toBe(true);
    expect(before).toBe(modelFile);
  });

  test('leaves the canvas its last drawable state', async ({ page }) => {
    await open(page);
    await openModel(page);
    await openSource(page);
    await typeUnknownMember(page);
    await expect(page.locator('[data-source-banner="off-grammar"]')).toBeVisible({ timeout: 30_000 });

    await showCanvas(page);
    // The boxes llama3-8b draws are still drawn, and the note says what they are.
    await expect(page.locator('.canvas.graph > .canvas-note').first()).toHaveText(
      /^last drawable state · /,
    );
    await expect(page.locator('.gcanvas [data-name="/instances/embed"]')).toBeVisible();
    await expect(page.locator('.gcanvas [data-name="/instances/final_n"]')).toBeVisible();
    await expect(page.locator('.gcanvas [data-fold="/compositions/decoder"]')).toBeVisible();
    await expect(page.locator('[data-source-banner="off-grammar"]')).toBeVisible();
  });

  test('saves an off-schema document only when the confirmation is taken', async ({ page }) => {
    await open(page);
    await openModel(page);
    await openSource(page);
    await typeUnknownMember(page);
    await expect(page.locator('[data-source-banner="off-grammar"]')).toBeVisible({ timeout: 30_000 });

    // The Examples workspace is read-only, so a Save hands the bytes over as a download (S18).
    let asked = '';
    page.on('dialog', (dialog) => {
      asked = dialog.message();
      void dialog.dismiss();
    });
    await showCanvas(page);
    await page.locator('.doc-foot .btn.pri').click();
    await expect.poll(() => asked, { timeout: 30_000 }).toContain('off the grammar');
    page.removeAllListeners('dialog');

    page.on('dialog', (dialog) => {
      void dialog.accept();
    });
    const [download] = await Promise.all([
      page.waitForEvent('download'),
      page.locator('.doc-foot .btn.pri').click(),
    ]);
    expect(download.suggestedFilename()).toBe('llama3-8b.json');
    expect(readFileSync(await download.path(), 'utf8')).toContain('"kernels": 1');
  });
});

test.describe('the keyboard, while the reader is typing (§4.4)', () => {
  test('leaves every accelerator that is not marked `whileTyping` to the editor', async ({
    page,
  }) => {
    // Monaco 0.56 takes its input through the EditContext API where a browser has one, so what
    // holds the focus is a `div[role="textbox"]` and not a textarea: the shell's `isTyping` read
    // the tag and answered *false*, which fired `Delete`, `F2` and the rest under the reader's
    // typing. What is asked here is the consequence — a `Delete` inside the source view deletes a
    // character and not the selected instance of the document.
    await open(page);
    await openModel(page);
    await page.locator('.gcanvas [data-name="/instances/final_n"]').click();
    await expect(page.locator('.insp .panel-body .insp-title')).toHaveText('final_n');
    await openSource(page);

    await selectRange(page, '"heads": {\n      "type"', '32');
    await page.keyboard.press('Delete');
    // No confirmation went up — `Edit ▸ Delete` would have asked what the cascade removes —
    // and the text lost the two characters the selection held.
    await expect(page.locator('.dlg')).toHaveCount(0);
    await expect.poll(async () => sourceText(page), { timeout: 30_000 }).toContain('"value": \n');
    // The document still has the instance the canvas had selected.
    expect(await sourceText(page)).toContain('"final_n"');
  });
});

test.describe('what a Save writes (§4.3, §4.10)', () => {
  test('is what the reader is looking at, even one keystroke after the edit', async ({ page }) => {
    // The path a flush exists for: `Ctrl+S` while the focus is in the editor, before the three
    // hundred milliseconds are up. Every path that reads a document's bytes goes through the
    // pane's own flush first, so the file gets what is on the screen and not what the document
    // held a moment ago.
    await open(page);
    await openModel(page);
    await openSource(page);
    await selectRange(page, '"heads": {\n      "type"', '32');
    await page.keyboard.type('16');
    const [download] = await Promise.all([
      page.waitForEvent('download'),
      page.keyboard.press('Control+S'),
    ]);
    expect(download.suggestedFilename()).toBe('llama3-8b.json');
    expect(readFileSync(await download.path(), 'utf8')).toBe(
      modelFile.replace('"value": 32,\n', 'KEEP').replace('"value": 32', '"value": 16').replace('KEEP', '"value": 32,\n'),
    );
  });
});

test.describe('Show in JSON (§4.10, §4.17)', () => {
  test('reveals the range of the selection — the member and its value, as S16 draws it', async ({
    page,
  }) => {
    await open(page);
    await openModel(page);
    await page.locator('.gcanvas [data-name="/instances/final_n"]').click();
    await expect(page.locator('.insp .panel-body .insp-title')).toHaveText('final_n');
    await openSource(page);
    await expect.poll(async () => selectedText(page), { timeout: 30_000 }).toMatch(/^"final_n": \{/);
    const shown = await selectedText(page);
    expect(shown.endsWith('}')).toBe(true);
    expect(shown).toContain('"primitive"');
  });

  test('is what a Problems row asks for too — §4.17’s third navigation', async ({ page }) => {
    await open(page);
    await openModel(page);
    await openSource(page);
    // A row about a place of the document: the editor's own notice about `kv_heads` is
    // deepseek's, so a refusal is made here instead, and the row is clicked in the panel.
    await selectRange(page, '"heads": {\n      "type"', '32');
    await page.keyboard.type('5');
    await expect(page.locator('.panel .prow').first()).toBeVisible({ timeout: 60_000 });
    await page.locator('.panel .prow .msg').first().click();
    await expect.poll(async () => selectedText(page), { timeout: 30_000 }).not.toBe('');
  });
});

test.describe('the schema Monaco is given (§4.10, §1)', () => {
  test('completes the members the grammar declares', async ({ page }) => {
    // "with the model schema attached (completion, hover from `description`s)". Nothing here
    // *validates* with it — the verdicts are the core's (D4) — but the help is the schema's,
    // which is §1's own sentence one component along.
    await open(page);
    await openModel(page);
    await openSource(page);
    await selectRange(page, '{\n  "schema"', '"model"');
    await page.keyboard.press('Control+Space');
    const suggestions = page.locator('.monaco-editor .suggest-widget .monaco-list-row');
    await expect(suggestions.first()).toBeVisible({ timeout: 60_000 });
    const labels = await suggestions.allInnerTexts();
    expect(labels.join(' ')).toContain('model');
    await page.keyboard.press('Escape');
  });
});

test.describe('the accessibility of what this feature draws (§4.21)', () => {
  /**
   * The page, checked by axe — feature 2.5's own audit, with its one exemption and one more.
   *
   * The **wordmark** is excluded from the colour-contrast rule alone (WCAG 1.4.3: "Text that is
   * part of a logo or brand name has no contrast requirement"), which is feature 2.5's.
   *
   * The **suggestion list** is excluded from `aria-required-children` and `aria-required-parent`,
   * and from nothing else. Measured: Monaco 0.56 draws the completion list as
   * `.monaco-list[role="listbox"]` whose rows carry `role="listitem"` — a `listbox` wants
   * `option` children and a `listitem` wants a `list` parent, so axe reports both, critically and
   * correctly. It is the widget's own markup and there is no option that changes it; fixing it
   * would mean re-drawing a third party's list from the outside. Every other rule runs over it,
   * the colour-contrast one included — and that is not an idle statement: it found the matched
   * part of a suggestion painted in Monaco's own white at 1.22:1 on `--bg-tint`, which is bound
   * to the ramp now (`source/monaco.ts`).
   */
  async function audit(page: Page): Promise<string[]> {
    const result = await new AxeBuilder({ page })
      .exclude('.wordmark')
      .exclude('.suggest-widget')
      .analyze();
    // Only where one is open: axe refuses an `include` that matches nothing.
    const widget =
      (await page.locator('.suggest-widget .monaco-list-row').count()) === 0
        ? { violations: [] }
        : await new AxeBuilder({ page })
            .include('.suggest-widget')
            .disableRules(['aria-required-children', 'aria-required-parent'])
            .analyze();
    const wordmark = await new AxeBuilder({ page })
      .include('.wordmark')
      .disableRules(['color-contrast'])
      .analyze();
    return [...result.violations, ...widget.violations, ...wordmark.violations].flatMap((one) =>
      one.nodes.map((node) => `${one.id} ${node.target.join(' ')}`),
    );
  }

  for (const theme of ['dark', 'light'] as const) {
    test(`finds none on the source view in the ${theme} theme`, async ({ page }) => {
      await open(page);
      await page.locator('nav.menu > div > button:text-is("View")').click();
      await page.locator(`[data-command="view.theme-${theme}"]`).click();
      await expect(page.locator('.app')).toHaveAttribute('data-scheme', theme);

      await openDerived(page);
      await openSource(page);
      expect(await audit(page)).toEqual([]);

      // The schema's own completion, which is Monaco's widget painted from the design's tokens.
      await selectRange(page, '{\n  "schema"', '"model"');
      await page.keyboard.press('Control+Space');
      await expect(page.locator('.suggest-widget .monaco-list-row').first()).toBeVisible({
        timeout: 60_000,
      });
      expect(await audit(page)).toEqual([]);
      await page.keyboard.press('Escape');

      // And with the banner up, which is the other state the pane has.
      await selectRange(page, '{\n  "schema"', '"model": "llama3-8b",', true);
      await paste(page, '\n  "kernels": 1,');
      await expect(page.locator('[data-source-banner="off-grammar"]')).toBeVisible({
        timeout: 30_000,
      });
      expect(await audit(page)).toEqual([]);
    });
  }
});
