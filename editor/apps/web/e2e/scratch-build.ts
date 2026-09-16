import { expect, type Locator, type Page } from '@playwright/test';

import { modelText } from './source-text.js';

/**
 * `llama3-8b`, built from an empty document by gestures alone — feature 2.19's own script.
 *
 * Every function below is something a person does: a menu entry of §4.4, a drop on the canvas
 * (§4.7), a field or a chip of a sheet (§4.11, §4.12), a connection between two port handles, an
 * entry of a context menu, a token of the location editor (§4.14). Nothing here writes JSON, calls the
 * store, or reaches past the page; the only thing it knows about the target is what
 * `data/models/llama3-8b.json` says the model *is* — its quantities, its families, its arguments
 * and its physical names — which is what an author transcribing the checkpoint would know.
 *
 * The **order** the declarations are written in is part of the build and not an accident: D2's
 * values, D3's tensors and D6's partition options are listed in the order the document declares
 * their sites and rules (`derive.py` walks the document, `d1.py` alone re-sorts). Measured by
 * permuting the corpus document and deriving it again: the quantities may move, nothing else may.
 * So the sites are created in the corpus's order, the rules in the corpus's order, and the two
 * boundary edges that must sit among the top-level rules are made before the sites they name are
 * moved inside the composition.
 */

/**
 * A line in the run's own log, so that a build that stops says where.
 *
 * Four hundred gestures is too many to find one's way through a stack trace: each phase of
 * {@link buildLlama3} names itself as it finishes, with the seconds it took.
 */
let began = Date.now();
function note(what: string): void {
  const since = Date.now() - began;
  began = Date.now();
  process.stdout.write(`      · ${what} (+${String(Math.round(since / 100) / 10)} s)\n`);
}

/** Where `File ▸ New Model` puts a document, and where this script saves it. */
export const UNTITLED = 'untitled.json';

/**
 * Where the built document is saved, and why it is not `models/llama3-8b.json`.
 *
 * What the path has to give is the **directory depth** the corpus's own document sits at, so that
 * `../primitive-library/` resolves to the base the library was gathered from (`bases_of` joins the
 * document's directory with what `primitive_libraries` writes, and nothing else). What it must not
 * be is a path the workspace already holds: the Examples workspace *is* the vendored corpus, and
 * saving onto `models/llama3-8b.json` is a conflict, which is the optimistic concurrency of §5.2
 * doing its job rather than a defect.
 */
export const SAVED = 'built/llama3-8b.json';

/** The base the corpus document pins, from the directory this script saves into. */
export const CORPUS_BASE = '../primitive-library/';

// ---------------------------------------------------------------------------------------------
// The gestures.
// ---------------------------------------------------------------------------------------------

/** Open the application and the vendored Examples workspace (§4.3, D11). */
export async function openEditor(page: Page): Promise<void> {
  await page.goto('/');
  await expect(page.locator('#root')).toHaveAttribute('data-state', 'ready');
  await expect(page.locator('#root')).toHaveAttribute('data-documents', 'ready');
  await page.locator('.nothing .link[data-open="examples"]').click();
  await page.locator('.dlg [data-workspace="examples"]').click();
  await expect(page.locator('footer.status [data-workspace]')).toHaveText('Examples');
}

/** Run a command of §4.4 by its menu entry, which is the way a person reaches it. */
export async function command(page: Page, menu: string, id: string): Promise<void> {
  await page.locator(`nav.menu > div > button:text-is("${menu}")`).click();
  await page.locator(`.menu-pop button[data-command="${id}"]`).click();
}

/** The Properties panel's body — every sheet of §4.11 is drawn in it. */
export function sheet(page: Page): Locator {
  return page.locator('.insp .panel-body');
}

/** One row of the argument sheet (§4.12), by the argument path it stands for. */
export function argument(page: Page, path: string): Locator {
  return sheet(page).locator(`[data-argument="${path}"]`);
}

/** A box of the canvas, by the place it stands for. */
export function box(page: Page, pointer: string): Locator {
  return page.locator(`[data-box="${pointer}"]`);
}

/** One port handle, by the box and the port it belongs to. */
export function port(page: Page, pointer: string, name: string): Locator {
  return page.locator(`[data-port="${pointer}:${name}"]`);
}

/** Select a place through the Model explorer's tree, opening the groups on the way down (§4.5). */
export async function select(page: Page, pointer: string): Promise<void> {
  const steps = pointer.split('/').filter((step) => step !== '');
  for (let depth = 1; depth < steps.length; depth += 1) {
    const at = `/${steps.slice(0, depth).join('/')}`;
    const chevron = page.locator(`.tree [data-chevron="${at}"]`);
    if ((await page.locator(`.tree [data-row="${at}"]`).count()) === 0) continue;
    if ((await chevron.count()) > 0 && (await chevron.innerText()) === '▸') await chevron.click();
  }
  await page.locator(`.tree [data-row="${pointer}"]`).click();
  await expect(sheet(page).locator('.insp-title')).toBeVisible();
}

/** The document's own bytes, read through `View ▸ JSON Source` (§4.10). */
export async function source(page: Page, path: string): Promise<string> {
  await command(page, 'View', 'view.json-source');
  await expect(page.locator('.src-pane[data-ready="true"]')).toBeVisible({ timeout: 60_000 });
  const shown = await modelText(page, path);
  await page.keyboard.press('Control+w');
  await expect(page.locator('.src-pane')).toHaveCount(0);
  return shown;
}

/**
 * Drop a primitive on the canvas — §4.7's first gesture.
 *
 * There is no palette yet (feature 3.1 builds it), so what is dispatched is the `DataTransfer`
 * the Library activity will carry; feature 2.9 built the *drop* and its own suite dispatches the
 * same transfer.
 */
export async function dropPrimitive(page: Page, primitive: string, x: number, y: number): Promise<void> {
  await page.locator('.canvas.graph').evaluate(
    (element, held: { json: string; x: number; y: number }) => {
      const transfer = new DataTransfer();
      transfer.setData('application/x-tensorspine-primitive', held.json);
      const corner = element.getBoundingClientRect();
      for (const kind of ['dragover', 'drop']) {
        element.dispatchEvent(
          new DragEvent(kind, {
            bubbles: true,
            cancelable: true,
            dataTransfer: transfer,
            clientX: corner.left + held.x,
            clientY: corner.top + held.y,
          }),
        );
      }
    },
    { json: JSON.stringify({ primitive, version: '1.0.0' }), x, y },
  );
}

/** Rename what the sheet has open, in its Identity row (§4.4: "edited in the sheet like any value"). */
export async function rename(page: Page, to: string): Promise<void> {
  const field = sheet(page).locator('input[data-name-field]');
  await field.fill(to);
  await field.press('Enter');
}

/**
 * Give the open declaration exactly these families — §4.11's chip editor.
 *
 * The wanted ones are added first and the proposed one taken away afterwards, because
 * `family_list` requires one member and the × of a lone chip is disabled until a second stands
 * beside it.
 */
export async function families(page: Page, want: readonly string[]): Promise<void> {
  const held = sheet(page).locator('[data-member="families"]');
  const written = await held.locator('.fchip .mono').allInnerTexts();
  for (const family of want) {
    if (written.includes(family)) continue;
    const field = held.locator('[data-add="families"]');
    await field.fill(family);
    await field.press('Enter');
  }
  for (const family of written) {
    if (want.includes(family)) continue;
    await held.locator(`[data-remove="${family}"]`).click();
  }
  await expect(held.locator('.fchip .mono')).toHaveText([...want]);
}

/**
 * The families of a place whose sheet is the **generated form** rather than §4.11's Identity
 * section — a composition's.
 *
 * The same `family_list` is drawn two ways: as the chip editor on an instance's own sheet (§4.11)
 * and as the generic list everywhere else (§4.16's own rendering of a list of scalars). Both are
 * gestures; which one a place gets is the sheet it belongs to.
 */
export async function listedFamilies(page: Page, want: readonly string[]): Promise<void> {
  const held = sheet(page).locator('[data-list="/families"]');
  const written = await held.locator('.fentry').count();
  for (let at = written; at < want.length; at += 1) {
    await held.locator('[data-add-item="/families"]').click();
  }
  for (const [at, family] of want.entries()) {
    const field = held.locator(`[data-entry="/families/${String(at)}"] [data-member-value="${String(at)}"]`);
    await field.fill(family);
    await field.blur();
  }
  for (let at = written - 1; at >= want.length; at -= 1) {
    await held.locator(`[data-remove-entry="/families/${String(at)}"]`).click();
  }
  await expect(held.locator('.fentry')).toHaveCount(want.length);
}

/** Set one argument row of §4.12 to a quantity — its source mode, then the quantity itself. */
export async function fromQuantity(page: Page, path: string, quantity: string): Promise<void> {
  const row = argument(page, path);
  await row.locator(`[data-modes="${path}"]`).selectOption('quantity');
  // The mode is what the row is drawn from, so the value control is the one the *new* mode has:
  // waited for by the row's own state rather than by the control, which is still the old one for
  // as long as React has not drawn the new one.
  await expect(row).toHaveAttribute('data-mode', 'quantity');
  await row.locator(`[data-value="${path}"]`).selectOption(quantity);
}

/** Set one argument row to a literal — a select where the declaration enumerates, a field else. */
export async function fromLiteral(
  page: Page,
  path: string,
  written: string,
  enumerated = false,
): Promise<void> {
  const row = argument(page, path);
  await row.locator(`[data-modes="${path}"]`).selectOption('literal');
  await expect(row).toHaveAttribute('data-mode', 'literal');
  const value = row.locator(`[data-value="${path}"]`);
  if (enumerated) await value.selectOption(written);
  else {
    await value.fill(written);
    await value.blur();
  }
}

/**
 * Connect an output handle to an input handle — §4.7's second gesture, in its **click** form.
 *
 * "A connection is one gesture with two shapes: press an output handle and release over an input,
 * or click one and then the other — the second is the keyboard's form, and both are one state
 * machine" (feature 2.9). The click form is what a script can rely on: a click is an actionability
 * check that re-resolves the element and waits for it to be stable and hittable, where a raw drag
 * measures a rectangle once and releases wherever that rectangle *was*. And the canvas moves while
 * a document is being built — a card grows its port handles when `describe` comes back (§5.4's "at
 * once") and ELK lays the drawing out again off the interface's thread, which moved a target 300 px
 * between a press and a release the first time this script was run.
 *
 * `wire` is the place the new rule is written at, and it is asserted: a gesture that lands nowhere
 * leaves the connection *armed* rather than refusing (§4.7), and says nothing.
 */
export async function connect(page: Page, from: Locator, to: Locator, wire: string): Promise<void> {
  await from.click();
  await expect(page.locator('.dghost')).toBeVisible();
  await to.click();
  await expect(page.locator(`[data-wire="${wire}"]`)).toHaveCount(1);
}

/** Take one entry of a box's context menu (§4.7), opened on the box's own name. */
export async function boxMenu(page: Page, pointer: string, entry: string): Promise<void> {
  await box(page, pointer).locator('.n-name').click({ button: 'right' });
  await expect(page.locator('.ctxmenu')).toBeVisible();
  await page.locator(`.ctxmenu button[data-entry="${entry}"]`).click();
}

/** What the status bar says about the two stages of §5.4, once they have both finished. */
export async function settled(page: Page): Promise<void> {
  await expect(page.locator('footer.status [data-validation]')).toHaveText('no problems', {
    timeout: 120_000,
  });
  await expect(page.locator('footer.status [data-derivation]')).toHaveText('derived · fresh', {
    timeout: 120_000,
  });
}

// ---------------------------------------------------------------------------------------------
// What each part of the document is, as the corpus writes it.
// ---------------------------------------------------------------------------------------------

/** One quantity of §4.11's quantity sheet. */
interface Quantity {
  readonly name: string;
  /** `cardinality`, `real` or `enum` — the alternative of `quantity_type` its chooser takes. */
  readonly type: 'cardinality' | 'real' | 'enum';
  /** The literal the source writes. */
  readonly value: string;
  /** An `enum` type's admissible values. */
  readonly values?: readonly string[];
  /** A literal that declares how it follows from others (V11), in §4.13's text form. */
  readonly derivation?: string;
}

const QUANTITIES: readonly Quantity[] = [
  { name: 'd', type: 'cardinality', value: '4096' },
  { name: 'ffn', type: 'cardinality', value: '14336' },
  { name: 'heads', type: 'cardinality', value: '32' },
  { name: 'kv_heads', type: 'cardinality', value: '8' },
  { name: 'head_dim', type: 'cardinality', value: '128', derivation: 'd div heads' },
  { name: 'layers', type: 'cardinality', value: '32' },
  { name: 'vocab', type: 'cardinality', value: '128256' },
  { name: 'eps', type: 'real', value: '1e-05' },
  { name: 'precision', type: 'enum', value: 'bf16', values: ['bf16'] },
];

/** One instance, as the palette drops it and the sheets then fill it in. */
interface Instance {
  /** The primitive to drop. */
  readonly primitive: string;
  /** What the drop proposes to call it (§9 Q4: the primitive name's last segment). */
  readonly proposed: string;
  /** What this document calls it. */
  readonly name: string;
  readonly families: readonly string[];
  /** Arguments written as a quantity, by argument path. */
  readonly quantities?: Readonly<Record<string, string>>;
  /** Arguments written as a literal, by argument path; `true` where the widget enumerates. */
  readonly literals?: readonly (readonly [string, string, boolean])[];
  /** Argument paths whose mode is `record` — written before the fields under them. */
  readonly records?: readonly string[];
}

const EMBED: Instance = {
  primitive: 'embed',
  proposed: 'embed',
  name: 'embed',
  families: ['input', 'embedding'],
  quantities: { width: 'd', vocabulary: 'vocab' },
};

const FINAL_N: Instance = {
  primitive: 'norm.rms',
  proposed: 'rms',
  name: 'final_n',
  families: ['norm'],
  quantities: { width: 'd', eps: 'eps' },
};

const LM_HEAD: Instance = {
  primitive: 'lm_head',
  proposed: 'lm_head',
  name: 'lm_head',
  families: ['output', 'head'],
  quantities: { width: 'd', vocabulary: 'vocab' },
};

/** The six sites of `decoder`, in the order the corpus declares them. */
const SITES: readonly Instance[] = [
  {
    primitive: 'norm.rms',
    proposed: 'rms',
    name: 'attn_n',
    families: ['norm'],
    quantities: { width: 'd', eps: 'eps' },
  },
  {
    primitive: 'attention.dense',
    proposed: 'dense',
    name: 'attn',
    families: ['sequence_operator'],
    quantities: { width: 'd', heads: 'heads', kv_heads: 'kv_heads', head_dim: 'head_dim' },
    literals: [
      ['mask', 'causal', true],
      ['rope.theta', '500000', false],
      ['rope.layout', 'split', true],
    ],
    records: ['rope'],
  },
  {
    primitive: 'residual.add',
    proposed: 'add',
    name: 'attn_r',
    families: ['residual'],
    quantities: { width: 'd' },
  },
  {
    primitive: 'norm.rms',
    proposed: 'rms',
    name: 'ffn_n',
    families: ['norm'],
    quantities: { width: 'd', eps: 'eps' },
  },
  {
    primitive: 'ffn.gated',
    proposed: 'gated',
    name: 'ffn',
    families: ['feed_forward'],
    quantities: { width: 'd', inner: 'ffn' },
    literals: [['activation', 'silu', true]],
  },
  {
    primitive: 'residual.add',
    proposed: 'add',
    name: 'ffn_r',
    families: ['residual'],
    quantities: { width: 'd' },
  },
];

/** One token of a physical name (§3.4, §4.14's editor). */
interface Token {
  readonly kind: 'string' | 'index';
  readonly text: string;
}

/** A tensor identity to write: the slot it is bound from, and where its tensor is. */
interface Located {
  /** The place of the site the slot belongs to. */
  readonly site: string;
  readonly slot: string;
  /** Whether it is a state port (D4's identities) rather than a parameter slot (D3's). */
  readonly state?: boolean;
  readonly location?: readonly Token[];
}

/** The literal prefix and the index every scoped location of `llama3-8b` begins with. */
const LAYER: readonly Token[] = [
  { kind: 'string', text: 'model.layers.' },
  { kind: 'index', text: 'layer' },
];

const ROOT_PARAMETERS: readonly Located[] = [
  { site: '/instances/embed', slot: 'weight', location: [{ kind: 'string', text: 'model.embed_tokens.weight' }] },
  { site: '/instances/final_n', slot: 'weight', location: [{ kind: 'string', text: 'model.norm.weight' }] },
  { site: '/instances/lm_head', slot: 'weight', location: [{ kind: 'string', text: 'lm_head.weight' }] },
];

const SITE_PARAMETERS: readonly Located[] = [
  { site: 'attn_n', slot: 'weight', location: [...LAYER, { kind: 'string', text: '.input_layernorm.weight' }] },
  { site: 'attn', slot: 'q', location: [...LAYER, { kind: 'string', text: '.self_attn.q_proj.weight' }] },
  { site: 'attn', slot: 'k', location: [...LAYER, { kind: 'string', text: '.self_attn.k_proj.weight' }] },
  { site: 'attn', slot: 'v', location: [...LAYER, { kind: 'string', text: '.self_attn.v_proj.weight' }] },
  { site: 'attn', slot: 'out', location: [...LAYER, { kind: 'string', text: '.self_attn.o_proj.weight' }] },
  { site: 'ffn_n', slot: 'weight', location: [...LAYER, { kind: 'string', text: '.post_attention_layernorm.weight' }] },
  { site: 'ffn', slot: 'gate', location: [...LAYER, { kind: 'string', text: '.mlp.gate_proj.weight' }] },
  { site: 'ffn', slot: 'up', location: [...LAYER, { kind: 'string', text: '.mlp.up_proj.weight' }] },
  { site: 'ffn', slot: 'out', location: [...LAYER, { kind: 'string', text: '.mlp.down_proj.weight' }] },
];

/** The composition this document builds, and the index its sites run over. */
const COMPOSITION = 'decoder';
const INDEX = 'layer';
/** What `Extract to Composition` proposes to call one (feature 2.14, §4.20). */
const PROPOSED_COMPOSITION = 'block';

// ---------------------------------------------------------------------------------------------
// The build.
// ---------------------------------------------------------------------------------------------

/** Add one quantity through `Model ▸ Add Quantity` and fill in its sheet (§4.11, §4.16). */
async function addQuantity(page: Page, quantity: Quantity): Promise<void> {
  note(`quantity ${quantity.name}`);
  await command(page, 'Model', 'model.add-quantity');
  await expect(sheet(page).locator('.insp-kind')).toHaveText('quantity');
  await rename(page, quantity.name);
  const type = sheet(page).locator('[data-chooser="/type"]');
  await type.locator('[data-modes="/type"]').selectOption(quantity.type);
  for (const [at, admitted] of (quantity.values ?? []).entries()) {
    await sheet(page).locator('[data-add-item="/type/values"]').click();
    const field = sheet(page).locator(`[data-entry="/type/values/${String(at)}"] input`).first();
    await field.fill(admitted);
    await field.blur();
  }
  const source_ = sheet(page).locator('[data-chooser="/source"]');
  const value = source_.locator('[data-member-value="value"]');
  await value.fill(quantity.value);
  await value.blur();
  if (quantity.derivation !== undefined) {
    await source_.locator('[data-add-member="/source/derivation"]').click();
    const written = source_.locator('[data-member="derivation"] .xtext');
    await written.fill(quantity.derivation);
    await written.press('Enter');
  }
}

/** Write the arguments and the families of whatever instance the sheet has open (§4.11, §4.12). */
async function fill(page: Page, instance: Instance): Promise<void> {
  await families(page, instance.families);
  for (const path of instance.records ?? []) {
    const row = argument(page, path);
    await row.locator(`[data-modes="${path}"]`).selectOption('record');
    await expect(row).toHaveAttribute('data-mode', 'record');
  }
  for (const [path, quantity] of Object.entries(instance.quantities ?? {})) {
    await fromQuantity(page, path, quantity);
  }
  for (const [path, written, enumerated] of instance.literals ?? []) {
    await fromLiteral(page, path, written, enumerated);
  }
}

/** Drop a primitive, name it, and write what the corpus says it carries. */
async function addInstance(page: Page, instance: Instance, x: number, y: number): Promise<void> {
  await dropPrimitive(page, instance.primitive, x, y);
  await expect(box(page, `/instances/${instance.proposed}`)).toBeVisible();
  await expect(sheet(page).locator('.insp-title')).toHaveText(instance.proposed);
  if (instance.name !== instance.proposed) {
    await rename(page, instance.name);
    await expect(box(page, `/instances/${instance.name}`)).toBeVisible();
  }
  await fill(page, instance);
}

/** Add a public input or output from `Model ▸ Add Input` / `Add Output` and fill its endpoint. */
async function addInterface(
  page: Page,
  which: 'input' | 'output',
  named: { readonly name: string; readonly instance: string; readonly port: string; readonly kind?: string; readonly generative?: boolean },
): Promise<void> {
  await command(page, 'Model', which === 'input' ? 'model.add-input' : 'model.add-output');
  await expect(sheet(page).locator('.insp-kind')).toHaveText(which);
  const at = which === 'input' ? '/to/0' : '/from';
  if (which === 'input') await sheet(page).locator('[data-add-item="/to"]').click();
  const where = which === 'input' ? `[data-entry="${at}"]` : `[data-section="${at}"]`;
  const instance = sheet(page).locator(`${where} [data-member-value="instance"]`);
  await instance.fill(named.instance);
  await instance.blur();
  const held = sheet(page).locator(`${where} [data-member="port"] [data-member-value="port"]`);
  await held.fill(named.port);
  await held.blur();
  if (named.kind !== undefined) {
    await sheet(page).locator('select[data-member-value="kind"]').selectOption(named.kind);
  }
  if (named.generative === true) {
    await sheet(page).locator('[data-member="generative"] [data-member-value="generative"]').check();
  }
  await rename(page, named.name);
}

/** Write one location with the token editor of §4.14 (S8's chips). */
async function writeLocation(page: Page, place: string, tokens: readonly Token[]): Promise<void> {
  await sheet(page).locator('[data-add-member="/location"]').click();
  await expect(sheet(page).locator('[data-chooser="/location"]')).toHaveAttribute('data-mode', 'tensor');
  const held = sheet(page).locator(`[data-tokens="${place}/location/tensor"]`);
  for (const [at, token] of tokens.entries()) {
    await held.locator('[data-token-add]').click();
    await expect(held.locator(`[data-token="${String(at)}"]`)).toBeVisible();
    if (token.kind !== 'string') {
      await held.locator(`select[data-token-kind="${String(at)}"]`).selectOption(token.kind);
    }
    const text = held.locator(`input[data-token-text="${String(at)}"]`);
    await text.fill(token.text);
    await text.blur();
  }
  await expect(held.locator('.tok:not(.add)')).toHaveCount(tokens.length);
}

/** Write the dtype selector `{"quantity": "precision"}` on the identity the sheet has open. */
async function writeDtype(page: Page, quantity: string): Promise<void> {
  await sheet(page).locator('[data-add-member="/dtype"]').click();
  await sheet(page).locator('[data-modes="/dtype"]').selectOption('quantity');
  // The chooser's own value: the member the mode names, drawn in the section's body.
  const field = sheet(page).locator('[data-chooser="/dtype"] [data-member-value="dtype"]');
  await field.fill(quantity);
  await field.blur();
}

/**
 * Bind one slot privately (§4.11), then give its identity a dtype and a location (§4.14).
 *
 * The slot's own row is where the first gesture is, and "Edit location…" beside it is what opens
 * the identity's sheet — which is where the other two are written.
 */
async function bind(page: Page, where: string, held: Located): Promise<void> {
  await select(page, where);
  const row = sheet(page).locator(held.state === true ? `[data-state="${held.slot}"]` : `[data-slot="${held.slot}"]`);
  await row.locator(`[data-bind-private="${held.slot}"]`).click();
  if (held.location === undefined) return;
  await row.locator(`[data-edit-identity="${held.slot}"]`).click();
  const place = await sheet(page).locator('[data-place]').innerText();
  await writeDtype(page, 'precision');
  await writeLocation(page, place, held.location);
}

/**
 * Build `llama3-8b` from an empty document, gesture by gesture.
 *
 * Returns nothing: what it leaves behind is a document open in the editor at {@link SAVED}, with
 * its base pinned as the corpus pins it, which the suite then validates, derives and exports.
 */
export async function buildLlama3(page: Page): Promise<void> {
  note('opening the editor');
  await openEditor(page);
  await command(page, 'File', 'file.new-model');
  await expect(page.locator(`.gcanvas[data-canvas="${UNTITLED}"]`)).toBeVisible();

  // The model identifier, in the Document sheet §4.4's own command opens (§4.11's last row).
  await command(page, 'Model', 'model.document-properties');
  const identifier = sheet(page).locator('[data-member="model"] [data-member-value="model"]');
  await identifier.fill('llama3-8b');
  await identifier.blur();

  // The three root instances, in the order the corpus declares them. A new document is off the
  // grammar until it has a site *and* both interfaces (feature 2.6's own finding), and `describe`
  // gates on the grammar (feature 1.6d) — so no card has a port handle at all until the two
  // interfaces below are written, which is why they come before anything that is connected.
  await dropPrimitive(page, EMBED.primitive, 200, 140);
  await expect(box(page, '/instances/embed')).toBeVisible();
  await dropPrimitive(page, FINAL_N.primitive, 200, 360);
  await expect(box(page, `/instances/${FINAL_N.proposed}`)).toBeVisible();
  await rename(page, FINAL_N.name);
  await dropPrimitive(page, LM_HEAD.primitive, 200, 580);
  await expect(box(page, '/instances/lm_head')).toBeVisible();

  note('the three root instances');
  await addInterface(page, 'input', { name: 'tokens', instance: 'embed', port: 'tokens', kind: 'token' });
  await addInterface(page, 'output', { name: 'logits', instance: 'lm_head', port: 'logits', generative: true });
  // On the grammar at last: the core answers, and the cards grow their handles.
  await expect(port(page, '/instances/embed', 'output')).toBeVisible({ timeout: 60_000 });

  note('the two public interfaces');
  for (const quantity of QUANTITIES) await addQuantity(page, quantity);

  note('the nine quantities');
  for (const instance of [EMBED, FINAL_N, LM_HEAD]) {
    await select(page, `/instances/${instance.name}`);
    await fill(page, instance);
  }

  // The first site of the composition, as a root instance: §4.20's "Extract to Composition" turns
  // selected **roots** into a new composition, and the edge into it becomes a boundary edge.
  note('the root instances’ arguments');
  await addInstance(page, SITES[0] as Instance, 620, 140);
  await command(page, 'View', 'view.reset-layout');
  await connect(
    page,
    port(page, '/instances/embed', 'output'),
    port(page, '/instances/attn_n', 'input'),
    '/bindings/values/attn_n.input',
  );
  await boxMenu(page, '/instances/attn_n', 'canvas.extract-to-composition');
  await expect(box(page, `/compositions/${PROPOSED_COMPOSITION}`)).toBeVisible();

  note('the composition, extracted from its first site');
  // The composition is the author's to name, and its range is the author's to write.
  await select(page, `/compositions/${PROPOSED_COMPOSITION}`);
  await rename(page, COMPOSITION);
  await select(page, `/compositions/${COMPOSITION}`);
  await listedFamilies(page, [COMPOSITION]);
  // The range the corpus writes: the literal the configuration gives, not the quantity (erratum
  // E10 — only the template `decoder-causal-yarn` names one).
  const stop = sheet(page).locator(`[data-entry="/indices/${INDEX}"] [data-member="stop"] .xtext`);
  await stop.fill('32');
  await stop.press('Enter');
  await expect(box(page, `/compositions/${COMPOSITION}`).locator('.g-count')).toHaveText('×32');

  // The other five, each made as a root — where the two that a boundary edge names are wired
  // before they move inside, so that those two rules are written among the top-level ones and in
  // the corpus's own order.
  note('the composition named, and its range');
  for (const site of SITES.slice(1)) {
    await addInstance(page, site, 620, 140);
    await command(page, 'View', 'view.reset-layout');
    if (site.name === 'attn_r') {
      await connect(
        page,
        port(page, '/instances/embed', 'output'),
        port(page, '/instances/attn_r', 'a'),
        '/bindings/values/attn_r.a',
      );
    }
    if (site.name === 'ffn_r') {
      await connect(
        page,
        port(page, '/instances/ffn_r', 'output'),
        port(page, '/instances/final_n', 'input'),
        '/bindings/values/final_n.input',
      );
    }
    await boxMenu(page, `/instances/${site.name}`, 'canvas.add-to-composition');
    await page.locator(`.ctxmenu button[data-entry="canvas.add-to-composition:${COMPOSITION}"]`).click();
    await expect(box(page, `/instances/${site.name}`)).toHaveCount(0);
  }

  note('the five other sites');

  // The exit edge names the last iteration, and §4.20's move writes that as a **copy** of the
  // bound — the grammar has no "last index" expression to refer to one — so it reads `32 - 1`
  // where the corpus's author wrote `layers - 1`. §4.13's editor is where an index expression is
  // said differently, and this is the one place this build uses it on an endpoint.
  await select(page, '/bindings/values/final_n.input');
  const exit = sheet(page).locator(`[data-entry="/from/instance/indices/${INDEX}"] .xtext`);
  await exit.fill('layers - 1');
  await exit.press('Enter');
  await expect(exit).toHaveValue('layers - 1');

  // The fourth and last top-level edge.
  await command(page, 'View', 'view.reset-layout');
  await connect(
    page,
    port(page, '/instances/final_n', 'output'),
    port(page, '/instances/lm_head', 'input'),
    '/bindings/values/lm_head.input',
  );

  note('the four top-level edges');
  await wireInside(page);
  note('the eight scoped rules');
  await bindAll(page);
  note('the thirteen identities');
  await saveWhereTheCorpusSits(page);
  note('saved where the corpus sits');
}

/** The composition's own eight scoped rules, in the corpus's order — §4.8, in the drill-in. */
async function wireInside(page: Page): Promise<void> {
  await boxMenu(page, `/compositions/${COMPOSITION}`, 'canvas.drill-in');
  await expect(page.locator('.drill')).toBeVisible();
  const at = (site: string): string => `/compositions/${COMPOSITION}/instances/${site}`;

  // The two carries first: "Connect from previous iteration…" writes the override on the producing
  // end and **proposes** the guard, which the toast's own action accepts (§4.8).
  for (const [site, into] of [
    ['attn_n', 'input'],
    ['attn_r', 'a'],
  ] as const) {
    await port(page, at('ffn_r'), 'output').click({ button: 'right' });
    await expect(page.locator('.ctxmenu')).toBeVisible();
    await page.locator(`.ctxmenu button[data-entry="canvas.connect-previous:${INDEX}"]`).click();
    await port(page, at(site), into).click();
    const toast = page.locator('.toast');
    await expect(toast.locator('.undo')).toHaveText(`Add guard ${INDEX} >= 1`);
    await toast.locator('.undo').click();
  }

  // Then the chain, in the order the corpus writes it.
  const chain: readonly (readonly [string, string, string, string])[] = [
    ['attn_n', 'output', 'attn', 'input'],
    ['attn', 'output', 'attn_r', 'b'],
    ['attn_r', 'output', 'ffn_n', 'input'],
    ['attn_r', 'output', 'ffn_r', 'a'],
    ['ffn_n', 'output', 'ffn', 'input'],
    ['ffn', 'output', 'ffn_r', 'b'],
  ];
  for (const [from, out, to, into] of chain) {
    await connect(
      page,
      port(page, at(from), out),
      port(page, at(to), into),
      `/compositions/${COMPOSITION}/bindings/values/${to}.${into}`,
    );
  }
}

/** Every identity of the document: three at the top level, nine and a state inside (§4.11, §4.14). */
async function bindAll(page: Page): Promise<void> {
  for (const held of ROOT_PARAMETERS) await bind(page, held.site, held);
  for (const held of SITE_PARAMETERS) {
    await bind(page, `/compositions/${COMPOSITION}/instances/${held.site}`, held);
  }
  await bind(page, `/compositions/${COMPOSITION}/instances/attn`, {
    site: 'attn',
    slot: 'kv',
    state: true,
  });
}

/**
 * Save the document where the corpus keeps it, and pin the base from there.
 *
 * A document made by `File ▸ New Model` sits at the workspace's root and is given the base it
 * resolves from there (`primitive-library/`); the corpus's own document sits one directory down
 * and writes `../primitive-library/`. Both name the same folder, and `bases_of` resolves both to
 * it — so the move and the one field are what make the document the corpus's, envelope and all.
 */
async function saveWhereTheCorpusSits(page: Page): Promise<void> {
  await command(page, 'File', 'file.save-as');
  const where = page.locator('.dlg input.ctl.wide.mono');
  await expect(where).toBeVisible();
  await where.fill(SAVED);
  note('the Save As dialog');
  // The Examples workspace is read-only, so the write is handed to the reader as a download
  // (feature 2.4) — which is not what says the save happened: the document's own path is, and the
  // tab that carries it (feature 2.6: "a tab's identity is its workspace and its path").
  await page.locator('.dlg [data-save-as]').click();
  await expect(page.locator(`.gcanvas[data-canvas="${SAVED}"]`)).toBeVisible({ timeout: 60_000 });
  note('the document saved one directory down');

  await command(page, 'Model', 'model.document-properties');
  const base = sheet(page).locator('[data-list="/primitive_libraries"] [data-member-value="base"]');
  await base.fill(CORPUS_BASE);
  await base.blur();
}


/** The first phases of {@link buildLlama3}, for a probe that does not need the whole build. */
export async function buildStart(page: Page): Promise<void> {
  await openEditor(page);
  await command(page, 'File', 'file.new-model');
  await expect(page.locator(`.gcanvas[data-canvas="${UNTITLED}"]`)).toBeVisible();
  await command(page, 'Model', 'model.document-properties');
  const identifier = sheet(page).locator('[data-member="model"] [data-member-value="model"]');
  await identifier.fill('llama3-8b');
  await identifier.blur();
  await dropPrimitive(page, EMBED.primitive, 200, 140);
  await expect(box(page, '/instances/embed')).toBeVisible();
  await dropPrimitive(page, FINAL_N.primitive, 200, 360);
  await rename(page, FINAL_N.name);
  await dropPrimitive(page, LM_HEAD.primitive, 200, 580);
  await expect(box(page, '/instances/lm_head')).toBeVisible();
  await addInterface(page, 'input', { name: 'tokens', instance: 'embed', port: 'tokens', kind: 'token' });
  await addInterface(page, 'output', { name: 'logits', instance: 'lm_head', port: 'logits', generative: true });
  await expect(port(page, '/instances/embed', 'output')).toBeVisible({ timeout: 60_000 });
  for (const quantity of QUANTITIES) await addQuantity(page, quantity);
}
