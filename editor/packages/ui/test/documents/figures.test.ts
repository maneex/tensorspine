import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import type { PyRecord } from '@tensorspine/lang';
import type { Lang, LibraryHandle } from '@tensorspine/lang/api';

import {
  elementsText,
  figureOf,
  groupedNumber,
  operationsText,
  renderedFormats,
  sizeText,
  statusFigures,
} from '../../src/documents/figures.js';
import { shapesFor } from '../../src/documents/shapes.js';
import { presentation } from '../../src/presentation/index.js';
import { core, corpusText, registry, repositoryRoot } from './source.js';

// The figures of §4.2's status bar: the renderings `tools/view.py` writes (finding F6 moves them
// here), and the walk that **finds** the four fields in the derived document rather than naming
// four paths in a component.

describe('the renderings, which are `view.py`’s own', () => {
  it('writes bytes as it does — GiB at two places, MiB at one, KiB at one or none', () => {
    expect(sizeText(0)).toBe('—');
    expect(sizeText(512)).toBe('512 B');
    expect(sizeText(1024)).toBe('1.0 KiB');
    expect(sizeText(10_240)).toBe('10 KiB');
    expect(sizeText(131_072)).toBe('128 KiB');
    expect(sizeText(521_216)).toBe('509 KiB');
    expect(sizeText(2 ** 20)).toBe('1.0 MiB');
    expect(sizeText(16_060_522_496)).toBe('14.96 GiB');
  });

  it('writes operations as it does — Gop and Mop at two places', () => {
    expect(operationsText(0)).toBe('—');
    expect(operationsText(999)).toBe('999 op');
    expect(operationsText(15_009_857_536)).toBe('15.01 Gop');
    expect(operationsText(2_500_000)).toBe('2.50 Mop');
  });

  it('groups a whole number the way `fmt_int` does', () => {
    expect(groupedNumber(1234567)).toBe('1 234 567');
    expect(elementsText(8_030_261_248)).toBe('8.03 G');
  });

  it('falls back to the product’s own number where a format has no rendering', () => {
    expect(figureOf(1234, undefined)).toEqual({ text: '1 234', exact: '1 234' });
    expect(figureOf(1234, { format: 'shape' })).toEqual({ text: '1 234', exact: '1 234' });
  });

  it('renders only formats the editor’s own schema admits', () => {
    // The same discipline §1 (d) puts on the core's semantic tables, one schema along: the table
    // is held to the enumeration rather than to a list written twice.
    const schema = JSON.parse(
      readFileSync(
        join(repositoryRoot, 'editor', 'schemas', 'tensorspine-editor-presentation.schema.json'),
        'utf8',
      ),
    ) as { $defs: { binding: { properties: { format: { enum: string[] } } } } };
    const admitted = schema.$defs.binding.properties.format.enum;
    expect(renderedFormats().filter((one) => !admitted.includes(one))).toEqual([]);
    // And every format the status bar's own bindings ask for has one, which is the half that
    // would otherwise show a raw number in the bar.
    const asked = presentation()
      .anchors.map((anchor) => presentation().at(anchor))
      .filter((binding) => binding?.statusBar !== undefined)
      .map((binding) => binding?.format ?? '');
    expect(asked.filter((one) => !renderedFormats().includes(one))).toEqual([]);
  });
});

describe('the status bar’s four figures, found in the derived document', () => {
  let derived: PyRecord;
  let stop: () => void;

  beforeAll(async () => {
    const session = await core();
    stop = session.stop;
    derived = await deriveLlama(session.lang, session.library);
  }, 120_000);

  afterAll(() => {
    stop();
  });

  it('are the four §4.2 names, in the order the bindings give them', () => {
    const figures = statusFigures(derived, shapesFor(registry()), presentation());
    expect(figures.map((one) => one.label)).toEqual([
      'params',
      '/ element',
      '/ cached position',
      'peak live',
    ]);
  });

  it('carry the numbers the core derived, rendered as the artboard writes them', () => {
    const figures = statusFigures(derived, shapesFor(registry()), presentation());
    expect(figures.map((one) => one.figure.text)).toEqual([
      // D3 `totals.bytes`, D5 `operations.element`, D4 `totals.append_bytes_per_cached_position`,
      // D2 `peak_live.bytes_per_element` — S18's own `14.96 GiB` and `15.01 Gop / element`, and
      // the inventory's erratum E14 ("509 KiB; 509 KiB is D2's whole-graph `peak_live`").
      '14.96 GiB',
      '15.01 Gop',
      '128 KiB',
      '509 KiB',
    ]);
    // The exact number is what the tooltip shows: no component rounds a figure away (§7).
    expect(figures[0]?.figure.exact).toBe(groupedNumber(16_060_522_496));
  });

  it('carry the epistemic status of a qualified value beside it, read from the schema', () => {
    const figures = statusFigures(derived, shapesFor(registry()), presentation());
    expect(figures[1]?.status).toBe('exact');
    expect(figures[0]?.status).toBeUndefined();
  });

  it('are found by the bindings and by nothing else: unmark one and it leaves the bar', () => {
    const marked = presentation();
    const without = {
      anchors: marked.anchors,
      at: (anchor: string) => {
        const binding = marked.at(anchor);
        if (binding?.statusBar?.label !== 'peak live') return binding;
        const { statusBar, ...rest } = binding;
        void statusBar;
        return rest;
      },
      firstOf(candidates: readonly string[]) {
        for (const anchor of candidates) {
          const found = this.at(anchor);
          if (found !== undefined) return found;
        }
        return undefined;
      },
    };
    expect(statusFigures(derived, shapesFor(registry()), without).map((one) => one.label)).toEqual([
      'params',
      '/ element',
      '/ cached position',
    ]);
  });
});

/** `llama3-8b`, derived — the document every figure above is read out of. */
async function deriveLlama(lang: Lang, library: LibraryHandle): Promise<PyRecord> {
  const tree = await lang.parse(corpusText('llama3-8b'));
  return lang.derive(tree, 'data/models/llama3-8b.json', { library });
}
