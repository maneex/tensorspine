import { describe, expect, it } from 'vitest';

import { pyStr, serialize, type SchemaFacts, type SiteDescription } from '@tensorspine/lang';
import { DocumentStore } from '@tensorspine/store';

import type { ArgumentRow, ArgumentSheet } from '../../src/sheet/arguments.js';
import { NO_ARGUMENT_FACTS } from '../../src/sheet/artifact.js';
import {
  blankOf,
  clearValue,
  literalMode,
  numberAsTyped,
  pinValue,
  writeLiteral,
  writeMode,
  writeTypedNumber,
} from '../../src/sheet/edits.js';
import { artifactOf, context, described, freshTree, sheetFor } from './source.js';

/**
 * What a row writes — plan §4.12's source modes, "defaults are not written" and `Pin value`.
 *
 * The three edits the feature's own block names are here, each made on the row and read back
 * through the **core**: `kv_heads` to 3 is V8's refusal, `eps` to 0 on `final_n` is the domain
 * error on its row, `output_gate` on takes the slot `q` away. Nothing is asserted about a verdict
 * this file computes; every one of them is what `describe` answered for the edited document.
 */

const LLAMA = 'llama3-8b';
const ATTN = 'decoder/attn[layer=0]';

/** A document in a store, with the sheet of one of its sites, re-described after every edit. */
class Sheeted {
  readonly store: DocumentStore;

  constructor(
    private readonly name: string,
    private readonly where: string,
  ) {
    this.store = new DocumentStore(freshTree(name), context.shapes);
  }

  /** The core's description of the site as the document stands. */
  site(): SiteDescription {
    const site = described(this.name, this.store.tree).sites.get(this.where);
    if (site === undefined) throw new Error(`no site ${this.where}`);
    return site;
  }

  /** The sheet of that site, as the interface builds it. */
  sheet(): ArgumentSheet {
    return sheetFor(this.name, this.store.tree, this.where);
  }

  row(path: string): ArgumentRow {
    const row = this.sheet().rows.find((one) => one.path === path);
    if (row === undefined) throw new Error(`no row ${path}`);
    return row;
  }

  /** What the artifact asserts about a row's place — what a literal is written as. */
  facts(row: ArgumentRow): SchemaFacts {
    const site = this.site();
    const artifact = artifactOf(pyStr(site.primitive), pyStr(site.version));
    return (artifact?.at(row.path) ?? NO_ARGUMENT_FACTS).facts;
  }
}

describe('a structural argument set to a value an invariant refuses', () => {
  it('is V8’s refusal, on the row and in the Problems the core answers (§4.12)', () => {
    const held = new Sheeted(LLAMA, ATTN);
    const row = held.row('kv_heads');
    const literal = literalMode(row.modes);
    expect(literal).toBeDefined();
    held.store.apply(writeLiteral(held.store.context, row, 3, held.facts(row)));

    const after = held.row('kv_heads');
    expect(after.mode).toBe('literal');
    expect(after.effective).toBe('3');
    const invariant = held.sheet().invariants[0];
    expect(invariant?.verdict).toBe('fails');
    expect(invariant?.shown).toBe('heads = 32, kv_heads = 3');
    expect(invariant?.problem?.message).toContain(
      "'heads is a multiple of kv_heads' does not hold (heads = 32, kv_heads = 3)",
    );
  });
});

describe('a real argument set below its domain', () => {
  it('is the domain error on the row, in the core’s own words', () => {
    const held = new Sheeted(LLAMA, 'final_n');
    const row = held.row('eps');
    held.store.apply(writeLiteral(held.store.context, row, 0, held.facts(row)));

    const after = held.row('eps');
    expect(after.domain).toBe('refused');
    // `0.0` and not `0`: a `real` argument is written as a float (D12, "the float-ness taken from
    // the declared type"), and the core prints the value it was given.
    expect(after.error).toBe("argument 'eps' = 0.0 is below the domain bound 0 (exclusive)");
  });
});

describe('a structural argument that changes which slots exist', () => {
  it('takes `q` away and puts `q_gated` there — the core’s answer, not the sheet’s', () => {
    const held = new Sheeted(LLAMA, ATTN);
    const row = held.row('output_gate');
    expect(held.site().parameters.find((slot) => slot.name === 'q')?.present).toBe(true);
    held.store.apply(writeLiteral(held.store.context, row, true, held.facts(row)));

    const site = held.site();
    expect(site.parameters.find((slot) => slot.name === 'q')?.present).toBe(false);
    expect(site.parameters.find((slot) => slot.name === 'q_gated')?.present).toBe(true);
    expect(site.parameters.find((slot) => slot.name === 'q_gated')?.boundBy).toBeNull();
    // And the validator says nothing about the binding that names the slot that went: it skips an
    // absent slot before it reads anything of it, which is why §4.17 gives the row to the editor.
    expect(site.parameters.find((slot) => slot.name === 'q')?.boundBy).toBeNull();
  });
});

describe('Pin value', () => {
  it('writes the effective value as a literal, and nothing else moves', () => {
    const held = new Sheeted(LLAMA, ATTN);
    const row = held.row('cross');
    expect(row.source).toBe('default');
    expect(row.written).toBe(false);
    held.store.apply(pinValue(held.store.context, row, held.facts(row)));

    const after = held.row('cross');
    expect(after.source).toBe('given');
    expect(after.written).toBe(true);
    expect(after.value).toBe(false);
    expect(serialize(held.store.tree)).toContain('"cross"');
  });

  it('writes a cardinality as a whole number and a real with its fraction (D12)', () => {
    const held = new Sheeted(LLAMA, ATTN);
    const heads = held.row('heads');
    held.store.apply(writeLiteral(held.store.context, heads, 16, held.facts(heads)));
    expect(serialize(held.store.tree)).toContain('"literal": 16');

    const scale = held.row('scale');
    held.store.apply(writeLiteral(held.store.context, scale, 0.25, held.facts(scale)));
    expect(serialize(held.store.tree)).toContain('"literal": 0.25');
  });
});

describe('a row put back on its default', () => {
  it('stores nothing — §4.12’s "defaults are not written"', () => {
    const held = new Sheeted(LLAMA, ATTN);
    const before = serialize(held.store.tree);
    const row = held.row('cross');
    held.store.apply(pinValue(held.store.context, row, held.facts(row)));
    expect(serialize(held.store.tree)).not.toBe(before);

    held.store.apply(clearValue(held.store.context, held.row('cross')));
    // Byte for byte what was read: the member goes from the map it was added to, and Immer's own
    // array patches put every other member back where it stood (feature 2.1).
    expect(serialize(held.store.tree)).toBe(before);
    expect(held.row('cross').source).toBe('default');
  });
});

describe('the source modes', () => {
  it('writes the member the schema discriminates the alternative by', () => {
    const held = new Sheeted(LLAMA, ATTN);
    const row = held.row('heads');
    const quantity = row.modes.find((mode) => mode.referent === 'quantity');
    expect(quantity).toBeDefined();
    held.store.apply(writeMode(held.store.context, row, quantity as never, 'layers'));
    expect(held.row('heads').value).toBe('layers');
    expect(held.row('heads').effective).toBe('32');
  });

  it('writes a typed number in the form it was typed in, not the declaration’s (V3’s own rule)', () => {
    // `rope.theta` is declared `real` and `llama3-8b` writes `500000` there — a whole number,
    // which `model.py` reads as a Python `int`, D1 carries as one and the Weisfeiler-Lehman
    // signature of `tests/signatures/llama3-8b.json` is taken over. A writer that took the
    // float-ness from the declaration (D12's parenthesis) could not spell that document at all,
    // which is what feature 2.19 met when it tried to build it.
    const held = new Sheeted(LLAMA, ATTN);
    const theta = held.row('rope.theta');
    const facts = held.facts(theta);
    expect(numberAsTyped('500000', facts)).toEqual({ kind: 'number', value: 500000, real: false });
    expect(numberAsTyped('500000.0', facts)).toEqual({ kind: 'number', value: 500000, real: true });
    expect(numberAsTyped('1e-05', facts)).toEqual({ kind: 'number', value: 1e-5, real: true });
    // A field being typed in is not a value: an empty one, a half-written exponent, an infinity.
    for (const text of ['', ' ', '1e', 'Infinity', 'x']) {
      expect(numberAsTyped(text, facts), text).toBeNull();
    }

    // And what it writes is what the document then holds, lexeme and all (D12).
    held.store.apply(writeTypedNumber(held.store.context, theta, '500000', facts));
    expect(serialize(held.store.tree)).toContain('"literal": 500000\n');
    held.store.apply(writeTypedNumber(held.store.context, held.row('rope.theta'), '5e5', facts));
    expect(serialize(held.store.tree)).toContain('"literal": 500000.0\n');

    // A value the editor *computes* is still written in the form the declaration asks for, which
    // is D12's own rule and is untouched: `Pin value` on a real writes a real.
    const scale = held.row('scale');
    expect(blankOf(held.facts(scale), scale.options)).toBe(0);
    held.store.apply(writeLiteral(held.store.context, scale, 0.5, held.facts(scale)));
    expect(serialize(held.store.tree)).toContain('"scale": {');
  });

  it('starts a literal from the first value the place admits, never from a default of its own', () => {
    const held = new Sheeted(LLAMA, ATTN);
    const mask = held.row('mask');
    expect(blankOf(held.facts(mask), mask.options)).toBe('causal');
    const scale = held.row('scale');
    expect(blankOf(held.facts(scale), scale.options)).toBe(0);
    const cross = held.row('cross');
    expect(blankOf(held.facts(cross), cross.options)).toBe(false);
  });
});
