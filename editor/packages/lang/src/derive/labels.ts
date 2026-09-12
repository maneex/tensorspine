/**
 * The value-type string of the inventory: `bf16[tokens, model.width=4096]`, `i32[tokens]`.
 *
 * This is a *label*, not a product: it renders one D2 value entry as a diagram prints it. It is in
 * the core, and not in a component, for the reason the plan's finding F6 gives — the conventions
 * live in `tools/view.py` today, `view.py` is to be removed once the editor is deployed, and "the
 * editor restates them from the derived document". The one-time check of that restatement is the
 * fixture `test/derive/view-labels.json`: the labels `--view` itself wrote for `llama3-8b`,
 * `voxtral-realtime` and `qwen3.5-35b-a3b`, recorded while the tool is still there.
 *
 * The convention, in `view.py`'s own words:
 *
 * > "A value is an array: its leading axis is the stream it lives on, and the shape D2 gives is
 * > the geometry of ONE element on it. That leading axis has no extent — how many elements an
 * > invocation carries is deployment intent, never a number the language states (§10.3) — so it is
 * > printed by name, at the count saying how many of the stream's own elements one of these is
 * > worth."
 *
 * So the string is the dtype, then the axes: the stream axis first (from the value's `count`, and
 * absent where there is none), then one `axis=extent` per axis of one element. "An extent is a
 * dimension, not a total, so it is printed whole, without the digit grouping a figure carries" —
 * and an extent D2 left undetermined prints as Python prints `None`, which is what a reader needs
 * to see rather than a blank.
 *
 * It reads a D2 value as data: the entry the core computed, or one parsed from a derived document
 * a workspace holds. Nothing of the schemas is named here (plan §1): `dtype`, `count` and `shape`
 * are the derived schema's member names, read off the value, never a vocabulary this file carries.
 */
import { pyEqual } from '../expr/arithmetic.js';
import { truthy, type PyValue } from '../expr/value.js';
import { demand, entries, listOf, optional } from '../library/access.js';
import { pyStr } from '../library/repr.js';

/**
 * `stream_axis(count)`: a value's leading axis, named by the stream it counts against.
 *
 * "One element per element of the stream is the stream's own name, one per eight is `audio/8`, and
 * a stream a transform inserted into another is a sum." A count that is neither one nor the
 * reciprocal of a whole number is written as a multiplier, `pixels×0.3` — no count of the corpus
 * is one, every one of them being `1.0` or a power of two's reciprocal.
 */
export function streamAxis(count: PyValue): string {
  const parts: string[] = [];
  // `(count or {}).items()`: a value with no count has no stream axis at all.
  for (const [name, value] of entries(truthy(count) ? count : {})) {
    if (pyEqual(value, 1n)) {
      parts.push(name);
      continue;
    }
    const reciprocal = truthy(value) ? 1 / Number(value) : Number.NaN;
    if (Number.isInteger(reciprocal)) parts.push(`${name}/${BigInt(Math.trunc(reciprocal))}`);
    // `view.py`'s own `${name}×${n}`, where `n` is what `JSON.parse` made of the count: the
    // viewer this reproduces is JavaScript embedded in the tool, so a count of `2.0` reads
    // `tokens×2` there and a count of `0` reads `tokens×0` — the value's float-ness is a
    // distinction the language keeps and this label never showed. Every other figure of the
    // products is written with `pyStr`, which is why this one says why it is not.
    else parts.push(`${name}×${String(Number(value))}`);
  }
  return parts.join(' + ');
}

/**
 * `value_geometry(v)`: the type of one D2 value — "the dtype, then the axes of the array, the
 * stream axis first, then the axes of one element of it".
 */
export function valueGeometry(value: PyValue): string {
  const count = optional(value, 'count', null);
  const axes = truthy(count) ? [streamAxis(count)] : [];
  for (const axis of shapeAxes(optional(value, 'shape', null))) axes.push(axis);
  return `${pyStr(optional(value, 'dtype', ''))}[${axes.join(', ')}]`;
}

/**
 * A derived shape's axes, each as `axis=extent` — what a D3 or a D4 row shows of its tensor.
 *
 * The half of {@link valueGeometry} that is not about a stream: the same rendering, in one place,
 * so a sheet's `[attention.heads=4096, model.width=4096]` and an edge's value type agree about
 * what a shape reads as (component inventory §7 — "no component adds, converts or rounds").
 */
export function shapeAxes(shape: PyValue): string[] {
  return listOf(shape ?? []).map(
    (axis) => `${pyStr(demand(axis, 'axis'))}=${pyStr(demand(axis, 'extent'))}`,
  );
}

/** A derived shape as one line: `[attention.heads=4096, model.width=4096]`, `[]` where it has none. */
export function shapeText(shape: PyValue): string {
  return `[${shapeAxes(shape).join(', ')}]`;
}
