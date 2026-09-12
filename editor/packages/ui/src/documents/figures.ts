/**
 * The figures of the derived document, rendered — feature 2.6's half of the status bar (§4.2).
 *
 * The component inventory's §7 is the rule: **"Every figure — D1–D6. No component adds, converts
 * or rounds a byte count; `bytes` is a presentation binding over the product's own number."** So
 * nothing here computes a figure: the derived document's number is the fact, the binding says how
 * it is written, and this is the writing.
 *
 * **The renderings are `tools/view.py`'s, and that is deliberate.** The plan's finding F6 decided
 * that `view.py` is removed when the editor is deployed and that its conventions — the value-type
 * string, the figure lines — "move into the editor's presentation layer". This is that move for
 * the figure lines; the Python is quoted beside each one, as every other port keeps it.
 *
 * ```
 * def fmt_bytes(n):
 *     if not n: return '—'
 *     if n >= 2 ** 30: return f"{n / 2 ** 30:.2f} GiB"
 *     if n >= 2 ** 20: return f"{n / 2 ** 20:.1f} MiB"
 *     if n >= 1024:    return f"{n / 1024:.0f} KiB" if n >= 10240 else f"{n / 1024:.1f} KiB"
 *     return f"{fmt_int(n)} B"
 * ```
 *
 * **Why the table's keys are written and the schema's are not.** `bytes`, `elements` and
 * `operations` are values of the *editor's own* format vocabulary — `editor/schemas/
 * tensorspine-editor-presentation.schema.json` — and not of the four schemas, though two of them
 * are spelled the same as a value of the derived schema's. They are written as bare property
 * names, never as string literals, and a test holds this table's key set to the editor's own
 * schema: the same discipline §1 (d) puts on the core's semantic tables, one schema along.
 */
import { formatNumber, type PyValue } from '@tensorspine/lang';

import type { Binding, Presentation } from '../presentation/index.js';

/** What a rendered figure carries: the line, and the exact number behind it for the tooltip. */
export interface Figure {
  /** The figure as it is shown: `14.96 GiB`, `15.01 Gop`, `—`. */
  readonly text: string;
  /** The number as the derived document holds it, grouped — §4.18's "exact value in the tooltip". */
  readonly exact: string;
}

/** What nothing at all is written as — `view.py`'s own em dash. */
const NOTHING = '—';

/**
 * `fmt_int`: a whole number with its thousands separated, as `view.py` groups them.
 *
 * A number that is **not** whole is written as the language writes one — D12's `formatNumber`,
 * CPython's own float layout — with its integer part grouped, which is what `f"{n:,}"` does to a
 * float in Python too. Rounding it away would be the interface inventing a figure (the component
 * inventory's §7): D3's `activated_fraction` is `7.79690618762475e-06` and reads as itself, which
 * is what feature 2.15's tables showed the first time one of them was drawn.
 */
export function groupedNumber(value: number): string {
  if (!Number.isFinite(value)) return String(value);
  const text = formatNumber(value, !Number.isInteger(value));
  const point = text.search(/[.e]/);
  const whole = point < 0 ? text : text.slice(0, point);
  return `${whole.replace(/\B(?=(\d{3})+(?!\d))/g, ' ')}${point < 0 ? '' : text.slice(point)}`;
}

/** `fmt_bytes`: B, KiB, MiB, GiB — the sizes §4.18 names, at `view.py`'s own precisions. */
export function sizeText(value: number): string {
  if (!value) return NOTHING;
  if (value >= 2 ** 30) return `${(value / 2 ** 30).toFixed(2)} GiB`;
  if (value >= 2 ** 20) return `${(value / 2 ** 20).toFixed(1)} MiB`;
  if (value >= 1024) return `${(value / 1024).toFixed(value >= 10240 ? 0 : 1)} KiB`;
  return `${groupedNumber(value)} B`;
}

/** `fmt_ops`: op, Mop, Gop. */
export function operationsText(value: number): string {
  if (!value) return NOTHING;
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)} Gop`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)} Mop`;
  return `${groupedNumber(value)} op`;
}

/** A count of elements, on the same scale and with no unit of its own. */
export function elementsText(value: number): string {
  if (!value) return NOTHING;
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)} G`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)} M`;
  return groupedNumber(value);
}

/**
 * The renderings, by the format the binding names.
 *
 * Three of the six the editor's presentation schema admits. `count`, `status` and `shape` are the
 * Derived panel's (feature 2.15) and are not written here on faith: a format with no rendering
 * falls back to the product's own number, which is §1's "unknown constructs get the generic
 * widget" applied to a figure.
 */
const RENDERINGS: Readonly<Record<string, (value: number) => string>> = {
  bytes: sizeText,
  elements: elementsText,
  operations: operationsText,
};

/** Every format this module renders, for the test that holds it to the editor's own schema. */
export function renderedFormats(): string[] {
  return Object.keys(RENDERINGS).sort();
}

/** A number under a binding's format, with the exact value beside it. */
export function figureOf(value: number, binding: Binding | undefined): Figure {
  const render = binding?.format === undefined ? undefined : RENDERINGS[binding.format];
  return {
    text: render === undefined ? groupedNumber(value) : render(value),
    exact: groupedNumber(value),
  };
}

/** One field of the status bar: what §4.2 puts there, as the bar draws it. */
export interface StatusFigure {
  /** Where it sits, from the binding. */
  readonly order: number;
  /** What is written beside it, from the binding — an English key of the dictionary. */
  readonly label: string;
  readonly figure: Figure;
  /** The epistemic status beside a qualified value, where the figure carries one. */
  readonly status?: string;
  /** The anchor the binding was read at, so a field can say what it is showing. */
  readonly anchor: string;
}

/** How deep the walk goes: the four fields sit at depth three, a qualified value at four. */
const DEPTH = 4;

/**
 * The status bar's figures, found in the derived document rather than named in the bar.
 *
 * The walk is over the *document* — the products the core derived — and it asks the schema
 * reading for the place it is at, exactly as a form does. A place whose binding carries
 * `statusBar` is a field; the number is the value there, or, where the value is a qualified one,
 * the member the schema declares as a number, with its status beside it.
 *
 * Bounded to {@link DEPTH} because the fields are totals and totals sit near the root: a walk of
 * the whole document would visit D1's two hundred nodes to find four numbers.
 */
export function statusFigures<S>(
  derived: PyValue,
  shapes: FigureShapes<S>,
  bindings: Presentation,
): StatusFigure[] {
  const found: StatusFigure[] = [];
  const walk = (value: PyValue, shape: S, depth: number): void => {
    const anchors = shapes.anchors(shape);
    const binding = bindings.firstOf(anchors);
    const field = binding?.statusBar;
    if (field !== undefined) {
      const read = numberIn(value, shape, shapes, bindings);
      if (read !== null) {
        found.push({
          order: field.order,
          label: field.label,
          figure: figureOf(read.value, binding),
          anchor: anchors[0] ?? '',
          ...(read.status === undefined ? {} : { status: read.status }),
        });
      }
      return;
    }
    if (depth >= DEPTH || !isRecord(value)) return;
    for (const [name, member] of Object.entries(value)) {
      walk(member, shapes.member(shape, name), depth + 1);
    }
  };
  walk(derived, shapes.root(), 0);
  return found.sort((a, b) => a.order - b.order);
}

/** What a figure's place answers: the number, and the status a qualified value carries. */
function numberIn<S>(
  value: PyValue,
  shape: S,
  shapes: FigureShapes<S>,
  bindings: Presentation,
): { value: number; status?: string } | null {
  if (typeof value === 'number') return { value };
  if (typeof value === 'bigint') return { value: Number(value) };
  if (!isRecord(value)) return null;
  // A qualified value: the member the schema declares as a number is the figure, and the one the
  // presentation shows as an epistemic status is the chip. Both are read from the schema, so
  // neither member is named here.
  let number: number | null = null;
  let status: string | undefined;
  for (const [name, member] of Object.entries(value)) {
    const under = shapes.member(shape, name);
    if (typeof member === 'number' || typeof member === 'bigint') {
      if (shapes.holdsNumber(under)) number = Number(member);
      continue;
    }
    if (typeof member === 'string' && bindings.firstOf(shapes.anchors(under))?.format === STATUS) {
      status = member;
    }
  }
  return number === null ? null : { value: number, ...(status === undefined ? {} : { status }) };
}

/**
 * The format that shows an epistemic status as a chip.
 *
 * A value of the *editor's* format vocabulary, and the one place this module has to compare one:
 * a qualified value is a number and a status, and which member is which is what the schema and
 * this binding say between them.
 */
const STATUS = 'status';

/**
 * What the walk needs of the schemas: a place, its member, its anchors, and whether it is a
 * number.
 *
 * An interface over the store's `SchemaShapes` rather than the class itself, so that this
 * module — which renders — carries no dependency on the store's reading of a document, and so
 * that a suite can walk a schema of its own.
 */
export interface FigureShapes<S> {
  /** The derived document's root. */
  root(): S;
  /** The place of a member. */
  member(shape: S, name: string): S;
  /** The anchors of a place, most specific first — what a binding is looked up by. */
  anchors(shape: S): readonly string[];
  /** Whether the schema at that place admits a number. */
  holdsNumber(shape: S): boolean;
}

function isRecord(value: PyValue): value is Readonly<Record<string, PyValue>> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
