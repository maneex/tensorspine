/**
 * The checkpoint half of V17: the port of `artifact.check` (plan §5.3, `checkCheckpoint`).
 *
 * §6, V17: "Against a checkpoint: every located tensor exists with the D3 shape — unit axes the
 * physical tensor has and the logical shape lacks being dropped — and the D3 dtype (I9)." That is
 * the whole of what this decides, and it decides it over two things and nothing else: the derived
 * document's **D3** (the Parameter Tensor Inventory, feature 1.8a) and a map of safetensors header
 * entries (`header.ts`). No file is read here, no location is evaluated here — `evaluate_location`
 * did that during validation (feature 1.6c) and D3 carries its answer.
 *
 * ## Errors, warnings, stats
 *
 * `artifact.check` answers `(errors, advisories, stats)`. The plan asks for the same three under
 * the editor's names (§4.19, "Checks"): "errors for absent tensors, shape or dtype mismatches,
 * stack counts, concat sums, slices that do not fit; **warnings** for every physical tensor no
 * location names (the model does not use it) and for every identity without a location while
 * others have one".
 *
 * - The **errors** are the tools' `errors`, in their order, word for word. `tests/run_artifact.py`
 *   matches substrings of them and the parity suite compares them whole, so the wording is a
 *   contract (plan §7 F1) and is not rephrased. Beside the words each problem carries the identity,
 *   the physical tensor and the file, because plan §3 asks the core to emit the pointer natively.
 * - The **warnings** lead with the tools' `advisories`, in the tools' order and wording — one per
 *   physical tensor no location names, which is Q7's "what must not be lost". After them come the
 *   editor's own: one per identity the document leaves unlocated while others are located. That
 *   second kind is the plan's, not the tools': it has no wording to reproduce, and it is stated
 *   here to be **unreachable from a valid document**, since V17 makes locations total or absent —
 *   a derived document exists only for a document that validated. It is emitted all the same,
 *   because `checkCheckpoint` is a function of a derived document and a header map, and a panel
 *   that met one should say so rather than stay silent.
 * - The **stats** are exactly the tools' three (`located`, `physical`, `unnamed`), which is what
 *   §4.19's header line and the status bar read: `738 tensors · 736 located · 2 named by no
 *   location`.
 *
 * ## Where the tools raise, this raises
 *
 * Two places, both unreachable through `derive` because the derived schema admits exactly four
 * location forms and 1.8e validates the emitted document against it, and both reproduced rather
 * than tidied — a caller may hand in a D3 it built itself:
 *
 * - a location form that is none of the four is one error line *and then* a `KeyError('concat')`
 *   out of `_names`, which walks the parts of a form it assumes is a `stack` or a `concat`;
 * - a `slice` along an axis whose logical extent is 1 raises `ValueError: <dim> is not in list`,
 *   the position list being built from the non-unit extents alone.
 *
 * ## One clause that cannot decide
 *
 * The whole-tensor test is `squeeze(h['shape']) != squeeze(logical) or len(h['shape']) <
 * len(squeeze(logical))`. Its second half is **dead**: it is reached only when the first is false,
 * and `squeeze(a) == squeeze(b)` makes the two squeezes the same length while a list is never
 * shorter than its own squeeze, so `len(h['shape']) >= len(squeeze(logical))` always holds there.
 * It is ported as written, as every dead branch of the tools has been, and the proof is a case of
 * the unit suite rather than a comment nobody checks. A finding for the tools.
 */
import { pyAdd, pyEqual, pyOrder } from '../expr/arithmetic.js';
import { PyIndexError, PyTypeError, PyValueError } from '../expr/errors.js';
import { isRecord, pythonTypeName, type PyValue } from '../expr/value.js';
import { demand, listOf, optional } from '../library/access.js';
import { pyRepr, pyStr } from '../library/repr.js';
import { comparePythonStrings } from '../schema/repr.js';
import { headerAt, type CheckpointHeaders, type HeaderEntry } from './header.js';

/** What a checkpoint problem is about, for a panel that groups or filters them. */
export type CheckpointProblemKind =
  /** A physical tensor a location names and the checkpoint has not. */
  | 'absent'
  /** A physical tensor whose shape is not the one D3 declares, unit axes dropped. */
  | 'shape'
  /** A physical tensor whose dtype is not D3's. */
  | 'dtype'
  /** The same, where the reader has no mapping for the checkpoint's dtype (`dtypes.ts`). */
  | 'dtype-unknown'
  /** A `stack` whose parts do not number the extent of the axis it stacks along. */
  | 'stack'
  /** A `concat` whose parts' extents do not sum to the extent of the axis. */
  | 'concat'
  /** A `slice` that does not fit the physical tensor it slices. */
  | 'slice'
  /** A location form that is none of the four (§3.4). */
  | 'form'
  /** A physical tensor no location names: the model does not use it (Q7). */
  | 'unnamed'
  /** An identity with no location while others have one; V17 makes it unreachable when valid. */
  | 'unlocated';

/** One verdict of the checkpoint check: the tools' line, and what it is about. */
export interface CheckpointProblem {
  /** The rule, as the tools' printer prefixes it: `[V17]`. */
  readonly code: 'V17';
  /** The tools' words, exactly, without that prefix. */
  readonly message: string;
  /** Whether the check fails on it (§4.19). */
  readonly severity: 'error' | 'warning';
  /** What the row is about, for the panel. */
  readonly kind: CheckpointProblemKind;
  /** The D3 identity the row is about; absent for a row about a physical tensor alone. */
  readonly identity?: string;
  /** The physical tensor the row names; absent for a row about an identity alone. */
  readonly tensor?: string;
  /** The checkpoint file that tensor is in, when the headers know it. */
  readonly file?: string;
}

/** The three counters `artifact.check` answers, and what §4.19's header line reads. */
export interface CheckpointStats {
  /** D3 entries carrying an evaluated location. */
  readonly located: number;
  /** Tensors the checkpoint holds. */
  readonly physical: number;
  /** Physical tensors no location names. */
  readonly unnamed: number;
}

/** `(errors, advisories, stats)`, under the editor's names. */
export interface CheckpointCheck {
  readonly errors: readonly CheckpointProblem[];
  /** The tools' advisories first, in their order, then the unlocated identities. */
  readonly warnings: readonly CheckpointProblem[];
  readonly stats: CheckpointStats;
}

/** `f"[{code}] {message}"`: the line `artifact.run` prints for an error. */
export function formatCheckpointProblem(problem: CheckpointProblem): string {
  return `[${problem.code}] ${problem.message}`;
}

/** `[d for d in shape if d != 1]`: the unit axes V17 drops before two shapes are compared. */
export function squeeze(shape: readonly PyValue[]): PyValue[] {
  return shape.filter((extent) => !pyEqual(extent, 1n));
}

/**
 * V17 against a checkpoint: `checkCheckpoint(derived, headers)`.
 *
 * `derived` is the derived document — `derive`'s own answer (§5.3) — whose `d3` member is what
 * the tools hand `artifact.check`.
 */
export function checkCheckpoint(derived: PyValue, headers: CheckpointHeaders): CheckpointCheck {
  const d3 = demand(derived, 'd3');
  const errors: CheckpointProblem[] = [];
  const used = new Set<string>();
  const unlocated: string[] = [];
  let located = 0;

  for (const entry of listOf(demand(d3, 'tensors'))) {
    const identity = pyStr(demand(entry, 'identity'));
    const location = optional(entry, 'location', null);
    if (location === null) {
      unlocated.push(identity);
      continue;
    }
    located += 1;
    const logical = listOf(demand(entry, 'shape')).map((axis) => demand(axis, 'extent'));
    checkPart(location, logical, demand(entry, 'dtype'), headers, identity, errors);
    for (const name of namesOf(location)) used.add(name);
  }

  const unnamed = Object.keys(headers)
    .filter((name) => !used.has(name))
    .sort(comparePythonStrings);
  const warnings: CheckpointProblem[] = unnamed.map((name) => {
    const entry = headerAt(headers, name) as HeaderEntry;
    return {
      code: 'V17',
      message:
        `physical tensor '${name}' (${pyStr(entry.dtype)} ${pyStr(entry.shape)}) ` +
        'is named by no location',
      severity: 'warning',
      kind: 'unnamed',
      tensor: name,
      file: entry.file,
    };
  });
  // "for every identity without a location while others have one" (§4.19). A document that
  // locates none of its identities is not partly located: V17's "totals or absent" is satisfied,
  // and the tools' own report says "no location: nothing to check against <checkpoint>".
  if (located > 0) {
    for (const identity of unlocated) {
      warnings.push({
        code: 'V17',
        message: `identity '${identity}' has no location while others have one`,
        severity: 'warning',
        kind: 'unlocated',
        identity,
      });
    }
  }

  return {
    errors,
    warnings,
    stats: { located, physical: Object.keys(headers).length, unnamed: unnamed.length },
  };
}

/**
 * One evaluated location against the shape it must fill: `_check_part`.
 *
 * The tools' function returns "the physical extent it contributes along a concat axis, when
 * asked"; no caller reads the return, the concat branch computing the extent itself, so the port
 * answers nothing and says so here rather than carrying a value nobody uses.
 */
function checkPart(
  location: PyValue,
  logical: readonly PyValue[],
  dtype: PyValue,
  headers: CheckpointHeaders,
  identity: string,
  errors: CheckpointProblem[],
): void {
  if (has(location, 'tensor')) {
    const name = pyStr(demand(location, 'tensor'));
    const entry = headerAt(headers, name);
    if (entry === undefined) {
      errors.push(absent(identity, name));
      return;
    }
    const want = squeeze(logical);
    if (!pyEqual(squeeze(entry.shape), want) || entry.shape.length < want.length) {
      errors.push({
        code: 'V17',
        message:
          `${identity}: '${name}' has shape ${pyStr(entry.shape)}, ` +
          `the document says ${pyStr(logical)}`,
        severity: 'error',
        kind: 'shape',
        identity,
        tensor: name,
        file: entry.file,
      });
    }
    dtypeProblem(entry, dtype, identity, name, errors);
    return;
  }

  if (has(location, 'stack')) {
    const stack = demand(location, 'stack');
    const dim = demand(stack, 'dim');
    const parts = listOf(demand(stack, 'parts'));
    const inner = withoutDim(logical, dim);
    const extent = itemAt(logical, dim);
    if (!pyEqual(BigInt(parts.length), extent)) {
      errors.push({
        code: 'V17',
        message:
          `${identity}: stack of ${String(parts.length)} along ` +
          `'${pyStr(demand(stack, 'axis'))}' of extent ${pyStr(extent)}`,
        severity: 'error',
        kind: 'stack',
        identity,
      });
    }
    for (const part of parts) checkPart(part, inner, dtype, headers, identity, errors);
    return;
  }

  if (has(location, 'concat')) {
    const concat = demand(location, 'concat');
    const dim = demand(concat, 'dim');
    let total: PyValue = 0n;
    for (const part of listOf(demand(concat, 'parts'))) {
      // "the part's extent along the axis is its own: take it from the checkpoint"
      const whole = namesIn(part)[0][0];
      const entry = whole === undefined ? undefined : headerAt(headers, whole);
      if (whole === undefined || entry === undefined) {
        checkPart(part, logical, dtype, headers, identity, errors); // reports the absence
        return;
      }
      const physical = squeeze(entry.shape);
      const want = squeeze(logical);
      if (physical.length !== want.length) {
        errors.push({
          code: 'V17',
          message:
            `${identity}: '${whole}' has shape ${pyStr(entry.shape)}, ` +
            `a part of ${pyStr(logical)}`,
          severity: 'error',
          kind: 'shape',
          identity,
          tensor: whole,
          file: entry.file,
        });
        return;
      }
      const own = [...logical];
      const extent = pyEqual(itemAt(logical, dim), 1n)
        ? 1n
        : itemAt(physical, BigInt(positionOf(logical, dim)));
      own[indexOf(own, dim)] = extent;
      checkPart(part, own, dtype, headers, identity, errors);
      total = pyAdd(total, extent);
    }
    const extent = itemAt(logical, dim);
    if (!pyEqual(total, extent)) {
      errors.push({
        code: 'V17',
        message:
          `${identity}: concat parts sum to ${pyStr(total)} along ` +
          `'${pyStr(demand(concat, 'axis'))}', the document says ${pyStr(extent)}`,
        severity: 'error',
        kind: 'concat',
        identity,
      });
    }
    return;
  }

  if (has(location, 'slice')) {
    const slice = demand(location, 'slice');
    const name = pyStr(demand(slice, 'tensor'));
    const entry = headerAt(headers, name);
    if (entry === undefined) {
      errors.push(absent(identity, name));
      return;
    }
    const physical = squeeze(entry.shape);
    const want = squeeze(logical);
    if (physical.length !== want.length) {
      errors.push({
        code: 'V17',
        message:
          `${identity}: '${name}' has shape ${pyStr(entry.shape)}, ` +
          `sliced for ${pyStr(logical)}`,
        severity: 'error',
        kind: 'shape',
        identity,
        tensor: name,
        file: entry.file,
      });
      return;
    }
    // The tools' order, kept: the position of the sliced axis among the non-unit ones, then the
    // other axes, then the offset and the extent — so an off-schema slice raises where they raise.
    const position = positionOf(logical, demand(slice, 'dim'));
    const others = (values: readonly PyValue[]): PyValue[] =>
      values.filter((_, index) => index !== position);
    const aligned = pyEqual(others(physical), others(want));
    const offset = demand(slice, 'offset');
    const end = pyAdd(offset, demand(slice, 'extent'));
    if (!aligned || pyOrder(itemAt(physical, BigInt(position)), end, '<') === -1) {
      errors.push({
        code: 'V17',
        message:
          `${identity}: '${name}' has shape ${pyStr(entry.shape)}; the slice ` +
          `[${pyStr(offset)}, ${pyStr(end)}) along '${pyStr(demand(slice, 'axis'))}' ` +
          `does not fit ${pyStr(logical)}`,
        severity: 'error',
        kind: 'slice',
        identity,
        tensor: name,
        file: entry.file,
      });
    }
    dtypeProblem(entry, dtype, identity, name, errors);
    return;
  }

  errors.push({
    code: 'V17',
    message: `${identity}: unknown location form ${pyRepr(Object.keys(isRecord(location) ? location : {}))}`,
    severity: 'error',
    kind: 'form',
    identity,
  });
}

/** `'name' in ev` on an evaluated location. */
function has(location: PyValue, name: string): boolean {
  return isRecord(location) && Object.hasOwn(location, name);
}

/** The absence of a physical tensor a location names. */
function absent(identity: string, name: string): CheckpointProblem {
  return {
    code: 'V17',
    message: `${identity}: physical tensor '${name}' is absent from the checkpoint`,
    severity: 'error',
    kind: 'absent',
    identity,
    tensor: name,
  };
}

/**
 * `h['dtype'] != dtype`, and the kind the disagreement gets.
 *
 * The comparison is the tools' own — the name the reader answered against the name D3 selected —
 * so a header whose dtype the table does not carry disagrees with every dtype of the language and
 * is refused. What changes is the **kind**: `dtype-unknown` says the reader has no mapping for the
 * checkpoint's name, which is a different thing to tell an author than a genuine mismatch, and it
 * is what `artifact/dtypes.ts` decided the pass-through would be marked with.
 */
function dtypeProblem(
  entry: HeaderEntry,
  dtype: PyValue,
  identity: string,
  name: string,
  errors: CheckpointProblem[],
): void {
  if (pyEqual(entry.dtype, dtype)) return;
  errors.push({
    code: 'V17',
    message: `${identity}: '${name}' is ${pyStr(entry.dtype)}, the document says ${pyStr(dtype)}`,
    severity: 'error',
    kind: entry.known ? 'dtype' : 'dtype-unknown',
    identity,
    tensor: name,
    file: entry.file,
  });
}

/** `w, s = _names(ev)` then `used.update(w); used.update(s)`: every physical name a location binds. */
function namesOf(location: PyValue): string[] {
  const [whole, sliced] = namesIn(location);
  return [...whole, ...sliced];
}

/**
 * `_names(ev)` as the tools write it: the whole names and the sliced ones, in two lists.
 *
 * The concat branch reads the first *whole* name of a part — `headers.get(names[0]) if names` —
 * and `check` reads both lists, so the pair is answered rather than one list. A form that is none
 * of the four raises `KeyError('concat')`, which is what `ev['concat']` does there: the
 * fall-through the tools never guarded.
 */
function namesIn(location: PyValue): [string[], string[]] {
  const whole: string[] = [];
  const sliced: string[] = [];
  collectNames(location, whole, sliced);
  return [whole, sliced];
}

function collectNames(location: PyValue, whole: string[], sliced: string[]): void {
  if (has(location, 'tensor')) {
    whole.push(pyStr(demand(location, 'tensor')));
    return;
  }
  if (has(location, 'slice')) {
    sliced.push(pyStr(demand(demand(location, 'slice'), 'tensor')));
    return;
  }
  const key = has(location, 'stack') ? 'stack' : 'concat';
  for (const part of listOf(demand(demand(location, key), 'parts'))) {
    collectNames(part, whole, sliced);
  }
}

/** `values[index]`: Python's list indexing, negatives from the end, `IndexError` past either. */
function itemAt(values: readonly PyValue[], index: PyValue): PyValue {
  return values[indexOf(values, index)] as PyValue;
}

/** The position `values[index]` denotes, as Python resolves it. */
function indexOf(values: readonly PyValue[], index: PyValue): number {
  if (typeof index !== 'bigint') {
    throw new PyTypeError(
      `list indices must be integers or slices, not ${pythonTypeName(index)}`,
    );
  }
  const at = index < 0n ? Number(index) + values.length : Number(index);
  if (at < 0 || at >= values.length) {
    throw new PyIndexError('list index out of range');
  }
  return at;
}

/** `[i for i, d in enumerate(logical) if d != 1].index(dim)`: the axis's place among the non-unit ones. */
function positionOf(logical: readonly PyValue[], dim: PyValue): number {
  const positions = logical
    .map((extent, index) => (pyEqual(extent, 1n) ? null : BigInt(index)))
    .filter((index): index is bigint => index !== null);
  const found = positions.findIndex((index) => pyEqual(index, dim));
  if (found < 0) throw new PyValueError(`${pyRepr(dim)} is not in list`);
  return found;
}

/** `logical[:dim] + logical[dim + 1:]`, with Python's slice bounds. */
function withoutDim(logical: readonly PyValue[], dim: PyValue): PyValue[] {
  if (typeof dim !== 'bigint') {
    throw new PyTypeError(`slice indices must be integers or None, not ${pythonTypeName(dim)}`);
  }
  const bound = (index: bigint): number => {
    const at = index < 0n ? Number(index) + logical.length : Number(index);
    return Math.min(Math.max(at, 0), logical.length);
  };
  return [...logical.slice(0, bound(dim)), ...logical.slice(bound(dim + 1n))];
}
