import type { PyValue } from '../../src/index.js';

/**
 * Applying the oracle's edits: one walk, shared by the suites that read an edited document.
 *
 * Features 1.3, 1.4 and 1.5 all record their cases the same way — *a pointer into a repository
 * file with a few values changed* — so that both implementations read the same bytes and apply
 * the same change, and neither side owns the input. This is that application, on the reading
 * (`toPython`) rather than on the tree, because the edited document is what the ported functions
 * take.
 *
 * How a fixture value is read is the caller's: 1.4's edits are plain JSON whole numbers, 1.5's are
 * the tagged encoding of `encoding.ts` (a case there turns on whether a value is `4096` or
 * `4096.0`). Hence the `read` parameter rather than one reading built in.
 */

/** One edit of a recorded case, as the oracle writes it. */
export interface Edit {
  pointer: string;
  op: 'set' | 'delete';
  value?: unknown;
}

/**
 * `record[name] = value`, safe for the one name JavaScript reads as the prototype.
 *
 * The model schema's `propertyNames` pattern admits `__proto__` (features 0.3, 1.1 and 1.4 each
 * met it), and plain assignment would set the prototype instead of adding a member.
 */
export function put(record: Record<string, PyValue>, name: string, value: PyValue): void {
  if (name === '__proto__') {
    Object.defineProperty(record, name, {
      value,
      writable: true,
      enumerable: true,
      configurable: true,
    });
  } else {
    record[name] = value;
  }
}

/**
 * One edit applied in place, at the pointer the fixture writes.
 *
 * The pointer is split and nothing is unescaped: no member name of a model document holds `/` or
 * `~`, which is the reading the generator takes too.
 */
export function applyEdit(
  root: PyValue,
  edit: Edit,
  read: (value: unknown) => PyValue,
  where: string,
): void {
  const steps = edit.pointer.split('/').slice(1);
  let cursor = root;
  for (const step of steps.slice(0, -1)) {
    cursor = Array.isArray(cursor)
      ? ((cursor as readonly PyValue[])[Number(step)] as PyValue)
      : ((cursor as Record<string, PyValue>)[step] as PyValue);
    if (cursor === undefined) throw new Error(`${where}: ${edit.pointer} names nothing`);
  }
  const last = steps[steps.length - 1] as string;
  if (Array.isArray(cursor)) {
    const list = cursor as PyValue[];
    if (edit.op === 'delete') list.splice(Number(last), 1);
    else list[Number(last)] = read(edit.value);
    return;
  }
  const record = cursor as Record<string, PyValue>;
  if (edit.op === 'delete') delete record[last];
  else put(record, last, read(edit.value));
}

/** Every edit of one case, in the order it was recorded. */
export function applyEdits(
  root: PyValue,
  edits: readonly Edit[],
  read: (value: unknown) => PyValue,
  where: string,
): void {
  for (const edit of edits) applyEdit(root, edit, read, where);
}
