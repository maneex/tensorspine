import { describe, expect, it } from 'vitest';

import { auditPresentation } from '../../src/presentation/audit.js';
import { presentation } from '../../src/presentation/load.js';
import { registry } from '../presentation/source.js';

// The snapshot layer holds "the generated forms and the rendered strings", and the first of them
// is the answer to plan §1 (a)'s second half: **what renders generically**.
//
// Most of the language does, and should — a dtype is a plain select, `min` prints as `min(a, b)`
// because its symbol is its own name, and a `oneOf` the interface has no editor for is a chooser
// over its alternatives' tags. None of that is a fault. What would be a fault is not knowing:
// a construct that quietly lost its binding, or a schema that grew a value nobody looked at.
// Recording the list makes either of those a reviewed diff.
//
// The three snapshots are the three ways it can move: the anchors the file binds, the enumerated
// values no binding names, and the union members no binding names.

const audit = auditPresentation(registry(), presentation());

/** The enumerated values no binding names, by the enumeration each belongs to. */
function enumerations(): Record<string, string[]> {
  const found: Record<string, string[]> = {};
  for (const one of audit.generic) {
    if (one.kind !== 'enumeration') continue;
    (found[one.anchor] ??= []).push(one.name);
  }
  return found;
}

/** The union members no binding names: one line each, the alternative's place and its tags. */
function alternatives(): string[] {
  return audit.generic
    .filter((one) => one.kind === 'alternative')
    .map((one) => `${one.anchor} ${one.name}`);
}

describe('presentation.json', () => {
  it('binds the anchors the snapshot records', () => {
    expect(presentation().anchors).toMatchSnapshot();
  });
});

describe('what the schemas carry and no binding names', () => {
  it('renders these enumerated values generically', () => {
    expect(enumerations()).toMatchSnapshot();
  });

  it('renders these union members generically', () => {
    expect(alternatives()).toMatchSnapshot();
  });
});
