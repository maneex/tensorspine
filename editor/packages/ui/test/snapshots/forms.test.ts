import { describe, expect, it } from 'vitest';

import { formLines, noteLines } from '../../src/forms/render.js';
import { formOf } from '../../src/forms/walk.js';
import { context, DERIVED, DOCUMENTATION, MODEL, UNIT } from '../forms/source.js';
import { registry } from '../presentation/source.js';

// Plan §1's catching rule (c): "The forms are exercised on every `$def` of every schema in a
// snapshot test, so a schema change shows up as a diff, not as a crash."
//
// This is that test, and it is the whole point of the feature. Every definition of the four
// schemas the plan names is walked with no value — the *shape* of the form, before anything is
// written — and the rows are recorded as lines. A line is `<indent><widget> <label>` followed by
// what the schema asserts: `required`, the branch a conditional requirement hangs on, a select's
// options, a chooser's modes with the widget each opens, the bounds, the name definition a map's
// keys are, and the reason a row renders generically when one applies.
//
// What a diff here means:
//
//   * a `$def` added or removed to a schema — a new block, or one gone;
//   * a `type`, `enum`, `const`, `required`, `minimum`, `pattern` or `propertyNames` changed — the
//     marks on one line;
//   * a `oneOf` gaining or losing an alternative — the modes in braces;
//   * a presentation binding added or moved — the widget on the left of a line;
//   * and, the case the rule is really for, a schema construct the walker has no reading of — a
//     `generic:` mark and a line in the notes, rather than a crash in front of a user.
//
// The definitions are read from the schema files themselves, in the order each file writes them,
// so a schema that grows a definition grows this snapshot without anybody adding a name.

interface Document {
  readonly $defs?: Record<string, unknown>;
}

/** The names a schema's `$defs` declares, in the order the file writes them. */
function definitionsOf(id: string): readonly string[] {
  const document = registry().byId(id)?.document as Document | undefined;
  return Object.keys(document?.$defs ?? {});
}

/** The form of one place, as lines. */
function lines(anchor: string, label: string): readonly string[] {
  return formLines(formOf(context(), { anchor, label }));
}

/** Every `$def` of a schema, as a map from its name to the rows its form has. */
function formsOf(id: string): Record<string, readonly string[]> {
  const found: Record<string, readonly string[]> = {};
  for (const name of definitionsOf(id)) found[name] = lines(`${id}#/$defs/${name}`, name);
  return found;
}

/** Every note every `$def` of a schema raises, in the order the walks raise them. */
function notesOf(id: string): readonly string[] {
  const found: string[] = [];
  for (const name of definitionsOf(id)) {
    for (const line of noteLines(formOf(context(), { anchor: `${id}#/$defs/${name}`, label: name }))) {
      found.push(`${name}: ${line}`);
    }
  }
  return found;
}

describe('every `$def` of the model schema renders', () => {
  it('has these rows', () => {
    expect(formsOf(MODEL)).toMatchSnapshot();
  });
});

describe('every `$def` of the primitive-library-unit schema renders', () => {
  it('has these rows', () => {
    expect(formsOf(UNIT)).toMatchSnapshot();
  });
});

describe('every `$def` of the derived schema renders', () => {
  it('has these rows', () => {
    expect(formsOf(DERIVED)).toMatchSnapshot();
  });
});

describe('every `$def` of the documentation schema renders', () => {
  it('has these rows', () => {
    expect(formsOf(DOCUMENTATION)).toMatchSnapshot();
  });
});

describe('the documents themselves render', () => {
  it('has these rows at the root of each schema that is a document', () => {
    // The model schema's root is the Document sheet of §4.11; the unit schema's is the primitive
    // editor's header (§4.22), and it is the one the root's `allOf` used to hide.
    expect({
      model: lines(`${MODEL}#`, 'model'),
      unit: lines(`${UNIT}#`, 'unit'),
      derived: lines(`${DERIVED}#`, 'derived'),
    }).toMatchSnapshot();
  });
});

describe('what the walker had no reading for', () => {
  it('lists it, once per construct, and never crashes (plan §1)', () => {
    expect({
      model: notesOf(MODEL),
      unit: notesOf(UNIT),
      derived: notesOf(DERIVED),
      documentation: notesOf(DOCUMENTATION),
    }).toMatchSnapshot();
  });
});
