import { describe, expect, it } from 'vitest';

import { parse } from '../../src/json/index.js';
import {
  PyKeyError,
  resolveQuantities,
  toPython,
  type PyRecord,
  type Quantities,
} from '../../src/expr/index.js';
import {
  assignmentNeeded,
  checkAssignment,
  checkQuantities,
  formatSemanticProblems,
  variableQuantities,
} from '../../src/validate/index.js';
import * as core from '../../src/index.js';

// `check_quantities`, `check_assignment` and `variable_quantities` (feature 1.5): the four rules
// the semantic stage decides before every other one, because every other one is written over the
// map they resolve to.
//
// Documents are written as JSON and read as the editor reads one — through the lexeme-preserving
// parser and the value model — so `4096` stays a whole number and `1e-05` a real all the way in,
// which is the distinction V3 is there to enforce.

/** A document with the quantities given, read as a document is read. */
function document(quantities: string): PyRecord {
  return toPython(parse(`{"quantities": ${quantities}}`)) as PyRecord;
}

/** An assignment, read the same way. */
function assignment(text: string): PyRecord {
  return toPython(parse(text)) as PyRecord;
}

/** The lines `check_quantities` would have produced, in order. */
function problems(model: PyRecord, given?: PyRecord): string[] {
  const resolved: Quantities = resolveQuantities(model, given);
  return formatSemanticProblems(checkQuantities(model, resolved));
}

const literal = (value: string, type = '{"kind": "cardinality"}'): string =>
  `{"type": ${type}, "source": {"kind": "literal", "value": ${value}}}`;
const derived = (expression: string, rest = ''): string =>
  `{"type": {"kind": "cardinality"}${rest}, "source": {"kind": "derived", "expression": ${expression}}}`;
const external = (rest = ''): string =>
  `{"type": {"kind": "cardinality"}${rest}, "source": {"kind": "external"}}`;
const POSITIVE = ', "domain": {"kind": "interval", "lower": {"value": {"literal": 1}, "inclusive": true}}';

describe('variableQuantities', () => {
  it('names the external quantities and everything derived from one, transitively', () => {
    const model = document(`{
      "width": ${external(POSITIVE)},
      "inner": ${derived('{"op": "multiply", "args": [{"quantity": "width"}, {"literal": 4}]}', POSITIVE)},
      "twice": ${derived('{"op": "multiply", "args": [{"quantity": "inner"}, {"literal": 2}]}', POSITIVE)},
      "heads": ${literal('32')},
      "head_dim": ${derived('{"op": "multiply", "args": [{"quantity": "heads"}, {"literal": 4}]}')}
    }`);
    expect([...variableQuantities(model)].sort()).toEqual(['inner', 'twice', 'width']);
  });

  it('reaches a derivation declared before the one it reads', () => {
    // The fixpoint is what makes declaration order irrelevant (§2.2: acyclic, in any order).
    const model = document(`{
      "twice": ${derived('{"op": "multiply", "args": [{"quantity": "inner"}, {"literal": 2}]}', POSITIVE)},
      "inner": ${derived('{"op": "multiply", "args": [{"quantity": "width"}, {"literal": 4}]}', POSITIVE)},
      "width": ${external(POSITIVE)}
    }`);
    expect([...variableQuantities(model)].sort()).toEqual(['inner', 'twice', 'width']);
  });

  it('names nothing in a document of model constants', () => {
    expect(variableQuantities(document(`{"d": ${literal('4096')}}`)).size).toBe(0);
  });

  it('raises for a document without quantities, as the tools do', () => {
    expect(() => variableQuantities(toPython(parse('{}')) as PyRecord)).toThrow(PyKeyError);
  });
});

describe('checkQuantities', () => {
  it('accepts a document of literals that conform', () => {
    const model = document(`{
      "d": ${literal('4096')},
      "eps": ${literal('1e-05', '{"kind": "real"}')},
      "precision": ${literal('"bf16"', '{"kind": "enum", "values": ["bf16"]}')}
    }`);
    expect(problems(model)).toEqual([]);
  });

  it('rewrites the type table’s word, once, and only the first occurrence', () => {
    // `_check_type` says `argument '…'` because the argument side is its other caller;
    // `check_quantities` returns `m.replace("argument '", "quantity '", 1)`.
    const model = document(`{"argument": ${literal('4096.0')}}`);
    expect(problems(model)).toEqual([
      "[V3] quantity 'argument' = 4096.0 is not a cardinality (non-negative integer)",
    ]);
  });

  it('refuses a variable quantity that declares no domain (V3)', () => {
    const model = document(`{
      "width": ${external(POSITIVE)},
      "inner": ${derived('{"op": "multiply", "args": [{"quantity": "width"}, {"literal": 4}]}')}
    }`);
    expect(problems(model, assignment('{"width": 3072}'))).toEqual([
      "[V3] quantity 'inner' depends on an external quantity and declares no domain " +
        '(a variable quantity declares one, §2.1)',
    ]);
  });

  it('refuses a derivation reading an undeclared quantity (V1), sorted, once per name', () => {
    const model = document(`{
      "d": ${derived('{"op": "multiply", "args": [{"quantity": "zeta"}, {"quantity": "alpha"}, {"quantity": "zeta"}]}')}
    }`);
    expect(problems(model)).toEqual([
      "[V1] quantity 'd': derivation reads undeclared quantity 'alpha'",
      "[V1] quantity 'd': derivation reads undeclared quantity 'zeta'",
      "[V10] quantity 'd': derivation does not resolve (a cycle, or a reference with no value)",
    ]);
  });

  it('refuses a cycle (V10) and leaves an unassigned external alone', () => {
    const cycle = document(`{
      "a": ${derived('{"op": "add", "args": [{"quantity": "b"}, {"literal": 1}]}', POSITIVE)},
      "b": ${derived('{"op": "add", "args": [{"quantity": "a"}, {"literal": 1}]}', POSITIVE)}
    }`);
    expect(problems(cycle)).toEqual([
      "[V10] quantity 'a': derivation does not resolve (a cycle, or a reference with no value)",
      "[V10] quantity 'b': derivation does not resolve (a cycle, or a reference with no value)",
    ]);
    // An external quantity with no assignment has no value either, and that is not a refusal:
    // the document is a family of graphs, and reading it needs one (§4.6).
    expect(problems(document(`{"width": ${external(POSITIVE)}}`))).toEqual([]);
  });

  it('checks a literal against its declared derivation (V11), comparing as Python compares', () => {
    const disagrees = document(`{
      "d": ${literal('4096')},
      "heads": ${literal('32')},
      "head_dim": {"type": {"kind": "cardinality"}, "source": {"kind": "literal", "value": 96,
        "derivation": {"op": "floor_divide", "args": [{"quantity": "d"}, {"quantity": "heads"}]}}}
    }`);
    expect(problems(disagrees)).toEqual([
      "[V11] quantity 'head_dim' = 96 disagrees with its derivation, which gives 128",
    ]);
    // `128 == 128.0` in Python, so a derivation answering a float agrees with a whole literal.
    const across = document(`{
      "d": ${literal('4096')},
      "heads": ${literal('32')},
      "head_dim": {"type": {"kind": "cardinality"}, "source": {"kind": "literal", "value": 128,
        "derivation": {"op": "divide", "args": [{"quantity": "d"}, {"quantity": "heads"}]}}}
    }`);
    expect(problems(across)).toEqual([]);
  });

  it('refuses a literal whose derivation does not resolve (V10), in its own words', () => {
    // Not the same line as a derived quantity's: "its derivation does not resolve".
    const model = document(`{
      "d": ${literal('4096')},
      "head_dim": {"type": {"kind": "cardinality"}, "source": {"kind": "literal", "value": 128,
        "derivation": {"op": "floor_divide", "args": [{"quantity": "d"}, {"literal": 0}]}}}
    }`);
    expect(problems(model)).toEqual([
      "[V10] quantity 'head_dim': its derivation does not resolve",
    ]);
  });

  it('checks the domain only when the type held', () => {
    const both = `${', "domain": {"kind": "interval", "lower": {"value": {"literal": 8192}, "inclusive": true}}'}`;
    expect(
      problems(document(`{"d": {"type": {"kind": "cardinality"}${both},
        "source": {"kind": "literal", "value": 4096}}}`)),
    ).toEqual(["[V3] quantity 'd' = 4096 is below the domain bound 8192 (inclusive)"]);
    expect(
      problems(document(`{"d": {"type": {"kind": "cardinality"}${both},
        "source": {"kind": "literal", "value": "four thousand"}}}`)),
    ).toEqual(["[V3] quantity 'd' = 'four thousand' is not a cardinality (non-negative integer)"]);
  });

  it('points every problem at the quantity’s declaration', () => {
    const model = document(`{
      "d": ${literal('4096.0')},
      "inner": ${derived('{"quantity": "nope"}')}
    }`);
    const found = checkQuantities(model, resolveQuantities(model));
    expect(found.map((one) => one.path)).toEqual([
      '/quantities/d',
      '/quantities/inner',
      '/quantities/inner',
    ]);
    expect(found[0]?.segments).toEqual(['quantities', 'd']);
  });
});

describe('checkAssignment', () => {
  const template = document(`{
    "layers": ${external(POSITIVE)},
    "precision": {"type": {"kind": "enum", "values": ["bf16", "f16"]},
                  "domain": {"kind": "set", "values": ["bf16", "f16"]},
                  "source": {"kind": "external"}},
    "d": ${literal('4096')}
  }`);

  it('says `assignment:` and keeps the type table’s word, unlike the quantity side', () => {
    const errors = formatSemanticProblems(
      checkAssignment(template, assignment('{"layers": 0, "precision": "fp4"}')),
    );
    expect(errors).toEqual([
      "[V3] assignment: argument 'layers' = 0 is below the domain bound 1 (inclusive)",
      "[V3] assignment: argument 'precision' = 'fp4' is not among ['bf16', 'f16']",
    ]);
  });

  it('walks the document’s quantities, so an unknown name is not refused', () => {
    // The tools never look at the assignment's own names; a name the document does not declare
    // is checked by nothing, which is a fact about `--assign`, not about the editor's sheet.
    expect(checkAssignment(template, assignment('{"lyaers": 0}'))).toEqual([]);
  });

  it('ignores a quantity that is not external, and a name the assignment leaves unset', () => {
    expect(checkAssignment(template, assignment('{"d": "not a number"}'))).toEqual([]);
    expect(checkAssignment(template, assignment('{}'))).toEqual([]);
    expect(checkAssignment(template)).toEqual([]);
  });

  it('points at the declaration that refused the value', () => {
    const found = checkAssignment(template, assignment('{"layers": 0}'));
    expect(found.map((one) => one.path)).toEqual(['/quantities/layers']);
  });
});

describe('assignmentNeeded', () => {
  const model = document(`{
    "width": ${external(POSITIVE)},
    "eps": {"type": {"kind": "real"},
            "domain": {"kind": "interval", "lower": {"value": {"literal": 0}, "inclusive": false}},
            "source": {"kind": "external", "default": {"literal": 1e-05}}},
    "layers": ${external(POSITIVE)},
    "d": ${literal('4096')}
  }`);

  it('lists the external quantities in the document’s order, and what must be supplied', () => {
    const needed = assignmentNeeded(model);
    expect(needed.external).toEqual(['width', 'eps', 'layers']);
    expect(needed.required).toEqual(['width', 'layers']);
  });

  it('reports what is unset, sorted, in the words the tools print', () => {
    expect(assignmentNeeded(model).unset).toEqual(['layers', 'width']);
    expect(assignmentNeeded(model).report).toBe("needs --assign for ['layers', 'width']");
    const partial = assignmentNeeded(model, assignment('{"width": 3072}'));
    expect(partial.unset).toEqual(['layers']);
    expect(partial.report).toBe("needs --assign for ['layers']");
  });

  it('reports nothing when every required name is supplied, or none is required', () => {
    const complete = assignmentNeeded(model, assignment('{"width": 3072, "layers": 26}'));
    expect(complete.unset).toEqual([]);
    expect(complete.report).toBeNull();
    const constants = assignmentNeeded(document(`{"d": ${literal('4096')}}`));
    expect(constants.external).toEqual([]);
    expect(constants.report).toBeNull();
  });
});

describe('the package surface', () => {
  it('exports the feature from the package root', () => {
    // Two star exports of one name are silently excluded rather than refused (feature 1.4), so a
    // collision between `validate/` and an older subsystem would make a name simply disappear.
    for (const name of [
      'checkQuantities',
      'checkAssignment',
      'variableQuantities',
      'assignmentNeeded',
      'checkType',
      'checkDomain',
      'semanticProblem',
      'withMessage',
      'formatSemanticProblem',
      'formatSemanticProblems',
    ]) {
      expect(typeof (core as unknown as Record<string, unknown>)[name], name).toBe('function');
    }
  });
});
