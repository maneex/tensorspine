import { describe, expect, it } from 'vitest';

import {
  auditPresentation,
  genericLines,
  presentationLines,
  type PresentationAudit,
} from '../../src/presentation/audit.js';
import { presentation, readPresentation } from '../../src/presentation/load.js';
import { startPresentation } from '../../src/presentation/index.js';
import { registry } from './source.js';

// The startup audit of plan §1 (a):
//
//   > A startup audit resolves every key of `presentation.json` against the loaded schemas; a key
//   > that resolves nowhere is an error in the log, and every enum value and `oneOf` member of the
//   > loaded schemas that no binding names is listed as "rendered generically".
//
// The two halves fail in opposite directions. A key that resolves nowhere is a binding that does
// nothing — the schema moved, or the pointer was mistyped — and the interface would go on
// rendering generically with nobody the wiser. A construct nothing names is *not* an error: most
// of the language renders generically on purpose. It has to be listed, that is all.

const MODEL = 'https://tensorspine.dev/schema/2.0/model.json';
const UNIT = 'https://tensorspine.dev/schema/2.0/primitive-library-unit.json';
const DERIVED = 'https://tensorspine.dev/schema/2.1/derived.json';

const audit = auditPresentation(registry(), presentation());

/** The audit of a handful of bindings written here, against the repository's schemas. */
function auditOf(bindings: Record<string, unknown>): PresentationAudit {
  return auditPresentation(registry(), readPresentation(bindings));
}

describe('the bindings the application ships', () => {
  it('every one of them names a place of the loaded schemas', () => {
    expect(audit.problems.map((one) => `${one.code} ${one.anchor}: ${one.message}`)).toEqual([]);
    expect(audit.resolved).toBe(audit.bindings);
    expect(audit.bindings).toBe(presentation().anchors.length);
  });

  it('resolves against the five schemas the registry indexes', () => {
    expect(audit.schemas).toContain(MODEL);
    expect(audit.schemas).toContain(UNIT);
    expect(audit.schemas).toContain(DERIVED);
  });

  it('runs at startup and writes what it found to the log', () => {
    const lines: string[] = [];
    const started = startPresentation(registry(), (line) => lines.push(line));
    expect(started.bindings).toBe(presentation());
    expect(started.audit.problems).toEqual([]);
    expect(lines).toEqual(presentationLines(started.audit));
    expect(lines).toHaveLength(1);
    expect(lines[0]).toMatch(/bindings resolved; \d+ enumerated values and \d+ union members/);
  });

  it('runs without a log, for a caller that has none yet', () => {
    expect(startPresentation(registry()).audit.problems).toEqual([]);
  });
});

describe('a key that resolves nowhere', () => {
  it('is an error naming the schema it was looked for in', () => {
    const found = auditOf({ [`${MODEL}#/$defs/instance_defintion`]: { role: 'node' } });
    expect(found.resolved).toBe(0);
    expect(found.problems).toEqual([
      {
        code: 'unresolved',
        anchor: `${MODEL}#/$defs/instance_defintion`,
        message: `names nothing in ${MODEL}`,
      },
    ]);
  });

  it('is told from a key whose schema is not loaded at all', () => {
    const anchor = 'https://tensorspine.dev/schema/2.0/derived.json#/$defs/d3';
    const found = auditOf({ [anchor]: { format: 'bytes' } });
    expect(found.problems.map((one) => one.code)).toEqual(['unknown-schema']);
    expect(found.problems[0]?.message).toMatch(/names no schema of the registry/);
  });

  it('is told from a key that is no anchor at all', () => {
    const found = auditOf({ 'model.json/$defs/dtype': { format: 'bytes' } });
    expect(found.problems.map((one) => one.code)).toEqual(['malformed-key']);
  });

  it('is reported through a chain of bare references, so a definition is reached', () => {
    // `#/properties/quantities` is `{"$ref": "#/$defs/quantity_map"}` and nothing else, and the
    // binding that declares the quantities is written at the *place*, not at the shape.
    expect(auditOf({ [`${MODEL}#/properties/quantities`]: { declares: 'quantity' } }).problems).toEqual(
      [],
    );
  });
});

describe('a key that resolves and still says nothing', () => {
  it("catches the plan's own Appendix B sketch, which gives the binary operator the n-ary symbols", () => {
    // Appendix B writes `binary_operation_expression/properties/op` with the symbols of `add`,
    // `subtract`, `multiply`, `divide`, `floor_divide`, `ceil_divide` and `modulo`. The
    // enumeration there admits five of the seven: `add` and `multiply` are
    // `nary_operation_expression`'s, and a symbol map that names them is a binding that resolves
    // and is wrong.
    const found = auditOf({
      [`${MODEL}#/$defs/binary_operation_expression/properties/op`]: {
        symbols: {
          add: { text: '+', form: 'infix' },
          subtract: { text: '-', form: 'infix' },
          multiply: { text: '*', form: 'infix' },
          divide: { text: '/', form: 'infix' },
        },
      },
    });
    expect(found.resolved).toBe(1);
    expect(found.problems.map((one) => one.message)).toEqual([
      "gives a symbol to 'add', which it does not admit",
      "gives a symbol to 'multiply', which it does not admit",
    ]);
  });

  it('reports symbols at a place that is neither an enumeration nor a union', () => {
    const found = auditOf({
      [`${MODEL}#/$defs/instance_definition`]: { symbols: { add: { text: '+', form: 'infix' } } },
    });
    expect(found.problems.map((one) => one.code)).toEqual(['symbol']);
    expect(found.problems[0]?.message).toMatch(/neither an enumeration nor a union/);
  });

  it('admits a symbol keyed by the tag of an alternative of a union', () => {
    expect(
      auditOf({ [`${MODEL}#/$defs/condition`]: { symbols: { all: { text: 'and', form: 'infix' } } } })
        .problems,
    ).toEqual([]);
    expect(
      auditOf({ [`${MODEL}#/$defs/condition`]: { symbols: { every: { text: 'and', form: 'infix' } } } })
        .problems.map((one) => one.message),
    ).toEqual(["gives a symbol to 'every', which it does not admit"]);
  });

  it('reports a face naming a member the definition does not declare', () => {
    const found = auditOf({
      [`${MODEL}#/$defs/instance_definition`]: { role: 'node', face: ['primitive', 'colour'] },
    });
    expect(found.problems.map((one) => one.message)).toEqual([
      "shows 'colour' on its face, which it does not declare",
    ]);
  });

  it('reports a face on a place that declares no member at all', () => {
    const found = auditOf({ [`${MODEL}#/$defs/dtype`]: { face: ['primitive'] } });
    expect(found.problems.map((one) => one.code)).toEqual(['face']);
  });
});

describe('a reference rule and the declaration it is scoped by', () => {
  const sites = `${MODEL}#/$defs/composition_definition/properties/instances`;

  it('is reported when it reads against an enclosing declaration and names none', () => {
    const found = auditOf({
      [sites]: { declares: 'site', refers: [{ tag: 'site', kind: 'tagged', under: 'scope' }] },
    });
    expect(found.problems.map((one) => one.message)).toEqual([
      "reads 'site' against an enclosing declaration and names none",
    ]);
  });

  it('is reported when the scope it names declares nothing', () => {
    const found = auditOf({
      [sites]: { declares: 'site', scope: `${MODEL}#/$defs/dtype`, refers: [] },
    });
    expect(found.problems.map((one) => one.message)).toEqual([
      `is scoped by ${MODEL}#/$defs/dtype, which declares nothing`,
    ]);
  });

  it('is reported when it restricts the search to a place that has no name', () => {
    const found = auditOf({
      [sites]: {
        declares: 'site',
        scope: `${MODEL}#/properties/compositions`,
        refers: [{ tag: 'site', under: 'everywhere' }],
      },
      [`${MODEL}#/properties/compositions`]: { declares: 'composition' },
    });
    expect(found.problems.map((one) => one.message)).toEqual([
      "restricts 'site' to 'everywhere', which names no place",
    ]);
  });
});

describe('what renders generically', () => {
  const named = (anchor: string, name: string): boolean =>
    audit.generic.some((one) => one.anchor === anchor && one.name === name);

  it('lists an operator with no symbol, and not one with a symbol', () => {
    const op = `${MODEL}#/$defs/nary_operation_expression/properties/op`;
    // `min` and `max` print as `min(a, b)` and `max(a, b)` — their symbol is their own name, so
    // writing one down would repeat the schema (plan §4.13). They are listed for that reason.
    expect(named(op, 'min')).toBe(true);
    expect(named(op, 'max')).toBe(true);
    expect(named(op, 'add')).toBe(false);
    expect(named(op, 'multiply')).toBe(false);
  });

  it('lists every dtype, because a dtype is a plain select', () => {
    const dtype = `${MODEL}#/$defs/dtype`;
    expect(named(dtype, 'bf16')).toBe(true);
    expect(named(dtype, 'f32')).toBe(true);
  });

  it('lists the alternatives of a union no binding names, and not the ones a symbol names', () => {
    const condition = `${MODEL}#/$defs/condition`;
    const alternatives = audit.generic.filter((one) => one.anchor.startsWith(`${condition}/oneOf/`));
    expect(alternatives.map((one) => one.name)).toEqual(['boolean', 'compare']);
  });

  it('counts every enumerated value and every union member of the loaded schemas', () => {
    const vocabulary = registry().vocabulary();
    const values = vocabulary.enums.reduce((total, one) => total + one.values.length, 0);
    const members = vocabulary.unions.reduce((total, one) => total + one.alternatives.length, 0);
    const listed = audit.generic.filter((one) => one.kind === 'enumeration').length;
    const alternatives = audit.generic.length - listed;
    expect(listed).toBeLessThan(values);
    expect(alternatives).toBeLessThan(members);
    // Nothing is listed twice, and nothing is listed that no schema writes.
    expect(new Set(audit.generic.map((one) => `${one.anchor} ${one.name}`)).size).toBe(
      audit.generic.length,
    );
  });

  it('has one line each for the log to unfold', () => {
    const lines = genericLines(audit);
    expect(lines).toHaveLength(audit.generic.length);
    expect(lines.every((line) => line.endsWith('renders generically'))).toBe(true);
  });
});
