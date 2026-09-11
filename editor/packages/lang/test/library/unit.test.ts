import { describe, expect, it } from 'vitest';

import {
  formatLibraryProblem,
  formatLibraryProblems,
  loadLibrary,
  memorySource,
  placeOf,
  validateUnit,
  validateUnitText,
  type Library,
} from '../../src/library/index.js';
import { parse } from '../../src/json/index.js';
import { repositorySchemas } from '../schema/repository.js';
import { nodeSource, overlay, REFERENCE_BASE, rejectionBase, repositoryPath } from './source.js';

// `validateUnit`, the live judgement of §4.22: the loader's checks on one unit against bases
// already gathered, as the primitive editor asks for them on every edit (D15).
//
// The two tests the feature names are here — every unit of the reference base returns nothing, and
// a unit citing `attention.headz` returns the tools' own message — with the rest of D15's list
// beside them: the identity/path agreement, the domain-bound rule, the `present`-guard rule, the
// payload-axis rule and the template pin.

const schemas = repositorySchemas();
const source = nodeSource();
const context = { schemas, source };
const library: Library = loadLibrary([REFERENCE_BASE], context);

/** One unit of a rejection base, judged against the reference base. */
function judgeFixture(base: string, unit: string): string[] {
  const path = `${rejectionBase(base)}/${unit}`;
  return formatLibraryProblems(
    validateUnitText(source.read(path), { base: rejectionBase(base), path }, library, context),
  );
}

describe('every unit of the reference base', () => {
  it('is judged clean, one by one', () => {
    const problems: string[] = [];
    for (const unit of library.bases[0]?.units ?? []) {
      const answered = validateUnit(
        unit.tree,
        { base: REFERENCE_BASE, path: unit.file },
        library,
        context,
      );
      problems.push(...formatLibraryProblems(answered));
    }
    expect(problems).toEqual([]);
  });

  it('includes the manifest, judged as a base unit', () => {
    const path = `${REFERENCE_BASE}/primitive-library.json`;
    expect(
      validateUnitText(source.read(path), { base: REFERENCE_BASE, path }, library, context),
    ).toEqual([]);
  });

  it('is 130 units and one manifest', () => {
    // The plan's Appendix A: "36 primitives (one a template), 38 axes, 56 precision roles, one
    // manifest" — the 130 units D15's closing test names.
    expect(library.bases[0]?.units).toHaveLength(130);
  });
});

describe('a unit citing what the primitive library does not hold', () => {
  it('answers the tools’ message for `attention.headz`', () => {
    expect(judgeFixture('primitive-library/unknown-axis', 'primitives/attention/dense/1.0.1.json'))
      .toEqual([
        `${rejectionBase('primitive-library/unknown-axis')}/primitives/attention/dense/1.0.1.json: ` +
          'unresolved reference(s)\n' +
          "    parameter 'q': axis 'attention.headz' is not in the primitive_library",
      ]);
  });

  it('points the row at the axis it could not resolve', () => {
    const base = rejectionBase('primitive-library/unknown-axis');
    const path = `${base}/primitives/attention/dense/1.0.1.json`;
    const problems = validateUnitText(source.read(path), { base, path }, library, context);
    expect(problems[0]?.kind).toBe('reference');
    expect(problems[0]?.detail[0]?.path).toBe('/definition/parameters/q/shape/axes/0/axis');
  });

  it('answers the tools’ message for a role with no precision rule', () => {
    expect(
      judgeFixture('primitive-library/unknown-precision-role', 'primitives/attention/dense/1.0.1.json'),
    ).toEqual([
      `${rejectionBase('primitive-library/unknown-precision-role')}/primitives/attention/dense/1.0.1.json: ` +
        'unresolved reference(s)\n' +
        "    parameter 'out': role 'attention.output_projektion' has no precision rule",
    ]);
  });

  it('answers the tools’ message for a guard on an argument nobody declares', () => {
    const lines = judgeFixture(
      'primitive-library/guard-on-undeclared-argument',
      'primitives/attention/dense/1.0.1.json',
    );
    expect(lines[0]).toContain("parameter 'q_gated' present_when: tests undeclared argument 'o_rank'");
  });

  it('refuses a domain bound that names an argument which may be absent', () => {
    const lines = judgeFixture(
      'primitive-library/domain-bound-optional-argument',
      'primitives/attention/dense/1.0.1.json',
    );
    expect(lines[0]).toContain(
      "argument 'head_dim': domain upper bound reads 'scale', which may be absent — a bound must " +
        'always resolve (§4.6)',
    );
  });

  it('refuses a comparison of a maybe-absent argument outside a present test, once per site', () => {
    const lines = judgeFixture(
      'primitive-library/absent-argument-comparison',
      'primitives/attention/dense/1.0.1.json',
    );
    // Four conditions read `streaming`; the refusal carries one line for each, in the order the
    // checker met them (present_when, carried_across, rule 0, rule 4).
    expect(lines[0]?.split('\n')).toHaveLength(5);
    expect(lines[0]).toContain(
      "state 'kv' present_when: compares 'streaming', which may be absent, outside a present test",
    );
    expect(lines[0]).toContain("state 'kv' rule 4: compares 'streaming'");
  });

  it('refuses a payload declared per position under a growing rule', () => {
    const lines = judgeFixture(
      'primitive-library/payload-position-axis',
      'primitives/sequence/gated_delta/1.0.1.json',
    );
    expect(lines[0]).toContain(
      "state 'conv'.w: a sequence.position axis under a window rule — a payload is declared per " +
        'position (§4.3)',
    );
  });

  it('refuses an invariant that reads an argument the primitive never declares', () => {
    const lines = judgeFixture(
      'primitive-library/invariant-undeclared-argument',
      'primitives/attention/dense/1.0.1.json',
    );
    expect(lines[0]).toContain(
      "invariant 3 ('a made-up argument is positive'): tests undeclared argument 'nonexistent'",
    );
  });
});

describe('the identity a unit’s path spells out', () => {
  it('refuses a version that disagrees with the file name', () => {
    expect(
      judgeFixture('primitive-library/version-path-mismatch', 'primitives/attention/dense/1.0.1.json'),
    ).toEqual([
      `${rejectionBase('primitive-library/version-path-mismatch')}/primitives/attention/dense/1.0.1.json: ` +
        "version '1.0.2', path says '1.0.1'",
    ]);
  });

  it('reads a primitive’s path as a dotted name and a version', () => {
    expect(placeOf({ base: 'b', path: 'b/primitives/attention/dense/1.0.0.json' })).toEqual({
      section: 'primitives',
      kind: 'primitive',
      name: 'attention.dense',
      version: '1.0.0',
    });
  });

  it('reads an axis and a role as a dotted name alone, and the manifest as the base', () => {
    expect(placeOf({ base: 'b', path: 'b/axes/model/width.json' })).toEqual({
      section: 'axes',
      kind: 'axis',
      name: 'model.width',
      version: null,
    });
    expect(placeOf({ base: 'b', path: 'b/precision/norm/scale.json' })?.kind).toBe('precision_role');
    expect(placeOf({ base: 'b', path: 'b/primitive-library.json' })?.kind).toBe('base');
  });

  it('reads a base that resolves to the root the paths are written against', () => {
    // `basesOf` normalises, so a document at the root of its workspace declaring the directory it
    // sits in gets the base `.` — `normalise('models/..')`. `join` does not normalise, so
    // `join('.', 'primitives')` is `./primitives` while the unit's path has become
    // `primitives/…`: the comparison has to be made on the normalised form or every unit of such
    // a base is refused as being under no section of it.
    expect(placeOf({ base: '.', path: 'primitives/norm/rms/1.0.0.json' })).toEqual({
      section: 'primitives',
      kind: 'primitive',
      name: 'norm.rms',
      version: '1.0.0',
    });
    expect(placeOf({ base: '.', path: 'primitive-library.json' })).toEqual({
      section: null,
      kind: 'base',
      name: '',
      version: null,
    });
    expect(placeOf({ base: 'models/..', path: 'axes/model/width.json' })).toEqual({
      section: 'axes',
      kind: 'axis',
      name: 'model.width',
      version: null,
    });
    // And a file that is under no section of it is still refused.
    expect(placeOf({ base: '.', path: 'elsewhere/one.json' })).toBeNull();
  });

  it('names no place for a file outside the base’s sections', () => {
    expect(placeOf({ base: 'b', path: 'b/elsewhere/one.json' })).toBeNull();
    const problems = validateUnitText('{}', { base: 'b', path: 'b/elsewhere/one.json' }, library, {
      schemas: null,
      source,
    });
    expect(problems[0]?.message).toBe(
      'b/elsewhere/one.json: not under primitives/, axes/ or precision/ of b',
    );
  });
});

describe('a template primitive’s pin', () => {
  const base = rejectionBase('primitive-library/template-missing');

  it('is checked against the base’s declared templates location', () => {
    const lines = judgeFixture('primitive-library/template-missing', 'primitives/decoder/causal_yarn/1.0.1.json');
    expect(lines[0]).toBe(
      `${base}/primitives/decoder/causal_yarn/1.0.1.json: template 'decoder-nowhere' 1.0.0 is ` +
        `not at ${repositoryPath('data', 'models')}/decoder-nowhere/1.0.0.json`,
    );
  });

  it('refuses a base that declares no templates location at all', () => {
    const lines = judgeFixture(
      'primitive-library/template-no-location',
      'primitives/decoder/causal_yarn/1.0.1.json',
    );
    expect(lines[0]).toContain('a template primitive, but its base declares no `templates` location (§4.6)');
  });

  it('refuses a template document whose model id is not the one pinned', () => {
    const lines = judgeFixture(
      'primitive-library/template-id-mismatch',
      'primitives/decoder/causal_yarn/1.0.1.json',
    );
    expect(lines[0]).toContain(
      "template 'decoder-causal-yarn' declares model id 'decoder_causal_yarn', the primitive says " +
        "'something_else'",
    );
  });

  it('accepts the reference base’s own template primitive', () => {
    const path = `${REFERENCE_BASE}/primitives/decoder/causal_yarn/1.0.0.json`;
    expect(
      validateUnitText(source.read(path), { base: REFERENCE_BASE, path }, library, context),
    ).toEqual([]);
  });

  it('reads the templates location of a base the gathered library has not seen', () => {
    // A base being written in the editor is not in the loaded library yet, so the manifest is
    // read again through the source; a manifest that says nothing leaves the pin unresolvable.
    const unit = source.read(`${REFERENCE_BASE}/primitives/decoder/causal_yarn/1.0.0.json`);
    const manifest = JSON.stringify({
      schema: 'tensorspine-primitive-library-unit/2.0',
      kind: 'base',
      name: 'demo.base',
      definition: { primitive_library: 'demo/base', title: 'Demo', templates: '../models/' },
    });
    const fresh = memorySource({
      'fresh/primitive-library.json': manifest,
      'fresh/primitives/decoder/causal_yarn/1.0.0.json': unit,
    });
    const path = 'fresh/primitives/decoder/causal_yarn/1.0.0.json';
    const problems = validateUnitText(unit, { base: 'fresh', path }, library, {
      schemas,
      source: overlay(fresh, nodeSource(repositoryPath('..'))),
    });
    // `fresh/../models/` is not where the corpus is, so the pin is refused — naming where it looked.
    expect(problems[0]?.message).toContain("template 'decoder-causal-yarn' 1.0.0 is not at models/");
  });
});

describe('a unit off the grammar', () => {
  it('is refused for that and nothing else, in eight lines at most', () => {
    const base = rejectionBase('primitive-library/state-evolution-typo');
    const path = `${base}/primitives/attention/dense/1.0.1.json`;
    const problems = validateUnitText(source.read(path), { base, path }, library, context);
    expect(problems).toHaveLength(1);
    expect(problems[0]?.kind).toBe('schema');
    expect(formatLibraryProblem(problems[0]!)).toBe(
      `${path}: off the primitive-library-unit schema\n` +
        "    definition/state_ports/kv/rules/4/evolution: 'apend' is not one of " +
        "['append', 'window', 'fixed']",
    );
  });

  it('carries the place a schema row names, which two rejection cases match on', () => {
    const base = rejectionBase('primitive-library/context-in-primitive');
    const path = `${base}/primitives/attention/dense/1.0.1.json`;
    const problems = validateUnitText(source.read(path), { base, path }, library, context);
    expect(problems[0]?.detail[0]?.message).toBe(
      "definition/state_ports/kv/rules/3/span: 'literal' is a required property",
    );
    expect(problems[0]?.detail[0]?.path).toBe('/definition/state_ports/kv/rules/3/span');
  });

  it('takes a tree the editor already parsed, without the JSON layer', () => {
    const path = `${REFERENCE_BASE}/axes/model/width.json`;
    expect(
      validateUnit(parse(source.read(path)), { base: REFERENCE_BASE, path }, library, context),
    ).toEqual([]);
  });
});

describe('a precision role', () => {
  it('is refused when its default is outside its own admissible set', () => {
    expect(
      judgeFixture('primitive-library/role-default-inadmissible', 'precision/state/kv_test.json'),
    ).toEqual([
      `${rejectionBase('primitive-library/role-default-inadmissible')}/precision/state/kv_test.json: ` +
        "default 'fp4' is not in the admissible set ['bf16', 'f16', 'f8e4m3']",
    ]);
  });
});
