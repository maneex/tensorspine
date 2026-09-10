import { describe, expect, it } from 'vitest';

import {
  basesOf,
  formatLibraryProblems,
  identityKey,
  librariesFor,
  loadLibrary,
  memorySource,
  primitiveOf,
  semanticVersion,
  templatePinOf,
  templatePrimitives,
  type Library,
} from '../../src/library/index.js';
import { toPython } from '../../src/expr/index.js';
import { parse } from '../../src/json/index.js';
import { loadSchemas } from '../../src/schema/index.js';
import { repositorySchemas } from '../schema/repository.js';
import {
  nodeSource,
  overlay,
  REFERENCE_BASE,
  rejectionBase,
  repositoryPath,
} from './source.js';

// The loader over the repository's own reference base — the vocabulary every corpus document
// pins — and over bases built in memory for the rules the reference base cannot exercise: a
// conflict between two bases, a monolithic base, a second version of one primitive.
//
// The counts asserted on the reference base are the plan's Appendix A, which is what the primitive
// editor must render: "36 primitives (one a template), 38 axes, 56 precision roles, one manifest;
// 203 argument and field declarations, every one described, 17 with value descriptions; 28
// primitives with external docs; three tags".

const schemas = repositorySchemas();
const source = nodeSource();

function reference(): Library {
  return loadLibrary([REFERENCE_BASE], { schemas, source });
}

/** Every argument and field declaration of a set of declarations, records flattened. */
function declarationsOf(args: unknown): Record<string, unknown>[] {
  if (args === null || typeof args !== 'object') return [];
  const found: Record<string, unknown>[] = [];
  for (const declaration of Object.values(args as Record<string, unknown>)) {
    if (declaration === null || typeof declaration !== 'object') continue;
    found.push(declaration as Record<string, unknown>);
    const type = (declaration as { type?: { kind?: unknown; fields?: unknown } }).type;
    if (type !== undefined && type.kind === 'record') found.push(...declarationsOf(type.fields));
  }
  return found;
}

/** A unit of the reference base, with some of its definition's members changed. */
function variantOf(unit: string, changes: Record<string, unknown>): string {
  const document = JSON.parse(source.read(unit)) as { definition: Record<string, unknown> };
  document.definition = { ...document.definition, ...changes };
  return JSON.stringify(document);
}

describe('the reference base', () => {
  const library = reference();

  it('loads with no problem at all', () => {
    expect(formatLibraryProblems(library.problems)).toEqual([]);
  });

  it('holds the counts of the plan’s Appendix A', () => {
    expect(library.byId.size).toBe(36);
    expect(library.axes.size).toBe(38);
    expect(library.precision.size).toBe(56);
    expect(library.bases).toHaveLength(1);
    expect(library.bases[0]?.manifest).not.toBeNull();
    expect(library.bases[0]?.units).toHaveLength(36 + 38 + 56);
    expect(templatePrimitives(library)).toEqual(new Set(['decoder.causal_yarn']));
    expect(library.templates.size).toBe(1);
  });

  it('resolves the one template primitive to its document', () => {
    const pin = library.templates.get(identityKey('decoder.causal_yarn', '1.0.0'));
    expect(pin?.path).toBe(repositoryPath('data', 'models', 'decoder-causal-yarn', '1.0.0.json'));
    expect(pin?.version).toBe('1.0.0');
    const definition = primitiveOf(library, { name: 'decoder.causal_yarn', version: '1.0.0' });
    expect(templatePinOf(library, definition ?? null)?.path).toBe(pin?.path);
    expect(library.bases[0]?.templates).toBe(repositoryPath('data', 'models'));
  });

  it('indexes every primitive by name and by identity', () => {
    for (const [key, version] of library.byId) {
      expect(key).toBe(identityKey(version.name, version.version));
      expect(version.base).toBe(REFERENCE_BASE);
      expect(version.file.startsWith(REFERENCE_BASE)).toBe(true);
    }
    expect(library.primitives.size).toBe(36);
    // Every identity of the reference base is at 1.0.0, so the by-name index is the by-id one.
    expect([...library.byId.values()].every((one) => one.version === '1.0.0')).toBe(true);
  });

  it('answers a pinned version first and falls back to the name', () => {
    expect(primitiveOf(library, { name: 'norm.rms', version: '1.0.0' })).toBeDefined();
    // A version nobody carries falls back to the name, so the caller reports the mismatch (V1).
    expect(primitiveOf(library, { name: 'norm.rms', version: '9.9.9' })).toBe(
      library.primitives.get('norm.rms'),
    );
    expect(primitiveOf(library, { name: 'norm.rmz', version: '1.0.0' })).toBeUndefined();
  });
});

describe('what the reference base declares', () => {
  const library = reference();
  const primitives = [...library.byId.values()].filter(
    (one) => !Object.hasOwn(one.definition as Record<string, unknown>, 'template'),
  );

  it('describes all 203 argument and field declarations, 17 with value descriptions', () => {
    const all = primitives.flatMap((one) =>
      declarationsOf((one.definition as { arguments?: unknown }).arguments),
    );
    expect(all).toHaveLength(203);
    expect(all.filter((one) => 'description' in one)).toHaveLength(203);
    expect(all.filter((one) => 'value_descriptions' in one)).toHaveLength(17);
  });

  it('gives 28 primitives external docs and the manifest three tags', () => {
    const documented = [...library.byId.values()].filter((one) =>
      Object.hasOwn(one.definition as Record<string, unknown>, 'external_docs'),
    );
    expect(documented).toHaveLength(28);
    const manifest = library.bases[0]?.manifest as { tags?: unknown[] };
    expect(manifest.tags).toHaveLength(3);
  });

  it('carries the state ports, transforms, sparsity and multiplicities Appendix A lists', () => {
    const states: Record<string, number> = {};
    const transforms: Record<string, string[]> = {};
    const sparse: string[] = [];
    const multiple: Record<string, string[]> = {};
    const shareable: string[] = [];
    let constants = 0;
    for (const one of primitives) {
      const definition = one.definition as {
        state_ports: Record<string, unknown>;
        domain_transforms?: { relation: string }[];
        sparsity?: unknown[];
        parameters: Record<string, { multiplicity?: unknown; sharing?: { kind?: string } }>;
        constants: Record<string, unknown>;
      };
      const count = Object.keys(definition.state_ports).length;
      if (count > 0) states[one.name] = count;
      if (definition.domain_transforms !== undefined) {
        transforms[one.name] = definition.domain_transforms.map((entry) => entry.relation);
      }
      if (definition.sparsity !== undefined) sparse.push(one.name);
      const withMultiplicity = Object.entries(definition.parameters)
        .filter(([, slot]) => slot.multiplicity !== undefined)
        .map(([name]) => name);
      if (withMultiplicity.length > 0) multiple[one.name] = withMultiplicity;
      if (Object.values(definition.parameters).some((slot) => slot.sharing?.kind === 'shareable')) {
        shareable.push(one.name);
      }
      constants += Object.keys(definition.constants).length;
    }
    expect(states).toEqual({
      'attention.dense': 1,
      'attention.latent_compressed': 5,
      'conditioning.scale': 1,
      conv_frontend: 2,
      'sequence.gated_delta': 2,
    });
    expect(transforms).toEqual({
      'attention.dense': ['align'],
      'conditioning.scale': ['align'],
      conv_frontend: ['merge'],
      'projector.patch_merge_bottleneck': ['merge'],
      'projector.patch_merge_mlp': ['merge'],
      'projector.temporal_stack': ['merge'],
      splice: ['insert'],
    });
    expect(sparse.sort()).toEqual([
      'conv_frontend',
      'embed',
      'embedding.token_auxiliary',
      'embedding.token_position',
      'embedding.token_position_type',
      'moe',
      'patch_embed',
    ]);
    expect(multiple).toEqual({
      moe: ['shared_gate', 'shared_up', 'shared_out'],
      'residual.stream_collapse': ['projection'],
      'residual.stream_expand': ['projection'],
    });
    expect(shareable.sort()).toEqual([
      'embed',
      'embedding.token_auxiliary',
      'embedding.token_position',
      'embedding.token_position_type',
      'lm_head',
      'residual.altup_correct',
      'residual.altup_predict',
    ]);
    // "no unit declares a constant slot" — the grammar admits one, the corpus has none (F8).
    expect(constants).toBe(0);
  });
});

describe('a base is a set: one identity, one content', () => {
  it('refuses a second base carrying attention.dense@1.0.0 differently, as V1', () => {
    const base = rejectionBase('primitive-library/base-conflict');
    const library = loadLibrary([base, REFERENCE_BASE], { schemas, source });
    const first = library.problems[0];
    expect(first?.code).toBe('V1');
    expect(first?.kind).toBe('conflict');
    expect(first?.message).toContain("identity ('attention.dense', '1.0.0') is also carried by");
    expect(first?.message).toContain('with different contents (V1)');
    // The refusal names the file it met second, and the one that provided the identity first.
    expect(first?.file).toBe(
      `${REFERENCE_BASE}/primitives/attention/dense/1.0.0.json`,
    );
    expect(first?.message).toContain(`${base}/primitives/attention/dense/1.0.0.json`);
  });

  it('accepts two bases carrying one identity with the same content', () => {
    const axis = source.read(repositoryPath('data', 'primitive-library', 'axes', 'model', 'width.json'));
    const library = loadLibrary(['a', 'b'], {
      schemas,
      source: memorySource({ 'a/axes/model/width.json': axis, 'b/axes/model/width.json': axis }),
    });
    expect(formatLibraryProblems(library.problems)).toEqual([]);
    expect(library.axes.size).toBe(1);
  });

  it('names an axis identity bare where it names a primitive identity as a tuple', () => {
    const axis = (space: string): string =>
      JSON.stringify({
        schema: 'tensorspine-primitive-library-unit/2.0',
        kind: 'axis',
        name: 'demo.one',
        definition: { space, summary: 'A demonstration axis.' },
      });
    const library = loadLibrary(['a', 'b'], {
      schemas,
      source: memorySource({
        'a/axes/demo/one.json': axis('value'),
        'b/axes/demo/one.json': axis('instance'),
      }),
    });
    expect(library.problems[0]?.message).toBe(
      'b/axes/demo/one.json: identity demo.one is also carried by a/axes/demo/one.json with ' +
        'different contents (V1)',
    );
  });
});

describe('the loader’s other refusals', () => {
  it('refuses a unit whose path does not spell its name', () => {
    const library = loadLibrary(['base'], {
      schemas,
      source: memorySource({
        'base/axes/demo/one.json': JSON.stringify({
          schema: 'tensorspine-primitive-library-unit/2.0',
          kind: 'axis',
          name: 'demo.two',
          definition: { space: 'value', summary: 'A demonstration axis.' },
        }),
      }),
    });
    expect(library.problems[0]?.message).toBe(
      "base/axes/demo/one.json: name 'demo.two', path says 'demo.one'",
    );
  });

  it('refuses a unit whose kind does not match its section', () => {
    const library = loadLibrary(['base'], {
      schemas,
      source: memorySource({
        'base/axes/demo/one.json': JSON.stringify({
          schema: 'tensorspine-primitive-library-unit/2.0',
          kind: 'precision_role',
          name: 'demo.one',
          definition: { admissible: ['bf16'], default: 'bf16', sensitivity: 'reduced' },
        }),
      }),
    });
    // The schema stage passes — a precision role is a well-formed unit — and the section decides.
    expect(library.problems[0]?.message).toBe(
      "base/axes/demo/one.json: kind 'precision_role', expected 'axis'",
    );
  });

  it('refuses a text that is not JSON at all, in CPython’s words', () => {
    const library = loadLibrary(['base'], {
      schemas,
      source: memorySource({ 'base/axes/demo/one.json': '{"schema": }' }),
    });
    expect(library.problems[0]?.code).toBe('V12');
    expect(library.problems[0]?.message).toBe(
      'base/axes/demo/one.json: Expecting value: line 1 column 12 (char 11) (V12)',
    );
  });

  it('refuses a duplicate member name under V12, after the schema stage', () => {
    // The schema stage reads the file with a plain `json.load` (last value wins), so a unit that
    // is on the schema under that reading is refused for the duplicate and not for the grammar.
    const unit =
      '{"schema": "tensorspine-primitive-library-unit/2.0", "kind": "axis",' +
      ' "name": "demo.one", "definition": {"space": "value", "space": "instance"}}';
    const library = loadLibrary(['base'], {
      schemas,
      source: memorySource({ 'base/axes/demo/one.json': unit }),
    });
    expect(library.problems[0]?.message).toBe(
      "base/axes/demo/one.json: duplicate member name 'space' (V12)",
    );
  });

  it('refuses a duplicate for the grammar first when the last value is off the schema', () => {
    const unit =
      '{"schema": "tensorspine-primitive-library-unit/2.0", "kind": "axis",' +
      ' "name": "demo.one", "definition": {"space": "value", "space": "sideways"}}';
    const library = loadLibrary(['base'], {
      schemas,
      source: memorySource({ 'base/axes/demo/one.json': unit }),
    });
    expect(formatLibraryProblems(library.problems)[0]).toBe(
      'base/axes/demo/one.json: off the primitive-library-unit schema\n' +
        "    definition/space: 'sideways' is not one of ['value', 'instance', 'storage']",
    );
  });

  it('refuses a precision role whose default is outside its own admissible set', () => {
    const base = rejectionBase('primitive-library/role-default-inadmissible');
    const library = loadLibrary([base, REFERENCE_BASE], { schemas, source });
    expect(library.problems[0]?.message).toBe(
      `${base}/precision/state/kv_test.json: default 'fp4' is not in the admissible set ` +
        "['bf16', 'f16', 'f8e4m3']",
    );
  });

  it('reports the missing unit schema before it reads anything', () => {
    const documentation = source.read(
      repositoryPath('schemas', 'tensorspine-documentation.schema.json'),
    );
    const partial = loadSchemas([{ path: 'schemas/documentation.json', text: documentation }], {
      origin: 'schemas',
    });
    const library = loadLibrary([REFERENCE_BASE], { schemas: partial, source });
    expect(formatLibraryProblems(library.problems)).toEqual([
      'no schema with $id ending in /primitive-library-unit.json under schemas/',
    ]);
  });
});

describe('a monolithic base', () => {
  // "A base is either an exploded directory … or a single monolithic file, still accepted." Its
  // units carry no file of their own, so they are never read against the unit schema.
  const monolithic = JSON.stringify({
    axes: { 'demo.one': { space: 'value', summary: 'One.' } },
    precision: { 'demo.role': { admissible: ['bf16'], default: 'bf16', sensitivity: 'reduced' } },
    primitives: {},
  });

  it('provides its axes and roles without a structural stage', () => {
    const library = loadLibrary(['bundle.json'], {
      schemas,
      source: memorySource({ 'bundle.json': monolithic }),
    });
    expect(formatLibraryProblems(library.problems)).toEqual([]);
    expect(library.axes.size).toBe(1);
    expect(library.precision.size).toBe(1);
    expect(library.bases[0]?.kind).toBe('file');
  });

  it('still checks a role’s default against its admissible set, naming the base file', () => {
    const broken = JSON.stringify({
      precision: { 'demo.role': { admissible: ['bf16'], default: 'fp4', sensitivity: 'reduced' } },
    });
    const library = loadLibrary(['bundle.json'], {
      schemas,
      source: memorySource({ 'bundle.json': broken }),
    });
    expect(library.problems[0]?.message).toBe(
      "bundle.json: default 'fp4' is not in the admissible set ['bf16']",
    );
  });
});

describe('the bases a document resolves from', () => {
  const modelPath = repositoryPath('data', 'models', 'llama3-8b.json');
  const model = toPython(parse(source.read(modelPath)));

  it('resolves `primitive_libraries` against the document’s own directory', () => {
    const resolved = basesOf(modelPath, model);
    expect(resolved.problem).toBeNull();
    expect(resolved.bases).toEqual([REFERENCE_BASE]);
  });

  it('refuses a document that is not tensorspine/2.0', () => {
    const resolved = basesOf('x.json', { schema: 'tensorspine/1.0' });
    expect(resolved.problem?.message).toBe(
      'x.json: expected tensorspine/2.0; convert supported legacy input with ' +
        'python3 tools/migrate.py INPUT -o OUTPUT',
    );
  });

  it('refuses a declared base that does not exist, as V1', () => {
    const library = librariesFor(modelPath, model, { schemas, source }, ['/nowhere/base']);
    expect(library.problems[0]?.code).toBe('V1');
    expect(library.problems[0]?.message).toBe(
      "llama3-8b.json: primitive library base '/nowhere/base' does not exist (V1)",
    );
  });

  it('loads the reference base for a corpus document', () => {
    const library = librariesFor(modelPath, model, { schemas, source });
    expect(formatLibraryProblems(library.problems)).toEqual([]);
    expect(library.byId.size).toBe(36);
  });
});

describe('semantic versions', () => {
  it('orders by the integers, and reads anything else as zero', () => {
    expect(semanticVersion('1.2.3')).toEqual([1, 2, 3]);
    expect(semanticVersion('1.10.0')).toEqual([1, 10, 0]);
    expect(semanticVersion('1.0')).toEqual([1, 0]);
    expect(semanticVersion('1.0.0-rc1')).toEqual([0, 0, 0]);
    expect(semanticVersion('')).toEqual([0, 0, 0]);
  });

  it('indexes a primitive by name at its highest version, 1.10.0 above 1.2.0', () => {
    const file = repositoryPath(
      'data',
      'primitive-library',
      'primitives',
      'norm',
      'rms',
      '1.0.0.json',
    );
    const later = memorySource({
      'extra/primitives/norm/rms/1.10.0.json': variantOf(file, {
        version: '1.10.0',
        summary: 'Ten.',
      }),
      'extra/primitives/norm/rms/1.2.0.json': variantOf(file, {
        version: '1.2.0',
        summary: 'Two.',
      }),
    });
    const library = loadLibrary([REFERENCE_BASE, 'extra'], {
      schemas,
      source: overlay(source, later),
    });
    expect(formatLibraryProblems(library.problems)).toEqual([]);
    expect(library.byId.size).toBe(38);
    expect((library.primitives.get('norm.rms') as { summary: string }).summary).toBe('Ten.');
  });
});
