/**
 * The schema registry: `tools/schema.py` and `validate.structural`, ported.
 *
 * `schema.py` builds a registry by reading every schema of the directory and indexing it under
 * its own `$id` — "the mapping is discovered, not declared", so that renaming a file or moving
 * the directory needs no change. This module does the same over the files the editor is handed:
 * the static build vendors `schemas/` and reads them at startup, a workspace may carry its own,
 * and the plan's §1 makes those files the only source of vocabulary and structure.
 *
 * On top of the index it holds the one thing D4 decides — Ajv, compiled from those same files:
 *
 * - `conforms` is the verdict. One compiled validator, one call, about a millisecond on the
 *   largest corpus document.
 * - `structural` is the verdict *explained*, in the words `--validate` prints: it runs the walk
 *   of `walk.ts` only when Ajv has refused, so a document that conforms costs one Ajv call and
 *   nothing else.
 * - `explain` is `structural` without that shortcut. It exists so that a test can require the two
 *   readings to agree on every document, unit and fixture the repository holds — which is what
 *   keeps Ajv the validator and the walk a phrasing of Ajv's answer, and is D4's own catching
 *   rule ("a disagreement is a finding, and the specification decides").
 *
 * The stage before the schema is the JSON layer: a duplicate member name is refused (V12), never
 * resolved to the last value. `structuralText` runs it, because that is the order
 * `validate.structural` runs them in and the rejection suite has a case for it.
 */
import Ajv2020, { type ErrorObject, type ValidateFunction } from 'ajv/dist/2020.js';

import { JsonParseError, parse } from '../json/parse.js';
import type { JsonValue } from '../json/tree.js';
import type { AssertionEngine } from './assertions.js';
import { absolutePath, deepest, sortByPlace, type SchemaError } from './errors.js';
import {
  instanceOf,
  pointerOf,
  whereOf,
  type Instance,
  type PathSegment,
  type SchemaNode,
  type SchemaObject,
} from './types.js';
import { vocabularyOf, type Vocabulary } from './vocabulary.js';
import { iterErrors, type Resolved, type WalkEnv } from './walk.js';

// Ajv is CommonJS and states its default export twice; every reading of it — Node's interop,
// Vite's, a bundler's — lands on one of the two, and the class is its own `default`.
const Ajv = (Ajv2020 as unknown as { default?: typeof Ajv2020 }).default ?? Ajv2020;

/** One schema file as the editor is handed it: where it came from, and its text. */
export interface SchemaFile {
  readonly path: string;
  readonly text: string;
}

/** One schema of the registry, indexed under the `$id` it declares. */
export interface LoadedSchema {
  /** Where the file came from, for the log and for the mismatch warning of plan §1. */
  readonly path: string;
  /** The published identity: `https://tensorspine.dev/schema/2.0/model.json`. */
  readonly id: string;
  /** The last segment of the identity without `.json`: `model`, `primitive-library-unit`. */
  readonly role: string;
  /** The schema itself, as the file writes it. */
  readonly document: SchemaObject;
}

/** A structural problem: one line of `--validate`'s schema stage, with its place as a pointer. */
export interface StructuralProblem {
  /**
   * `schema` for the grammar stage, `V12` for the JSON layer that runs before it, `registry` for
   * the one thing `validate.structural` reports before either — that no schema of the role asked
   * for was loaded. The three are rendered differently, which is why they are told apart here.
   */
  readonly code: 'schema' | 'V12' | 'registry';
  /** The tools' message, word for word. */
  readonly message: string;
  /** Where in the document, as a JSON pointer (RFC 6901); `''` at the root. */
  readonly path: string;
  /** The same place as its segments, for a store that walks the tree. */
  readonly segments: readonly PathSegment[];
  /** The keyword that refused it, `null` for the JSON layer and for the `false` schema. */
  readonly keyword: string | null;
}

/** How a document is read: as the tools read it, or as the library loader reads a unit. */
export interface StructuralOptions {
  /**
   * Report the leaf behind each error rather than the error itself — `schema.deepest`, which is
   * what `primitive_library` prints for a unit that is off the unit schema.
   */
  readonly deepest?: boolean;
}

/** Raised when a file handed to {@link loadSchemas} is not a schema the registry can index. */
export class SchemaLoadError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SchemaLoadError';
  }
}

/** The registry: the schemas by identity, Ajv over them, and what the interface reads. */
export interface SchemaRegistry {
  /** Where the schemas came from, as the "no schema with $id ending in …" line names it. */
  readonly origin: string;
  /** Every schema indexed, in the order the files were read. */
  readonly schemas: readonly LoadedSchema[];
  /** The schema of that identity, or `undefined`. */
  byId(id: string): LoadedSchema | undefined;
  /** `schema.locate`: the schema whose `$id` ends in `/<role>.json`. */
  locate(role: string): LoadedSchema | undefined;
  /** Ajv's verdict on a tree against the schema of that role. */
  conforms(tree: JsonValue, role?: string): boolean;
  /** The verdict explained: the problems `--validate`'s schema stage would print. */
  structural(tree: JsonValue, role?: string, options?: StructuralOptions): StructuralProblem[];
  /** `structural`, but the walk always runs: the reading a parity test holds Ajv's against. */
  explain(tree: JsonValue, role?: string, options?: StructuralOptions): StructuralProblem[];
  /** The JSON layer (V12) and then the schema stage, from a document's text. */
  structuralText(text: string, role?: string, options?: StructuralOptions): StructuralProblem[];
  /** Every enum and every `oneOf` of the loaded schemas, by pointer (plan §1). */
  vocabulary(): Vocabulary;
}

/** The tools' rendering of one problem: `<where>: <message>`, `[V12] <message>`, or bare. */
export function formatProblem(problem: StructuralProblem): string {
  if (problem.code === 'V12') return `[V12] ${problem.message}`;
  if (problem.code === 'registry') return problem.message;
  return `${whereOf(problem.segments)}: ${problem.message}`;
}

/** The lines `validate.structural` returns for a document: {@link formatProblem} over each. */
export function formatProblems(problems: readonly StructuralProblem[]): string[] {
  return problems.map((problem) => formatProblem(problem));
}

/** The directory the files came from, as the tools' "no schema …" line names it. */
function commonDirectory(files: readonly SchemaFile[]): string {
  const directories = files.map((file) => file.path.replace(/[^/\\]*$/, '').replace(/[/\\]$/, ''));
  const first = directories[0];
  if (first === undefined) return 'schemas';
  return directories.every((directory) => directory === first) && first !== '' ? first : 'schemas';
}

/** The segments of a JSON pointer, unescaped. */
function pointerSegments(pointer: string): string[] {
  if (pointer === '' || pointer === '#') return [];
  const body = pointer.startsWith('#') ? pointer.slice(1) : pointer;
  if (!body.startsWith('/')) throw new SchemaLoadError(`'${pointer}' is not a JSON pointer`);
  return body
    .slice(1)
    .split('/')
    .map((segment) => decodeURIComponent(segment).replace(/~1/g, '/').replace(/~0/g, '~'));
}

/** The node a JSON pointer names inside a schema document. */
function nodeAt(document: SchemaObject, pointer: string, ref: string): SchemaNode {
  let node: unknown = document;
  for (const segment of pointerSegments(pointer)) {
    if (node === null || typeof node !== 'object') {
      throw new SchemaLoadError(`'${ref}' names nothing: '${segment}' has no container`);
    }
    node = Array.isArray(node)
      ? node[Number(segment)]
      : (node as Record<string, unknown>)[segment];
  }
  if (node === undefined) throw new SchemaLoadError(`'${ref}' names nothing in the registry`);
  return node as SchemaNode;
}

/**
 * Ajv, asked one keyword at a time.
 *
 * The schema compiled for a keyword is that keyword's own value, taken from the file, with the
 * siblings it cannot be read without: `additionalProperties: false` needs to know which names
 * `properties` and `patternProperties` cover, and `items: false` how many `prefixItems` covers.
 * Those siblings are reduced to `true`, so the compiled schema asserts the one keyword and
 * descends into nothing.
 */
class KeywordAssertions implements AssertionEngine {
  private readonly cache = new WeakMap<SchemaObject, Map<string, ValidateFunction | null>>();

  constructor(private readonly ajv: InstanceType<typeof Ajv2020>) {}

  failures(node: SchemaObject, keyword: string, data: unknown): readonly ErrorObject[] {
    const validate = this.validator(node, keyword);
    if (validate === null) return [];
    return validate(data) ? [] : (validate.errors ?? []);
  }

  private validator(node: SchemaObject, keyword: string): ValidateFunction | null {
    let byKeyword = this.cache.get(node);
    if (byKeyword === undefined) {
      byKeyword = new Map();
      this.cache.set(node, byKeyword);
    }
    const known = byKeyword.get(keyword);
    if (known !== undefined) return known;
    const built = this.build(node, keyword);
    const compiled = built === null ? null : this.ajv.compile(built);
    byKeyword.set(keyword, compiled);
    return compiled;
  }

  private build(node: SchemaObject, keyword: string): SchemaObject | null {
    // The tools build their validator without a format checker, so `format` asserts nothing.
    if (keyword === 'format') return null;
    if (keyword === 'additionalProperties') {
      return {
        properties: allowAll(node['properties']),
        patternProperties: allowAll(node['patternProperties']),
        additionalProperties: false,
      };
    }
    if (keyword === 'items') {
      const prefixItems = node['prefixItems'];
      return {
        prefixItems: Array.isArray(prefixItems) ? prefixItems.map(() => true) : [],
        items: node['items'],
      };
    }
    return { [keyword]: node[keyword] };
  }
}

/** The names a `properties`-shaped keyword covers, each mapped to the schema that allows anything. */
function allowAll(declared: unknown): Record<string, true> {
  if (declared === null || typeof declared !== 'object') return {};
  const allowed: Record<string, true> = {};
  for (const name of Object.keys(declared)) allowed[name] = true;
  return allowed;
}

/**
 * The registry over a set of schema files.
 *
 * The files are indexed by the `$id` each declares, as `schema.discover` does; a file that
 * declares none is not part of the namespace and is skipped, exactly as the tools skip it.
 */
export function loadSchemas(
  files: readonly SchemaFile[],
  options: { readonly origin?: string } = {},
): SchemaRegistry {
  const schemas: LoadedSchema[] = [];
  const byId = new Map<string, LoadedSchema>();
  // `allErrors` because a message that names every missing property needs every one of them;
  // `strict: false` and `validateFormats: false` because `Draft202012Validator(schema,
  // registry=…)` is built with neither a strict mode nor a format checker, and the two must read
  // the same schemas the same way; `ownProperties` because a member named `toString` is a member
  // and not something inherited from `Object.prototype`.
  const ajv = new Ajv({
    allErrors: true,
    strict: false,
    validateFormats: false,
    ownProperties: true,
  });

  // `schema.discover` reads `sorted(glob(...))` and keeps one entry per identity, so of two files
  // declaring the same `$id` the later one is the one indexed. The files arrive here in whatever
  // order the caller read them, and are put in that order first so that the same file wins.
  const ordered = [...files].sort((left, right) =>
    left.path < right.path ? -1 : left.path > right.path ? 1 : 0,
  );

  for (const file of ordered) {
    let document: unknown;
    try {
      document = JSON.parse(file.text);
    } catch (error) {
      throw new SchemaLoadError(`${file.path}: not JSON (${(error as Error).message})`);
    }
    if (document === null || typeof document !== 'object' || Array.isArray(document)) {
      throw new SchemaLoadError(`${file.path}: a schema is a JSON object`);
    }
    const identity = (document as Record<string, unknown>)['$id'];
    if (typeof identity !== 'string' || identity === '') continue;
    const loaded: LoadedSchema = {
      path: file.path,
      id: identity,
      role: identity.slice(identity.lastIndexOf('/') + 1).replace(/\.json$/, ''),
      document: document as SchemaObject,
    };
    const already = schemas.findIndex((schema) => schema.id === identity);
    if (already >= 0) schemas.splice(already, 1);
    schemas.push(loaded);
    byId.set(identity, loaded);
    ajv.removeSchema(identity);
    ajv.addSchema(loaded.document, identity);
  }

  const origin = options.origin ?? commonDirectory(files);
  const assertions = new KeywordAssertions(ajv);
  const env: WalkEnv = {
    assertions,
    resolve(ref: string, base: string): Resolved {
      const hash = ref.indexOf('#');
      const uri = hash < 0 ? ref : ref.slice(0, hash);
      const pointer = hash < 0 ? '' : ref.slice(hash);
      const target = uri === '' ? byId.get(base) : byId.get(uri);
      if (target === undefined) {
        throw new SchemaLoadError(`'${ref}' names a schema the registry does not hold`);
      }
      return { node: nodeAt(target.document, pointer, ref), base: target.id };
    },
  };

  const compiled = new Map<string, ValidateFunction>();
  const validatorFor = (schema: LoadedSchema): ValidateFunction => {
    const known = compiled.get(schema.id);
    if (known !== undefined) return known;
    const made = ajv.getSchema(schema.id);
    if (made === undefined) {
      throw new SchemaLoadError(`${schema.path}: Ajv did not compile ${schema.id}`);
    }
    compiled.set(schema.id, made);
    return made;
  };

  const locate = (role: string): LoadedSchema | undefined =>
    schemas.find((schema) => schema.id.endsWith(`/${role}.json`));

  const missingRole = (role: string): StructuralProblem[] => [
    {
      code: 'registry',
      message: `no schema with $id ending in /${role}.json under ${origin}/`,
      path: '',
      segments: [],
      keyword: null,
    },
  ];

  const walk = (
    instance: Instance,
    schema: LoadedSchema,
    options_: StructuralOptions,
  ): StructuralProblem[] => {
    const raised = iterErrors(instance, schema.document, schema.id, env);
    const selected = options_.deepest === true ? deepest(raised) : sortByPlace(raised);
    return selected.map((error) => problemOf(error));
  };

  const registry: SchemaRegistry = {
    origin,
    schemas,
    byId: (id) => byId.get(id),
    locate,
    conforms(tree, role = 'model') {
      const schema = locate(role);
      if (schema === undefined) return false;
      // No schema of the registry is asynchronous, so the verdict is a boolean and not a promise.
      return validatorFor(schema)(instanceOf(tree).plain) === true;
    },
    explain(tree, role = 'model', options_ = {}) {
      const schema = locate(role);
      if (schema === undefined) return missingRole(role);
      return walk(instanceOf(tree), schema, options_);
    },
    structural(tree, role = 'model', options_ = {}) {
      const schema = locate(role);
      if (schema === undefined) return missingRole(role);
      const instance = instanceOf(tree);
      const validate = validatorFor(schema);
      if (validate(instance.plain) === true) return [];
      const explained = walk(instance, schema, options_);
      if (explained.length > 0) return explained;
      // Ajv refused and the walk found nothing to say: a divergence between the two, which the
      // parity suite requires never to happen. Report Ajv's own words rather than nothing, so
      // that a refusal is never silent.
      return (validate.errors ?? []).slice(0, 1).map((error) => ({
        code: 'schema' as const,
        message: `${error.instancePath}: ${error.message ?? 'is not valid'} (Ajv)`,
        path: error.instancePath,
        segments: pointerSegments(error.instancePath),
        keyword: error.keyword,
      }));
    },
    structuralText(text, role = 'model', options_ = {}) {
      let tree: JsonValue;
      try {
        tree = parse(text);
      } catch (error) {
        if (error instanceof JsonParseError) {
          // `validate.structural` runs the JSON layer first and reports a duplicate as
          // `[V12] duplicate member name 'x' (V12)`. It has no line for a document that is not
          // JSON at all — `json.load` raises out of it uncaught — so the parse error's own text
          // is stated under the same code rather than crashing: a text that cannot be read is
          // refused either way, and the reader is told why.
          return [
            { code: 'V12', message: error.message, path: '', segments: [], keyword: null },
          ];
        }
        throw error;
      }
      return registry.structural(tree, role, options_);
    },
    vocabulary: () => vocabularyOf(schemas),
  };
  return registry;
}

/** One walked error as a problem: the tools' message, the place as a pointer. */
function problemOf(error: SchemaError): StructuralProblem {
  const segments = absolutePath(error);
  return {
    code: 'schema',
    message: error.message,
    path: pointerOf(segments),
    segments,
    keyword: error.keyword,
  };
}
