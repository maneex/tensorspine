/**
 * `primitive_library.load`: the bases gathered, and every refusal they carry.
 *
 * "Bases form a set: an identity is provided by whichever base carries it, and two bases carrying
 * one identity with different contents are a conflict (V1), never a choice." A unit's path
 * reproduces its identity dot by dot, a primitive's file name is its version, and a disagreement
 * between the path and what is written inside is a refusal, not a preference (§8.2).
 *
 * The tools raise at the first refusal and stop. The port keeps their **staging** — the manifest
 * before the units, the schema before the reading, the whole set gathered before any
 * cross-reference is resolved — and continues past the point they would abort at, which is what
 * plan §3 asks of the port ("extra rows, never fewer") and what the Problems panel needs. That
 * makes the parity contract exact and easy to state:
 *
 * > the first problem, rendered by `formatLibraryProblem`, is the text the tools would have
 * > raised; the reference base yields none.
 *
 * Continuation never *removes* a problem, because it never skips a check the tools reached: a unit
 * the schema refused is left out of the gathered set, as it is left out by an abort, and every
 * problem that follows is marked `afterRefusal`.
 *
 * Two things the tools do that the port deliberately does not:
 *
 * - **The cache.** `load_for` memoises by the set of bases; in the editor that is the worker's
 *   business (§5.6: "library load once per session, cached; a saved unit reloads its base alone"),
 *   and a loader that cached would hand the same object to two documents.
 * - **The working directory.** `bases_of` calls `os.path.abspath`, which resolves against the
 *   process's cwd. The core has none, so a base is resolved lexically against the document that
 *   declares it — the reading feature 0.6 already took for `Workspace.resolve`.
 */
import type { JsonValue } from '../json/tree.js';
import { toPython, type PyRecord, type PyValue } from '../expr/value.js';
import { pyEqual } from '../expr/arithmetic.js';
import { comparePythonStrings } from '../schema/repr.js';
import {
  formatProblem,
  type SchemaRegistry,
  type StructuralProblem,
} from '../schema/registry.js';
import { asText, get, has, members } from './access.js';
import { basename, dirname, join, normalise, relative } from './paths.js';
import { afterRefusal, libraryProblem, type LibraryProblem } from './problems.js';
import { primitiveReferences, type ReferenceLibrary } from './references.js';
import { readRefusal, readText, type ReadText } from './read.js';
import { pyRepr } from './repr.js';
import type { LibrarySource } from './source.js';
import { pinnedTemplate, templateInterface, type TemplatePin } from './template.js';

/** The `schema` every unit of a base declares. */
export const UNIT_SCHEMA = 'tensorspine-primitive-library-unit/2.0';

/** `SECTIONS`: the directory a kind of unit lives in, and the `kind` its file must declare. */
export const SECTIONS = [
  ['primitives', 'primitive'],
  ['axes', 'axis'],
  ['precision', 'precision_role'],
] as const;

/** One of the three sections of an exploded base. */
export type LibrarySection = (typeof SECTIONS)[number][0];

/** What the loader was given to read with. */
export interface LibraryContext {
  /**
   * The schemas every exploded unit is read against. `null` skips the structural stage, as
   * `schema_dir=None` does — which is what a monolithic base gets, its units carrying no file.
   */
  readonly schemas: SchemaRegistry | null;
  /** Where the bytes come from. */
  readonly source: LibrarySource;
  /** `models_base`: one templates location for every base, overriding each manifest (tests only). */
  readonly modelsBase?: string | null;
}

/** One unit of a base, as the load read it. */
export interface LibraryUnit {
  /** The file, as every refusal about it names it. */
  readonly file: string;
  /** The base it belongs to, as the load was given it. */
  readonly base: string;
  readonly section: LibrarySection;
  /** The `kind` the file declares, which the section decides. */
  readonly kind: (typeof SECTIONS)[number][1];
  /** The identity its path spells out: `attention.dense`, `model.width`, `norm.scale`. */
  readonly name: string;
  /** A primitive's version, from its file name; `null` for an axis and a precision role. */
  readonly version: string | null;
  /** The whole file, as the evaluators read a document. */
  readonly unit: PyValue;
  /** `unit['definition']`. */
  readonly definition: PyValue;
  /** The ordered tree, for a caller that opens the unit in the primitive editor (D1). */
  readonly tree: JsonValue;
}

/** One primitive version of the gathered set, with where it came from. */
export interface PrimitiveVersion {
  readonly name: string;
  readonly version: string;
  readonly definition: PyValue;
  /** The unit file, or the monolithic base's own path — `origin[('primitives', …)]`. */
  readonly file: string;
  /** The base that provided it — `origin[('base-of', …)]`. */
  readonly base: string;
}

/** One base of the load. */
export interface LoadedBase {
  readonly path: string;
  /** An exploded directory, or a single monolithic file, still accepted. */
  readonly kind: 'directory' | 'file';
  /** `primitive-library.json`'s definition, `null` when the base carries none. */
  readonly manifest: PyValue | null;
  /** Where its template documents live, resolved against the base; `null` when it says nothing. */
  readonly templates: string | null;
  readonly units: readonly LibraryUnit[];
}

/** The primitive library the bases denote — `cat`, with the loader's refusals beside it. */
export interface Library extends ReferenceLibrary {
  /** The bases, in the order they were given. */
  readonly bases: readonly LoadedBase[];
  /** `by_id`, keyed `<name>@<version>`: one identity, one content. */
  readonly byId: ReadonlyMap<string, PrimitiveVersion>;
  /** `primitives`, by name alone: the highest version, for a reading that does not pin. */
  readonly primitives: ReadonlyMap<string, PyValue>;
  readonly axes: ReadonlyMap<string, PyValue>;
  readonly precision: ReadonlyMap<string, PyValue>;
  /** The template document each template primitive pins, keyed `<name>@<version>`. */
  readonly templates: ReadonlyMap<string, TemplatePin>;
  /** Every refusal, in the order the tools would have met them; the first is the one they raise. */
  readonly problems: readonly LibraryProblem[];
}

/** A primitive identity as a key of {@link Library.byId}; `@` cannot occur in either half. */
export function identityKey(name: string, version: string): string {
  return `${name}@${version}`;
}

/**
 * `primitive_library.primitive`: the primitive a `{name, version}` reference designates — the
 * pinned version first, falling back to the name so that a version mismatch is reported by the
 * caller rather than looking like an absent primitive.
 */
export function primitiveOf(
  library: Library,
  reference: { name: string; version: string },
): PyValue | undefined {
  const pinned = library.byId.get(identityKey(reference.name, reference.version));
  if (pinned !== undefined) return pinned.definition;
  return library.primitives.get(reference.name);
}

/** `primitive_library.template_primitives`: the names whose primitive pins a template (§4.6). */
export function templatePrimitives(library: Library): Set<string> {
  const found = new Set<string>();
  for (const [name, definition] of library.primitives) {
    if (has(definition, 'template')) found.add(name);
  }
  return found;
}

/**
 * `primitive_library.template_path`: the template file of a template primitive, as pinned and
 * checked at load. `undefined` when the load did not resolve it — where the tools raise, since a
 * load that resolved nothing has already refused.
 */
export function templatePinOf(library: Library, definition: PyValue): TemplatePin | undefined {
  for (const [key, pin] of library.templates) {
    const version = library.byId.get(key);
    if (version !== undefined && pyEqual(version.definition, definition)) return pin;
  }
  return undefined;
}

/** Every unit of every base, in the order the load read them — what §5.3 calls "units". */
export function libraryUnits(library: Library): LibraryUnit[] {
  return library.bases.flatMap((base) => [...base.units]);
}

/**
 * The interface each template primitive presents, keyed `<name>@<version>` — §5.3's "template
 * interfaces", computed from the documents the load pinned.
 *
 * The tools compute it at the call site, in `validate.analyse`; here it is computed from the
 * library alone because a template instance's node and the primitive editor's header both read it
 * before any document is validated (D9, §4.22).
 */
export function templateInterfaces(library: Library): Map<string, PyRecord> {
  const out = new Map<string, PyRecord>();
  for (const [key, pin] of library.templates) {
    const version = library.byId.get(key);
    if (version === undefined) continue;
    out.set(key, templateInterface(version.definition, pin.document));
  }
  return out;
}

/** `_semver`: the version as a tuple of integers, `(0, 0, 0)` for anything else. */
export function semanticVersion(version: string): number[] {
  const parts = version.split('.');
  const out: number[] = [];
  for (const part of parts) {
    // `int(x)` accepts surrounding whitespace and a sign, and refuses everything else here.
    if (!/^\s*[+-]?\d+\s*$/.test(part)) return [0, 0, 0];
    out.push(Number.parseInt(part.trim(), 10));
  }
  return out;
}

/** Python's ordering of two version tuples: element by element, then by length. */
function compareVersions(left: readonly number[], right: readonly number[]): number {
  const shared = Math.min(left.length, right.length);
  for (let index = 0; index < shared; index += 1) {
    const a = left[index] as number;
    const b = right[index] as number;
    if (a !== b) return a < b ? -1 : 1;
  }
  return left.length - right.length;
}

/** A mutable store of one kind of identity, with where each one came from. */
interface Provided {
  readonly values: Map<string, PyValue>;
  readonly origin: Map<string, string>;
}

/** The load in progress: the stores, the origins, and the problems in the tools' own order. */
export class Gathering {
  readonly primitives: Provided = { values: new Map(), origin: new Map() };
  readonly axes: Provided = { values: new Map(), origin: new Map() };
  readonly precision: Provided = { values: new Map(), origin: new Map() };
  readonly baseOf = new Map<string, string>();
  readonly problems: LibraryProblem[] = [];

  /** A refusal, marked as one the tools would never have reached once there is an earlier one. */
  refuse(problem: LibraryProblem): void {
    this.problems.push(this.problems.length === 0 ? problem : afterRefusal(problem));
  }

  /**
   * `_provide`: one identity, one content.
   *
   * `label` is what the message writes after `identity` — a primitive's is a Python tuple, an
   * axis's and a role's the bare name, because that is what `f"{key}"` does to each.
   */
  provide(store: Provided, key: string, label: string, definition: PyValue, where: string): void {
    const current = store.values.get(key);
    if (current === undefined) {
      store.values.set(key, definition);
      store.origin.set(key, where);
      return;
    }
    if (pyEqual(current, definition)) return;
    this.refuse(
      libraryProblem(
        'conflict',
        where,
        `${where}: identity ${label} is also carried by ${store.origin.get(key) ?? 'None'} with ` +
          'different contents (V1)',
        { code: 'V1' },
      ),
    );
  }
}

/**
 * The primitive library gathered from the given bases.
 *
 * The order of the bases matters to two things and to nothing else: which base an identity's
 * origin names in a conflict, and which base a template primitive's `templates` location is taken
 * from. It never decides *which* content wins — a disagreement is a refusal.
 */
export function loadLibrary(bases: readonly string[], context: LibraryContext): Library {
  const { schemas, source } = context;
  const modelsBase = context.modelsBase ?? null;
  const gathering = new Gathering();
  const loaded: LoadedBase[] = [];

  if (schemas !== null && schemas.locate('primitive-library-unit') === undefined) {
    // `load` raises this before it reads anything; the registry writes the same sentence.
    gathering.refuse(
      libraryProblem(
        'schema',
        '',
        `no schema with $id ending in /primitive-library-unit.json under ${schemas.origin}/`,
      ),
    );
    return library([], gathering, new Map(), gathering.problems);
  }

  for (const base of bases) {
    if (source.isDirectory(base)) {
      loaded.push(readDirectoryBase(base, gathering, context, modelsBase));
    } else {
      loaded.push(readMonolithicBase(base, gathering, context, modelsBase));
    }
  }

  // A precision role's default must be one of the dtypes it admits; the roles are walked in the
  // order they were provided, which is the order `precision.items()` walks them in.
  for (const [name, role] of gathering.precision.values) {
    const fallback = get(role, 'default');
    const admissible = get(role, 'admissible');
    const admits =
      Array.isArray(admissible) &&
      (admissible as readonly PyValue[]).some((one) => pyEqual(one, fallback));
    if (admits) continue;
    const where = gathering.precision.origin.get(name) ?? name;
    gathering.refuse(
      libraryProblem(
        'precision',
        where,
        `${where}: default '${asText(fallback)}' is not in the admissible set ${pyRepr(admissible)}`,
      ),
    );
  }

  const byId = identities(gathering);
  const templates = new Map<string, TemplatePin>();
  const primitives = highestVersions(byId);
  const resolved: ReferenceLibrary = {
    axes: gathering.axes.values,
    precision: gathering.precision.values,
  };
  const templatesOf = new Map(loaded.map((one) => [one.path, one.templates]));

  for (const version of sortedIdentities(byId)) {
    const problems = primitiveReferences(version.definition, resolved);
    if (problems.length > 0) {
      gathering.refuse(
        libraryProblem('reference', version.file, `${version.file}: unresolved reference(s)`, {
          detail: problems.map((one) => ({
            message: one.message,
            path: `/definition${one.path}`,
            segments: ['definition', ...one.segments],
          })),
        }),
      );
      continue;
    }
    if (!has(version.definition, 'template')) continue;
    const location = templatesOf.get(version.base) ?? null;
    if (location === null) {
      gathering.refuse(
        libraryProblem(
          'template',
          version.file,
          `${version.file}: a template primitive, but its base declares no \`templates\` ` +
            'location (§4.6)',
        ),
      );
      continue;
    }
    const pinned = pinnedTemplate(version.definition, version.file, location, source, schemas);
    if (pinned.problem !== null) gathering.refuse(pinned.problem);
    if (pinned.pin !== null) {
      templates.set(identityKey(version.name, version.version), pinned.pin);
    }
  }

  return library(loaded, gathering, templates, gathering.problems, byId, primitives);
}

/** The `Library` from what the load gathered. */
function library(
  bases: readonly LoadedBase[],
  gathering: Gathering,
  templates: ReadonlyMap<string, TemplatePin>,
  problems: readonly LibraryProblem[],
  byId: ReadonlyMap<string, PrimitiveVersion> = new Map(),
  primitives: ReadonlyMap<string, PyValue> = new Map(),
): Library {
  return {
    bases,
    byId,
    primitives,
    axes: gathering.axes.values,
    precision: gathering.precision.values,
    templates,
    problems,
  };
}

/** `by_id`, in the order the units were provided. */
function identities(gathering: Gathering): Map<string, PrimitiveVersion> {
  const out = new Map<string, PrimitiveVersion>();
  for (const [key, definition] of gathering.primitives.values) {
    const cut = key.lastIndexOf('@');
    out.set(key, {
      name: key.slice(0, cut),
      version: key.slice(cut + 1),
      definition,
      file: gathering.primitives.origin.get(key) ?? key,
      base: gathering.baseOf.get(key) ?? '',
    });
  }
  return out;
}

/**
 * `sorted(by_id.items())`: by name, then by version, as Python orders two tuples of strings —
 * never as one `name@version` string, which would compare `.` with `@` and put `a.b@0` first.
 */
function sortedIdentities(byId: ReadonlyMap<string, PrimitiveVersion>): PrimitiveVersion[] {
  return [...byId.values()].sort((left, right) => {
    const order = comparePythonStrings(left.name, right.name);
    return order !== 0 ? order : comparePythonStrings(left.version, right.version);
  });
}

/**
 * The index by name alone, "for readings that address a primitive by name (the linter's inventory,
 * D1's template walk); an instance always pins". The highest version wins, and a tie keeps the one
 * provided first, as `>` on two equal tuples does.
 */
function highestVersions(byId: ReadonlyMap<string, PrimitiveVersion>): Map<string, PyValue> {
  const out = new Map<string, PyValue>();
  const chosen = new Map<string, number[]>();
  for (const one of byId.values()) {
    const version = semanticVersion(one.version);
    const current = chosen.get(one.name);
    if (current === undefined || compareVersions(version, current) > 0) {
      out.set(one.name, one.definition);
      chosen.set(one.name, version);
    }
  }
  return out;
}

/** An exploded base: its manifest, then every unit of its three sections. */
function readDirectoryBase(
  base: string,
  gathering: Gathering,
  context: LibraryContext,
  modelsBase: string | null,
): LoadedBase {
  const { source } = context;
  const manifest = readManifest(base, gathering, context);
  let templates: string | null = null;
  if (manifest !== null && has(manifest, 'templates')) {
    templates = normalise(join(base, asText(get(manifest, 'templates'))));
  }
  if (modelsBase !== null) templates = modelsBase;

  const units: LibraryUnit[] = [];
  for (const [section, kind] of SECTIONS) {
    const root = join(base, section);
    if (!source.isDirectory(root)) continue;
    const files = [...source.find(root)].sort(comparePythonStrings);
    for (const file of files) {
      const unit = readUnit(file, root, base, section, kind, gathering, context);
      if (unit === null) continue;
      units.push(unit);
      provideUnit(unit, gathering);
    }
  }
  return { path: base, kind: 'directory', manifest, templates, units };
}

/** `_manifest`: the base's own `primitive-library.json`, `null` when it carries none. */
export function readManifest(
  base: string,
  gathering: Gathering,
  context: LibraryContext,
): PyValue | null {
  const path = join(base, 'primitive-library.json');
  const source = context.source;
  if (!source.isFile(path)) return null;
  const read = readFile(path, gathering, context, 'primitive-library-unit');
  if (read === null) return null;
  const kind = get(read.value, 'kind');
  if (!pyEqual(kind, 'base')) {
    gathering.refuse(
      libraryProblem('identity', path, `${path}: kind ${pyRepr(kind)}, expected 'base'`),
    );
    return null;
  }
  return get(read.value, 'definition');
}

/** One unit file: the schema stage, the reading, and the agreement between path and identity. */
function readUnit(
  file: string,
  root: string,
  base: string,
  section: LibrarySection,
  kind: (typeof SECTIONS)[number][1],
  gathering: Gathering,
  context: LibraryContext,
): LibraryUnit | null {
  const read = readFile(file, gathering, context, 'primitive-library-unit');
  if (read === null) return null;
  const parts = relative(file, root).replace(/\.json$/, '').split('/');
  const name = section === 'primitives' ? parts.slice(0, -1).join('.') : parts.join('.');
  const version = section === 'primitives' ? (parts[parts.length - 1] as string) : null;
  const problem = identityProblem(file, read.value, kind, name, version);
  if (problem !== null) {
    gathering.refuse(problem);
    return null;
  }
  const definition = get(read.value, 'definition');
  return { file, base, section, kind, name, version, unit: read.value, definition, tree: read.tree };
}

/**
 * The four agreements `_units` requires between a unit and its place: the format it declares, the
 * kind its section decides, the name its path spells, and — for a primitive — the version its file
 * name is (§8.2).
 */
export function identityProblem(
  file: string,
  unit: PyValue,
  kind: string,
  name: string,
  version: string | null,
): LibraryProblem | null {
  const at = (name_: string): PyValue => get(unit, name_);
  const refuse = (message: string): LibraryProblem =>
    libraryProblem('identity', file, `${file}: ${message}`);
  if (!pyEqual(at('schema'), UNIT_SCHEMA)) {
    return refuse(`schema ${pyRepr(at('schema'))}, expected ${pyRepr(UNIT_SCHEMA)}`);
  }
  if (!pyEqual(at('kind'), kind)) {
    return refuse(`kind ${pyRepr(at('kind'))}, expected ${pyRepr(kind)}`);
  }
  if (!pyEqual(at('name'), name)) {
    return refuse(`name ${pyRepr(at('name'))}, path says ${pyRepr(name)}`);
  }
  if (version === null) return null;
  const declared = get(at('definition'), 'version');
  if (!pyEqual(declared, version)) {
    return refuse(`version ${pyRepr(declared)}, path says ${pyRepr(version)}`);
  }
  return null;
}

/** A unit into the store its section decides. */
function provideUnit(unit: LibraryUnit, gathering: Gathering): void {
  if (unit.section === 'primitives') {
    const key = identityKey(unit.name, unit.version ?? '');
    const label = `(${pyRepr(unit.name)}, ${pyRepr(unit.version ?? null)})`;
    gathering.provide(gathering.primitives, key, label, unit.definition, unit.file);
    if (!gathering.baseOf.has(key)) gathering.baseOf.set(key, unit.base);
    return;
  }
  const store = unit.section === 'axes' ? gathering.axes : gathering.precision;
  gathering.provide(store, unit.name, unit.name, unit.definition, unit.file);
}

/**
 * A monolithic base: one file carrying `primitives`, `axes` and `precision` as maps.
 *
 * It is "still accepted", and its units are never checked against the unit schema — they carry no
 * file of their own, which is what `schema_dir=None` means at that call site.
 */
function readMonolithicBase(
  base: string,
  gathering: Gathering,
  context: LibraryContext,
  modelsBase: string | null,
): LoadedBase {
  const read = readFile(base, gathering, context, null);
  const templates = modelsBase;
  if (read === null) return { path: base, kind: 'file', manifest: null, templates, units: [] };
  const document = read.value;
  const section = (name: string): [string, PyValue][] => members(get(document, name));
  for (const [name, definition] of section('primitives')) {
    const version = get(definition, 'version');
    const key = identityKey(name, asText(version));
    const label = `(${pyRepr(name)}, ${pyRepr(version)})`;
    gathering.provide(gathering.primitives, key, label, definition, base);
    if (!gathering.baseOf.has(key)) gathering.baseOf.set(key, base);
  }
  for (const [name, definition] of section('axes')) {
    gathering.provide(gathering.axes, name, name, definition, base);
  }
  for (const [name, definition] of section('precision')) {
    gathering.provide(gathering.precision, name, name, definition, base);
  }
  return { path: base, kind: 'file', manifest: null, templates, units: [] };
}

/**
 * One file, read in the tools' own order: the schema stage on the plain reading, then `read_json`
 * with its duplicate, legacy and revision refusals. `null` when the file is refused.
 */
export function readFile(
  path: string,
  gathering: Gathering,
  context: LibraryContext,
  role: string | null,
): ReadText | null {
  const reading = readText(path, context.source.read(path));
  if (reading.read === null) {
    gathering.refuse(reading.problem as LibraryProblem);
    return null;
  }
  if (role !== null && context.schemas !== null) {
    const problems = context.schemas.structural(reading.read.tree, role, { deepest: true });
    if (problems.length > 0) {
      gathering.refuse(offSchema(path, problems.slice(0, 8)));
      return null;
    }
  }
  const refusal = readRefusal(path, reading.read);
  if (refusal !== null) {
    gathering.refuse(refusal);
    return null;
  }
  return reading.read;
}

/**
 * `<path>: off the primitive-library-unit schema`, with the eight deepest lines under it.
 *
 * The cap is `problems[:8]`, and it is the tools': a unit off the schema in twenty places is
 * refused in eight lines. The lines the cap drops are dropped here too, so that the refusal's text
 * is the text the rejection suite matches; the schema stage answers them all to a caller that
 * wants them (§4.22's Ajv row is that caller).
 */
export function offSchema(path: string, problems: readonly StructuralProblem[]): LibraryProblem {
  return libraryProblem('schema', path, `${path}: off the primitive-library-unit schema`, {
    detail: problems.map((problem) => ({
      message: formatProblem(problem),
      path: problem.path,
      segments: problem.segments,
    })),
  });
}

// --- the model's own bases -------------------------------------------------

/** The bases a document resolves from, and the refusal a document that is not one carries. */
export interface BasesResult {
  readonly bases: readonly string[];
  readonly problem: LibraryProblem | null;
}

/**
 * `bases_of`: the bases a document resolves from — the caller's override when it names some, else
 * the document's own `primitive_libraries` entries, taken relative to the document's directory.
 */
export function basesOf(
  modelPath: string,
  model: PyValue,
  override?: readonly string[],
): BasesResult {
  const revision = get(model, 'schema');
  if (revision !== 'tensorspine/2.0') {
    return {
      bases: [],
      problem: libraryProblem(
        'revision',
        modelPath,
        `${modelPath}: expected tensorspine/2.0; convert supported legacy input with ` +
          'python3 tools/migrate.py INPUT -o OUTPUT',
      ),
    };
  }
  if (override !== undefined && override.length > 0) {
    return { bases: override.map((one) => normalise(one)), problem: null };
  }
  const here = dirname(modelPath);
  const declared = get(model, 'primitive_libraries');
  if (!Array.isArray(declared)) {
    return {
      bases: [],
      problem: libraryProblem(
        'base',
        modelPath,
        `${modelPath}: no primitive_libraries to resolve from`,
      ),
    };
  }
  const bases = (declared as readonly PyValue[]).map((entry) =>
    normalise(join(here, asText(get(entry, 'base')))),
  );
  return { bases, problem: null };
}

/**
 * `load_for`: the primitive library a document resolves from, its declared bases checked to exist
 * first — "a declared base that does not exist is a rejection (V1), not a fallback to some other
 * primitive_library".
 */
export function librariesFor(
  modelPath: string,
  model: PyValue,
  context: LibraryContext,
  override?: readonly string[],
): Library {
  const gathering = new Gathering();
  const resolved = basesOf(modelPath, model, override);
  if (resolved.problem !== null) {
    gathering.refuse(resolved.problem);
    return library([], gathering, new Map(), gathering.problems);
  }
  const missing = resolved.bases.filter((base) => !context.source.exists(base));
  if (missing.length > 0) {
    for (const base of missing) {
      gathering.refuse(
        libraryProblem(
          'base',
          modelPath,
          `${basename(modelPath)}: primitive library base '${base}' does not exist (V1)`,
          { code: 'V1' },
        ),
      );
    }
    return library([], gathering, new Map(), gathering.problems);
  }
  return loadLibrary(resolved.bases, context);
}

/** The reading `toPython` gives a tree, for a caller that already holds one. */
export function documentOf(tree: JsonValue): PyValue {
  return toPython(tree);
}
