/**
 * `validateUnit`: the loader's checks on **one** unit, against the bases already gathered.
 *
 * The tools have no such call — a unit is checked on the way into a base and nowhere else — but
 * the editor needs one, and D15 says what it covers: "the unit schema, the path/identity
 * agreement, every cross-reference (axes, roles, ports named by effects and transforms, argument
 * paths in conditions, domain bounds on required-or-defaulted arguments, an optional argument
 * compared only under a `present` test, a `sequence.position` payload axis only under `fixed`, a
 * template's file, name, version and id)". §4.22 says when: on every edit, as the unit is typed,
 * beside Ajv's own row.
 *
 * It is not a second implementation. Every check here is the function `loadLibrary` calls at the
 * same point of the same staging — the schema stage, `read_json`'s refusals, `_units`' four
 * agreements, `primitive_references`, the precision default, `_pinned_template` — so parity on the
 * rejection suite is parity for this call too, and a rule cannot move in one without moving in the
 * other.
 *
 * What it does **not** cover is the one check that is not about a unit: two bases carrying one
 * identity with different contents (V1) is a fact about a *set*, and `loadLibrary` is where it is
 * found. A unit that would conflict is otherwise perfectly well-formed.
 *
 * Staging is the tools': meaning assumes grammar. A unit off the unit schema is refused for that
 * and nothing else is said about it, exactly as `_units` says nothing else — which is also why the
 * editor shows Ajv's rows beside these (§4.22) rather than expecting these to carry them.
 */
import type { JsonValue } from '../json/tree.js';
import { toPython, type PyValue } from '../expr/value.js';
import { pyEqual } from '../expr/arithmetic.js';
import { asText, get, has } from './access.js';
import { join, normalise, relative } from './paths.js';
import { libraryProblem, type LibraryProblem } from './problems.js';
import { primitiveReferences } from './references.js';
import { readRefusal, readText, type ReadText } from './read.js';
import { pyRepr } from './repr.js';
import { pinnedTemplate } from './template.js';
import {
  identityProblem,
  offSchema,
  SECTIONS,
  type Library,
  type LibraryContext,
  type LibrarySection,
} from './load.js';

/** Where a unit sits: the base it belongs to, and its own file. */
export interface UnitLocation {
  /** The base, as the load was given it. */
  readonly base: string;
  /** The unit's file, as every refusal about it names it. */
  readonly path: string;
}

/** What a unit's path says it is. */
export interface UnitPlace {
  /** `null` for the base manifest, which lives at the root and has no section. */
  readonly section: LibrarySection | null;
  /** The `kind` the place requires: `primitive`, `axis`, `precision_role` or `base`. */
  readonly kind: string;
  /** The identity the path spells out; `''` for the manifest. */
  readonly name: string;
  /** A primitive's version, from its file name; `null` otherwise. */
  readonly version: string | null;
}

/**
 * What a unit's path says it is, or `null` when the path names no place of the base.
 *
 * `_units` cannot reach that state — it globs the three section roots — so there is no wording to
 * reproduce; the port states the refusal in its own voice rather than guessing a section, since
 * guessing is what would let a misplaced file be read as an identity it does not have (§8.2).
 */
export function placeOf(where: UnitLocation): UnitPlace | null {
  const base = normalise(where.base);
  const path = normalise(where.path);
  if (path === join(base, 'primitive-library.json')) {
    return { section: null, kind: 'base', name: '', version: null };
  }
  for (const [section, kind] of SECTIONS) {
    const root = join(base, section);
    if (path !== root && !path.startsWith(`${root}/`)) continue;
    const parts = relative(path, root).replace(/\.json$/, '').split('/');
    if (section === 'primitives') {
      return {
        section,
        kind,
        name: parts.slice(0, -1).join('.'),
        version: parts[parts.length - 1] as string,
      };
    }
    return { section, kind, name: parts.join('.'), version: null };
  }
  return null;
}

/**
 * The loader's verdict on one unit, as a list of problems: empty when the unit loads.
 *
 * `unit` is the ordered tree the editor holds (D1). The JSON layer has already run — the store's
 * own parse refused a duplicate member name — so this begins where `_units` begins, at the schema
 * stage; {@link validateUnitText} is the entry for a caller that has only the bytes.
 */
export function validateUnit(
  unit: JsonValue,
  where: UnitLocation,
  library: Library,
  context: LibraryContext,
): LibraryProblem[] {
  if (context.schemas !== null) {
    const problems = context.schemas.structural(unit, 'primitive-library-unit', { deepest: true });
    if (problems.length > 0) return [offSchema(where.path, problems.slice(0, 8))];
  }
  return judge({ tree: unit, value: toPython(unit), duplicate: null }, where, library, context);
}

/** {@link validateUnit} from a unit's bytes: the JSON layer, then everything above. */
export function validateUnitText(
  text: string,
  where: UnitLocation,
  library: Library,
  context: LibraryContext,
): LibraryProblem[] {
  const reading = readText(where.path, text);
  if (reading.read === null) return [reading.problem as LibraryProblem];
  if (context.schemas !== null) {
    const problems = context.schemas.structural(reading.read.tree, 'primitive-library-unit', {
      deepest: true,
    });
    if (problems.length > 0) return [offSchema(where.path, problems.slice(0, 8))];
  }
  return judge(reading.read, where, library, context);
}

/** Everything after the schema stage: `read_json`'s refusals, the place, the identity, the kind. */
function judge(
  read: ReadText,
  where: UnitLocation,
  library: Library,
  context: LibraryContext,
): LibraryProblem[] {
  const refusal = readRefusal(where.path, read);
  if (refusal !== null) return [refusal];

  const place = placeOf(where);
  if (place === null) {
    return [
      libraryProblem(
        'identity',
        where.path,
        `${where.path}: not under primitives/, axes/ or precision/ of ${where.base}`,
      ),
    ];
  }
  if (place.section === null) return manifestProblems(where.path, read.value);

  const identity = identityProblem(where.path, read.value, place.kind, place.name, place.version);
  if (identity !== null) return [identity];

  const definition = definitionOf(read.value);
  if (place.section === 'precision') return precisionProblems(where.path, definition);
  if (place.section === 'axes') return [];
  return primitiveProblems(where, definition, library, context);
}

/** `_manifest`'s one check beyond the schema: the file declares the base kind. */
function manifestProblems(path: string, unit: PyValue): LibraryProblem[] {
  const kind = get(unit, 'kind');
  if (pyEqual(kind, 'base')) return [];
  return [libraryProblem('identity', path, `${path}: kind ${pyRepr(kind)}, expected 'base'`)];
}

/** A precision role's default must be one of the dtypes it admits. */
function precisionProblems(path: string, definition: PyValue): LibraryProblem[] {
  const fallback = get(definition, 'default');
  const admissible = get(definition, 'admissible');
  const admits =
    Array.isArray(admissible) &&
    (admissible as readonly PyValue[]).some((one) => pyEqual(one, fallback));
  if (admits) return [];
  return [
    libraryProblem(
      'precision',
      path,
      `${path}: default '${asText(fallback)}' is not in the admissible set ${pyRepr(admissible)}`,
    ),
  ];
}

/** A primitive's cross-references, and then — for a template primitive — its pin. */
function primitiveProblems(
  where: UnitLocation,
  definition: PyValue,
  library: Library,
  context: LibraryContext,
): LibraryProblem[] {
  const problems = primitiveReferences(definition, library);
  if (problems.length > 0) {
    return [
      libraryProblem('reference', where.path, `${where.path}: unresolved reference(s)`, {
        detail: problems.map((one) => ({
          message: one.message,
          path: `/definition${one.path}`,
          segments: ['definition', ...one.segments],
        })),
      }),
    ];
  }
  if (!has(definition, 'template')) return [];
  const location = templatesLocation(where.base, library, context);
  if (location === null) {
    return [
      libraryProblem(
        'template',
        where.path,
        `${where.path}: a template primitive, but its base declares no \`templates\` location ` +
          '(§4.6)',
      ),
    ];
  }
  const pinned = pinnedTemplate(definition, where.path, location, context.source, context.schemas);
  return pinned.problem === null ? [] : [pinned.problem];
}

/**
 * Where a base's template documents live.
 *
 * The gathered library already knows, for a base it loaded. A base being written in the editor is
 * not in it yet, so the manifest is read again — and its own refusals are not repeated here: a
 * broken manifest is the manifest's row, not this unit's.
 */
function templatesLocation(
  base: string,
  library: Library,
  context: LibraryContext,
): string | null {
  if (context.modelsBase !== undefined && context.modelsBase !== null) return context.modelsBase;
  const known = library.bases.find((one) => normalise(one.path) === normalise(base));
  if (known !== undefined) return known.templates;
  const path = join(base, 'primitive-library.json');
  if (!context.source.isFile(path)) return null;
  const reading = readText(path, context.source.read(path));
  if (reading.read === null) return null;
  const declared = get(get(reading.read.value, 'definition'), 'templates');
  if (typeof declared !== 'string') return null;
  return normalise(join(base, declared));
}

/** `unit['definition']`, or `null` for a unit that carries none. */
function definitionOf(unit: PyValue): PyValue {
  return get(unit, 'definition');
}
