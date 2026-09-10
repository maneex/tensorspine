/**
 * The compatibility boundary: `tools/migrate.py`'s `legacy_layout`, and nothing more of it.
 *
 * `primitive_library.read_json` refuses a unit written in the earlier untagged field layout —
 * `contract` for `primitive`, `occurrence` for `instance`, `law` for `evolution`, `catalog` for
 * `primitive_library` — and tells the reader to convert it, rather than aliasing the old names
 * ("this module is the compatibility boundary; current readers never alias old keys"). The editor
 * loads whatever base a workspace holds, so it needs that refusal; it does not need the
 * conversion, which stays a Python command line.
 *
 * `vocabulary` classifies *structural* key names, and its two exclusions are what keep an
 * authored identity from being mistaken for one: a map keyed by names the author chose
 * (`instances`, `arguments`, `axes`, …) contributes nothing from its keys, and the value of an
 * opaque field (`arguments`, `assignment`, `indices`, …) is user data that is never walked at
 * all. The tables below are `migrate.py`'s own; they name no vocabulary of the five schemas — a
 * legacy name is by definition absent from them — so the audit of plan §1 (d) has nothing to hold
 * them to but this: they are the old spelling, frozen, and they change only when `migrate.py`
 * changes.
 */
import { isRecord, type PyValue } from '../expr/value.js';

/** `migrate.KEYS`: the legacy spelling of each renamed field. */
const LEGACY_KEYS: ReadonlySet<string> = new Set([
  'contract',
  'contracts',
  'occurrence',
  'occurrences',
  'law',
  'state_laws',
  'by_law',
  'cut',
  'cuts',
  'partitions',
  'per_contract',
]);

/** `set(migrate.KEYS.values())`: the current spelling of each renamed field. */
const CURRENT_KEYS: ReadonlySet<string> = new Set([
  'primitive',
  'primitives',
  'instance',
  'instances',
  'evolution',
  'state_evolutions',
  'by_evolution',
  'graph_split',
  'graph_splits',
  'partition_options',
  'per_primitive',
]);

/** `migrate.NAMED_MAPS`: fields whose member names are authored identities, not format names. */
const NAMED_MAPS: ReadonlySet<string> = new Set([
  'occurrences',
  'instances',
  'compositions',
  'quantities',
  'constants',
  'nodes',
  'contracts',
  'primitives',
  'axes',
  'precision',
  'inputs',
  'outputs',
  'state_ports',
  'parameter_slots',
  'constant_slots',
  'arguments',
  'indices',
  'assignment',
  'roles',
  'tolerance',
  'hook_map',
  'per_contract',
  'per_primitive',
]);

/** `migrate.OPAQUE`: fields whose values are user data and are never walked. */
const OPAQUE: ReadonlySet<string> = new Set([
  'arguments',
  'assignment',
  'indices',
  'hook_map',
  'tolerance',
  'delivery',
  'artifact',
]);

/** `migrate.vocabulary`: which of the two field layouts a document is written in, or neither. */
export function layoutVocabulary(value: PyValue, parent: string | null = null): Set<string> {
  const found = new Set<string>();
  if (parent !== null && OPAQUE.has(parent)) return found;
  if (Array.isArray(value)) {
    for (const item of value as readonly PyValue[]) {
      for (const one of layoutVocabulary(item)) found.add(one);
    }
    return found;
  }
  if (!isRecord(value)) return found;
  for (const key of Object.keys(value)) {
    if (parent === null || !NAMED_MAPS.has(parent)) {
      if (LEGACY_KEYS.has(key) || key === 'catalog') found.add('legacy');
      if (CURRENT_KEYS.has(key) || key === 'primitive_libraries' || key === 'primitive_library') {
        // D1's expansion provenance already used `instances` under `d1`, so that one name there
        // says nothing about the layout.
        if (!(key === 'instances' && parent === 'd1')) found.add('current');
      }
    }
    for (const one of layoutVocabulary(value[key] as PyValue, key)) found.add(one);
  }
  return found;
}

/** `migrate.legacy_layout`: whether the document is written in the earlier field layout. */
export function legacyLayout(document: PyValue): boolean {
  return layoutVocabulary(document).has('legacy');
}
