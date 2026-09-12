/**
 * Anchors: `<$id>#<JSON pointer>`, the way a place of a schema is named outside the schema.
 *
 * The plan's §1 binds presentation to *schema anchors* — "keyed by JSON pointers into the
 * schemas (`…/model.json#/$defs/instance_definition`)" — and the store already answers in the
 * same shape (`Place.anchor`, feature 2.1). So an anchor is one string with two parts: the `$id`
 * the registry indexes a schema under, and an RFC 6901 pointer into that schema's document.
 *
 * Resolving one is the registry's business and not the interface's, which is why it lives here:
 * the presentation audit of §1 (a) has to say whether a key *resolves*, and a rule with two
 * implementations is the thing D3 forbids. `vocabulary.ts` writes the same anchors from the
 * other side, and reads them back through this module.
 *
 * Two conventions, both taken from the schemas as they are written:
 *
 * - the root of a schema is `<$id>#`, and `<$id>#/` would be the member named `''` — which is
 *   RFC 6901 and not a typo to be forgiven;
 * - a percent escape is *not* decoded. A pointer inside a URI fragment may be percent-encoded,
 *   but the schemas write none and the registry's own `$ref` resolution is what handles a
 *   reference; an anchor is written by hand in `presentation.json`, so what it says is what it
 *   names.
 */
import type { SchemaRegistry } from './registry.js';
import { isSchemaObject, type SchemaNode, type SchemaObject } from './types.js';

/** An anchor split into the schema it names and the pointer inside it. */
export interface Anchor {
  /** The `$id` the registry indexes the schema under. */
  readonly schema: string;
  /** The JSON pointer, `''` at the root and `/$defs/dtype` inside. */
  readonly pointer: string;
}

/** A place of a schema, as {@link resolveAnchor} answers it. */
export interface Resolution {
  readonly schema: string;
  readonly pointer: string;
  /** `<$id>#<pointer>` — the anchor, rebuilt, so a caller can key by what was resolved. */
  readonly anchor: string;
  /** The node itself, as the file writes it. */
  readonly node: SchemaNode;
}

/** A JSON pointer segment, escaped as RFC 6901 escapes it — `json/pointer.ts`'s, re-exported. */
export { pointerSegment } from '../json/pointer.js';

/** `<$id>#<pointer>` from its two parts. */
export function anchorOf(schema: string, pointer: string): string {
  return `${schema}#${pointer}`;
}

/**
 * An anchor split at its `#`, or `null` when it carries none.
 *
 * The `$id`s of this repository's schemas carry no fragment of their own, so the first `#` is
 * the separator and everything after it is the pointer.
 */
export function parseAnchor(anchor: string): Anchor | null {
  const hash = anchor.indexOf('#');
  if (hash < 0) return null;
  const pointer = anchor.slice(hash + 1);
  if (pointer !== '' && !pointer.startsWith('/')) return null;
  return { schema: anchor.slice(0, hash), pointer };
}

/** The steps of a JSON pointer, unescaped. `'#'` and `''` are the root and have none. */
export function pointerSteps(pointer: string): string[] {
  const body = pointer.startsWith('#') ? pointer.slice(1) : pointer;
  if (body === '') return [];
  return body
    .slice(1)
    .split('/')
    .map((step) => step.replace(/~1/g, '/').replace(/~0/g, '~'));
}

/** The node a JSON pointer names inside a schema document, or `null` when it names nothing. */
export function nodeAtPointer(document: SchemaObject, pointer: string): SchemaNode | null {
  if (pointer === '' || pointer === '#') return document;
  const body = pointer.startsWith('#') ? pointer.slice(1) : pointer;
  if (!body.startsWith('/')) return null;
  let node: unknown = document;
  for (const step of pointerSteps(body)) {
    if (node === null || typeof node !== 'object') return null;
    node = Array.isArray(node)
      ? node[Number(step)]
      : (node as Record<string, unknown>)[step];
    if (node === undefined) return null;
  }
  return node as SchemaNode;
}

/** The place an anchor names in the registry, or `null` when nothing of the registry has it. */
export function resolveAnchor(registry: SchemaRegistry, anchor: string): Resolution | null {
  const parsed = parseAnchor(anchor);
  if (parsed === null) return null;
  const schema = registry.byId(parsed.schema);
  if (schema === undefined) return null;
  const node = nodeAtPointer(schema.document, parsed.pointer);
  if (node === null) return null;
  return {
    schema: schema.id,
    pointer: parsed.pointer,
    anchor: anchorOf(schema.id, parsed.pointer),
    node,
  };
}

/**
 * The place an anchor names, with a chain of bare `$ref`s followed to the node that says
 * something.
 *
 * A schema writes `{"$ref": "#/$defs/quantity_map"}` where it means the map, and a reader that
 * stopped at the reference would find neither its `properties` nor its `enum`. Only a *bare*
 * reference is followed — a node that carries a `$ref` beside other keywords says something of
 * its own, and 2020-12 applies both.
 */
export function followAnchor(
  registry: SchemaRegistry,
  anchor: string,
  limit = 8,
): Resolution | null {
  let current = resolveAnchor(registry, anchor);
  for (let depth = 0; depth < limit; depth += 1) {
    if (current === null || !isSchemaObject(current.node)) return current;
    const ref = current.node['$ref'];
    const bare = typeof ref === 'string' && Object.keys(current.node).every((key) => key === '$ref');
    if (!bare) return current;
    const hash = ref.indexOf('#');
    const uri = hash < 0 ? ref : ref.slice(0, hash);
    const inside = hash < 0 ? '' : ref.slice(hash + 1);
    const next = resolveAnchor(registry, anchorOf(uri === '' ? current.schema : uri, inside));
    if (next === null) return current;
    current = next;
  }
  return current;
}
