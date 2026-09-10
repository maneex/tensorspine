/**
 * A document to normalise, written as the text a workspace holds.
 *
 * `loadModel` reads a text, so the tests state their input as one: what is being checked is the
 * expansion of §5.2 rule 7, and writing the input as JSON keeps the integer/float distinction and
 * the member order of the document visible where they belong — in the document — rather than in a
 * pile of `0n`s. Nothing here is validated: `normalise` runs before every other rule, so a
 * document only has to be shaped enough for the members it reads.
 */

/** What the composition of {@link document} holds, all of it optional but its scoped bindings. */
export interface Composition {
  /** The `indices` map, as JSON text; one index `i` over `[0, 2)` by default. */
  indices?: string;
  /** The site names; `a` and `b` by default. */
  sites?: readonly string[];
  /** The `bindings` member, as JSON text; omitted entirely when absent. */
  bindings?: string;
}

const RANGE = '{"start": {"literal": 0}, "stop": {"literal": 2}, "step": {"literal": 1}}';

/** One site, shaped as the grammar shapes an instance. */
function site(name: string): string {
  return `"${name}": {"primitive": {"name": "demo.unit", "version": "1.0.0"}, ` +
    `"arguments": {}, "families": ["demo"]}`;
}

/**
 * A document with one composition `c`, and the top-level bindings a caller gives.
 *
 * `top` is the four maps of `bindings`, as JSON text, so that a test can put a rule where a scoped
 * one would collide with it; the four are all present, as the grammar requires them to be.
 */
export function document(composition: Composition, top = ''): string {
  const indices = composition.indices ?? `{"i": ${RANGE}}`;
  const sites = (composition.sites ?? ['a', 'b']).map(site).join(', ');
  const scoped = composition.bindings === undefined ? '' : `,\n      "bindings": ${composition.bindings}`;
  const bindings = top === ''
    ? '{"values": {}, "parameters": {}, "constants": {}, "states": {}}'
    : top;
  return `{
  "schema": "tensorspine/2.0",
  "model": "demo/1.0",
  "primitive_libraries": [{"base": "../primitive-library/"}],
  "quantities": {},
  "constants": {},
  "instances": {${site('root')}},
  "compositions": {
    "c": {
      "indices": ${indices},
      "families": ["demo"],
      "instances": {${sites}}${scoped}
    }
  },
  "bindings": ${bindings},
  "interfaces": {"inputs": {}, "outputs": {}}
}
`;
}
