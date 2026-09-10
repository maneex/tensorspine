/**
 * Template primitives: the document a unit pins, and the interface it presents.
 *
 * A template primitive carries `template` instead of the computational objects: "its meaning is
 * the family of graphs a template — a parameterized model definition — denotes, one per
 * assignment (§4.6)". Two things follow, and both are here.
 *
 * **The pin is checked at load.** `primitive_library._pinned_template` resolves
 * `<templates>/<name>/<version>.json`, requires it to be there, to be on the model schema, to
 * carry the pinned version and to declare the pinned model id. A base with a template primitive
 * and no `templates` location in its manifest is refused for that alone: the identity would name
 * a document nobody can find.
 *
 * **The interface is computed.** `validate.template_interface` turns the template into a primitive
 * definition a call site can be validated against: one argument per *external* quantity, with its
 * type, its domain and its declared default; the template's public inputs as input ports and its
 * public outputs as output ports. It lives in `validate.py` in the tools and here in the library
 * because the plan's §5.3 makes template interfaces part of what `loadLibrary` answers — the
 * primitive editor's header and a template instance's node both read it (§4.22, D9).
 *
 * The two literals the interface writes — the precision role `activation.hidden` and the domain
 * kind `inherit` — are `template_interface`'s own, quoted from the tools. They are not a
 * vocabulary the editor chose: a template exposes activations and inherits its outputs' domain
 * from expansion, and changing either would change what a call site is validated against.
 */
import { hasKey, isRecord, type PyRecord, type PyValue } from '../expr/value.js';
import { pyEqual } from '../expr/arithmetic.js';
import type { SchemaRegistry } from '../schema/registry.js';
import { formatProblem } from '../schema/registry.js';
import { get, has, members } from './access.js';
import { join } from './paths.js';
import { libraryProblem, type LibraryProblem } from './problems.js';
import { readRefusal, readText } from './read.js';
import { pyRepr, pyStr } from './repr.js';
import type { LibrarySource } from './source.js';

/** The precision role every port of a template interface carries (`validate.template_interface`). */
const TEMPLATE_PORT_ROLE = 'activation.hidden';

/** The template document a template primitive pins, once the load has checked it. */
export interface TemplatePin {
  /** The template's name, as the primitive pins it. */
  readonly name: string;
  /** The version pinned, which the document must carry (§4.6). */
  readonly version: string;
  /** Where the document is: `<templates>/<name>/<version>.json`. */
  readonly path: string;
  /** The document's text, for a caller that wants to open it as the editor holds documents. */
  readonly text: string;
  /** The document as the evaluators read one. */
  readonly document: PyValue;
}

/** The pin, or the refusal that keeps it from being one. */
export interface PinResult {
  readonly pin: TemplatePin | null;
  readonly problem: LibraryProblem | null;
}

/**
 * `_pinned_template`: the template file a template primitive pins, once it is known to exist, to
 * be a model document, and to carry the pinned version and id (§4.6).
 *
 * `where` is what the refusal names — the unit's own file, as the load's `origin` records it.
 */
export function pinnedTemplate(
  definition: PyValue,
  where: string,
  location: string,
  source: LibrarySource,
  schemas: SchemaRegistry | null,
): PinResult {
  const reference = get(definition, 'template');
  const pinnedName = get(reference, 'name');
  const pinnedVersion = get(reference, 'version');
  const identity = get(reference, 'id');
  const name = pyStr(pinnedName);
  const version = pyStr(pinnedVersion);
  const path = join(location, name, `${version}.json`);
  if (!source.isFile(path)) {
    return {
      pin: null,
      problem: libraryProblem(
        'template',
        where,
        `${where}: template '${name}' ${version} is not at ${path}`,
      ),
    };
  }
  const text = source.read(path);
  const reading = readText(path, text);
  if (reading.read === null) {
    return { pin: null, problem: reading.problem };
  }
  if (schemas !== null) {
    const problems = schemas.structural(reading.read.tree, 'model', { deepest: true }).slice(0, 8);
    if (problems.length > 0) {
      return {
        pin: null,
        problem: libraryProblem('schema', where, `${where}: template ${path} is off the model schema`, {
          detail: problems.map((problem) => ({
            message: formatProblem(problem),
            path: problem.path,
            segments: problem.segments,
          })),
        }),
      };
    }
  }
  const refusal = readRefusal(path, reading.read);
  if (refusal !== null) return { pin: null, problem: refusal };
  const document = reading.read.value;
  const carried = get(document, 'version');
  if (!pyEqual(carried, pinnedVersion)) {
    return {
      pin: null,
      problem: libraryProblem(
        'template',
        where,
        `${where}: pins template '${name}' at ${version}, but ${path} carries version ` +
          `${pyRepr(carried)}`,
      ),
    };
  }
  const declared = get(document, 'model');
  if (!pyEqual(declared, identity)) {
    return {
      pin: null,
      problem: libraryProblem(
        'template',
        where,
        `${where}: template '${name}' declares model id ${pyRepr(declared)}, the primitive ` +
          `says ${pyRepr(identity)}`,
      ),
    };
  }
  return { pin: { name, version, path, text, document }, problem: null };
}

/**
 * `_to_primitive_expression`: a template default, written over quantities, as a primitive
 * expression over arguments. `null` when it reads something a caller cannot supply — a derived
 * quantity, a quantity the template does not declare, an operator over one of those.
 */
export function toPrimitiveExpression(expression: PyValue, template: PyValue): PyValue | null {
  if (!isRecord(expression)) return null;
  if (hasKey(expression, 'literal')) return expression;
  if (hasKey(expression, 'quantity')) {
    const name = get(expression, 'quantity');
    const quantities = get(template, 'quantities');
    if (typeof name !== 'string' || !has(quantities, name)) return null;
    const source = get(get(quantities, name), 'source');
    const kind = get(source, 'kind');
    if (kind === 'external') return { argument: name };
    if (kind === 'literal') return { literal: get(source, 'value') };
    return null;
  }
  if (hasKey(expression, 'op')) {
    const given = get(expression, 'args');
    if (!Array.isArray(given)) return null;
    const args: PyValue[] = [];
    for (const one of given as readonly PyValue[]) {
      const converted = toPrimitiveExpression(one, template);
      if (converted === null) return null;
      args.push(converted);
    }
    return { op: get(expression, 'op'), args };
  }
  return null;
}

/**
 * `validate.template_interface`: the primitive a template primitive presents to a caller.
 *
 * One argument per external quantity of the template, with its type, domain and declared default;
 * the template's public inputs as input ports with their kinds, its public outputs as output ports
 * whose domains expansion resolves (§4.6). Every other object of a primitive definition is empty:
 * a template owns no parameter, no constant, no state and no partition option of its own — what
 * its sites own is theirs.
 */
export function templateInterface(definition: PyValue, template: PyValue): PyRecord {
  const args: Record<string, PyValue> = {};
  for (const [name, quantity] of members(get(template, 'quantities'))) {
    const source = get(quantity, 'source');
    if (get(source, 'kind') !== 'external') continue;
    const declaration: Record<string, PyValue> = {
      type: get(quantity, 'type'),
      required: true,
      structural: true,
    };
    if (has(quantity, 'domain')) declaration['domain'] = get(quantity, 'domain');
    if (has(source, 'default')) {
      const fallback = toPrimitiveExpression(get(source, 'default'), template);
      if (fallback !== null) {
        declaration['default'] = fallback;
        // Python assigns into the dictionary, so `required` keeps the place it was written in.
        declaration['required'] = false;
      }
    }
    args[name] = declaration;
  }
  const interfaces = get(template, 'interfaces');
  const inputs: Record<string, PyValue> = {};
  for (const [name, declared] of members(get(interfaces, 'inputs'))) {
    inputs[name] = {
      role: TEMPLATE_PORT_ROLE,
      domain: { kind: get(declared, 'kind'), from: { self: true } },
    };
  }
  const outputs: Record<string, PyValue> = {};
  for (const [name] of members(get(interfaces, 'outputs'))) {
    outputs[name] = {
      role: TEMPLATE_PORT_ROLE,
      domain: { kind: 'inherit', from: { self: true } },
    };
  }
  return {
    version: get(definition, 'version'),
    arguments: args,
    ports: { inputs, outputs },
    parameters: {},
    constants: {},
    state_ports: {},
    partition_options: [],
  };
}
