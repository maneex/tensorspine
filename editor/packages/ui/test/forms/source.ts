import { parse, type JsonValue } from '@tensorspine/lang';

import { formContext, type FormContext } from '../../src/forms/walk.js';
import { presentation } from '../../src/presentation/load.js';
import { registry } from '../presentation/source.js';

export { readRepositoryFile, registry, repositoryRoot } from '../presentation/source.js';

/** The `$id` of each schema the forms are generated over. */
export const MODEL = 'https://tensorspine.dev/schema/2.0/model.json';
export const UNIT = 'https://tensorspine.dev/schema/2.0/primitive-library-unit.json';
export const DERIVED = 'https://tensorspine.dev/schema/2.1/derived.json';
export const DOCUMENTATION = 'https://tensorspine.dev/schema/2.0/documentation.json';

let once: FormContext | null = null;

/** One context for the whole suite: the schemas cost ~200 ms and the shapes memoise. */
export function context(): FormContext {
  once ??= formContext(registry(), presentation());
  return once;
}

/** A value as the store holds it: the lexeme-preserving tree, from its JSON text. */
export function tree(text: string): JsonValue {
  return parse(text);
}
