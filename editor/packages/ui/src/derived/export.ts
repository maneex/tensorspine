/**
 * `File ▸ Export Derived Document…` — plan §4.4 and §4.18.
 *
 * > **Export Derived Document…** writes the JSON, validated by the core against the derived schema
 * > before writing, as the tools do.
 *
 * Three facts about it, each with its reason.
 *
 * **The file is `<model>.derived.json`**, which §4.4 states and which is the name the repository's
 * own corpus writes (`data/models/*.derived.json`). It goes through `Workspace.write`, so a folder
 * the browser can write takes it in place and a read-only snapshot hands it to the user as a
 * download — one path, and no second reading of what a deployment can do (feature 2.4).
 *
 * **The check is the core's own self-check, run again here.** `derive` already refuses a document
 * it could not validate (`DerivedSchemaError`, feature 1.8e), so a derived document that reached
 * the panel has passed once; running `structural(tree, 'derived')` on the page's registry before
 * writing is what §4.18 asks for and what catches the one case the first check cannot — a
 * workspace that carries its own `schemas/` (plan §1) whose derived schema is not the build's.
 *
 * **The bytes are the core's serializer, and they are not `--derive`'s.** `--derive` writes
 * `indent=1`, the one output of the repository that does not write `indent=2` (feature 1.8e), so
 * an exported document is the *document* the tools write and not their layout. `serialize` ends
 * the file with a newline as it does for a model document (D12); the parity contract is a deep
 * equality, which is what the feature's own browser test asserts.
 */
import {
  formatProblems,
  serialize,
  toJsonValue,
  type PyValue,
  type SchemaRegistry,
} from '@tensorspine/lang';

/** The role the registry indexes the derived schema under — its file's own name. */
export const DERIVED_ROLE = 'derived';

/** What a derived document is written to, beside the document it is derived from. */
export function derivedPathOf(path: string): string {
  return `${path.replace(/\.json$/, '')}.derived.json`;
}

/** What an export answers: the bytes, and what the schema said about them. */
export interface DerivedExport {
  readonly text: string;
  /** The derived schema's refusals, in the core's own words; empty when it conforms. */
  readonly problems: readonly string[];
}

/**
 * A derived document as bytes, checked against the derived schema first.
 *
 * The refusals are returned rather than raised: §4.4's own rule for a Save is that the file is the
 * user's ("saving an off-schema document is allowed with a confirmation"), and the caller is what
 * decides whether a refusal stops the write or lands in the log beside it.
 */
export function derivedBytes(derived: PyValue, registry: SchemaRegistry | null): DerivedExport {
  const tree = toJsonValue(derived);
  const text = serialize(tree);
  if (registry === null) return { text, problems: [] };
  if (registry.conforms(tree, DERIVED_ROLE)) return { text, problems: [] };
  return { text, problems: formatProblems(registry.structural(tree, DERIVED_ROLE)) };
}
