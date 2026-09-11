/**
 * `tools/lint.py`, function by function.
 *
 * Three rules, in the order `run` concatenates them — the primitives no linted document calls,
 * the axes and precision roles no primitive cites, and the advisories `validate.analyse` noticed
 * and did not refuse — then the printing, which is where the tools deduplicate.
 */
import { missingAssignment } from '../expr/model.js';
import { isRecord, toPython, type PyRecord, type PyValue } from '../expr/value.js';
import { parse } from '../json/parse.js';
import { demand, entries, has } from '../library/access.js';
import {
  identityKey,
  libraryUnits,
  templateOf,
  type Library,
  type LibrarySection,
} from '../library/load.js';
import { pyStr } from '../library/repr.js';
import { comparePythonStrings } from '../schema/repr.js';
import { formatProblem, type SchemaRegistry } from '../schema/registry.js';
import { analyse } from '../validate/index.js';
import { valueToken } from '../validate/graph.js';

/**
 * One document `--lint` was given: the path it was named by, and its bytes.
 *
 * The tools take paths and open the files; the core reads nothing (§5.3), so the caller — the
 * worker, a test, a Node command — hands it both. The path is not decoration: `model_advisories`
 * scopes a finding by `os.path.basename(path)`, so two documents of the same base name produce
 * findings the report deduplicates as one.
 */
export interface LintDocument {
  /** The path, as the tools were given it. */
  readonly path: string;
  /** The document's text, byte for byte as the file holds it. */
  readonly text: string;
}

/** What `--lint` needs beside the documents: the grammar, and the gathered primitive library. */
export interface LintOptions {
  /** The schemas: `model_advisories` crosses `validate.structural` before it analyses. */
  readonly schemas: SchemaRegistry;
  /**
   * The primitive library, gathered.
   *
   * `lint.run` loads it itself — from the command line's bases, else the first document's own —
   * which is the half of it that reads the filesystem. Resolving the bases (`basesOf`) and
   * gathering them (`loadLibrary`) is the caller's, exactly as it is for `analyse`.
   */
  readonly library: Library;
}

/** Which of `lint.py`'s four message shapes a finding is. */
export type LintRule =
  /** `uncalled_primitives`: the primitive library carries it, no linted document calls it. */
  | 'uncalled'
  /** `unreferenced_vocabulary`: an axis or a precision role no primitive cites. */
  | 'vocabulary'
  /** `model_advisories`, first branch: the document is off the grammar, so it is not analysed. */
  | 'off-schema'
  /** `model_advisories`, second branch: a line `validate.analyse` put in `advisories`. */
  | 'advisory';

/**
 * One lint finding: the scope and the message the report prints, with the pointer beside them.
 *
 * "A lint finding is an opinion a reasonable author may decline" — there is no severity and no
 * rule code, and `--lint` always exits 0. The `file` is plan §3's obligation ("the core emits
 * `{code, message, path, node?, rule?, file?}` natively — the message with the tools' wording for
 * parity, the pointer beside it"): the unit a curation question is about, or the document a
 * document-scoped finding is about.
 */
export interface LintFinding {
  readonly rule: LintRule;
  /** `primitive_library`, or the document's base name — the scope the report prints. */
  readonly scope: string;
  /** The tools' words, so that a report is comparable line for line. */
  readonly message: string;
  /** The unit file or the document the finding is about, when the reading gives one. */
  readonly file?: string;
}

/**
 * `os.path.basename`, which on the platform the tools run on is the text after the last `/`.
 *
 * The core has no path module and no platform: a workspace path is `/`-separated everywhere
 * (§5.2), and this is the one place a lint message shows a path at all.
 */
export function baseName(path: string): string {
  const cut = path.lastIndexOf('/');
  return cut === -1 ? path : path.slice(cut + 1);
}

/**
 * `_called_primitives`: the primitive names a document calls, templates followed.
 *
 * A name is added to `seen` before its template is walked, so a template that calls back into a
 * primitive the caller already named is not walked twice; the closure is therefore the same
 * whatever order the names come in, which is what makes this reproducible where the tools iterate
 * a Python set.
 *
 * The tools open the template's file again; here it is the document the loader already pinned —
 * `template_path` names the file, and the load read it to check its version and its id — which is
 * the same bytes without a second read. An unresolved template raises as `template_path` raises.
 */
export function calledPrimitives(
  document: PyValue,
  library: Library,
  seen: Set<string> = new Set<string>(),
): Set<string> {
  // `model['instances']` and `model['compositions']`: the tools index, so a document carrying
  // neither — which the grammar requires, and which this rule runs *before* the grammar is
  // checked — raises the `KeyError` here instead of being reported as off the schema.
  //
  // `names` is a Python *set*, which is why the walk below can be reproduced at all: its order is
  // hash-seeded there, and the closure it computes is the same whatever that order is. `known` is
  // that set's own deduplication; it decides nothing on its own, since `seen` skips a repeat one
  // line later, and it is kept because the tools' collection is a set and not a list.
  const names: PyValue[] = [];
  const known = new Set<string>();
  const called = (instance: PyValue): void => {
    const name = demand(demand(instance, 'primitive'), 'name');
    const token = valueToken(name);
    if (known.has(token)) return;
    known.add(token);
    names.push(name);
  };
  for (const [, instance] of entries(demand(document, 'instances'))) called(instance);
  for (const [, composition] of entries(demand(document, 'compositions'))) {
    for (const [, instance] of entries(demand(composition, 'instances'))) called(instance);
  }
  for (const name of names) {
    const token = valueToken(name);
    if (seen.has(token)) continue;
    seen.add(token);
    // `cat['primitives'].get(name)`: by name alone, the highest version — the reading D1 takes
    // too. A name that is not a string matches no key, as a Python dictionary of string keys
    // answers `None` for one.
    const definition = typeof name === 'string' ? library.primitives.get(name) : undefined;
    if (definition !== undefined && has(definition, 'template')) {
      calledPrimitives(templateOf(library, definition).document, library, seen);
    }
  }
  return seen;
}

/**
 * `uncalled_primitives`: what the primitive library carries that none of the linted documents
 * calls.
 *
 * "Not dead code: the primitive library is an open vocabulary and the corpus is a falsification
 * sample, not the definition. A curation question." The count in the message is the number of
 * documents *given*, not the number of distinct ones, so the line moves with the set linted —
 * which is why a parity comparison must lint the same set.
 */
export function uncalledPrimitives(
  documents: readonly LintDocument[],
  library: Library,
): LintFinding[] {
  const called = new Set<string>();
  for (const document of documents) {
    for (const token of calledPrimitives(loadedAsPlainJson(document.text), library)) {
      called.add(token);
    }
  }
  const findings: LintFinding[] = [];
  for (const [name, definition] of sortedByName(library.primitives)) {
    if (called.has(valueToken(name))) continue;
    const version = pyStr(demand(definition, 'version'));
    const file = library.byId.get(identityKey(name, version))?.file;
    findings.push({
      rule: 'uncalled',
      scope: 'primitive_library',
      message:
        `primitive '${name}' ${version} is in the primitive library, ` +
        `called by none of the ${documents.length} model(s) linted`,
      ...(file === undefined ? {} : { file }),
    });
  }
  return findings;
}

/**
 * `unreferenced_vocabulary`: the axes and precision roles no primitive cites.
 *
 * "Cited" is a string *equal* to the name, anywhere inside a primitive's definition: `_strings`
 * walks the values of the map — never the member names — and yields every string whole, so a
 * shape's `axis`, an effect's port name and a `note` whose entire text is the name all count,
 * while the same name inside a sentence does not. And the map walked is `cat['primitives']`,
 * which carries **one definition per name, the highest version**: a name cited only by an older
 * version of a primitive reads as uncited.
 *
 * A `storage` axis is skipped: "cited by the derivation (D3's storage axis, §3.4), never by a
 * primitive shape". It is the one space with that exemption, and the audit of plan §1 (d) is what
 * requires a space the grammar gains to reach a decision instead of falling through.
 */
export function unreferencedVocabulary(library: Library): LintFinding[] {
  const cited = new Set<string>();
  for (const [, definition] of library.primitives) collectStrings(definition, cited);
  const files = unitFiles(library);
  const finding = (section: LibrarySection, name: string, message: string): LintFinding => {
    const file = files.get(`${section} ${name}`);
    return {
      rule: 'vocabulary',
      scope: 'primitive_library',
      message,
      ...(file === undefined ? {} : { file }),
    };
  };
  const findings: LintFinding[] = [];
  for (const [name, definition] of sortedByName(library.axes)) {
    if (demand(definition, 'space') === 'storage') continue;
    if (cited.has(name)) continue;
    findings.push(finding('axes', name, `axis '${name}' is cited by no primitive`));
  }
  for (const [name] of sortedByName(library.precision)) {
    if (cited.has(name)) continue;
    findings.push(
      finding('precision', name, `precision role '${name}' is cited by no primitive`),
    );
  }
  return findings;
}

/**
 * `model_advisories`: what the validator noticed and does not refuse.
 *
 * "The analysis assumes the grammar, so a document off the schema is reported as such and not
 * analysed: `--validate` is where its refusal belongs." A document that still needs an assignment
 * is passed over in silence — `--lint` takes no `--assign`, so a template is never analysed here
 * — and everything else is analysed and its `advisories` reported, whether or not the analysis
 * also refused the document.
 */
export function modelAdvisories(
  documents: readonly LintDocument[],
  options: LintOptions,
): LintFinding[] {
  const findings: LintFinding[] = [];
  for (const document of documents) {
    const scope = baseName(document.path);
    const problems = options.schemas.structuralText(document.text, 'model');
    const first = problems[0];
    if (first !== undefined) {
      findings.push({
        rule: 'off-schema',
        scope,
        message: `off the schema, not analysed (--validate refuses it): ${formatProblem(first)}`,
        file: document.path,
      });
      continue;
    }
    const read = toPython(parse(document.text)) as PyRecord;
    if (missingAssignment(read).length > 0) continue;
    for (const line of analyse(read, options.library).advisories) {
      findings.push({ rule: 'advisory', scope, message: line, file: document.path });
    }
  }
  return findings;
}

/**
 * `lint.run`'s findings, in its order: every rule over the given documents.
 *
 * Nothing here is a refusal, and the list is the three rules concatenated — undeduplicated, as
 * the tools build it. {@link lintReport} is what removes the repeats, because that is where the
 * tools remove them.
 */
export function lint(documents: readonly LintDocument[], options: LintOptions): LintFinding[] {
  return [
    ...uncalledPrimitives(documents, options.library),
    ...unreferencedVocabulary(options.library),
    ...modelAdvisories(documents, options),
  ];
}

/** `run`'s `seen` filter: one finding per distinct `(scope, message)`, in the order met. */
export function distinctFindings(findings: readonly LintFinding[]): LintFinding[] {
  const seen = new Set<string>();
  const kept: LintFinding[] = [];
  for (const finding of findings) {
    const key = JSON.stringify([finding.scope, finding.message]);
    if (seen.has(key)) continue;
    seen.add(key);
    kept.push(finding);
  }
  return kept;
}

/** One printed finding: `  W  <scope>: <message>`. */
export function formatLintFinding(finding: LintFinding): string {
  return `  W  ${finding.scope}: ${finding.message}`;
}

/**
 * What `lint.run` prints, line for line — the findings, then the count or "nothing to report".
 *
 * The line `--lint` prints *above* this block (`lint  N document(s)`) is the command line's, not
 * `lint.py`'s, so it is not here: a caller that reproduces the command writes it itself.
 */
export function lintReport(findings: readonly LintFinding[]): string[] {
  const kept = distinctFindings(findings);
  if (kept.length === 0) return ['  nothing to report'];
  return [
    ...kept.map((finding) => formatLintFinding(finding)),
    `  ${kept.length} advisory finding(s) — nothing blocking`,
  ];
}

/**
 * `json.load(open(path))`: the reading `_called_primitives` takes.
 *
 * Plain, so a duplicate member name resolves to the last value rather than being refused — which
 * is not the reading `model_advisories` takes two rules later (`validate.structural` refuses it
 * as V12), and the difference is the tools' own. A text that is not JSON at all raises here, as
 * it does there: `_called_primitives` catches `OSError` and nothing else.
 */
function loadedAsPlainJson(text: string): PyValue {
  return toPython(parse(text, { duplicates: 'last' }));
}

/** `_strings`: every string inside a value, the member names of a record left out. */
function collectStrings(value: PyValue, into: Set<string>): void {
  if (isRecord(value)) {
    for (const [, one] of entries(value)) collectStrings(one, into);
    return;
  }
  if (Array.isArray(value)) {
    for (const one of value as readonly PyValue[]) collectStrings(one, into);
    return;
  }
  if (typeof value === 'string') into.add(value);
}

/** `sorted(mapping.items())` on a mapping whose keys are the names: Python's code-point order. */
function sortedByName(mapping: ReadonlyMap<string, PyValue>): [string, PyValue][] {
  return [...mapping].sort((left, right) => comparePythonStrings(left[0], right[0]));
}

/**
 * The unit file of every axis and precision role of the gathered bases, by section and name.
 *
 * The section is part of the key because an axis and a precision role may carry the same name —
 * the reference base has no such pair, the `lint` suite's own base does — so passing the
 * primitives over is a narrowing of the walk and not a rule: their entries could not be read back
 * under either of the two keys this map is asked for.
 */
function unitFiles(library: Library): Map<string, string> {
  const files = new Map<string, string>();
  for (const unit of libraryUnits(library)) {
    if (unit.section === 'primitives') continue;
    const key = `${unit.section} ${unit.name}`;
    if (!files.has(key)) files.set(key, unit.file);
  }
  return files;
}
