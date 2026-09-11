/**
 * Lint: the port of `tools/lint.py` (plan §5.3, the `lint.py` line of the `validate` row).
 *
 * > "Nothing here is a refusal. §8.1 makes explicit refusal the obligation of the primitive, so
 * > anything the primitive can judge is already handled by `--validate`. What is left over is
 * > outside the document's jurisdiction: the consistency of the repository around it, and the
 * > curation of an open primitive_library. A lint finding is an opinion a reasonable author may
 * > decline."
 *
 * Three rules and a report:
 *
 * - **uncalled primitives** — what the gathered bases carry that none of the linted documents
 *   calls, templates followed. A *curation* question about an open vocabulary, not dead code.
 * - **unreferenced vocabulary** — axes and precision roles no primitive cites, the `storage`
 *   space excepted, since D3's storage axis is cited by the derivation and never by a shape.
 * - **the validator's advisories** — the lines `validate.analyse` put aside without refusing, and
 *   the one report about a document it could not analyse at all because it is off the grammar.
 *
 * Two things about the answer are worth stating before anything reads it.
 *
 * It is **set-dependent**: "called by none of the N model(s) linted" names the size of the set,
 * and the set decides what is called, so the same document linted alone and linted with the
 * corpus produces different lines. Feature 0.1 recorded that for the oracle; it is a property of
 * the rule, and the editor's own Problems panel inherits it — a lint of the open document is not
 * a lint of the workspace.
 *
 * It runs **before** the grammar, in part: `uncalled_primitives` indexes `instances` and
 * `compositions` on every document given, so a document missing either raises out of `--lint`
 * before `model_advisories` gets the chance to report it as off the schema. The port reproduces
 * that rather than reordering the rules, and states it as a finding.
 *
 * Reading is the caller's, here as everywhere in this package: `lint.run` loads the primitive
 * library from the command line's bases or the first document's own and opens every file it was
 * given, and what is ported is the part that decides — the caller hands it the gathered library
 * (`basesOf`, `loadLibrary`) and each document's text.
 *
 * **One divergence, stated.** `model_advisories` crosses `validate.structural`, which runs three
 * stages: the JSON layer (V12), then `primitive_library.read_json` — the legacy field layout and
 * an unsupported `schema` revision — and then the grammar. Feature 1.1's `structuralText` is the
 * first and the third; the middle one is not in it, and `derive/products.ts`'s own gate — which
 * the tools take through the same `validate.structural` — does not run it either. A document carrying a legacy revision is therefore reported off the schema with the
 * grammar's line (`schema: 'tensorspine/2.0' was expected`) where the tools report the loader's
 * (`<path>: unsupported revision 'tensorspine/1.0'; convert … (V12)`) — the same verdict, another
 * reason. No document, rejection case or base of the repository reaches it; closing it needs a
 * rendering `StructuralProblem` has not (the tools print that line *bare*) and a path
 * `structuralText` is not given, so it is a change set of its own over feature 1.1's module and
 * its parity suite. The unit suite pins the line the port answers, with the tools' own quoted
 * beside it as it was measured, so that the gap cannot go quiet.
 */
export {
  baseName,
  calledPrimitives,
  distinctFindings,
  formatLintFinding,
  lint,
  lintReport,
  modelAdvisories,
  uncalledPrimitives,
  unreferencedVocabulary,
  type LintDocument,
  type LintFinding,
  type LintOptions,
  type LintRule,
} from './lint.js';
