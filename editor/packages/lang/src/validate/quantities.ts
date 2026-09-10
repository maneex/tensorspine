/**
 * Quantities and assignments: the port of `check_quantities`, `check_assignment` and
 * `variable_quantities` of `tools/validate.py`.
 *
 * A document's quantities are the first thing the semantic stage decides, because everything else
 * is written over them: `analyse` resolves them, refuses what does not hold here, and evaluates
 * every range, guard, argument and location in the map that comes out. Four rules meet in one
 * screen of code:
 *
 * - **V3** — every quantity conforms to its declared type and domain, "derived ones included",
 *   and a *variable* quantity declares a domain. Which quantities are variable is not declared:
 *   §2.1 derives it from the sources — external, or derived from a variable one, transitively —
 *   because "declaring that classification would copy the rule that derives it (§4.4)".
 * - **V1** — a derivation reads only declared quantities.
 * - **V10** — a derivation resolves: a cycle, or a reference with no value, is a refusal, never a
 *   guess (I7).
 * - **V11** — a literal that declares how it follows from the others agrees with it (I8): "a
 *   transcription that disagrees with the structure is refused".
 *
 * **The assignment (§4.6).** A parameterized document denotes one graph per admissible
 * assignment, and "admissibility must be decidable at the call site", so an assignment is checked
 * against the declared types and domains of the external quantities exactly as a primitive's
 * arguments are — the same {@link checkType} and {@link checkDomain}, under a different heading.
 * What an assignment leaves unset is not an error: the document is a family, and reading it
 * requires one, which is what {@link assignmentNeeded} reports and what `--validate`, `--d1` and
 * `--derive` print as a *skip*, never a failure ("the skip is printed, never silent (I7)").
 *
 * **The wording, exactly.** `_check_type` and `_check_domain` say `argument '…'` because the
 * argument side is their other caller; `check_quantities` rewrites the first occurrence of that
 * word in every line it returns, and `check_assignment` does not — an assignment *is* an argument
 * list at a call site. Both are the parity contract (`tests/rejections/models.json` matches
 * `quantity 'head_dim' = 96 disagrees with its derivation`), so the rewrite is reproduced where
 * the tools do it: over the whole list, at the end.
 *
 * **What each function is handed.** `analyse` hands `check_quantities` the *normalised* document
 * (`model.load`) and the map `resolve_quantities` answers; `run`, `--d1`, `--derive` and `--lint`
 * hand `check_assignment` and `missing_assignment` the document as read. The two readings differ
 * only in `bindings` and `compositions` (feature 1.4), which no rule of this module looks at, and
 * the parity suite pins that they agree here.
 */
import { pyEqual } from '../expr/arithmetic.js';
import {
  externalNames,
  missingAssignment,
  modelValue,
  quantityReferences,
  type Quantities,
} from '../expr/model.js';
import { member, UNRESOLVED, type PyRecord, type PyValue } from '../expr/value.js';
import { comparePythonStrings } from '../schema/index.js';
import { demand, entries, has, optional } from '../library/access.js';
import { pyRepr } from '../library/repr.js';

import { checkDomain, checkType } from './conformance.js';
import { semanticProblem, withMessage, type SemanticProblem } from './problems.js';

/**
 * The quantities whose value depends on the assignment (§2.1): the external ones, and the derived
 * ones that read one, transitively.
 *
 * A fixpoint, because a derivation may read a derivation: the corpus has none — its documents
 * declare no variable quantity at all, every one of them being a model constant — and the
 * template's eight are external, so the transitive step is exercised by the editor's own cases
 * and by the one rejection fixture (`v3-derived-variable-without-domain`).
 */
export function variableQuantities(model: PyRecord): Set<string> {
  const quantities = demand(model, 'quantities');
  const variable = new Set<string>();
  for (const [name, declaration] of entries(quantities)) {
    if (demand(demand(declaration, 'source'), 'kind') === 'external') variable.add(name);
  }
  let changed = true;
  while (changed) {
    changed = false;
    for (const [name, declaration] of entries(quantities)) {
      if (variable.has(name)) continue;
      const source = demand(declaration, 'source');
      if (demand(source, 'kind') !== 'derived') continue;
      const reads = quantityReferences(demand(source, 'expression'));
      for (const one of reads) {
        if (!variable.has(one)) continue;
        variable.add(name);
        changed = true;
        break;
      }
    }
  }
  return variable;
}

/**
 * `check_quantities`: every quantity resolves (V10), reads only declared quantities (V1),
 * conforms to its declared type and domain (V3), declares a domain when it depends on the
 * assignment (V3), and — when a literal declares how it follows from the others — agrees with
 * that derivation (V11).
 *
 * The map is what `resolveQuantities` answered under the assignment in force: a name it
 * does not carry has no value, which is a refusal for a derived quantity and nothing at all for
 * an external one — an unassigned external is the template's normal state, reported by
 * {@link assignmentNeeded}, not by a rule.
 */
export function checkQuantities(model: PyRecord, quantities: Quantities): SemanticProblem[] {
  const problems: SemanticProblem[] = [];
  const declaredQuantities = demand(model, 'quantities');
  const declared = new Set(entries(declaredQuantities).map(([name]) => name));
  const variable = variableQuantities(model);
  for (const [name, declaration] of entries(declaredQuantities)) {
    const at = ['quantities', name];
    const source = demand(declaration, 'source');
    const kind = demand(source, 'kind');
    if (variable.has(name) && !has(declaration, 'domain')) {
      problems.push(
        semanticProblem(
          'V3',
          `quantity '${name}' depends on an external quantity and declares no domain ` +
            '(a variable quantity declares one, §2.1)',
          at,
        ),
      );
    }
    const expression =
      kind === 'derived'
        ? optional(source, 'expression', null)
        : optional(source, 'derivation', null);
    if (expression !== null) {
      const undeclared = [...quantityReferences(expression)]
        .filter((read) => !declared.has(read))
        .sort(comparePythonStrings);
      for (const read of undeclared) {
        problems.push(
          semanticProblem(
            'V1',
            `quantity '${name}': derivation reads undeclared quantity '${read}'`,
            at,
          ),
        );
      }
    }
    if (!quantities.has(name)) {
      if (kind === 'derived') {
        problems.push(
          semanticProblem(
            'V10',
            `quantity '${name}': derivation does not resolve ` +
              '(a cycle, or a reference with no value)',
            at,
          ),
        );
      }
      continue;
    }
    const value = quantities.get(name) as PyValue;
    const before = problems.length;
    checkType(value, demand(declaration, 'type'), name, problems);
    if (problems.length === before && has(declaration, 'domain')) {
      checkDomain(value, demand(declaration, 'domain'), name, problems);
    }
    locate(problems, before, at);
    if (kind === 'literal' && has(source, 'derivation')) {
      const derived = modelValue(demand(source, 'derivation'), quantities);
      if (derived === UNRESOLVED) {
        problems.push(
          semanticProblem('V10', `quantity '${name}': its derivation does not resolve`, at),
        );
      } else if (!pyEqual(derived, value)) {
        problems.push(
          semanticProblem(
            'V11',
            `quantity '${name}' = ${pyRepr(value)} disagrees with its derivation, ` +
              `which gives ${pyRepr(derived)}`,
            at,
          ),
        );
      }
    }
  }
  // The type table says `argument`, because the argument side is its other caller; here every
  // line speaks of a quantity. The tools rewrite the first occurrence, over the whole list, once.
  return problems.map((problem) =>
    withMessage(problem, problem.message.replace("argument '", "quantity '")),
  );
}

/**
 * `check_assignment`: an assignment against the external quantities it supplies — types and
 * domains, as at any call site (§4.6).
 *
 * Only the names the assignment carries are checked, and only where the document declares them
 * external: a name it does not declare is *not* refused — the tools iterate the document's
 * quantities and never look at the assignment's own names — and a name the assignment leaves
 * unset belongs to {@link assignmentNeeded}, not here.
 *
 * The place a problem carries is the quantity's declaration. An assignment is not part of the
 * document — it comes from `--assign`, from a template's call site, or from the editor's preview
 * sidecar (plan §5.5) — so the pointer names what refused the value rather than where the value
 * was written.
 */
export function checkAssignment(model: PyRecord, assignment?: PyRecord): SemanticProblem[] {
  const problems: SemanticProblem[] = [];
  const given = assignment ?? {};
  for (const [name, declaration] of entries(demand(model, 'quantities'))) {
    if (demand(demand(declaration, 'source'), 'kind') !== 'external') continue;
    if (!Object.hasOwn(given, name)) continue;
    const value = member(given, name) as PyValue;
    const before = problems.length;
    checkType(value, demand(declaration, 'type'), name, problems);
    if (problems.length === before && has(declaration, 'domain')) {
      checkDomain(value, demand(declaration, 'domain'), name, problems);
    }
    locate(problems, before, ['quantities', name]);
  }
  return problems.map((problem) => withMessage(problem, `assignment: ${problem.message}`));
}

/** What a document still needs before it denotes one graph, and what the tools print about it. */
export interface AssignmentNeeded {
  /** Every external quantity, in the document's own order — the order the sheet shows them in. */
  readonly external: readonly string[];
  /** Those with no declared default, which an assignment must supply: `external_names(False)`. */
  readonly required: readonly string[];
  /** Those of {@link required} the assignment leaves unset, sorted: `missing_assignment`. */
  readonly unset: readonly string[];
  /**
   * `needs --assign for ['eps', 'head_dim', …]`, or `null` when nothing is unset.
   *
   * The fragment the three call sites share: `--validate` prints `schema ok; semantic ` before it,
   * `--d1` and `--derive` print `skipped: `, and each counts the document as *skipped*, never as
   * failed — "a template with no assignment is skipped, not failed: it is a family of graphs, and
   * refusing it would report a defect where there is none".
   */
  readonly report: string | null;
}

/**
 * The assignment-needed report: what `missing_assignment` answers, with the names around it the
 * editor's assignment sheet is generated from (plan §4.6's sheet, D9).
 *
 * A document with no external quantity — every model of the corpus — reports nothing unset, which
 * is what makes it readable on its own.
 */
export function assignmentNeeded(model: PyRecord, assignment?: PyRecord): AssignmentNeeded {
  const unset = missingAssignment(model, assignment);
  return {
    external: [...externalNames(model)],
    required: [...externalNames(model, false)],
    unset,
    report: unset.length === 0 ? null : `needs --assign for ${pyRepr(unset)}`,
  };
}

/** The problems appended since `from`, given the place in the document they are about. */
function locate(problems: SemanticProblem[], from: number, segments: readonly string[]): void {
  for (let index = from; index < problems.length; index += 1) {
    const problem = problems[index] as SemanticProblem;
    problems[index] = semanticProblem(problem.code, problem.message, segments);
  }
}
