# Acceptance fixtures

Documents and primitive-library bases written for one reason: **the grammar they use is written
nowhere in `data/models/`, `data/primitive-library/` or `tests/rejections/`.** The language admits
them, the schemas admit them, and no file of the repository exercises them — so the TypeScript core
could read any of these constructs differently from `tools/` and every parity suite would stay
green. That is the gap the implementation plan records as finding **F8**, and these files are how
it is closed editor-side.

They are **inputs, not expectations.** `pnpm oracle` runs `tools/tensorspine` over every one of them
exactly as it runs it over the corpus — `--validate`, `--d1`, `--derive`, and `artifact.check`
against a checkpoint synthesised from each one's own D3 — and records what the tools answered under
`editor/tests/oracle/out/fixtures/`. `packages/lang/test/parity/fixtures.test.ts` is where the core
is held to it, product by product and byte for byte. Nothing in this directory says what an answer
should be, and nothing here has to be refreshed when the tools change.

Two audits guard them without needing Python: `editor/tests/audit/acceptance-fixtures.test.ts`
(each construct is still in the file that exists for it) and the "expectations that are not the
oracle's" register in `editor/tests/audit/parity-job.test.ts`.

## Layout

```
fixtures/
├── models/      documents, each opened under editor/tests/fixtures/models/<name>.json
├── base/        tensorspine.editor-fixtures — the one unit of the repository with a constant slot
└── scratch/     tensorspine.scratch.derive  — two primitives whose port shapes may not resolve
```

Every document declares `../../../../data/primitive-library/` first, so the reference base is the
shared vocabulary and a fixture base sits beside it exactly as a laboratory's own base would (plan
§9 Q6). Both fixture bases reuse the reference base's axes and precision roles and declare none of
their own.

## The documents

| Document | What it exists for |
|---|---|
| `grid-composition.json` | a composition of **two indices** (`row` × `col`, a 2×3 grid), with a scoped carry on each index and a guard reading both; a **`concat`** location; the five operators the corpus and the reference base never write — `divide`, `min`, `max`, `negate`, `absolute` — in derived quantities, three of them deciding the grid's own ranges |
| `top-level-for-each.json` | a **top-level `for_each`** as an author writes it, not as `model.normalise` hoists one: a value rule and a parameter rule repeating over their own index range, and a composition whose range starts at 1 with a step of 2 |
| `declared-constant.json` | a document **`constants`** map, a `bindings.constants` rule and its **scoped** form, binding the constant slot of `fixture.position_bias`; and a **`stack` along an axis that is not the first** |
| `root-level-when.json` | a **root-level `when`**: a root instance a guard keeps, one a guard removes, a top-level binding naming the absent instance, a binding a guard removes and one it keeps — in all five forms of the model condition language, where the corpus writes `compare` and `all` alone; a **boolean** and a **physical** quantity and an **upper-inclusive** domain bound |
| `one-template-instance.json` | a caller whose **public interfaces name a template instance**, which `inputs_at` and `outputs_at` resolve *through* — the composite's feed `embed` and `lm_head` instead |
| `one-template-instance-dtype.json` | the same caller with the three readings of a dtype selector inside the instance told apart: the caller's quantity `f32`, the argument's literal `f16`, the role's default `bf16` |
| `scratch-blank.json` | a port shape that does not resolve on the **input** side, where `derive` takes a byte size unguarded and raises out of the whole derivation |
| `scratch-fed.json` | the same on the **output** side, where the same figure is guarded and the derived document carries the blank its schema admits |

The last four were strings inside the unit suites until feature 1.13, with answers read off
`tools/derive.py` once, by hand, for want of an oracle; they are files now so that the oracle owns
them.

## What they showed

Three facts the corpus cannot show, recorded here because a reader of these files will wonder:

- **A composition of several indices has no layer graph split.**
  `derive._structural_graph_splits` emits one per prefix of a composition's index *only* when the
  composition has exactly one index (`len(comp[1]) == 1`), so `grid-composition` gets family splits
  and nothing else. Reproduced, not corrected.
- **The five operators are declared, not all of them read.** `min`, `max` and `absolute` decide
  `grid-composition`'s index ranges, so they run inside `index_grid` and the expansion follows
  them; `divide` and `negate` stand as derived quantities that nothing consumes, because no
  argument of the reference base takes a real that may be negative or fractional. Both are still
  resolved and type-checked (V3, V10) on every reading of the document, which is what the fixture
  is for.
- **`constants` reaches no rule and no product.** `validate.analyse` never reads
  `bindings.constants` and `derive` never reads `constants`, so a constants rule naming a constant
  the document does not declare, an instance that does not exist and a slot that does not exist is
  accepted by `--validate` — V1's resolution, V4's shape agreement and V7's totality are all
  unimplemented for constants, and D1–D6 carry no trace of one. `declared-constant.json` is written
  as the language says it should be, so the day the rules arrive the fixture still holds.

## Adding one

A fixture is added by writing the document (and, where it needs one, the unit), running
`pnpm oracle` and then `pnpm test:parity`. The parity suite reads the directory listing, so a
document added without an expectation fails rather than sitting unread; the audit's construct table
is where the *reason* for the new file is written down.
