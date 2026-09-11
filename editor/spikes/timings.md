# Timings — the budgets of the plan’s §5.6, measured

*Written by `pnpm spike:timings`, which runs `editor/packages/lang/test/api/timings.test.ts`.*

Machine: 8 × Intel(R) Core(TM) i7-4790K CPU @ 4.00GHz, Node v22.22.3.
Every figure is the best of a few runs — what the editor pays once the code is warm, which is
what it pays — taken on an idle machine, which is where a budget means anything. The documents
are the largest of the repository’s own `data/models/` with `llama3-8b` beside them, read
through the `Lang` API of feature 1.11; "across the worker boundary" is the same call through
a structured clone. What `pnpm check` asserts on every commit is not the budget but a
regression guard at three times it: the suite runs beside five other test files there, and a
budget asserted flat would fail for the machine’s reasons rather than the code’s. Making a
budget itself fail the job is feature X.2’s, on an idle runner.

| Measure | Budget | Measured | Worst on | Verdict |
|---|---|---|---|---|
| loadSchemas (registry compiled) | — | 49.36 ms | `schemas/` | — |
| loadLibrary, cold (first gather of the reference base) | 300 ms | 264.13 ms | `data/primitive-library` | within — the files are already in memory: the caller reads them (§5.3) |
| loadLibrary, warm | 300 ms | 27.45 ms | `data/primitive-library` | within |
| Ajv on a corpus document | 5 ms | 0.45–3.70 ms | `deepseek-v4-pro` | within |
| validate, in this thread | 300 ms | 23.03–99.45 ms | `deepseek-v4-pro` | within |
| validate, across the worker boundary | 300 ms | 23.94–101.41 ms | `deepseek-v4-pro` | within |
| derive, in this thread | 2000 ms | 55.06–245.94 ms | `qwen3.5-397b` | within |
| derive, across the worker boundary | 2000 ms | 64.30–270.70 ms | `qwen3.5-397b` | within |
| expand (D1), across the worker boundary | 2000 ms | 8.90–31.17 ms | `deepseek-v4-pro` | within |
| describe, whole document, across the boundary | 20 ms | 37.69–175.04 ms | `deepseek-v4-pro` | **over on 6 of 6** — the semantic stage it reads is 26–100 ms on its own (feature 1.6c) |
| describe, one site, across the boundary | 20 ms | 5.35–24.96 ms | `deepseek-v4-pro` | **over on 1 of 6** — the analysis is reused; what is left is one site described and the document’s compatibility walk |
| check, across the boundary | 20 ms | 0.09 ms | `deepseek-v4-pro` | within |
| validateUnit, across the boundary | 20 ms | 5.42 ms | `attention.dense@1.0.0` | within |
| structured clone of a document’s facts | — | 7.96–25.29 ms | `deepseek-v4-pro` | — once out of the worker; the analysis is not among them, and costs 8–37 ms more if it is |
| structured clone of a derived document | — | 8.66–17.24 ms | `qwen3.5-397b` | — |

## What the figures say

- **The two budgets this feature is held to are met with room.** The whole semantic stage and
  the whole derivation stay well inside §5.6’s 300 ms and 2 s, in this thread and across the
  boundary alike, and the suite asserts both on every run. The boundary costs the clone and
  nothing else: a derivation across it is within a tenth of the same derivation here.
- **`describe` over a whole document cannot meet the 20 ms round trip, and never could.**
  Feature 1.6d had already measured why — `analyse` alone is 10–93 ms — and the clone of every
  site’s facts is on top. What the sheet actually asks for is one site (plan §3: "`describe`
  for the selected instance"), and `only` is what answers that: the analysis is the session’s
  already, and one site is described and carried.
- **One site still pays the document’s compatibility walk.** Feature 1.6d measured it at
  3.9–63.7 ms: the partners a slot may join are answered from every identity instance of the
  graph, so narrowing the *sites* does not narrow it. It is inside the budget on every document
  measured but the largest, `deepseek-v4-pro`. A finding for feature 2.10: the
  sheet needs the facts on every keystroke and the compatibility lists only when a chip’s menu
  opens, so the knob that would close the gap is a `describe` that leaves them out — not asked
  for by any document of the plan, and therefore not invented here.
- **`check` and `validateUnit` meet the budget by a wide margin**, because neither analyses:
  `check` reads the analysis the session kept for that document (feature 1.6d’s decision), and
  `validateUnit` reads one unit against a library already gathered.
- **The analysis stays in the worker, and that is what the clone figures are for.** The whole
  `Description` of the largest corpus document costs 45 ms to clone and its sites alone 25; the
  facts the API answers are the sites, so the analysis — which the interface displays nothing
  of — never crosses. `check` is what reads it, where it is.
- **The library is gathered once a session**, inside its 300 ms, from files the caller read:
  the core opens nothing (§5.3). Every later document names the same handle.
