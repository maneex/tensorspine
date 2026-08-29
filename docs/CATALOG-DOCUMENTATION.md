# Catalog documentation model — proposal

> Give every catalog unit a fixed place to say what it is, for people, without letting that prose
> change what the unit means; then render the whole catalog from its definitions and that prose
> with `tensorspine --document catalog`.

*Proposal, 28 August 2026. Companion to `schemas/tensorspine-documentation.schema.json`,
`tools/document.py` and the generated [catalog reference](CATALOG-REFERENCE.md). Non-normative:
the [specification](SPECIFICATION.md) remains the authority on what a catalog unit means.*

---

## 1 — What is being proposed

Two things, one depending on the other:

1. **A documentation model**: a closed set of documentation fields that a contract, an axis, a
   precision role and a base manifest may carry, with a fixed shape for each, expressed as a JSON
   Schema (`schemas/tensorspine-documentation.schema.json`). It is modelled on the documentation
   constructs of OpenAPI 3.1 — `info`, `summary`/`description`, `externalDocs`, `tags`,
   `deprecated` — with the same rule OpenAPI and JSON Schema both apply: **documentation never
   changes what a document denotes.**
2. **A generator**: `tensorspine --document catalog` reads the catalog and writes one Markdown file
   that contains *everything* the catalog says — every argument, port, slot, state rule, cost and
   partition of every contract, rendered in readable notation — plus the documentation fields
   where they exist.

The generator is useful on a catalog with no documentation fields at all, because the definitions
already carry most of what a reader needs. The documentation model is what turns that rendering
from a *listing* into a *reference*: the one-line summary in an index, the paragraph that says when
to use `attention.dense` rather than `attention.latent_compressed`, the paper that introduced the
primitive.

### 1.1 — Constraints the specification imposes

The specification already decides most of the design:

| Rule | Consequence for documentation |
|---|---|
| **§10.2 mutation test** — a field ignored by every conforming implementation is a comment | Documentation fields are comments *by construction*. `--validate`, `--d1` and D2–D6 never read them. Only `--document` and, later, `--lint` do. |
| **I7 — no silent defaults** | The generator never invents prose. A unit without a `summary` is rendered without one and counted as undocumented. It is not summarised from its `note`. |
| **§3.4 — each fact has one authority** | A `description` explains meaning; it never restates a type, a default, a shape or a law. Those are rendered from the definition, so a restatement could only drift from it. |
| **O0.6 — every language is closed** | The set of documentation fields is closed and schema-checked. There is no `x-` extension namespace: an open comment channel would be a field with no schema. |
| **§8.2 — contract identity is immutable** | Editing documentation does not create a new contract version: the meaning is unchanged. A deprecation is advice, not a change of denotation. |

### 1.2 — OpenAPI as the model

| OpenAPI 3.1 | Tensorspine catalog | Notes |
|---|---|---|
| `info` (`title`, `summary`, `description`, `contact`, `license`) | Base manifest `catalog.json`: `title`, `summary`, `description`, `contact`, `license` | `info.version` is deliberately absent: a catalog has no global version (§8.2). |
| `tags` (declared at the top, cited by operations) | Base manifest `tags: [{name, summary, description?, external_docs?}]`; units cite by name | Namespaces (`attention.*`) already group by structure; a tag groups by a property that cuts across them. |
| `externalDocs` (one object) | `external_docs` (a list of `{url, title?, description?, kind?}`) | A primitive routinely has several sources: the paper, the reference implementation, the discussion that fixed a convention. `kind` lets the renderer group them. |
| Operation `summary`, `description` | Unit-level `summary` (one line, ≤ 120 characters, plain text) and `description` (CommonMark) | On contracts, axes, precision roles and the base. |
| Operation `deprecated: true` | `deprecated: {reason, superseded_by?: {name, version}}` | An object rather than a boolean: the useful fact is what to use instead, and the renderer links it. |
| Parameter `description`, `deprecated` | `description` and `deprecated: {reason, superseded_by?: <argument>}` on every argument and record field | |
| Schema `description` on every property | `description` on every port, parameter slot, constant slot, state port, payload component, state operation, state rule, partition, cost entry, sparsity unit, domain transform | One field per site; the site's other facts are already declared. |
| `x-enum-descriptions` (vendor extension, no standard) | `value_descriptions: {value: text}` on an enum argument | Keys must be declared values; checked by the generator. |
| Example Object | — | Not adopted (§2.5). |
| `x-*` specification extensions | — | Rejected: the vocabulary is closed (O0.6). |

## 2 — The documentation model

### 2.1 — Where the fields live

Documentation fields sit **inside `definition`, beside `note`**, at the site they document. A
contract's `summary` is a sibling of its `version`; an argument's `description` is a sibling of its
`type`. Nothing moves; nothing is wrapped. The existing `note` field keeps its role.

| Unit kind | Unit-level fields | Element-level fields |
|---|---|---|
| Contract (primitive) | `summary`, `description`, `external_docs`, `tags`, `deprecated` | `description` on every argument, record field, port, parameter slot, constant slot, state port, payload component, state operation, state rule, partition, cost entry, sparsity unit, domain transform; `value_descriptions` and `deprecated` on arguments |
| Contract (template) | same | none: its elements are derived from the template |
| Axis | `summary`, `description`, `external_docs`, `tags`, `deprecated` | — |
| Precision role | `summary`, `description`, `external_docs`, `tags`, `deprecated` | — |
| Base manifest | `title`, `summary`, `description`, `contact`, `license`, `external_docs`, `tags` (declarations) | — |

Shape axes (`{name, axis, nature, extent}`) carry no documentation: the local name and the axis
identity are self-explanatory, and the axis unit carries the prose.

### 2.2 — The fields

The schema is `schemas/tensorspine-documentation.schema.json` (`$id`
`https://tensorspine.dev/schema/2.0/documentation.json`). Its `$defs` are the authority; this table is
the reading guide.

| Field | Shape | Rule |
|---|---|---|
| `summary` | string, one line, ≤ 120 characters | What the unit *is*. Plain text: it lands in index rows and table cells. |
| `description` | string, CommonMark | What the unit *means* and how to use it. Never restates a fact the definition carries. |
| `external_docs` | `[{url, title?, description?, kind?}]`, `kind ∈ {paper, reference_implementation, specification, discussion, other}` | `url` is absolute, or a path relative to the repository root (rewritten by the renderer relative to the output file). |
| `tags` | `[tag_name]`, unique | A tag a unit cites should be declared by a base manifest; an undeclared tag is an advisory finding. |
| `deprecated` | `{reason, superseded_by?}` | On a unit: `superseded_by` is `{name, version}`. On an argument: another argument's name. |
| `value_descriptions` | `{value: description}` | Only on an enum-typed argument; every key must be a declared value. |
| `title`, `contact`, `license` | as OpenAPI `info` | Base manifest only. |
| `tags` (base) | `[{name, summary, description?, external_docs?}]` | Declarations, as OpenAPI's top-level `tags`. |

### 2.3 — `note` and `description` are different fields

The catalog carries 79 `note` fields. They are **maintainers' asides**: why a default is what it is,
why a fused axis has no factors, why a port is optional. That is the *why*, written by and for the
people who curate the catalog. A `description` is the *what*, written for the people who write
models against the catalog and the people who implement runtimes from it.

Both are kept and both are rendered — the description as prose, the note quoted as a
"Note (maintainers)". No note was rewritten or moved in this change.

### 2.4 — Integration into the catalog grammar

The catalog grammar, `schemas/tensorspine-catalog-unit.schema.json`, is in the tree and every unit of
every base is read against it when a catalog is loaded (`catalog.load`), so a unit outside the
vocabulary is a load error, never an advisory finding. The documentation fields are declared in that
grammar at the site they document, by reference: `schemas/tensorspine-documentation.schema.json` is
the single source of their shapes, and the catalog grammar `$ref`s its definitions (`summary`,
`description`, `external_docs`, `tags`, `deprecated`, `value_descriptions`; `title`, `contact`,
`license` and the tag declarations on the base) with `additionalProperties: false` around them, which
is the closure O0.6 asks for. One definition, two readers: the loader refuses a malformed field, the
generator renders a well-formed one.

### 2.5 — What is deliberately not in the model

- **No examples.** OpenAPI's Example Object has no counterpart. The catalog says what a primitive
  *is*; what invoking it with particular arguments produces belongs to a model document, and the
  corpus already provides twelve of them.
- **No `x-` extensions.** Closed vocabulary (O0.6).
- **No documentation version or `since`.** There is no catalog version to date it against; the
  file's history is in git.
- **No documentation on shape axes** (§2.1).
- **One language.** `description` is a string, not a language map. If bilingual descriptions are
  wanted later, the change is local to `$defs/description` and `summary`.

## 3 — The generator: `tensorspine --document catalog`

```sh
python3 tools/tensorspine --document catalog -o docs/CATALOG-REFERENCE.md
python3 tools/tensorspine --document catalog                                 # to stdout
python3 tools/tensorspine --document catalog --catalog other/catalog -o out/  # writes out/catalog.md
```

- **Inputs.** The catalog bases (`--catalog`, default `data/catalog/`) and the documentation schema
  (`--schemas`, default `schemas/`); the template of a template contract is resolved from the
  location its base declares (`templates`) and its pinned name, version and id are checked when
  the catalog is loaded. The model documents given as `PATH`s are not read by this command.
- **Output.** `-o FILE`, `-o DIR` (writes `DIR/catalog.md`), or stdout when `-o` is omitted; the
  status line then goes to stderr so the page can be piped. The output is **deterministic**: the
  same catalog gives the same bytes at the same location, with no timestamp, so the file can be
  committed and diffed.
- **Exit status.** `0` written; `1` refused — an unreadable catalog, a malformed documentation
  field — with every cause on stderr and nothing written. Findings that are legal but worth knowing
  (a tag no base declares, a condition citing an argument the contract does not declare) go to the
  findings appendix and never block.

### 3.1 — What the page contains

| Section | Source | Content |
|---|---|---|
| Head | base manifest | Title, summary, description, contact, license, external docs; bases consulted; counts. |
| Contents, How to read | — | Navigation; the notation: expressions in infix, conditions in words, shapes as `[name: extent]`, structural arguments, ordered rules. |
| Overview | all units | One index table per kind: contract with summary and shape (`17 args · 2→1 ports · 9 params · state kv`), axes, precision roles. |
| Contracts | contract units, grouped by namespace | Per contract: summary, tags, description, note, external docs, an at-a-glance row, then **Arguments** (with nested record fields and enum value descriptions), **Ports**, **Parameters** (with sharing, presence, multiplicity, declared views), **Constant slots**, **State ports** (presence, key axes, payload, permitted operations, carrying condition, ordered derivation rules), **Effects**, **Logical cost** (corrections), **Structured sparsity** (units), **Semantic partitions**, **Domain transforms**. A template contract shows its template: resolved path, the template's external quantities as arguments with their domains, its public interfaces, and the transitive closure of contracts it cites — the consumer's capability cost (§8.1). |
| Axes, Precision roles | axis and role units | Tables with summary, then details for units with a description. |
| Tags | base manifest | Declared tags. A unit that carries a tag says so in its own section. |
| Appendix A | derived | Closed vocabulary in use: every value of every closed enumeration at least one unit uses (laws, access geometries, sharing, communications, natures, domains, dtypes…), with how many units use it. A runtime that implements these values implements the catalog as it stands. |
| Appendix B | derived | Documentation coverage per site kind, and the undocumented units by name. |
| Appendix C | derived | Findings. |

### 3.2 — What it does not do

- **No reverse references.** A unit's page says what the unit declares and nothing about who uses
  it: no "cited by" on contracts, axes or precision roles, no index of models by contract, no index
  of slots by role. Those are facts about the corpus, not about the catalog.
- **No template expansion.** It does not expand templates (D1), nor derive parameter tensors,
  states, costs or partitions from them (D3–D6): those are products of the compiler, and the page
  says so where a template contract is rendered.
- **No grammar check of its own.** It reads the catalog through `catalog.load`, which already
  refuses a unit off the grammar or with an identity disagreeing with its path; the generator adds
  no check beyond the documentation fields.

### 3.3 — Where the output lives

`docs/CATALOG-REFERENCE.md`, regenerated whenever a unit changes and committed, so the reference is
readable on the repository without running anything. The documentation plan reserves
`docs/CATALOG.md` for a hand-written guide to the catalog's *organisation* — resolution order, how
to add a unit, versioning. That guide should link to the generated reference for the lists rather
than carry them: a list maintained by hand drifts.

## 4 — What this change contains

- `schemas/tensorspine-documentation.schema.json` — the model, 20 `$defs`.
- `tools/document.py`, `tools/tensorspine --document` — the generator.
- **The catalog, documented in full.** Every one of the 34 contracts, 35 axes and 54 precision roles
  carries a `summary` and a `description`; every argument, record field, port, parameter slot, state
port, payload component, state operation, state rule, partition, cost entry and domain transform of
  every primitive contract carries a `description`, as does every cost correction and sparsity
  unit; every enum argument has `value_descriptions`.
  The base manifest carries a title, summary, description, a `specification` external doc and
  three declared tags — `sequence-operator`, `multimodal`, `parallel-residual` — which the
  contracts concerned cite. Contracts whose primitive has a canonical paper cite it in
  `external_docs`; those that describe a single model's construction (Gemma 3n conditioning, the
  Mistral patch merger, the residual helpers) cite nothing rather than something approximate.
- `docs/CATALOG-REFERENCE.md` — the rendered catalog.

Rendering the whole catalog once surfaced a defect the validator of the time did not see: two
slots of `attention.dense` guarded by arguments the contract never declared. The catalog loader now
resolves every argument path a condition cites, so that class of defect is a load error, and the
guards are gone.

`--validate` and `--lint` give the same results before and after: the documentation fields are
inert, as §10.2 requires.

## 5 — Authoring guidance

1. **Write to the field's contract.** A summary is one line under 120 characters and is not
   CommonMark. A description does not restate a type, a default, a shape or a law; it says what
   the unit computes, when to choose it over its neighbours, and what its arguments change.
2. **Cite what introduced the primitive**, not what happens to use it: the paper as `paper`, the
   `modeling_x.py` as `reference_implementation`. When no canonical source exists, cite nothing.
3. **Keep `note` for the why.** A maintainer's aside about a decision stays a note; do not fold it
   into the description, and do not duplicate it there.
4. **Tag across namespaces, not along them.** `attention.*` needs no tag; "carries state" does.
5. **Regenerate and read the appendices.** A new unit without a summary shows up in coverage; a key
   with a typo shows up in findings.

## 6 — Open questions and follow-ups

- **Lint.** Three advisory rules fall out of the generator and belong in `--lint` so they run with
  the rest of the hygiene checks: a unit without a summary, a cited tag no base declares, a
  condition citing an undeclared argument.
- **`--document model`.** The same renderer applied to a model document — quantities, occurrences
  with their resolved arguments, bindings, interfaces — would give the tutorial material the
  documentation plan asks for, from the corpus itself. The `--document WHAT` form leaves room for
  it.
- **One language or two.** If descriptions are ever wanted in two languages, `summary` and
  `description` become language maps; everything else is unchanged.
