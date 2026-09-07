<a id="catalog-documentation-model-proposal"></a><a id="primitive_library-documentation-model-proposal"></a>

<a id="catalog-documentation-model--proposal"></a><a id="primitive-library-documentation-model--proposal"></a>

# Primitive Library documentation model — proposal

> Give every primitive library unit a fixed place to say what it is, for people, without letting that prose
> change what the unit means; then render the whole primitive library from its definitions and that prose
> with `tensorspine --document primitive-library`.

*Proposal, 28 August 2026. Companion to `schemas/tensorspine-documentation.schema.json`,
`tools/document.py` and the generated [primitive library reference](https://maneex.github.io/tensorspine/primitive-library/). Non-normative:
the [specification](SPECIFICATION.md) remains the authority on what a primitive library unit means.*

---

<a id="1--what-is-being-proposed"></a>

## 1 — What is being proposed

Two things, one depending on the other:

1. **A documentation model**: a closed set of documentation fields that a primitive, an axis, a
   precision role and a base manifest may carry, with a fixed shape for each, expressed as a JSON
   Schema (`schemas/tensorspine-documentation.schema.json`). It is modelled on the documentation
   constructs of OpenAPI 3.1 — `info`, `summary`/`description`, `externalDocs`, `tags`,
   `deprecated` — with the same rule OpenAPI and JSON Schema both apply: **documentation never
   changes what a document denotes.**
2. **A generator**: `tensorspine --document primitive-library` reads the primitive library and writes one Markdown file
   that contains *everything* the primitive library says — every argument, port, slot, state rule, cost and
   partition of every primitive, rendered in readable notation — plus the documentation fields
   where they exist.

The generator is useful on a primitive library with no documentation fields at all, because the definitions
already carry most of what a reader needs. The documentation model is what turns that rendering
from a *listing* into a *reference*: the one-line summary in an index, the paragraph that says when
to use `attention.dense` rather than `attention.latent_compressed`, the paper that introduced the
primitive.

<a id="11--constraints-the-specification-imposes"></a>

### 1.1 — Constraints the specification imposes

The specification already decides most of the design:

| Rule | Consequence for documentation |
|---|---|
| **§10.2 mutation test** — a field ignored by every conforming implementation is a comment | Documentation fields are comments *by construction*. `--validate`, `--d1` and D2–D6 never read them. Only `--document` and, later, `--lint` do. |
| **I7 — no silent defaults** | The generator never invents prose. A unit without a `summary` is rendered without one and counted as undocumented. It is not summarised from its `note`. |
| **§3.4 — each fact has one authority** | A `description` explains meaning; it never restates a type, a default, a shape or a evolution. Those are rendered from the definition, so a restatement could only drift from it. |
| **O0.6 — every language is closed** | The set of documentation fields is closed and schema-checked. There is no `x-` extension namespace: an open comment channel would be a field with no schema. |
| **§8.2 — primitive identity is immutable** | Editing documentation does not create a new primitive version: the meaning is unchanged. A deprecation is advice, not a change of denotation. |

<a id="12--openapi-as-the-model"></a>

### 1.2 — OpenAPI as the model

| OpenAPI 3.1 | TensorSpine primitive library | Notes |
|---|---|---|
| `info` (`title`, `summary`, `description`, `contact`, `license`) | Base manifest `primitive-library.json`: `title`, `summary`, `description`, `contact`, `license` | `info.version` is deliberately absent: a primitive library has no global version (§8.2). |
| `tags` (declared at the top, cited by operations) | Base manifest `tags: [{name, summary, description?, external_docs?}]`; units cite by name | Namespaces (`attention.*`) already group by structure; a tag groups by a property that graph splits across them. |
| `externalDocs` (one object) | `external_docs` (a list of `{url, title?, description?, kind?}`) | A primitive routinely has several sources: the paper, the reference implementation, the discussion that fixed a convention. `kind` lets the renderer group them. |
| Operation `summary`, `description` | Unit-level `summary` (one line, ≤ 120 characters, plain text) and `description` (CommonMark) | On primitives, axes, precision roles and the base. |
| Operation `deprecated: true` | `deprecated: {reason, superseded_by?: {name, version}}` | An object rather than a boolean: the useful fact is what to use instead, and the renderer links it. |
| Parameter `description`, `deprecated` | `description` and `deprecated: {reason, superseded_by?: <argument>}` on every argument and record field | |
| Schema `description` on every property | `description` on every port, parameter slot, constant slot, state port, payload component, state operation, state rule, partition, cost entry, sparsity unit, domain transform | One field per site; the site's other facts are already declared. |
| `x-enum-descriptions` (vendor extension, no standard) | `value_descriptions: {value: text}` on an enum argument | Keys must be declared values; checked by the generator. |
| Example Object | — | Not adopted (§2.5). |
| `x-*` specification extensions | — | Rejected: the vocabulary is closed (O0.6). |

<a id="2--the-documentation-model"></a>

## 2 — The documentation model

<a id="21--where-the-fields-live"></a>

### 2.1 — Where the fields live

Documentation fields sit **inside `definition`, beside `note`**, at the site they document. A
primitive's `summary` is a sibling of its `version`; an argument's `description` is a sibling of its
`type`. Nothing moves; nothing is wrapped. The existing `note` field keeps its role.

| Unit kind | Unit-level fields | Element-level fields |
|---|---|---|
| Primitive Definition (primitive) | `summary`, `description`, `external_docs`, `tags`, `deprecated` | `description` on every argument, record field, port, parameter slot, constant slot, state port, payload component, state operation, state rule, partition, cost entry, sparsity unit, domain transform; `value_descriptions` and `deprecated` on arguments |
| Primitive Definition (template) | same | none: its elements are derived from the template |
| Axis | `summary`, `description`, `external_docs`, `tags`, `deprecated` | — |
| Precision role | `summary`, `description`, `external_docs`, `tags`, `deprecated` | — |
| Base manifest | `title`, `summary`, `description`, `contact`, `license`, `external_docs`, `tags` (declarations) | — |

Shape axes (`{name, axis, nature, extent}`) carry no documentation: the local name and the axis
identity are self-explanatory, and the axis unit carries the prose.

<a id="22--the-fields"></a>

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

<a id="23--note-and-description-are-different-fields"></a>

### 2.3 — `note` and `description` are different fields

Primitive Library `note` fields are **maintainers' asides**: why a default is what it is,
why a fused axis has no factors, why a port is optional. That is the *why*, written by and for the
people who curate the primitive library. A `description` is the *what*, written for the people who write
models against the primitive library and the people who implement runtimes from it.

Both are kept and both are rendered — the description as prose, the note quoted as a
"Note (maintainers)". No note was rewritten or moved in this change.

<a id="integration-into-the-catalog-grammar"></a><a id="integration-into-the-primitive_library-grammar"></a>

<a id="24--integration-into-the-catalog-grammar"></a><a id="24--integration-into-the-primitive-library-grammar"></a>

### 2.4 — Integration into the primitive library grammar

The primitive library grammar, `schemas/tensorspine-primitive-library-unit.schema.json`, is in the tree and every unit of
every base is read against it when a primitive library is loaded (`primitive_library.load`), so a unit outside the
vocabulary is a load error, never an advisory finding. The documentation fields are declared in that
grammar at the site they document, by reference: `schemas/tensorspine-documentation.schema.json` is
the single source of their shapes, and the primitive library grammar `$ref`s its definitions (`summary`,
`description`, `external_docs`, `tags`, `deprecated`, `value_descriptions`; `title`, `contact`,
`license` and the tag declarations on the base) with `additionalProperties: false` around them, which
is the closure O0.6 asks for. One definition, two readers: the loader refuses a malformed field, the
generator renders a well-formed one.

<a id="25--what-is-deliberately-not-in-the-model"></a>

### 2.5 — What is deliberately not in the model

- **No examples.** OpenAPI's Example Object has no counterpart. The primitive library says what a primitive
  *is*; what invoking it with particular arguments produces belongs to a model definition. The
  generated [status page](https://maneex.github.io/tensorspine/status/) lists the corpus.
- **No `x-` extensions.** Closed vocabulary (O0.6).
- **No documentation version or `since`.** There is no primitive library version to date it against; the
  file's history is in git.
- **No documentation on shape axes** (§2.1).
- **One language.** `description` is a string, not a language map. If bilingual descriptions are
  wanted later, the change is local to `$defs/description` and `summary`.

<a id="the-generator-tensorspine---document-catalog"></a><a id="the-generator-tensorspine---document-primitive_library"></a>

<a id="3--the-generator-tensorspine---document-catalog"></a><a id="3--the-generator-tensorspine---document-primitive-library"></a>

## 3 — The generator: `tensorspine --document primitive-library`

```sh
python3 tools/tensorspine --document primitive-library -o /tmp/PRIMITIVE-LIBRARY-REFERENCE.md
python3 tools/tensorspine --document primitive-library                                 # to stdout
python3 tools/tensorspine --document primitive-library --primitive-library other/primitive-library -o out/  # writes out/primitive-library.md
```

- **Inputs.** The primitive library bases (`--primitive-library`, default `data/primitive-library/`) and the documentation schema
  (`--schemas`, default `schemas/`); the template of a template primitive is resolved from the
  location its base declares (`templates`) and its pinned name, version and id are checked when
  the primitive library is loaded. The model definitions given as `PATH`s are not read by this command.
- **Output.** `-o FILE`, `-o DIR` (writes `DIR/primitive-library.md`), or stdout when `-o` is omitted; the
  status line then goes to stderr so the page can be piped. The output is **deterministic**: the
  same primitive library gives the same bytes at the same logical location, with no timestamp. The site
  build generates it in temporary storage; it is never committed.
- **Exit status.** `0` written; `1` refused — an unreadable primitive library, a malformed documentation
  field — with every cause on stderr and nothing written. Findings that are valid but worth knowing
  (a tag no base declares, a condition citing an argument the primitive does not declare) go to the
  findings appendix and never block.

<a id="31--what-the-page-contains"></a>

### 3.1 — What the page contains

| Section | Source | Content |
|---|---|---|
| Head | base manifest | Title, summary, description, contact, license, external docs; bases consulted; counts. |
| Contents, How to read | — | Navigation; the notation: expressions in infix, conditions in words, shapes as `[name: extent]`, structural arguments, ordered rules. |
| Overview | all units | One index table per kind: primitive summary and shape, axes, precision roles. |
| Primitives | primitive units, grouped by namespace | Per primitive: summary, tags, description, note, external docs, an at-a-glance row, then **Arguments** (with nested record fields and enum value descriptions), **Ports**, **Parameters** (with sharing, presence, multiplicity, declared views), **Constant slots**, **State ports** (presence, key axes, payload, permitted operations, carrying condition, ordered derivation rules), **Effects**, **Logical cost** (corrections), **Structured sparsity** (units), **Semantic partition options**, **Domain transforms**. A template primitive shows its template: resolved path, the template's external quantities as arguments with their domains, its public interfaces, and the transitive closure of primitives it cites — the consumer's capability cost (§8.1). |
| Axes, Precision roles | axis and role units | Tables with summary, then details for units with a description. |
| Tags | base manifest | Declared tags. A unit that carries a tag says so in its own section. |
| Appendix A | derived | Closed vocabulary in use: every value of every closed enumeration at least one unit uses (evolutions, access geometries, sharing, communications, natures, domains, dtypes…), with how many units use it. A runtime that implements these values implements the primitive library as it stands. |
| Appendix B | derived | Documentation coverage per site kind, and the undocumented units by name. |
| Appendix C | derived | Findings. |

<a id="32--what-it-does-not-do"></a>

### 3.2 — What it does not do

- **No reverse references.** A unit's page says what the unit declares and nothing about who uses
  it: no "cited by" on primitives, axes or precision roles, no index of models by primitive, no index
  of slots by role. Those are facts about the corpus, not about the primitive library.
- **No template expansion.** It does not expand templates (D1), nor derive parameter tensors,
  states, costs or partition options from them (D3–D6): those are products of the compiler, and the page
  says so where a template primitive is rendered.
- **No grammar check of its own.** It reads the primitive library through `primitive_library.load`, which already
  refuses a unit off the grammar or with an identity disagreeing with its path; the generator adds
  no check beyond the documentation fields.

<a id="33--where-the-output-lives"></a>

### 3.3 — Where the output lives

The [primitive library reference](https://maneex.github.io/tensorspine/primitive-library/) is generated from the primitive library
at site build and never committed. A hand-written guide to primitive library organisation should link to the
generated reference for lists rather than copy them: a list maintained by hand drifts.

<a id="4--what-this-change-contains"></a>

## 4 — What this change contains

- `schemas/tensorspine-documentation.schema.json` — the documentation model.
- `tools/document.py`, `tools/tensorspine --document` — the generator.
- **The primitive library, documented in full.** Units and their elements carry the applicable summaries,
  descriptions, enum descriptions and external references. The generated reference reports
  documentation coverage and omissions directly from those fields.
- [Primitive Library reference](https://maneex.github.io/tensorspine/primitive-library/) — the rendered primitive library,
  generated only at site build.

Rendering the primitive library surfaced guards that cited undeclared arguments. The primitive library loader resolves
every argument path a condition cites, so that class of defect is now a load error.

`--validate` and `--lint` give the same results before and after: the documentation fields are
inert, as §10.2 requires.

<a id="5--authoring-guidance"></a>

## 5 — Authoring guidance

1. **Write to the field's primitive.** A summary is one line under 120 characters and is not
   CommonMark. A description does not restate a type, a default, a shape or a evolution; it says what
   the unit computes, when to choose it over its neighbours, and what its arguments change.
2. **Cite what introduced the primitive**, not what happens to use it: the paper as `paper`, the
   `modeling_x.py` as `reference_implementation`. When no canonical source exists, cite nothing.
3. **Keep `note` for the why.** A maintainer's aside about a decision stays a note; do not fold it
   into the description, and do not duplicate it there.
4. **Tag across namespaces, not along them.** `attention.*` needs no tag; "carries state" does.
5. **Regenerate and read the appendices.** A new unit without a summary shows up in coverage; a key
   with a typo shows up in findings.

<a id="6--open-questions-and-follow-ups"></a>

## 6 — Open questions and follow-ups

- **Lint.** Advisory rules fall out of the generator and belong in `--lint` so they run with
  the rest of the hygiene checks: a unit without a summary, a cited tag no base declares, a
  condition citing an undeclared argument.
- **`--document model`.** The same renderer applied to a model definition — quantities, instances
  with their resolved arguments, bindings, interfaces — would give the tutorial material the
  documentation plan asks for, from the corpus itself. The `--document WHAT` form leaves room for
  it.
- **One language or two.** If descriptions are ever wanted in two languages, `summary` and
  `description` become language maps; everything else is unchanged.
