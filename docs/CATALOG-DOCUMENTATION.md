# Catalog documentation model — proposal

> Give every catalog unit a fixed place to say what it is, for people, without letting that prose
> change what the unit means; then render the whole catalog from its definitions and that prose
> with `armature --document catalog`.

*Proposal, 28 August 2026. Companion to `schemas/armature-documentation.schema.json`,
`tools/document.py` and the generated [catalog reference](CATALOG-REFERENCE.md). Non-normative:
the [specification](SPECIFICATION.md) remains the authority on what a catalog unit means.*

---

## 1 — What is being proposed

Two things, one depending on the other:

1. **A documentation model**: a closed set of documentation fields that a contract, an axis, a
   precision role and a base manifest may carry, with a fixed shape for each, expressed as a JSON
   Schema (`schemas/armature-documentation.schema.json`). It is modelled on the documentation
   constructs of OpenAPI 3.1 — `info`, `summary`/`description`, `externalDocs`, `tags`,
   `deprecated`, `examples` — with the same rule OpenAPI and JSON Schema both apply:
   **documentation never changes what a document denotes.**
2. **A generator**: `armature --document catalog` reads the catalog and the corpus and writes one
   Markdown file that contains *everything* the catalog says — every argument, port, slot, state
   rule, cost and partition of every contract, rendered in readable notation — plus the
   documentation fields where they exist, plus cross-references no single unit can know (which
   models cite a contract, which slots share a precision role, which enumeration values are in
   use).

The generator is useful today, on a catalog that has almost no documentation fields, because the
definitions already carry most of what a reader needs. The documentation model is what turns that
rendering from a *listing* into a *reference*: the one-line summary in an index, the paragraph that
says when to use `attention.dense` rather than `attention.latent_compressed`, the paper that
introduced the primitive, the worked example.

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

| OpenAPI 3.1 | Armature catalog | Notes |
|---|---|---|
| `info` (`title`, `summary`, `description`, `contact`, `license`) | Base manifest `catalog.json`: `title`, `summary`, `description`, `contact`, `license` | `info.version` is deliberately absent: a catalog has no global version (§8.2). |
| `tags` (declared at the top, cited by operations) | Base manifest `tags: [{name, summary, description?, external_docs?}]`; units cite by name | Namespaces (`attention.*`) already group by structure; a tag groups by a property that cuts across them. |
| `externalDocs` (one object) | `external_docs` (a list of `{url, title?, description?, kind?}`) | A primitive routinely has several sources: the paper, the reference implementation, the discussion that fixed a convention. `kind` lets the renderer group them. |
| Operation `summary`, `description` | Unit-level `summary` (one line, ≤ 120 characters, plain text) and `description` (CommonMark) | On contracts, axes, precision roles and the base. |
| Operation `deprecated: true` | `deprecated: {reason, superseded_by?: {name, version}}` | An object rather than a boolean: the useful fact is what to use instead, and the renderer links it. |
| Parameter `description`, `deprecated` | `description` and `deprecated: {reason, superseded_by?: <argument>}` on every argument and record field | |
| Schema `description` on every property | `description` on every port, parameter slot, constant slot, state port, payload component, state operation, state rule, partition, cost entry, domain transform, alias rule | One field per site; the site's other facts are already declared. |
| `x-enum-descriptions` (vendor extension, no standard) | `value_descriptions: {value: text}` on an enum argument | Keys must be declared values; checked by the generator. |
| Example Object (`summary`, `description`, `value`) | `examples: [{name, summary?, description?, arguments, model?}]` on a contract | **Executed**, not displayed: the renderer applies the contract and prints the consequences. |
| `x-*` specification extensions | — | Rejected: the vocabulary is closed (O0.6). |

## 2 — The documentation model

### 2.1 — Where the fields live

Documentation fields sit **inside `definition`, beside `note`**, at the site they document. A
contract's `summary` is a sibling of its `version`; an argument's `description` is a sibling of its
`type`. Nothing moves; nothing is wrapped. The existing `note` field keeps its role.

| Unit kind | Unit-level fields | Element-level fields |
|---|---|---|
| Contract (primitive) | `summary`, `description`, `external_docs`, `tags`, `deprecated`, `examples` | `description` on every argument, record field, port, parameter slot, constant slot, state port, payload component, state operation, state rule, partition, cost entry, domain transform, alias rule; `value_descriptions` and `deprecated` on arguments |
| Contract (delegated body) | same unit-level fields | none: its elements are derived from the body |
| Axis | `summary`, `description`, `external_docs`, `tags`, `deprecated` | — |
| Precision role | `summary`, `description`, `external_docs`, `tags`, `deprecated` | — |
| Base manifest | `title`, `summary`, `description`, `contact`, `license`, `external_docs`, `tags` (declarations) | — |

Shape axes (`{name, axis, nature, extent}`) carry no documentation: the local name and the axis
identity are self-explanatory, and the axis unit carries the prose.

### 2.2 — The fields

The schema is `schemas/armature-documentation.schema.json` (`$id`
`https://armature.dev/schema/2.0/documentation.json`). Its `$defs` are the authority; this table is
the reading guide.

| Field | Shape | Rule |
|---|---|---|
| `summary` | string, one line, ≤ 120 characters | What the unit *is*. Plain text: it lands in index rows and table cells. |
| `description` | string, CommonMark | What the unit *means* and how to use it. Never restates a fact the definition carries. |
| `external_docs` | `[{url, title?, description?, kind?}]`, `kind ∈ {paper, reference_implementation, specification, discussion, other}` | `url` is absolute, or a path relative to the repository root (rewritten by the renderer relative to the output file). |
| `tags` | `[tag_name]`, unique | A tag a unit cites should be declared by a base manifest; an undeclared tag is an advisory finding. |
| `deprecated` | `{reason, superseded_by?}` | On a unit: `superseded_by` is `{name, version}`. On an argument: another argument's name. |
| `examples` | `[{name, summary?, description?, arguments, model?}]` | `arguments` is a *literal* argument map — JSON scalars, objects for records — checked like an occurrence (V2, V3), defaults applied, consequences rendered. `model` names the corpus document the example comes from. |
| `value_descriptions` | `{value: description}` | Only on an enum-typed argument; every key must be a declared value. |
| `title`, `contact`, `license` | as OpenAPI `info` | Base manifest only. |
| `tags` (base) | `[{name, summary, description?, external_docs?}]` | Declarations, as OpenAPI's top-level `tags`. |

### 2.3 — `note` and `description` are different fields

The catalog already carries 79 `note` fields. They are **maintainers' asides**: why a default is
what it is, why a fused axis has no factors, why a port is optional. That is the *why*, written by
and for the people who curate the catalog. A `description` is the *what*, written for the people
who write models against the catalog and the people who implement runtimes from it.

Both are kept and both are rendered — the description as prose, the note quoted as a
"Note (maintainers)". Nothing is migrated automatically. Some unit-level notes do read like
summaries (`"RMSNorm. eps affects no template"`, `"Gated FFN: gate and up projections…"`); promoting
those to `summary` is an editorial pass, unit by unit, not a rule the generator applies (I7).

### 2.4 — Examples are executed

An OpenAPI example is displayed. An Armature example is **applied**: the renderer resolves the
argument map against the contract exactly as `--validate` resolves an occurrence — unknown
argument refused, missing required argument refused, enum value checked, defaults applied in
dependency order — and then prints what follows:

- which parameter slots exist for these arguments, with numeric shapes and element counts;
- which state ports exist, **which derivation rule fires** (the first whose condition holds), the
  payload per indexed position, and the bytes per position at the roles' default dtypes;
- the logical cost, evaluated.

An example the contract rejects is a documentation error and a refusal: the command exits 1 and
writes nothing. This is what keeps examples true — the same reason OpenAPI validators check that
`example` values conform to their schema.

The three examples on `attention.dense@1.0.0` are taken from the corpus (`llama3-8b`,
`whisper-large-v3`, `voxtral-realtime`) and show three different rules firing: plain causal
append, cross-attention indexed by the source and frozen once it is complete, and a ring of 750
positions where the window rule wins over the streaming rule. See the
[rendered section](CATALOG-REFERENCE.md#contract-attention.dense-1.0.0).

### 2.5 — Integration into the catalog grammar

The unit envelope schema, `schemas/armature-catalog-unit.schema.json`, has just been added to the
tree; the body grammar it references, `https://armature.dev/schema/2.0/catalog.json`, is still
absent. The copy in `temp/backup/` validates every live unit unchanged, so reinstating it is a copy.
Once it is there, the documentation model plugs into it by `$ref` — one property added per site, no
new site. The `additionalProperties: false` of each definition then rejects any documentation key
the model does not know, which is the closure O0.6 asks for.

| Catalog schema `$def` | Properties to add | From `documentation.json#/$defs/…` |
|---|---|---|
| `contract_definition`, `composite_contract_definition` | `summary`, `description`, `external_docs`, `tags`, `deprecated`, `examples` | `contract_documentation` (its `properties`) |
| `axis_definition`, `precision_rule` | `summary`, `description`, `external_docs`, `tags`, `deprecated` | `unit_documentation` |
| catalog-unit `base_definition` | `title`, `summary`, `description`, `contact`, `license`, `external_docs`, `tags` | `base_documentation` |
| `argument_definition` (and therefore record fields) | `description`, `value_descriptions`, `deprecated` | `argument_documentation` |
| `port_definition`, `parameter_definition`, `constant_slot_definition`, `state_port_definition`, `payload_component`, `state_operation`, `state_rule`, `partition_definition`, `cost_entry`, `domain_transform`, `alias_rule` | `description` | `element_documentation` |

Until then, the generator is the check: it extracts the documentation keys of every site, validates
them against the fragment above, and refuses on any error. A key that is neither grammar nor
documentation is reported as an advisory finding, not refused — the catalog schema, not the
generator, is the authority on grammar.

### 2.6 — What is deliberately not in the model

- **No `x-` extensions.** Closed vocabulary (O0.6).
- **No documentation version or `since`.** There is no catalog version to date it against; the
  file's history is in git.
- **No documentation on shape axes** (§2.1).
- **No structured "rejection conditions" prose.** `constraints` is an array of bare conditions in the
  grammar, so a constraint has nowhere to carry a `description`. Giving constraints an object form
  (`{condition, description?}`) is a grammar change, listed under open questions; no live unit
  declares a constraint today.
- **One language.** `description` is a string, not a language map. If bilingual descriptions are
  wanted later, the change is local to `$defs/description` and `summary`.

## 3 — The generator: `armature --document catalog`

```sh
python3 tools/armature --document catalog -o docs/CATALOG-REFERENCE.md   # the corpus is data/models/
python3 tools/armature --document catalog data/models/llama3-8b.json      # cross-reference one model, to stdout
python3 tools/armature --document catalog --catalog other/catalog -o out/  # writes out/catalog.md
```

- **Inputs.** The catalog bases (`--catalog`, default `data/catalog/`), the documentation schema
  (`--schemas`, default `schemas/`), and the model documents given as `PATH`s — the **corpus** used
  for cross-references only: which document cites which contract, and where the body of a delegated
  contract is (`<last URI segment>.json` in the corpus directories). No `PATH` means every document
  of `data/models/`, as for the other commands.
- **Output.** `-o FILE`, `-o DIR` (writes `DIR/catalog.md`), or stdout when `-o` is omitted; the
  status line then goes to stderr so the page can be piped. The output is **deterministic**: the
  same catalog and corpus give the same bytes, with no timestamp, so the file can be committed and
  diffed.
- **Exit status.** `0` written; `1` refused — an unreadable catalog, a malformed documentation
  field, an example the contract rejects — with every cause on stderr and nothing written. Findings
  that are legal but worth knowing (an unknown key, a tag no base declares, an example naming a
  model outside the corpus, a condition citing an argument the contract does not declare) go to the
  findings appendix and never block.

### 3.1 — What the page contains

| Section | Source | Content |
|---|---|---|
| Head | base manifest | Title, summary, description, contact, license, external docs; bases consulted; counts. |
| Contents, How to read | — | Navigation; the notation: expressions in infix, conditions in words, shapes as `[name: extent]`, template arguments, ordered rules, executed examples. |
| Overview | all units | One index table per kind: contract with summary and shape (`17 args · 2→1 ports · 9 params · state kv`), axes, precision roles, each with how many documents or slots cite it. |
| Contracts | contract units, grouped by namespace | Per contract: summary, tags, description, note, external docs, an at-a-glance row, then **Arguments** (with nested record fields and enum value descriptions), **Ports**, **Parameters** (with sharing, presence, multiplicity, declared views), **Constant slots**, **State ports** (presence, key axes, payload, permitted operations, ordered derivation rules), **Effects**, **Logical cost**, **Semantic partitions**, **Domain transforms**, **Constraints**, **Examples** (executed), **Cited by**. A delegated contract shows its body: resolved path, the body's external quantities as arguments with their domains, its public interfaces, and the transitive closure of contracts it cites — the consumer's capability cost (§8.1). |
| Axes, Precision roles | axis and role units | Tables with summary and "cited by" (every contract whose shapes, key axes or partitions use the axis; every slot, port or component with the role), then details for units with a description. |
| Tags | base manifest | Declared tags with the units citing them. |
| Corpus cross-reference | corpus | Per document: model id, kind (a parametric body lists its external quantities), contracts cited with site counts. |
| Appendix A | derived | Roles by slot: what a precision policy decides at once. |
| Appendix B | derived | Closed vocabulary in use: every value of every closed enumeration at least one unit uses (laws, access geometries, sharing, communications, natures, domains, dtypes…), with the units. A runtime that implements these values implements the catalog as it stands. |
| Appendix C | derived | Documentation coverage per site kind, and the undocumented units by name. |
| Appendix D | derived | Findings. |

### 3.2 — What it does not do

It does not expand delegated bodies (D1), nor derive parameter tensors, states, costs or
partitions from them (D3–D6): those are products of the compiler, and the page says so where a
delegated contract is rendered. It does not check units against the catalog grammar — the envelope
schema is in the tree, the body grammar it references is not yet — but it will refuse a catalog
whose unit identities disagree with their paths, as `catalog.load` does.

### 3.3 — Where the output lives

`docs/CATALOG-REFERENCE.md`, regenerated whenever a unit changes and committed, so the reference is
readable on the repository without running anything. The documentation plan reserves
`docs/CATALOG.md` for a hand-written guide to the catalog's *organisation* — resolution order, how
to add a unit, versioning. That guide should link to the generated reference for the lists rather
than carry them: a list maintained by hand drifts.

## 4 — What this change demonstrates

- `schemas/armature-documentation.schema.json` — the model, 24 `$defs`.
- `tools/document.py`, `tools/armature --document` — the generator.
- Four units documented as a specimen, one of each kind: the base manifest (`title`, `summary`,
  `description`, a `specification` external doc, one declared tag `sequence-operator`),
  `attention.dense@1.0.0` (unit summary and description, five papers, every argument, port, slot,
  state component, operation, rule and partition described, enum value descriptions for `mask`,
  three executed examples), the axis `model.width`, the precision role `state.kv`.
- `docs/CATALOG-REFERENCE.md` — the rendered catalog: 36 contracts, 37 axes, 54 precision roles,
  12 corpus documents.

Rendering the whole catalog surfaced one thing the validator does not see: in
`attention.dense@1.0.0`, the `qkv` and `out` slots are guarded by `q_rank absent` and
`o_rank absent`, and `attention.dense` declares neither argument — the guards can never fire (they
are copied from `attention.latent_compressed`). Harmless today, since an absent argument is
"absent"; the finding is in Appendix D, and it is a candidate lint rule.

`--validate` and `--lint` give the same results before and after: the documentation fields are
inert, as §10.2 requires.

## 5 — Authoring guidance and migration

1. **Contracts first.** 36 units, and the ones readers need most. For each: a `summary`, a
   `description` that says what the primitive computes and when to choose it over its
   neighbours, `external_docs` with the paper and the reference implementation, a `description`
   per argument, and one executed example taken from the corpus (with `model` set). Arguments and
   state ports matter more than ports and partitions.
2. **Then precision roles** (54) — a summary each; a description where the sensitivity is not
   obvious (the `note` on `moe.router` is a good description already).
3. **Then axes** (37) — a summary each; most are one line.
4. **Promote, do not duplicate.** Where a unit-level `note` describes the unit, move it to
   `summary` (shortened to a line) or `description`; keep it as a `note` where it explains a
   decision.
5. **Write to the field's contract.** A summary is one line under 120 characters and is not
   CommonMark. A description does not restate a type, a default, a shape or a law. An example is a
   literal argument map that the contract accepts.
6. **Regenerate and read the appendices.** Coverage shows what is left; findings show what the
   renderer could not honour.

## 6 — Open questions and follow-ups

- **Lint.** Three advisory rules fall out of the generator and belong in `--lint` so they run with
  the rest of the hygiene checks: a unit without a summary, a cited tag no base declares, a
  condition citing an undeclared argument. Example validation stays in `--document`, which
  already refuses.
- **Catalog grammar in the tree.** Reinstate `catalog.json` from `temp/backup/` beside the envelope
  schema now in `schemas/`, with the properties of §2.5 added, and give `--validate` a stage that
  checks every unit of every base against them. Until then a documentation key with a typo is an
  advisory finding, not a refusal.
- **`constraints` as objects** (`{condition, description?}`) so that rejection conditions can be
  documented. Grammar change; no live unit affected.
- **Body version pin.** §4.6 says the delegated contract names its body's version; the live
  `decoder.causal_yarn` declares `uri` and `id` only. The renderer prints the version when it is
  there. Whether the pin belongs in the catalog grammar is a specification question, not a
  documentation one.
- **`--document model`.** The same renderer applied to a model document — quantities, occurrences
  with their resolved arguments, bindings, interfaces — would give the tutorial material the
  documentation plan asks for, from the corpus itself. The `--document WHAT` form leaves room for
  it.
- **One language or two.** If descriptions are ever wanted in two languages, `summary` and
  `description` become language maps; everything else is unchanged.
