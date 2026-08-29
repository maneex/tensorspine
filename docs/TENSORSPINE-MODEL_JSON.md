# Tensorspine model JSON — practical format guide

> Represent every model as a **finite graph of parameterized primitive occurrences**, then derive
> everything else from their contracts.

*Tensorspine model schema 2.0 — revised 29 August 2026.*

This is the practical, non-normative guide to reading and authoring a `tensorspine/2.0` JSON model.
The [JSON Schema](../schemas/tensorspine.schema.json) defines the concrete grammar, while the
[language specification](SPECIFICATION.md) is the sole normative authority for validity and
denotation. Human-authored models may instead use [TSPL 1.0](TENSORSPINE_MODEL_TSPL.md), whose
compiler emits this JSON format without applying semantic rewrites. The [README](../README.md)
provides motivation and repository orientation; the
[architecture guide](ARCHITECTURE.md) explains the design rationale; and the
[glossary](GLOSSARY.md) provides a single terminology index. If this guide conflicts with the
specification, the specification wins.

---

## §1 — How to use this guide

A Tensorspine model **is a graph**. Its nodes are occurrences of primitives; its edges are explicit
bindings. State behaviour, parameter inventories, port shapes, logical costs and legal semantic
partitions are consequences of primitive contracts applied to each occurrence's arguments.

> **The model declares causes. Primitive contracts derive consequences.**

This guide follows the JSON fields from the model document through contracts, validation, and
derived products. It summarizes the governing principle only to orient the field descriptions; the
specification defines the language's scope, required properties, and semantics. Terms such as
*occurrence*, *composition*, *contract*, and *liveness* link back to one lookup point in the
glossary.

---

## §2 — The Tensorspine 2.0 model document

The schema requires nine top-level fields, plus an optional `version` — the document's version as a
representation, which a template must carry (§4.6). Fixed-shape objects use
`additionalProperties: false`, so unknown fields are rejected rather than ignored; maps such as
`quantities` and `occurrences` deliberately accept user-named entries that match their value schema.

```json
{
  "schema": "tensorspine/2.0",
  "model": "authoritative-model-id",
  "catalog": [{ "base": "catalog/" }],
  "quantities": {},
  "constants": {},
  "occurrences": {},
  "compositions": {},
  "bindings": {
    "values": {},
    "parameters": {},
    "constants": {},
    "states": {}
  },
  "interfaces": {
    "inputs": {},
    "outputs": {}
  }
}
```

This is an outline, not a complete valid model: a document must contain at least one root occurrence
or composition, and must expose at least one public input and one public output.

| Field | Purpose |
|---|---|
| `schema` | Must be exactly `tensorspine/2.0`. |
| `model` | Stable, authoritative model identifier. |
| `catalog` | One or more catalog bases; one identity carried by two bases with different contents is a conflict (V1). There is no global catalog version. |
| `quantities` | Typed scalar facts, variables and derivations. |
| `constants` | Non-learned numeric tensors or buffers, identified by content. |
| `occurrences` | Root graph nodes. |
| `compositions` | Finite indexed families of generated occurrences. |
| `bindings` | Value edges and the identities of parameters, constants and states. |
| `interfaces` | Public inputs — the ports they feed, their kind, stream and fragmentation — and public outputs. |

### 2.1 — Quantities and expressions

Quantities occupy one flat namespace. Each quantity declares three independent facts:

- a `regime`: `model_constant` or `model_variable`;
- a type: `cardinality`, `real`, `physical`, `enum` or `boolean`;
- a source: `literal`, `external` or `derived`.

A `model_constant` may be literal or derived. A `model_variable` must declare an interval or set
domain and may be external or derived. This distinction makes templates explicit: an external
quantity is an input to graph construction, not a runtime load variable such as batch size.

Expressions are tagged unions, never ambiguous strings. A scalar expression is one of:

- `{"literal": ...}`, `{"quantity": ...}` or `{"index": ...}`;
- a unary, binary or n-ary operation with `op` and `args`;
- a conditional with `if`, `then` and `else`.

The operator set is closed. It includes addition, multiplication, subtraction, division, ceiling
and floor division, modulo, minimum, maximum, negation and absolute value. Conditions use explicit
boolean composition and comparisons.

A `derived` quantity is an expression over other quantities: it resolves in any declaration order,
acyclically (V10), and must conform to its declared type and domain (V3) — `divide` yields a real,
so a cardinality is derived with `floor_divide` or `ceil_divide`. A `literal` quantity may carry a
`derivation` too: the value read from configuration and the way it follows from the structure are
then checked against each other (V11), so `head_dim: 96` next to `d: 4096, heads: 32` is refused.
Epistemic status and provenance are not fields of a model quantity; they belong to contract costs
and to the derived products.

### 2.2 — External constants

`constants` describes non-learned numeric data without placing tensor contents inline. Each entry
has:

- a SHA-256 content digest and an optional URI;
- a shape of catalog axes with extents, including factors when an axis has been flattened (V4 compares
  axis identities and extents, never local names);
- a dtype, written directly or selected by a quantity;
- an optional multiplicity expression.

An occurrence consumes such data through a contract constant slot, which is connected by a
`bindings.constants` entry. Learned weights belong to parameter bindings instead.

### 2.3 — Occurrences

Every occurrence has a stable identifier and requires:

- a contract reference containing `name` and semantic `version`;
- a complete argument map after declared contract defaults are applied;
- at least one addressable `family`.

An occurrence may also carry a model-level `when` condition, which controls whether the site exists
after expansion; dtypes are selected on parameter and state identities (§2.5), never on occurrences. It never contains code, a kernel name, a
parameter inventory, state descriptors or port connections. Model `when`, contract `present_when`,
and contract-rule `when` have distinct contexts; see [`when` and
`present_when`](GLOSSARY.md#when-and-present_when).

Contract arguments may be scalar expressions or recursively tagged records, and every one has a
declared type in the contract — there is no opaque argument: a variant of a primitive that the
contract does not name is added under a new contract version (§7), never passed through unread.
Inline numeric tensors are deliberately excluded: structural metadata belongs in arguments, while
learned and non-learned numeric tensors have explicit identities and bindings.

### 2.4 — Compositions and deterministic expansion

A composition is a named finite family of occurrence sites. It declares:

- one or more index ranges, each with `start`, `stop` and `step` expressions;
- at least one family name;
- one or more occurrence sites.

Ranges must resolve to finite integer sequences; several ranges form a grid. A site may carry a
`when` condition over the indices and quantities, and an argument may be an `if`/`then`/`else`
expression over them, so a periodic or piecewise layer pattern is one composition over a flat layer
index rather than one site per case. A binding is emitted only where every occurrence it names is
emitted, so the guard of a site is written once, on the site — never repeated on its edges,
parameters or states:

```json
"decoder": {
  "indices": { "layer": { "start": {"literal": 0}, "stop": {"quantity": "layers"}, "step": {"literal": 1} } },
  "families": ["decoder"],
  "occurrences": {
    "attn":  { "contract": {"name": "attention.dense", "version": "1.0.0"}, "arguments": { … },
               "families": ["sequence_operator"],
               "when": { "compare": { "operator": "equal",
                         "left": { "op": "modulo", "args": [ {"index": "layer"}, {"literal": 4} ] },
                         "right": {"literal": 3} } } },
    "gdn":   { …, "when": { "compare": { "operator": "not_equal", … } } },
    …
  },
  "bindings": {
    "values": {
      "attn.norm_in": { "from": {"site": "attn_n", "port": "output"}, "to": {"site": "attn", "port": "input"} },
      "attn_n.carry": { "from": {"site": "ffn_r", "port": "output",
                                 "indices": { "layer": { "op": "subtract", "args": [ {"index": "layer"}, {"literal": 1} ] } } },
                        "to":   {"site": "attn_n", "port": "input"},
                        "when": { "compare": { "operator": "greater_or_equal", "left": {"index": "layer"}, "right": {"literal": 1} } } }
    },
    "parameters": { "attn.qkv": { "members": [ {"site": "attn", "parameter": "qkv"} ] } },
    "states":     { "attn.kv":  { "members": [ {"site": "attn", "state": "kv"} ] } }
  }
}
```

`attn.norm_in`, `attn.qkv` and `attn.kv` exist exactly where `attn` does. The carry edge keeps a
guard of its own because it states a fact of its own — the first layer has no predecessor: an index
outside the composition's ranges is a rejection (V1), never a silent omission; only an occurrence
absent *by its guard* makes a binding absent. A rule with several members is emitted where all of
them are.

A composition's own `bindings` are written against its sites and are sugar for top-level rules:
`decoder.attn.norm_in` with `for_each` = the composition's ranges, and each `site` endpoint the
generated occurrence at the current indices — overridden where `indices` says so, as in the carry
edge. A scoped parameter or state rule without a `tensor` / `identity` names it
`<composition>.<rule>`, indexed by the composition's indices. The D1 identifier of a generated
occurrence is `<composition>/<occurrence>[<index>=<value>,…]`, indices in name order; an occurrence
of a template is prefixed by its instance (`text/decoder/attn[layer=3]`). Identifiers are
representation: two expansions denote the same graph when they correspond up to occurrence renaming
(§5.2).

A `when` that cannot be decided — over an index that does not exist, or a quantity with no value —
is a rejection (V10), never false. Compositions are syntactic sugar: every validity rule applies
after expansion, and expansion is deterministic as a *set* of occurrences, edges and identities,
whatever the order of the document's members; the canonical listing is sorted. Reusable
parameterized submodels are represented separately through template contracts, not nested
composition syntax.

### 2.5 — Bindings

`bindings` contains four required maps:

| Map | Meaning |
|---|---|
| `values` | A directed edge from one output port to one input port. |
| `parameters` | A logical tensor identity, the contract parameter slots it satisfies and optionally its `dtype`. Multiple members express weight tying; tied members must be compatible — `shareable` on both sides, roles listed in each other's sharing rules, equal shapes (V15). |
| `constants` | A top-level constant and the contract constant slots that consume it. |
| `states` | A persistent-storage identity, every state port that shares it and optionally its `dtype`. |

Top-level bindings may use `for_each` and `when` to describe regular families, and a composition
carries the bindings among its own sites (§2.4). Root and generated occurrence selectors are
explicit; flow is never inferred from ordering or mutation of a named residual.

A state binding carries the graph-level facts that no primitive can derive, and only those:

- its identity, whose indices say which repetition indices distinguish allocations;
- its member state ports — several members under one identity is sharing;
- optionally, a `dtype` for its payload, admissible for every component's role (V14); absent, each
  role's default applies.

The instance key of an allocation is derived: the identity's indices times the contract's
`key_axes` (session, branch). Liveness is one class per distinct key; how many classes are active
at once, and how often a state is visited per request, are deployment intent (§10.3) supplied to
the derived products, never written in the model. A document references only its own quantities,
indices and arguments: there is no `context` namespace.

These fields do not duplicate the contract's state descriptor. The contract defines payload,
conditional presence, growth law, access geometry, key axes, permitted operations and the condition
under which the state is carried across fragments of its stream; the binding defines identity,
sharing and dtype. Whether a state survives between fragments follows from that condition and the
input's `fragmented` flag (§2.6): nothing is written twice.

After composition expansion, evaluation of model `when` conditions, and resolution of contract
`present_when` guards, bindings must be total and unique. Every required input, parameter slot,
constant slot and state slot must be accounted for exactly once, except where one declared identity
intentionally has several members; a binding whose occurrences are absent by their guards is simply
absent (§2.4), and every output port must be consumed or exposed (V13).

### 2.6 — Public interfaces

Every public input lists the value ports it feeds — one or more — and declares its indexing
domain: a `kind` (`sequence`, `token`, `position` or `patch`), and either the stream it introduces,
named after the input, or an existing stream it joins (`stream`). It may be `fragmented`: its
elements arrive over several invocations. Every public output names one source value port and
states whether it is `generative`; its domain is derived from the port, never written.

```json
"interfaces": {
  "inputs": {
    "tokens": { "to": [ { "occurrence": {"kind": "root", "occurrence": "embed"}, "port": "tokens" } ],
                "kind": "token" },
    "audio":  { "to": [ { "occurrence": {"kind": "root", "occurrence": "conv_frontend"}, "port": "frames" } ],
                "kind": "position", "fragmented": true }
  },
  "outputs": {
    "main": { "from": { "occurrence": {"kind": "root", "occurrence": "lm_head"}, "port": "logits" },
              "generative": true }
  }
}
```

An indexing domain is a pair (kind, stream), and V5 requires it to agree on every edge — a public
input is an edge like any other. Contract ports either declare a kind or `inherit` the occurrence's
own domain; a contract that legitimately changes domain declares a transform: `merge` (a projector
turns `merge_count` patches into one token-kind element of the same stream), `align`
(cross-attention reads `source_values` in another stream and answers in its own), `insert` (a splice
inserts one stream's elements into another). A generative output has kind `token`. Multiple inputs
and outputs are allowed, including non-generative and per-token outputs. An interface is an
additional reference to an existing graph value, not an invented operation.

---

## §3 — Catalog contracts

`catalog` is a list of bases. Each occurrence independently pins a contract by
`{name, version}`; there is no catalog-wide version whose meaning has to be coordinated across
all primitives.

A complete primitive contract provides the consequences needed to interpret an occurrence:

| Contract element | Content |
|---|---|
| **Arguments** | Types, required status, explicit defaults, conditional presence (`present_when`) and structural arguments. |
| **Value ports** | Typed inputs and outputs, shapes, roles and domains — a kind, or `inherit`. |
| **Parameters and constants** | Conditional logical slots, shapes, precision roles and sharing rules. |
| **State ports** | Conditional presence, payload components, key axes, ordered derivation rules and the condition under which the state is carried across fragments. |
| **Effects** | The ports read and written. |
| **Logical cost** | Derived from the parameter inventory — two operations per weight element per element of the output domain, at the activated fraction of a sparse unit — plus the contract's declared **corrections**: guarded entries, each an expression with a status, counted per `element`, `cached_position`, `sequence` or `invocation`; every entry whose condition holds contributes. Never executed FLOPs. |
| **Semantic partitions** | Axes along which partitioning preserves meaning and the resulting logical communication; every contract states at least one, `any_axis` or `none`. |
| **Sparsity** | One or more **units** for a primitive activating only some parameters per element: the slots and axis that form a unit, the policy that selects units (an argument, an input port, or the element itself), the count activated per element and the union bound per invocation. A lookup table is the limiting case: one row per element (§4.5). |
| **Domain transforms** | `merge`, `align`, `insert`: how a port's domain relates to the occurrence's own (§2.6). |

Every primitive has a contract, including embeddings, feed-forward blocks, mixtures of experts,
patch embeddings, projectors, residual operations, poolers and output heads. A primitive needs a
state contract only when it exposes persistent state.

Contract defaults are not silent implementation defaults. For example, the dense-attention
contract explicitly defines `kv_heads` to default to `heads`. The default is versioned,
inspectable and applied before argument validation.

A contract element uses `present_when` when its existence depends on resolved arguments. An ordered
contract rule instead uses `when` to say when that rule applies. Neither field is the model-level
`when` used during graph expansion; see [`when` and
`present_when`](GLOSSARY.md#when-and-present_when).

### 3.1 — State is split between contract and graph

For each state port, a contract may derive:

- conditional presence in either direction: an argument may add or remove the state;
- payload components, shapes, dtypes and multiplicity;
- evolution law, such as append, bounded window or fixed size;
- the stream along which it grows: the occurrence's own, or that of one of its input ports;
- access geometry, sharing capability and permitted operations;
- modulators — span, stride — and the condition under which it is carried across fragments of its
  stream.

The graph then supplies how many occurrences exist and which state ports name the same storage:

```text
state slots       = contract(primitive, arguments) × expanded occurrences
state allocations = equivalence classes induced by state bindings
```

This distinction matters. A cross-attention contract can know that its cache is indexed by the
stream arriving on `source_values`, but it cannot know which encoder value is wired to that port. Likewise, it can define a
shareable KV payload without knowing which non-adjacent layers actually share one identity.

### 3.2 — Implementation candidates are outside the model

Backend, guards, memory layout, fusion, workspace, algorithms, physical traffic, supported kernel
partitions and actual collectives belong to an implementation candidate, not to the model or its
logical contract. Two implementations may have different physical costs while denoting the same
expanded logical graph.

---

## §4 — What the model does not declare

| Excluded fact | Where it belongs |
|---|---|
| State payload descriptors, evolution laws, access geometry and permitted operations | Primitive contract |
| Port and logical-slot shapes that follow from primitive arguments | Primitive contract |
| Logical operation count and logical memory traffic | Primitive contract |
| Kernel, backend, fusion, workspace, physical layout and executed FLOPs | Implementation candidate |
| Hardware placement, topology and resolved sharding plan | Compilation or deployment control |
| Batch size, active sequence count and admission policy | Deployment intent or online control |
| Cache pages, block tables and other runtime data structures | Runtime implementation |

The model **does** declare dtypes on parameter and state identities, explicit value flow, parameter
and state identities, and public inputs with their kind, stream and fragmentation.
Those are model-specific facts, not consequences of a primitive in isolation.

---

## §5 — Validation, expansion and derived products

From the repository root, the current entry point is:

```sh
python3 tools/tensorspine --validate
python3 tools/tensorspine --lint
python3 tools/tensorspine --d1 data/models/llama3-8b.json -o /tmp/llama3-8b.d1.json
```

`--validate` checks the model schema and then performs semantic validation: catalog resolution,
arguments and defaults, types (V3: enums, cardinalities, reals, booleans and records, recursively,
after contract defaults are applied; an inapplicable field is refused, not ignored), shapes,
domains, total bindings and value-graph acyclicity. The catalog itself is read against
`schemas/tensorspine-catalog-unit.schema.json` when loaded, so its vocabulary is closed by grammar,
not by convention; `tests/run_rejections.py` holds one document or catalog base per required
rejection case. A validation failure is a reasoned refusal. `--lint` reports advisory findings and
deliberately exits successfully.

The template needs an assignment when validated on its own:

```sh
python3 tools/tensorspine --validate data/models/decoder-causal-yarn/1.0.0.json \
  --assign '{"width":3072,"layers":26,"heads":32,"kv_heads":8,"head_dim":128,"inner":9216,"eps":0.00001,"precision":"bf16"}'
```

As of this revision, the eleven concrete corpus documents validate as written against the grammar
and the rules of the specification (the tools that check them are being brought to this revision). The template,
`decoder-causal-yarn/1.0.0.json`, is schema-valid and also passes semantic validation with the
assignment above; the catalog manifest says where templates live (`templates`), one immutable file
per version.

The language defines six derived products:

| Product | Content | Current repository support |
|---|---|---|
| **D1** | Expanded occurrences, value edges and families | Emitted by `--d1` |
| **D2** | Values, shapes and the payload of every legal cut | Specified, not yet emitted |
| **D3** | Parameter tensors, roles, selected dtypes, sensitivity, shapes and sharing | Specified; validation resolves slots, identities and dtypes |
| **D4** | Complete state descriptors, instances, derived keys, state liveness, visits per phase, carrying and operations | Specified; validation resolves slots, identities and carrying |
| **D5** | Logical costs and cut payloads | Validation derives parameter elements, operations per element and per cached position (`--validate` stats); cut payloads not yet |
| **D6** | Legal cuts and semantic partition axes | Specified, not yet emitted |

A valid document must reject unresolved references, missing required arguments, undeclared
arguments, invalid enum values, incompatible shapes or domains, combinational value cycles, unfed or
twice-fed ports, unbound or twice-bound slots, dangling outputs, inadmissible dtypes, incompatible
tied members, a carried state on a stream that is not fragmented, incompatible state identities and
unresolvable repetition ranges (V1–V16). Unknown input is never accepted on the assumption that it
does not matter.

---

## §6 — Coverage cases in the current corpus

Four models exercise topology that a homogeneous decoder does not:

| Model | Property exercised | Tensorspine 2.0 representation |
|---|---|---|
| **Whisper large-v3** | Cross-attention reads a different trunk | `cross: true` and an explicit encoder-to-decoder edge into `source_values`; the contract derives KV state indexed by that port's stream (`audio`, position kind) and frozen once it is complete. |
| **Gemma 3n** | Non-adjacent layers share cache storage | State bindings merge 30 expanded state slots into 20 identities; shared identities use session and branch key axes but no layer key. |
| **Voxtral Realtime** | State survives fragmented input invocations | The `audio` input is `fragmented` and the encoder attention has `streaming: true`, under which the contract carries its KV across fragments; the pairing is checked (V16), and nothing is declared on the binding. |
| **ColBERT v2** | Per-token, non-generative output | A token-domain output with `generative: false`; the expanded model has no state slots. |

These cases expose three distinctions that a state growth law alone cannot capture:

1. **Growth needs a stream.** `append` is ambiguous unless the contract states whether it follows
   the occurrence's own stream or the one arriving on an input port.
2. **State presence is conditional in both directions.** Plain non-causal self-attention is
   stateless, while cross-attention (`cross: true`) or streaming mode can require state.
3. **Persistence across fragments is derived, not declared.** The input says it is fragmented; the
   contract says under which arguments the state is carried; the boundary follows, and is never
   folded into a runtime-specific cache type.

---

## §7 — Extension, identity and rejection

Extensions affect existing documents and consumers differently:

| Extension | Existing documents | Consumer consequence |
|---|---|---|
| **New primitive** | No breakage | New capability when a model uses it |
| **New optional argument or argument with a declared default** | No breakage | New capability when used |
| **New value of a closed derived property** | No breakage | New explicitly rejectable strategy |
| **Template contract** | No breakage | No new capability if every contract in its expanded template is already supported |

“No breakage” does not mean “free”. A consumer still has to implement any new vocabulary that a
model actually uses. Delegation is the only case that can genuinely reuse existing capabilities
without requiring a new primitive implementation.

Compatible catalog extensions do not require a new `tensorspine/2.x` model-language version. They do
have to preserve every previously published contract identity: an existing `{name, version}` pair
is one immutable file that never changes meaning. Every change is a new version file — patch when
no product changes, minor when it only adds, major when an existing occurrence would mean something
else (§8.2) — and a pin is always exact.

Rejection and identity solve different problems:

- an unknown primitive, field, argument, value or combination is rejected with a reason;
- a missing required argument is also rejected;
- a pinned contract version identifies the exact meaning the author intended.

Without exhaustive rejection, a consumer could silently discard information it does not understand.
Without immutable contract identities, an older consumer could accept familiar syntax while applying
obsolete semantics.

---

## §8 — Lessons from derivation

Deriving consequences from contracts has already uncovered several facts that hand-written state
blocks obscured:

1. **Non-causal attention is not one case.** Batch self-attention with `mask: none` has no KV state;
   cross-attention (`cross: true`) and streaming attention do.
2. **State-port names come from contracts.** A model cannot invent aliases such as `compressed` or
   `window` when the contract exposes `kv`, `sliding` or `index`; sharing names storage through
   bindings instead.
3. **A growth law needs a frame of reference.** `append` alone does not reveal which stream drives
   growth, so state budgeting requires the contract to say: its own, or an input port's.
4. **Structural arguments belong at the occurrence.** Window span, recurrent depth, stride
   and similar causes cannot live only in hand-written consequences if state and parameter slots are
   to be derived.
5. **Topology and state semantics are separate authorities.** Contracts describe what one state
   slot means, including when it is carried across fragments; bindings describe which occurrences
   share it and how many live identities exist; inputs say whether they are fragmented.

That separation is the point of Tensorspine 2.0: a model remains a compact declaration of structure,
while every reusable consequence has one versioned, inspectable source of truth.
