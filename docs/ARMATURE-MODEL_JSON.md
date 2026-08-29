# Armature model JSON — practical format guide

> Represent every model as a **finite graph of parameterized primitive occurrences**, then derive
> everything else from their contracts.

*Armature model schema 2.0 — revised 28 August 2026.*

This is the practical, non-normative guide to reading and authoring an `armature/2.0` JSON model.
The [JSON Schema](../schemas/armature.schema.json) defines the concrete grammar, while the
[language specification](SPECIFICATION.md) is the sole normative authority for validity and
denotation. The [README](../README.md) provides motivation and repository orientation; the
[architecture guide](ARCHITECTURE.md) explains the design rationale; and the
[glossary](GLOSSARY.md) provides a single terminology index. If this guide conflicts with the
specification, the specification wins.

---

## §1 — How to use this guide

An Armature model **is a graph**. Its nodes are occurrences of primitives; its edges are explicit
bindings. State behaviour, parameter inventories, port shapes, logical costs and legal semantic
partitions are consequences of primitive contracts applied to each occurrence's arguments.

> **The model declares causes. Primitive contracts derive consequences.**

This guide follows the JSON fields from the model document through contracts, validation, and
derived products. It summarizes the governing principle only to orient the field descriptions; the
specification defines the language's scope, required properties, and semantics. Terms such as
*occurrence*, *composition*, *contract*, and *liveness* link back to one lookup point in the
glossary.

---

## §2 — The Armature 2.0 model document

The schema requires nine top-level fields, plus an optional `version` — the document's version as a
representation, which a template must carry (§4.6). Fixed-shape objects use
`additionalProperties: false`, so unknown fields are rejected rather than ignored; maps such as
`quantities` and `occurrences` deliberately accept user-named entries that match their value schema.

```json
{
  "schema": "armature/2.0",
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
| `schema` | Must be exactly `armature/2.0`. |
| `model` | Stable, authoritative model identifier. |
| `catalog` | One or more catalog bases, consulted in order. There is no global catalog version. |
| `quantities` | Typed scalar facts, variables and derivations. |
| `constants` | Non-learned numeric tensors or buffers, identified by content. |
| `occurrences` | Root graph nodes. |
| `compositions` | Finite indexed families of generated occurrences. |
| `bindings` | Value edges and the identities of parameters, constants and states. |
| `interfaces` | Public inputs and outputs with explicit indexing domains. |

### 2.1 — Quantities and expressions

Quantities occupy one flat namespace. Each quantity declares three independent facts:

- a `regime`: `model_constant` or `model_variable`;
- a type: `cardinality`, `real`, `physical`, `enum` or `boolean`;
- a source: `literal`, `external` or `derived`.

A `model_constant` may be literal or derived. A `model_variable` must declare an interval or set
domain and may be external or derived. This distinction makes templates explicit: an external
quantity is an input to graph construction, not a runtime load variable such as batch size.

Expressions are tagged unions, never ambiguous strings. A scalar expression is one of:

- `{"literal": ...}`, `{"quantity": ...}`, `{"index": ...}` or `{"context": ...}`;
- a unary, binary or n-ary operation with `op` and `args`;
- a conditional with `if`, `then` and `else`;
- a normative interface call with a pinned contract, result status and provenance.

The operator set is closed. It includes addition, multiplication, subtraction, division, ceiling
and floor division, modulo, minimum, maximum, negation and absolute value. Conditions use explicit
boolean composition and comparisons.

Every derivation carries an epistemic `status` (`exact`, `upper_bound`, `lower_bound` or `estimate`)
and a non-empty `provenance` list. Bounds used for state liveness and visit counts may be exact or
upper bounds only.

### 2.2 — External constants

`constants` describes non-learned numeric data without placing tensor contents inline. Each entry
has:

- a SHA-256 content digest and an optional URI;
- a named-axis shape, including factors when an axis has been flattened;
- a dtype, written directly or selected by a quantity;
- an optional multiplicity expression.

An occurrence consumes such data through a contract constant slot, which is connected by a
`bindings.constants` entry. Learned weights belong to parameter bindings instead.

### 2.3 — Occurrences

Every occurrence has a stable identifier and requires:

- a contract reference containing `name` and semantic `version`;
- a complete argument map after declared contract defaults are applied;
- at least one addressable `family`.

An occurrence may also provide logical dtype selections and a model-level `when` condition, which
controls whether the site exists after expansion. It never contains code, a kernel name, a
parameter inventory, state descriptors or port connections. Model `when`, contract `present_when`,
and contract-rule `when` have distinct contexts; see [Conditions](GLOSSARY.md#conditions).

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

Ranges must resolve to finite integer sequences. Bindings refer to generated nodes through a
selector containing the composition name, local occurrence name and index assignment. The canonical
D1 identifier is `<composition>/<occurrence>[<index>=<value>,…]`.

Compositions are syntactic sugar. Every validity rule applies after expansion, and expansion must
produce the same unique identifiers, nodes and edges on every run. Reusable parameterized submodels
are represented separately through template contracts, not nested composition syntax.

### 2.5 — Bindings

`bindings` contains four required maps:

| Map | Meaning |
|---|---|
| `values` | A directed edge from one output port to one input port. |
| `parameters` | A logical tensor identity and the contract parameter slots it satisfies. Multiple members express weight tying. |
| `constants` | A top-level constant and the contract constant slots that consume it. |
| `states` | A persistent-storage identity and every state port that shares it. |

Bindings may use `for_each` and model-level `when` to describe regular families. Root and generated
occurrence selectors are explicit; flow is never inferred from ordering or mutation of a named
residual.

A state binding carries graph-level facts that no primitive can derive:

- its member state ports;
- key axes and the equality relation that defines sharing;
- liveness classes and an exact or upper-bounded maximum number of active classes;
- visit bounds by execution unit and phase;
- optionally, what survives an invocation boundary and in which domain.

These fields do not duplicate the contract's state descriptor. The contract defines payload,
conditional presence, growth law, access geometry and permitted operations; the binding defines
identity, graph topology and lifetime.

After composition expansion, evaluation of model `when` conditions, and resolution of contract
`present_when` guards, bindings must be total and unique. Every required input, parameter slot,
constant slot and state slot must be accounted for exactly once, except where one declared identity
intentionally has several members.

### 2.6 — Public interfaces

Every public input selects a destination value port and declares an indexing domain. Every public
output selects a source value port, declares its indexing domain and states whether it is
`generative`.

The schema recognises `sequence`, `token`, `position`, `patch` and `fragment` domains, each with a
named source. Multiple inputs and outputs are allowed, including non-generative and per-token
outputs. An interface is an additional reference to an existing graph value, not an invented
operation.

---

## §3 — Catalog contracts

`catalog` is an ordered list of bases. Each occurrence independently pins a contract by
`{name, version}`; there is no catalog-wide version whose meaning has to be coordinated across
all primitives.

A complete primitive contract provides the consequences needed to interpret an occurrence:

| Contract element | Content |
|---|---|
| **Arguments** | Types, required fields, explicit defaults, invariants and structural arguments. |
| **Value ports** | Typed inputs and outputs, shapes, roles and indexing domains. |
| **Parameters and constants** | Conditional logical slots, shapes, precision roles and sharing rules. |
| **State ports** | Conditional presence, payload components, key axes and ordered derivation rules. |
| **Effects** | The values read and written and any permitted aliasing. |
| **Logical cost** | Logical operations and bytes read or written, never executed FLOPs. |
| **Semantic partitions** | Axes along which partitioning preserves meaning and the resulting logical communication. |
| **Domain transforms** | Explicit relationships between different indexing domains. |

Every primitive has a contract, including embeddings, feed-forward blocks, mixtures of experts,
patch embeddings, projectors, residual operations, poolers and output heads. A primitive needs a
state contract only when it exposes persistent state.

Contract defaults are not silent implementation defaults. For example, the dense-attention
contract explicitly defines `kv_heads` to default to `heads`. The default is versioned,
inspectable and applied before argument validation.

A contract element uses `present_when` when its existence depends on resolved arguments. An ordered
contract rule instead uses `when` to say when that rule applies. Neither field is the model-level
`when` used during graph expansion; see [Conditions](GLOSSARY.md#conditions).

### 3.1 — State is split between contract and graph

For each state port, a contract may derive:

- conditional presence in either direction: an argument may add or remove the state;
- payload components, shapes, dtypes and multiplicity;
- evolution law, such as append, bounded window or fixed size;
- the indexing source relative to which it grows;
- access geometry, sharing capability and permitted operations;
- modulators such as window span, rank, depth or stream boundary behaviour.

The graph then supplies how many occurrences exist and which state ports name the same storage:

```text
state slots       = contract(primitive, arguments) × expanded occurrences
state allocations = equivalence classes induced by state bindings
```

This distinction matters. A cross-attention contract can know that its cache is indexed by its
source, but it cannot know which encoder value is wired to that source. Likewise, it can define a
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

The model **does** declare logical dtypes where the schema permits them, explicit value flow,
parameter and state identities, state liveness, invocation boundaries and public indexing domains.
Those are model-specific facts, not consequences of a primitive in isolation.

---

## §5 — Validation, expansion and derived products

From the repository root, the current entry point is:

```sh
python3 tools/armature --validate
python3 tools/armature --lint
python3 tools/armature --d1 data/models/llama3-8b.json -o /tmp/llama3-8b.d1.json
```

`--validate` checks the model schema and then performs semantic validation: catalog resolution,
arguments and defaults, types (V3: enums, cardinalities, reals, booleans and records, recursively,
after contract defaults are applied; an inapplicable field is refused, not ignored), shapes,
domains, total bindings and value-graph acyclicity. The catalog itself is read against
`schemas/armature-catalog-unit.schema.json` when loaded, so its vocabulary is closed by grammar,
not by convention; `tests/run_rejections.py` holds one document or catalog base per required
rejection case. A validation failure is a reasoned refusal. `--lint` reports advisory findings and
deliberately exits successfully.

The template needs an assignment when validated on its own:

```sh
python3 tools/armature --validate data/models/decoder-causal-yarn.json \
  --assign '{"width":3072,"layers":26,"heads":32,"kv_heads":8,"head_dim":128,"inner":9216,"eps":0.00001,"precision":"bf16"}'
```

As of this revision, the eleven concrete corpus documents validate as written. The twelfth document,
`decoder-causal-yarn.json`, is schema-valid and also passes semantic validation with the assignment
above.

The language defines six derived products:

| Product | Content | Current repository support |
|---|---|---|
| **D1** | Expanded occurrences, value edges and families | Emitted by `--d1` |
| **D2** | Values, shapes and value liveness at graph cuts | Specified, not yet emitted |
| **D3** | Parameter tensors, roles, shapes and sharing | Specified; validation resolves slots and identities |
| **D4** | Complete state descriptors, instances, keys, state liveness and operations | Specified; validation resolves slots and identities |
| **D5** | Logical costs and cut traffic | Specified, not yet emitted |
| **D6** | Legal cuts and semantic partition axes | Specified, not yet emitted |

A valid document must reject unresolved references, missing required arguments, undeclared
arguments, invalid enum values, incompatible shapes or domains, combinational value cycles,
unbound slots, incompatible state identities and unresolvable repetition ranges. Unknown input is
never accepted on the assumption that it does not matter.

---

## §6 — Coverage cases in the current corpus

Four models exercise topology that a homogeneous decoder does not:

| Model | Property exercised | Armature 2.0 representation |
|---|---|---|
| **Whisper large-v3** | Cross-attention reads a different trunk | An explicit encoder-to-decoder value edge and `source` argument; the contract derives KV state indexed by the source and frozen once that source is complete. |
| **Gemma 3n** | Non-adjacent layers share cache storage | State bindings merge 30 expanded state slots into 20 identities; shared identities use session and branch key axes but no layer key. |
| **Voxtral Realtime** | State survives fragmented input invocations | The encoder state binding retains all keys across the `fragment` domain sourced from `audio`. |
| **ColBERT v2** | Per-token, non-generative output | A token-domain output with `generative: false`; the expanded model has no state slots. |

These cases expose three distinctions that a state growth law alone cannot capture:

1. **Growth needs an indexing source.** `append` is ambiguous unless the contract states whether it
   follows the current sequence or a named source.
2. **State presence is conditional in both directions.** Plain non-causal self-attention is
   stateless, while a named cross-attention source or fragmented streaming mode can require state.
3. **Persistence across invocations is a graph fact.** `invocation_boundary` records what survives
   and the domain over which it is carried; it is not folded into a runtime-specific cache type.

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

Compatible catalog extensions do not require a new `armature/2.x` model-language version. They do
have to preserve every previously published contract identity: an existing `{name, version}` pair
must never change meaning. Publish changed contract contents under a new contract version; if an
existing argument changes meaning, a new contract identity is mandatory.

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
   cross-attention with a named source and fragmented streaming attention do.
2. **State-port names come from contracts.** A model cannot invent aliases such as `compressed` or
   `window` when the contract exposes `kv`, `sliding` or `index`; sharing names storage through
   bindings instead.
3. **A growth law needs a frame of reference.** `append` alone does not reveal which sequence drives
   growth, so state budgeting requires the contract's indexing source.
4. **Structural arguments belong at the occurrence.** Window span, recurrent depth, stride
   and similar causes cannot live only in hand-written consequences if state and parameter slots are
   to be derived.
5. **Topology and state semantics are separate authorities.** Contracts describe what one state
   slot means; bindings describe which occurrences share it, how many live identities exist and what
   survives an invocation boundary.

That separation is the point of Armature 2.0: a model remains a compact declaration of structure,
while every reusable consequence has one versioned, inspectable source of truth.
