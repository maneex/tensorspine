# Tensorspine glossary

This glossary is the non-normative lookup point for Tensorspine terminology. Each entry gives a
reader-facing summary and links to the document that owns the precise definition. The
[language specification](SPECIFICATION.md) is authoritative for validity and denotation; the
[JSON Schema](../schemas/tensorspine.schema.json) is authoritative for the concrete model grammar. If
a summary here conflicts with either authority, use the authority. The [README](../README.md) gives
the project orientation, and the [model JSON guide](TENSORSPINE-MODEL_JSON.md) explains how to author
the concrete format.

Concept names are written in ordinary text. Literal JSON fields and values are written as `code`.

## A

### Argument

A scalar, enum, boolean, or record supplied by an occurrence to its primitive contract. Arguments
describe structural causes such as width, head count, or mask kind; learned and non-learned numeric
tensors are not arguments. See [Specification §3.3](SPECIFICATION.md#33--arguments-o91-o94) and
[Model guide §2.3](TENSORSPINE-MODEL_JSON.md#23--occurrences).

### Assignment

A set of values supplied for a parameterized model document's external quantities. An admissible
assignment turns the document's graph family into one concrete graph. See
[Specification §4.6](SPECIFICATION.md#46--template-contracts).

### Axis

A catalog identity for a semantic dimension, such as `model.width` or `attention.heads`. A shape
uses an axis together with an extent; flattened axes retain their factorization so semantic
partition axes are not silently lost. See
[Specification §4.1](SPECIFICATION.md#41--contract-contents-o92-semantic-part).

## B

### Binding

A graph declaration that records a model-specific relationship. Value bindings are directed edges
from an output port to an input port. Parameter, constant, and state bindings associate one identity
with the contract slots that use it. See [Specification §3.4](SPECIFICATION.md#34--bindings) and
[Model guide §2.5](TENSORSPINE-MODEL_JSON.md#25--bindings).

## C

### Catalog

The ordered vocabulary resolved by a model. It contains independently identified axes, precision
roles, and versioned contracts; it has no single global version. See
[Model guide §3](TENSORSPINE-MODEL_JSON.md#3--catalog-contracts) and
[Specification §8.2](SPECIFICATION.md#82--identity-and-versioning).

### Composition

A named, finite, indexed family of occurrence sites, with the bindings among those sites. A site
may be guarded by `when`, so one composition over a flat layer index expresses a periodic pattern.
Expansion turns sites into ordinary occurrences with deterministic identifiers and scoped bindings
into top-level rules. It is authoring shorthand, not a runtime node or a nested model. See
[Specification §5.1](SPECIFICATION.md#51--the-expanded-graph-is-authoritative) and [Model guide
§2.4](TENSORSPINE-MODEL_JSON.md#24--compositions-and-deterministic-expansion).

### Communication

The logical data movement implied when a semantic partition cuts model values. Contracts describe
this consequence independently of physical collectives, topology, or measured traffic. See
[Specification §4.1](SPECIFICATION.md#41--contract-contents-o92-semantic-part).

### Constant

A non-learned numeric tensor or buffer with explicit identity, shape, and dtype, optionally located
by a URI and identified by content digest. It is distinct from a quantity whose regime is
`model_constant`. See [Model guide §2.2](TENSORSPINE-MODEL_JSON.md#22--external-constants) and
[Specification §3](SPECIFICATION.md#3--occurrences).

### Contract

The immutable, versioned semantic definition of a primitive. An occurrence pins a contract by name
and version; the contract declares arguments and derives ports, logical tensors, state, effects,
costs, and legal partitions. It describes meaning, not a kernel or backend implementation. See
[Specification §4](SPECIFICATION.md#4--primitive-semantic-contracts) and
[Model guide §3](TENSORSPINE-MODEL_JSON.md#3--catalog-contracts).

## D

### Denotation

The graph meaning of a valid document. A concrete document denotes one finite expanded graph; a
parameterized document denotes one such graph for every admissible assignment. See
[Specification §5](SPECIFICATION.md#5--denotation).

### Domain

See [Indexing domain](#indexing-domain). A quantity's declared set or interval is instead its value
domain; it constrains admissible scalar values and does not index a port. See
[Specification §2.1](SPECIFICATION.md#21--quantities-o21-o22-o23-o04).

### Derived products (D1–D6)

The products that can be computed from a valid model and its referenced contracts:

| Product | Meaning |
|---|---|
| **D1** | Expanded occurrences, value edges, and families |
| **D2** | Values, shapes, and value liveness at graph cuts |
| **D3** | Parameter tensors, including roles, shapes, and sharing |
| **D4** | Complete state: descriptors, instances, derived keys and liveness, and operations |
| **D5** | Logical costs and cut traffic |
| **D6** | Legal cuts and semantic partition axes |

See [Specification §7](SPECIFICATION.md#7--required-derived-products).

## E

### Expression

A tagged, inspectable construction for a scalar value or condition. Model expressions read model
quantities and composition indices; contract expressions read resolved primitive arguments. The two
contexts are deliberately separate. See [Model guide
§2.1](TENSORSPINE-MODEL_JSON.md#21--quantities-and-expressions) and [Specification
§2.2](SPECIFICATION.md#22--derivation-algebra-o01-o02-o03-o05-o06).

### Extent

The expression giving the size of one named shape axis. An extent is a value; the axis supplies the
dimension's semantic identity. See [Specification
§4.1](SPECIFICATION.md#41--contract-contents-o92-semantic-part).

## F

### Family

A named, addressable grouping of occurrences. Families allow later derivations to refer to a set of
nodes without enumerating every expanded occurrence. See
[Specification §3.1](SPECIFICATION.md#31--identity-o21-o35).

## G

### Generative

An output property stating whether a public output participates in generation. Tensorspine also permits
non-generative and per-token outputs. See
[Specification §2.3](SPECIFICATION.md#23--public-inputs-and-outputs-o81-o82-o83-o42).

## I

### Identity

An explicit declaration that answers “which logical thing is this?” Contract identity pins meaning;
parameter, constant, and state identities group the slots that refer to the same logical resource;
occurrence identity names a graph node. Matching names or shapes do not implicitly create identity.
See [Specification §§3.1–3.4](SPECIFICATION.md#3--occurrences).

### Indexing domain

What indexes a value or state port, such as a sequence, token, position, patch, or stream fragment,
together with the named source of that domain. It distinguishes, for example, the current sequence
from an encoder source sequence. See
[Specification §5.3](SPECIFICATION.md#53--indexing-domains-and-invocation-boundaries).

### Interface

A public model input or output attached to an existing occurrence port. Each interface declares its
indexing domain; outputs also state whether they are generative. See
[Specification §2.3](SPECIFICATION.md#23--public-inputs-and-outputs-o81-o82-o83-o42) and
[Model guide §2.6](TENSORSPINE-MODEL_JSON.md#26--public-interfaces).

### Invocation boundary

The declaration, on a state identity, of what survives into the next invocation and the indexing
domain across which it is carried. It makes fragmented or streaming execution explicit without
introducing a value-graph cycle. See [Specification
§5.3](SPECIFICATION.md#53--indexing-domains-and-invocation-boundaries).

## L

### Liveness

The word is qualified by what is live:

- **Value liveness** identifies graph values live across a cut and therefore the cut's logical
  payload (D2).
- **State liveness** is the number of distinct state-allocation equivalence classes active at once.
  It is derived: one class per distinct instance key, the key being the identity's indices times the
  contract's `key_axes`. The count of active classes is deployment intent, not a model fact.

State liveness sizes simultaneous memory; visits size computation. See
[Specification §4.4](SPECIFICATION.md#44--information-supplied-by-the-graph).

### Model document

An `tensorspine/2.0` declaration of model identity (and, for a template, its version), catalog bases,
quantities — literal, external with domain and optional default, or derived — constants,
occurrences, compositions with their scoped bindings, bindings, and public interfaces. It declares
graph-specific causes and relationships; contracts derive reusable consequences. See [Specification
§2](SPECIFICATION.md#2--model-document) and [Model guide
§2](TENSORSPINE-MODEL_JSON.md#2--the-tensorspine-20-model-document).

## N

### Nature

A structural classification attached to a shape dimension, such as `feature`, `projection`, or
`structural`. The axis names the semantic dimension, while nature records how that dimension is used
at a particular site. See
[Specification §4.1](SPECIFICATION.md#41--contract-contents-o92-semantic-part).

## O

### Occurrence

A node in the model graph. It has a stable identity, references one versioned contract, supplies
that contract's arguments, and belongs to one or more families. It is an invocation of a primitive,
not the primitive definition itself. See [Specification §3](SPECIFICATION.md#3--occurrences) and
[Model guide §2.3](TENSORSPINE-MODEL_JSON.md#23--occurrences).

### Operator

A member of the closed scalar-expression vocabulary, such as addition, multiplication, ceiling
division, comparison, or boolean conjunction. Operators are data, not executable function bodies.
See [Specification §2.2](SPECIFICATION.md#22--derivation-algebra-o01-o02-o03-o05-o06).

## P

### Parameter tensor

A learned logical tensor whose slot, shape, and precision role are derived by a contract. A
parameter binding gives it model-level identity and expresses weight tying when one identity has
multiple members. See [Specification §3.4](SPECIFICATION.md#34--bindings).

### Partition

A division along a contract-declared semantic axis that preserves the primitive's meaning. A
contract derives the resulting logical communication; hardware placement and the actual collective
remain outside the model. See
[Specification §4.1](SPECIFICATION.md#41--contract-contents-o92-semantic-part).

### Port

A named interface on a primitive contract. A value port carries a temporary graph value; a state
port refers to logical storage that persists across token invocations. Occurrence bindings connect
ports but do not redefine them. See [Specification §3](SPECIFICATION.md#3--occurrences).

### Primitive

A reusable semantic operation kind, such as an embedding, attention operation, feed-forward
operation, or residual addition. A contract defines the primitive; an occurrence places one use of
it in a model graph. See [Specification §4](SPECIFICATION.md#4--primitive-semantic-contracts).

### `present_when`

See [`when` and `present_when`](#when-and-present_when).

## Q

### Quantity

A named scalar fact or variable in a model document. Its regime, type, and source are independent:
for example, an externally assigned cardinality and a literal model constant are both quantities.
Runtime load variables such as active batch size do not belong here. See
[Specification §2.1](SPECIFICATION.md#21--quantities-o21-o22-o23-o04) and
[Model guide §2.1](TENSORSPINE-MODEL_JSON.md#21--quantities-and-expressions).

## R

### Regime

Whether a quantity is a `model_constant` or a `model_variable`. Regime does not say where the value
comes from; that is the quantity's source. See
[Specification §2.1](SPECIFICATION.md#21--quantities-o21-o22-o23-o04) and
[Model guide §2.1](TENSORSPINE-MODEL_JSON.md#21--quantities-and-expressions).

### Role

A catalog identity describing the semantic use of a value or logical tensor for precision policy,
such as `activation.hidden` or `attention.qkv_projection`. A role is not a dtype; its catalog entry
defines admissible dtypes and a default. See
[Specification §4.1](SPECIFICATION.md#41--contract-contents-o92-semantic-part).

## S

### Sharing

The use of one logical resource by multiple sites. A contract declares whether and at what
granularity sharing is semantically permitted; model bindings declare the actual parameter or state
identity shared by occurrences. Sharing is explicit and is never inferred from equal shapes or
similar names. See [Specification §3.4](SPECIFICATION.md#34--bindings) and
[Model guide §3.1](TENSORSPINE-MODEL_JSON.md#31--state-is-split-between-contract-and-graph).

### Source

The word is qualified by context:

- a **quantity source** says whether a quantity is literal, external, or derived;
- an **indexing source** names the sequence, patch stream, or fragment stream indexed by a domain;
- a **value source** is the producer endpoint of a value binding.

These are not interchangeable fields. See [Model guide
§2](TENSORSPINE-MODEL_JSON.md#2--the-tensorspine-20-model-document).

### Semantics

The rules that determine what a document means, including expansion, validity, and derived
consequences. In this repository, semantics are defined only by the language specification; prose
in the README, model guide, and glossary is explanatory. See
[Specification §1](SPECIFICATION.md#1--scope-and-authority).

### State

Logical storage exposed through a state port and preserved across token invocations. The contract
derives its payload, evolution, access geometry, and permitted operations; model state bindings
declare identity, members and any invocation boundary; keys, liveness and visits are derived. See
[Specification §§4.3–4.4](SPECIFICATION.md#43--state-derivation).

### State identity

A model binding that declares which state-port members name the same logical storage. It is a graph
fact, separate from the state descriptor derived by a contract. See
[Specification §3.4](SPECIFICATION.md#34--bindings).

### Structural argument

A contract argument flagged `structural: true`: it decides which parameter slots, ports or state
ports exist, or their shapes (`heads`, `window`, `kv_heads`). A non-structural argument (`rope`,
`activation`) changes only the computation. See
[Model guide §3](TENSORSPINE-MODEL_JSON.md#3--catalog-contracts).

## T

### Template

A model document with external quantities: it denotes a family of graphs, one per admissible
assignment, and is instantiated by a template contract. Templates live alongside the models
(`data/models/decoder-causal-yarn.json`). See
[Specification §4.6](SPECIFICATION.md#46--template-contracts).

### Template contract

A contract whose computation is provided by a template rather than by a consumer capability. The
contract keeps its own semantic identity and pins the template's version; its parameters, states,
costs and partitions are derived from the expanded template. An occurrence of a template contract
is an *instance*. See [Specification §4.6](SPECIFICATION.md#46--template-contracts).

### Type

The declared kind of a quantity, such as cardinality, real, physical, enum, or boolean. Type is
independent of regime and source. Ports and tensors separately carry shape and precision-role
information. See [Specification §2.1](SPECIFICATION.md#21--quantities-o21-o22-o23-o04).

## V

### Visits

How often a state is visited per indexing domain and phase — once per token for a decoder cache,
once per source for a frozen cross-attention cache, once per fragment for a streaming encoder. The
rate is derived from the value graph, the generative outputs and the invocation boundaries; the
counts (tokens per request, fragments per stream) are deployment intent. Visits size computation;
state liveness sizes memory. See
[Specification §4.4](SPECIFICATION.md#44--information-supplied-by-the-graph).

### `when` and `present_when`

Three condition sites use similar names but different inputs and effects:

| Site | Evaluated from | Effect |
|---|---|---|
| Model-level `when` | Model quantities and in-scope composition indices | Controls whether an occurrence site or binding is emitted during expansion |
| Contract element `present_when` | Resolved primitive arguments | Controls whether a port, parameter, constant, or state slot exists |
| Contract-rule `when` | Resolved primitive arguments | Selects which ordered derivation rule applies to an element that exists |

They are not aliases and cannot be moved between contexts. See
[Specification §5.2](SPECIFICATION.md#52--deterministic-expansion),
[Model guide §2.3](TENSORSPINE-MODEL_JSON.md#23--occurrences), and
[Model guide §3](TENSORSPINE-MODEL_JSON.md#3--catalog-contracts).
