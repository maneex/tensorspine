# TensorSpine glossary

This glossary is the non-normative lookup point for TensorSpine terminology. Each entry gives a
reader-facing summary and links to the document that owns the precise definition. The
[language specification](SPECIFICATION.md) is authoritative for validity and denotation; the
[JSON Schema](../schemas/tensorspine.schema.json) is authoritative for the concrete model grammar.
If a summary here conflicts with either authority, use the authority. The [README](../README.md)
gives the project orientation, and the [model JSON guide](TENSORSPINE-MODEL_JSON.md) explains how to
author the concrete format.

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

### Axis space

Every axis belongs to the value space — the dimensions of values and tensors — or to the instance
space (`instance.session`, `instance.branch`), the key axes along which state allocations are
distinct. See [Specification §4.3](SPECIFICATION.md#43--state-derivation).

## B

### Backend

The hardware a primitive implementation targets — cpu, nvidia, neuron, tpu. Outside the language: a
model document never names one, and a serving application may target several. When it must select
one, it does so through opaque physical parameters passed beside the contract arguments; `backend`
is one optional key ([Capabilities](../generators/CAPABILITIES.md)).

### Branch

A separately admitted choice within one contract version: an enum value, a record field, or an
optional argument. An implementation may support some branches before others; its capabilities
manifest states which, and the generated branch ledger lists what remains for one model and one
implementation. See [Capabilities](../generators/CAPABILITIES.md).

### Binding

A graph declaration that records a model-specific relationship. Value bindings are directed edges
from an output port to an input port. Parameter, constant, and state bindings associate one identity
with the contract slots that use it. See [Specification §3.4](SPECIFICATION.md#34--bindings) and
[Model guide §2.5](TENSORSPINE-MODEL_JSON.md#25--bindings).

## C

### Carrying

A contract's condition, declared once on a state port, under which the state survives between the
invocations that deliver successive fragments of its stream. With the input's `fragmented` flag it
derives whether a state persists across fragments; a self-indexed state that is not carried is reset
at each fragment, a carried state must sit on a fragmented stream (V16), and the model declares
nothing about it. This replaces the former invocation boundary. See
[Specification §4.3](SPECIFICATION.md#43--state-derivation) and
[§5.3](SPECIFICATION.md#53--indexing-domains-streams-and-fragmentation).

### Catalog

The vocabulary resolved by a model: a set of bases containing independently identified axes,
precision roles, and versioned contracts; it has no single global version, and one identity carried
by two bases with different contents is a conflict (V1). See
[Model guide §3](TENSORSPINE-MODEL_JSON.md#3--catalog-contracts) and
[Specification §8.2](SPECIFICATION.md#82--identity-and-versioning).

### Composition

A named, finite, indexed family of occurrence sites, with the bindings among those sites. A site
may be guarded by `when`, so one composition over a flat layer index expresses a periodic pattern;
a binding is emitted only where the sites it names are, so the guard is never repeated on the
bindings. Expansion turns sites into ordinary occurrences with deterministic identifiers and scoped
bindings into top-level rules. It is authoring shorthand, not a runtime node or a nested model. See
[Specification §5.1](SPECIFICATION.md#51--the-expanded-graph-is-authoritative) and [Model guide
§2.4](TENSORSPINE-MODEL_JSON.md#24--compositions-and-deterministic-expansion).

### Communication

The logical data movement implied when a semantic partition cuts model values. Contracts describe
this consequence independently of physical collectives, topology, or measured traffic. See
[Specification §4.1](SPECIFICATION.md#41--contract-contents-o92-semantic-part).

### Constant

A non-learned numeric tensor or buffer with explicit identity, shape, and dtype, optionally located
by a URI and identified by content digest. It is distinct from a scalar quantity. See
[Model guide §2.2](TENSORSPINE-MODEL_JSON.md#22--external-constants) and
[Specification §3](SPECIFICATION.md#3--occurrences).

### Conformer

An implementation of a contract version whose outputs and states agree with that version's witness
on its unit fixtures, within the tolerance stated for the implementation's compute dtype. Primitive
implementations in serving applications and the ZML example are conformers; they do not redefine
the primitive. See
[Specification §4.2](SPECIFICATION.md#42--computation-may-be-delegated-identity-may-not).

### Contract

The immutable, versioned semantic definition of a primitive. An occurrence pins a contract by name
and version; the contract declares arguments and derives ports, logical tensors, state, effects,
costs, and legal partitions. Its reference implementation is the witness that fixes what the
primitive computes; all optimized implementations are conformers. See
[Specification §4](SPECIFICATION.md#4--primitive-semantic-contracts) and
[Model guide §3](TENSORSPINE-MODEL_JSON.md#3--catalog-contracts).

### Correction

A cost entry a contract declares because the inventory rule cannot see it: a condition over the
arguments, an expression, a status, and what it is counted per — an element of the output domain, a
cached position of a state, a sequence, or an invocation. Every entry whose condition holds
contributes. See [Specification §4.1](SPECIFICATION.md#41--contract-contents-o92-semantic-part).

### Coverage

Closure of the in-scope graph class over the contracts available in the pinned catalog: every such
graph has at least one representation. It is not a claim that every architecture has a contract or
that every implementation supports every contract. See
[Specification §1](SPECIFICATION.md#1--scope-and-authority).

### Cut

A partition of the emitted occurrences into two blocks. It is **legal** when every crossing edge is
directed from the first block to the second. Its **payload** is the set of values on the crossing
edges — the values live at the cut — each sized per invocation by its port shape and the number of
its domain elements. See [Specification §7](SPECIFICATION.md#7--required-derived-products).

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
| **D2** | Values, shapes, and the payload of every legal cut |
| **D3** | Parameter tensors, including roles, selected dtypes, sensitivity, shapes, and sharing |
| **D4** | Complete state: descriptors, instances, derived keys and liveness, visits per phase, carrying, and operations |
| **D5** | Logical costs and cut payloads |
| **D6** | Legal cuts and semantic partition axes |

See [Specification §7](SPECIFICATION.md#7--required-derived-products).

## E

### Expression

A tagged, inspectable construction for a scalar value or condition. Model expressions read model
quantities and composition indices; contract expressions read resolved primitive arguments. The two
contexts are deliberately separate. See [Model guide
§2.1](TENSORSPINE-MODEL_JSON.md#21--quantities-and-expressions) and [Specification
§2.2](SPECIFICATION.md#22--derivation-algebra-and-qualified-values-o01-o02-o03-o05-o06).

### Extent

The expression giving the size of one named shape axis. An extent is a value; the axis supplies the
dimension's semantic identity. See [Specification
§4.1](SPECIFICATION.md#41--contract-contents-o92-semantic-part).

## F

### Family

A named, addressable grouping of occurrences. Families allow later derivations to refer to a set of
nodes without enumerating every expanded occurrence. See
[Specification §3.1](SPECIFICATION.md#31--identity-o21-o35).

### Fragmented input

A public input whose elements arrive over several invocations, declared on the input
(`fragmented`). Together with a contract's carrying condition it decides which states survive
between fragments. See
[Specification §5.3](SPECIFICATION.md#53--indexing-domains-streams-and-fragmentation).

### Fixture

A safetensors file carrying reproducible conformance evidence and schema-checked metadata. A
**unit fixture** records one contract witness on generated inputs, parameters and states; every
conformer runs it. An **integration fixture** records a delivery implementation at a model's legal
cuts and states; every implementation running that model compares against it. See the
[fixture-format guide](TENSORSPINE-FIXTURE.md).

## G

### Generative

An output property stating whether a public output participates in generation: each generated
element is delivered to the output's stream in the next invocation, so a generative output has kind
`token`. Generation is autoregressive; diffusion-style regeneration of every position is outside
the current scope. TensorSpine also permits non-generative and per-token outputs. See [Specification
§2.3](SPECIFICATION.md#23--public-inputs-and-outputs-o81-o82-o83-o42).

## I

### Generator

A repository executable that builds and runs a model graph—or an extract between legal cuts—over
primitive implementations. The reference generator is the repository's target generator: it runs
the contract reference implementations and carries their witnesses. The ZML generator is an
example conformer. A serving application does not embed a TensorSpine generator; it implements the
subset of TensorSpine contracts it supports and may consume TensorSpine's derived facts and tooling
directly. See [Capabilities](../generators/CAPABILITIES.md). Not to be confused with a *generative*
output, which feeds its stream back.

### Identity

An explicit declaration that answers “which logical thing is this?” Contract identity pins meaning;
parameter, constant, and state identities group the slots that refer to the same logical resource;
occurrence identity names a graph node. Matching names or shapes do not implicitly create identity.
See [Specification §§3.1–3.4](SPECIFICATION.md#3--occurrences).

### Indexing domain

What indexes a value or state port: a pair (kind, stream). The kind says what one element is —
`sequence`, `token`, `position` (a frame of a sampled signal) or `patch`; the stream is the public
input that introduced the elements. It distinguishes, for example, the text tokens from the audio
frames an encoder produced. Domains must agree on every edge (V5) unless a contract transform says
how they relate. See
[Specification §5.3](SPECIFICATION.md#53--indexing-domains-streams-and-fragmentation).

### Interface

A public model input or output attached to existing occurrence ports. An input names the ports it
feeds — one or more — and declares its kind, the stream it introduces or joins, and whether it is
fragmented; an output names one port and states whether it is generative, its domain being that of
the port. See
[Specification §2.3](SPECIFICATION.md#23--public-inputs-and-outputs-o81-o82-o83-o42) and
[Model guide §2.6](TENSORSPINE-MODEL_JSON.md#26--public-interfaces).

### Invocation

One evaluation of the expanded graph on one delivery of its inputs: all elements of a non-fragmented
input, one fragment of a fragmented one; when an output is generative, each generated element is
delivered to the output's stream in the next invocation. The value graph is acyclic within an
invocation; recurrence passes through state ports. See
[Specification §7](SPECIFICATION.md#7--required-derived-products).

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

### Location

Where a parameter identity's tensor is stored in the artifact the document wraps: one physical
tensor, a `stack` of locations along an axis, a `concat` of locations along an axis, or a `slice`
of one physical tensor. Declared on the binding, evaluated into D3, checked against a checkpoint's
headers by V17. See [Specification §3.4](SPECIFICATION.md#34--bindings) and
[Model guide](TENSORSPINE-MODEL_JSON.md#locating-the-weights).

### Model document

A `tensorspine/2.0` declaration of model identity (and, for a template, its version), catalog
bases, quantities — literal, external with domain and optional default, or derived — constants,
occurrences, compositions with their scoped bindings, bindings with their identities and dtypes, and
public interfaces. It declares
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
See [Specification §2.2](SPECIFICATION.md#22--derivation-algebra-and-qualified-values-o01-o02-o03-o05-o06).

## P

### Parameter tensor

A learned logical tensor whose slot, shape, and precision role are derived by a contract. A
parameter binding gives it model-level identity and expresses weight tying when one identity has
multiple members. See [Specification §3.4](SPECIFICATION.md#34--bindings).

### Partition

A division along a contract-declared semantic axis that preserves the primitive's meaning, with
the logical communication it implies. Every contract states its partitions: one or more axes,
`any_axis` for an elementwise primitive, or `none`; an empty list is refused. Hardware placement
and the actual collective remain outside the model. See
[Specification §4.1](SPECIFICATION.md#41--contract-contents-o92-semantic-part).

### Payload

Of a cut: see [Cut](#cut). Of a state: what one position holds, for `append` and `window`, or the
whole state, for `fixed`; a window's ring is `span` payloads. See
[Specification §4.3](SPECIFICATION.md#43--state-derivation).

### Phase

For a generative document, the invocations that consume supplied elements (prefill) and those that
consume one generated element (decode). Visit rates are derived per phase; element counts are
deployment intent. See [Specification §7](SPECIFICATION.md#7--required-derived-products).

### Physical tensor

A tensor as a checkpoint stores it, under its own name, shape and dtype. A location maps every
logical tensor of D3 onto physical tensors; the language guarantees each logical tensor once (V7)
and each physical tensor bound once (V17, I9). The encoding of the files themselves is outside the
language ([Specification §10.3](SPECIFICATION.md#103--explicitly-separate-concerns)).

### Port

A named interface on a primitive contract. A value port carries a temporary graph value; a state
port refers to logical storage that persists across token invocations. Occurrence bindings connect
ports but do not redefine them. See [Specification §3](SPECIFICATION.md#3--occurrences).

### Primitive

A reusable semantic operation kind, such as an embedding, attention operation, feed-forward
operation, or residual addition. A contract defines the primitive; an occurrence places one use of
it in a model graph. See [Specification §4](SPECIFICATION.md#4--primitive-semantic-contracts).

### Primitive implementation

Application-owned code that executes a contract version or one of its branches: kernels, fusions,
layouts, collectives and data paths optimized for target hardware. A capabilities manifest states
the implemented subset; conformance means its outputs and states match the contract's witness. It
does not define the primitive. See [Capabilities](../generators/CAPABILITIES.md).

### `present_when`

See [`when` and `present_when`](#when-and-present_when).

## Q

### Quantity

A named scalar in a model document, with a type and a literal, external or derived source. It is
variable when its source is external or depends transitively on an external quantity; a variable
declares its admissible domain. Runtime load variables such as active batch size do not belong here. See
[Specification §2.1](SPECIFICATION.md#21--quantities-o21-o22-o23-o04) and
[Model guide §2.1](TENSORSPINE-MODEL_JSON.md#21--quantities-and-expressions).

## R

### Role

A catalog identity describing the semantic use of a value or logical tensor for precision policy,
such as `activation.hidden` or `attention.qkv_projection`. A role is not a dtype; its catalog entry
defines admissible dtypes, a default and a sensitivity. A dtype selected on a parameter or state
identity must be admissible for every member's role (V14); absent, the default applies. See
[Specification §4.1](SPECIFICATION.md#41--contract-contents-o92-semantic-part).

## S

### Sensitivity

A precision role's advice on quantisation — `quantizable`, `reduced` or `full_precision` — carried
into D3 with every tensor's role and selected dtype. See
[Specification §7](SPECIFICATION.md#7--required-derived-products).

### Semantics

The rules that determine what a document and its primitives mean. The specification governs
validity and graph denotation; each contract version's reference implementation—the witness—governs
what that primitive computes. Prose in the README, guides and glossary is explanatory. See
[Specification §1](SPECIFICATION.md#1--scope-and-authority) and
[§4.1](SPECIFICATION.md#41--contract-contents-o92-semantic-part).

### Serving application

The system that connects a model, requests and target infrastructure. Its harness maps derived
model facts to admission, cache, batching, routing, placement and scheduling decisions; its
primitive implementations execute the supported TensorSpine contracts with application-specific
optimizations. See [README §1](../README.md#1-why-tensorspine) and the
[harness guide](HARNESS.md).

### Sharing

The use of one logical resource by multiple sites. A contract declares whether and at what
granularity sharing is semantically permitted; model bindings declare the actual parameter or state
identity shared by occurrences. Tied parameter members must be compatible: `shareable` on both
sides, roles listed in each other's sharing rules, equal shapes (V15). Sharing is explicit and is
never inferred from equal shapes or similar names. See [Specification §3.4](SPECIFICATION.md#34--bindings) and
[Model guide §3.1](TENSORSPINE-MODEL_JSON.md#31--state-is-split-between-contract-and-graph).

### Source

The word is qualified by context:

- a **quantity source** says whether a quantity is literal (read from configuration, optionally
  with the derivation it must agree with), external (supplied by an assignment), or derived (an
  expression over other quantities, exact by construction);
- a **stream** is the public input that introduced the elements a domain indexes (see
  [Stream](#stream));
- a **value source** is the producer endpoint of a value binding.

These are not interchangeable fields. See [Model guide
§2](TENSORSPINE-MODEL_JSON.md#2--the-tensorspine-20-model-document).

### Sparsity unit

A contract's declaration that only some of its parameters are activated per element: the slots and
the axis that form one activatable unit, the policy that selects units — an argument, the value on
an input port, or the element itself — the count activated per element, and the upper bound on the
union an invocation may activate. A lookup table is the limiting case: one row per element. Cost
derivation counts a unit's weights at the activated fraction per element and in full for residency
and worst-case transfer. See [Specification §4.5](SPECIFICATION.md#45--structured-sparsity-o66).

### State

Logical storage exposed through a state port and preserved across invocations. The contract derives
its payload, evolution, access geometry, permitted operations and carrying condition; model state
bindings declare identity, members and dtype; keys, liveness, visits and carrying are derived. The
closed evolution vocabulary is `append`, `window` and `fixed`; another evolution is a new state law.
See
[Specification §§4.3–4.4](SPECIFICATION.md#43--state-derivation).

### State identity

A model binding that declares which state-port members name the same logical storage. It is a graph
fact, separate from the state descriptor derived by a contract. See
[Specification §3.4](SPECIFICATION.md#34--bindings).

### Status page

The generated, untracked page reporting catalog and corpus state, implementation coverage and recorded
verification for the commit from which the site was built. It reports project state, never language
validity or primitive meaning. See the [status page](https://maneex.github.io/tensorspine/status/).

### Stream

The public input that introduced the elements a domain indexes; with the kind it forms the indexing
domain. An input introduces the stream named after it or joins an existing one; a contract
transform can `merge` a stream into fewer elements of another kind, `align` a port carrying another
stream onto the occurrence's own, or `insert` one stream's elements into another. See
[Specification §5.3](SPECIFICATION.md#53--indexing-domains-streams-and-fragmentation).

### Structural argument

A contract argument flagged `structural: true`: it decides which parameter slots, ports or state
ports exist, or their shapes (`heads`, `window`, `kv_heads`). A non-structural argument (`rope`,
`activation`) changes only the computation. See
[Model guide §3](TENSORSPINE-MODEL_JSON.md#3--catalog-contracts).

## T

### Template

A model document with external quantities: it denotes a family of graphs, one per admissible
assignment, and is instantiated by a template contract. Templates live where the catalog manifest
says (`templates`), one immutable file per version (`data/models/decoder-causal-yarn/1.0.0.json`).
See
[Specification §4.6](SPECIFICATION.md#46--template-contracts).

### Template contract

A contract whose computation is provided by a template rather than by a consumer capability. The
contract keeps its own semantic identity and pins the template's version; its parameters, states,
costs and partitions are derived from the expanded template. An occurrence of a template contract
is an *instance*. See [Specification §4.6](SPECIFICATION.md#46--template-contracts).

### Transform

A contract's declaration of how a port's domain relates to the occurrence's own: `merge`, `align`
or `insert`. Transforms also carry element counts, so a cut's payload is sized correctly after a
projector or a splice. See
[Specification §5.3](SPECIFICATION.md#53--indexing-domains-streams-and-fragmentation).

### Trunk

A maximal connected component of the expanded value graph. Several may coexist in one document,
and a trunk may carry no state. See
[Specification §5.2](SPECIFICATION.md#52--deterministic-expansion).

### Type

The declared kind of a quantity, such as cardinality, real, physical, enum, or boolean. Type is
independent of source. Ports and tensors separately carry shape and precision-role information. See
[Specification §2.1](SPECIFICATION.md#21--quantities-o21-o22-o23-o04).

## V

### Visits

How often a state is visited per indexing domain and phase — once per element for a decoder cache,
once per source element for a frozen cross-attention cache, once per fragment for a streaming
encoder. The rate is derived from the value graph, the generative outputs and the fragmentation of
inputs and is data-independent; a graph whose visits depend on element values is outside the
current scope. Counts (tokens per request, fragments per stream) are deployment intent. Visit rates
size computation; state liveness sizes memory. See
[Specification §4.4](SPECIFICATION.md#44--information-supplied-by-the-graph).

## W

### Witness

The reference implementation supplied with one immutable contract version and run by the reference
generator, bound to that identity with a tolerance per compute dtype and unit fixture ids. It is the
authority for what the primitive computes; every other implementation is a conformer. The
contract's description, cited sources and integration fixtures remain the authority for whether a
witness is correct. See
[Specification §4.1](SPECIFICATION.md#41--contract-contents-o92-semantic-part).

### `when` and `present_when`

Three condition sites use similar names but different inputs and effects:

| Site | Evaluated from | Effect |
|---|---|---|
| Model-level `when` | Model quantities and in-scope composition indices | Controls whether an occurrence site or binding is emitted during expansion; a binding is also absent wherever an occurrence it names is absent, so a site's guard is not repeated on its bindings |
| Contract element `present_when` | Resolved primitive arguments | Controls whether a port, parameter, constant, or state slot exists |
| Contract-rule `when` | Resolved primitive arguments | Selects which ordered derivation rule applies to an element that exists |

They are not aliases and cannot be moved between contexts. See
[Specification §5.2](SPECIFICATION.md#52--deterministic-expansion),
[Model guide §2.3](TENSORSPINE-MODEL_JSON.md#23--occurrences), and
[Model guide §3](TENSORSPINE-MODEL_JSON.md#3--catalog-contracts).
