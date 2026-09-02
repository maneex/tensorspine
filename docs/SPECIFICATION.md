# TensorSpine Language Specification

> Represent every in-scope model as a **finite graph of parameterized primitive occurrences**, then
> derive all other information from the primitives' contracts.

*TensorSpine 2.0 — language specification, revision of 29 August 2026. The language version is the
`tensorspine/2.0` tag every document and schema carries; catalog units carry their own versions
(§8.2). This document is the sole normative authority for language validity and denotation, and is
self-contained for both: neither depends on source code or tools. The concrete model JSON grammar is
the [model schema](../schemas/tensorspine.schema.json); the concrete derived encoding is described by
the [derived-document guide](TENSORSPINE-DERIVED_JSON.md) and its schema. Every requirement
identifier cited here (O‑, I‑, N‑) is stated in Appendix A or §9.*

For motivation and repository orientation, read the [README](../README.md). For a field-by-field
JSON authoring guide, read [TensorSpine model JSON](TENSORSPINE-MODEL_JSON.md). The
[glossary](GLOSSARY.md) summarizes terminology and points back to its canonical sections; the
[architecture guide](ARCHITECTURE.md) explains the design rationale. None of those explanatory
documents adds requirements to this specification.

## §1 — Scope and authority

This document specifies what a model document can express and what a primitive contract must contain
so all consequences can be derived.

**In scope:** inference; a graph that is **statically finite per invocation**; the graph class of
§5.3 — a finite directed acyclic value graph per invocation, recurrence only through state ports —
over contracted primitives.

**Out of scope:** training; fault tolerance and session recovery; hot reconfiguration; resource
topology; runtime data structures; placement; and hardware assignment.

Three statements bound the class, so that "exact" and "derived" below mean what they say:

- **Visits are data-independent.** Which occurrences an element visits, and how often a state is
  visited, is fixed by the graph and the delivery of its inputs (§7), never by the value of an
  element. A model whose per-token visits depend on the data — a mixture of depths, an early exit
  — is out of scope until a status for visits exists.
- **The generative model is autoregressive.** A generative output delivers one element per
  invocation to its stream, consumed by the next (§7); a model that regenerates every position at
  each invocation — a diffusion language model — is out of scope.
- **The state laws are three.** A state evolves by `append`, `window` or `fixed` (O5.2); a state
  that evolves otherwise — a retained prefix kept beside a sliding window — has no representation
  and is a new law: a capability every consumer must implement (§8.1), not a rearrangement.

**Coverage** is closure of that class over the available contracts:

> Every graph of the class above whose occurrences use available contracts — any wiring, any sharing
> pattern, any set of public inputs and outputs, any periodic or irregular repetition — has **at
> least one** representation.

The claim is about the graph language and holds by construction of the class; it says nothing
about the contracts. A node with no contract is not a counter-example but a new contract (§8.1),
and a document is only as expressive as the catalog it pins. The reference corpus
(`data/models/`, §10.1) is the evidence for the claim and the regression set; it is not part of
the definition. A graph of the class with no representation would be a counter-example.

Three authoring conveniences the graph language does without, stated rather than discovered:
sharing across a template boundary is not expressible (the flat form of the same graph is); a
contract's port set is static, so a variable number of operands is expressed by chaining or by
stacking along an axis; there are no indexed literal tables, so a per-repetition hyperparameter
is an `if/then/else` over the index or one site per value.

### 1.1 — Required properties

| Property | Meaning | Status |
|---|---|---|
| **Functional denotation** | A document denotes exactly one graph: denotation is total and unambiguous. A parameterized document denotes one graph **per admissible assignment** (§4.6). | **required** |
| **Coverage** | Every in-scope graph has at least one representation. | **required** |
| *Unique representation* | Each graph has exactly one representation. | *not required* |

Functional denotation is not injectivity. Injectivity would forbid two documents from denoting the
same graph; TensorSpine explicitly does not require a canonical representation.

### 1.2 — Governing principle

> **The model declares causes. Primitive contracts derive consequences.**

Apply this field test: *Could two models using the same primitive with the same arguments
legitimately give this field different values?* If not, the field does not belong in the model
document.

## §2 — Model document

A model document contains exactly:

| Element | Content |
|---|---|
| **Identity** | A stable, authoritative model identifier (O1.2), and, for a template, the version of this representation (§4.6). |
| **Catalog bases** | The catalog bases whose units the occurrences pin, as locations relative to the document; a base that does not exist is a rejection (V1), and two bases carrying one identity with different contents are a conflict (V1). Contract identity is per occurrence, `{name, version}`, immutable (O1.1, §8.2); there is no catalog-wide version. |
| **Quantities** | The namespace of dimensions and scalars (§2.1). |
| **External constants** | Non-learned numeric data with identity, shape, and type (§3). |
| **Occurrences** | Graph nodes (§3). |
| **Compositions** | Finite indexed families of occurrence sites whose expansion is normative (§5). |
| **Bindings** | Value edges, tensor bindings, constant bindings, and state identities (§3.4). |
| **Public inputs and outputs** | Interfaces with indexing domains (§2.3). |

### 2.1 — Quantities *(O2.1, O2.2, O2.3, O0.4)*

Quantities form one flat namespace, declared once and referenced everywhere. Each quantity declares
two independent properties, and a third follows from them:

- **Dimensional type:** a cardinality (non-negative integer); a ratio or real hyperparameter bounded
  by contract; a physical quantity with a **unit** (bytes, elements, tokens, seconds, operations);
  or an enum or boolean, which participates in arithmetic only through an allowed conditional.
- **Provenance:** a literal value, read from configuration, optionally with the **derivation** by
  which it follows from other quantities (V11 checks the two agree); an **external** value supplied
  by an assignment (§4.6), with a declared domain and optionally a declared default; or a
  **derived** value, an expression of the algebra of §2.2 over other quantities.

A quantity is a *model variable* when its source is external or is a derivation that reads a
variable quantity transitively, and then declares a domain (O2.2, V3); otherwise it is a *model
constant*, known when the document is written. Declaring that classification would copy the rule
that derives it (§4.4). Variability and type are independent: a model constant need not be an
integer.

Load variables such as batch size and sequence count do **not** belong in the model document.

### 2.2 — Derivation algebra and qualified values *(O0.1, O0.2, O0.3, O0.5, O0.6)*

A derivation expressed only as a function in a general-purpose language moves code instead of making
the derivation verifiable.

**O0.1 —** Every derived quantity is expressed in a closed, normative, inspectable scalar algebra
over declared quantities and literals. The operator set is exactly: addition, subtraction,
multiplication, division, floor division, ceiling division, modulo, minimum, maximum, negation,
absolute value, and a conditional `if/then/else` over a condition of the closed condition language
(presence tests, comparisons, and boolean composition). An operator outside this set is a rejection.

**O0.2 —** A derivation outside this algebra uses a normative interface with specified inputs,
semantics, and conformance tests. No function body is read or trusted; its result is at least a
qualified bound. No such interface is defined in this revision, and the model grammar admits a call
to one only once it is: adding it is a compatible extension (§8.1), while a call nothing can read
would be a comment (§10.2).

**O0.3 —** A derived quantity of a model document is exact and its provenance is the expression
itself: it reads only declared quantities, resolves acyclically, and conforms to the declared type
and domain — the rounding of a division is chosen explicitly (floor or ceiling). Epistemic status
and provenance are declared where a value can be a bound or an estimate: the logical cost a
contract states (§4.1) and the derived products (§7).

**O0.5 — Qualified values.** A value stated by a contract correction (§4.1), by a sparsity bound
(§4.5), or supplied to a product from deployment intent (§10.3) carries a **status**: `exact`;
`upper_bound` or `lower_bound`, a one-sided guarantee on a non-negative quantity; or `estimate`, no
guaranteed relation to the true value — whether because information was lost in combination or
because the value is an expectation supplied from outside. Expected values are never derived by the
language. Model quantities are exact (O0.3).

Status propagates through the algebra by the table below, where every operand is a non-negative
real, `E` is exact, `U` an upper bound, `L` a lower bound, `S` an estimate, and any operation with
an `S` operand yields `S`:

| operation | E,E | E,U / U,E | E,L / L,E | U,U | L,L | U,L / L,U |
|---|---|---|---|---|---|---|
| `add`, `multiply`, `min`, `max` | E | U | L | U | L | S |
| `subtract` a−b | E | a−U: L · U−b: U | a−L: U · L−b: L | S | S | U−L: U · L−U: L |
| `divide` a/b, `floor_divide`, `ceil_divide` (b > 0) | E | a/U: L · U/b: U | a/L: U · L/b: L | S | S | U/L: U · L/U: L |
| `negate`, `absolute` | not applicable to qualified values: a rejection | | | | | |
| `if/then/else` | the condition ranges over exact values only; the result has the chosen branch's status | | | | | |

A qualified value is rounded in the direction its status requires — an upper bound up, a lower
bound down; an exact value by the operator chosen (O0.3). No propagation turns an estimate into a
bound.

**O0.6 —** Every language introduced by this specification is closed and decidable or uses a
normative interface. This applies to the size algebra, the contract-condition language in §4.3, and
the expansion rules in §5. Calling a representation “symbolic” does not make it decidable.

### 2.3 — Public inputs and outputs *(O8.1, O8.2, O8.3, O4.2)*

Each public **input** declares the occurrence ports it feeds — one or more — and its indexing
domain (§5.3): a kind, and either the stream it introduces (named after the input) or an existing
stream it joins. It may be **fragmented**: its elements arrive over several invocations. Each
public **output** names one occurrence port and states whether it is generative; its domain is
that of the port. A public input feeds a port that nothing else feeds (V7); several outputs may name
one port (O8.3). A generative output has kind `token`.

The language must support:

- multiple simultaneous outputs, including **non-generative** outputs; a classifier exposing two
  probabilities is valid (O8.2);
- auxiliary outputs that reference existing values without introducing a new construction: they are
  additional output entries, not invented operations (O8.3);
- explicit seeding of every flow; implicit seeds are forbidden (O4.2).

## §3 — Occurrences

An occurrence is a graph node. The language keeps these five categories distinct:

| Category | Meaning | Example |
|---|---|---|
| **Primitive argument** | A scalar, enum, or record of such values. | `heads`, `mask`, `inner` |
| **Parameter tensor** | A **learned** weight with shape, role, and sharing rule. | Attention Q/K/V/O projections |
| **External constant or buffer** | Non-learned numeric data with identity, shape, and type. | A position table |
| **Value port** | A temporary input or output. | Residual activation |
| **State port** | Storage that survives a token invocation. | A cache |

### 3.1 — Identity *(O2.1, O3.5)*

Every occurrence has a stable identifier, unique after expansion (§5.2), and belongs to one or more
named, addressable families. Families allow sets of occurrences to be referenced without
enumeration.

### 3.2 — Contract reference *(O1.1, §8.2)*

An occurrence references a contract through an immutable identity, `{name, version}` with a semantic
version (§8.2). Capability rejection (§8.1) answers “Can this consumer execute it?”
Contract identity answers “Which exact meaning did the author reference?” Neither replaces the
other.

### 3.3 — Arguments *(O9.1, O9.4)*

A model names a primitive and supplies arguments; it never describes the primitive's computation.

An argument is a scalar, enum, or record of such values. Inline numeric tensors are forbidden. The
boundary is between inline structural metadata and external numeric tensors described as parameters
or constants with identity, shape, and type. Data nature—not size—determines the category (N4).

### 3.4 — Bindings

**Value edges** *(O4.4)* are explicit source-to-destination bindings:
`occurrence.port → occurrence.port`. Producer-consumer flow is never inferred from implicit mutation
of a named flow. Explicit edges determine the values live at any cut and therefore the cut's
**logical payload** (O4.5), which is not actual traffic.

**Parameter-tensor bindings** *(O6.2)* identify the logical tensors consumed by each occurrence and
any sharing between occurrences. With tied embeddings, one physical tensor satisfies two logical
parameters and is counted once.

**Locations.** A parameter binding may say where its identity's tensor is stored in the artifact
the document wraps — a `location`: one physical tensor (`tensor`); one location per coordinate of a
named axis of the slot's shape, its names carrying that coordinate (`stack`); locations laid
consecutively along an axis (`concat`); a region of one physical tensor along an axis, at an
offset, of the logical extent (`slice`). A physical name is tagged data — literal strings, a
composition index printed in decimal, the coordinate of an enclosing stack — never a format
string; the axes it names are the slot's shape axis names. A document locates all of its parameter
identities or none (V17); a tied identity has one location; the tensors of a template instance
cannot be located from outside — a stated limit of this version, the flat form being locatable.

**Parameter identities** are compatible only between slots whose contracts declare them
`shareable`, whose roles are listed in each other's sharing rules, and whose shapes are equal under
V4 (V15).

**State identities** declare when occurrences use the **same storage**. Sharing is a topological fact
that no primitive contract can know: the contract derives a state descriptor, while the graph
declares which occurrence reads another occurrence's state.

> **Use one construction for every case.** Non-adjacent layer cache sharing and cross-attention
> reading encoder state are both state-port identities plus an indexing domain (§5.3); state carried
> between fragments follows from the contract's carrying condition and the input's fragmentation
> (§5.3). Case-specific fields would reintroduce redundant semantics.

Each fact has one authority. If a primitive argument determines a binding, no other field redeclares
it; if the binding is declared, no argument duplicates it.

### 3.5 — Flow multiplicity *(O4.1, O4.3)*

A value port has a shape and a role (O4.1). A residual carried as several parallel streams is a
shape axis of the port (`residual.stream`); the primitive that combines the streams is a contracted
occurrence, never an annotation (O4.3).

## §4 — Primitive semantic contracts

A primitive contract is the immutable, versioned semantic definition referenced by an occurrence.
It declares the primitive's arguments and derives its ports, logical tensors, state, effects, costs,
and legal partitions. Contracts provide coverage: a new primitive can be added without changing the
language.

### 4.1 — Contract contents *(O9.2, semantic part)*

| Element | Content |
|---|---|
| **Semantic identity** | The operation's meaning — stated by the unit in prose and the sources it cites, and fixed by its witness (below) — and its immutable reference identity. |
| **Arguments** | Types, allowed values, required status, declared defaults, conditional presence (`present_when`). |
| **Value ports** | Inputs and outputs with shape functions. |
| **Logical tensors** | A symbolic learned-parameter inventory derived from arguments: shapes, roles, and sharing rules (O6.1). |
| **State ports** | Ports conditional on arguments, with their derivation function (§4.3). |
| **Effects** | The ports the operation reads and writes; and whether, under a condition over the arguments, the operation reads positions of its stream beyond those of the element it produces — **across positions** (O9.5): a convolution reads neighbouring frames, attention every earlier position, a pooler with a reduction the whole sequence, while a merge reads only the group its transform declares. |
| **Logical cost** | Derived from the inventory: for every parameter slot, two operations per weight element per element of the occurrence's output domain, scaled by the activated fraction of a sparse unit (§4.5); the bytes of the inventory, in full for residency. A contract declares only **corrections**: an ordered list of entries, each guarded by a condition over the arguments, each stating an expression, a status (O0.5) and what it is counted per — `element` of the output domain, `cached_position` of a state, `sequence`, or `invocation`. Every entry whose condition holds contributes. Never operations actually executed. Known approximations are documented, not modelled: a strided convolution's first kernel (per input frame), a per-head normalisation scale, a pooler with `reduce`. |
| **Semantic partitions** | Axes whose partition preserves meaning — an argument axis, an instance-key axis, a state payload axis, `any_axis`, or `none` — with the logical communication each implies, several when the partition admits several patterns, and the **granularity** a shard keeps whole along the axis, one unless declared (O7.1); at least one entry. A contract declares what it knows of its own axes: which of the patterns an implementation realises is the manifest's, and consistency across occurrences is compilation's (§7, §10.3). |
| **Domain transforms** | How a port's domain relates to the occurrence's own: `merge`, `align`, `insert` (§5.3). |
| **Witness** | For every published `{name, version}`, one reference implementation supplied with the contract and executed by the reference generator is the authority for what the primitive computes (O1.3), at a tolerance stated per compute dtype; every other implementation is a **conformer**, checked against the witness on the unit fixtures it produces. The witness is not part of the unit — a field no reading consumes would be a comment (§10.2) — but of the reference generator's manifest, which binds the implementation entry point, its tolerances and its fixtures to the identity (Architecture §2); the unit's prose and cited sources are the meaning the witness must agree with (§10.2). |

Anything a contract does not declare it cannot interpret, and is rejected (§8.1): the closed
vocabulary is the rejection condition.

Within a declared shape, the **axis** supplies a dimension's catalog identity, its **extent** is the
expression that supplies the dimension's size, and its **nature** describes the dimension's use at
that site. A port or logical tensor's **role** identifies its semantic use for precision policy; the
role is distinct from the dtype selected under that policy (V14).

Every primitive has a contract; only primitives with state ports have a state contract. Token
embeddings, feed-forward networks, mixtures of experts, patch projections, and output heads are
full primitives. Only a **sequence operator**—attention or a substitute that communicates tokens
along the sequence—carries state.

A logical-tensor inventory is mandatory: without it, an implementation cannot construct a module,
verify a checkpoint, compute parameter size, or choose a partition unit. Deriving only cache
semantics is insufficient.

A contract never contains a generator, a backend, a kernel, a memory layout, a fusion, a workspace,
an executed-operation count, an implementation-supported partition, or an actual collective. Those
belong to a **generator** — the implementation that builds and runs the graph over its own primitives,
on the **backends** (the hardware) it targets — which is outside this specification. Several correct
generators may have different costs without changing the logical graph. One of them, the reference
generator, executes the witness of every contract version (above); being the authority for the
computation does not make its costs, layouts or fusions the contract's.

### 4.2 — Computation may be delegated; identity may not

An implementation may provide computation, but the contract still fixes semantic identity, ports,
types, shapes, arguments, effects, and rejection conditions. Otherwise wiring cannot be checked,
candidates compared, or costs derived safely.

The delegate is checked against the witness (§4.1): an implementation of a contract version
conforms when its outputs and states agree with the witness's, within the tolerance stated for its
compute dtype, on the witness's unit fixtures. Two conformers therefore agree with each other
within the sum of their tolerances, and the agreement of two generators is a statement about the
contract's meaning, not about an oracle outside the language.

The computation provider may itself be a parameterized model document: the template contract
defined by §4.6. Delegation relaxes none of these obligations.

### 4.3 — State derivation

For each state port, the contract declares:

- **Conditional presence, in both directions.** Arguments may add or remove a port. Non-causal
  attention has no cache; the same encoder gains one when it attends across the fragments of a
  fragmented stream.
- **Payload.** Each storage component has a shape, type, and multiplicity. One state may contain
  several components of different types (O5.1). A payload is declared **per position** of the
  stream the state grows along: one cached position for `append` and `window`, whose extent is the
  law's and whose span is a modulator; the whole state for `fixed`. A payload therefore carries a
  `sequence.position` axis only if every rule of its port is `fixed`; a contract that does otherwise
  is refused when the catalog is loaded.
- **Derived properties.** These are consequences of arguments and never appear in the model:

| Property | Reference | Meaning |
|---|---|---|
| **Evolution law** | O5.2 | How extent evolves: append, bounded window, or fixed size. |
| **Indexing domain** | §5.3 | The stream along which the state grows: the occurrence's own (`self`) or that of one of its input ports (`port`); the law alone is ambiguous. |
| **Access geometry** | O5.3 | `logical_position`, `ring`, `aggregate`, or `selected` (positions chosen per query by the primitive), as consumed properties rather than runtime data-structure names. |
| **Sharing** | O5.3 | The granularity at which sessions may share. |
| **Key axes** | O5.5 | The instance axes (session, branch) along which allocations of this port are distinct; with the identity's indices they form the instance key (§4.4). |
| **Permitted operations** | O5.4 | The effects the state admits: read, append, evict, write. |
| **Modulators** | O5.8 | Span and stride, expressions over the arguments. |
| **Carrying** | §5.3 | A condition over the arguments under which the state survives between the invocations that deliver successive fragments of its stream (`carried_across`); declared once per state port, not per rule. A state indexed by a source stream is frozen once that stream is complete, by definition. |

The condition language is closed and decidable (O0.6): presence tests, comparisons of expressions,
negation, conjunction, disjunction, and the constants. An argument that may be absent — optional,
without a declared default — may be compared or computed with only under a `present` test of it; a
contract that does otherwise is refused when the catalog is loaded. Rules are ordered; the first
matching rule wins; a present state port that no rule matches for an occurrence's arguments is a
rejection of the document (V9).
Conditions apply to argument **combinations**: two coexisting mechanisms produce two ports, while
either mechanism alone produces one.

A flattened shape declares its decomposition (O5.10). Flattening can hide a partition axis. Without
a decomposition, placement derivation must use the non-partitionable case and report **information
loss**, never present non-partitionability as a known fact.

### 4.4 — Information supplied by the graph

> `instantiated states = contract(primitive, arguments) × graph occurrences × state identities`

A contract supplies the descriptor; the graph supplies instance count and relationships. The model
declares only what no derivation can produce: which state ports name the same storage (a state
identity with several members) and which repetition indices distinguish allocations (the identity's
indices). Whether a state survives between fragments follows from the contract's carrying condition
and the input's fragmentation (§5.3). Everything else about instances is a consequence, computed by
the same rule for every document; a field the author could only fill by copying that rule is not a
declaration (§1.2, §10.2).

**State liveness** is the number of distinct state-allocation equivalence classes that may be active
at once. It sizes simultaneous state memory. This is distinct from **value liveness**, which records
the graph values live across a cut and determines that cut's logical payload.

| Element | Reference | Authority |
|---|---|---|
| **Instance key** | O5.5 | **Derived:** the identity's indices × the contract's `key_axes` (session, branch). Sharing is declared by listing several members under one identity, never by a relation over keys. |
| **Liveness law** | O5.5 | **Derived:** one class per distinct instance key. How many classes are active at once is deployment intent (§10.3): the model fixes the class structure, never the count, and never the raw product of dimensions. |
| **Visit rate** | O3.2 | **Derived** per indexing domain and phase (§7), from the value graph, the generative outputs (§2.3) and the fragmentation of inputs (§5.3). Visits size computation; liveness sizes memory. Counts — tokens per request, fragments per stream — are deployment intent. |
| **Cardinality** | O5.7 | The model: any finite number of states with distinct natures, with no language-defined upper bound. |

A model document therefore never references a deployment or request fact: there is no open
`context` namespace. Every reference in a document resolves to one of its own quantities, indices
or arguments (V1); a derived product such as D4 takes deployment intent as a separate input.

### 4.5 — Structured sparsity *(O6.6)*

When a primitive activates only some of its parameters per element, its contract declares one or
more **sparsity units**: the parameter slots that form one unit and the axis along which units are
laid out; the **policy** that selects units — an argument, the value on an input port, or the
element itself (its position, its type); the count **activated per element**; and an **upper bound
on the union of units activated per invocation**. A lookup is the limiting case: unit = one row of
the table along the lookup axis, one activated per element, union bounded by the axis extent. Cost
derivation counts a unit's weights at the activated fraction per element, and in full for residency
and worst-case transfer.

A per-element count is insufficient because an invocation's union may include every unit. This
yields three separate qualified quantities: exact resident cost, upper-bounded worst-case transfer,
and estimated expected transfer — the last an input with status `estimate`, never derived (O0.5).

### 4.6 — Template contracts

A contract may be implemented by a **template**—a parameterized model document—instead of a consumer
capability. The catalog designates the template; it does not describe it:

```json
"decoder.causal_yarn": { "version": "1.0.0",
  "template": { "name": "decoder-causal-yarn", "version": "1.0.0",
                "id": "decoder_causal_yarn" } }
```

A template contract's **ports** are the template's public inputs and outputs, with their domains;
its **arguments** are the template's external quantities, with their types, domains and defaults;
`template.id` is the template's `model` identifier. The template is a document
`<name>/<version>.json` in the location the base manifest declares (`templates`, relative to the
base); a reference resolves only when that document exists and carries the pinned name, version and
id (V1). Template occurrences become occurrences of the calling graph with identifiers prefixed by
the invoking occurrence (§5.2).

The template is not the contract identity: two templates may realize one semantic identity, and a
template name identifies a realization, not a meaning. The realization is nevertheless pinned —
the contract names the template version, and the template carries that version — and how template
and contract versions relate is defined in §8.2.

A parameterized document denotes a family, not one graph. Its external-source quantities are
contract arguments supplied by an **assignment**, except those with declared defaults:

> For every admissible assignment, a parameterized document denotes exactly one graph.

Admissibility must be decidable at the call site. The contract therefore exposes the declared types
and domains of external quantities, and assignments are checked like other arguments (I7). Omitting
those domains would make denotation silently partial.

An external quantity may declare a default in the template, which the contract exposes as a
primitive default (§4.1). The default is a §2.2 expression over literals and **other arguments**;
its dependency graph is acyclic. The argument becomes optional without violating I7: a declared
default is not a silent default. Adding an external quantity with a declared default is therefore
backward-compatible and requires no change to existing call sites.

Delegation removes the need for a new consumer capability (§8.1), but none of §4.1's contract
contents. Logical tensors, state ports, logical costs, and semantic partitions are derived from the
expanded template. Declaring them empty would be false: the template may contain a sequence operator
and most of the invoking model's parameters, with counts dependent on the assignment.

> **Delegation moves derivation; it never removes the obligation to derive.**

The contract citation graph is acyclic. A template may cite another template contract, but a
cycle is a reasoned rejection (§8.1), not recursion.

Two instances of a template share neither state nor tensors. Sharing across the boundary is not
expressible (§1); any future form must use ports, never common tensor identities.

## §5 — Denotation

### 5.1 — The expanded graph is authoritative

A document denotes one finite occurrence graph. Families, stacks, trunks, repetitions, and patterns
are syntactic sugar with the expansion defined here. All validity rules apply to the expanded graph.

A **composition** is a named, finite, indexed family of occurrence sites. Expansion evaluates
its index ranges and conditions, then emits ordinary occurrences with deterministic identifiers; a
composition is not a runtime node or an independently nested graph.

### 5.2 — Deterministic expansion

These rules are normative:

1. **Resolvable counts:** every repetition count resolves to an integer from declared quantities. A
   count depending on a runtime decision cannot be expanded and is rejected.
2. **Identifiers.** A root occurrence is identified by its name; a generated occurrence by
   `<composition>/<site>[<index>=<value>,…]`, indices in name order; an occurrence of a template by
   `<instance>/<identifier in the template>`. A scoped rule `R` of composition `C` is `C.R`; an
   identity it names without declaring one is `C.R[<index>=<value>,…]`. These identifiers cannot
   collide with handwritten ones. They are representation: two expanded graphs are the same when a
   one-to-one correspondence of occurrences preserves contracts, arguments, families, every value
   edge, the grouping of slots into parameter, constant and state identities, the interface names
   and the identity names.
3. **Resolved references.** Every quantity, occurrence, port, family, identity, stream, catalog
   base and template reference resolves (V1). A binding is emitted only where every occurrence it
   names is emitted; an occurrence absent by its guard is not a reference failure, an unknown name
   or an index outside a composition's ranges is one. A rule naming several members is emitted
   where all are.
4. **Set semantics.** The expanded graph is a set of occurrences, edges, families and identities;
   expansion is deterministic as a set, whatever the order of members in the document. Any listing
   is conforming; the canonical listing orders occurrences by identifier, edges by (source,
   destination), identities by name.
5. **Trunks.** A trunk is a maximal connected component of the expanded value graph; several may
   coexist (O3.3), and a trunk may carry no state.
6. **Guards:** a site or a binding may carry a `when` condition over the document's quantities and
   the indices in scope; it is evaluated at every index, and a site or rule whose guard is false is
   not emitted. A guard that cannot be decided is a rejection (V10), never false: an undecidable
   guard would otherwise drop occurrences silently (I7). Guards are how a periodic layer pattern is
   written as one composition — a site present when `layer mod 4 = 3`, a carry edge present when
   `layer ≥ 1` — instead of one site per case.
7. **Scoped bindings:** a composition may carry bindings written against its own sites. Each such
   rule `R` of composition `C` denotes exactly the top-level rule `C.R` whose `for_each` is `C`'s
   index ranges and whose site endpoints select the generated occurrence at the current indices,
   overridden index by index where the endpoint says so; a scoped parameter or state rule without a
   declared identity names it `C.R`, indexed by `C`'s indices. Nothing is expressible in the scoped
   form that is not expressible at the top level; the scoped form only removes the repetition of the
   composition's ranges and selectors, and every reading expands it before any other rule applies.
   The emitted rule carries the presence of its endpoints by rule 3. A scoped rule whose expanded
   name collides with a top-level rule is rejected (V12).
8. **Templates.** An instance expands its template under the assignment formed by its arguments;
   the template's inputs are fed by the instance's input edges and its outputs feed the instance's
   output edges; two instances share nothing.

The concrete representations use three related but non-interchangeable condition fields:

- model-level `when` is evaluated from model quantities and in-scope composition indices and
  controls whether a graph site or binding is emitted;
- contract-level `present_when` is evaluated from resolved primitive arguments and controls whether
  a port, parameter, constant, or state slot exists;
- `when` on an ordered contract rule is also evaluated from resolved arguments and controls whether
  that rule applies to a present element.

All conditions that affect denotation must resolve deterministically. A field from one context
cannot substitute for a field from another.

### 5.3 — Indexing domains, streams and fragmentation

An **indexing domain** is a pair (kind, stream). The **kind** says what one element is: `sequence`
— one element per sequence; `token` — one per token of a text sequence; `position` — one per frame
of a sampled signal; `patch` — one per image or video patch. The **stream** is the public input that
introduced the elements. A public input introduces the stream named after it, or joins an existing
one.

A contract port declares a kind, or `inherit`. The **occurrence's own domain** is the common domain
of its input ports that are not transformed; V5 requires them to agree, so an elementwise primitive
with two inputs has one domain. An input port declaring a kind constrains the kind of the edge into
it and takes the edge's stream; an output port declaring a kind has that kind and the occurrence's
stream unless a transform says otherwise; a port declared `inherit` from an input port has that
port's domain.

An occurrence whose every input port is transformed has no own domain; its outputs' domains come
from its transforms.

A **transform** relates a port to the occurrence's domain: `merge` — the output carries the input
port's stream at the output's kind (which may be the same kind), one element per `factor` input
elements; `align` — the input port carries another domain and the output stays in the occurrence's
domain; `insert` — the input port's elements enter the occurrence's stream, which the output keeps.
Transforms carry element counts: after a `merge` a stream has one element per `factor`; after an
`insert` it has the inserted stream's elements in addition.

**V5:** on every edge the source port's (kind, stream) equals the destination port's, unless the
destination is the `from_port` of a transform of its contract. A public input is an edge like any
other.

Value edges form an acyclic graph within one invocation. Time recurrence must pass through a state
port, never an implicit combinational cycle.

A public input may be **fragmented**: its elements arrive over several invocations. A state whose
carrying condition holds (§4.3) survives between the invocations that deliver successive fragments
of its stream; the graph stays finite per invocation. A state indexed by a source stream
(`indexed_by: {port}`) grows along that stream, across its fragments, and is frozen once the stream
is complete. A self-indexed state that is not carried is reset at each fragment of its stream. An
occurrence whose state is carried must sit on a fragmented stream (V16); nothing is declared twice.

An occurrence that reads across positions of a fragmented stream (its contract's `across_positions`
condition holds, §4.1) must carry a state across that stream's fragments — a present state port
on that stream, self-indexed with its carrying condition holding, or indexed by the port that
carries the stream — or the fragments would compute something the whole stream would not; the
document is rejected (V18). A `merge` transform on a fragmented stream reads whole groups of
`factor` elements and nothing across them: its invariance under fragmentation holds when every
fragment delivers whole groups, so a fragmented stream has a **fragment alignment** — the least
common multiple of the cumulative merge factors of the values on it — that every fragment
delivers a multiple of. It is a deployment obligation derived by the language and reported with
the stream (D2), never declared: a declared alignment would copy that rule (§4.4).

## §6 — Static semantics

A document is valid only if every rule below holds. Failure produces a reasoned rejection, never an
implicit default.

| Rule | Requirement |
|---|---|
| **V1** | Every quantity, occurrence, port, family, identity, stream, catalog base and template reference resolves (I1). Two bases carrying one `{name, version}` with different contents are a conflict, not a choice. |
| **V2** | Every required contract argument is present; every undeclared argument is rejected; declared defaults are applied before checking. |
| **V3** | Types, enums, record fields, units and domains conform, for arguments and for quantities, derived ones included; an enum value outside its declared set is rejected; a number literal is an instance of its declared type (`32.0` is not a cardinality); a quantity selecting a dtype is an enum over dtypes. |
| **V4** | Shapes compose: source and destination shapes have the same rank and, position by position, the same axis identity and equal extents; local names and natures are not compared; factors are compared when both declare them. The same rule binds a constant slot to its constant and the members of a parameter identity. |
| **V5** | Indexing domains agree on every edge (§5.3). |
| **V6** | The value graph is acyclic within an invocation. |
| **V7** | Bindings are total and unique: every input port of an emitted occurrence is fed exactly once; every parameter, constant and state slot present under its `present_when` is bound exactly once; a slot absent by its condition is bound by nothing. |
| **V8** | Conditional argument combinations satisfy the contract. Meaningless combinations are excluded by invariant, never merely by a missing guard (I11). |
| **V9** | A state identity connects only compatible ports: the same applicable rule, the same key axes, and equal payload shapes and indexing domains; a present state port that no rule matches is rejected. |
| **V10** | Every repetition, guard and derivation resolves under §5.2 and §2.2. |
| **V11** | A literal quantity that declares its derivation agrees with it (I8). |
| **V12** | Every construction has one reading; references and literals are unambiguous (I5, I6); a document or unit with duplicate member names in any object is rejected. |
| **V13** | Every output port of an emitted occurrence is consumed by an edge or exposed by a public output. |
| **V14** | A dtype selected for a parameter identity is admissible for the role of every member; one selected for a state identity, for the role of every payload component of every member; absent, each role's default applies. |
| **V15** | Parameter identity compatibility (§3.4). |
| **V16** | An occurrence whose state is carried across fragments (its contract's `carried_across` condition holds) sits on a fragmented stream. |
| **V17** | Locations are total or absent: a document with one located parameter identity locates every parameter identity instance. A physical name is bound by one identity; the slices of one physical tensor do not overlap and do not coexist with a whole binding of it; a `stack` names an axis of the slot and its part carries that coordinate; a `slice` offset resolves to a non-negative integer, and a slice is not a part of a concat; a document that locates its weights does not instantiate a template. Against a checkpoint: every located tensor exists with the D3 shape — unit axes the physical tensor has and the logical shape lacks being dropped — and the D3 dtype (I9). |
| **V18** | An occurrence whose contract reads across positions (§4.1), on a fragmented stream, carries a state across the fragments of that stream (§5.3). |

## §7 — Required derived products

An **invocation** is one evaluation of the expanded graph on one delivery of its inputs: all
elements of a non-fragmented input, one fragment of a fragmented one; when an output is generative,
each generated element is delivered to the output's stream in the next invocation. An input may
deliver zero elements in an invocation — its stream complete, or unused by that invocation. An
occurrence is evaluated in an invocation when every input port receives elements, excepting a
port that is the source of an `insert` transform of its contract, and a port whose elements a
state of the occurrence indexed by that port already holds in full (an `append` state; a `window`
holds a suffix and exempts nothing); an occurrence not evaluated delivers
nothing downstream, and a state indexed by a stream that delivered nothing is not visited. An
input is **required** for an output when, on a first delivery, the output is not evaluated without it. A generative
document has two **phases**: the invocations consuming supplied elements (prefill) and those
consuming one generated element (decode). A **cut** is a partition of the emitted occurrences into
two blocks; it is **legal** when every crossing edge is directed from the first block to the second.
The **payload** of a cut is the set of values on its crossing edges — the values live at the cut —
each sized by its port shape times the number of its domain elements in the invocation, element
counts following the transforms of §5.3. A state instance is **visited** in every invocation that
evaluates a member occurrence: an `append` or `window` state indexed by its own stream is written
once per new element and read once per element produced; a state indexed by a source stream is
written once per source element and frozen when the source is complete; a `fixed` state is read and
written once per element. Element counts are deployment intent.

A valid document and its referenced contracts make all products below computable without inference
code or human knowledge of a named mechanism:

| Product | Content |
|---|---|
| **D1** | **Expanded graph:** occurrences, edges, and families. |
| **D2** | **Values:** the value and shape inventory; the payload of every legal cut — the values live at it, sized per invocation; the peak of live values along one order of the graph, the activation peak of an invocation; and the fragment alignment of every fragmented stream (§5.3). |
| **D3** | **Parameter tensors:** shapes, sharing, and total count; the role, selected dtype and sensitivity of every tensor; when the document locates its weights, the evaluated location of every tensor. |
| **D4** | **Complete state:** descriptors, instances, keys, state liveness, visits per phase, and permitted operations. |
| **D5** | **Logical costs:** parameters, activations, state per element, computation — derived from the inventory and the declared corrections (§4.1) — and the payload crossing each legal cut per invocation. |
| **D6** | **Legal cuts and semantic partition axes:** the legal cuts of the expanded graph, and for every occurrence the partitions its contract declares with their communications and granularity; a flattened axis without factors is reported as information loss (O5.10). Partitions are declared per occurrence; their consistency across occurrences — the residual width through norm, add and feed-forward, a head partition aligned to the KV groups of its layer — is compilation's (§10.3), the axis identities on D1's edges being what a compiler aligns. |

These are compilation outputs. Their encoding and the decisions that consume them are outside this
specification.

## §8 — Extension and rejection

### 8.1 — Extension kinds

Extensions have different effects:

| Extension | Existing documents | Consumer |
|---|---|---|
| **New primitive** | No breakage. | A new capability when a model uses it. |
| **New argument** that is optional or has a declared default | No breakage. | A new capability. |
| **New value** of a closed derived property | No breakage. | A new, explicitly rejectable capability. |
| **Template contract** (§4.6) | No breakage. | No new capability if the transitive closure of its template is supported; otherwise the cost is exactly the missing primitives, never the composite. |

These columns describe different concerns. Backward-compatible additions are not free: consumers
must implement them when a model uses them. Syntactic position—open argument or closed
property—does not measure implementation cost.

Only templates can have zero consumption cost, which is how the catalog remains open. The
first three extensions add vocabulary that must be implemented. A template composes existing
vocabulary. Its cost is computed from the transitive closure of cited contracts; a template that
cites an unsupported primitive merely hides that cost.

Every primitive, argument, field, reference, or combination that a contract cannot interpret is
**rejected with a reason**. No implicit interpretation or default is allowed. The representation of
rejection—code, exception, or return value—is implementation-defined.

Rejection is exhaustive in both directions: a missing required argument and an argument absent from
the contract are both errors. Otherwise a consumer could silently accept information it does not
understand.

### 8.2 — Identity and versioning

Capability rejection and contract identity are orthogonal. A published `{name, version}` is one
immutable file; a pin names exactly one file, and resolution is exact. Every change is a new version
file whose digits classify the change:

| Digit | The publisher asserts |
|---|---|
| **patch** | The declared meaning is unchanged — the unit's prose and cited sources — and no product (D1–D6) changes for any existing occurrence: documentation, notes, a template rewritten to the same family. The witness may change only where it was wrong against the declared meaning or against the integration fixtures of a model that uses the contract (§10.2); its unit fixtures are then re-recorded, and a conformer that matched the old witness may now disagree — the accepted cost of a patch, stated in the version note. |
| **minor** | Additive only: an argument with a declared default, an optional port, a new enum value, a new partition entry, a correction that did not apply before. Every existing occurrence denotes, derives and computes the same, and no implementation of the previous version breaks. |
| **major** | An existing valid occurrence would denote, derive or compute something different, or an implementation of the previous version breaks. |

Pins stay exact for reproducibility. A consumer implementing `name X.Y` may accept any pin `X.Y'`,
`Y' ≤ Y`, and rejects the rest explicitly (§8.1). A template is versioned the same way, in its own
file; a contract pins one template version; a template rewrite is a patch of the contract pinning
the new version, and earlier contract versions keep resolving. "Same family" is asserted by the
digit and may be checked on any assignment by expansion (§5.2 rule 2). **Specification stage:**
until the first release of a catalog, its units are edited in place at `1.0.0`.

A compatible addition need not change the model-language version, but every prior contract identity
must remain immutable. Otherwise occurrences cannot reference reproducible meanings.

### 8.3 — Benefit of closure

An extension's cost is bounded and known in advance. This is not an expressiveness obligation—no
language element carries it—but the criterion that justifies the mechanism.

## §9 — Invariants and non-requirements

### 9.1 — Invariants

| Invariant | Requirement |
|---|---|
| **I1** | Every reference resolves. |
| **I2** | Declared properties and present fields are coherent; an inapplicable field is forbidden, not merely ignored. |
| **I3** | Every input port is fed; writes target only ports. |
| **I4** | Every bound resolves to a value or a closed §2.2 expression. |
| **I5** | References and literals are unambiguous. |
| **I6** | Every construction has one reading. |
| **I7** | **No silent defaults:** every non-derivable answer produces a reasoned rejection. |
| **I8** | A literal value and its declared derivation agree. |
| **I9** | The described model and loaded artifact are mutually compatible: every logical tensor is located, and no physical tensor is bound twice (V17). |
| **I11** | A meaningless combination is excluded by invariant, never merely by a missing guard. |

I5 and I6 are expressiveness invariants, not concrete syntax rules. Known conformance cases include
a dimension named like an enum value and a two-key object indistinguishable from a two-entry map.

I10—detecting a decision based on stale measurements—belongs to deployment control and is outside
this specification.

### 9.2 — Non-requirements

| Rule | Exclusion | Reason |
|---|---|---|
| **N1** | No tensor algebra: no value expressions, loops, or indexing. | This is the level of an IR; §2.2 shape arithmetic remains allowed. |
| **N2** | No generator, backend or kernel scheduling. | These belong to generators. |
| **N3** | No training. | It is a separate problem. |
| **N4** | No inline numeric tensors. | See §3.3. |
| **N5** | No resource topology. | It can change without changing the model. |
| **N6** | No hardware assignment: placement, resolved affinity, or cross-domain composition rules. | They depend on parallelism chosen outside the model. |
| **N7** | No scheduling or preemption policy. | The language makes such policies computable; it does not choose them. |
| **N8** | No runtime data structures such as pages, tables, or index layouts. | The model declares logical geometry; a runtime declares its cost. |
| **N9** | No fault tolerance, recovery, hot redistribution, or reservation protocol. | See §1. |

## §10 — Coverage and rejection cases

Every requirement maps to the language construction that carries it. Examples illustrate rules; they
never define them.

### 10.1 — Corpus cases

The corpus is evidence, not definition (§1). Each case below names the construction that carries it:

| Case | Property tested | Construction |
|---|---|---|
| Homogeneous dense decoder | Trivial case | §3, §5.2 |
| Regular heterogeneous pattern | Families and instance counts | §3.1, §4.4 |
| **Three simultaneous state natures** | Unbounded cardinality and different operations | O5.7, §4.3 |
| **Four states, irregular pattern, residual multiplicity** | Multiplicity and conditional combinations | §3.5, §4.3 |
| **Non-generative, multimodal, tied weights** | Non-generative output, tensor sharing, and multiple trunks | §2.3, §3.4, §5.2 |
| **Mixture of experts with batch > 1** | Batch union differs from per-element count | §4.5 |
| **Flattened shape** | Lost partition axis is reported | O5.10 |
| **Encoder-decoder** | State read by another trunk | §3.4, §5.3 |
| **Non-adjacent layer cache sharing** | State identity | §3.4 |
| **Fragmented streaming input** | Fragmentation and carrying | §5.3 |
| **Per-token output** | Output indexing domain | §2.3, §5.3 |

### 10.2 — Required rejection cases

The following are rejected at the stated level: unresolved reference, or one identity carried by
two bases with different contents (V1); missing required argument (V2); undeclared argument (V2);
out-of-enum value, or a literal that is not an instance of its type (V3); incompatible shapes (V4);
incompatible indexing domains (V5); value cycle (V6); unfed port, unbound tensor or state slot, or
a port or slot fed or bound twice (V7); meaningless combination (V8); state identity between
incompatible ports (V9); unresolvable repetition, guard or derivation (V10); a literal quantity
disagreeing with its declared derivation (V11); duplicate member names in an object (V12); an
output consumed by nothing (V13); an inadmissible dtype (V14); incompatible members of a parameter
identity (V15); a carried state on a stream that is not fragmented (V16); and an occurrence
reading across positions of a fragmented stream with no state carried across its fragments (V18).

A catalog is refused when loaded if a contract compares or computes with an optional argument that
has no default outside a `present` test of it; if a precision role's default is not admissible; if
a sparsity policy names an argument or input port the contract does not declare; if a state payload
carries a `sequence.position` axis under a rule that is not `fixed` (§4.3); if a template
contract's template is absent from the declared location or carries another name, version or id.

**Release rule.** A catalog is released — its base tagged — only when every contract version it
publishes has a witness (§4.1); until then it is at the specification stage (§8.2), and the
reference generator's coverage report lists the versions without one. The witness is the authority
for what a primitive computes; the authority for whether the witness is *right* stands above it:
the contract's declared meaning — its description and the sources it cites — and the integration
fixtures recorded from the delivery implementation of a model that uses the contract. A witness
corrected toward that authority is a patch of the contract (§8.2); a witness that changes what a
contract means is a new identity.

**Mutation test:** for every normative construction, removing a field, breaking its reference, or
changing its value must either change the expanded denotation or cause rejection. A field ignored by
every conforming implementation is a comment, even if the document remains valid.

### 10.3 — Explicitly separate concerns

| Concern | Authority |
|---|---|
| Configuration transition classes and costs | Deployment control |
| Decidable representation of a configuration space | Deployment control |
| Request admission and length commitment | Online control |
| Dependency graph between consumer questions | Deployment control |
| Cross-memory-domain composition of a quantity | Compilation for selected parallelism |
| Load variables such as batch and concurrent sequences | Deployment intent |
| Physical artifact encoding: file format, sharding, quantisation containers. The mapping of physical tensors onto D3 identities is the document's own (`location`, §3.4, V17) | Artifact format |

## Appendix A — Requirements

The identifiers cited throughout this document, stated once. A citation names the construction that
carries the requirement; this appendix is the requirement. Gaps in the numbering are reserved.

| Id | Requirement |
|---|---|
| **O0.1** | Derived quantities are expressed in the closed scalar algebra of §2.2. |
| **O0.2** | A derivation outside the algebra uses a normative interface with declared inputs, semantics and conformance tests; its result is at least a qualified bound. None is defined in this revision. |
| **O0.3** | A derived quantity of a model is exact, reads only declared quantities, resolves acyclically and conforms to its declared type and domain; epistemic status and provenance are declared on contract costs and derived products. |
| **O0.4** | Quantities form one flat namespace: each is declared once and referenced by name everywhere. |
| **O0.5** | A qualified value carries a status — exact, upper bound, lower bound, or estimate — that propagates through the algebra by the table of §2.2; rounding follows the status; no propagation turns an estimate into a bound. |
| **O0.6** | Every language of this specification is closed and decidable, or uses a normative interface. |
| **O1.1** | A contract is referenced through an immutable identity, `{name, version}`, whose meaning never changes (§8.2). |
| **O1.2** | A model document carries a stable, authoritative identifier. |
| **O1.3** | Every published contract version has one witness — a reference implementation supplied with the contract, executed by the reference generator and bound to the identity by its manifest with a tolerance per compute dtype — that is the authority for what the primitive computes; every other implementation is a conformer, checked against it (§4.1, §4.2). |
| **O2.1** | Every quantity and every occurrence has one stable identifier, unique after expansion. |
| **O2.2** | Every quantity declares its type and its source; it is a model variable when its source is external or reads one, transitively, and then declares a domain; otherwise a model constant. |
| **O2.3** | Every quantity declares its dimensional type; physical quantities carry a unit. |
| **O3.2** | The rate at which a state is visited, per indexing domain and phase, is derived from the graph; counts are deployment intent (§4.4, §7). |
| **O3.3** | Several trunks may coexist in one document; a trunk may carry no state. |
| **O3.5** | Every occurrence belongs to one or more named families, addressable without enumeration. |
| **O4.1** | Every value port has a shape and a role; residual multiplicity is a shape axis. |
| **O4.2** | Every flow is seeded explicitly by a public input; implicit seeds are forbidden. |
| **O4.3** | Residual multiplicity other than one is explicit, and the combining operation is a contracted primitive. |
| **O4.4** | Value flow is declared by explicit port-to-port edges, never inferred from ordering or from mutation of a named flow. |
| **O4.5** | The logical payload of a cut is the set of values live across it, determined by the explicit edges; it is not measured traffic. |
| **O5.1** | A state payload may hold several components of different shapes and types. |
| **O5.2** | Each state port derives an evolution law: append, bounded window, or fixed size. |
| **O5.3** | Each state port derives its access geometry — logical position, ring, aggregate, or selected — and the granularity at which sessions may share it, as consumed properties, never as runtime data-structure names. |
| **O5.4** | Each state port derives its permitted operations: the effects it admits. |
| **O5.5** | The instance key of an allocation is derived from the identity's indices and the contract's key axes; liveness is one class per distinct key, never the raw product of dimensions. |
| **O5.7** | A model may declare any finite number of states of distinct natures; the language sets no upper bound. |
| **O5.8** | Modulators of a state — span, stride — are expressions over the primitive's arguments. |
| **O5.10** | A flattened shape declares its decomposition into named axes; without one, placement derivation reports information loss rather than asserting non-partitionability. |
| **O6.1** | Every contract derives its logical tensor inventory from its arguments: shapes, roles and sharing rules. |
| **O6.2** | Parameter bindings identify the logical tensors each occurrence consumes and every sharing between occurrences; a tied tensor is counted once. |
| **O6.6** | A primitive that activates only some of its parameters per element declares its sparsity units: the activatable unit, the policy that selects units (an argument, an input port, or the element itself), the count activated per element, and an upper bound on the union activated per invocation. |
| **O7.1** | A contract declares the axes along which partition preserves meaning, the logical communication each partition implies — several when the partition admits several patterns — and the granularity a shard keeps whole along the axis. |
| **O8.1** | Every public input names one or more occurrence ports and declares its kind and stream; every public output names one occurrence port and whether it is generative. |
| **O8.2** | A document may expose several outputs at once, including non-generative ones. |
| **O8.3** | An auxiliary output is an additional reference to an existing value, never a new operation. |
| **O9.1** | A model names primitives and supplies arguments; it never describes a primitive's computation. |
| **O9.2** | A contract declares the elements of §4.1: arguments, ports, logical tensors, state ports, effects, corrections, partitions and transforms. |
| **O9.4** | An argument is a scalar, an enum, or a record of such values; inline numeric tensors are forbidden. |
| **O9.5** | A contract declares, as a condition over its arguments, whether the primitive reads positions of its stream beyond those of the element it produces; an occurrence that does so on a fragmented stream carries a state across the fragments (V18). |

The I‑ and N‑ series are stated in §9. O2.4 and O5.9 are withdrawn; their numbers are reserved.
