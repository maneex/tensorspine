# Tensorspine Language Specification

> Represent every in-scope model as a **finite graph of parameterized primitive occurrences**, then
> derive all other information from the primitives' contracts.

*Tensorspine 2.0 — language specification, revision of 29 August 2026. The language version is the
`tensorspine/2.0` tag every document and schema carries; catalog units carry their own versions
(§8.2). This document is the sole normative authority for the language. It is self-contained:
syntax, validity, and denotation do not depend on source code, tools, or other repository artifacts.
Every requirement identifier it cites (O‑, I‑, N‑) is stated in Appendix A or §9.*

For motivation and repository orientation, read the [README](../README.md). For a field-by-field
authoring guide, read [Tensorspine model JSON](TENSORSPINE-MODEL_JSON.md). The
[glossary](GLOSSARY.md) summarizes terminology and points back to its canonical sections; the
[architecture guide](ARCHITECTURE.md) explains the design rationale. None of those explanatory
documents adds requirements to this specification.

## §1 — Scope and authority

This document specifies what a model document can express and what a primitive contract must contain
so all consequences can be derived.

**In scope:** inference; infrastructure fixed for the duration of an analysis; a graph that is
**statically finite per invocation**; and the architectures of the reference corpus, `data/models/`
in the repository, which is the coverage set this revision answers for. Candidate architectures are
drawn from what serving engines such as vLLM run at a given date; “all vLLM models” is not a finite
set, the corpus is, and a model that cannot be written is a counter-example to be filed against it.

**Out of scope:** training; fault tolerance and session recovery; hot reconfiguration; resource
topology; runtime data structures; placement; and hardware assignment.

The coverage claim is relative and falsifiable:

> Every finite graph in this scope, composed of primitives with available contracts, has **at least
> one** representation.

This does not claim coverage of every present and future model. An open primitive vocabulary admits
new **nodes**; coverage of graph **topologies** is bounded by the graph class of §5.3: a finite
directed acyclic value graph per invocation, with recurrence only through state ports.

### 1.1 — Required properties

| Property | Meaning | Status |
|---|---|---|
| **Functional denotation** | A document denotes exactly one graph: denotation is total and unambiguous. A parameterized document denotes one graph **per admissible assignment** (§4.6). | **required** |
| **Coverage** | Every in-scope graph has at least one representation. | **required** |
| *Unique representation* | Each graph has exactly one representation. | *not required* |

Functional denotation is not injectivity. Injectivity would forbid two documents from denoting the
same graph; Tensorspine explicitly does not require a canonical representation.

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
| **Catalog bases** | The ordered catalog bases whose units the occurrences pin, as locations relative to the document; a base that does not exist is a rejection (V1). Contract identity is per occurrence, `{name, version}`, immutable (O1.1, §8.2); there is no catalog-wide version. |
| **Quantities** | The namespace of dimensions and scalars (§2.1). |
| **External constants** | Non-learned numeric data with identity, shape, and type (§3). |
| **Occurrences** | Graph nodes (§3). |
| **Compositions** | Finite indexed families of occurrence sites whose expansion is normative (§5). |
| **Bindings** | Value edges, tensor bindings, constant bindings, and state identities (§3.4). |
| **Public inputs and outputs** | Interfaces with indexing domains (§2.3). |

### 2.1 — Quantities *(O2.1, O2.2, O2.3, O0.4)*

Quantities form one flat namespace, declared once and referenced everywhere. Each quantity has three
independent properties:

- **Regime:** a model constant known when written, or a variable with a declared domain.
- **Dimensional type:** a cardinality (non-negative integer); a ratio or real hyperparameter bounded
  by contract; a physical quantity with a **unit** (bytes, elements, tokens, seconds, operations);
  or an enum or boolean, which participates in arithmetic only through an allowed conditional.
- **Provenance:** a literal value, read from configuration, optionally with the **derivation** by
  which it follows from other quantities (V11 checks the two agree); an **external** value supplied
  by an assignment (§4.6), with a declared domain and optionally a declared default; or a
  **derived** value, an expression of the algebra of §2.2 over other quantities.

Regime and type are independent: a model constant need not be an integer.

When several authorities constrain a domain—model capability, implementation limit, and deployment
choice—each declares its own constraint, and the binding constraint must remain identifiable (O2.4).
Load variables such as batch size and sequence count do **not** belong in the model document.

### 2.2 — Derivation algebra *(O0.1, O0.2, O0.3, O0.5, O0.6)*

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

**O0.5 — Epistemic status is an algebra, not a label.** Where a product combines qualified values,
status propagates according to the operation's monotonicity: dividing by an upper bound produces a
lower bound. Rounding direction belongs to the status, never to the operator. Monotonicity is
relative to a domain, and each node must prove that it stays within that domain. Propagation never
turns an estimate into a bound.

**O0.6 —** Every language introduced by this specification is closed and decidable or uses a
normative interface. This applies to the size algebra, the contract-condition language in §4.3, and
the expansion rules in §5. Calling a representation “symbolic” does not make it decidable.

### 2.3 — Public inputs and outputs *(O8.1, O8.2, O8.3, O4.2)*

Each public input and output declares:

- the attached occurrence and port;
- its indexing domain (§5.3), such as per token, sequence, patch, or fragment;
- for an output, whether it is generative.

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
version. Capability rejection (§8.1) answers “Can this consumer execute it?”
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
parameters, is counted and loaded once, and must not appear twice in an artifact.

**State identities** declare when occurrences use the **same storage**. Sharing is a topological fact
that no primitive contract can know: the contract derives a state descriptor, while the graph
declares which occurrence reads another occurrence's state.

> **Use one construction for every case.** Non-adjacent layer cache sharing, cross-attention reading
> encoder state, and state carried between fragments are all state-port identities plus an indexing
> domain (§5.3). Case-specific fields would reintroduce redundant semantics.

Each fact has one authority. If a primitive argument determines a binding, no other field redeclares
it; if the binding is declared, no argument duplicates it.

### 3.5 — Flow multiplicity *(O4.1, O4.3)*

A value port has a shape, type, and multiplicity. Residual multiplicity other than one—for example,
parallel residual streams combined by an operation—is explicit. The combining operation is a
contracted primitive, not an annotation.

## §4 — Primitive semantic contracts

A primitive contract is the immutable, versioned semantic definition referenced by an occurrence.
It declares the primitive's arguments and derives its ports, logical tensors, state, effects, costs,
and legal partitions. Contracts provide coverage: a new primitive can be added without changing the
language.

### 4.1 — Contract contents *(O9.2, semantic part)*

| Element | Content |
|---|---|
| **Semantic identity** | The operation's meaning and immutable reference identity. |
| **Arguments** | Types, allowed values, required status, declared defaults, and cross-constraints. |
| **Value ports** | Inputs and outputs with shape functions. |
| **Logical tensors** | A symbolic learned-parameter inventory derived from arguments: shapes, roles, and sharing rules (O6.1). |
| **State ports** | Ports conditional on arguments, with their derivation function (§4.3). |
| **Effects and aliasing** | What the operation writes and what it may overlap. |
| **Logical cost** | Operations and bytes logically read and written, never operations actually executed. |
| **Semantic partitions** | Axes whose partition preserves meaning and the resulting logical communication (O7.1). |

Anything a contract does not declare it cannot interpret, and is rejected (§8.1): the closed
vocabulary is the rejection condition.

Within a declared shape, the **axis** supplies a dimension's catalog identity, its **extent** is the
expression that supplies the dimension's size, and its **nature** describes the dimension's use at
that site. A port or logical tensor's **role** identifies its semantic use for precision policy; the
role is distinct from the dtype selected under that policy.

Every primitive has a contract; only primitives with state ports have a state contract. Token
embeddings, feed-forward networks, mixtures of experts, patch projections, and output heads are
full primitives. Only a **sequence operator**—attention or a substitute that communicates tokens
along the sequence—carries state.

A logical-tensor inventory is mandatory: without it, an implementation cannot construct a module,
verify a checkpoint, compute parameter size, or choose a partition unit. Deriving only cache
semantics is insufficient.

A contract never contains a backend, kernel, memory layout, fusion, workspace, executed-operation
count, implementation-supported partition, or actual collective. Those belong to an
**implementation candidate**, which is outside this specification. Multiple correct candidates may
have different costs without changing the logical graph.

### 4.2 — Computation may be delegated; identity may not

An implementation may provide computation, but the contract still fixes semantic identity, ports,
types, shapes, arguments, effects, and rejection conditions. Otherwise wiring cannot be checked,
candidates compared, or costs derived safely.

The computation provider may itself be a parameterized model document: the template contract
defined by §4.6. Delegation relaxes none of these obligations.

### 4.3 — State derivation

For each state port, the contract declares:

- **Conditional presence, in both directions.** Arguments may add or remove a port. Non-causal
  attention has no cache; the same encoder gains one when processing a fragmented stream.
- **Payload.** Each storage component has a shape, type, and multiplicity. One state may contain
  several components of different types (O5.1).
- **Derived properties.** These are consequences of arguments and never appear in the model:

| Property | Reference | Meaning |
|---|---|---|
| **Evolution law** | O5.2 | How extent evolves: append, bounded window, or fixed size. |
| **Indexing domain** | §5.3 | The sequence relative to which state grows; the law alone is ambiguous. |
| **Access geometry** | O5.3 | Logical position, block, sparse key, or aggregate, expressed as consumed properties rather than runtime data-structure names. |
| **Sharing** | O5.3 | The granularity at which sessions may share. |
| **Key axes** | O5.5 | The instance axes (session, branch) along which allocations of this port are distinct; with the identity's indices they form the instance key (§4.4). |
| **Permitted operations** | O5.4 | Operations derived from evolution law × geometry, with preconditions, including conditions that make an operation unavailable (O5.9). |
| **Modulators** | O5.8 | Scope, sampling step, rank, and depth, represented as §2.1 quantities. |

The condition language is closed and decidable (O0.6). It supports presence tests and argument
equality or inequality, combined by conjunction and disjunction. Rules are ordered; the first
matching rule wins. Conditions apply to argument **combinations**: two coexisting mechanisms produce
two ports, while either mechanism alone produces one.

A flattened shape declares its decomposition (O5.10). Flattening can hide a partition axis. Without
a decomposition, placement derivation must use the non-partitionable case and report **information
loss**, never present non-partitionability as a known fact.

### 4.4 — Information supplied by the graph

> `instantiated states = contract(primitive, arguments) × graph occurrences × state identities`

A contract supplies the descriptor; the graph supplies instance count and relationships. The model
declares only what no derivation can produce: which state ports name the same storage (a state
identity with several members), which repetition indices distinguish allocations (the identity's
indices), and what survives an invocation boundary (§5.3). Everything else about instances is a
consequence, computed by the same rule for every document; a field the author could only fill by
copying that rule is not a declaration (§1.2, §10.2).

**State liveness** is the number of distinct state-allocation equivalence classes that may be active
at once. It sizes simultaneous state memory. This is distinct from **value liveness**, which records
the graph values live across a cut and determines that cut's logical payload.

| Element | Reference | Authority |
|---|---|---|
| **Instance key** | O5.5 | **Derived:** the identity's indices × the contract's `key_axes` (session, branch). Sharing is declared by listing several members under one identity, never by a relation over keys. |
| **Liveness law** | O5.5 | **Derived:** one class per distinct instance key. How many classes are active at once is deployment intent (§10.3): the model fixes the class structure, never the count, and never the raw product of dimensions. |
| **Visit rate** | O3.2 | **Derived** per indexing domain and phase, from the value graph, the generative outputs (§2.3) and the invocation boundaries (§5.3). Visits size computation; liveness sizes memory. Counts — tokens per request, fragments per stream — are deployment intent. |
| **Cardinality** | O5.7 | The model: any finite number of states with distinct natures, with no language-defined upper bound. |

A model document therefore never references a deployment or request fact: there is no open
`context` namespace. Every reference in a document resolves to one of its own quantities, indices
or arguments (V1); a derived product such as D4 takes deployment intent as a separate input.

### 4.5 — Structured sparsity *(O6.6)*

When a primitive activates only some tensors per token, its contract declares the **activatable
unit**, **routing policy**, **count activated per token**, and an **upper bound on the union of
units activated per batch**.

Per-token count is insufficient because a batch's union may include every unit. This yields three
separate qualified quantities: exact resident cost, upper-bounded worst-case transfer, and estimated
expected transfer.

### 4.6 — Template contracts

A contract may be implemented by a **template**—a parameterized model document—instead of a consumer
capability. The catalog designates the template; it does not describe it:

```json
"decoder.causal_yarn": { "version": "1.0.0",
  "template": { "name": "decoder-causal-yarn", "version": "1.0.0",
                "id": "decoder_causal_yarn" } }
```

The template is not the contract identity. Two templates implementing the same semantic identity are
interchangeable; a template name identifies a realization, not a meaning. The realization is
nevertheless pinned: the contract names the template version, and the template carries that version.
Otherwise an in-place template change could silently change the contract's denotation.

The two versions distinguish different facts:

> **Template versions distinguish representations; contract versions distinguish meanings.**

Because unique representation is not required (§1.1), two templates may denote the same graph
family. A template version may change without changing contract identity while that family remains
invariant. Changing the family creates a new contract identity: for a template contract, the
template defines the meaning.

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

Template occurrences become occurrences in the calling graph, with identifiers prefixed by the
invoking occurrence. Two invocations share neither state nor tensors. Sharing across the boundary is
not expressible; any future form must use ports, never common tensor identities.

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
2. **Hygienic identifiers:** expansion creates unique identifiers through a fixed rule that cannot
   collide with handwritten identifiers.
3. **Resolved references:** every quantity, occurrence, port, family, and tensor reference resolves;
   unresolved references are rejected (I1).
4. **Order:** a composition emits occurrences in its defined order, identically on every reading:
   compositions in document order, sites in document order within each, index grids in declaration
   order of the ranges with values ascending.
5. **Multiple trunks:** several trunks may coexist (O3.3), and a trunk may carry no state.
6. **Guards:** a site or a binding may carry a `when` condition over the document's quantities and
   the indices in scope; it is evaluated at every index, and a site or rule whose guard is false is
   not emitted. A guard that cannot be decided is a rejection (V10), never false: an undecidable
   guard would otherwise drop occurrences silently (I7). Guards are how a periodic layer pattern is
   written as one composition — a site present when `layer mod 4 = 3`, an edge present when
   `layer ≥ 1` — instead of one site per case.
7. **Scoped bindings:** a composition may carry bindings written against its own sites. Each such
   rule `R` of composition `C` denotes exactly the top-level rule `C.R` whose `for_each` is `C`'s
   index ranges and whose site endpoints select the generated occurrence at the current indices,
   overridden index by index where the endpoint says so; a scoped parameter or state rule without a
   declared identity names it `C.R`, indexed by `C`'s indices. Nothing is expressible in the scoped
   form that is not expressible at the top level; the scoped form only removes the repetition of the
   composition's ranges and selectors, and every reading expands it before any other rule applies.
   A scoped rule whose expanded name collides with a top-level rule is rejected (V1).

The concrete representations use three related but non-interchangeable condition fields:

- model-level `when` is evaluated from model quantities and in-scope composition indices and
  controls whether a graph site or binding is emitted;
- contract-level `present_when` is evaluated from resolved primitive arguments and controls whether
  a port, parameter, constant, or state slot exists;
- `when` on an ordered contract rule is also evaluated from resolved arguments and controls whether
  that rule applies to a present element.

All conditions that affect denotation must resolve deterministically. A field from one context
cannot substitute for a field from another.

### 5.3 — Indexing domains and invocation boundaries

Every value and state port declares an **indexing domain**: what indexes it, such as token position
in the current sequence, position in a source sequence, patch, or stream fragment.

The same concept governs both interface granularity and the axis along which state grows.

Value edges form an acyclic graph within one invocation. Time recurrence must pass through a state
port, never an implicit combinational cycle.

State carried to the next invocation declares its **invocation boundary**: what is preserved and
under which indexing domain. This supports streaming input while the graph remains statically finite
per invocation.

## §6 — Static semantics

A document is valid only if every rule below holds. Failure produces a reasoned rejection, never an
implicit default.

| Rule | Requirement |
|---|---|
| **V1** | Every quantity, occurrence, port, family, tensor, catalog base, and template reference resolves (I1). |
| **V2** | Every required contract argument is present; every undeclared argument is rejected; declared defaults are applied before checking. |
| **V3** | Types, enums, record fields, units and domains conform, for arguments and for quantities, derived ones included; an enum value outside its declared set is rejected. |
| **V4** | Shapes compose: each source-port output shape equals the destination-port input shape. |
| **V5** | Indexing domains are compatible across every edge. |
| **V6** | The value graph is acyclic within an invocation. |
| **V7** | Bindings are total: every input port is fed and every logical tensor is bound. |
| **V8** | Conditional argument combinations satisfy the contract. Meaningless combinations are excluded by invariant, never merely by a missing guard (I11). |
| **V9** | A state identity connects only compatible ports: the same applicable rule, the same key axes, and equal payload shapes and indexing domains. |
| **V10** | Every repetition, guard and derivation resolves under §5.2 and §2.2. |
| **V11** | A literal quantity that declares its derivation agrees with it (I8). |
| **V12** | Every construction has one reading; references and literals are unambiguous (I5, I6). |

## §7 — Required derived products

A valid document and its referenced contracts make all products below computable without inference
code or human knowledge of a named mechanism:

| Product | Content |
|---|---|
| **D1** | **Expanded graph:** occurrences, edges, and families. |
| **D2** | **Values:** the value and shape inventory, including value liveness at every cut. |
| **D3** | **Parameter tensors:** shapes, roles, sharing, and total count. |
| **D4** | **Complete state:** descriptors, instances, keys, state liveness, and permitted operations. |
| **D5** | **Logical costs:** parameters, activations, state per token, computation, and cut traffic. |
| **D6** | **Legal cuts and semantic partition axes.** |

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

Capability rejection and contract identity are orthogonal:

| Change | Mechanism |
|---|---|
| A primitive, argument, or value is **added**. | Explicit rejection; no new model-language version. |
| The meaning of an existing argument **changes**. | A **new contract identity**; an old consumer would otherwise silently apply the old meaning. |
| A template is rewritten but its denoted graph family is unchanged. | Update the pinned **template version**; retain contract identity (§4.6). |
| A template's denoted family changes. | A **new contract identity**; for a template contract, the template defines the meaning. |

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
| **I9** | The described model and loaded artifact are mutually compatible: every logical tensor is bound, and no physical tensor is accidentally bound twice. |
| **I11** | A meaningless combination is excluded by invariant, never merely by a missing guard. |

I5 and I6 are expressiveness invariants, not concrete syntax rules. Known conformance cases include
a dimension named like an enum value and a two-key object indistinguishable from a two-entry map.

I10—detecting a decision based on stale measurements—belongs to deployment control and is outside
this specification.

### 9.2 — Non-requirements

| Rule | Exclusion | Reason |
|---|---|---|
| **N1** | No tensor algebra: no value expressions, loops, or indexing. | This is the level of an IR; §2.2 shape arithmetic remains allowed. |
| **N2** | No backend or kernel scheduling. | These belong to implementation candidates. |
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

| Case | Property tested | Construction |
|---|---|---|
| Homogeneous dense decoder | Trivial case | §3, §5.2 |
| Regular heterogeneous pattern | Families and instance counts | §3.1, §4.4 |
| **Three simultaneous state natures** | Unbounded cardinality and different operations | O5.7, §4.3 |
| **Four states, irregular pattern, residual multiplicity** | Multiplicity and conditional combinations | §3.5, §4.3 |
| **Non-generative, multimodal, tied weights** | Non-generative output, tensor sharing, and multiple trunks | §2.3, §3.4, §5.2 |
| **Mixture of experts with batch > 1** | Batch union differs from per-token count | §4.5 |
| **Flattened shape** | Lost partition axis is reported | O5.10 |
| **Encoder-decoder** | State read by another trunk | §3.4, §5.3 |
| **Non-adjacent layer cache sharing** | State identity | §3.4 |
| **Fragmented streaming input** | Invocation boundary | §5.3 |
| **Per-token output** | Output indexing domain | §2.3, §5.3 |

### 10.2 — Required rejection cases

The following are rejected at the stated level: unresolved reference (V1); missing required argument
(V2); undeclared argument (V2); out-of-enum value (V3); incompatible shapes (V4); incompatible
indexing domains (V5); value cycle (V6); unfed port or unbound tensor (V7); meaningless combination
(V8); state identity between incompatible ports (V9); unresolvable repetition, guard or derivation
(V10); and a literal quantity disagreeing with its declared derivation (V11).

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
| Physical artifact encoding and mapping to fragments | Artifact descriptor |

## Appendix A — Requirements

The identifiers cited throughout this document, stated once. A citation names the construction that
carries the requirement; this appendix is the requirement. Gaps in the numbering are reserved.

| Id | Requirement |
|---|---|
| **O0.1** | Derived quantities are expressed in the closed scalar algebra of §2.2. |
| **O0.2** | A derivation outside the algebra uses a normative interface with declared inputs, semantics and conformance tests; its result is at least a qualified bound. None is defined in this revision. |
| **O0.3** | A derived quantity of a model is exact, reads only declared quantities, resolves acyclically and conforms to its declared type and domain; epistemic status and provenance are declared on contract costs and derived products. |
| **O0.4** | Quantities form one flat namespace: each is declared once and referenced by name everywhere. |
| **O0.5** | Where a product combines qualified values, status propagates by the monotonicity of each operation; propagation never turns an estimate into a bound. |
| **O0.6** | Every language of this specification is closed and decidable, or uses a normative interface. |
| **O1.1** | A contract is referenced through an immutable identity, `{name, version}`, whose meaning never changes (§8.2). |
| **O1.2** | A model document carries a stable, authoritative identifier. |
| **O2.1** | Every quantity and every occurrence has one stable identifier, unique after expansion. |
| **O2.2** | Every quantity declares its regime: a model constant, or a variable with a declared domain. |
| **O2.3** | Every quantity declares its dimensional type; physical quantities carry a unit. |
| **O2.4** | When several authorities constrain a domain, each constraint is declared separately and the binding one remains identifiable. |
| **O3.2** | The rate at which a state is visited, per indexing domain and phase, is derived from the graph; counts are deployment intent (§4.4). |
| **O3.3** | Several trunks may coexist in one document; a trunk may carry no state. |
| **O3.5** | Every occurrence belongs to one or more named families, addressable without enumeration. |
| **O4.1** | Every value port has a shape, a type and a multiplicity. |
| **O4.2** | Every flow is seeded explicitly by a public input; implicit seeds are forbidden. |
| **O4.3** | Residual multiplicity other than one is explicit, and the combining operation is a contracted primitive. |
| **O4.4** | Value flow is declared by explicit port-to-port edges, never inferred from ordering or from mutation of a named flow. |
| **O4.5** | The logical payload of a cut is the set of values live across it, determined by the explicit edges; it is not measured traffic. |
| **O5.1** | A state payload may hold several components of different shapes and types. |
| **O5.2** | Each state port derives an evolution law: append, bounded window, or fixed size. |
| **O5.3** | Each state port derives its access geometry and the granularity at which sessions may share it, as consumed properties, never as runtime data-structure names. |
| **O5.4** | Each state port derives its permitted operations from evolution law and geometry, with their preconditions. |
| **O5.5** | The instance key of an allocation is derived from the identity's indices and the contract's key axes; liveness is one class per distinct key, never the raw product of dimensions. |
| **O5.7** | A model may declare any finite number of states of distinct natures; the language sets no upper bound. |
| **O5.8** | Modulators of a state — span, stride, rank, depth — are expressions over the primitive's arguments. |
| **O5.9** | A contract states the conditions under which a permitted operation becomes unavailable. |
| **O5.10** | A flattened shape declares its decomposition into named axes; without one, placement derivation reports information loss rather than asserting non-partitionability. |
| **O6.1** | Every contract derives its logical tensor inventory from its arguments: shapes, roles and sharing rules. |
| **O6.2** | Parameter bindings identify the logical tensors each occurrence consumes and every sharing between occurrences; a tied tensor is counted and loaded once. |
| **O6.6** | A primitive that activates only some of its tensors per token declares the activatable unit, the routing policy, the count activated per token, and an upper bound on the union activated per batch. |
| **O7.1** | A contract declares the axes along which partition preserves meaning and the logical communication each partition implies. |
| **O8.1** | Every public input and output names its occurrence and port and declares its indexing domain. |
| **O8.2** | A document may expose several outputs at once, including non-generative ones. |
| **O8.3** | An auxiliary output is an additional reference to an existing value, never a new operation. |
| **O9.1** | A model names primitives and supplies arguments; it never describes a primitive's computation. |
| **O9.2** | A contract declares the elements of §4.1. |
| **O9.4** | An argument is a scalar, an enum, or a record of such values; inline numeric tensors are forbidden. |

The I‑ and N‑ series are stated in §9.

