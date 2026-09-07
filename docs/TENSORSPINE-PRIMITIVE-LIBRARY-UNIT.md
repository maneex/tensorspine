<a id="tensorspine-catalog-unit"></a><a id="tensorspine-primitive_library-unit"></a>

# TensorSpine primitive library unit

> One logical vocabulary unit — a primitive definition, a named axis, or a precision role — written down so a consumer can implement it in advance.

*Companion to `schemas/tensorspine-primitive-library-unit.schema.json` (`$id`
`https://tensorspine.dev/schema/2.0/primitive-library-unit.json`). Non-normative: the
[specification](SPECIFICATION.md) is the authority for what a primitive means and which instances
are valid (§4, §6); this document says why the schema has the objects it has, and describes each
one. The generated [primitive library reference](https://maneex.github.io/tensorspine/primitive-library/) renders the
units themselves.*

---

<a id="1--why-a-schema-and-why-one-per-file"></a>

## 1 — Why a schema, and why one per file

The primitive library is the closed vocabulary the language shares between a model author and every
implementation (§4.3, O0.6): the set of primitives a document may name, the axes its shapes are
built from, and the precision roles its tensors carry. A model definition references this vocabulary;
it never inlines it. So the vocabulary needs a grammar of its own, separate from the model
grammar — this schema — against which every unit is checked when a primitive library base is loaded. A unit
outside it is a load error naming the file, never a document read with a guess (I7).

One unit is one file, and its path reproduces its identity: `primitives/attention/dense/1.0.0.json`
is the primitive `attention.dense` at version `1.0.0`, `axes/model/width.json` the axis
`model.width`, `precision/norm/scale.json` the role `norm.scale`. A primitive carries its version as
the file name, so an identity `{name, version}` is one file and never changes meaning (§8.2). A
disagreement between the path and the identity written inside is a refusal, not a preference.

Every enumeration in the schema — argument kinds, state evolution rules, access geometries, sharing
granularities, partition communications, precision sensitivities — is closed. There is no opaque
argument type: what a consumer cannot enumerate in advance it cannot implement in advance.

<a id="2--the-four-kinds-of-unit"></a>

## 2 — The four kinds of unit

A unit's top level is `{schema, kind, name, definition}`, where `schema` is the constant
`tensorspine-primitive-library-unit/2.0` and `kind` is one of:

| `kind` | `definition` is | Identity |
|---|---|---|
| `primitive` | a **primitive definition** (a template primitive when it carries `template`) | `{name, version}` |
| `axis` | an **axis definition**: the space a shape extent lives in (`value`, `instance` or `storage`) | `name` |
| `precision_role` | a **precision role**: the admissible dtypes of a tensor and its numerical sensitivity | `name` |
| `base` | the **base itself**: a `primitive-library.json` manifest naming the base and where its templates live | — |

Documentation fields (`summary`, `description`, `tags`, `external_docs`, `value_descriptions`) are
defined once in `documentation.json` and referenced from every site that may carry them; they carry
meaning for a reader, never for a consumer, and a unit without them is rendered from its definition
alone.

<a id="the-primitive-contract"></a><a id="the-primitive-primitive"></a>

<a id="3--the-primitive-contract"></a><a id="3--the-primitive-definition"></a><a id="3--the-primitive-primitive"></a>

## 3 — The primitive definition

A `primitive_primitive` is the whole of what a model references when it names a primitive: the
knobs it exposes, the tensors it reads and writes, the learned parameters and states it owns, and
the facts a serving system derives from all of these. Its objects:

| Object | What it declares |
|---|---|
| `version` | The primitive's semantic version, matching the file name (§8.2). |
| `arguments` | The knobs the model supplies — see [§4](#4--argument-and-field-declarations). |
| `ports` | The value inputs and outputs, each with a shape function over the arguments and an indexing domain (§5.3). |
| `parameters` | The learned-tensor inventory: role, shape, optional `present_when`, optional `multiplicity` (§3.4), sharing. |
| `constants` | Tensors fixed by the primitive rather than learned. |
| `state_ports` | The states the primitive carries across invocations — evolution, access geometry, sharing, payload, indexing rule (§4.3). |
| `effects` | The ports the operation reads and writes, and its `across_positions` condition (§4.1, O9.5). |
| `invariants` | The relations the resolved arguments must satisfy — see [§5](#5--invariants). |
| `partition_options` | The axes along which the operation may be split, each with its communication (§7, D6). |
| `domain_transforms` | How an output's indexing domain relates to an input's — a merge, an insert (§5.3). |
| `logical_cost` | The arithmetic per element, as an expression over the arguments (D5). |
| `sparsity` | The activation-sparsity unit and policy, where the primitive has one. |

A **template primitive** carries `template` instead of the computational objects: it pins a model
document as a reusable sub-graph, versioned in its own file, expanded at every call site (§4.6). Its
arguments are the template's external quantities.

<a id="4--argument-and-field-declarations"></a>

## 4 — Argument and field declarations

An `argument_declaration` is one knob; a `field_declaration` is one field of a record argument, and
has the same shape. Both are:

| Field | Meaning |
|---|---|
| `type` | The kind and, where the kind needs it, its detail: `cardinality`, `real`, `physical` (with a `unit`), `boolean`, `enum` (with `values`), or `record` (with `fields`). |
| `required` | Whether the model must supply it. |
| `structural` | Whether it decides which slots, ports or states exist, or their shapes — as opposed to changing only the computation. |
| `default` | The value applied when the model omits it (§4.6), an expression over the other arguments. |
| `domain` | A numeric precondition — see below. |
| `present_when` | The condition under which the argument is applicable; supplying it otherwise is a refusal, not a value that is ignored (I2). |

### The `domain`

A `domain` is a numeric precondition checked under **V3**, after the type, at every call site — the
admissibility of one argument's value (§4.6). It is admitted on a `cardinality`, `real` or
`physical` argument only; a relation between *two* arguments is an invariant, not a domain
([§5](#5--invariants)). Two shapes:

- an **interval** — a `lower` and/or an `upper` bound, each a `value` and whether it is
  `inclusive`. A bound's value is a scalar literal or a reference to another argument
  (`{"argument": "heads"}`): `kv_heads` in `[1, heads]` names `heads`. The referenced argument must
  be required or defaulted, so the bound always resolves at a call site; the primitive library loader refuses
  a bound that names an argument which may be absent.
- a **set** — a finite list of admissible scalar `values`.

A physical argument in `tokens`, `elements`, `bytes` or `operations` is additionally a whole number
(only `seconds` is real); a fractional count is refused under V3 before the domain is even read.

<a id="5--invariants"></a>

## 5 — Invariants

`invariants` is the list of relations the resolved arguments must satisfy for an instance to be
admissible — **V8** (§6). Each entry is `{holds, description}`: `holds` a condition in the unit
schema's own grammar (`compare`, `modulo` and the other operators, `all`/`any`/`not`, `present`),
`description` the words the validator prints when it does not hold —
`[V8] attn @decoder/attn: 'heads is a multiple of kv_heads' does not hold (heads = 32, kv_heads = 3)`.

A domain constrains one argument; an invariant relates several: `heads mod kv_heads = 0`,
`top_k ≤ experts`, `value_heads mod key_heads = 0`. Invariants are evaluated after the types and
domains, on the arguments with defaults applied, in the instance's index environment. A condition
that reads an argument which may be absent — a yarn-only bound under a scaling that is not yarn —
guards it: `any[not present X, …]` holds vacuously when `X` is absent, the dual of the `all[present
X, …]` a presence-gated field uses. An argument left unresolved by a V3 problem is skipped, so a
domain failure and an invariant do not both fire on the same value; every other undecidable
condition is a refusal, never read as false (§4.3). The loader refuses an invariant that reads an
undeclared argument, or that compares a maybe-absent argument outside a `present` test of it, so a
condition that cannot be decided at a call site is caught before any document is read.

<a id="6--axes-precision-roles-and-the-base"></a>

## 6 — Axes, precision roles, and the base

An `axis_definition` names a shape space: a `space` (`value` — a feature of the data;
`instance` — a session or branch key; `storage` — an address that is not an axis of the
computation, such as the leading `storage.multiplicity` of a slot with copies, §3.4), and a
`summary`. A shape extent cites an axis; a consumer that partition options a tensor reasons in these
spaces.

A `precision_role_definition` names the numerical class of a tensor: its `admissible` dtypes, its
`default`, and its `sensitivity` (how much its precision matters). Every port, parameter, constant
and state component cites a role; V14 checks a document's dtype choices against the role's
admissible set.

A `base_definition` is a base's own `primitive-library.json`: the base `name`, and `templates` — where the
template documents its template primitives pin are found, relative to the base (§4.6). A base is a
set of units; two bases carrying one identity with different contents are a conflict, never a
choice (V1).

<a id="7--reading-and-checking-one"></a>

## 7 — Reading and checking one

```sh
python3 tools/tensorspine --validate        # loads the primitive library a document declares; a broken unit is refused, naming the file
python3 tools/tensorspine --document primitive-library -o /tmp/PRIMITIVE-LIBRARY-REFERENCE.md   # every unit, definitions and documentation, as Markdown
python3 tools/tensorspine --document primitive-schema -o /tmp/schemas/   # one JSON Schema per primitive version, from its arguments (non-normative)
python3 tests/run_rejections.py             # the primitive library cases: a unit off the schema, or citing what the primitive library does not hold, is refused
```

When a primitive library base is loaded, every unit is checked against this schema, and the cross-references a
unit makes — an axis, a precision role, a port, an argument path named by a condition, a domain
bound, an invariant — are resolved against the gathered base. What the grammar cannot see (a domain
bound on a maybe-absent argument, an invariant on an undeclared one) the loader checks besides. The
validator, reading a model definition, is the authority for whether an instance's arguments satisfy
the primitive; this schema is the authority for whether the primitive itself is well-formed.
