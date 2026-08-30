# Capabilities of a generator

*A manifest per generator — the implementation that builds and runs a graph over its own primitives — for two readers and nothing else: a **runtime**, which infers from it and
from a document's derived products whether it can run that document for a given delivery of
inputs, before anything is loaded; and a **maintainer**, who reads from it which of the catalog's
vocabulary and which of the corpus is not covered yet. The language does not own the manifest
(§4.1: generators are outside it); it owns the words the manifest is made of, and
the tool that reads it. Every generator writes its manifest from its code — the tables its kernels
are written against, never a hand-maintained list — and commits the result beside a test that
regenerates it.*

The grammar is `generators/capabilities.schema.json`. The reader is `tensorspine --capabilities
MANIFEST MODEL…` (can this candidate run these documents, for the inputs `--inputs` names — by
default those D2 marks required for the generative output) and `--coverage` (what the catalog and
the corpus still need). The reference generator's manifest is `generators/reference/capabilities.json`,
written by `ref.py capabilities`.

## What a manifest states

| Section | Content | Vocabulary it draws on |
|---|---|---|
| `generator` | name, version (the code's, e.g. a commit), the program that wrote the manifest, date | — |
| `compute_dtypes`, `parameter_dtypes` | dtypes computed in; storage dtypes the loader reads | the catalog's dtype names |
| `state_laws`, `access`, `sharing` | the state laws, access geometries and cross-session sharing granularities implemented | `state_rule.law`, `.access`, `.sharing` (catalog-unit schema) |
| `partitions` | the semantic partition communications the generator can realise (`all_reduce`, …); empty for a single-machine generator | `partition.communication` |
| `domains` | the indexing-domain kinds and transforms handled; whether fragmented inputs are | `port_domain.kind`, `domain_transform.relation`, `fragmented` |
| `sessions_per_invocation` | 1 for a sequential runtime; more for a batching one (D4's `instance.session` axis) | §7, D4 instance keys |
| `locations` | the location forms the loader assembles | §3.4 |
| `contracts` | per contract version: the argument values implemented, combinations refused, the state ports' laws, the transforms realised, notes | the contract's own arguments and enums |

## The argument rules

Per argument, or per record field:

| Rule | Meaning |
|---|---|
| `"any"` | every value, present or absent |
| `"absent"` | only when the argument is absent |
| `[v, …]` | absent, or one of these literals |
| `{"absent": b, "values": [v, …]}` | an enum or boolean: may or may not be absent; when present, one of these |
| `{"absent": b, "fields": {…}}` | a record, checked field by field; an unknown field is refused |

`excluding` lists argument combinations refused although each value alone is implemented — a
manifest by independent values over-approximates, and this is where it says so. An argument the
table does not name is refused: the manifest is closed, like the contract.

## The physical parameters, and the backend

Generating an extract mostly depends on the **backend** — the hardware the generator targets: cpu,
nvidia, neuron, tpu — and on choices that are nobody's but the serving application's and the
kernel's: block sizes, kernel selection, layouts. Those *physical parameters* are **opaque** to the
language and to the manifest: a contract between the serving application and the generator's
primitives, neither typed nor validated here. What the language fixes is the channel: a generator
passes them to a primitive *beside* the contract arguments, never merged with them, addressed by
occurrence, by site pattern or by contract; the derivation (D1–D6) never sees them; the targeted
`backend` is one optional key among them — a generator that targets one backend needs none.

## What is not in it

Evidence and performance. Which fixtures a generator reproduces, and how closely, belongs with
its tests; physical costs — the remaining half of Architecture §6's open question — and anything a
document cannot ask (throughput, latency, memory beyond the declared bytes) are not capabilities.
A manifest says what can be evaluated; it does not promise speed.

## Reading it

**Can this generator run this document?** For every occurrence the delivery evaluates (§7): its
contract version is in `contracts`, its D1 arguments pass the rules, its D4 states' laws and access
geometries are in the manifest; every D3 dtype is in `parameter_dtypes`; every location form is in
`locations`; a fragmented input needs `domains.fragmented`; the streams' kinds and the transforms
the evaluated occurrences use are in `domains`. The first failure is the answer.

**What is left to cover?** Every catalog contract without an entry; for every entry, the enum
values, booleans and records of the contract the rules do not admit; every corpus document that
cannot run, with its first reason. The report is computed, never written.
