# Capabilities of an implementation

*A capabilities manifest declares the subset of TensorSpine contracts and branches one
implementation supports. A **runtime** uses it with a model's derived products to decide whether
that implementation can run the requested inputs before loading anything. A **maintainer** uses it
to see which catalog vocabulary and corpus documents remain unsupported. TensorSpine defines the
format and vocabulary, and provides the tooling that builds the manifest from the implementation's
primitive support and reads it. The manifest is generated from code, never maintained as a model
allow-list.*

The grammar is `generators/capabilities.schema.json`. The
[harness guide](../docs/HARNESS.md) maps this admission result and D1–D6 to serving decisions. The
reader is `tensorspine --capabilities MANIFEST MODEL…` (can this candidate run these documents, for the inputs `--inputs` names — by
default those D2 marks required for the generative output) and `--coverage` (what the catalog and
the corpus still need). The reference generator's manifest is `generators/reference/capabilities.json`,
written by `ref.py capabilities`.

## What a manifest states

| Section | Content | Vocabulary it draws on |
|---|---|---|
| `generator` | implementation name and version, the program that generated the manifest, date | — |
| `compute_dtypes`, `parameter_dtypes` | dtypes computed in; storage dtypes the loader reads | the catalog's dtype names |
| `state_laws`, `access`, `sharing` | the state laws, access geometries and cross-session sharing granularities implemented | `state_rule.law`, `.access`, `.sharing` (catalog-unit schema) |
| `partitions` | the semantic partition communications the implementation can realise (`all_reduce`, …); empty for a single-machine implementation | `partition.communication` |
| `domains` | the indexing-domain kinds and transforms handled; whether fragmented inputs are | `port_domain.kind`, `domain_transform.relation`, `fragmented` |
| `sessions_per_invocation` | 1 for a sequential runtime; more for a batching one (D4's `instance.session` axis) | §7, D4 instance keys |
| `locations` | the location forms the loader assembles | §3.4 |
| `contracts` | per contract version: the argument values implemented, combinations refused, the state ports' laws, the transforms realised, notes | the contract's own arguments and enums |
| `role` | `witness` for the reference generator, which executes the reference implementations supplied with contracts; absent, or `conformer`, for an implementation checked against their fixtures (Specification §4.1, O1.3) | — |
| `contracts.*.witness` | In a witness manifest, per contract version: the supplied reference implementation's `kernel` entry point (relative to the manifest), the `tolerance` a conformer must meet per compute dtype, and its unit `fixtures`, each at `fixtures/contracts/<id>.safetensors` beside the manifest. The reader refuses missing files and witness blocks in conformer manifests. | the fixture schema (`docs/TENSORSPINE-FIXTURE.md`) |

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

Executing an extract mostly depends on the **backend** — the hardware the implementation targets:
cpu, nvidia, neuron, tpu — and on choices that belong to the serving application and its kernels:
block sizes, kernel selection and layouts. Those *physical parameters* are **opaque** to the
language and to the manifest. A serving application passes them to a primitive implementation
*beside* the contract arguments, never merged with them, addressed by occurrence, site pattern or
contract; derivation never sees them. The targeted `backend` is one optional key among them; an
implementation that targets one backend needs none.

## What is not in it

Conformance results and performance. A witness manifest binds each contract to its reference
implementation, tolerances and fixture ids; whether another implementation reproduces those
fixtures, and how closely, belongs with its tests. Physical costs — the remaining half of
Architecture §6's open question — and anything a document cannot ask (throughput, latency, memory
beyond the declared bytes) are not capabilities. A manifest says what can be evaluated; it does not
promise correctness or speed.

## Reading it

**Can this implementation run this document?** For every occurrence the delivery evaluates (§7): its
contract version is in `contracts`, its D1 arguments pass the rules, its D4 states' laws and access
geometries are in the manifest; every D3 dtype is in `parameter_dtypes`; every location form is in
`locations`; a fragmented input needs `domains.fragmented`; the streams' kinds and the transforms
the evaluated occurrences use are in `domains`. The first failure is the answer.

## Branch ledger

`--coverage` produces the **branch ledger**: the to-do list for each model-and-implementation pair.
It lists every absent contract entry; within present entries, every enum value, boolean, record
field and optional argument not admitted; then every corpus document that
cannot run and its reasons. The ledger is computed from the catalog, model and manifest, never
tracked. For a contract without an entry every branch of its arguments is listed, so the to-do list
per model and implementation is complete. On a witness manifest the report ends with the contract
versions still without a witness — a catalog is released only when there is none (§10.2) — and
`--strict` exits 1 while there is one, for the tag workflow. The documentation build renders the
reference manifest's ledger at the generated
[branch-ledger page](https://maneex.github.io/tensorspine/branch-ledger/).
