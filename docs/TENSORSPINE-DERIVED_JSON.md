# Tensorspine derived document — the products, as JSON

> A valid model document and its contracts make six products computable without inference code
> (Specification §7). This document is how the repository writes them down: one JSON per model,
> one schema, D1 required and D2–D6 optional.

*Companion to `schemas/tensorspine-derived.schema.json` (`$id`
`https://tensorspine.dev/schema/2.0/derived.json`), emitted by `tensorspine --d1` and
`tensorspine --derive`. Non-normative: the [specification](SPECIFICATION.md) defines what the
products *are* (§7) and leaves their encoding open; this is the repository's encoding.*

---

## 1 — Why a schema, and why one document

The specification says what D1–D6 contain and refuses to say how they are written: "their
encoding and the decisions that consume them are outside this specification" (§7). An emitter
therefore chooses an encoding, and a consumer has to be able to check it. The schema is that
check — the same role `tensorspine.schema.json` plays for a model document — and the emitters
run it on their own output before writing anything: a document the tool cannot vouch for is not
written.

The six products are one document, not six, for one reason: every product refers to D1's
identifiers. A D2 value is `node.port`, a D3 member is `node.slot`, a D4 member is
`node.state`, a D6 cut is a set of nodes. Split across files, a consumer would have to join by
identifier and *trust* that both files came from the same expansion under the same assignment;
in one document that is a fact. `d1` is therefore required and everything else optional:
`--d1` writes the graph alone, `--derive` writes all six, and both validate against the same
schema.

What the document does **not** contain follows from §4.1 and §10.3: no kernel, no executed
FLOPs, no physical traffic, no placement. A cut's payload is the logical payload of O4.5, a cost
is a logical cost, a partition is a semantic partition; the machine, the workload and the
implementation are inputs a consumer adds.

## 2 — Head

| Field | Content |
|---|---|
| `schema` | `tensorspine-derived/2.0` |
| `model` | The model document's identifier. |
| `catalog` | The bases the document declares (§2). |
| `assignment` | The values supplied for the external quantities — empty for a concrete model, the call-site or `--assign` values for a template (§4.6). One document is one assignment. |
| `d1` … `d6` | The products, described below. `d1` is required. |

### Identifiers

Every product refers to the expanded graph by the identifiers of §5.2 rule 2: a root
occurrence by its name (`embed`), a generated one as `<composition>/<site>[<index>=<value>,…]`
(`decoder/attn[layer=3]`), an occurrence inside a template instance prefixed by the instance
(`text/decoder/attn[layer=3]`). A port, slot or state of a node is `node.name`. A parameter or
state identity instance is the identity's name with its evaluated indices
(`decoder.attn.q[layer=3]`, `tied_embeddings`, `shared.sliding.kv`), prefixed by its instance
when it lives inside a template. Identifiers are representation, not meaning (§5.2): two
documents denote the same graph when they correspond up to occurrence renaming.

### Counts

Element counts are never numbers: how many tokens a request has is deployment intent (§10.3).
A `count` is a combination of the public inputs' counts — `{"tokens": 1.0}` means one element
per element of the `tokens` input; `{"pixels": 0.0625}` one per sixteen pixels after a
`merge` by 16; `{"tokens": 1.0, "pixels": 0.0625}` a token stream after a `splice` inserted
the merged patches (§5.3). A consumer multiplies by the counts it knows.

### Statuses

A value that can be a bound or an estimate carries a `status` — `exact`, `upper_bound`,
`lower_bound`, `estimate` — and the totals of D5 carry the status the algebra of §2.2 gives a
sum: an estimate absorbs, opposite bounds combine into an estimate, a one-sided bound survives.
Everything derived from the inventory is exact; a status other than `exact` always comes from a
contract's declared correction or sparsity bound.

## 3 — D1, the expanded graph

| Field | Content |
|---|---|
| `nodes` | One entry per emitted occurrence, keyed by identifier: `contract` `{name, version}`, `arguments` after declared defaults were applied (records as objects), `families` (the site's and its composition's). |
| `edges` | Every value edge: `rule` (the binding, `<composition>.<rule>` for a scoped one), `from` and `to` as `{node, port}`. An edge into a template instance fans out to the template's destinations; an edge out of one starts at the template's source. |
| `interfaces.inputs` | Per public input: `to` (the ports it feeds, expanded), `kind`, optional `stream` (the stream it joins) and `fragmented`. |
| `interfaces.outputs` | Per public output: `node`, `port`, `generative`. Its domain is derived (§2.3) and appears in D2. |
| `topological_order` | One order of the nodes compatible with the edges (V6). |
| `instances` | Template instances that were expanded: identifier, contract, the assignment their arguments formed. Absent when there is none. |

The listing is the canonical one (§5.2 rule 4): nodes by identifier, edges by (source,
destination). An occurrence absent by its guard is absent, and so is every binding that named
it (rule 3); no guard is repeated on a binding.

## 4 — D2, values and cuts

| Field | Content |
|---|---|
| `streams` | Every stream a public input introduced: `kind` and its `count` — after transforms, so a merged stream has fewer elements and an inserted-into stream more (§5.3). |
| `values` | One entry per value, keyed by its source `node.port`: `to` (every consumer), `shape` (axis identity and evaluated extent, factors when declared), `role`, `dtype` (the role's default: the language selects dtypes on identities, not on activations), `elements`, `bytes_per_element`, `domain` `{kind, stream}`, `count`. A public input is an edge like any other (§5.3): the value it delivers is an entry named by the input (`input`), whose shape is that of the ports it feeds, with `required_for` — the public outputs not evaluated without it on a first delivery — and `required`, whether that list is empty (§7: an input whose elements are only inserted into another stream may deliver nothing); a value a public output exposes carries `exposed` (the output names) and is listed whether or not an edge consumes it. |
| `cuts` | One entry per structural legal cut (§7): `cut` (its name), `kind` — `layer` for the ancestor closure of a composition prefix (`decoder[layer<=3]`), `family` for the ancestor closure of a family (`family:encoder`) — `sizes` of the two blocks, `payload` (the distinct values crossing, each with `bytes_per_element` and `count`), `bytes_per_element` (their sum) and `bytes_per_invocation` (the sum weighted by the counts, per public input). |

A cut is legal by construction: the ancestor closure of a set of nodes is downward closed, so
every crossing edge points out of it. A value is counted once per cut however many consumers it
has across it — the residual carried to two ports of the next layer is one value. Cuts inside a
template instance are not enumerated at the caller's level; derive the template itself, under
the instance's assignment, for those.

## 5 — D3, parameter tensors

One entry per parameter identity instance, a tied tensor once.

| Field | Content |
|---|---|
| `identity` | The identity instance (`decoder.attn.q[layer=3]`). |
| `members` | The slots it satisfies, `node.slot`; several for a tied tensor. |
| `contract`, `slot`, `role` | Of the first member; V15 makes the others compatible. |
| `sensitivity` | The role's quantisation advice (`quantizable`, `reduced`, `full_precision`), carried through as §7 requires. |
| `dtype` | The dtype the binding selects, else the role's default (V14). |
| `shape` | Axis identity and evaluated extent, with factors when the contract declared them. |
| `multiplicity`, `elements`, `bytes` | `elements` includes the multiplicity; `bytes` is elements × the dtype's width. |
| `tied` | Whether the identity has several members. |
| `location` | When the document locates its weights: the evaluated location — physical names with indices and coordinates substituted, a `stack` expanded into its parts, a `slice` with its `offset` and `extent`, `dim` the position of the named axis in the shape. What a loader reads; what `--checkpoint` checks against the file headers (V17). |
| `sparsity` | When the slot belongs to a sparsity unit (§4.5): the unit's index and axis, `activated_per_element`, `units` (the axis extent) and `activated_fraction`. A lookup table is `1 / vocabulary`. |
| `totals` | `tensors`, `elements`, `bytes`, `tied`. |

## 6 — D4, states

One entry per state identity instance.

| Field | Content |
|---|---|
| `identity`, `members`, `contract`, `state` | As for tensors; `members` are `node.state`. |
| `law`, `access`, `sharing` | Of the rule that applies to the members' arguments (§4.3): `append`/`window`/`fixed`; `logical_position`/`ring`/`aggregate`/`selected`; `by_position`/`by_source`/`within_span`/`at_fork_point`. |
| `stream` | The stream the state grows along `{kind, stream}` — the occurrence's own, or the stream of the port it is indexed by. |
| `indexed_by_source` | True for a state indexed by an input port's stream (cross-attention): it is frozen once that stream is complete (§5.3). |
| `instance_key` | The identity's indices followed by the contract's key axes (O5.5): one allocation per distinct key. |
| `carried_across_fragments` | Derived: the contract's carrying condition holds and the stream is a fragmented input (§5.3, V16). |
| `span`, `stride` | Evaluated modulators (O5.8), when the rule declares them. |
| `payload` | Per component: `role`, `dtype` (the binding's selection or the role's default), `shape`, `elements`, `bytes`. |
| `bytes_per_cached_position`, `bytes_bounded` | The payload's bytes; for a `window`, the bytes of a full ring. |
| `operations` | The effects the state admits (O5.4). |
| `visits` | The rule of §7 in words: when the state is written and read, per element of its stream or of the source stream. |
| `totals` | `identities`, `by_law`, `append_bytes_per_cached_position` (the "cache bytes per token" of a decoder), `bounded_bytes`, `fixed_bytes`, `carried`. |

## 7 — D5, logical costs

| Field | Content |
|---|---|
| `parameters` | Resident `elements` and `bytes` — a tied tensor once — with status `exact`. |
| `operations` | Per `element`, `cached_position`, `sequence` and `invocation`: the inventory rule of §4.1 (two operations per weight element per element, at the activated fraction of a sparsity unit) plus every applying correction, each as `{value, status}`. |
| `corrections` | Every correction that applies: `node`, `contract`, `entry`, `value`, `status`, `per`. |
| `sparsity` | Every sparsity unit in use: `activated_per_element`, `units`, `activated_fraction`, and the `union_per_invocation` bound with its status. |
| `state` | The D4 totals: append bytes per cached position, bounded bytes, fixed bytes. |
| `cuts` | The payload of every D2 cut, per element and per invocation. |

Never operations actually executed (§4.1). The known approximations are documented in §4.1 of
the specification and not modelled here.

## 8 — D6, legal cuts and partitions

| Field | Content |
|---|---|
| `cuts` | The D2 cuts by name, with their block sizes and the number of crossing values. |
| `partitions` | For every node — template instances expanded — every partition its contract declares whose condition holds: `target` (an argument axis, an instance-key axis, a state payload axis, `any_axis`, `none`) and the logical `communication` (`none`, `all_reduce`, `all_gather`, `all_to_all`). |
| `information_loss` | Every parameter slot axis whose extent is a product and that declares no factors (O5.10): partitionability along its factors is unknown and is reported as such, never as non-partitionability. |

Partitions are semantic: an implementation may support fewer, and which of these cuts is a good
one is decided with the machine's topology and the workload (§10.3).

## 9 — Generating

```sh
python3 tools/tensorspine --d1     data/models/llama3-8b.json -o /path/     # graph only
python3 tools/tensorspine --derive data/models/llama3-8b.json -o /path/     # D1 to D6
python3 tools/tensorspine --derive -o /path/                                 # whole corpus
python3 tools/tensorspine --derive data/models/decoder-causal-yarn/1.0.0.json -o /path/ \
  --assign '{"width":3072,"layers":26,"heads":32,"kv_heads":8,"head_dim":128,
             "inner":9216,"eps":0.00001,"precision":"bf16"}'
python3 tests/run_derived.py     # every document on the schema, and its facts
```

A file is named `<model>.d1.json` or `<model>.derived.json`; a template's,
`<name>@<version>.derived.json`. `tests/run_derived.py` checks that every document of the
corpus validates against the schema and that what it says agrees with the validator and with
independently known facts (Llama 3 8B's 128 KiB per token, Whisper's cross-attention cache on
the audio stream, Voxtral's carried encoder states).
