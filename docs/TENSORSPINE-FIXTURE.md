# TensorSpine fixture — what a generator is checked against, as one file

> A fixture is recorded once and run by every generator's harness: the language owns the format,
> not the harness that happened to write it first.

*Companion to `schemas/tensorspine-fixture.schema.json` (`$id`
`https://tensorspine.dev/schema/2.0/fixture.json`). Non-normative: the
[specification](SPECIFICATION.md) declares the witness of a contract version (§4.1, O1.3) and what
a conformer owes it (§4.2); this document is how the evidence is written down.*

---

## 1 — Why a format, and why one

Two kinds of evidence say that an implementation of the language is right. At the **unit** level a
contract version has one witness — the reference generator's kernel for it — and every other
implementation is a conformer, checked against what the witness computes (Specification §4.1,
§4.2). At the **integration** level a whole model is compared with the implementation that
delivered it, `transformers` by usage, at the legal cuts of its document and at every state after
the prefill: the check that a transcribed document's wiring is right. Both kinds were already
produced and consumed — the reference generator dumped `transformers` at its cuts, and the ZML
generator read those dumps — as a naming convention two harnesses shared. A convention is not a
format: nothing said what a file had to contain, nothing refused a file that lacked it, and a third
generator would have had to read one harness's code to learn what the other wrote.

The fixture schema is that statement. A fixture is a safetensors file; its `__metadata__` is the
document below, one JSON value per key, and its tensors are named by the grammar of its kind. The
container is safetensors because every generator already reads it — the checkpoints are — and
because a header can be read without loading a tensor. A file whose metadata is off the schema is
not a fixture: the reference generator refuses to write one and refuses to read one.

## 2 — The two kinds

| Kind | Produced by | What it holds | Who runs it |
|---|---|---|---|
| `unit` | the witness of one contract version, on random parameters and inputs (`ref.py witness NAME@VERSION`) | the one-occurrence document it ran, its parameters, and the inputs, positions, states and outputs of each invocation | every conformer of that contract version, at the tolerance of its compute dtype |
| `integration` | the delivery implementation of a whole model at the document's legal cuts (`fixtures/dump_hf.py`) | the output of every layer, every state after the prefill, the logits, the greedy tokens, and the provenance of the weights | every generator that runs the document |

A unit fixture is small — the witness runs at small arguments (a width of 64, a vocabulary of 256)
— and there are several per contract, one per branch worth a case. An integration fixture is a
truncated model (the first few layers), because that is what fits beside `transformers` on one
machine; the untruncated greedy tokens are recorded separately by the generator's test.

## 3 — The metadata

### Common

| Field | Content |
|---|---|
| `schema` | `tensorspine-fixture/1` |
| `kind` | `unit` or `integration` |
| `compute` | The dtype the recorded tensors were computed in (`f32`, `bf16`, `f16`). The tensors themselves are stored as f32, so that a comparison never rounds twice. |
| `tolerance` | Per compute dtype of the implementation being checked: `{atol, rtol}`, the test being `|a − b| ≤ atol + rtol·|b|` element by element, `b` the recorded value. A dtype without an entry is not covered by the fixture — a harness refuses to compare rather than guess. |

### `unit`

| Field | Content |
|---|---|
| `id` | `<name>@<version>/<case>` — what the witness's manifest lists under `witness.fixtures`. |
| `contract` | `{name, version}`. |
| `arguments` | The occurrence's resolved arguments, declared defaults applied: D1's. |
| `document` | The one-occurrence model document the witness ran: the occurrence under its arguments, one public input per input port (a `token` input for a port that inherits its kind, else the port's kind), one public output per output port, every slot bound to an identity named after it and located at the fixture's own `param/<identity>` key — the fixture is its own checkpoint, so a conformer loads it the way it loads any artifact. Its catalog base is relative to the fixture's own directory. |
| `invocations` | One entry per invocation, in order: the elements each public input delivers, by name (`[{"input": 5}, {"input": 3}]`: five elements, then three); an input absent from an entry delivers nothing, as an insert transform's source may (§7). States carry over; positions continue per stream. |
| `seed` | The seed of the parameters and inputs. The witness regenerates the fixture from it, which is how a silent change of the witness is caught (§5). |
| `witness` | `generator` (`reference`), its `version` (a commit), the `kernel` file, and the library `versions`. |

Tensor keys: `param/<identity>` for every D3 identity of the document; `in/<k>/<input>` for every
input delivering in invocation `k` (from 0), `positions/<k>/<stream>` for each stream's elements
in it and `out/<k>/<output>` for every output it produced; `state/<k>/<identity>/<component>` for
every D4 state after invocation `k` — a growing state holds the positions written so far, a window
its span in chronological order zero-padded before the first write, a fixed state its whole
payload.

### `integration`

| Field | Content |
|---|---|
| `document` | The corpus document, by name. |
| `artifact` | `name` (the directory, never a path of the machine that made it), `id` (the published identifier), and `index_sha256` for a sharded checkpoint or `header_sha256` for one file without an index. |
| `delivery` | `implementation` (`transformers`), the `program` that dumped it, the library `versions`. |
| `truncation` | `composition` and `layers`: the document is truncated to as many layers of that composition before it is compared. |
| `ids`, `tokens` | The prompt, and the greedy tokens the delivery implementation produced after it (empty for a document without a generative output). |
| `hook_map` | How the delivery implementation's names met D1 values and D4 states — data, and the only place they meet. |

Tensor keys: `value/<D1 value>` for the output of every layer (the values crossing D6's layer
cuts) and every exposed output; `state/<D4 identity>/<component>` for every state after the
prefill; `logits/last` and `logits/argmax` for a generative document.

## 4 — Reading one

```sh
python3 generators/reference/ref.py compare OURS FIXTURE          # tolerance from the fixture, for OURS's compute dtype
python3 generators/reference/tests/run_reference.py               # every integration fixture whose checkpoint is on disk; every unit fixture regenerated
python3 tests/run_fixtures.py                                     # every committed fixture is on the schema and its keys on the grammar
```

A harness reads the metadata first (`compare.read_fixture` refuses a file off the schema), takes
the ids, the truncation and the tolerance from it, runs its own generator, and compares key by
key. Nothing about a fixture is written in a harness: the ZML harness reads the same files as the
reference generator's, and a generator that is not in this repository reads them the same way.

## 5 — Regenerating one

A unit fixture is a function of the witness and the seed. The reference generator's test
regenerates every committed unit fixture in memory and compares it with the file: a difference
beyond the fixture's own `f32` tolerance means the witness changed. That is refused unless the
contract version changed with it — a change of meaning is a new identity — or the change is a
correction of the witness toward the contract's declared meaning, which is a patch whose
re-recorded fixtures say so in the version note (Specification §8.2). `ref.py witness
NAME@VERSION --record` rewrites the files; nothing else does.

An integration fixture is regenerated by `fixtures/dump_hf.py` against the artifact, and the
reference generator's test compares its own run against it whenever the artifact is on disk
(`$TENSORSPINE_MODEL_ARTIFACTS/weights/<artifact.name>`), else says `skip`.
