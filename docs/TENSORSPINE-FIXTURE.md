# TensorSpine fixture format

> Record conformance evidence once; run the same schema-checked file against every implementation.

*Companion to `schemas/tensorspine-fixture.schema.json` (`$id`
`https://tensorspine.dev/schema/2.0/fixture.json`). Non-normative: the
[specification](SPECIFICATION.md) declares the witness of a contract version (§4.1, O1.3) and what
a conformer owes it (§4.2); this document is how the evidence is written down.*

---

## 1 — Why a format, and why one

Two kinds of evidence cover different boundaries:

- A **unit fixture** records the reference implementation supplied with one contract version. The
  reference generator executes it; every optimized implementation of that contract is a conformer
  checked against the result (Specification §4.1–§4.2).
- An **integration fixture** records a model's delivery implementation at the legal cuts and states
  of its TensorSpine model document. It checks that the document reproduces the delivered model's
  wiring.

The fixture schema is that statement. A fixture is a safetensors file; its `__metadata__` is the
document below, one JSON value per key, and its tensors are named by the grammar of its kind. The
container matches the model-artifact ecosystem and lets readers inspect metadata without loading a
tensor. TensorSpine's conformance tooling refuses a file whose metadata or tensor names are outside
the schema.

## 2 — The two kinds

| Kind | Produced by | What it holds | Who runs it |
|---|---|---|---|
| `unit` | Contract reference implementation, on generated parameters and inputs (`ref.py witness NAME@VERSION`) | One-occurrence document, parameters, inputs, positions, states and outputs for each invocation | Every conformer of that contract version, at the tolerance for its compute dtype |
| `integration` | Model delivery implementation at the document's legal cuts (`fixtures/dump_hf.py`) | Cut values, post-prefill states, exposed outputs, generated tokens, the non-token inputs the prefill delivered, and artifact provenance | Every implementation that runs the document |

Unit fixtures use deliberately small arguments and cover distinct contract branches. An integration
fixture may truncate a composition; its metadata records that boundary, so no reader guesses it.

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
| `inputs` | The provenance of every non-token public input the prefill delivered, keyed by the input's name — one entry per `in/<input>` tensor key, and no other. A recording names its `source` (a file name, never a path), the `sha256` of that file, its `origin` (where the recording came from: a repository, an archive item, a URL) and the `licence` it is used under — a fixture is committed and read elsewhere, so open content carries its attribution itself, and the test requires the two whenever `source` is present — and `processor` (how the tensor was made from it: the extractor and its configuration, in words). A setting the delivery chose rather than recorded — the number of delay tokens of a streaming transcription — has no file: `processor` alone says what chose it, and `source` and `sha256` are absent together. Absent when the prompt is the only input. The delivery's preprocessing is outside the document, as a tokenizer is; the fixture holds its result and says how it was made. |
| `hook_map` | How the delivery implementation's names met D1 values and D4 states — data, and the only place they meet. |

Tensor keys: `in/<input>` for every public input the prefill delivered other than the token input
`ids` names — element-major, one row per element, the input value's D2 shape per row (`[3000, 128]`
for Whisper's `audio`; what `Session.run` checks a delivered input against — for an input whose
stream the token input joins, every element the prefill and the recorded steps consumed, the
prompt taking its tokens' share and each step one token's, as D2's count says); `value/<D1
value>` for the output of every layer (the values crossing D6's layer cuts) and every exposed
output; `state/<D4 identity>/<component>` for every state after the prefill — a growing state's
positions, a `window` state's valid tail (the last `span` positions written, in order, and
nothing before the first write: what the delivery's cache holds), a fixed state's payload;
`logits/last` and `logits/argmax` for a generative document.

## 4 — Reading one

```sh
python3 generators/reference/ref.py compare OURS FIXTURE          # tolerance from the fixture, for OURS's compute dtype
python3 generators/reference/tests/run_reference.py               # every integration fixture whose checkpoint is on disk; every unit fixture regenerated
python3 tests/run_fixtures.py                                     # every committed fixture is on the schema and its keys on the grammar
```

A conformance runner reads the metadata first (`compare.read_fixture` refuses a file off the
schema), takes the ids, truncation and tolerance from it, runs the implementation under test, and
compares key by key. The reference and ZML runners consume the same files; another implementation
does not need repository-specific fixture knowledge.

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
