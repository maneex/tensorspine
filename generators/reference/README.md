# Reference generator

The reference generator is the repository's target generator and the language's review instrument.
It runs the reference implementation supplied with each contract; once bound to that contract
version, the implementation is its witness and the authority for what the primitive computes. The
TensorSpine conformance tooling checks optimized implementations against the corresponding unit
fixtures. The reference generator also loads a model document, its derived products and artifact into one `forward`, compares
integration fixtures from the delivery implementation at legal cuts and states, and provides a
small chat. It is not a serving system.

The generated [status page](https://maneex.github.io/tensorspine/status/) lists the contracts and
documents the generator admits, fixture tolerances and provenance, recorded token sequences, and
whole-model checks. It reads the same manifest, `verified.py` and fixture metadata as the test
suite, so this README describes the method rather than copying results.

The suite checks conformers against witness-produced unit fixtures. Separately, it compares values
at every legal cut, persistent state after prefill and exposed outputs against integration fixtures
dumped from `transformers`; generative models also compare greedy tokens. A document without a
generative output runs once: `run` prints its exposed outputs, the test compares them, and no decode
loop is entered.

The generator supports session-based chat, greedy or sampled streaming, artifact-owned chat
templates and optional decode compilation. `--max-ram GIB` runs a model in blocks at D6 legal cuts,
loading and releasing each block per invocation; the suite checks its outputs against one-block
execution and the command prints the resulting traffic.

## Capabilities

`generators/reference/capabilities.json` is the generator's manifest (the format is
[`generators/CAPABILITIES.md`](../CAPABILITIES.md)): what it can evaluate, in the language's
vocabulary, generated from the kernels' `CAPABILITIES` tables — the same tables `supports()` is
computed from, so the manifest cannot drift from the code, and the test regenerates it to prove
it. Two readers: a runtime asks `tensorspine --capabilities generators/reference/capabilities.json
MODEL…` whether this generator can run a document for a delivery of inputs, before loading anything;
a maintainer asks `--coverage` what the catalog and the corpus still need. `ref.py capabilities
[--check]` regenerates it and validates its names against the catalog.

## Commands

Every command takes a model document (derived in-process) or a derived document, and the options
`--device cpu|cuda[:i]` (default `cpu`), `--compute f32|bf16` (default fp32 on CPU, bf16 on CUDA),
`--capacity N` positions per session (default 1024), `--max-ram GIB` (blocks at legal cuts, below),
`--truncate decoder.layer=N` (a truncated document, for smoke tests) and `--set path=value` (any
other document edit before deriving).

Weights, derived documents and dumps are **runtime inputs**, not part of this repository, and one
shell variable names all of them — `$TENSORSPINE_MODEL_ARTIFACTS`, the same one the ZML generator reads:

```
$TENSORSPINE_MODEL_ARTIFACTS/
  derived/                 the derived documents
  weights/<artifact>/      what `--checkpoint` is pointed at; a symlink when they live elsewhere
  dumps/<model>/           what a run left behind
```

`GLOSSARY.md` calls one of those weight directories *"the artifact the document wraps"*. Nothing in
this repository has a default inside the tree, and the test harness says `skip` for whatever is
absent rather than looking in a home directory.

**Two words, and they are not synonyms** — the language uses both. An **artifact** is the container
the document wraps, format-agnostic: `ARCHITECTURE.md` lists it beside the implementation, the
deployment intent and the hardware, and `SPECIFICATION.md` I9 says *"the described model and loaded
artifact are mutually compatible"*. A **checkpoint** is one concrete safetensors directory — what
`--checkpoint` is pointed at and what V17 is checked against. So the prose here says *artifact*, and
*checkpoint* appears only where a flag or V17 does. In the examples below, set `IDS`, `STEPS`,
`CAPACITY`, `RAM_GIB`, `CUT` and `LAYERS` for the run being exercised.


```sh
CK="$TENSORSPINE_MODEL_ARTIFACTS/weights/Meta-Llama-3-8B"
MODEL=data/models/llama3-8b.json
R=generators/reference/ref.py

python3 $R info    $MODEL --capacity "$CAPACITY"                 # D3/D4 bytes, free memory, refusals
python3 $R verify  $MODEL --checkpoint "$CK"                     # V17 from headers; no tensor read
python3 $R run     $MODEL --checkpoint "$CK" --ids "$IDS" --steps "$STEPS"
python3 $R info    $MODEL --capacity "$CAPACITY" --max-ram "$RAM_GIB"
python3 $R run     $MODEL --checkpoint "$CK" --ids "$IDS" --max-ram "$RAM_GIB"
python3 $R chat    $MODEL --checkpoint "$CK" --capacity "$CAPACITY" --max-new-tokens "$STEPS"
# the comparison against transformers, at every legal cut and every state after prefill
python3 $R run     $MODEL --checkpoint "$CK" --truncate "$CUT" \
                   --ids "$IDS" --steps "$STEPS" --dump /tmp/ours.safetensors
python3 generators/reference/fixtures/dump_hf.py --model "$CK" --document llama3-8b \
                   --layers "$LAYERS" --ids "$IDS" --steps "$STEPS" --out /tmp/theirs.safetensors
python3 $R compare /tmp/ours.safetensors /tmp/theirs.safetensors     # at the fixture's own tolerance; --atol/--rtol override
python3 $R witness attention.dense@1.0.0            # the unit fixtures of a contract version, regenerated and compared; --record writes them
python3 $R witness all --record                       # every case every kernel declares (docs/TENSORSPINE-FIXTURE.md)
python3 generators/reference/tests/run_reference.py [--compile] [--full]   # random weights; fixtures and full models when artifacts are present
```

### Chat

`ref.py chat MODEL --checkpoint DIR` verifies the artifact against the document, prints the
feasibility line, then prompts `you> `; the reply streams after `bot> `; an empty line quits. The
session persists across turns — positions accumulate up to `--capacity`, and exceeding it is a
refusal, not an eviction (eviction is a contract property, `window`, not a runtime policy).

| Option | Meaning |
|---|---|
| `--max-new-tokens N` | Upper bound on the reply. |
| `--temperature T`, `--top-p P`, `--seed N` | Sampling; zero temperature selects greedy decoding. |
| `--compile` | `torch.compile` of the decode step; compilation occurs on the first decode |
| `--truncate decoder.layer=N` | Run a truncated graph as a smoke test of the loop. |

The tokenizer and the chat template come from the artifact; only the tokenizer is instantiated,
never the model class. Each turn renders the whole transcript and feeds the suffix beyond what the
session has consumed; when the template rewrites the history (Qwen 3.5 strips earlier thinking
blocks), the prefix check says `(the template rewrote the prefix: session restarted)` and rebuilds
the session from the whole transcript. A base model without a template (Llama 3 8B) gets a plain
`User: … / Assistant:` transcript. Stop tokens come from `generation_config.json` and the tokenizer.
The CLI does not expose Qwen's template option `enable_thinking=False`.

The reference implementation uses a per-token Python recurrence and may upcast each operation to
fp32. Speed is a non-goal.

### Memory

`load_parameters` memory-maps safetensors shards; the kernel pages a weight in when an operation
touches it and the operating system may evict it under pressure. Anonymous allocations are the
per-operation compute-dtype copy of a weight, states and activations. `info` reports declared bytes
from D3/D4 and the computed resident set: states and the largest temporary on CPU, weights on CUDA,
or the largest block and its payload under `--max-ram`.

### Blocks under `--max-ram`

`--max-ram GIB` makes the streaming explicit and deterministic. The plan cuts D1's order into
blocks at D6's `layer` cuts — membership is the ancestor closure of each cut's payload producers,
so it holds for any document; legality (every crossing edge forward) is D6's — and merges
consecutive layers greedily while a block's parameters, the payload crossing into it for
`--capacity` elements, the states and the largest per-operation temporary stay under the bound.
Every invocation then materialises one block at a time (an owned copy on the device; on CUDA the
block is what lives on the card), runs it, and releases it; a tied identity used by two blocks is
held and loaded by both. The outputs are the one-block outputs bit for bit — the test checks random
weights and integration fixtures — and the cost is printed. Whenever `--max-ram` is set, `info`,
`run` and `chat` print the cut summary first: one line per block with the legal cut it opens with
and the one it closes at (D6's names), its nodes, its parameter bytes and the payload crossing into
it, then what stays resident and the traffic — the whole model's bytes per decode step. A bound
below one layer plus the resident part is refused with the computed requirement. This mode exposes
storage performance by design; prefetching the next
block, reading directly into its owned buffer and storage selection remain engine concerns.



What it reads: the derived document only (D1 graph and arguments, D2 shapes, D3 tensors, D4
states, D6 cuts) — derived in-process from a model document through `tools/derive.py` — never
the model source or the catalog. What it knows about a model: nothing; `--truncate` and `--set`
edit the *document* before deriving and are a test convenience.

An input the document declares may deliver nothing in an invocation (§7): the occurrences it alone
would reach are not evaluated and need no kernel, `splice` keeps its `text`, and only the inputs D2
marks `required_for` the output at hand are refused when absent. The suite compares a multimodal
document's text-only path with its text-trunk document. Sending an image waits on the language saying
where the inserted elements go (`splice`
placement, a parked finding).

Conventions: a value's tensor has the element axis first, then the port's axes in the contract's
order; one sequence, no batch axis; parameters stay at their D3 dtype and are upcast per
operation (`--compute f32` on CPU, `bf16` on CUDA); `append` states are buffers of `--capacity`
positions written at a cursor, and a kernel masks beyond the length. Each kernel's docstring lists
the contract's branches it implements or refuses, and the conventions the contract leaves open.
