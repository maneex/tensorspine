# Reference generator

The reference *generator* — a loader and a reference implementation of the primitives, in PyTorch — turns a model document,
its derived products and the model artifact it names into a module with one `forward`, a comparison
against the official implementation at every layer boundary, and a small chat. Not a serving
system. Its plan, and the location plan whose weight locations it loads by, are working notes
outside the tree.

**Status: M0, M1, M2 and M4; M3 (CUDA) waits for a GPU.** Six models run from their documents
and the artifacts they name, with no per-model code, and agree with `transformers` (a seventh
document, the multimodal `qwen3.8-27b`, runs on text and gives `qwen3.8-27b-text`'s logits bit for
bit — below):

- `llama3-8b` (six contracts, one state law): on the 3-layer fixture, layer outputs within 2e-7,
  KV states within 3e-6, logits within 8e-6; on the full 32-layer model the eight greedy tokens
  after *"The capital of France is"* are `transformers`' — *" Paris. It is located in the north"*.
- `qwen3.5-4b-text` (seven contracts; `append`, `window`/`ring` and `fixed` states; the 3:1
  gated-delta / gated-attention hybrid from D1; tied embeddings): on the 4-layer fixture, layer
  outputs within 1e-6, the KV, convolution-history and recurrent states within 5e-6, logits and
  greedy tokens identical; on the full 32-layer model the eight greedy tokens after *"The capital
  of France is"* are `transformers`' — *" Paris.\nA. True\nB"* — at ~4–10 s per token here.
- `qwen3.8-27b-text` (the same seven contracts, 64 layers, 50.1 GiB — larger than this machine's
  memory): on its 4-layer fixture within 4e-6 of `transformers` at every cut and state; the full
  model runs under `--max-ram 8` in seven blocks (below).
- `shieldstral-3b` (Llama's six contracts with YaRN rotary scaling and tied embeddings — the text
  trunk of a Mistral 3 multimodal artifact, `mistralai/Shieldstral-1.0-3B`, one `model.safetensors`
  without an index; its Pixtral tower is located but not evaluated on text): on the 3-layer fixture
  within 3.9e-6 of `transformers` at every cut and state; on the full 26-layer model the eight greedy
  tokens after *"Hello! Can you help me plan a birthday party?"* are `transformers`' — *"no</s>no</s>…"*,
  a moderation fine-tune's verdict. On *"The capital of France is"* the model's first token is a tie
  in bf16 ("no" and "yes" both at 28.25) that fp32 breaks the other way: a tolerance question, not a
  kernel's, and the reason the prompt is a different one.
- `colbert-v2` (BERT-base, post-LN, **stateless**: `embedding.token_position_type`, `norm.layer`,
  `ffn.dense`, bidirectional `attention.dense` with biases, `pooler` — one 128-vector per token,
  L2-normalised; `colbert-ir/colbertv2.0`, fp32, one file): the whole 12-layer model within 3.8e-6 of
  `transformers` at every layer and on the embeddings of *"[CLS] [Q] what is the capital of france ? [SEP]"*.
  A document without a generative output is one invocation: `run` prints the exposed outputs, the
  test compares them, nothing is decoded and no state exists.
- `qwen3.5-35b-a3b` (the 397B's structure at its quantities: 40 layers, 256 routed experts of
  which 8 per token plus a gated shared expert, `moe` in every layer; `Qwen/Qwen3.5-35B-A3B`, 72 GB,
  every tensor located and — a first — the fused expert layout verified against the shards): on
  the 2-layer fixture (fp32) within 3.8e-6 of `transformers` at every cut and state; on the 4-layer
  fixture, which adds the attention layer, `transformers` runs in bf16 (four fp32 layers exceed the
  memory here) and the fp32 reference is within 8.2e-2 absolute of it, tokens identical — the
  tolerance recorded beside the fixture; the full model streams 65 GiB per token under `--max-ram`.

Kernels: `embed`, `embedding.token_position_type`, `norm.rms` (zero-centred scales), `norm.layer`,
`attention.dense` (causal or bidirectional, GQA, biases, RoPE full or partial, YaRN scaling, gated
query, Q/K RMS norms), `residual.add`, `ffn.gated`, `ffn.dense`, `moe` (learned softmax routing,
renormalised or not, fused experts, gated shared experts), `lm_head`, `pooler`,
`sequence.gated_delta`. A session-based chat (`ref.py chat`), greedy or sampled, streaming, through
the artifact's own chat template when it has one (Qwen 3.5 4B answers with its thinking block;
when its template strips earlier thinking from the history, the prefix check rebuilds the session
and says so) or a plain transcript for a base model; `--compile` for the decode step. `--max-ram GIB` runs a model in blocks of layers cut at D6's legal
cuts, loading and releasing each block every invocation (M4): the same logits bit for bit, the
traffic printed. Next: M3 (CUDA).

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
*checkpoint* appears only where a flag or V17 does.


```sh
CK="$TENSORSPINE_MODEL_ARTIFACTS/weights/Meta-Llama-3-8B"
R=generators/reference/ref.py

python3 $R info    data/models/llama3-8b.json --capacity 4096          # bytes from D3/D4, free memory, refusals
python3 $R verify  data/models/llama3-8b.json --checkpoint $CK         # V17 against the file headers, nothing read
python3 $R run     data/models/llama3-8b.json --checkpoint $CK --ids 128000,791,6864,315,9822,374 --steps 7
python3 $R run     data/models/llama3-8b.json --random --set quantities.d.source.value=64 …   # no checkpoint
python3 $R info    data/models/llama3-8b.json --capacity 4096 --max-ram 6                    # the partition and its traffic
python3 $R run     data/models/llama3-8b.json --checkpoint $CK --ids 128000,791,6864,315,9822,374 --max-ram 6
python3 $R chat    data/models/llama3-8b.json --checkpoint $CK --capacity 512 --max-new-tokens 60
python3 $R chat    data/models/qwen3.5-4b-text.json --checkpoint "$TENSORSPINE_MODEL_ARTIFACTS/weights/Qwen3.5-4B" \
                   --capacity 1024 --max-new-tokens 200
# the comparison against transformers, at every legal cut and every state after prefill
python3 $R run     data/models/llama3-8b.json --checkpoint $CK --truncate decoder.layer=3 \
                   --ids 128000,791,6864,315,9822,374 --steps 3 --dump /tmp/ours.safetensors
python3 generators/reference/fixtures/dump_hf.py --model $CK --document llama3-8b --layers 3 \
                   --ids 128000,791,6864,315,9822,374 --steps 3 --out /tmp/theirs.safetensors
python3 $R compare /tmp/ours.safetensors /tmp/theirs.safetensors     # at the fixture's own tolerance; --atol/--rtol override
python3 $R witness attention.dense@1.0.0            # the unit fixtures of a contract version, regenerated and compared; --record writes them
python3 $R witness all --record                       # every case every kernel declares (docs/TENSORSPINE-FIXTURE.md)
python3 generators/reference/tests/run_reference.py [--compile] [--full]   # M0 on random weights; M1/M2 on the fixtures when the artifacts are on disk
```

### Chat

`ref.py chat MODEL --checkpoint DIR` verifies the artifact against the document, prints the
feasibility line, then prompts `you> `; the reply streams after `bot> `; an empty line quits. The
session persists across turns — positions accumulate up to `--capacity`, and exceeding it is a
refusal, not an eviction (eviction is a contract property, `window`, not a runtime policy).

| Option | Meaning |
|---|---|
| `--max-new-tokens N` (256) | the reply's bound; Qwen 3.5 thinks before it answers, so give it room or it stops mid-thought |
| `--temperature T` (0 = greedy), `--top-p P`, `--seed N` | sampling |
| `--compile` | `torch.compile` of the decode step; the first decode pays ~20 s |
| `--truncate decoder.layer=4` | a truncated model: loads in seconds, answers garbage — a smoke test of the loop |

The tokenizer and the chat template come from the artifact; only the tokenizer is instantiated,
never the model class. Each turn renders the whole transcript and feeds the suffix beyond what the
session has consumed; when the template rewrites the history (Qwen 3.5 strips earlier thinking
blocks), the prefix check says `(the template rewrote the prefix: session restarted)` and rebuilds
the session from the whole transcript. A base model without a template (Llama 3 8B) gets a plain
`User: … / Assistant:` transcript. Stop tokens come from `generation_config.json` and the tokenizer.
Not exposed yet: switching Qwen's thinking off through the template (`enable_thinking=False`).

What to expect on this CPU: about 4–10 s per token for the 4B, ~7 s per token for the 8B (its 15 GB
of weights are memory-mapped and paged from disk at every step — see below) — a Qwen answer with
its thinking block is a few minutes. That is the reference implementation's per-token Python
recurrence and per-operation fp32 upcast, not a defect; speed is a non-goal.

### Memory

`load_parameters` memory-maps the safetensors shards: 291 tensors, 15 GiB declared, load in 0.1 s
and add ~20 MiB of resident memory; the kernel pages a weight in when an operation touches it and
evicts it under pressure. The only anonymous allocations are the per-operation fp32 upcast of one
weight (2 GiB for the 8B's `lm_head`, 2.5 GiB for the 4B's tied embedding), the states and the
activations. A model larger than the page cache therefore runs, slowly, by implicit streaming.
`info` reports the declared bytes (D3, D4) and the bytes a run holds: on CPU the states and the
largest temporary; on CUDA the weights too; under blocks the largest block with its payload.

### Blocks under `--max-ram`

`--max-ram GIB` makes the streaming explicit and deterministic. The plan cuts D1's order into
blocks at D6's `layer` cuts — membership is the ancestor closure of each cut's payload producers,
so it holds for any document; legality (every crossing edge forward) is D6's — and merges
consecutive layers greedily while a block's parameters, the payload crossing into it for
`--capacity` elements, the states and the largest per-operation temporary stay under the bound.
Every invocation then materialises one block at a time (an owned copy on the device; on CUDA the
block is what lives on the card), runs it, and releases it; a tied identity used by two blocks is
held and loaded by both. The outputs are the one-block outputs bit for bit — the test checks it on
random weights and on both fixtures — and the cost is printed. Whenever `--max-ram` is set, `info`,
`run` and `chat` print the cut summary first: one line per block with the legal cut it opens with
and the one it closes at (D6's names), its nodes, its parameter bytes and the payload crossing into
it, then what stays resident and the traffic — the whole model's bytes per decode step. A bound below one layer plus the resident part is refused with the numbers. On
llama3-8b, `--max-ram 6` gives six blocks of about 3 GiB and 14.96 GiB of traffic per token.

Measured on `qwen3.8-27b-text` — 50.1 GiB of bf16 weights on a 31 GB machine with about 10 GB
free, a SATA SSD underneath — under `--max-ram 8`: 7 blocks at legal cuts, planned resident
7.84 GiB, **allocated peak 8.29 GiB** (`fixtures/peak_rss.py`; the excess is the unbudgeted
activations), 19.3 GiB of file-backed pages beside it that the kernel may drop; 145 s for the
prefill and ~119 s per decode step, the process in disk wait at ~360 MiB/s with under one core
busy — 50 GiB per token through a 500 MB/s disk. Tokens `[11751, 13, 198]` (" Paris.\n"), the
same as the memory-mapped one-block run's, which allocated only 0.90 GiB (23 GiB of file-backed
pages) and took 106 s for the prefill and ~98 s per step: the copy per block costs about twenty
seconds a token here, and buys a bound. The mode measures the disk, by design; prefetching the
next block, reading straight into the owned buffer, more cache or a faster disk are the levers,
and none of them is the reference generator's business.



What it reads: the derived document only (D1 graph and arguments, D2 shapes, D3 tensors, D4
states, D6 cuts) — derived in-process from a model document through `tools/derive.py` — never
the model source or the catalog. What it knows about a model: nothing; `--truncate` and `--set`
edit the *document* before deriving and are a test convenience.

An input the document declares may deliver nothing in an invocation (§7): the occurrences it
alone would reach are not evaluated and need no kernel, `splice` keeps its `text`, and only the
inputs D2 marks `required_for` the output at hand are refused when absent. A multimodal document
therefore runs on text with its vision tower silent — the test proves `qwen3.8-27b` on text gives
`qwen3.8-27b-text`'s logits bit for bit. Sending an image waits on the language saying where the
inserted elements go (the `splice` placement, a parked finding).

Conventions: a value's tensor has the element axis first, then the port's axes in the contract's
order; one sequence, no batch axis; parameters stay at their D3 dtype and are upcast per
operation (`--compute f32` on CPU, `bf16` on CUDA); `append` states are buffers of `--capacity`
positions written at a cursor, and a kernel masks beyond the length. Each kernel's docstring lists
the contract's branches it implements or refuses, and the conventions the contract leaves open.

What no fixture here measures: YaRN's extrapolation regime — positions beyond `orig_ctx` (16384 for
Shieldstral). The arithmetic is position-independent, so a short prompt already exercises the ramp
and the attention factor; the regime itself is not measured. The frequencies are checked against
`transformers`' own YaRN parameters in the test, without an artifact.
