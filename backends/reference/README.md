# Reference backend

A loader and a reference implementation of the primitives, in PyTorch, that turn a model document,
its derived products and the checkpoint it names into a module with one `forward`, a comparison
against the official implementation at every layer boundary, and a small chat. Not a serving
system. The plan is [`docs/reference-backend-plan.md`](../../docs/reference-backend-plan.md); the
weight locations it loads by are [`docs/location-plan.md`](../../docs/location-plan.md).

**Status: M2.** Two models run from their documents and the checkpoints they name, with no
per-model code, and agree with `transformers`:

- `llama3-8b` (six contracts, one state law): on the 3-layer fixture, layer outputs within 2e-7,
  KV states within 3e-6, logits within 8e-6; on the full 32-layer model the eight greedy tokens
  after *"The capital of France is"* are `transformers`' — *" Paris. It is located in the north"*.
- `qwen3.5-4b-text` (seven contracts; `append`, `window`/`ring` and `fixed` states; the 3:1
  gated-delta / gated-attention hybrid from D1; tied embeddings): on the 4-layer fixture, layer
  outputs within 1e-6, the KV, convolution-history and recurrent states within 5e-6, logits and
  greedy tokens identical; on the full 32-layer model the eight greedy tokens after *"The capital
  of France is"* are `transformers`' — *" Paris.\nA. True\nB"* — at ~4–10 s per token here.

Kernels: `embed`, `norm.rms` (zero-centred scales), `attention.dense` (causal, GQA, RoPE full or
partial, gated query, Q/K RMS norms), `residual.add`, `ffn.gated`, `lm_head`,
`sequence.gated_delta`. A session-based chat (`ref.py chat`), greedy or sampled, streaming, through
the checkpoint's own chat template when it has one (Qwen 3.5 4B answers with its thinking block;
when its template strips earlier thinking from the history, the prefix check rebuilds the session
and says so) or a plain transcript for a base model; `--compile` for the decode step. Next: M3
(CUDA), M4 (blocks under `--max-ram`).

## Commands

Every command takes a model document (derived in-process) or a derived document, and the options
`--device cpu|cuda[:i]` (default `cpu`), `--compute f32|bf16` (default fp32 on CPU, bf16 on CUDA),
`--capacity N` positions per session (default 1024), `--truncate decoder.layer=N` (a truncated
document, for smoke tests) and `--set path=value` (any other document edit before deriving).

```sh
CK=~/work/perso/huggingface/Meta-Llama-3-8B
R=backends/reference/ref.py

python3 $R info    data/models/llama3-8b.json --capacity 4096          # bytes from D3/D4, free memory, refusals
python3 $R verify  data/models/llama3-8b.json --checkpoint $CK         # V17 against the file headers, nothing read
python3 $R run     data/models/llama3-8b.json --checkpoint $CK --ids 128000,791,6864,315,9822,374 --steps 7
python3 $R run     data/models/llama3-8b.json --random --set quantities.d.source.value=64 …   # no checkpoint
python3 $R chat    data/models/llama3-8b.json --checkpoint $CK --capacity 512 --max-new-tokens 60
python3 $R chat    data/models/qwen3.5-4b-text.json --checkpoint ~/work/perso/huggingface/Qwen3.5-4B \
                   --capacity 1024 --max-new-tokens 200
# the comparison against transformers, at every legal cut and every state after prefill
python3 $R run     data/models/llama3-8b.json --checkpoint $CK --truncate decoder.layer=3 \
                   --ids 128000,791,6864,315,9822,374 --steps 3 --dump /tmp/ours.safetensors
python3 backends/reference/fixtures/dump_hf.py --model $CK --layers 3 --ids 128000,791,6864,315,9822,374 \
                   --steps 3 --out /tmp/theirs.safetensors
python3 $R compare /tmp/ours.safetensors /tmp/theirs.safetensors [--atol 1e-3 --rtol 1e-2]
python3 backends/reference/tests/run_reference.py [--compile] [--full]   # M0 on random weights; M1/M2 on the fixtures when the checkpoints are on disk
```

### Chat

`ref.py chat MODEL --checkpoint DIR` verifies the checkpoint against the document, prints the
feasibility line, then prompts `you> `; the reply streams after `bot> `; an empty line quits. The
session persists across turns — positions accumulate up to `--capacity`, and exceeding it is a
refusal, not an eviction (eviction is a contract property, `window`, not a runtime policy).

| Option | Meaning |
|---|---|
| `--max-new-tokens N` (256) | the reply's bound; Qwen 3.5 thinks before it answers, so give it room or it stops mid-thought |
| `--temperature T` (0 = greedy), `--top-p P`, `--seed N` | sampling |
| `--compile` | `torch.compile` of the decode step; the first decode pays ~20 s |
| `--truncate decoder.layer=4` | a truncated model: loads in seconds, answers garbage — a smoke test of the loop |

The tokenizer and the chat template come from the checkpoint; only the tokenizer is instantiated,
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
activations. A model larger than the page cache therefore runs, slowly, by implicit streaming;
`--max-ram` (M4, planned) makes that explicit and deterministic — blocks at D6's legal cuts, loaded
and released in order — and is what CUDA needs. `info` still counts the declared bytes against the
free memory and is conservative on CPU for that reason.


What it reads: the derived document only (D1 graph and arguments, D2 shapes, D3 tensors, D4
states, D6 cuts) — derived in-process from a model document through `tools/derive.py` — never
the model source or the catalog. What it knows about a model: nothing; `--truncate` and `--set`
edit the *document* before deriving and are a test convenience.

Conventions: a value's tensor has the element axis first, then the port's axes in the contract's
order; one sequence, no batch axis; parameters stay at their D3 dtype and are upcast per
operation (`--compute f32` on CPU, `bf16` on CUDA); `append` states are buffers of `--capacity`
positions written at a cursor, and a kernel masks beyond the length. Each kernel's docstring lists
the contract's branches it implements or refuses, and the conventions the contract leaves open.
