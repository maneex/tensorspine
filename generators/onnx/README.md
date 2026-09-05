# ONNX generator

The third generator: it **emits** an ONNX graph from a derived document and a checkpoint, and
runs it through onnxruntime. Where the reference executes each contract's kernel and ZML traces
the graph into MLIR, this generator writes the graph down in the interchange form serving
engines read. It knows nothing about any model: the derived document (D1–D6, `tensorspine
--derive`) is all it reads, never the model source or the catalog.

## What is emitted

One ONNX graph is **one invocation** (Specification §7), for one set of delivered inputs:

- every public input delivered is a graph input with a dynamic element axis (`tokens` as int64
  identifiers; a floating input in the compute dtype) — under `--batch aligned`, a batch axis before
  it (Batching, below);
- the positions of every stream delivered are an input (`positions/<stream>`, int64);
- every D4 state is an input and an output — `state/<identity>/<component>` in, `state_out/…`
  out — since ONNX has no mutable state: the **session** keeps the tensors between runs, an
  `append` state as the positions held so far (`Concat` of the past and the new);
- every D3 identity is an initializer at its stored dtype (bf16 stays bf16), cast to the compute
  dtype once where a primitive uses it (the runtime folds the cast); a tied identity is one
  initializer, and an embedding table is gathered before it is cast;
- every occurrence a delivery evaluates is emitted by its contract's primitive
  (`primitives/<name>.py`), in D1's order, its outputs named `<node>.<port>`; the exposed outputs
  are graph outputs, and with `--dump` so is every value crossing a D6 layer cut.

Opset 17, IR 8. The **target** is an argument of the generator (`--target`), the runtime the graph
is emitted for: `onnx`, the default, composes every primitive from the standard operators and the
graph runs on any runtime; `onnxruntime` emits that runtime's fused operators wherever a primitive
has a form for it (below), and the portable form everywhere else. The session emits one graph per
distinct delivery (a prefill with the audio and a decode without it would be two) and caches the
onnxruntime session for each.

## Targets: opaque arguments flowing to the primitives

The target is one of the **physical parameters** — the opaque arguments the language leaves to a
generator ([`generators/CAPABILITIES.md`](../CAPABILITIES.md); derivation never sees them) —
flowing from the generator to its primitives: the `backend` key, set for every occurrence by
`--target` and overridden per occurrence by `--physical` (a file keyed by exact identifier, by
site pattern with `*`, or by contract version, the more specific entry winning), so one graph may
mix targets. Every other key flows the same way and a primitive reads it from its context
(`ctx.physical`; the fused attention takes `rotary_cache`, the positions its cos/sin caches cover).

The generator dispatches on the target by directory, with a fallback: `primitives/<name>.py` is
the portable emitter of a contract, `primitives/<target>/<name>.py` the target's own, laid over
the portable table contract by contract (`registry.load_primitives(target)`); a contract the
target has no file for keeps the portable form, and a target's emitter falls back to the portable
one for the branches its fused operator does not cover. A new target is a directory. The
`onnxruntime` target today (`com.microsoft`):

- `attention.dense`: causal attention over an `append` state with a full-head rope or none as one
  `GroupQueryAttention`, the rotary inside it from cos/sin caches (YaRN's frequencies and attention
  factor folded into the caches); a non-causal mask, a partial or interleaved rope, an output gate
  or a head size that is not a multiple of 16 keeps the portable form;
- `norm.rms`: `SimplifiedLayerNormalization`, and when its input is the sum a `residual.add`
  occurrence just produced, the two occurrences as one `SkipSimplifiedLayerNormalization` whose
  sum output replaces the Add for every consumer — a fusion across occurrences, reasoned from the
  topology (the value's origin), never from the model's name;
- `ffn.gated`: the activation as `QuickGelu` with alpha 1 (SiLU exactly), `Gelu` or `FastGelu`.

A fused form is checked as the composed one is: the same unit and integration fixtures at the same
tolerances, under `--target onnxruntime`.

## Commands

```sh
export TENSORSPINE_MODEL_ARTIFACTS=~/somewhere/tensorspine        # weights/<artifact>/, derived/
python3 tools/tensorspine --derive data/models/llama3-8b.json -o "$TENSORSPINE_MODEL_ARTIFACTS/derived"
D="$TENSORSPINE_MODEL_ARTIFACTS/derived/llama3-8b.derived.json"
CK="$TENSORSPINE_MODEL_ARTIFACTS/weights/Meta-Llama-3-8B"
T=generators/onnx/tsonnx.py

python3 $T info $D                                              # counts, and the refusals over what the delivery evaluates
python3 $T emit $D --checkpoint "$CK" --out model.onnx [--dump]  # the graph of one invocation (external data above 1.5 GB)
python3 $T run  $D --checkpoint "$CK" --ids 128000,791 --steps 8 [--dump ours.safetensors] [--out model.onnx]
python3 $T run  $D --checkpoint "$CK" --ids 128000,791 --target onnxruntime      # the fused forms (com.microsoft)
python3 $T run  $D --checkpoint "$CK" --ids 128000,791 --ids 128000,9906 --batch aligned --capacity 64   # two sessions, one batch
python3 $T run  $D --checkpoint "$CK" --ids 128000,791 --physical phys.json       # e.g. {"decoder/attn[layer=*]": {"backend": "onnxruntime", "rotary_cache": 4096}}
python3 $T run  $D --random --ids 1,2,3                          # parameters drawn from the D3 shapes
python3 $T capabilities [--check]                                # the manifest, from the emitters' tables
python3 generators/onnx/tests/run_onnx.py [--target onnxruntime] # the harness (below)
```

`run` prefills the prompt, decodes greedily, and with `--dump` writes the values at every layer
cut, the states after the prefill and the logits in the reference's dump form, which
`generators/reference/ref.py compare` reads against a fixture.

## Evidence

`tests/run_onnx.py`, run from the repository root: every corpus document derives and reads back;
every unit fixture the manifest admits (`generators/reference/fixtures/contracts/`) is emitted
from the fixture's own document with the fixture as its checkpoint and compared, output by output
and state by state, within the tolerance the fixture states for f32 — this generator as a
**conformer** (Specification §4.2); every integration fixture whose document the manifest can run
and whose checkpoint is on disk is compared at every layer cut, on every state after the prefill
and on the logits, then on the greedy tokens; the manifest regenerates identically and the
language's reader agrees it can run `llama3-8b`. Absent checkpoints are `skip`, never failures.
The harness takes `--target`: under `onnxruntime` every check runs through the fused forms and
passes at the same tolerances (the fused attention and norm sit farther from transformers than the
composed forms, within the fixtures' tolerance).

## Capabilities

`generators/onnx/capabilities.json` (the format is [`generators/CAPABILITIES.md`](../CAPABILITIES.md))
is generated from the primitives' `CAPABILITIES` tables by `tsonnx.py capabilities` and regenerated
by the harness; a conformer's manifest, without a `witness` block. Seven contracts today (`embed`,
`norm.rms`, `residual.add`, `ffn.gated`, `attention.dense` with `append` states and causal or no
mask, `lm_head`, `splice` for the text-only path of a multimodal document); `tensorspine
--capabilities generators/onnx/capabilities.json MODEL…` says which documents it can run. The
manifest is the portable table — what the generator can run is the same under every target — and
a target's fused form is a note on the contract it covers (`target onnxruntime: …`). Compute is f32; `window` and `fixed` states, cross attention, non-token
inputs and a merged domain's positions are not emitted yet and are refused by the manifest, not
guessed.

## Batching

The language describes one session's invocation and leaves batching downstream (harness guide
§8; the batch size is a load variable, out of the model document, Specification §2.1; every
state port is keyed by session through its instance key, §4.4). Like the target, the batch layout
is an argument of the generator, one per graph, reaching every primitive as `ctx.layout`
(batch-plan B02): `none`, the default, one session on the element axis as above; `aligned`, a
batch axis before it. On the aligned layout every session in an invocation delivers the same
count on each stream — equal-length prompts in a prefill, one token each at decode, so sessions
prefilled apart decode together — and no element is padding; the `append` states are buffers of a
capacity per stream (`--capacity`, as the reference's session takes it), `[b, capacity, …]` in and
out, and the positions each session holds on a stream go in as `held/<stream>`, an int64 `[b]`
beside `positions/<stream>`, now `[b, n]`. The portable attention scatters the new keys and values
at each session's held length and masks by position, with the dtype's lowest value rather than
−∞ so a masked row stays finite; the onnxruntime target's GroupQueryAttention takes its native
batched form, the state buffers shared between past and present and `seqlens_k` each session's
length. Linear layers are `MatMul` against the transposed weight on this layout (`Gemm` on the
other).

`tsonnx.py run … --ids A --ids B --batch aligned --capacity N` runs one session per prompt;
`session.Batch(emitter, graph, sessions, capacity)` is the programmatic form (`prefill`, `decode`,
`run`), its outputs per session, element-major, as a `Session`'s. Unequal prompts in one batch are
refused (prefill them apart, or take the reference's packed layout); at most
`SESSIONS_PER_INVOCATION` (16, the manifest's `sessions_per_invocation`) ride together; onnxruntime's
GroupQueryAttention takes a fresh prefill of several sessions or a one-token decode of several, not
a multi-token continuation of several, which the batch refuses under that target (batch-plan
finding 3). Batching is invisible: the harness checks a batch of one and of two against the
sessions alone on the tiny Llama, bit for bit on this box, and on every integration fixture's
checkpoint the prompt and its reverse as one batch against each alone, at the fixture's tolerance.
