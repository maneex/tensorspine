# ONNX generator

The third generator: it **emits** an ONNX graph from a derived document and a checkpoint, and
runs it through onnxruntime. Where the reference executes each contract's kernel and ZML traces
the graph into MLIR, this generator writes the graph down in the interchange form serving
engines read. It knows nothing about any model: the derived document (D1–D6, `tensorspine
--derive`) is all it reads, never the model source or the catalog.

## What is emitted

One ONNX graph is **one invocation** (Specification §7), for one set of delivered inputs:

- every public input delivered is a graph input with a dynamic element axis (`tokens` as int64
  identifiers; a floating input in the compute dtype);
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

Opset 17, IR 8, no contrib operators: RMS norm, rotary and attention are composed from the
standard operators. The session emits one graph per distinct delivery (a prefill with the audio
and a decode without it would be two) and caches the onnxruntime session for each.

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
python3 $T run  $D --random --ids 1,2,3                          # parameters drawn from the D3 shapes
python3 $T capabilities [--check]                                # the manifest, from the emitters' tables
python3 generators/onnx/tests/run_onnx.py                        # the harness (below)
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

## Capabilities

`generators/onnx/capabilities.json` (the format is [`generators/CAPABILITIES.md`](../CAPABILITIES.md))
is generated from the primitives' `CAPABILITIES` tables by `tsonnx.py capabilities` and regenerated
by the harness; a conformer's manifest, without a `witness` block. Six contracts today (`embed`,
`norm.rms`, `residual.add`, `ffn.gated`, `attention.dense` with `append` states and causal or no
mask, `lm_head`); `tensorspine --capabilities generators/onnx/capabilities.json MODEL…` says which
documents it can run. Compute is f32; `window` and `fixed` states, cross attention, non-token
inputs and a merged domain's positions are not emitted yet and are refused by the manifest, not
guessed.
