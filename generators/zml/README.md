# The ZML generator

The second generator: one Zig type for every document, primitives one file each, the graph traced
into MLIR through [ZML](https://github.com/zml/zml) and compiled by XLA. It consumes a **derived
document** (D1–D6) and a checkpoint, and nothing else — no model document, no catalog, no Python.
Deriving here would be a second implementation of the language.

The reference generator is the oracle: the same fixtures, the same D6 cuts, the same dumped states.

> **Three documents run whole**, and the manifest says four of the corpus's fourteen can.
>
> **`qwen3.5-4b-text` runs whole** — a hybrid: 24 gated-delta layers and 8 attention layers, whose
> states are `fixed` and `window` beside the KV cache. `[11751, 13, 198, 32, 13, 2912, 198, 33]`,
> again the reference generator's own recorded tokens. Its **56 states become four buffers**, and
> the whole run holds 8.58 GiB against 7.83 GiB of weights — a document built of these has almost
> nothing to page. Its four layer cuts and every state of the reference's fixture agree within
> **2.53e-06**.
>
> `colbert-v2` — a retrieval encoder with no generative output and no state at all — matches the
> reference generator's fixture within **1.21e-06** on its embeddings, every row L2-normalised to
> one, on 0.41 GiB of weights.
>
> **`llama3-8b` runs whole.** All 32 layers, from the derived document and the checkpoint alone:
> `[12366, 13, 1102, 374, 7559, 304, 279, 10411]` — the reference generator's own recorded tokens,
> and `transformers`'. 195 occurrences as 16 programs; 370 MiB to compile, 15.28 GiB resident after
> loading 14.96 GiB of weights, 18.3 GiB peak.

## Running a model

Three steps; the first two happen once.

### 1. Build

ZML is Bazel-only (`build.zig` is empty), so **the ZML workspace is the build root** and these
sources are injected into it — nothing in the ZML checkout is edited, and no path outside this
repository is written down in it. Two directories are involved, so both are shell variables rather
than prose: `$ZML_HOME` is the ZML checkout, `$TENSORSPINE_GENERATOR` is *this* directory, absolute
because Bazel resolves it from a build root that is somewhere else.

```sh
export ZML_HOME=~/somewhere/zml                      # the ZML checkout
export TENSORSPINE_GENERATOR=$PWD/generators/zml     # from this repository's root

cd "$ZML_HOME" && ./bazel.sh build \
  --inject_repository=tensorspine="$TENSORSPINE_GENERATOR" @tensorspine//:tspl
```

The binary lands at `$ZML_HOME/bazel-bin/external/+local_repository+tensorspine/tspl`.

### 2. Derive the document

Offline, once per model, through the language's own tool — the generator never derives:

```sh
export TENSORSPINE_ARTIFACTS=~/somewhere/tensorspine    # your choice; see below
mkdir -p "$TENSORSPINE_ARTIFACTS/derived"

tools/tensorspine --derive data/models/llama3-8b.json -o "$TENSORSPINE_ARTIFACTS/derived"
```

### 3. Run

```sh
"$ZML_HOME"/bazel-bin/external/+local_repository+tensorspine/tspl \
  --derived="$TENSORSPINE_ARTIFACTS/derived/llama3-8b.derived.json" \
  --checkpoint="$TENSORSPINE_ARTIFACTS/weights/Meta-Llama-3-8B" \
  --steps=8 --compute=bf16 --split=16
```

That is the invocation that produced the tokens above. **`--compute=bf16 --split=16` are what make
it fit** on a 31 GiB machine: without them the same run wants about 40 GiB. The prompt defaults to
the reference fixture's identifiers; `--ids=128000,791,…` overrides it.

Nothing about that command is llama's. The hybrid is the same invocation against another document:

```sh
"$ZML_HOME"/bazel-bin/external/+local_repository+tensorspine/tspl \
  --derived="$TENSORSPINE_ARTIFACTS/derived/qwen3.5-4b-text.derived.json" \
  --checkpoint="$TENSORSPINE_ARTIFACTS/weights/Qwen3.5-4B" \
  --ids=760,6511,314,9338,369 --steps=8 --compute=bf16 --split=8
```

A full-model run is slow on CPU — 32 layers through 16 programs for every token — so it is worth
`nohup`-ing and watching the `rss after …` lines it prints.

### Chatting

```sh
tspl --derived=DOC --checkpoint=CK --chat --compute=bf16 --split=16 --capacity=512
```

A turn is tokenised, fed through the graph, and answered until a stop identifier or
`--steps` tokens; the session's states are carried from one turn to the next, so the
conversation *is* the growing state. The tokenizer comes from the checkpoint's
`tokenizer.json` (`--tokenizer=PATH` overrides it), and the turn ends on whichever of
`<|end_of_text|>`, `<|eot_id|>`, `</s>` or `<|im_end|>` that tokenizer knows.

The tokenizer, the stopping rule and the notion of a turn are all the serving
application's — the language describes a token stream, not what a token is, nor when a
conversation ends. `llama3-8b` is a base model and will happily run to `--steps`; an
instruction-tuned mirror stops itself.

Only one arity is compiled, for a single element, and the prompt is fed through it a
token at a time. A compiled graph has static shapes, so a prefill of *n* tokens would be
a different program for every *n* a conversation happens to produce; feeding one at a
time costs the same forward passes and compiles once. A serving application that cared
would compile a few prefill widths and pad to them.

### Other things `tspl` does

```sh
# which contracts this generator has no primitive for, per document
tspl --derived=DOC --refusals

# evaluate one D2 value and write its bytes: the ancestor closure only, loading only the
# tensors that closure needs — how to bisect a numerical disagreement
tspl --derived=DOC --checkpoint=CK --until='decoder/ffn_r[layer=0].output' --out=v.bin

# every state as well, one file per D4 identity whatever layout held it
tspl --derived=DOC --checkpoint=CK --steps=4 --dump=DIR

# the emitted IR, for reading what XLA was given and what it made of it
tspl --derived=DOC --checkpoint=CK --until=VALUE --dump-mlir=DIR
```

### Serving choices

None of these changes the graph, the numbers or the tokens; all of them change what a run holds.
They are arguments because the decisions are the serving application's, not the document's — the
same standing as the reference generator's `--max-ram`.

| Option | What it decides |
|---|---|
| `--split=<n>` | compile and run the graph as *n* programs in sequence. XLA CPU upcasts bf16 matmuls to f32, so one program's scratch holds an f32 copy of every weight its matmuls touch; cutting bounds that to the largest group. Measured on 8 layers: 14.3 GiB at 1, 9.3 at 4, 8.5 at 8 — same tokens throughout. |
| `--compute=<dtype>` | `f32` (default) or `bf16`. bf16 is the checkpoint's own precision, as ZML's hand-written models use; f32 is what makes a comparison against the reference a comparison of the mathematics rather than of two roundings. |
| `--capacity=<n>` | positions a growing state holds — deployment intent, not a document fact (§7). |
| `--separate-states` | one buffer per D4 identity instead of one per family. The packed layout is the default; both give identical results. |

`tspl` reports its resident set after every phase — locate, compile, load, and each invocation — so
where memory goes is visible rather than inferred. That report is what found ZML's CPU default of
**four devices**, which puts a copy of every replicated parameter on each: 3.18 GiB of weights held
12.93 GiB until the generator asked for one device.

### Build modes

The default (`-c dbg`) is what every measurement here used and what runs the full model; it also
gives leak checking, which is worth having. `-c opt` runs faster and takes far longer to get —
Bazel rebuilds XLA from scratch, the better part of an hour. The harness takes `--compilation-mode`
and defaults to the debug build, so a first run is not an hour of XLA.

To avoid repeating the inject flag, put it in the ZML checkout's `user.bazelrc` — Bazel expands
`%workspace%` there, its equivalent of `CMAKE_SOURCE_DIR`, so a path relative to the ZML checkout
stays relative:

```
build --inject_repository=tensorspine=%workspace%/../<this-repository>/generators/zml
```

The test harness needs neither: it computes its own location and passes the flag.

## Testing

```sh
export ZML_HOME=~/somewhere/zml                   # the ZML checkout
export TENSORSPINE_ARTIFACTS=~/somewhere/tensorspine
generators/zml/tests/run_zml.py
```

It builds the target itself, derives all 14 corpus documents, and checks that what `tspl` reads out
of each equals what `tools/derive.py` put in. Then, for whichever checkpoints it is given — all
against the reference generator's committed fixtures:

| Document | Checked |
|---|---|
| `llama3-8b` | the embedding, the first norm, three layer cuts, all six KV components |
| `colbert-v2` | two layer cuts and the embeddings |
| `qwen3.5-4b-text` | four layer cuts, and **every state**: three convolution histories, three recurrent matrices, one KV cache |
| `qwen3.8-27b-text` | the same, at other quantities — 48 value heads against 32, a 10240-wide convolution against 8192 |

The two hybrids are checked because one document could have been fitted and two at different
sizes cannot: the second needed no code at all, only its checkpoint.

Last, it regenerates the capabilities manifest and diffs it, and asks the language's own reader
whether the manifest can run the document.

## Where the runtime files live

Derived documents, weights and dumps are **runtime inputs**, not part of this repository: none has a
fixed place in the tree, and nothing here names one. **Three shell variables, all naming
directories**, none with a default inside the tree:

| Variable | What it holds | Flag | Unset |
|---|---|---|---|
| `ZML_HOME` | the ZML checkout that is the build root | `--zml` | the harness skips and says so |
| `TENSORSPINE_GENERATOR` | this directory, absolute, for `--inject_repository` | — | the harness computes it |
| `TENSORSPINE_ARTIFACTS` | everything a run needs, in one place | `--artifacts` | documents go somewhere temporary, numerical checks skipped |

A run needs the derived document **and** the weights it locates tensors in; they are one deployment,
so they are one directory:

```
$TENSORSPINE_ARTIFACTS/
  derived/                 llama3-8b.derived.json, qwen3.5-4b-text.derived.json, …
  weights/                 one directory per artifact — a symlink when they live elsewhere
    Meta-Llama-3-8B/       what `--checkpoint` is pointed at
    Qwen3.5-4B/
  dumps/<model>/           what a run left behind, named by D4 identity
```

`GLOSSARY.md` calls one of those weight directories *"the artifact the document wraps"*, and
`tools/artifact.py` is the language's own reader for them — the variable is that word's plural, and
holds the documents beside them. Which subdirectory belongs to which document is a table in the
harness, not a guess: weights that are absent make their checks say so rather than search.

**Weights are big and shared.** Nothing needs moving: point `weights` at wherever they already are.

```sh
export TENSORSPINE_ARTIFACTS=~/somewhere/tensorspine
mkdir -p "$TENSORSPINE_ARTIFACTS"
ln -s ~/somewhere/huggingface "$TENSORSPINE_ARTIFACTS/weights"
```

## Why the build is wired this way

ZML must stay the root module: `bzlmod` ignores `single_version_override` declared anywhere but the
root, and ZML's `MODULE.bazel` carries four of them with patch labels relative to its own root. A
`bazel_dep` in the other direction is not an option either — this directory depends on `@zml//zml`,
so ZML depending on it as a module would be a cycle in the module graph. `--inject_repository`
declares the repository per invocation instead, so a `git pull` in a repository this one does not
own cannot break the wiring. `REPO.bazel` here is the boundary marker Bazel requires; it declares
nothing.

What a ZML update *can* break is the code: these sources are written against the commit below, and
`zml/`'s API is not frozen.

| | |
|---|---|
| ZML | upstream `6e171a5` |
| Zig | 0.16.0, from ZML's toolchain — not from `PATH` |
| Bazel | 9.1.1 via `./bazel.sh` (bazelisk) |

## What is here

| Path | |
|---|---|
| `main.zig` | `tspl` — the command line |
| `chat.zig` | a conversation: tokenizer, turns, stopping |
| `session.zig` | the compiled arities, and one invocation through them |
| `graph.zig` | the derived document as Zig data |
| `plan.zig` | what to evaluate, in what order, cut into programs |
| `loader.zig` | parameters, from D3's locations |
| `state.zig` | D4's laws, once; and the layout the serving application chose |
| `emit.zig` | the plan walked once, emitting MLIR |
| `primitive.zig`, `registry.zig`, `primitives/` | one file per contract version |
| `primitive-abi.schema.json`, `PRIMITIVE-ABI.md` | the boundary a primitive that is *not* linked in would arrive through: JSON request, MLIR response. Specified; nothing is built behind it. |
| `tensorspine-zml-coverage.md` | a static audit of what ZML's numerical surface already covers, 29 Aug 2026 |

## Reading

- `generators/CAPABILITIES.md` — what a generator's manifest states, and the rules its arguments follow
- `docs/TENSORSPINE-DERIVED_JSON.md` — the document this generator consumes
- `generators/reference/` — the first generator, and the oracle
