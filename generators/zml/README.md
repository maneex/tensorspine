# The ZML generator

The ZML generator is a language-review instrument and a conformer: its primitive implementations
are checked against the contract reference implementations. It traces a derived graph into MLIR through
[ZML](https://github.com/zml/zml) and compiles it with XLA. It consumes a **derived document**
(D1–D6) and a model artifact, and nothing else—no model document, catalog or Python. Deriving here
would be a second implementation of the language.

The numerical suite uses the shared fixtures, D6 cuts and dumped states. The generated
[status page](https://maneex.github.io/tensorspine/status/) reports current manifest admission,
fixture tolerances and recorded token sequences without copying them into this README.

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
export TENSORSPINE_MODEL_ARTIFACTS=~/somewhere/tensorspine    # your choice; see below
mkdir -p "$TENSORSPINE_MODEL_ARTIFACTS/derived"

tools/tensorspine --derive data/models/llama3-8b.json -o "$TENSORSPINE_MODEL_ARTIFACTS/derived"
```

### 3. Run

```sh
"$ZML_HOME"/bazel-bin/external/+local_repository+tensorspine/tspl \
  --derived="$TENSORSPINE_MODEL_ARTIFACTS/derived/llama3-8b.derived.json" \
  --checkpoint="$TENSORSPINE_MODEL_ARTIFACTS/weights/Meta-Llama-3-8B" \
  --max-tokens=8 --compute=bf16 --split=16
```

`--compute` and `--split` are serving choices used to bound the run's resident set. The prompt
defaults to the reference fixture's identifiers; `--ids="$IDS"` overrides it.

Nothing about that command is llama's. The hybrid is the same invocation against another document:

```sh
"$ZML_HOME"/bazel-bin/external/+local_repository+tensorspine/tspl \
  --derived="$TENSORSPINE_MODEL_ARTIFACTS/derived/qwen3.5-4b-text.derived.json" \
  --checkpoint="$TENSORSPINE_MODEL_ARTIFACTS/weights/Qwen3.5-4B" \
  --ids="$IDS" --max-tokens=8 --compute=bf16 --split=8
```

A full-model CPU run can be long; `tspl` prints `rss after …` lines for monitoring.

### Answering a prompt

```sh
tspl --derived=DOC --checkpoint=CK --prompt='The capital of France is' \
  --max-tokens=64 --compute=bf16 --split=16 --capacity=512
```

The text is tokenised, fed through the graph, and answered on standard output until a
stop identifier or `--max-tokens` tokens, whichever comes first. Nothing is read from the
terminal, which is what a script on a machine reached over `ssh` needs.

`--chat` is the same turn, read from the terminal and repeated:

```sh
tspl --derived=DOC --checkpoint=CK --chat --compute=bf16 --split=16 --capacity=512
```

A turn is tokenised, fed through the graph, and answered until a stop identifier or
`--max-tokens` tokens; the session's states are carried from one turn to the next, so the
conversation *is* the growing state. The tokenizer comes from the artifact's
`tokenizer.json` (`--tokenizer=PATH` overrides it), and the turn ends on whichever of
`<|end_of_text|>`, `<|eot_id|>`, `</s>` or `<|im_end|>` that tokenizer knows.

The tokenizer, the stopping rule and the notion of a turn are all the serving
application's — the language describes a token stream, not what a token is, nor when a
conversation ends. `llama3-8b` is a base model and will happily run to `--max-tokens`; an
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
tspl --derived=DOC --checkpoint=CK --max-tokens=4 --dump=DIR

# the emitted IR, for reading what XLA was given and what it made of it
tspl --derived=DOC --checkpoint=CK --until=VALUE --dump-mlir=DIR

# a unit fixture (docs/TENSORSPINE-FIXTURE.md) as a conformer: the fixture is the checkpoint, the
# inputs of invocation k are read from DIR/in.<k>.<name>.bin, the result and every state written back
tspl --derived=DOC --checkpoint=FIXTURE --unit=DIR --invocations="$INVOCATIONS" --compute=f32
```

### Serving choices

None of these changes the graph, the numbers or the tokens; all of them change what a run holds.
They are arguments because the decisions are the serving application's, not the document's — the
same standing as the reference generator's `--max-ram`.

| Option | What it decides |
|---|---|
| `--split=<n>` | compile and run the graph as *n* programs in sequence. XLA CPU upcasts bf16 matmuls to f32, so one program's scratch holds an f32 copy of every weight its matmuls touch; cutting bounds that copy to the largest group. |
| `--compute=<dtype>` | `f32` (default) or `bf16`. bf16 is the artifact's own precision, as ZML's hand-written models use; f32 is what makes a comparison against the reference a comparison of the mathematics rather than of two roundings. |
| `--capacity=<n>` | positions a growing state holds — deployment intent, not a document fact (§7). |
| `--separate-states` | one buffer per D4 identity instead of one per family. The packed layout is the default; both give identical results. |

`tspl` reports its resident set after every phase — locate, compile, load and each invocation — so
where memory goes is visible rather than inferred. The generator selects one device explicitly to
avoid backend-default replication.

### Build modes

The default (`-c dbg`) gives leak checking and runs the full model. `-c opt` runs faster and takes
longer to build because Bazel rebuilds XLA. The harness takes `--compilation-mode` and defaults to
the debug build.

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
export TENSORSPINE_MODEL_ARTIFACTS=~/somewhere/tensorspine
generators/zml/tests/run_zml.py
```

It builds the target, derives the corpus and checks the following evidence:

| Check | Evidence |
|---|---|
| Derived-document reader | Values read by `tspl` equal the output of `tools/derive.py` |
| Primitive computation | Outputs and states agree with witness-produced unit fixtures |
| Whole-model wiring | Legal-cut values, states and exposed outputs agree with integration fixtures |
| Generative behavior | Greedy tokens agree with the fixture when the document is generative |
| Capability declaration | The regenerated manifest matches the committed one and admits the tested document through TensorSpine's reader |

Documents with different quantities exercise the same primitive paths so a hard-coded fit cannot
pass; another size requires a document and artifact, not model-specific generator code.

## Where the runtime files live

Derived documents, weights and dumps are **runtime inputs**, not part of this repository: none has a
fixed place in the tree, and nothing here names one. **Three shell variables, all naming
directories**, none with a default inside the tree:

| Variable | What it holds | Flag | Unset |
|---|---|---|---|
| `ZML_HOME` | the ZML checkout that is the build root | `--zml` | the harness skips and says so |
| `TENSORSPINE_GENERATOR` | this directory, absolute, for `--inject_repository` | — | the harness computes it |
| `TENSORSPINE_MODEL_ARTIFACTS` | everything a run needs, in one place | `--model-artifacts` | documents go somewhere temporary, numerical checks skipped |

A run needs the derived document **and** the weights it locates tensors in; they are one deployment,
so they are one directory:

```
$TENSORSPINE_MODEL_ARTIFACTS/
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

**Two words, and they are not synonyms** — the language uses both. An **artifact** is the container
the document wraps, format-agnostic: `ARCHITECTURE.md` lists it beside the implementation, the
deployment intent and the hardware, and `SPECIFICATION.md` I9 says *"the described model and loaded
artifact are mutually compatible"*. A **checkpoint** is one concrete safetensors directory — what
`--checkpoint` is pointed at and what V17 is checked against. So the prose here says *artifact*, and
*checkpoint* appears only where a flag or V17 does.

**Weights are big and shared.** Nothing needs moving: point `weights` at wherever they already are.

```sh
export TENSORSPINE_MODEL_ARTIFACTS=~/somewhere/tensorspine
mkdir -p "$TENSORSPINE_MODEL_ARTIFACTS"
ln -s ~/somewhere/huggingface "$TENSORSPINE_MODEL_ARTIFACTS/weights"
```

## Why the build is wired this way

ZML must stay the root module: `bzlmod` ignores `single_version_override` declared anywhere but the
root, and ZML's `MODULE.bazel` carries four of them with patch labels relative to its own root. A
`bazel_dep` in the other direction is not an option either — this directory depends on `@zml//zml`,
so ZML depending on it as a module would be a cycle in the module graph. `--inject_repository`
declares the repository per invocation instead, so a `git pull` in a repository this one does not
own cannot break the wiring. `REPO.bazel` here is the boundary marker Bazel requires; it declares
nothing.

What a ZML update can break is source compatibility because `zml/`'s API is not frozen. ZML's own
toolchain selects Zig and Bazel; the harness is the compatibility check.

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

## Reading

- `generators/CAPABILITIES.md` — what a generator's manifest states, and the rules its arguments follow
- `docs/TENSORSPINE-DERIVED_JSON.md` — the document this generator consumes
- `generators/reference/` — the contract witnesses and integration reference
