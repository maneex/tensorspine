# The ZML generator

The second generator: one Zig type for every document, primitives one file each, the graph traced
into MLIR through [ZML](https://github.com/zml/zml) and compiled by XLA. It consumes a **derived
document** (D1–D6) and a checkpoint, and nothing else — no model document, no catalog, no Python.
Deriving here would be a second implementation of the language.

The reference generator is the oracle: the same fixtures, the same D6 cuts, the same dumped states.

> **`llama3-8b` runs whole.** All 32 layers, from the derived document and the checkpoint alone:
> `[12366, 13, 1102, 374, 7559, 304, 279, 10411]` — the reference generator's own recorded tokens,
> and `transformers`'. 195 occurrences as 16 programs; 370 MiB to compile, 15.28 GiB resident after
> loading 14.96 GiB of weights, 18.3 GiB peak.

## Running a model

Three steps; the first two happen once.

### 1. Build

ZML is Bazel-only (`build.zig` is empty), so **the ZML workspace is the build root** and these
sources are injected into it — nothing in the ZML checkout is edited, and no path outside this
repository is written down in it. From the ZML checkout, with `<generator>` the absolute path of
this directory:

```sh
./bazel.sh build --inject_repository=tensorspine=<generator> @tensorspine//:tspl
```

The binary lands at `bazel-bin/external/+local_repository+tensorspine/tspl`.

### 2. Derive the document

Offline, once per model, through the language's own tool — the generator never derives:

```sh
export TENSORSPINE_RUNTIME_DIR=~/somewhere/tensorspine-runtime    # your choice; see below
mkdir -p "$TENSORSPINE_RUNTIME_DIR"

tools/tensorspine --derive data/models/llama3-8b.json -o "$TENSORSPINE_RUNTIME_DIR"
```

### 3. Run

```sh
<zml>/bazel-bin/external/+local_repository+tensorspine/tspl \
  --derived="$TENSORSPINE_RUNTIME_DIR/llama3-8b.derived.json" \
  --checkpoint="$TENSORSPINE_CHECKPOINT" \
  --steps=8 --compute=bf16 --split=16
```

That is the invocation that produced the tokens above. **`--compute=bf16 --split=16` are what make
it fit** on a 31 GiB machine: without them the same run wants about 40 GiB. The prompt defaults to
the reference fixture's identifiers; `--ids=128000,791,…` overrides it.

A full-model run is slow on CPU — 32 layers through 16 programs for every token — so it is worth
`nohup`-ing and watching the `rss after …` lines it prints.

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
build --inject_repository=tensorspine=%workspace%/../<repo>/generators/zml
```

The test harness needs neither: it computes its own location and passes the flag.

## Testing

```sh
export ZML_HOME=<the zml checkout>
export TENSORSPINE_CHECKPOINT=<the safetensors repository>
generators/zml/tests/run_zml.py
```

It builds the target itself, derives all 14 corpus documents, checks that what `tspl` reads out of
each equals what `tools/derive.py` put in, then compares `llama3-8b`'s embedding, its first norm,
its first three layer cuts and all six KV components against the reference's fixture — 25 checks.

## Where the runtime files live

Derived documents, checkpoints and dumps are **runtime inputs**, not part of this repository: none
of them has a fixed place in the tree, and nothing here names one. Three environment variables name
them, each overridden by a flag, and **none has a default inside the tree**:

| Variable | What it holds | Flag | Unset |
|---|---|---|---|
| `ZML_HOME` | the ZML checkout that is the build root | `--zml` | the harness skips and says so |
| `TENSORSPINE_RUNTIME_DIR` | where derived documents and dumps go | `--runtime-dir` | a temporary directory, deleted after |
| `TENSORSPINE_CHECKPOINT` | the safetensors repository D3's locations name | `--checkpoint` | the numerical checks are skipped |

So `$TENSORSPINE_RUNTIME_DIR` is wherever you point it — it is a convention, not a location this
repository owns. Set it to keep derived documents between runs; leave it unset to let the harness
work in a temporary directory and clean up.

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
