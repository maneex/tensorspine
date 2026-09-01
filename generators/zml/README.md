# The ZML generator

The second generator: one Zig type for every document, primitives one file each, the graph traced
into MLIR through [ZML](https://github.com/zml/zml) and compiled by XLA. It consumes a **derived
document** (D1–D6) and a checkpoint, and nothing else — no model document, no catalog, no Python.
Deriving here would be a second implementation of the language.

The reference generator is the oracle: the same fixtures, the same D6 cuts, the same dumped states.

## Building

ZML is Bazel-only (`build.zig` is empty), so **the ZML workspace is the build root** and
these sources are injected into it. ZML must stay the root module: `bzlmod` ignores
`single_version_override` declared anywhere but the root, and ZML's `MODULE.bazel` carries
four of them with patch labels relative to its own root. A `bazel_dep` in the other
direction is not an option either — this directory depends on `@zml//zml`, so ZML
depending on it as a module would be a cycle in the module graph.

`--inject_repository` names the repository on the command line, so **nothing in the ZML
checkout is edited** and no path is written down in this repository. From the ZML
checkout, with `<generator>` the absolute path of this directory:

```sh
./bazel.sh build -c opt --inject_repository=tensorspine=<generator> @tensorspine//:tspl
```

**`-c opt` is not optional for running a model.** Bazel's default is a debug build, where
`init.gpa` is a leak-checking allocator that does not return freed pages; the loader's
per-tensor host staging then accumulates, and 3.18 GiB of weights leave **12.93 GiB**
resident. `tspl` reports its resident set after each phase — locate, compile, load, and
every invocation — so this is visible rather than inferred.

`REPO.bazel` here is the boundary marker Bazel requires; it declares nothing.

To avoid repeating the flag, put it in the ZML checkout's `user.bazelrc` — Bazel expands
`%workspace%` there, its equivalent of `CMAKE_SOURCE_DIR`, so a path relative to the ZML
checkout stays relative:

```
build --inject_repository=tensorspine=%workspace%/../<repo>/generators/zml
```

The test harness needs neither: it computes its own location and passes the flag.

A derived document comes from the language's own tool, run once, offline:

```sh
tensorspine --derive data/models/llama3-8b.json -o "$TENSORSPINE_RUNTIME_DIR"
```

### Where the runtime files live

Derived documents, checkpoints and dumps are **runtime inputs**, not part of this
repository: none of them has a fixed place in the tree, and nothing here names one. The
test harness reads three environment variables, and takes an explicit flag over any of
them:

| Variable | What it holds | Flag |
|---|---|---|
| `ZML_HOME` | the ZML checkout that is the build root | `--zml` |
| `TENSORSPINE_RUNTIME_DIR` | where derived documents and dumps go (a temporary directory by default) | `--runtime-dir` |
| `TENSORSPINE_CHECKPOINT` | the safetensors repository D3's locations name | `--checkpoint` |

Unset, the harness derives into a temporary directory and skips the numerical checks — so
it runs anywhere, and says which checks it did not make.

```sh
generators/zml/tests/run_zml.py                    # structure only
generators/zml/tests/run_zml.py --checkpoint=DIR   # and the numbers
```

### Pinned

| | |
|---|---|
| ZML | upstream `a37f903c` (2 Aug 2026) |
| Zig | 0.16.0, from ZML's toolchain — not from `PATH` |
| Bazel | 9.1.1 via `./bazel.sh` (bazelisk) |

Nothing has to be edited in the ZML checkout, so a `git pull` there cannot break the
wiring. What a ZML update can break is the code: these sources are written against the
commit above, and `zml/`'s API is not frozen.

## What is here

| Path | |
|---|---|
| `main.zig` | `tspl` — the command line |
| `graph.zig` | the derived document as Zig data |
| `primitive-abi.schema.json`, `PRIMITIVE-ABI.md` | the boundary a primitive that is *not* linked in would arrive through: JSON request, MLIR response. Specified; nothing is built behind it. |
| `tensorspine-zml-coverage.md` | a static audit of what ZML's numerical surface already covers, 29 Aug 2026 |

## Reading

- `generators/CAPABILITIES.md` — what a generator's manifest states, and the rules its arguments follow
- `docs/TENSORSPINE-DERIVED_JSON.md` — the document this generator consumes
- `generators/reference/` — the first generator, and the oracle
