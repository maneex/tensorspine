# The ZML generator

The second generator: one Zig type for every document, primitives one file each, the graph traced
into MLIR through [ZML](https://github.com/zml/zml) and compiled by XLA. It consumes a **derived
document** (D1–D6) and a checkpoint, and nothing else — no model document, no catalog, no Python.
Deriving here would be a second implementation of the language.

The reference generator is the oracle: the same fixtures, the same D6 cuts, the same dumped states.

## Building

ZML is Bazel-only (`build.zig` is empty), so **the ZML workspace is the build root** and reaches
these sources through a `local_repository`. ZML must stay the root module: `bzlmod` ignores
`single_version_override` declared anywhere but the root, and ZML's `MODULE.bazel` carries four of
them with patch labels relative to its own root. A `bazel_dep` in the other direction is not an
option either — this directory depends on `@zml//zml`, so ZML depending on it as a module would be a
cycle in the module graph.

Append to the `MODULE.bazel` of your ZML checkout, with `path` pointing at this directory:

```python
tensorspine_repo = use_repo_rule("@bazel_tools//tools/build_defs/repo:local.bzl", "local_repository")

tensorspine_repo(
    name = "tensorspine",
    path = "../armature/master/generators/zml",
)
```

`REPO.bazel` in this directory is the boundary marker Bazel requires; it declares nothing.

Then, from the ZML checkout:

```sh
./bazel.sh build @tensorspine//:tspl
./bazel.sh run   @tensorspine//:tspl -- --derived=/path/to/llama3-8b.derived.json
```

A derived document comes from the language's own tool, run once, offline:

```sh
tensorspine --derive data/models/llama3-8b.json -o /some/dir/
```

### Pinned

| | |
|---|---|
| ZML | upstream `a37f903c` (2 Aug 2026) |
| Zig | 0.16.0, from ZML's toolchain — not from `PATH` |
| Bazel | 9.1.1 via `./bazel.sh` (bazelisk) |

Those two `MODULE.bazel` lines live in a repository this one does not own, so a `git pull` there can
drop them. That is the intended failure — Bazel says the module is missing rather than doing
something subtle — and re-applying them is the fix.

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
