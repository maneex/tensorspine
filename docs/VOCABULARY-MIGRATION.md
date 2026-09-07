# Canonical vocabulary migration

This is a breaking representation revision from baseline commit
`9da0897`. Primitive names, model identities, tensor keys,
numerical fixture payloads and V/I/O rule identifiers retain their meaning.
The complete previous corpus is recoverable from that commit. Identity in
Specification §8.2 includes immutable published files. Every reference primitive
and the `decoder-causal-yarn` template therefore move from `1.0.0` to `2.0.0`:
the interface revision is breaking even though computation is unchanged. The
converter contains the explicit successor map. Previous published files remain
available at the baseline Git revision; they are never relabelled as current units.

## Rename manifest

| Surface | Previous | Current |
|---|---|---|
| Model and derived library references | `catalog` | `primitive_libraries` |
| Library identity, Python module | `catalog` | `primitive_library` |
| Library path, CLI option and document command | `catalog` | `primitive-library` |
| Library collection and directory | `contracts` | `primitives` |
| Primitive citation | `contract` | `primitive` |
| Library unit discriminator | `contract` | `primitive` |
| Declaration collections and selectors | `occurrences`, `occurrence` | `instances`, `instance` |
| State rule selector | `law` | `evolution` |
| State capability collection | `state_laws` | `state_evolutions` |
| Derived state inventory | `by_law` | `by_evolution` |
| Graph boundary | `cut`, `cuts` | `graph_split`, `graph_splits` |
| Semantic decomposition declarations | `partitions` | `partition_options` |
| Boundary predicates | `legal` | `valid` |
| Schema/module/document names | `catalog`, `contract_schema`, `CATALOG` | `primitive-library`, `primitive_schema`, `PRIMITIVE-LIBRARY` |

Python identifiers use underscores, display paths and CLI forms use hyphens.
Primitive Definition describes a reusable specification; Primitive Reference is
its name/version citation; Primitive Instance Site is a declaration before
expansion and Primitive Instance is one concrete use. Node denotes its graph
representation. Model Segment denotes a chosen fragment, not a partition option.

## Format versions

| Format | Previous | Current |
|---|---|---|
| Model Definition | `tensorspine/2.0` | `tensorspine/3.0` |
| Primitive Library unit | `tensorspine-catalog-unit/2.0` | `tensorspine-primitive-library-unit/3.0` |
| Derived Products | `tensorspine-derived/2.1` | `tensorspine-derived/3.0` |
| Capabilities | `tensorspine-capabilities/1` | `tensorspine-capabilities/2` |
| Fixture metadata | `tensorspine-fixture/1` | `tensorspine-fixture/2` |
| Primitive ABI request/response | `tensorspine-primitive-request/1`, `tensorspine-primitive-response/1` | `tensorspine-primitive-request/2`, `tensorspine-primitive-response/2` |

Schema URLs move to the matching revision; shared documentation annotations move
to the 3.0 namespace so current schema references form one closed registry.
Historical Git records, this guide, conversion code/tests and compatibility URLs
are intentional exceptions to the vocabulary audit. Third-party names and
ordinary uses such as array slicing or legal obligations retain their meaning.

## Conversion

Run `python3 tools/migrate.py old.json -o new.json`. Supported legacy revisions
are listed above. Current input is explicitly refused. Duplicate keys, mixed
spellings and unsupported revisions are rejected without writing output.
The converter preserves declared identities and arbitrary argument names.
Use `--library-base` to name the migrated library directory when converting a
model to a new location; relative bases are calculated from the output document.
Safetensors conversion changes only the JSON metadata header; tensor descriptors,
keys and payload bytes remain identical. Convert libraries and referenced
templates together before validating a migrated model.

The baseline and migrated full D1–D6 documents are compared after the explicit
field/version map; no semantic fields are removed to obtain equality. Existing
graph signatures remain the numerical and topology baseline. See the verification
record below for available integration checks and environmental limitations.
