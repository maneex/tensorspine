# TensorSpine Editor

A graphical editor for the model definitions of this repository: it opens a `tensorspine/2.0`
document, draws it, edits it through generated forms and property sheets, judges it, derives its
products and writes it back byte-for-byte. It also authors primitive library units. The first
deployment is a static web application — no server, no interpreter, the browser alone.

Two rules shape everything here.

* **The schemas are the source of truth.** An item of information that can be inferred from
  `schemas/` is never hard-coded in the interface: every form is generated from a schema, every
  enumeration is read from one, and what a schema cannot say (that an instance is drawn as a
  node, that bytes are shown in MiB) lives in one presentation file bound to schema anchors.
* **One implementation of every rule, in `packages/lang`.** The language core is a TypeScript
  port of `tools/`, held to parity with it by the oracle below. The interface asks the core and
  displays the answer; it implements no rule of the specification.

## Layout

```
editor/
├── packages/lang     the language core: lexeme-preserving JSON, the schema registry,
│                     expressions, the library loader, validation, expansion, derivation,
│                     the checkpoint check, lint, describe and check. No DOM.
├── packages/store    the document store: the ordered tree, commands and undo, the sidecars
├── packages/ui       React: shell, activities, canvas, generated forms, sheets, panels
├── apps/web          the static application built by Vite
└── tests/            the audits (tests/audit) and the parity oracle (tests/oracle)
```

## Install

Node 22 or later and pnpm 12. The repository pins the package manager, so
[corepack](https://nodejs.org/api/corepack.html) — `corepack enable` — is enough to get the
right pnpm; otherwise install pnpm 12 yourself.

```sh
cd editor
pnpm install
pnpm exec playwright install chromium     # once, for the end-to-end layer
```

## Run

```sh
pnpm dev        # the application on Vite's dev server
pnpm build      # the static build, into apps/web/dist
```

## Test

`pnpm check` runs every layer, in this order, and is what CI runs:

| Layer | Command | What it holds |
|---|---|---|
| typecheck | `pnpm typecheck` | every package under strict TypeScript |
| lint | `pnpm lint` | ESLint over the workspace, type-aware |
| unit | `pnpm test:unit` | `packages/*/test/**` |
| parity | `pnpm test:parity` | `packages/lang/test/parity/**`, against the oracle |
| snapshot | `pnpm test:snapshot` | `packages/ui/test/snapshots/**` |
| audit | `pnpm audit:rules` | `tests/audit/**`: the standing rules of the workspace |
| end-to-end | `pnpm test:e2e` | Playwright over the built application, headless Chromium |

`pnpm test` runs the unit and snapshot layers together, which is the loop to keep open while
working. No test is ever disabled to make the suite pass.

## The oracle

The parity layer compares the core with the tools that are the authority — `tools/` — over the
repository's own material. `pnpm oracle` produces that material:

```sh
pnpm oracle     # needs python3 3.11 and jsonschema==4.25.0, both only for this
```

It runs `tools/tensorspine` on every model of `data/models/` and on the template under the
assignment `tests/signature.py` records, and writes `tests/oracle/out/`: the `--d1` and
`--derive` documents, the `--validate` and `--lint` output, the per-primitive argument schemas
of `--document primitive-schema`, the rejection suite's expectations and the signature suite's
signatures, and a `manifest.json` naming all of it with the digests of `tools/` and `schemas/`.

`tests/oracle/out/` is gitignored: it is regenerated, never committed. The generator refuses to
run without the tools rather than fall back on stale output.

Python runs here and in CI, and nowhere else: the editor itself never starts an interpreter, and
an audit test holds it to that.

## Continuous integration

`.github/workflows/editor.yml` defines the `editor` job: Node 22, pnpm from the pinned
`packageManager`, Python 3.11 with `jsonschema==4.25.0` for the oracle, Chromium for Playwright,
then `pnpm oracle` and `pnpm check`.

## The toolchain

TypeScript 6, Vite 8, Vitest 5, ESLint 10 with typescript-eslint 8, Playwright 1.63. TypeScript
is held at 6 on purpose: typescript-eslint 8 refuses TypeScript 7, so the workspace stays on the
line the linter supports until that changes.
