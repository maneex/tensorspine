# TensorSpine Editor

A graphical editor for the model definitions of this repository: it opens a `tensorspine/2.0`
document, draws it, edits it through generated forms and property sheets, judges it, derives its
products and writes it back byte-for-byte. It also authors primitive library units. The first
deployment is a static web application — no server, no interpreter, the browser alone.

Two rules shape everything here.

* **The schemas are the source of truth.** An item of information that can be inferred from
  `schemas/` is never hard-coded in the interface: every form is generated from a schema, every
  enumeration is read from one, and what a schema cannot say (that an instance is drawn as a
  node, that bytes are shown in MiB) lives in one presentation file bound to schema anchors —
  `packages/ui/src/presentation.json`, resolved against the loaded schemas at startup.
* **One implementation of every rule, in `packages/lang`.** The language core is a TypeScript
  port of `tools/`, held to parity with it by the oracle below. The interface asks the core and
  displays the answer; it implements no rule of the specification.

## Layout

```
editor/
├── packages/lang     the language core: lexeme-preserving JSON, the schema registry,
│                     expressions, the library loader, validation, expansion, derivation,
│                     the checkpoint check, lint, describe and check. No DOM.
├── packages/store    the document store: the ordered tree, commands and undo, the sidecars,
│                     and the platform interfaces (@tensorspine/store/platform) with their stub
├── packages/ui       React: shell, activities, canvas, generated forms, sheets, panels.
│                     Three entry points: the package root (the walker, the forms and the
│                     presentation bindings — no React), `/shell` (the chrome) and
│                     `/layout` (ELK), plus `/style.css`
├── apps/web          the static application built by Vite, and the browser's platform:
│                     the workspace, settings, drafts, the shell
├── schemas/          the editor's own schemas — the layout sidecar, the presentation
│                     bindings — with their companion notes
└── tests/            the audits (tests/audit) and the parity oracle (tests/oracle)
```

## The platform

Everything the editor cannot compute for itself — the files it reads and writes, the settings it
remembers, the drafts it autosaves, who the user is, and the few things only a shell can do —
goes through one interface, `Platform`, declared in `@tensorspine/store/platform` and implemented
per deployment. No package reaches around it: an audit holds every `packages/*/src` to that, and
CI builds the application against a stub platform so that a leak is a build failure.

| What | The static application's answer |
|---|---|
| a workspace the editor can write | a folder opened through the File System Access API and written in place, its handle kept in IndexedDB so it reopens with one permission prompt (Chromium today) |
| a workspace it cannot | a folder upload or a drop, read-only, where Save hands the document to the user as a download and the folder is left alone — what Firefox and Safari get |
| the **Examples** workspace | the corpus and the reference base vendored with the build, read-only, always available |
| settings | `localStorage`, loaded once and written through |
| drafts | IndexedDB |
| sign-in | none: the static application has no accounts, and everything it computes runs in the browser |

Each of them keeps working where the browser refuses to store anything at all — a private window,
site data blocked — for the session alone, and says so.

## The shell

One window laid out as VS Code lays one out: a bar with seven menus, an activity rail, a side
bar, the editor's tabs, a bottom panel, the Properties region and a status bar. Every region
remembers its size; every panel is a tab that can be dragged to the other region; the
arrangement lives in the settings.

* **Every command is registered once** and is reachable from its menu, from the command palette
  (Ctrl/⌘ + Shift + P) and, later, from a context menu. A command no feature has wired yet is
  still there and says so in the Log when it is chosen — hiding it would hide the plan from the
  person reading the menu. A shortcut is an accelerator and never the only way to reach
  anything; Help ▸ Keyboard Shortcuts lists all of them.
* **Two themes, and the machine decides between them by default.** The tokens are the design
  pass's own (`packages/ui/src/shell/tokens.css`, vendored from `plans/graph-editor-design/`
  and held to it); `View ▸ Theme: Light / Dark / System` pins one, and the choice is remembered.
* **The lockup is the repository's.** The monogram of `docs/tensorspine.svg` as
  `docs/style/nav.html` arranges it, vendored by `pnpm logo` and held equal to its source by an
  audit: placed and scaled, never redrawn.
* **Every string the interface shows is in one dictionary**, keyed by the English sentence
  itself, so a translation is a map and not a rewrite.

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
pnpm vendor     # the schemas, corpus, reference base and generated artifacts, for the application
pnpm logo       # re-vendor the wordmark's monogram from docs/style/nav.html (rarely needed)
pnpm dev        # the application on Vite's dev server
pnpm build      # the static build, into apps/web/dist (vendors first)
```

## What the application ships

There is no server and no interpreter behind the editor, so everything it reads is copied into
`apps/web/public/vendor/` at build time, from the repository it is part of:

| Under the vendor root | What it is |
|---|---|
| `schemas/` | the repository's `schemas/`, byte for byte, indexed at startup by `$id` |
| `data/models/` | the corpus — the read-only **Examples** workspace |
| `data/primitive-library/` | the reference base |
| `generated/primitive-schema/` | one JSON Schema per non-template primitive version, as `--document primitive-schema` writes them |
| `generated/primitive-library.md` | the library reference, as `--document primitive-library` writes it |
| `vendor.json` | the manifest: the commit vendored, and every file with its length and its sha256 |

`data/` keeps the repository's layout on purpose: a document resolves its bases relative to
itself (`"base": "../primitive-library/"`) and a base manifest resolves its templates the same
way, so the Examples workspace opens exactly as `data/` does.

The two generated artifacts are the tools' own output, consumed and never regenerated in
JavaScript, so `pnpm vendor` needs python3 and `jsonschema` as the oracle does. It refuses
without them rather than ship a stale copy: what it needs is checked before anything is removed,
a run that cannot finish leaves no half vendor behind, and the manifest is written last — a
directory without `vendor.json` is not a vendor. `apps/web/public/vendor/` is gitignored: it is
regenerated, never committed.

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

The audit layer also runs `pnpm vendor` into a temporary directory and holds what it wrote to
the repository's own files, so a broken vendor fails `pnpm check` and CI needs no step of its
own for it. The end-to-end layer vendors for real before it builds, because the static
application *is* its vendored schemas, corpus and reference base: an application built without
them opens no Examples workspace and reads no schema, and a suite against it would be testing
something nobody deploys. That build also emits two pages the deployment does not have —
`stub.html`, the renderer against the stub platform, and the browser layer's own driver — and an
audit holds `pnpm build` to emitting neither.

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

The build scripts under `scripts/` are TypeScript that Node runs itself, through type stripping
(`--experimental-strip-types`, passed so that the Node 22 lines that still need the flag run
them too); they are typechecked and linted with everything else.
