# The language core in a Web Worker — what a browser answers

*Feature 1.11. `pnpm spike:lang` serves the page; `apps/web/e2e/worker.spec.ts` drives it.
Measured 11 September 2026 on 8 × Intel i7-4790K, Chromium (Playwright's Chrome for Testing),
Node v22.22.3.*

The unit layer exercises the whole protocol over a `MessageChannel`, which is a true
structured-clone boundary — measured: a `postMessage` between two ports of one channel serializes,
hands the other side a copy, and refuses a symbol with the same `DataCloneError` a thread would.
Three claims are not true of one thread, and this page is where a browser is asked about them.

The page under test imports the application's own `startLang`
(`apps/web/src/lang/connect.ts`), so the worker Vite emits for it is the worker the static build
emits. The material — the schemas, the reference base, two corpus documents — is served from the
repository at `/material.json` rather than from `apps/web/public/vendor/`, which `pnpm vendor`
writes and the CI job does not run (feature 0.2).

## 1 — A derivation does not block the page

§5.3: "It runs in a Web Worker, so validation and derivation never block the UI." Asked with a
`requestAnimationFrame` loop running across the call:

| Document | Derivation | Frames | Rate |
|---|---|---|---|
| `llama3-8b` | 43.2 ms | 3 | 69 fps |
| `deepseek-v4-pro` | 193.6 ms | 12 | 62 fps |

The page kept a full sixty frames a second through a derivation that is two hundred milliseconds
of synchronous work. On the page's own thread the loop would have stopped dead for its whole
length — twelve frames missed, which is what the §5.6 canvas budget is about.

## 2 — A cancel message reaches a worker already at work

Both forms, across the thread:

- **superseded** — two `derive` calls on one document: the first rejects with
  `LangCancelled { reason: 'superseded' }` and its result is dropped; the second answers the six
  products. "One in-flight derivation per document" (§5.3).
- **requested** — a caller's `AbortSignal` aborted in the same tick as the call: the derivation
  never starts.

Both work because the session yields to the task queue before its heavy stage. That yield is the
only moment a synchronous core can be told to stop, and a macrotask is what it must be: a
`message` event is delivered as a task, so a microtask checkpoint would run the whole call before
the cancel could arrive.

## 3 — The engine is in the worker's chunk, not the page's

The built page, with the two entry points of `@tensorspine/lang/api`:

| Chunk | Size | Holds |
|---|---|---|
| `index-*.js` | 138 kB | the proxy, the codec, the JSON layer, Ajv and the schema registry |
| `worker-*.js` | 302 kB | all of the above **plus** the validator, the expansion and the derivation |

The page carries Ajv on purpose: §5.4 runs it **synchronously on the interface's own thread**
("Ajv runs synchronously, < 5 ms on a corpus document"), so the registry belongs there. What it
must never carry is the engine, and it does not — neither `analyse` nor any refusal only a
derivation can raise appears in it. The e2e asserts that, which is what the split of
`@tensorspine/lang/api` (the caller's half) from `@tensorspine/lang/api/engine` (the side that
runs the core) exists for: before the split, importing the engine's entry put the whole session,
host and `createLang` into the page's chunk though nothing on the page used them.

This is feature 0.4's lesson in its third instance (ELK out of the shell's first bundle), and
feature 0.5's in its second (the Hub client out of it).

## 4 — The pipeline of §5.4, end to end

`parse` → `validate` → `expand` → `derive` → `describe` for one site, over the boundary, on a
warm page:

| Document | Whole pipeline | Nodes | Verdict |
|---|---|---|---|
| `llama3-8b` | 426 ms (the first call, which gathers the reference base) | 195 | no problems |
| `deepseek-v4-pro` | 338 ms | 382 | no problems |

The bytes round-trip: `serialize(parse(text))` equals the file, across the boundary and back
(D12). The counters are the tools' own, key for key — `instances=195 | edges=258 | tensors=291 |
state_slots=32` for `llama3-8b`. The per-call budgets are in `editor/spikes/timings.md`.

## What this page is not

It is not a shell. It renders nothing of a document and decides nothing: every answer is the
core's, reached through the `Lang` interface, and the page exists so that a browser can be asked
the three questions above. The application's own entry point (`apps/web/src/main.ts`) is still
feature 0.1's skeleton; feature 2.5 builds the shell, and `startLang` is what it will call.
