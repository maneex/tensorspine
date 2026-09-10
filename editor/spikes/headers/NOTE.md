# Spike: safetensors headers in the browser

Feature 0.5 of `plans/graph-editor-implementation-plan.md`, run 10 September 2026.
**Question:** can a page read a safetensors header — and nothing else — from a local file, and
from the Hub by range request through `@huggingface/hub`, as plan §2 D10 and §4.19 assume?
**Answer: yes, in all three engines**, with one reservation that is not about CORS: the Hub
client's *default* transport is Xet, which needs readable byte streams, and WebKit has none. The
plain range transport (`xet: false`) reads the same header in every engine, for half the bytes.

## What is here

| File | What it is |
|---|---|
| `page/safetensors.ts` | the reader: eight-byte length, JSON header, `name → {dtype, shape, file}` in the language's dtype names — `tools/artifact.py`'s two reads, over anything sliceable |
| `page/synthetic.ts` | a synthetic `.safetensors`, header and payload, built in the page |
| `page/probe.ts` | the six cases, `window.spike.run([…])`, and the URL-driven run that posts its report back |
| `page/index.html` | the page, built by Vite as a static bundle |
| `serve.ts` | builds the page and serves it; Playwright's second `webServer`, and the runner's server |
| `run.ts` | `pnpm spike:headers [--hub]` — opens the page in every engine on the box and writes `engines.json` |
| `engines.json` | what each engine answered, with the request logs: the evidence this note quotes |

The reader is **spike code, not the core**. Feature 1.9 ports `artifact.py` into `packages/lang`
with the dtype table under the audit every semantic table gets (plan §1); what is proven here is
that a browser can do the reading at all, and what it costs. The cases that need no network are
held to account on every commit by `apps/web/e2e/headers.spec.ts` — the browser layer of §0.3 —
which also holds the spike's dtype table to `tools/artifact.py`'s own dictionary.

## What a header read costs

**Local files** (Chromium; a `File` on the page's file input, which is what a picker and a drop
give the editor). `TENSORSPINE_CHECKPOINT=<file-or-directory> pnpm --filter @tensorspine/web
exec playwright test --grep "real checkpoint"`:

| File | On disk | Header | Read | Tensors | ms |
|---|---|---|---|---|---|
| `Meta-Llama-3-8B/model-00001-of-00004` | 4 976 698 672 | 9 512 | **9 520** | 82 | 1.6 |
| `…-00002-of-00004` | 4 999 802 720 | 12 120 | 12 128 | 104 | 6.0 |
| `…-00003-of-00004` | 4 915 916 176 | 11 656 | 11 664 | 100 | 5.9 |
| `…-00004-of-00004` | 1 168 138 808 | 560 | 568 | 5 | 1.5 |
| `Voxtral-Mini-4B-Realtime-2602/model` | 8 859 446 848 | 88 120 | 88 128 | 711 | 7.4 |

**The whole `Meta-Llama-3-8B` checkpoint — 16 060 556 376 bytes, 291 tensors — is 33 880 bytes
and 15 ms of reading.** §4.19's "a 16 GB checkpoint costs a few kilobytes of reading" is right,
and the 291 tensors are the same 291 the Hub answers for the same repository.

**Synthetic files**, the case the plan asks for, in every engine (`engines.json`):

| Case | File | Header | Read | Chromium | Firefox | WebKitGTK |
|---|---|---|---|---|---|---|
| `synthetic-blob` | 1 362 B | 994 B | 1 002 B | 6 ms | 4 ms | 5 ms |
| `synthetic-large` | 268 435 540 B | 76 B | **84 B** | 104 ms → 0.7 ms | 4 ms → 1 ms | 261 ms → 1 ms |
| `opfs-file` | 67 108 945 B | 73 B | 81 B | 1.5 ms (write 198) | 1 ms (write 563) | no OPFS |

The two figures of `synthetic-large` are the first read of a freshly built blob and the second:
**the first read pays for the engine storing a 256 MiB blob, not for reading a header** — a cost
only synthetic material pays, since a `File` on disk costs 1.6 ms on 4.98 GB. The reader asks for
exactly two slices, `[0, 8)` and `[8, 8+n)`, and the case fails if it asks for anything else; a
counting wrapper around the real `Blob` is what proves it, so the bytes counted are the bytes the
browser was asked to produce.

Malformed prefixes are refused rather than allocated — a zero length, a length past the end of
the file, a file shorter than the prefix. `artifact.py` has no such guard because it is handed a
checkpoint directory; a page is handed whatever the user drops on it.

## The Hub, by range request

`@huggingface/hub` 2.16.3, `parseSafetensorsMetadata`, against two pinned revisions. Every
request the client made was recorded by a counting `fetch` (the client takes one), so these are
the requests a page actually causes:

| Case | Repository | Transport | Requests | Bytes | ms |
|---|---|---|---|---|---|
| `hub-range` | `openai-community/gpt2` (548 MB, one file) | `xet: false` — plain `/resolve/` ranges | **4** | **14 860** | 965 |
| `hub-single` | the same | default — Xet | 7 | 34 237 | 1 018 |
| `hub-sharded` | `NousResearch/Meta-Llama-3-8B` (16 GB, 4 shards + index) | default — Xet | 25 | 384 558 | 1 475 |

The plain transport is the reading D10 describes, and it is exactly the reader above with the
slices turned into range requests:

```
GET /openai-community/gpt2/raw/<sha>/model.safetensors                          134 B   (does it exist)
GET /openai-community/gpt2/resolve/<sha>/model.safetensors   Range: bytes=0-0   435 B   (size, and the file info)
GET …/resolve/<sha>/model.safetensors                        Range: bytes=0-7     8 B   (the length prefix)
GET …/resolve/<sha>/model.safetensors                        Range: bytes=8-14290  14 283 B  (the header)
```

Bytes are the responses' `Content-Length`, so a response that carries none counts zero — which is
why Firefox's sharded figure (360 612) is 24 kB under Chromium's: the index came back from
`/api/resolve-cache/` without a length.

**CORS is not an obstacle, and the answer is the server's, not the engine's.** `huggingface.co`
answers the preflight for a `Range` request with `access-control-allow-headers: range`,
`access-control-allow-methods: GET`, `access-control-max-age: 86400` and an
`access-control-expose-headers` list that carries `Accept-Ranges` and `Content-Range`; the 302
goes to `us.aws.cdn.hf.co`, which answers a preflight with `*` for origin, methods and headers
and serves the 206 with `access-control-allow-origin: *`. The Xet endpoints
(`/api/models/…/xet-read-token`, `cas-server.xethub.hf.co`, the xorb CDN) answer cross-origin
too — WebKitGTK reached all of them and failed only when the client built a byte stream.

## The three engines

Measured on this box, each one opening the same built page:

| Engine | Version | `synthetic-*` | OPFS | Hub, plain ranges | Hub, Xet |
|---|---|---|---|---|---|
| Chromium (Blink) | Debian 149.0.7827.53; Playwright 1.63 runs Chrome for Testing 153.0.8010.12 | ok | ok | ok | ok |
| Firefox (Gecko) | 151.0.3 | ok | ok | ok | ok |
| WebKitGTK (WebKit) | 2.50.6, WebKit 605.1.15 (`Version/60.5`) | ok | **absent** | ok | **fails** |

WebKit's failure is one line: `TypeError: ReadableByteStreamController is not implemented`, thrown
where `XetBlob` builds `new ReadableStream({ type: "bytes" })`. The capability probe in each
report says the same thing before any case runs (`readableByteStreams: false`).

**What this says about Safari.** WebKitGTK is the same engine family as Safari — the same WebCore
and JavaScriptCore, a different network backend (libsoup rather than CFNetwork) and a different
port's feature set — so it is evidence about Safari, not a measurement of it: no Safari runs on
Linux, and none was available here. Stated exactly:

- **Measured in WebKit**: `Blob.slice` and `Blob.arrayBuffer` read a header the same way; a
  cross-origin `Range` request to the Hub and to its CDN works; the plain transport reads the
  header of a 548 MB file in 4 requests and 14 860 bytes; readable byte streams are absent, and
  with them the Hub client's Xet transport; the Origin Private File System is absent in this
  build, as is `showDirectoryPicker`.
- **Reasoned, not measured**: readable byte streams are a long-standing gap of the engine rather
  than of this port, so Safari is expected to refuse the Xet transport as well. The conservative
  reading — the one this note recommends — is that the editor never depends on it: the plain
  range transport is proven in every engine, is half the bytes, and is what §4.19 describes.
- **Not this feature's**: the file-picking side of Safari and Firefox (no `showDirectoryPicker`,
  hence D11's read-only snapshot and Save-as-download) is feature 0.6's, which has the
  capability rows of `engines.json` to start from.

## Findings

1. **The Hub client's default transport is Xet.** `parseSafetensorsMetadata` → `downloadFile`
   uses `XetBlob` whenever the Hub offers it, and only `xet: false` gets the plain `/resolve/`
   range reads. It matters twice: WebKit cannot run the Xet path at all, and the Xet path costs
   more for a header — 7 requests and 34 kB against 4 and 15 kB, because each slice re-fetches
   the whole 15 546-byte xorb chunk it falls in, the same chunk twice. Feature 4.1's
   `HubCheckpoint` should ask for the plain transport.
2. **`xet` is forwarded but not declared.** `parseSafetensorsMetadata` passes its parameter
   object on to `downloadFile`, which reads `xet`, but its own parameter type does not name it,
   so a caller needs a cast (`page/probe.ts` carries it with the reason). Worth an upstream issue
   or a pinned wrapper in 4.1 rather than a cast in the application.
3. **`artifact.py`'s `DTYPES` no longer covers the format's vocabulary, and its fallback does not
   land in the schema's enum.** The table maps twelve names; the vocabulary `@huggingface/hub`
   recognises adds `U16`, `U32`, `U64`, `C64`, `F4`, `FP4`, `F6_E2M3`, `F6_E3M2`, `F8_E8M0`,
   `E8M0`, `UE8`, `F8_E4M3FNUZ` and `F8_E5M2FNUZ`, and `read_headers` falls back to
   `name.lower()` — `U16` becomes `u16` and `F8_E4M3FNUZ` becomes `f8_e4m3fnuz`, neither of
   which is a value of the schema's `dtype`; conversely the enum's `u4`, `i4` and `f8e4m3fn`
   have no spelling in the table. V17 compares that name with D3's, so
   an unmapped dtype reads today as a dtype mismatch. Feature 1.9 decides what the core does with
   a dtype the table does not carry (refusal, warning, or pass-through marked as unknown); the
   spike reproduces the tools' behaviour exactly, fallback included, and the e2e test holds the
   table to the tools' dictionary so a divergence shows up as a failure rather than as drift.
4. **A header read costs the header, on every source measured** — 33 880 bytes for the whole
   16 GB `Meta-Llama-3-8B`, 84 bytes on a 256 MiB blob, 14 860 over the wire for a 548 MB Hub
   file. What is *not* free is the first read of a blob the page has just built (104 ms for
   256 MiB in Chromium, 261 in WebKit, 4 in Firefox, then a millisecond or less): the engine has to store it.
   The editor never builds one, so this is a caution for tests, not for the application.
5. **The Hub client is 59 kB of the built page, and it stays out of the first chunk.** The page's
   own chunk is 12.7 kB; `@huggingface/hub` lands in a chunk of its own because the import is
   dynamic — 0.4's lesson about ELK and the first bundle, in its second instance. 4.1 should keep
   the dynamic import.
6. **Where the tests live.** They are in the browser layer (`apps/web/e2e/headers.spec.ts`),
   not in a spike-only Vitest project: `tests/audit/check-command.test.ts` fixes the set of
   projects to the six the plan names, and a browser claim belongs in the browser layer anyway.
   The Hub case is tagged `@network`; it runs on a developer box that can reach the Hub, and is
   skipped in CI on purpose — a suite that reaches a third party on every commit fails for
   reasons of its own.
7. **`TENSORSPINE_CHECKPOINT`** is this spike's name for "a `.safetensors` file, or a directory
   holding some". The repository has no convention for it — `tools/tensorspine --checkpoint` is
   an argument, and `TENSORSPINE_MODEL_ARTIFACTS` names the fixtures, not the checkpoints — so
   the name is proposed here and 1.9 or 4.1 may confirm or change it.

## How to reproduce

```sh
cd editor
pnpm test:e2e                                     # the cases, in Playwright's Chromium
pnpm spike:headers                                # the three engines, no network
pnpm spike:headers --hub                          # …and the Hub; writes engines.json
TENSORSPINE_CHECKPOINT=~/…/Meta-Llama-3-8B \
  pnpm --filter @tensorspine/web exec playwright test --grep "real checkpoint"
TENSORSPINE_HUB=0 pnpm test:e2e                   # what CI runs: no Hub
```

`pnpm spike:headers` needs whatever engines the box has — it names the ones it cannot find and
carries on. On this one: `/usr/bin/chromium`, `/usr/bin/firefox`, and WebKitGTK's `MiniBrowser`
under `xvfb-run`, since it has no headless mode.
