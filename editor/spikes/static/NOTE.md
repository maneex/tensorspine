# Spike: the static build and the browser workspace

Feature 0.6 of `plans/graph-editor-implementation-plan.md`, run 10 September 2026.
**Question:** can the editor be a static page served under a base path — no server at all — that
opens a local folder, reads and writes it in place, remembers it across a reload, and falls back
to an honest read-only snapshot where the browser has no writable picker (plan §2 D11, §4.3,
§5.2)?
**Answer: yes**, with one boundary that is the browser's and not the design's: **only Chromium
has the writable picker**. Firefox has every other piece — directory handles, `createWritable`,
handle persistence in IndexedDB — and no way to *obtain* a handle to the user's folder; WebKit
has none of it. So the snapshot path is not a courtesy for old browsers: it is what two of the
three engines get, and it is measured here alongside the writable one.

## What is here

| File | What it is |
|---|---|
| `page/workspace.ts` | the `Workspace` of plan §5.2 as a page needs it: paths, entries, revisions, refusals, and the relative resolution `primitive_libraries[].base` uses |
| `page/directory.ts` | the writable path: `FileSystemDirectoryHandle` — list, read, write in place with optimistic concurrency, `mkdir`, `watch` by polling |
| `page/snapshot.ts` | the read-only path: a folder upload or a drop, where `write` **downloads** the file and the revision does not move |
| `page/recents.ts` | the handle in IndexedDB, and the permission model (`queryPermission` / `requestPermission`) where the engine has one |
| `page/shell.ts` | the shell of artboards S17 and S18: Open Folder, the snapshot banner, the file list, Save, the recent workspaces, the status bar |
| `page/probe.ts` | the seven cases, `window.spike.run([…])`, and the URL-driven run that posts its report back |
| `serve.ts` | builds the page **under a base path** and serves it under that path and nowhere else |
| `run.ts` | `pnpm spike:static` — builds under three bases, opens the page in every engine on the box, writes `engines.json` |
| `engines.json` | what each engine answered: the evidence this note quotes |

The workspace here is **spike code, not `packages/store`**. Feature 2.4 owns `Platform` and will
inherit this shape and these findings; what is proven here is that the shape can be implemented
in a browser at all, and what it costs. The cases that need no unautomated engine are held to
account on every commit by `apps/web/e2e/static.spec.ts` — the browser layer of §0.3.

## The static build under a base path

The page is built by Vite with `base` as a parameter and served by a server that answers
**only** under that base: a build whose assets were named from the origin's root gets a 404 here
rather than passing by accident. Three builds, from `run.ts`:

| Base | What `index.html` names | JS | HTML |
|---|---|---|---|
| `/editor/` — what the feature asks for | `/editor/assets/index-<hash>.js` | 24 669 | 7 765 |
| `/tensorspine/editor/` — what the published site needs | `/tensorspine/editor/assets/index-<hash>.js` | 24 669 | 7 777 |
| `./` — prefix-agnostic | `./assets/index-<hash>.js` | 24 669 | 7 759 |

**The base path is a build parameter, and the published one is not `/editor/`.** The
documentation site is served at `https://maneex.github.io/tensorspine/`, so the editor beside it
is `/tensorspine/editor/` — one path segment more than the feature's own wording. Three ways out,
and 2.18 chooses: build with `--base /tensorspine/editor/`; build with `./` and let every URL be
relative, which is what the site's own pages already do (`%ROOT%` in `docs/style/nav.html`); or
read the base from the page at runtime. The measurement above says the choice is cheap *today* —
the JavaScript chunk is byte-identical under all three bases, and only `index.html` differs —
but that holds only while the page names no asset from JavaScript. A build with a worker, a
dynamic import or an image (which the editor will have: `packages/lang` runs in a worker, per
§5.3) carries the base inside the chunk as well, and then the build and the deployment path are
one decision. `tools/site.sh` already writes `.nojekyll` at the site root, so a directory of
Vite assets under it is served as it stands.

## The writable path: a folder read and written in place

Everything below the picker goes through `FileSystemDirectoryHandle` and `FileSystemFileHandle`
and through nothing else, which is why the tests can exercise it against an **Origin Private
File System** handle: OPFS answers the same interface, and the picker is only where the handle
comes from. What the round trip proves, on every commit:

- `mkdir` creates a nested directory (`primitive-library/primitives/norm.rms`);
- `write` creates the file — **and never the directories above it**: a write into a folder that
  does not exist is refused, so a typo in a path is a refusal instead of a new tree on disk;
- `list` answers entries with their kind, sorted by name; `read` returns the text and a revision;
- `write(path, text, expect)` refuses when the file has moved on since it was read
  (`models/spike.json changed since it was read (…:63 is not …:56)`), and the refused write
  leaves the file exactly as it was;
- `resolve('models/spike.json', '../primitive-library/')` is `primitive-library` — a relative
  reference resolves against the directory of the file it is written in, which is what
  `primitive_libraries[].base` means (§4.3);
- a missing file is `not-found`, not an exception with a DOM name in it.

The whole round trip — a dozen operations — costs **75 ms in Chromium and 1 155 ms in Firefox**
over OPFS: Firefox's `createWritable` … `close` cycle is around 90 ms against Chromium's 8 to 16,
which is worth knowing before the autosave of §4.3 writes a draft every 30 seconds.

### `watch` by polling, and what a revision can tell apart

`watch` lists the subtree and compares each file's modification time and size with the last
listing; the caller gets the difference. A change made **behind the workspace's back** — through
a second handle, as another editor or a `git checkout` would — is reported within one poll
interval in both engines (30 ms at `pollMs: 30`), for `changed`, `added` and `removed` alike, and
nothing at all is reported after the unsubscribe.

One poll of a `data/`-sized tree — 146 files in two directories — costs **22 to 25 ms** in
Chromium (measured once, in Playwright's Chromium 153, over OPFS; a real folder on disk was not
measured and will cost more). At the plan's implied second-scale interval that is a few per cent
of a core, which is affordable; at 100 ms it would not be. The per-commit case reports the cost
on its own one-file tree, which is under 5 ms everywhere.

**A revision is `lastModified:size`, and that is not injective.** Two writes of the *same length*
inside the same millisecond produce the same revision — and it happens: in one of the three runs
made here, two consecutive 64-byte writes in Chromium landed on the same millisecond, where the
recorded run has them 8 ms apart. Firefox never collided, since a write costs it around 90 ms.
The case therefore reports `distinguishableImmediately` as a measurement on every run, and the
test asserts that it was made rather than what it said. What follows is that the optimistic
concurrency above is **best-effort**: it catches an external change of a different length, or one
more than a millisecond apart — which is every change a human or another process makes — and it
cannot catch two same-length writes racing inside one millisecond. Feature 2.6 decides whether
that is enough (it is, for "the file changed under you, reload?") or whether the revision needs a
content hash, which costs a read of every file per poll instead of a stat.

## The handle in IndexedDB, and the one prompt

D11: "its handle kept in IndexedDB so the workspace reopens after a reload with one permission
prompt". Measured, in the application and not only in a case:

- a `FileSystemDirectoryHandle` is stored in IndexedDB and read back as **the same folder** —
  `isSameEntry` says so, and the folder's files are readable through the handle that came back;
- remembering the same folder twice is one entry, because `isSameEntry` is what identity means
  (two handles to one directory are not the same object);
- after a reload the workspace is **offered by name and never re-picked**: the page asks
  `queryPermission` first, reopens with no gesture when the grant is still held, and asks
  `requestPermission` exactly once — on the click that reopens it — when it is not. The
  end-to-end test drives all three states through the shell: pick, reload with the grant
  withheld (one prompt, one only), reload with the grant held (no prompt at all).

**Chromium and Firefox disagree about permissions, and the disagreement is in Firefox's favour
here.** Chromium answers `granted` for an OPFS handle and carries the full permission model;
**Firefox has no `queryPermission` and no `requestPermission` at all** (`handlePermissions:
false`), so a handle it hands back is simply usable. The workspace therefore treats "the engine
has no permission query" as *usable*, and says `unsupported` rather than pretending a grant was
asked for. If Firefox ever ships a directory picker, this is the line to revisit.

**A trap for every later feature that touches recents.** Reading a `FileSystemDirectoryHandle`
back out of IndexedDB **kills the renderer** in Playwright's Chromium — both the headless shell
and the full build of Chrome for Testing 153.0.8010.12 — when the browser context is the default
*ephemeral* one: not a `DataCloneError`, not a rejected promise, a dead page ("Target page,
context or browser has been closed"). The same code is fine in Debian's Chromium 149 with a
profile, and fine in Playwright when the context is a **persistent** one
(`chromium.launchPersistentContext`). Storing the handle is fine either way; it is the read back
that dies. So the two tests that persist a handle open their own persistent profile
(`inPersistentProfile` in `apps/web/e2e/static.spec.ts`), and 2.6's suite will have to do the
same. A handle store is not something a private window can offer, which is the honest reading of
the browser's behaviour — but a crash is not how it should be said.

## The read-only path: a folder upload, and Save as a download

Where there is no writable picker, a folder chosen through `<input type="file" webkitdirectory>`
(or dropped, where `webkitGetAsEntry` gives the tree) is a **snapshot**: the files the browser
copied, and no way back. Measured through the shell, on a folder holding the real
`data/models/llama3-8b.json` and `data/primitive-library/primitive-library.json`:

- the chosen folder's own name is stripped, so the workspace's paths are the same ones the
  writable side answers (`models/llama3-8b.json`), and `resolve` behaves identically;
- the document reaches the editor **byte for byte**, and Save hands the browser those same bytes
  as a download named `llama3-8b.json` — checked against the corpus file itself, not against a
  re-serialisation;
- **the folder on disk is untouched** after the save, which the test asserts by reading it back
  in Node;
- the banner of S18 is shown and says what Save will do; the button says it too
  (`Save — download llama3-8b.json`), rather than pretending to write;
- `write` reports the revision **unchanged**, because nothing moved (plan §5.2);
- `mkdir` is refused, in words that name the way out: *"this workspace is a read-only snapshot:
  … cannot be created. Open the folder in a browser with the writable picker, or create it on
  disk."* **This is the one part of §4.3 the snapshot path cannot do at all**: "New Base creates
  the folder and manifest" needs a folder to create. 3.2 has to say what New Base offers there —
  a downloaded folder is not a thing a browser can hand over — and the honest answer is probably
  that a base is created only in a writable workspace;
- `watch` registers the watcher and never calls it: nothing can change under a copy.

`webkitdirectory` is the one File System capability **all three engines have**, along with
`<a download>` and object URLs — so the snapshot path works everywhere, which is what makes it
the fallback rather than a Chromium-only convenience.

## The three engines

Measured on this box by `pnpm spike:static`, each engine opening the same built page under
`/editor/` (`engines.json`):

| Capability | Chromium 149.0.7827.53 | Firefox 151.0.3 | WebKitGTK 2.50.6 |
|---|---|---|---|
| **`showDirectoryPicker`** — the writable workspace | **yes** | **no** | **no** |
| `showOpenFilePicker` / `showSaveFilePicker` | yes | no | no |
| `FileSystemDirectoryHandle`, `values()` | yes | yes | no |
| `FileSystemFileHandle.createWritable` | yes | yes | no |
| `queryPermission` / `requestPermission` | yes | **no** | no |
| Origin Private File System | yes | yes | no |
| `navigator.storage.persist` | yes | yes | no |
| handle stored and read back in IndexedDB | yes | yes | — (no handles) |
| `webkitdirectory` | yes | yes | yes |
| `webkitGetAsEntry` | yes | yes | yes |
| `DataTransferItem.getAsFileSystemHandle` | yes | no | no |
| `<a download>`, object URLs, IndexedDB | yes | yes | yes |
| **Which path the editor gets** | **read-write folder** | **read-only snapshot** | **read-only snapshot** |

Every capability row feature 0.5 recorded is **confirmed**, none contradicted:
`showDirectoryPicker` in Chromium alone, OPFS in Chromium and Firefox but not WebKitGTK,
`createWritable` in Chromium and Firefox, `webkitdirectory` everywhere. What is added here is the
rest of the row: no permission model in Firefox, no `getAsFileSystemHandle` outside Chromium, and
the fact that Firefox's *only* missing piece is the picker itself.

**A folder dropped on the window is not necessarily a snapshot.** In Chromium
`DataTransferItem.getAsFileSystemHandle` gives a real directory handle, so a drop is the *same*
writable workspace the picker gives; §4.3 describes a drop as a read-only snapshot, which is
right for Firefox and Safari and unnecessarily poor for Chromium. The shell here takes the handle
when the engine offers one and falls back to the snapshot otherwise; 2.6 should keep that order.

**What this says about Safari.** WebKitGTK is the same engine family — the same WebCore and
JavaScriptCore, a different port — so it is evidence about Safari, not a measurement of it; no
Safari runs on Linux and none was available here. Stated exactly:

- **Measured in WebKit**: no `showDirectoryPicker`, no `FileSystemDirectoryHandle` at all, no
  Origin Private File System; `webkitdirectory`, `webkitGetAsEntry`, `<a download>`, object URLs
  and IndexedDB all present, and the snapshot path passes end to end.
- **Reasoned, not measured**: published support tables give Safari the Origin Private File
  System, which this WebKitGTK build does not have, so the *storage* side is probably better in
  Safari than what was measured here — which matters for the drafts store of §5.2, not for the
  workspace. The writable picker is Chromium-only in those same tables, and nothing measured
  here suggests otherwise. The conservative reading, and the one D11 already takes: Safari gets
  the snapshot path.

## Findings

1. **The base path is a build parameter, and the site's is `/tensorspine/editor/`, not
   `/editor/`.** The feature's own wording names the shorter path; the documentation site is
   published under `/tensorspine/`, so the editor beside it is one segment deeper. Three
   possibilities are measured above; the relative build (`./`) is the one that needs no
   deployment knowledge at build time, and is what the site's own pages already do. For 2.18.
2. **Reading a directory handle back out of IndexedDB kills Playwright's Chromium in an
   ephemeral context.** Chrome for Testing 153, headless shell and full build alike; Debian's
   Chromium 149 with a profile is fine, and so is Playwright with
   `chromium.launchPersistentContext`. Any suite that touches recent workspaces must use a
   persistent profile, and any code that reads a handle back should be prepared for it to fail
   in a private window. For 2.4 and 2.6, and for whoever writes the next e2e suite.
3. **A revision of `lastModified:size` is best-effort.** Two same-length writes inside one
   millisecond are indistinguishable, and that was observed in Chromium. It catches every change
   a human or another process makes and cannot catch a race; 2.6 decides whether the reload
   prompt needs more than that.
4. **Firefox has the whole writable workspace except the picker** — handles, `createWritable`,
   directory iteration, OPFS, IndexedDB persistence — and **no permission model at all**. Two
   consequences: the code must treat a missing `queryPermission` as "usable" rather than "denied"
   (it does), and the day Firefox ships a picker the editor gains a read-write workspace there
   with no change beyond the one line that asks for a folder.
5. **`write` must not create the directories above the file.** A workspace path is user input;
   `mkdir` is the operation that makes a folder, and Save into a mistyped folder should be a
   refusal, not a new tree. Stated here because the File System Access API makes the other
   behaviour a one-character difference (`{ create: true }` on the parent lookup).
6. **New Base has no answer on the snapshot path.** §4.3 says "New Base creates the folder and
   manifest"; a browser that cannot write a folder cannot do it, and a download of one file is
   not a base. 3.2 decides — the conservative reading is that a base is created only in a
   writable workspace, and the command is disabled with the reason shown.
7. **A drop can be writable.** `getAsFileSystemHandle` in Chromium turns a dropped folder into
   the same handle the picker gives, so §4.3's "a folder dropped onto the window … a read-only
   snapshot" is right only where the picker is missing. Prefer the handle, fall back to the copy.
8. **The snapshot path is the majority path, not the fallback.** Two of the three engines get it,
   so its chrome (the banner, the Save wording, the untouched folder) is not an edge case to
   polish later: it is what a Firefox or Safari user sees on every save.
9. **The page's own weight is nothing, and that is the point.** 24.7 kB of JavaScript and 7.8 kB
   of HTML for both workspace paths, the shell and the cases; the vendored schemas, corpus and
   reference base (0.2's 2.1 MB) and the language core are what the real page will carry. The
   startup budget of §5.6 (2 s to the shell) has room.

## How to reproduce

```sh
cd editor
pnpm test:e2e                        # the cases, in Playwright's Chromium, on every commit
pnpm spike:static                    # the three engines; writes engines.json
pnpm spike:static --engines=firefox  # one of them
node --experimental-strip-types spikes/static/serve.ts --port 4175 --base /editor/
                                     # …then open http://127.0.0.1:4175/editor/ by hand
```

`pnpm spike:static` needs whatever engines the box has — it names the ones it cannot find and
carries on. On this one: `/usr/bin/chromium`, `/usr/bin/firefox`, and WebKitGTK's `MiniBrowser`
under `xvfb-run`, since it has no headless mode. The unattended run builds the download but does
not hand it to the browser, so opening three engines does not litter the machine with files.
