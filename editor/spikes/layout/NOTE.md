# Spike: ELK top-to-bottom layout

Feature 0.4 of `plans/graph-editor-implementation-plan.md`, run 10 September 2026.
**Question:** does `elkjs`, in a worker, lay the folded document (roots, compositions as compound
nodes, terminals) and the expanded graph out top-to-bottom, readably, within budget?
**Answer: yes**, with one reservation, stated below, about laying a whole expanded graph out when
the model carries values across the length of it — `gemma3n-kvshare` here, and
`whisper-large-v3` in the corpus at large.

## What is here

`pnpm spike:layout` (from `editor/`) writes every file beside this note. It needs `pnpm oracle`
first: the expanded graphs are the tools' own `--d1`, since the core that will expand a document
in the browser is feature 1.7.

| File | Reading | Where |
|---|---|---|
| `<model>.folded.svg` | root instances, every composition **opened as a compound node** over one representative iteration, interfaces as terminals | §4.7 |
| `<model>.collapsed.svg` | the same, every composition shrunk to one box carrying its boundary handles | §4.7, artboard S1 |
| `<model>.expanded.svg` | D1 — every emitted node, every edge, in D1's topological order | §4.9, artboard S12 |
| `measurements.json` | what each layout cost and how large the drawing came out | §5.6 |

The code: `packages/ui/src/layout/elk.ts` is the layout — boxes, containment, edges in, absolute
coordinates out, and nothing of the language; `graphs.ts` beside this note is the projection of a
document and of D1 into those boxes, which is spike code because reading a document means naming
the schemas' vocabulary and §1 forbids that in `packages/ui` (the core answers it from 2.9 on);
`svg.ts` draws; `render.ts` runs it, in this thread and through a worker thread; `worker.ts` is
the worker's half. The properties are held to account in
`packages/ui/test/layout/elk.test.ts`, on these same graphs.

## Measured, on the development box (i7-4790K, Node v22.22.3)

| Diagram | boxes | edges | ms | ms through a worker | drawing |
|---|---|---|---|---|---|
| llama3-8b · folded | 12 | 12 | 22 | 168 | 315 × 1351 |
| llama3-8b · collapsed | 6 | 5 | 9 | 13 | 324 × 747 |
| llama3-8b · expanded | 195 | 258 | 139 | 200 | 318 × 12468 |
| gemma3n-kvshare · folded | 25 | 31 | 31 | 58 | 1023 × 2490 |
| gemma3n-kvshare · collapsed | 8 | 8 | 8 | 15 | 331 × 1031 |
| gemma3n-kvshare · expanded | 455 | 634 | **1504** | 1880 | 6643 × 25542 |
| deepseek-v4-pro · folded | 42 | 47 | 35 | 61 | 762 × 4344 |
| deepseek-v4-pro · collapsed | 23 | 24 | 14 | 25 | 722 × 2551 |
| deepseek-v4-pro · expanded | 382 | 505 | 176 | 281 | 1181 × 24180 |

Milliseconds are the best of three runs in one warm process. The first layout of a session costs
113 ms whatever the graph: constructing the engine loads a megabyte and a half of compiled ELK,
which is also what the 168 ms of the first worker figure is. The worker adds 20–40 % on top of
the layout for the two structured clones — `Graph` in, `Layout` out — and buys the thing §5.6
asks for: the interface is never blocked.

**The folded canvas — the one an author edits — is 8 to 42 boxes and lays out in 8 to 35 ms**,
so Auto-layout and Reset Layout are a frame or two of work and nothing has to be staged. The
figure that needs watching is the expanded view's.

**The reservation, and what actually costs.** `gemma3n-kvshare`'s whole expanded graph costs
1.5 s and comes out 6 643 px wide, where `deepseek-v4-pro`'s 382 nodes cost 176 ms. Laying every
D1 the oracle writes out says why — the cost tracks **long-range edges**, not node count (best of
two warm runs, an edge counted as long when it spans more than twenty positions of the
topological order):

| model | nodes | edges | long | ms | width |
|---|---|---|---|---|---|
| whisper-large-v3 | 485 | 675 | 31 | **1588** | 1910 |
| gemma3n-kvshare | 455 | 634 | 41 | **1498** | 6643 |
| voxtral-realtime | 382 | 522 | 27 | 959 | 1840 |
| deepseek-v4-pro | 382 | 505 | 1 | 229 | 1181 |
| qwen3.8-27b | 552 | 733 | 1 | 203 | 594 |
| qwen3.5-397b | 528 | 701 | 1 | 201 | 594 |
| qwen3.5-35b-a3b | 408 | 541 | 1 | 166 | 594 |
| qwen3.8-27b-text | 387 | 514 | 0 | 145 | 441 |
| llama4-scout | 291 | 386 | 0 | 111 | 356 |
| shieldstral-3b(-composite) | 307 | 406 | 1 | 109–111 | 654 |
| decoder-causal-yarn@1.0.0 | 156 | 206 | 0 | 107 | 318 |
| llama3-8b · qwen3.5-4b-text | 195 | 258 | 0 | 74–76 | 318–441 |
| colbert-v2 | 74 | 97 | 0 | 72 | 287 |

The corpus's **largest** D1 — `qwen3.8-27b`, 552 nodes — is one of the cheapest, and the
expensive ones are the models that carry a value across the length of the graph. Whisper's
`decoder.cross_source` takes the encoder's output to the cross-attention of all 29 decoder layers;
voxtral's `decoder.time_scale.condition` takes one embedding to all 26; gemma's AltUp path
computes `aux_select` at the head of an iteration and reads it at `condition` at its foot, 30
times, and its longest edge spans 428 of the 455 nodes. A layered algorithm turns each such edge
into a chain of dummy nodes, one per layer crossed, and both the time and the width follow.

Options do not rescue it: thoroughness 1 instead of 7 saves under 10 %, polyline routing trades
width for height, `nodePlacement.strategy = SIMPLE` narrows gemma's drawing to 1 099 px but
stretches it to 42 194 px, and `nodePlacement.strategy = NETWORK_SIMPLEX` overflows the
JavaScript stack outright on it. So the defaults stay ELK's own, and the view §4.9 already describes — read-only,
**filtered by index range** (artboard S12 opens on `layer 0–3`, 25 nodes of 195) and virtualised
— is the reading to build; the whole graph at once is the worst case, not the normal one, and on
`whisper-large-v3` that worst case sits at the two-second budget rather than under it.

## The four properties, and what holds them

The test file asserts, on all nine graphs:

1. **no two boxes overlap** — for every pair where neither box contains the other, since a
   compound node contains its children by construction;
2. **every edge's source is above its target** — its box wholly above, which is what
   "top-to-bottom" means; it holds on all nine, folded and expanded;
3. **every compound node contains its children**, with the room its header declared — one
   composition for `llama3-8b`, one for `gemma3n-kvshare`, three for `deepseek-v4-pro`;
4. **the largest expanded graph is laid out in under two seconds.**

Property 2 is only meaningful because the folded graph is acyclic, and it is acyclic only because
a carry edge is not an edge: `ffn_r[layer−1] → attn_n` is a badge on the receiving card (§4.8,
D8), exactly as `--view` treats it. The test asserts the acyclicity too, so the day a projection
starts drawing carries as edges, it says so instead of quietly producing a back-edge.

## Compared with `--view`'s reading

`tools/view.py` hands Graphviz a `rankdir="TB"` graph and the editor asks ELK for `DOWN`, so the
two agree on the axis and, on `llama3-8b`, on the drawing: `in · tokens`, `embed`, the `decoder`
box with `layer = 0…32 · ×32`, `final_n`, `lm_head`, `out · logits · generative`, one under the
other in that order — the collapsed SVG here is `--view`'s picture with the design pass's ink and
the boundary handles S1 asks for (`attn_n[layer=0].input`, `attn_r[layer=0].a`,
`ffn_r[layer=subtract(layers, 1)].output`) written on the box instead of implied. Where they part
is the composition: `--view` cannot draw one, because Graphviz gets a graph in which the
composition is a single opaque node and the interior is a *second* diagram in a collapsed section
below, so the edge `decoder.entry` stops at the box and the reader jumps to another picture to
find `attn_n`; ELK's compound node draws the same edge into the site it actually names, with the
six sites, their scoped edges, their guards and their carry badges in place, and — on
`gemma3n-kvshare` — shows in one picture what no page shows today: the seventeen sites, the four
guarded ones side by side in their two alternatives (`attn`/`attn_full`, `ffn_sparse`/`ffn`), and
the `aux` edge running from `embed` past the whole box into `aux_select`. The other two
differences are deliberate and belong to later features: `--view` labels an edge with its value
type from D2 (`bf16[tokens, model.width=4096]`) where §4.7 labels it with the binding's rule name
and puts the type behind View ▸ Show Edge Types, and `--view` writes D3/D4/D5 figures into every
box where these cards carry none, because a figure is the core's answer and the core does not
exist yet (§4.7's derived line arrives with 2.15). Nothing of `--view`'s reading is lost by the
move to ELK; what is gained is that the picture is the thing being edited, and that D1 gets a
diagram at all — `--view` has none.

## What this leaves for 2.9 and 2.16

- **Sizes are estimated here.** Card geometry is `_ts.css`'s (`.node` 214 px wide, `.term` 34 px
  tall, `.xn2` one 28 px row) and the text widths are computed from the type sizes, since a spike
  has no DOM. The real canvas measures the rendered card and lays out again when a card's size
  changes — which it will, since §4.7's card grows the ports, slot chips, structural summary and
  derived line the core answers.
- **A compound box cannot be given a minimum size.** `elk.nodeSize.minimumSize` with
  `MINIMUM_SIZE` in `elk.nodeSize.constraints` is ignored on a node with children — measured
  against elkjs 0.12, in both the bare and the enum-set spelling — so a composition box is exactly
  as wide as what it holds, and §4.7's header (name, index range, count, families) can be wider
  than every card inside it. The drawing here cuts the line with an ellipsis, which is what a
  card's `text-overflow` does; 2.9 either does the same or widens the box by padding it.
- **Ports are labels, not ELK ports.** Edges here attach to a box's border, which is what S1
  draws (the wire meets the card's top edge while the port names sit on its sides). Landing an
  edge on a named handle means declaring ELK ports with `portConstraints` and a side; that is a
  decision for 2.9, and this module passes none today.
- **The worker in the browser** is `elkjs/lib/elk-api.js` over `elk-worker.min.js`, handed to
  `layout()` as `settings.engine`; the module speaks only that interface and imports the bundled
  build as its default, so nothing here has to change for the application to move it off the main
  thread.
- **ELK must not reach the first bundle.** `elkjs/lib/elk.bundled.js` is 1.6 MB of compiled Java;
  re-exporting the layout from `packages/ui/src/index.ts` put all of it into the static
  application's chunk (881 B → 1 432 kB, 443 kB gzipped) even though nothing imported `layout`.
  So the layout is the package's second entry point, `@tensorspine/ui/layout`, and the index says
  why; 2.9 loads it where the canvas is laid out — in the worker — and the shell keeps its
  startup budget (§5.6).
- **Edge labels cost height and time.** ELK reserves a band between two layers for the labels it
  is given: `deepseek-v4-pro`'s folded graph is 4 344 px tall with the rule names and 3 148 px
  without, and takes 83 ms instead of 43 ms. Showing them is a View toggle in §4.7, so the layout
  is re-run when it changes — and the expanded view, which has 634 of them, passes none.
