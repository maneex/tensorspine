# The layout sidecar — `<model>.layout.json`

*Companion note to `tensorspine-editor-layout.schema.json`. Written with feature 2.1 of the
editor's implementation plan.*

## Why the file exists

A `tensorspine/2.0` document is a graph and nothing else. It says which primitives are
instantiated, how they are bound and what their quantities are; it does not say where a node is
drawn, which group is folded, or what the canvas is looking at. That is deliberate on the
language's side and it is the editor's problem, not the language's: the model grammar closes
every object (`additionalProperties: false`), and the specification's mutation test (§10.2) says
that a field no conforming implementation reads is a comment. A `position` written into an
instance would be refused by the loader, and rightly.

So the editor writes a **sidecar** — a second file beside the document, `<model>.layout.json` —
and the document travels without it. A document shared without its sidecar lays out
automatically, which is what the editor does by default anyway (ELK, layered, top to bottom), so
nothing is lost but the user's own arrangement.

## What it holds

| Member | What it is |
|---|---|
| `schema` | `tensorspine-editor-layout/1` — the format, tagged, so that a later revision is read rather than guessed |
| `positions` | a place per node the user moved, keyed by **the document's own path** |
| `collapsed` | the paths of the groups the user folded or unfolded |
| `viewport` | where the canvas was looking |
| `preview_assignment` | the values a *template* document is previewed under, one per external quantity |
| `expanded_view` | what the read-only expanded (D1) view is filtered to |

### The keys are document paths

`instances/embed`, `compositions/decoder/instances/attn`, `interfaces/inputs/tokens`: a key is the
path of the place it is about, with its steps joined by slashes. A step is a member name or an
array index, and the document's root has no key — nothing is positioned at the document itself.

Keying by the document's own paths is what makes the two files answerable to each other:

- **A key that names a place the document no longer has is dropped when the document is opened,
  with a line in the log.** A position for a node that is not there is one the user can neither
  see nor correct, so it is not kept in case the node comes back.
- **A rename carries its keys.** Renaming an instance rewrites every reference to it in the
  document, and the sidecar's keys follow: a key at or below a place that moved moves with it.
  The editor's store answers the list of moved places with the command that made them, and the
  layout store applies it.

`preview_assignment` is the exception, and it is deliberate: its keys are *quantity names*, not
paths, and they are not dropped against the document. An assignment is not part of the document
— it comes from a call site, from `--assign`, or from this file — and the tools' own assignment
check never looks at the assignment's own names, so a stale entry costs nothing and a quantity
that comes back finds the value it had.

### Positions are overrides, not the layout

The default is automatic layout. An entry in `positions` says "the user put this here"; *Reset
layout* empties the map, and the next layout is the automatic one again. The file therefore says
nothing about a document nobody has rearranged, which is why a sidecar for an untouched document
is `{"schema": …, "positions": {}, "collapsed": []}` and not a copy of the drawing.

## Its own history

Layout moves are recorded in a log of their own, separate from the document's. An undo of a
semantic edit ("Add instance attn") must not shuffle positions, and an undo of a move must not
touch the document. The two stores never see each other's history.

## How it is written

Two-space indentation and a trailing newline, through the plain writer. The core's serializer is
the writer of record for the *interchange* formats — the model documents and the library units,
whose bytes the corpus fixes and whose numbers carry a lexical reading the language depends on
(a real is written with a fraction or an exponent). A sidecar is the editor's own file, no tool
of the repository reads or writes one, and its numbers are canvas coordinates with no such
reading, so it is written the plain way.

## What this note does not settle

The shape of `expanded_view.filters` is the one the expanded graph needs today: the families and
the index ranges the plan names as that view's two filters. The feature that builds the expanded
graph is what will confirm it, and a member added there is a revision of this format — the
`schema` constant is what says so.
