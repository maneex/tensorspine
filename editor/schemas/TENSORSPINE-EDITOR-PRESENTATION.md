# The presentation bindings — `packages/ui/src/presentation.json`

*Companion note to `tensorspine-editor-presentation.schema.json`. Written with feature 2.2 of the
editor's implementation plan.*

## Why the file exists

The editor's governing rule is that **an item of information that can be inferred from a schema is
never hard-coded in the GUI**. Every form is generated from a `$def`, every select's options are
an `enum`, every tagged union's tags are the `required` keys of its `oneOf` members, and every
tooltip is a `description`. A test — `editor/tests/audit/no-hard-coding.test.ts` — extracts the
101 values the four schemas enumerate and fails the build if one of them appears as a string
literal anywhere in `packages/ui/src` or `packages/store/src`.

But some things a schema cannot say, and they still have to be said somewhere:

* that an `instance_definition` is drawn as a **node** and a `composition_definition` as a
  **group**, that a `value_binding` is an **edge** labelled by the name of the map member it is
  written under, that a `public_input` is a terminal on the **left**;
* that a `physical_name` gets a token editor, a `scalar_expression` an expression tree, a unit's
  `shape` the axis-row editor, a state port's `rules` an ordered list where dragging a row is a
  real edit;
* that `add` prints as `+` between its arguments and `absolute` as `abs(…)` around its own;
* that a `bytes` of a derived product is shown in B/KiB/MiB/GiB, with the exact number in the
  tooltip;
* and **what a name refers to** — which the schemas are silent about, as feature 2.1 measured.

The plan's answer is one data file, keyed by JSON pointers into the schemas. Not one file per
concern, not a table in a component, not a constant beside the code that uses it: **one**, so that
the exemption to the no-hard-coding rule is a place rather than a habit. The audit test states it
that way — every `.json` under a package's `src/` must be this file — so a second data file fails
the build whether or not it carries vocabulary.

## The key is an anchor

`https://tensorspine.dev/schema/2.0/model.json#/$defs/instance_definition`: the `$id` the registry
indexes a schema under, `#`, and an RFC 6901 pointer into that schema's document. It is the same
string the store computes for a place of a document (`Place.anchor`, feature 2.1), so a binding
and an occurrence are keyed by one scheme and a lookup is a map read.

A place of a document is described by a *chain* of schema nodes — the place itself, then whatever
its `$ref` names, then that node's `allOf` — and the store answers the chain most specific first.
`Presentation.firstOf` takes it in that order, so a binding written for a definition
(`…/derived.json#/$defs/count`) serves every place that refers to it, and one written for a place
(`…#/$defs/composition_definition/properties/indices`) wins over it there.

**The place, not the definition, whenever the two differ in meaning.**
`…/model.json#/$defs/index_ranges` is a composition's index *declarations* where a composition
writes it and a *reference* to them where a value binding's `for_each` does. One definition, two
meanings: so the declaration is bound at
`…#/$defs/composition_definition/properties/indices`, and never at the shape it refers to.

## What a binding may carry

| Member | What it says |
|---|---|
| `role` | `node`, `group`, `edge`, `terminal` — how the construct is drawn |
| `side` | `left`, `right` — which side of the canvas a terminal sits on |
| `face` | the members shown on a node's face, in order; each must be a member the definition declares |
| `structuralSummary` | whether the face carries the instance's structural arguments — *which* they are is the core's answer |
| `label` | `$key`: the construct is labelled by the name of the map member it is written under |
| `widget` | the editor bound to a construct the generic walker has no reading for |
| `references` | what a picker inside that editor offers, by the member each referent is named under |
| `picker` | the list a select is filled from when it is not an enumeration: the base's axes, its precision roles, the versions it carries of a pinned primitive, the family names the workspace's documents write |
| `create` | the label of the action beside that picker when what is wanted does not exist yet |
| `prefix` | what is printed before a name written at this place, so a text form tells an index from a quantity |
| `symbols` | a symbol per value of the enumeration at this anchor, or per tag of the union at it |
| `keywords` | the word written before each member of the construct at this anchor: `if … then … else …` |
| `format` | how a figure is shown: `bytes`, `elements`, `operations`, `count`, `status`, `shape` |
| `statusBar` | this figure is one of the status bar's four totals, in this position and under this label |
| `declares` | the kind of thing the map at this anchor declares |
| `scope` | the anchor of the declaration this one lives inside |
| `refers` | where a name declared here is referred to |

A member name outside that set is **refused when the file is read**: a mistyped `widgit` would
otherwise do nothing at all and nobody would learn of it. A member *value* outside the sets above
is not refused at startup — "unknown constructs get the generic widget" is the plan's rule for a
construct the interface cannot render, and the same answer is the right one for a rendering the
interface does not know yet — but the schema states the admissible values and the audit layer
holds the shipped file to it, so a typo fails the build.

### Symbols

`{"text": "+", "form": "infix"}`. The **form** is not inferable from the arity, which is why it is
written rather than computed: `negate` and `absolute` are both unary and print `-a` and `abs(a)`;
`add` and `min` are both n-ary and print `a + b` and `min(a, b)`.

An operator with **no** symbol prints and parses as `name(args)` (plan §4.13) and is listed by the
startup audit as rendered generically. `min` and `max` therefore carry none: their symbol *is*
their name, and writing it down would be repeating the schema.

The same member binds the alternatives of a union by their tag, which is how the three boolean
connectives get their text: `all` prints as `and`, `any` as `or`, `not` as `not`. It is also how
`present` gets its own — §4.13's symbol list ends with `present(path)`, and a presence test whose
member carried no symbol would print as the bare argument path it holds, which is exactly how a
*reference* to that argument prints. `boolean` and `compare` carry none, and need none: a boolean
condition prints as its own literal and a comparison through its operator's symbol.

An infix symbol also carries a **precedence**, and for the same reason the form is written rather
than computed: an operator's enumeration says nothing about how a reader groups it. `layers - 1`,
`$layer mod 5 = 4` and `heads mod kv_heads = 0` are all written without parentheses on artboard S7,
and `(a + b) * c` cannot be; the numbers are what decide, and the parser of §4.13 reads the same
ones so that text and tree round-trip. An infix symbol with no precedence parenthesises every
nested application — always correct, rarely readable.

### Keywords

`{"if": "if", "then": "then", "else": "else"}`. A symbol names one operator and stands in one
place — before its operand, between two, or around a list — and §4.13's symbol list also carries
`if … then … else …`, which is none of those: three operands, each introduced by a word of its
own. So the binding is a word **per member**, written at the alternative's own anchor, and the
order the words come in is the *schema's* member order rather than the file's.

Without it a conditional expression would fall to §1's generic form, `label(a, b, c)` — and the
label a chooser gives that alternative is `if | then | else`, which is no name a text form can
write or a parser can read. The audit checks each key against the members the definition declares,
as it checks a `face`.

### Prefixes

`$layer` is an index and `d` is a quantity, and no schema says so: both are written as an
`identifier` under a member of a one-member object. The `prefix` binding at the member's own anchor
is what puts the `$` there, which is what lets a reader and a parser tell the two apart (artboard
S7). A place with no `prefix` prints its name bare, which is every other referent.

### Figures

Plan §1 says it in one line — "that `bytes` is shown in MiB" — and the component inventory's §7
says why it is a binding and not a helper: *no component adds, converts or rounds a byte count*.
The derived document's number is the fact; `format` is its rendering.

Every member of the derived schema whose name says bytes carries the binding, and an audit test
derives that set **from the schema** rather than listing it, so a byte figure added to a derived
product fails the build until it is bound.

The renderings themselves are `tools/view.py`'s: `14.96 GiB`, `128 KiB`, `15.01 Gop`, an em dash
for nothing at all. That script is removed when the editor is deployed (the editor plan's finding
F6 — "the conventions … move into the editor's presentation layer"), so the editor is where they
live from now on, in `packages/ui/src/documents/figures.ts`, with the Python beside them in the
comments as every other port keeps it.

### The status bar's totals

§4.2 gives the status bar ten fields and four of them are figures: *D3 total bytes · D5 operations
per element · D4 append bytes per cached position · D2 peak live bytes per element*. Which four
they are is a choice about the interface, and the derived schema cannot state it — so it is
`statusBar` here, `{"order": 1, "label": "parameters"}`, and the bar **finds** them by walking the
derived document against the derived schema rather than reading four paths written into a
component. A fifth figure joins the bar by gaining the member, and a figure that moved in the
schema moves with it.

The label is English and is the key of the interface's own dictionary (§4.21): the schema's member
name (`append_bytes_per_cached_position`) is not a label a bar can carry, and inventing one in a
component is what this file exists to prevent.

## What a name refers to

This is the half the schemas cannot state at all, and it is worth being precise about why.

The *name definitions* are discovered: whatever a `propertyNames` points at is a definition of
names, which in the model schema is `identifier` and `qualified_name`. From them the store finds
the members that **hold** a name (`quantity`, `index`, `instance`, `composition`, `site`, `port`,
`parameter`, `state`, `constant`, `axis`, `coordinate`, `stream`, `name`) and the members that
hold a **map keyed by** one (`quantities`, `instances`, `compositions`, `indices`, `for_each`,
`arguments`, `record`, `inputs`, `outputs`, `values`, `parameters`, `constants`, `states`).

Nothing links the two. A `$ref` names a *shape*, not a scope: no keyword says that the member
`quantity` holds the key of the map `quantities`, or that a generated selector's `instance` names
a *site* of the composition written beside it. So the store answers **occurrences** — tag, kind,
place, and the names written beside it — and a command that renames or deletes is handed the
selectors that say which occurrences it is about.

Those selectors are written here:

```json
"…/model.json#/$defs/composition_definition/properties/instances": {
  "declares": "site",
  "scope": "…/model.json#/properties/compositions",
  "refers": [
    { "tag": "site", "kind": "tagged", "under": "scope" },
    { "tag": "instance", "kind": "tagged", "scopedBy": "composition" }
  ]
}
```

read as: *a site of a composition is named as a `site` inside that composition's own bindings, and
as the `instance` of a selector that writes the composition's name beside it.*

**The scope** is the enclosing declaration. A site and an index cannot be referred to without
saying which composition's they are, so a rule that names `under` or `scopedBy` is read against
it: its path for `under`, its own name for the qualifier. The caller supplies it —
`referenceSelectors(binding, {path, name})` — because the *document* is where the scope is, and
the binding says only that there is one.

**The empty list is a statement.** `interfaces/outputs` declares `output` and refers to nothing:
no construct of a `tensorspine/2.0` document names a public output, so renaming one rewrites
nothing. A referent with no binding at all would be a gap; one with `"refers": []` is an answer.

## The audit

Plan §1 (a):

> A startup audit resolves every key of `presentation.json` against the loaded schemas; a key that
> resolves nowhere is an error in the log, and every enum value and `oneOf` member of the loaded
> schemas that no binding names is listed as "rendered generically".

`startPresentation(registry, log)` is that audit. It runs when the schemas are loaded — the
vendored ones, or a workspace's own — and it answers, besides the errors, the whole list of what
renders generically, which the Log panel shows on demand. The list is long on purpose: most of the
language renders generically and should, a dtype being a plain select and `min` a function of its
own name. What matters is that it is a **list somebody can read** rather than a discovery made in
front of a user.

Three checks go past the letter of (a), each the same failure one level down:

* `symbols` naming a value the enumeration at that anchor does not admit. The plan's own Appendix B
  sketch does this: it gives `binary_operation_expression` the symbols of `add` and `multiply`,
  which belong to `nary_operation_expression`. A binding that resolves and still says nothing is
  worth the same error as one that does not resolve.
* `face` naming a member the definition does not declare.
* a scoped reference rule with no `scope` to be read against, or a `scope` that declares nothing.

The CI half is `editor/tests/audit/presentation.test.ts`: it runs the audit against the
repository's `schemas/` — which is what the static build vendors, byte for byte — and requires
**no** problem at all, holds the file to the schema beside this note, and checks the two sets the
file cannot be allowed to drift on: every byte figure of the derived schema is bound, and every
map the schemas key by a name either declares a referent or is listed there with the reason it
does not.

## What it may not carry

* **No rule of the language.** Whether an argument applies, which ports exist, what a shape
  evaluates to, whether an edge may be made: the core answers those, and a binding that tried to
  say one of them would be a second implementation (plan D3).
* **No vocabulary the schemas already carry.** A binding names an enum value only as the *key* of
  a `symbols` map, where the audit checks it against the enumeration at its own anchor.
* **Nothing about a document.** The bindings are per schema place, not per model: what a
  particular document draws is the document's, and where it draws it is the layout sidecar's.
