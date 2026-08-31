# The ZML generator's primitive ABI

*One primitive, asked to emit the body of one occurrence. The request is a projection of the derived
document — D1's arguments, D2's and D3's shapes, D4's state rules — plus the opaque physical
parameters; the response is one MLIR function, or the reasons there is none. The grammar is
`generators/zml/primitive-abi.schema.json`. This is the ZML generator's boundary, not the
language's: the language owns the words the request is made of and nothing about how a primitive is
obtained.*

## Why there is a boundary at all

A ZML primitive does not compute. It runs once, while the graph is being built, and **emits MLIR**;
XLA compiles the result. So the artefact a primitive fundamentally is, is IR — and IR that depends
on more than the contract: on this occurrence's concrete shapes, and on physical parameters the
serving application chooses, `backend` among them. ZML itself works this way, dispatching on a
runtime backend value at emission time.

Every primitive the generator ships is linked in and needs no boundary. The boundary exists for the
primitive that is **not** linked in — a contract the catalog has and this build does not. Without
one, a new primitive means a new runtime binary on every machine before that model can be served.

Two properties follow, and they are why the boundary is JSON in and MLIR text out:

- **Nothing of the runtime's internals crosses it.** No `zml.Tensor`, no `Shape`, no compilation
  context — so the callee is not pinned to a Zig version or a ZML commit, which is what makes a
  published primitive outlive the runtime that loads it.
- **There is no emitter interface to design.** The response is a whole function, so the boundary
  needs no op-construction calls, no handle table, no callbacks. What the primitive may say is
  bounded by StableHLO, which is versioned by its own compatibility policy rather than by ours.

## The entry points

| Symbol | Signature | Returns |
|---|---|---|
| `tsp_abi_version` | `int (void)` | The ABI generation this object implements. A mismatch is refused before anything is asked of it. |
| `tsp_capabilities` | `const char * (void)` | The primitive's entry in the manifest's rule grammar (`generators/CAPABILITIES.md`), so a fetched primitive is declared the same way a linked one is, and the manifest stays derived from code. |
| `tsp_emit` | `const char * (const char *request)` | A response document, always. Refusal is a response, not an error. |
| `tsp_free` | `void (const char *)` | Releases a buffer the object returned. The object allocates; the object frees. |

The mechanism that resolves these symbols is deliberately **not** specified here — a shared object,
a WebAssembly module with the same four exports, or a linked-in table. The boundary is the durable
decision; how the callee is loaded is not, and the same request and response survive changing it.

## The request

| Section | Content | Where it comes from |
|---|---|---|
| `occurrence` | The node this body is for | D1's node identifier |
| `contract` | name and version | D1 |
| `arguments` | resolved, defaults applied | D1, verbatim |
| `signature` | the function type the host will call | D2 shapes, D3 shapes, D4 payloads, and the deployment's workload |
| `states` | the law, access and sharing of each state held | D4's state rule |
| `physical` | opaque, `backend` among its keys | the deployment, or the serving application |

Nothing in it is new vocabulary. Every field is a projection of a document that already has a schema,
which is what makes the request auditable: it can be checked against the derived document it claims
to come from.

### The host dictates the signature

The request states the exact function type, in order, that the host will call. The primitive fills
the body; it does not choose the interface. A returned function whose type differs from the request
is refused before it is appended to the module.

This is what makes a fetched primitive checkable. Validation is mechanical — parse, compare the
function type, splice — so a malformed or dishonest object fails immediately, locally, and with a
message naming the mismatch, instead of contributing a subtly wrong graph that only a fixture
disagreement would ever catch.

Four kinds of operand are named, and exactly one name applies to each:

| Kind | Leading axis | Shape from |
|---|---|---|
| `port` | the elements of this invocation | D2's per-element shape |
| `slot` | none | D3, after the location is assembled |
| `state` | the capacity the deployment fixed | D4's payload |
| `positions` | the elements of the named stream | one index per element |

Shapes are **concrete and never symbolic**. D2 carries per-element shapes and D4 carries bytes per
position; resolving those into extents needs elements per invocation and a state capacity, both of
which are deployment intent, not document facts. The host resolves them before it asks, so the
primitive emits for the shapes it is given and never has to be shape-polymorphic.

## The response

One MLIR `func.func` in textual form, or a list of refusals — never both, never neither.

Refusals use the vocabulary the generators already share: one `name=value` per argument, record
field or state rule the primitive does not implement. That is the same report the linked-in registry
produces before any weight is read, so a fetched primitive refuses in the same words as a linked one
and the runtime's decision does not depend on which it got.

`notes` records a convention read into a contract that leaves one open — as the reference generator's
kernels do in their headers. A note is evidence, never authority: where a note and the contract
disagree, the contract wins and the note has become a finding.

## Where the body is spliced

The generator emits every occurrence as a `stablehlo.composite` named for its contract and version,
with the D1 arguments as composite attributes and the emitted function as its decomposition. That
seam is the same whether the body came from a linked-in primitive or from this boundary, and it is
what a backend with a faster kernel pattern-matches to substitute its own.

An emitted body naming an operation from a dialect the host has not registered fails at parse —
loudly, and before it can affect the graph.

## What is not in it

The physical parameters' grammar: the language declares them opaque and this boundary passes them
through. Whether the shipped object is a shared object or a WebAssembly module, and where it is
fetched from. Anything the runtime could ask a linked-in primitive but not a fetched one — there is
nothing here a linked primitive may do and a fetched one may not, which is the property that keeps
the two tiers interchangeable.
