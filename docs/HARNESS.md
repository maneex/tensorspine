# Serving decisions from TensorSpine facts

*Practical, non-normative guide for serving-application maintainers.*

A TensorSpine derived document states what the model requires and which logical choices preserve
its meaning. A harness combines those facts with its generated capabilities manifest, the artifact,
the workload and the machine. TensorSpine does not choose admission policy, page size, cache tier,
placement or scheduling.

The tables use three statuses:

- **exact** — determined by the model and its pinned contracts, possibly after the named deployment
  input is supplied;
- **bound** — safe for planning, but not an equality;
- **not derivable** — the derived document lacks a fact required to decide; the harness must supply
  or measure it.

Field paths are rooted at the single [derived document](TENSORSPINE-DERIVED_JSON.md).

## 1. Admission

Admission is exact relative to one implementation's capabilities manifest. TensorSpine tooling
builds that manifest from the implementation's primitive support; it is not a handwritten model
allow-list.

| Decision fact | Product and field | Deployment input | Status |
|---|---|---|---|
| Primitive version and argument branch required by every node | `d1.nodes.*.contract`, `d1.nodes.*.arguments` | Generated capabilities manifest | exact |
| Value dtype and input domain required at every edge | `d2.values[].dtype`, `d2.values[].domain` | Generated capabilities manifest | exact |
| State law, access, sharing and operations required | `d4.states[].law`, `.access`, `.sharing`, `.operations` | Generated capabilities manifest | exact |
| Logical tensor expected from the checkpoint | `d3.tensors[].shape`, `.dtype`, `.location` | Artifact headers and loader support | exact |
| Whether installed kernels fit memory and meet an SLO | Logical sizes in D2–D5 | Kernel costs, workspace, hardware, workload and SLO | not derivable |

`tensorspine --capabilities MANIFEST MODEL` performs the compatibility check. A successful result
means that the manifest admits the model's declared branches; it is not a performance promise.

## 2. Paging

| Decision fact | Product and field | Deployment input | Status |
|---|---|---|---|
| Bytes added for one cached position | `d4.states[].bytes_per_cached_position` where `.law == "append"` | Positions retained per active allocation | exact |
| Maximum logical bytes of a sliding window | `.span`, `.bytes_bounded` where `.law == "window"` | Active allocation count | bound |
| Fixed-state bytes per allocation | Sum of `.payload[].bytes` where `.law == "fixed"` | Active allocation count | exact |
| Allocation-key dimensions | `.identity`, `.instance_key` | Active values for session and branch axes | exact |
| Whether semantic eviction is permitted | `"evict" in .operations` | None | exact |
| Physical page size, allocator metadata and fragmentation | No field | Allocator and device-memory policy | not derivable |

The model supplies byte coefficients and allocation-key structure. Context lengths and the number
of simultaneously live sessions are workload inputs; page size is an engine choice.

## 3. Sharing and prefix reuse

| Decision fact | Product and field | Deployment input | Status |
|---|---|---|---|
| Granularity at which sharing is semantically permitted | `d4.states[].sharing` | None | exact |
| State ports that are already one logical storage identity | `.identity`, `.members` | None | exact |
| Stream and allocation dimensions that a reuse key must distinguish | `.stream`, `.indexed_by_source`, `.instance_key` | Runtime values for each key axis | exact |
| Whether two sessions contain the same reusable prefix or source | No content or lineage field | Token/source digest and fork lineage | not derivable |
| Bytes saved by reuse | Logical state bytes | Live reuse graph, retention and allocator overhead | not derivable |

`sharing` grants a kind of reuse; it never proves that two requests have equal content. The harness
must establish equality or lineage before aliasing storage.

## 4. Offload

| Decision fact | Product and field | Deployment input | Status |
|---|---|---|---|
| Logical payload for append or fixed state | `.bytes_per_cached_position`, `.payload[].bytes` | Live positions and allocations | exact |
| Maximum logical payload for window state | `.bytes_bounded` | Active allocations | bound |
| Whether a state may be evicted | `.operations` contains `evict` | None | exact |
| State that survives fragmented input deliveries | `.carried_across_fragments` | Fragment schedule | exact |
| Bytes in the physical transfer representation | No physical-layout field | Kernel layout, packing and compression | not derivable |
| Tier, eviction time and prefetch time | No field | Bandwidth, latency, pressure and scheduling policy | not derivable |

Logical eviction permission is necessary, not sufficient, for offload. Restoring an engine's
physical representation without changing semantics remains the engine's responsibility.

## 5. Prefill/decode handoff

| Decision fact | Product and field | Deployment input | Status |
|---|---|---|---|
| State indexed by a source and frozen when that source completes | `.indexed_by_source`, `.stream` | Source-completion event | exact |
| Maximum logical bytes to hand off for window state | `.bytes_bounded` | Active allocations | bound |
| Logical bytes to hand off for fixed state | Sum of `.payload[].bytes` where `.law == "fixed"` | Active allocations | exact |
| Logical bytes to hand off for append state | `.bytes_per_cached_position` | Retained positions per allocation | exact |
| Values crossing a selected graph cut | `d2.cuts[].payload`, `.bytes_per_invocation` | Selected cut and invocation input counts | exact |
| Which state is read in a later serving phase | `.visits` describes the rule but has no structured phase set | Phase schedule and engine execution plan | not derivable |
| Serialization, transport and destination layout | No field | Primitive implementation, network and receiving allocator | not derivable |

A handoff planner must transfer every live state allocation the destination will read. It must not
infer that set from a model name or from `carried_across_fragments`, which answers a different
question: survival across deliveries of a fragmented public input.

## 6. Extracts and placement

| Decision fact | Product and field | Deployment input | Status |
|---|---|---|---|
| Structurally legal graph extracts | `d6.cuts[].cut`, `.kind`, `.sizes` | None | exact |
| Logical payload crossing an extract boundary | Matching `d2.cuts[].payload`, `.bytes_per_invocation` | Invocation input counts | exact |
| Resident parameter bytes and artifact locations | `d3.tensors[].bytes`, `.location`; `d3.totals.bytes` | Artifact availability | exact |
| Peak live value bytes in topological order | `d2.peak_live` | One-invocation input counts | exact |
| Append and fixed-state footprint | D4 byte fields and `.instance_key` | Active key values and retained positions | exact |
| Window-state footprint | `.bytes_bounded`, `.instance_key` | Active key values | bound |
| Meaning-preserving partition choices per node | `d6.partitions[].node`, `.target`, `.communication`, `.granularity` | None | exact |
| Supported physical partition, collective and placement | No field in D1–D6 | Implementation capabilities, topology, workload and policy | not derivable |

A D6 partition is local to the named node. Its `communication` lists the patterns the contract
admits, one or several; its `granularity` is what a shard keeps whole along the axis, the KV group
for attention's heads. Axis identities on D1 edges let an engine coordinate choices across nodes;
TensorSpine does not emit a whole-model sharding plan.

## 7. Concrete D4 readers

The following Python-shaped readers show the minimum checks a harness needs. They are pseudocode,
not a TensorSpine library API. `need` refuses when either the derived field or the required runtime
fact is absent.

```python
class MissingFact(Exception):
    pass


def need(obj, key):
    if key not in obj or obj[key] is None:
        raise MissingFact(key)
    return obj[key]


def allocation_key(state, key_values):
    """Resolve D4 key-axis names to one runtime allocation key."""
    axes = need(state, "instance_key")
    try:
        values = tuple((axis, key_values[axis]) for axis in axes)
    except KeyError as error:
        raise MissingFact(f"runtime value for {error.args[0]}") from error
    return need(state, "identity"), values
```

### Paged append with logical positions

`logical_position` makes the logical address explicit. Page size remains deployment policy.

```python
def append_address(state, key_values, logical_position, positions_per_page):
    if need(state, "law") != "append" or need(state, "access") != "logical_position":
        raise ValueError("not a pageable append state")
    if positions_per_page <= 0 or logical_position < 0:
        raise ValueError("invalid physical page geometry")
    page, offset = divmod(logical_position, positions_per_page)
    return allocation_key(state, key_values), page, offset
```

### A window read across a wrapped ring

The harness supplies the exclusive right edge, `produced_until`. D4 supplies the retention span.
The returned physical slots may wrap; for example, consecutive logical positions can map to the end
and then the beginning of the ring.

```python
def ring_addresses(state, key_values, logical_positions, produced_until):
    if need(state, "law") != "window" or need(state, "access") != "ring":
        raise ValueError("not a window ring")
    span = need(state, "span")
    if (not isinstance(span, (int, float)) or isinstance(span, bool)
            or span <= 0 or span != int(span)):
        raise ValueError("D4 span must be a positive integer")
    span = int(span)
    first_retained = max(0, produced_until - span)
    if any(p < first_retained or p >= produced_until for p in logical_positions):
        raise KeyError("requested position is not retained")
    key = allocation_key(state, key_values)
    return [(key, position % span) for position in logical_positions]
```

### A fixed state at a fork

`at_fork_point` permits fork-scoped reuse. A mutable fixed state must be copied before parent and
child diverge; D4 does not supply the fork lineage or the copy primitive.

```python
def fork_fixed_state(state, parent_key_values, child_key_values, storage, clone):
    if need(state, "law") != "fixed" or need(state, "sharing") != "at_fork_point":
        raise ValueError("not a fork-copyable fixed state")
    if not {"read", "write"}.issubset(set(need(state, "operations"))):
        raise ValueError("state cannot be copied and then updated")
    parent = allocation_key(state, parent_key_values)
    child = allocation_key(state, child_key_values)
    storage[child] = clone(storage[parent])
    return child
```

### Prefix-sharing keys

The enum selects the required runtime proof. The proof itself is deliberately not in D4.

```python
def prefix_share_key(state, runtime):
    mode = need(state, "sharing")
    stream = tuple(sorted(need(state, "stream").items()))

    if mode == "by_position":
        return mode, stream, need(runtime, "prefix_digest"), need(runtime, "logical_position")
    if mode == "by_source":
        return mode, stream, need(runtime, "source_digest")
    if mode == "within_span":
        return (mode, stream, need(state, "span"), need(runtime, "retained_span_digest"),
                need(runtime, "right_edge"))
    if mode == "at_fork_point":
        return mode, stream, need(runtime, "lineage"), need(runtime, "fork_position")
    raise ValueError(f"unknown D4 sharing mode: {mode!r}")
```

The key is an equality witness for harness policy, not a prescribed storage key. An engine may use a
collision-resistant digest, verified token sequence, source object identity or another proof with
equivalent semantics.

## 8. Batching

The language describes one session's invocation. Batching is downstream:

1. Give every active session and branch the runtime values named by each
   `d4.states[].instance_key`.
2. Allocate state once per distinct resolved key. Before prefix reuse, N distinct session values
   make session-keyed state additive across N sessions.
3. Multiply append bytes by retained positions; use `bytes_bounded` for window state and the fixed
   payload bytes for fixed state.
4. Read `d2.peak_live` for the logical value peak of one invocation. If the field is absent, stop:
   the activation peak is not derivable from the remaining products. Physical batched peak still
   depends on the engine's schedule, fusion, workspace and buffer reuse.

TensorSpine therefore supplies the state terms and the single-invocation logical value peak. The
harness supplies concurrency, batching strategy and every physical-memory term.
