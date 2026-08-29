# TensorSpine primitive coverage in ZML

Repository location: `backends/zml/tensorspine-zml-coverage.md`

Audit date: 2026-08-29

## Scope and snapshots

This report maps the TensorSpine reference catalog in
`/home/corwin/work/perso/armature/master` to the public numerical APIs and the
tracked model implementations in `/home/corwin/work/perso/zml`.

- TensorSpine base commit: `3d6bf1fd25b4832cbdcff33c389c4c9f7d918bfe`
- ZML base commit: `a37f903c3fa01599587254b404506fcc9fedd395`
- Catalog denominator: 35 contract files: 34 logical primitives and one
  template (`decoder.causal_yarn`).
- The catalog also contains 35 named axes and 54 precision roles. They are not
  primitive rows, but their coverage is assessed separately below.
- Both worktrees were dirty. None of the catalog files or the ZML core/model
  files used as evidence were modified by this audit. Untracked model code was
  not counted as coverage.

Primary sources:

- [TensorSpine catalog metadata](/home/corwin/work/perso/armature/master/data/catalog/catalog.json)
- [TensorSpine generated catalog reference](/home/corwin/work/perso/armature/master/docs/CATALOG-REFERENCE.md)
- [ZML NN helpers](/home/corwin/work/perso/zml/zml/nn.zig)
- [ZML tensor operations](/home/corwin/work/perso/zml/zml/tensor.zig)
- [ZML higher-order operations](/home/corwin/work/perso/zml/zml/ops.zig)
- [ZML attention dispatch](/home/corwin/work/perso/zml/zml/attention.zig)
- [ZML MoE dispatch](/home/corwin/work/perso/zml/zml/moe/moe.zig)
- [Tracked Qwen 3.5 model](/home/corwin/work/perso/zml/examples/llm/models/qwen3_5/model.zig)
- [Tracked Qwen 3.5 MoE model](/home/corwin/work/perso/zml/examples/llm/models/qwen3_5_moe/model.zig)
- [Tracked Llama model](/home/corwin/work/perso/zml/examples/llm/models/llama/model.zig)

## Executive result

There are three materially different answers to “coverage”:

1. **TensorSpine integration coverage is 0/35.** ZML has no TensorSpine
   contract registry, `{name, version}` resolver, document importer, parameter
   slot mapper, or automatic lowering. No tracked ZML source refers to
   TensorSpine.
2. **The low-engineering-distance tier is 12/34 primitive families (35.3%).**
   Eight contracts have an exact or very thin mapping to existing public
   operations; another four have useful end-to-end variants in tracked model
   code. The latter four are not complete over every legal TensorSpine
   argument. This is a portability estimate, not a count of TensorSpine-shaped
   APIs: only `embed` and `norm.layer` have close, reusable high-level semantic
   counterparts as-is.
3. **Numerical building-block coverage is much broader than 12/34.** The other
   22 primitives can plausibly be authored from public `Tensor`, `nn`, and
   `ops` APIs. They do not currently exist as reusable ZML implementations and
   have no TensorSpine conformance tests, so this report does not call them
   implemented.

Tracked model code contains a numerical path for eight families overall:
`attention.dense`, `embed`, `ffn.gated`, `lm_head`, `moe`, `norm.rms`,
`residual.add`, and `sequence.gated_delta`. That evidence covers particular
model variants, not each contract's complete argument space.

The 34 primitive rows divide as follows:

| Level | Count | Meaning |
|---|---:|---|
| **Direct/thin** | 8 | Existing helper or short, unambiguous expression; no new algorithm or state machinery. |
| **Proven subset** | 4 | A tracked model implements a useful end-to-end instance, but ZML has no contract-complete wrapper. |
| **Building blocks only** | 22 | The operator surface appears sufficient, but there is no tracked reusable implementation or conformance test. |
| **Template** | 1 extra | `decoder.causal_yarn` expands to other primitives and is not in the 34-primitive denominator. |

“Building blocks only” is deliberately not “supported.” It says that an
implementation is technically plausible, not that it has been written,
validated, optimized, or made portable across ZML backends.

## Comprehensive primitive map

### Attention

| TensorSpine contract | Level | Closest ZML implementation/primitives | Actual coverage and gap |
|---|---|---|---|
| `attention.dense@1.0.0` | **Proven subset** | `nn.Linear`, `nn.rope`, `nn.causalAttnMask`, `nn.sdpa`; backend dispatcher `attention.attention`; `Tensor.sigmoid`; model-owned KV caches | Tracked Llama and Qwen code proves causal self-attention and GQA. Qwen also proves learned Q/K RMS scales, partial RoPE, output sigmoid gate, and explicit KV cache updates. Its source contains multi-section RoPE helpers, but the tracked text path does not exercise the multimodal branch. `nn.sdpa` accepts independent Q/K/V and an arbitrary mask, so unmasked and cross-attention math is available; `causalAttnMask` supports a sliding window. There is no single contract implementation covering fused QKV loading, all bias combinations, chunk masks, multimodal sections, cross-attention source domains, position-dependent temperature, streaming, or the TensorSpine ring/window state laws. Optimized `attention.attention` is causal and consumes already projected Q/K/V. |
| `attention.latent_compressed@1.0.0` | **Building blocks only** | `nn.Linear`, `nn.rmsNorm`, `nn.rope`, `Tensor.softmaxBiased`, `Tensor.topK`, `Tensor.gather`, `Tensor.conv1d`, slicing/scatter/update ops, `ops.while` | Low-rank Q/output projections, attention sinks, compressors, sparse top-k indexing, and explicit cache tensors are expressible. `softmaxBiased` is a particularly close match for learned sink mass. No latent-compressed attention, compression cache, sparse index, eviction policy, optimized kernel, model implementation, or test exists in tracked ZML. This is a substantial implementation project, not practical current support. |

### Conditioning

| TensorSpine contract | Level | Closest ZML implementation/primitives | Actual coverage and gap |
|---|---|---|---|
| `conditioning.layer_select@1.0.0` | **Direct/thin** | `Tensor.choose1d(.layer, layer)` or `Tensor.gather` | Exact numerical selection. ZML does not validate the TensorSpine `model.layer` axis identity or contract bounds beyond ordinary shape/index checks. |
| `conditioning.multiplicative@1.0.0` | **Building blocks only** | Two `nn.Linear`/`Tensor.dot` calls, `Tensor.mul`, `nn.rmsNorm`, learned-scale multiply | The full short formula is available, but no layer or test implements this contract. |

### Decoder template

| TensorSpine contract | Level | Closest ZML implementation/primitives | Actual coverage and gap |
|---|---|---|---|
| `decoder.causal_yarn@1.0.0` | **Template** | Tracked Llama `TransformerLayer`; `nn.RopeOpts.Scaling.yarn`; dense attention, gated FFN, RMSNorm, add/no-op rows below | The template’s components can be composed and YaRN has dedicated parsing/math/tests. There is no TensorSpine template expander or one callable ZML decoder primitive. Parameter layouts also differ in places (for example, TensorSpine’s fused QKV/gate-up slots versus commonly split ZML model tensors). |

### Embeddings and output

| TensorSpine contract | Level | Closest ZML implementation/primitives | Actual coverage and gap |
|---|---|---|---|
| `embed@1.0.0` | **Direct/thin** | `nn.TokenEmbedding.forward` (`Tensor.gather`) | Exact lookup, used by tracked Llama and Qwen models. Vocabulary-shard communication is model/sharding configuration, not inferred from the contract. |
| `embedding.token_auxiliary@1.0.0` | **Building blocks only** | Token/table gathers, `nn.Linear`, `nn.rmsNorm`, learned-scale multiply, `splitAxis`/`reshape` | All arithmetic and the layer-axis result shape can be assembled. No auxiliary embedding implementation, parameter-slot loader, or test exists. |
| `embedding.token_position@1.0.0` | **Building blocks only** | `nn.TokenEmbedding` plus position-table `gather` and `Tensor.add` | Straightforward composition; no reusable learned-position embedding layer in tracked ZML. |
| `embedding.token_position_type@1.0.0` | **Building blocks only** | Three table gathers, two adds, `nn.LayerNorm` | BERT-style math is available. No BERT embedding implementation/test and no implicit source for position or segment identifiers. |
| `lm_head@1.0.0` | **Direct/thin** | `nn.Linear.forward` without bias, or `embedding.weight.dot(hidden, feature_axis)` for tied weights | Exact projection. Tracked Llama/LFM code demonstrates both an independent head and tied unembedding. ZML does not enforce TensorSpine sharing-role compatibility. |

### Feed-forward networks

| TensorSpine contract | Level | Closest ZML implementation/primitives | Actual coverage and gap |
|---|---|---|---|
| `ffn.dense@1.0.0` | **Building blocks only** | Two `nn.Linear` calls; `Tensor.silu`, `Tensor.gelu`, `Tensor.relu().powByConst(2)`; selection/masking ops | Basic dense FFN and biases are easy to compose. The optional time-conditioning path and activation sparsity have no implementation. Important semantic gap: ZML’s function named `gelu` is the tanh approximation, so it maps to TensorSpine `gelu_tanh`, not TensorSpine’s distinct exact `gelu`; no public `erf`/exact-GELU op was found. |
| `ffn.gated@1.0.0` | **Proven subset** | Fused `nn.Linear` + `chunkExact`, or split gate/up `nn.Linear`s; activation + `mul`; down `nn.Linear` | Tracked Llama/Qwen models prove the SiLU/SwiGLU path, usually with separate gate/up tensors. Biases are supported by `nn.Linear`. Exact GELU, time conditioning, and activation sparsity remain absent. Do **not** map this contract to `Tensor.swiglu`: that helper forms an outer product and is not the ordinary gated-FFN semantics used by ZML models. |

### Basic plumbing

| TensorSpine contract | Level | Closest ZML implementation/primitives | Actual coverage and gap |
|---|---|---|---|
| `identity@1.0.0` | **Direct/thin** | Return the input `Tensor` unchanged | Exact. There is intentionally no separate ZML operation. |

### Hyper-connections, MoE, and MTP

| TensorSpine contract | Level | Closest ZML implementation/primitives | Actual coverage and gap |
|---|---|---|---|
| `mix.collapse@1.0.0` | **Building blocks only** | `merge`/`reshape`, `Tensor.dot`/`nn.Linear`, elementwise coefficient math, reduction | Required projections and per-stream reduction are expressible. No mHC head, parameter layout, or test exists. |
| `mix.doubly_stochastic@1.0.0` | **Building blocks only** | `nn.Linear`, `reshape`, `exp`/`div`/`sum`, compile-time loop or `ops.while`, batched `dot` | Sinkhorn normalization and stream mixing can be authored. No hyper-connection layer, convergence handling, or conformance test exists. |
| `moe@1.0.0` | **Proven subset** | `Tensor.topK`, `softmax`/`sigmoid`, gathers; `moe.forwardMoe`; ordinary `nn.Linear` for router/shared experts | Tracked Qwen 3.5 MoE proves learned routing, top-k softmax scores, fused gated experts, a shared expert, and a sigmoid shared gate. `forwardMoe` begins after routing: it consumes top-k IDs/weights and is not the full contract. Hash routing, `sqrtsoftplus`, score bias, every `norm_topk`/scale combination, and `swiglu_limit` are not packaged. Fused MoE has CUDA/ROCm/oneAPI, TPU, and Metal choices but returns `UnimplementedMoEBackend` for CPU. |
| `mtp.merge@1.0.0` | **Building blocks only** | Feature-axis `Tensor.concatenate` followed by bias-free `nn.Linear`/`dot` | Exact short formula is available. No MTP head/trunk, tied-parameter orchestration, or test exists. |

### Normalization

| TensorSpine contract | Level | Closest ZML implementation/primitives | Actual coverage and gap |
|---|---|---|---|
| `norm.layer@1.0.0` | **Direct/thin** | `nn.LayerNorm.forward` | Exact scale+bias LayerNorm over the last feature axis. TensorSpine axis identity is not checked. |
| `norm.rms@1.0.0` | **Direct/thin** | `nn.rmsNorm(x, axis, eps).mul(weight.broad(...))` | Exact thin composition. `nn.rmsNorm` alone only normalizes and does not apply the contract’s learned `weight`. Tracked Llama uses the scale directly, while tracked Qwen computes `normalized * (weight + 1)`; a TensorSpine adapter must convert that stored Qwen convention to the contract’s effective scale. |

### Frontends, patching, pooling, projectors, and splicing

| TensorSpine contract | Level | Closest ZML implementation/primitives | Actual coverage and gap |
|---|---|---|---|
| `conv_frontend@1.0.0` | **Building blocks only** | Two `Tensor.conv1d` calls with configured stride/padding, broadcast bias adds, optional position gather/add | The numerical stem is available. No Whisper/audio frontend or learned-position wrapper exists. |
| `patch_embed@1.0.0` | **Building blocks only** | Flatten one patch with `merge`/`reshape`, then `nn.Linear`/`dot`; alternatively `conv2d` for batched image extraction; bias and position add | The contract already presents one spatial/temporal patch, so flatten+projection handles images and video patches. ZML has no tracked vision patch layer, 3-D patch-convolution wrapper, or test. |
| `pooler@1.0.0` | **Building blocks only** | `nn.Linear`, optional `nn.normalizeL2`, then `Tensor.mean`, `choose1d` for CLS, or no reduction | All enum branches are expressible. No pooler API/model/test exists. |
| `projector.patch_merge_bottleneck@1.0.0` | **Building blocks only** | RMS normalization + learned scale, group/flatten with `splitAxis`/`merge`, then three `nn.Linear`/`dot` projections | Arithmetic and static `merge_count` transform are available. No vision-language projector or patch-domain validation exists. |
| `projector.patch_merge_mlp@1.0.0` | **Building blocks only** | `nn.LayerNorm`, `splitAxis`/`merge`, two biased `nn.Linear`s, activation | Arithmetic is available. No Qwen vision merger implementation/test exists; the TensorSpine contract text mentions a non-linearity but exposes no activation argument, so a future adapter must pin the intended fixed activation. |
| `projector.temporal_stack@1.0.0` | **Building blocks only** | Group frames with `splitAxis`/`merge`, then two bias-free `nn.Linear`s | Exact static stacking/projection is expressible. No Voxtral/audio adapter implementation/test exists. |
| `splice@1.0.0` | **Building blocks only** | `scatterSlices`, `dynamicUpdateSlice`, `Tensor.select`, or `concatenate`, depending on placement representation | Placement operations exist, but ZML has no token-domain/placeholder abstraction. TensorSpine’s ports also do not carry numerical placement indices, so an adapter needs graph/domain metadata or an extra runtime mask/index; this is not a direct two-tensor ZML function. |

### Residual and parallel-stream operations

| TensorSpine contract | Level | Closest ZML implementation/primitives | Actual coverage and gap |
|---|---|---|---|
| `residual.add@1.0.0` | **Direct/thin** | `Tensor.add` | Exact and ubiquitous in tracked transformer layers. |
| `residual.altup_correct@1.0.0` | **Building blocks only** | `dot`, `sub`, `mul`, `add`, stream selection, learned-scale multiply | Arithmetic is available. No AltUp state/active-stream convention or implementation/test exists. |
| `residual.altup_predict@1.0.0` | **Building blocks only** | RMS normalization + scale, router `dot`, batched stream `dot`, reshape, selection | Arithmetic is available. No AltUp implementation or active-layer schedule exists. |
| `residual.combine@1.0.0` | **Direct/thin** | `left.scale(ls).add(right.scale(rs)).scale(os)` | Exact short expression for all scale arguments. |
| `residual.laurel@1.0.0` | **Building blocks only** | Two bias-free `nn.Linear`s followed by `nn.rmsNorm` and learned-scale multiply | Exact short branch is available. No LAuReL layer/test exists. |
| `residual.stream_collapse@1.0.0` | **Building blocks only** | Select/split streams, one `nn.Linear` per auxiliary stream, add/reduce | Exact static composition is available. No reusable layer or stream-axis invariant exists. |
| `residual.stream_expand@1.0.0` | **Building blocks only** | One `nn.Linear` per auxiliary stream and `Tensor.stack` | Exact static composition is available. No reusable layer or stream-axis invariant exists. |
| `residual.stream_inject@1.0.0` | **Building blocks only** | `dynamicUpdateSlice`, `scatterSlices`, or slice+`concatenate` | Replacement is expressible once the active stream index is materialized. TensorSpine carries that selection structurally; ZML has no corresponding abstraction. |

### Recurrent sequence operation

| TensorSpine contract | Level | Closest ZML implementation/primitives | Actual coverage and gap |
|---|---|---|---|
| `sequence.gated_delta@1.0.0` | **Proven subset** | Projection `nn.Linear`s, depthwise `Tensor.conv1d`, `nn.GatedDeltaNet.forward`, RMSNorm/gate/output projection, explicit cache tensors | This is the strongest advanced match. `nn.GatedDeltaNet` implements the recurrent delta-rule core with state input/output and contains an inline numerical test. Tracked Qwen 3.5 code adds QKV/z/a/b projections, L2 Q/K normalization, causal depthwise convolution, conv history, recurrent cache, gated RMSNorm, and output projection. The library helper is still only the recurrence, while other `out_gate` choices and the no-convolution form lack one contract wrapper. ZML model code actually generalizes key/value dimensions beyond the catalog’s single `head_dim`. |

## Variant-level audit of the complex rows

### `attention.dense`

| Contract feature | ZML status |
|---|---|
| Q/K/V and output projections; optional biases | Available through `nn.Linear`. Tracked models cover common bias-free and biased layouts, but TensorSpine’s canonical fused `qkv` slot is not loaded by a generic adapter. |
| Dense MHA, GQA, and MQA head ratios | `nn.sdpa` handles fewer KV heads by splitting/repeating the query-head grouping. GQA is exercised; there is no contract-level cardinality validator. |
| `mask: causal` | Implemented by `nn.causalAttnMask` and optimized `attention.attention` backends. |
| `mask: none` | Implemented by calling `nn.sdpa` without a mask; not exposed by the optimized causal dispatcher. |
| `mask: chunked` | Constructible with `iota`, integer arithmetic, comparisons, and selection. No helper, chunk-cache implementation, or test. |
| Sliding `window` | Mask construction is directly supported by `causalAttnMask(..., window_len)`. TensorSpine’s ring-cache state law is not. |
| Cross-attention `source`/`source_values` | Independent Q/K/V tensors make the math possible in `nn.sdpa`. There is no source-domain abstraction or tracked cross-attention layer. |
| `streaming` and KV state | Fixed explicit caches are proven in Llama/Qwen. Generic append/read, fragment continuity, window eviction, session/branch keys, and sharing laws are absent. |
| RoPE base/layout/fraction and scaling | `nn.RopeOpts` covers three layouts, partial rotation, default/Llama3/YaRN/linear/proportional scaling. Multi-section RoPE is model-local and not exercised by the tracked text path. |
| Q/K RMSNorm or LayerNorm, optional learned scales | Both normalizers are available; Qwen proves learned RMS scales. LayerNorm and scale-free combinations are only compositions. |
| Position-dependent `temperature` | Expressible by scaling Q per query position before `sdpa`; no dedicated option implements the TensorSpine record. |
| Per-head sigmoid `output_gate` | Proven by tracked Qwen attention. |

### `ffn.dense` and `ffn.gated`

| Contract feature | ZML status |
|---|---|
| `activation: silu` | Direct `Tensor.silu`; gated SiLU is proven in multiple tracked models. |
| `activation: gelu_tanh` | `Tensor.gelu` is this approximation despite its shorter name. |
| `activation: gelu` (exact) | **Gap.** No public `erf` or exact-GELU operation was found. |
| `activation: relu2` | Thin composition `x.relu().powByConst(2)`. |
| Input/output biases | `nn.Linear.bias` covers both. |
| Fused gate/up slot | A single `nn.Linear` followed by `chunkExact` matches it. Tracked models commonly use two separate linears, so loading needs a layout transform. |
| `condition_dim` / `condition: time` | Projection operations exist; no FFN implementation or input convention applies the optional path. |
| `activation_sparsity` | Top-k, comparison, and mask/select operations exist; selection semantics are not implemented or tested as an FFN option. |

### `moe`

| Contract feature | ZML status |
|---|---|
| Learned router | Proven in tracked Qwen 3.5 MoE with `nn.Linear` + `topK`. |
| Hash router / hash table | Table `gather` is available; no router implementation/test. |
| Softmax scoring | Proven over selected experts in Qwen. |
| Sigmoid scoring | Direct operation; no full routing policy wrapper. |
| `sqrtsoftplus` scoring | Composable from `exp`, `log`, `add`, and `sqrt`; no helper/test. |
| `norm_topk`, output `scale`, and score bias | Reductions/division, `scale`, and `add` exist. Not represented as `forwardMoe` options. |
| Routed expert computation | `moe.forwardMoe` is optimized and accepts top-k IDs/weights, gate-up/down tensors, optional biases, and optional quantization scales. |
| Shared experts and shared gate | Proven in tracked Qwen model code outside `forwardMoe`. |
| `swiglu_limit` | Generic `clamp` can express it, but fused MoE interfaces do not expose it. |
| CPU | **Backend gap for fused MoE.** `Backend.auto` returns `UnimplementedMoEBackend`; a generic expert implementation would be separate work. |

### `sequence.gated_delta`

| Contract feature | ZML status |
|---|---|
| QKV/z/a/b and output projections | Proven in tracked Qwen code with `nn.Linear`. |
| Delta-rule recurrent matrix | Implemented in `nn.GatedDeltaNet.forward`, including explicit input/output state and an inline numerical test. |
| Optional causal depthwise convolution | `Tensor.conv1d` and history tensors are proven for the convolution-present Qwen variant. No packaged convolution-absent branch. |
| Fixed recurrent state and convolution history | Proven as model-owned `GatedDeltaNetCache` tensors. TensorSpine session/branch/fork and ring-law metadata are not represented. |
| `out_gate: silu` | Proven by Qwen’s gated RMSNorm wrapper. |
| `out_gate: sigmoid`, `none`, or `swish` | Elementwise operations are available; no contract wrapper or per-variant test. |
| Head cardinalities | The recurrence validates tagged shapes. Model code supports distinct key/value dimensions, which is a superset of this catalog version, but does not itself validate the TensorSpine single-`head_dim` restriction. |

## Observed axis-name mapping

ZML tags are model conventions, not registered semantic identities. The
following are the recurring correspondences in tracked Llama/Qwen code; an
adapter must still make them explicit and must not assume that equal strings
have TensorSpine catalog identity automatically.

| TensorSpine axis/concept | Common ZML tag(s) | Caveat |
|---|---|---|
| `model.width` | `.d`; projection output often `.dout` | `.dout` must usually be renamed back to `.d`. |
| `model.vocabulary` | `.voc` | Sharing and vocabulary partition semantics are manual. |
| `model.layer` | `.layer` | Frequently a cache-storage axis rather than an activation axis. |
| `sequence.position` | `.s`; attention views `.q` and `.k` | Query/key renames carry operational meaning only by convention. |
| `attention.heads`, `attention.kv_heads` | `.h` | Q and KV tensors can both use `.h` while having different extents. |
| `attention.head_dim` | `.hd` | Gated-delta model code instead uses `.khd`/`.vhd` before renaming. |
| `ffn.inner` / fused projection width | Commonly `.dout` | No catalog-level distinction among inner, gate, and projection axes. |
| `moe.experts` | `.expert`; selected route `.top_expert` | Expert and route axes are model/backend conventions. |
| `deltanet.value_heads` / value dimension | `.vh`, `.vhd`, then recurrence `.h`, `.v` | Requires explicit renames around `nn.GatedDeltaNet`. |
| `deltanet` fused convolution width | `.mix` | Model-local tag. |
| `residual.stream`, conditioning, audio, and vision axes | No stable tracked convention covering the catalog | A future adapter must define these. |

## What the ZML primitives actually provide

The map above rests on these public capabilities:

- `nn.Linear` supplies dense projection, optional bias, and scaled low-precision
  paths.
- `nn.TokenEmbedding`, `nn.LayerNorm`, `nn.rmsNorm`, and `nn.normalizeL2`
  cover the common embedding/normalization kernels.
- `nn.RopeOpts` includes default, Llama 3, YaRN, linear, and proportional
  scaling plus three layouts and partial rotary dimensions.
- `nn.sdpa` supports GQA by splitting query heads, arbitrary additive masks,
  a tensor scale, and softmax bias. `attention.attention` adds causal optimized
  backend dispatch but is narrower semantically.
- `nn.GatedDeltaNet` implements a state-passing recurrence using
  `stablehlo.while`.
- `moe.forwardMoe` implements routed expert execution after the caller has
  selected experts and weights.
- `Tensor` supplies general dot, 1-D/2-D convolution, elementwise math,
  reductions, gather/scatter, top-k/sort, slice/update, reshape/split/merge,
  stack/concatenate, broadcasting, comparisons, and selection.
- `ops` supplies general reduction/window reduction, while/if, gather/scatter,
  collectives, manual sharding computations, composites, and custom calls.

That is enough to make most catalog formulas writable. It is not a substitute
for contract implementations, especially for stateful attention, sparse latent
attention, expert routing policy, and graph-domain transforms.

## Cross-cutting catalog coverage

TensorSpine contracts specify more than numerical formulas. Those extra
semantics are mostly manual in ZML:

| TensorSpine concern | Closest ZML concept | Coverage |
|---|---|---|
| Contract `{name, version}` and template resolution | None | **Missing.** No registry/importer/lowering. |
| Named value axes | `Shape.Tag` strings and tagged-axis operations | **Mechanism only.** Tags can encode names, but ZML has no TensorSpine identity registry or nature/extent validation. ZML rank is capped at 8. |
| Token/patch/position domains and domain transforms | Tensor shapes/tags plus model logic | **Missing as a type system.** Merge/splice/alignment semantics must be lowered manually. |
| Parameter slots and roles | Tensor fields in Zig structs / `TensorStore` names | **Manual.** No role, presence, multiplicity, canonical-layout, or compatibility checks. Fused-vs-split layouts need adapter transforms. |
| Parameter sharing | Reusing the same `Tensor` field | **Possible, not enforced.** No catalog-role check for tied embedding/head weights. |
| State ports, laws, access, session/branch keys | Explicit tensor arguments, `Bufferized` structs, model-owned KV/GDN caches | **Manual/partial.** ZML can carry/update state, but window/ring/append/evict/fork-sharing laws are not declared or checked. |
| Effects | Function dataflow and optional buffer donation/reuse | **Not represented as TensorSpine read/write metadata.** |
| Precision roles | Per-tensor `DataType`, conversion, scaled dot and quantized backend fields | **Storage capability, no role policy.** ZML supports broad float/int/complex types, but does not validate any of the 54 catalog role constraints. |
| Semantic partitions | `Shape.withPartitioning`, `Sharding`, `ops.allReduce`, `manualComputation` | **Powerful but manual.** TensorSpine partition declarations are not consumed or verified. |
| Logical cost metadata | None equivalent | **Missing.** |

This is why “all the necessary tensor operations exist” and “ZML covers the
catalog” are not equivalent statements.

## Important semantic and operational gaps

1. **Exact GELU is not mapped.** `Tensor.gelu` explicitly implements the tanh
   approximation. It covers `gelu_tanh`; TensorSpine separately defines exact
   `gelu`. A new op/composite/custom call is needed for strict coverage.
2. **State policy is model-owned.** Dense-attention KV caches and gated-delta
   state exist in models, but ZML has no generic implementation of TensorSpine
   append/read/evict, ring/window, session/branch, and sharing laws.
3. **Domain semantics are absent.** `splice`, cross-domain attention, and
   patch/frame merges need indices or grouping supplied by model code; shape
   tags do not encode TensorSpine domains.
4. **Optimized kernels cover narrower interfaces.** Attention dispatch takes
   projected Q/K/V and is causal. MoE dispatch takes already routed IDs and
   weights. The higher-level projections, routing, scoring, gates, and state
   remain outside those APIs.
5. **Backend portability is not uniform.** In particular, fused MoE has no CPU
   backend in `Backend.auto`; building-block compositions are not evidence of
   acceptable performance on every target.
6. **Parameter layouts do not line up automatically.** TensorSpine canonically
   describes fused QKV and gate/up slots in several contracts, while tracked
   ZML models often load split projections. The math matches, but direct tensor
   loading requires concat/split/view transformations and sharing checks.
   Effective-value conventions can differ too: tracked Qwen stores an RMSNorm
   delta and applies `weight + 1`, while `norm.rms` defines `weight` as the full
   learned scale.

## Model-level consequences

The TensorSpine signature fixtures make the practical gaps visible:

- `llama3-8b` uses six families: `attention.dense`, `embed`, `ffn.gated`,
  `lm_head`, `norm.rms`, and `residual.add`. Tracked ZML Llama code contains the
  numerical path for all six, although there is no TensorSpine importer and
  canonical parameter layouts can differ.
- `qwen3.5-397b` uses twelve families. Tracked ZML Qwen 3.5/Qwen 3.5 MoE code
  proves the seven text-side families `attention.dense`, `embed`, `lm_head`,
  `moe`, `norm.rms`, `residual.add`, and `sequence.gated_delta`. The five
  vision/multimodal-side families `ffn.dense`, `norm.layer`, `patch_embed`,
  `projector.patch_merge_mlp`, and `splice` are not implemented as that model
  path.
- No tracked ZML model demonstrates the distinctive catalog features of the
  DeepSeek V4 fixture (latent-compressed attention, hyper-connections, MTP),
  Gemma 3n fixture (auxiliary conditioning, AltUp, LAuReL), Whisper/Voxtral
  fixtures (audio frontend/adapters), or ColBERT fixture (BERT embeddings and
  pooler).

## Recommended path to real catalog coverage

1. Add a TensorSpine adapter layer that resolves contract identities, maps
   canonical axes to ZML tags, validates argument domains, and records the
   state/precision/partition metadata that ZML currently leaves to model code.
2. Wrap and test the 12 direct/proven rows first. The wrappers should own
   fused/split parameter conversion and must test every TensorSpine enum branch,
   not only the variants used by Llama/Qwen.
3. For immediate Qwen multimodal coverage, implement the five missing Qwen
   rows: dense FFN, LayerNorm wrapper, patch embedding, patch-merge MLP, and a
   placement-aware splice API.
4. Treat latent-compressed attention, hyper-connections/MTP, and
   conditioning/AltUp/LAuReL as separate feature projects; generic-op
   expressibility understates their state, correctness, and performance work.
5. Generate conformance tests from catalog contracts. At minimum test port
   shapes, optional parameter presence, state transitions, every enum branch,
   canonical tensor layouts, and declared partition communication.

## Validation note

This is a static source audit. `bazel test //zml:test` could not be run in the
restricted workspace: the bare `bazel` command was absent, and the repository
wrapper then failed because its output base under `/home/corwin/.cache/bazel`
was not writable. No runtime result is inferred for the 22 building-block-only
rows.
