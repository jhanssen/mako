# Performance work: what landed, what didn't

Measured on an M5 Max (128 GB, 40-core GPU) unless noted. Numbers are from interleaved A/B runs where it mattered:
identical runs vary up to ~12 ms/step over long benchmark stretches, and the machine drifted ~15% slower over one
evening in both arms, so single runs or runs hours apart are not comparable.

## GLM-5.3-Flash (Maya-S IQ2_XXS, 96.5 GB; experts ~86 GB)

### Landed

| Change | Effect |
| --- | --- |
| Expert arena: mlock'ed anonymous memory wrapped no-copy in Metal buffers, GPU-side routing and expert->slot table, misses read by the CPU (pread + F_NOCACHE) while the resident experts run | 75 GB arena, ~97% hits, ~41 ms/token at the time (from ~0.93 s/token streaming on the M4) |
| Routing-profile placement (`mako_calibrate`, profiles keyed by the routers' SHA-256) + decayed-count eviction | which experts start resident; counting rules beyond decayed counts gained nothing measurable |
| Batched prefill: layer-major groups (each routed expert loaded once per layer and group), matmul2d GEMMs on half activations (38-44 TFLOP/s), batched indexer selection | ~458 tok/s on the 2.4K-token corpus |
| Short-prompt prefill: per-layer lendable arena slots for scratch, misses read 4 experts at a time with each batch committed once resident | at 15K context: 64 tokens 1098 -> 593 ms, 256 1821 -> 1141, 512 2039 -> 1698, 1024 3142 -> 2698 |
| 16-token GEMM tile for IQ types at <= 16 rows per expert | bit-identical to the 32/64-row tiles; matmul2d setup was the small-row cost |
| Speculative decoding with the next-token block (two-token verify, rollback of the recurrent state) | 15-25% faster than plain decode at 83-100% acceptance; break-even ~35-45% |
| Router before the shared expert in decode (the CPU starts on misses earlier) | -0.5 ms/token |
| Busy-spin on the routing event instead of a blocking wait | blocking was +1 ms/token; no throttling seen |
| 128- and 256-token GEMM tiles for IQ weights where they pad no more than 64-token ones (each weight dequantized once per tile; the dequantization does not overlap the multiply and cost ~21%) | `bench/gemm_bench` (288 experts, GLM shapes): 114 tokens each 39.5 -> 33.0 ms a layer, 1024 tokens 312 -> 248 ms (48 -> 60 TFLOP/s of the 63 half x half reaches); GLM prefill +7-8% (547 -> 592, 546 -> 586 tok/s in A/B pairs), experts section -19%; logits unchanged |
| The same tiles for Q6_K, Q8_0, F16 and F32 weights (dense projections: 4096 tokens were 64 tiles, each dequantizing every weight) | Q6_K 4096x8192 at 4096 tokens 13.7 -> 10.3 ms (40 -> 53 TFLOP/s), 8192x4096 8.6 -> 5.8 ms; GLM prefill +6-10% (534 -> 586, 549 -> 581 tok/s), KDA section -13%, FFN -15%; logits unchanged |
| MLA per-head GEMMs (k_b, v_b: 64 heads, each its own dispatch into one buffer) in a concurrent encoder | 5.0% -> 3.0% of prefill GPU time: in a serial encoder each waited for the one before (one wave of 256 threadgroups each, ~1500 a prefill); logits unchanged |
| KDA q/k l2 norm: one simdgroup per 128-wide row (was a 256-thread threadgroup and a threadgroup reduction), summing in the same order | 2.7% -> 0.7% of prefill GPU time (`prefill_sweep --parts`); logits bit-identical. Summing in a different order instead moved prefill-vs-decode KL under the ablation test from 2.8e-9 to 1.5e-5 and KL against the reference by ~10%: GLM-5.3's outputs react strongly to rounding, so the tolerance checks act close to bit-exact |
| KDA scan with each thread's 32 state columns in registers (was threadgroup memory, read and written twice per token: ~2.2 TB/s of traffic) | scan ~1.05 s -> ~0.3 s on 4096 tokens; prefill +6-8% (562 -> 609, 464 -> 493 tok/s in A/B pairs); logits unchanged against the reference |

Decode at 15K context after these: ~46-54 ms per speculative step (~1.95 tokens): GPU kernels ~36 ms, expert reads
10-19 ms (not overlapped), drafting ~2.3 ms. The big GEMVs already run at ~400 GB/s of the 553 GB/s measured (614
GB/s specified); the estimated bandwidth floor is ~20 ms/step.

### Tried and rejected

| Idea | Result |
| --- | --- |
| Expert read-ahead in decode (next layers' routers on this layer's FFN input) | top-8 catches 48-54% of misses but wastes ~21 reads/step; the window before the next routing (~0.6 ms) is about one 7 MB read (best 0.69 ms). Best variant (2 layers ahead, top-8, one read, worker thread): +1.4%. Reverted |
| Same-layer prediction before attention | 47% of misses, 31 wasted reads/step |
| Leaving prefill out of the eviction scores, or one score update per prefill group | steady decode misses +20% (19-21 vs 15-17 a step); the spike after a long prompt unchanged |
| Reserving the prefill chunk's scratch inside the arena budget | 512-token prefill 1477 ms (better) but decode +3% (~1 ms/token) |
| Few-token register GEMM for experts at <= 2 rows | +24-29% on short prompts, but a token's result then depends on how many others share its expert (logit KL ~3e-4 between kernels, drafts 8.7e-3): kept off |
| De-duplicating experts shared by the two verify tokens | they share 1.68 of 8 experts a layer (~10% pairs): not worth it |
| Two-draft chaining | estimated +5-10% (each extra verify token ~15 ms/step); not built |
| MTLIO instead of pread for expert reads | no faster at large blocks, much slower at small blocks or low concurrency |
| KDA scan loading 8 tokens' q/k/g at once (one barrier per 8) | KDA section +15%: 30 KB of threadgroup memory halves occupancy |
| KDA scan prefetching the next token into registers, double-buffered (one barrier per token) | no change |
| KDA scan with 8 threads per row (16 rows a threadgroup) | KDA section +9% against 4 threads per row; it also moved the second token of a two-token step by KL 4e-6 against single steps (cause not found) |
| GEMM instead of the GEMV for small-output projections at >= 256 tokens (router, KDA gates) | prefill +3.5%, but KL against the reference 0.00719 -> 0.00845 |
| The same for the KDA gate projections only (f_a, g_a, beta), with and without a second GEMM on the activations' rounding error (hi + lo halves) | KL 0.00719 -> 0.00784 / 0.00755 (sparse-attention check), top-1 below its limit: activation precision is not the cause (half-rounded weights, inferred); the most it could gain was 75-177 ms of KDA (1-2%) |
| Sparse attention output in 64-column tiles (8 simdgroups on a 64x64 matmul2d tile) | pathologically slow: the end-to-end prefill tests took ~15 min instead of ~15 s |
| Sparse attention without the sum pass when there is one split (output written in place) | no change (5.5% vs 5.3% of prefill GPU time) |
| Sparse-attention groups of up to 128 tokens (was 32) | no change (sparse attention and selection 744 vs 748 ms): the kernels, not the group size, bound it |
| Per-kernel telemetry to find slow kernels | timestamps around each dispatch inflate small kernels several-fold (one "slow" shared-expert GEMV was 93 us measured, 13 us real); repeat-in-place costs (`MAKO_REPEAT`) replaced it |

### Open

- Decode misses still rise after a long prompt, less since staging (GLM: ~27 vs ~15 a step for the first 40 steps).
- Expert GEMMs at few rows per expert in short prefills (512 tokens: experts ~714 ms GPU against a ~150 ms
  weight-read bound).

## DeepSeek-V4.1-Flash-1017 (Maya-S IQ2_XXS, ~178 GB; ~40% of experts resident on 128 GB)

### Landed

| Change | Effect |
| --- | --- |
| Shared runtime with GLM (arena, decode MoE with CPU-served misses, MoE prefill) | decode 21 tok/s on a short prompt at first run (46.9 ms/step, 14.5 ms of it reading ~13 missing experts) |
| One T-token layer path for decode and prefill; attention reads window and compressed keys in place | prefill working: 1300 tokens 111 tok/s before tuning |
| Attention through matmul2d (half operands staged in threadgroup memory, float accumulation) | attention section 2.5 s -> 0.83 s on 1300 tokens; prefill 130 tok/s; KL vs the reference unchanged (0.0061) |
| Prefill group 8192 (was 2048) | 2.4K-token prompt 146 -> 226 tok/s: each group re-reads every missing expert it routes to (~92 GB at ~12 GB/s, ~8 s) |
| Staged prefill reads: a prefill group of >= 512 tokens routes to nearly every expert, so the arena's missing experts of every layer are read ahead (8 threads, 1 MiB reads, 16 GB/s) into a 3 GB ring while the GPU works, and run from there without becoming resident | 2.4K-token prompt 275 -> 325-349 tok/s (A/B in sequence); now read-bound (~6.1 s of reads in ~7 s). The first version (6 threads, whole-part reads) read at 10.6 GB/s and gained nothing |
| Expert cache margin a quarter of RAM from 96 GB up (was 40%): DeepSeek 74 -> 94.5 GB, GLM-5.3 74 -> 86 GB (every expert resident) | DeepSeek at 90 GB: prefill +16% (352 vs 304 tok/s), decode after the prompt +13% (misses 28.8 -> 19.8 a step). Stress test (160 GB written beside decode): lowest available 20.4 GB (DeepSeek) / 32.4 GB (GLM), no swapping |
| Routing profile through prefill (per-token counts) | the profile now reflects routing frequency; first attempt counted each expert once per group and made decode worse (41 misses/step) |

Prefill on prompts of a few thousand tokens is SSD-bound: ~6 s of expert reads once per group (overlapped with the
~1.8 ms/token of GPU work since staging). Decode runs 13-21 tok/s; misses per step vary 15-40 with content and history.

### Tried and rejected

| Idea | Result |
| --- | --- |
| Attention on fp32 simdgroup matrices | slower than the scalar kernel: ablation showed the multiply-accumulates themselves bound it (~0.9 TFLOP/s; they do not reach the matrix units) |
| Staged prefill reads for GLM-5.3 with a 3 GB ring | prefill -11% (429 -> 382 tok/s at 4K): the ring's borrowed memory evicts resident experts. Sized to two layers' missing experts instead (~0.55 GB): prefill unchanged, decode after a 4K prompt ~4.5% faster (first 40 steps 32.1 -> 26.6 misses a step: prefill no longer evicts the experts steps use). Landed that way |
| Marking slots filled by prefill misses as preferred victims (no staging) | no change (38.6 vs 38.7 misses a step after the prompt): in a group nearly every resident is in use, so each miss still evicts one of decode's experts before any marked slot frees up |
| One 1024-thread threadgroup per token for the attention output (no repeated reads of probabilities and keys) | no change: redundant reads were not the cost |
| Per-row key descriptors with vector loads | no change for the output kernel (helped the score kernel slightly) |
| More query tokens per attention dispatch (32 -> 256) | no change: occupancy was not the cost |

### Open

- Prefill reads (~6 s a group, now overlapped with the GPU work): more resident experts or fewer bytes per expert.
- Decode misses (15-40 a step).
- DSpark (opt-in, `--dspark DIR`): greedy output identical to plain decoding (verify up to 8 tokens runs the decode
  path); 19.8 -> 22.5 tok/s on code (93% of drafts kept), 23.5 -> 22.8 on a story (70%), its 8 GB costing the cache
  ~9%. Draft rounds take ~11 ms (the reference engine's ~2.4). Measured groundwork (positions 600-900 of a prose text, 94.5 GB cache): a step of 1 / 2 / 4 /
  6 tokens costs 47-62 / 63 / 101 / 155-185 ms with ~12 misses per token at every width (tokens in a step share
  almost no experts; wider steps still pay off because the dense weights are read once and one step's misses are
  read in parallel). The reference engine's DSpark: on code 97% of drafts kept, 4.6 tokens a round (-> ~1.5-1.65x
  here); on a story 1.34 tokens a round, drafts offered in 113 of 299 rounds (-> ~0-8% here, before its 8 GB).

### Where a decode step goes (2026-10-25, short story prompt, Q8_0 dense, 94.4 GB expert cache)

`ds_run --gen 200 --spec 0 --telemetry layers|parts|kernels`. A step is ~41 ms, GPU-bound (inputs and encoding <1 ms):

| | ms/step | Note |
| --- | ---: | --- |
| GPU waiting for missing experts | 12.2 | ~10.7 misses of ~10 MB; one 7-14 MB read peaks at 10.5-11.9 GB/s on this SSD (`io_bench batch`), so ~0.9-1.1 ms each |
| Resident experts (IQ2_XXS / IQ3_XXS indexed GEMV) | 7.8 | the IQ kernels run at 360-385 GB/s (Q8_0 at 566-580 on the same shapes): ~4.6 ms at bandwidth |
| Attention output (out_a, out_b) | 6.1 | 3.2 GB of Q8_0, near bandwidth |
| Attention q/kv projections | 4.8 | 2.2 GB of Q8_0 |
| Shared expert | 3.6 | 1.5 GB of Q8_0 |
| Router + routing | 2.1 | the F32 router GEMV is ~0.6; the route kernel and the event signal the rest |
| hc and norm (attention and FFN) | 1.9 | small kernels |
| Attention core | 1.8 | |
| Miss experts, Engram, compressor, combine, hc post | 2.4 | |

Weights read per token: ~8.5 GB dense + ~2.5 GB routed experts = ~11 GB, ~19.6 ms at 560 GB/s; GPU compute is
~28-30 ms. A step issues ~1800 dispatches; a tiny dependent dispatch costs ~2 us, so ~3.6 ms is launch overhead.

Dense re-quantization at load (`ds_run --dense-quant`, 1300 positions against the reference):

| Dense | Top-1 | KL mean | Median step |
| --- | ---: | ---: | ---: |
| Q8_0 (stored) | 98.2% | 0.00557 | 46.0 ms |
| Q6_K | 97.5% | 0.00755 | 40.9 ms |
| Q5_K | 96.8% | 0.00964 | 43.5 ms |
| Q4_K | 96.9% | 0.01609 | 38.6 ms |

Cache-aware routing (`--miss-penalty F`: subtracted from a non-resident expert's selection score, its weight
unchanged), 1300 positions against the reference:

| Penalty | Misses/step | Top-1 | KL mean | Median step |
| --- | ---: | ---: | ---: | ---: |
| 0 | 13.4 | 98.2% | 0.00557 | 44.4 ms |
| 0.02 | 10.9 | 98.5% | 0.00648 | 41.3 ms |
| 0.05 | 8.6 | 98.2% | 0.00670 | 40.2 ms |
| 0.1 | 6.6 | 98.0% | 0.00881 | 37.9 ms |
| 0.2 | 4.4 | 97.5% | 0.00999 | 35.4 ms |
| 0.5 | 2.1 | 96.5% | 0.01852 | 33.4 ms |

It beats dense re-quantization per unit of KL; output then depends on what is cached (not reproducible across
sessions), and long sessions are unmeasured.

mako's own IQ2_XXS decode GEMV (grid table as half in threadgroup memory, signs by negation, the per-format kernel's
summation order): bit-identical, 361 -> 405 GB/s, resident experts 7.8 -> 7.35 ms a step (+1.5% tok/s). The same for
IQ3_XXS (385 -> 440 GB/s) differs from the per-format kernel at rounding level (1.4e-7 relative: the compiler fuses the
original's arithmetic differently, and reproducing its expression form did not match it). Accuracy is unchanged
(DeepSeek 1300 positions vs the reference: top-1 98.2% both, KL 0.00557 -> 0.00572; GLM-5.3 decode vs the reference
KL 0.0114, top-1 94.5%), so it is used; the GLM prefill-vs-decode limit was re-baselined (1.2e-4 -> 3.1e-4), and the
session check's failure was a Session bug (a new session over a used engine did not reset it). More rows per simdgroup (8, 16) is slower for both: the kernels are bound
by per-weight work, not by re-reading the activations.

Decode at long context (code text, 100 greedy tokens after the prompt): 16K tokens 54.5 ms a step (18.9 tok/s), 32K
54.3 ms (18.8); short prompts ~41 ms. Attention grows modestly (core 1.8 -> 2.8 ms, indexer 0 -> 1.4 / 2.4 ms); most
of the difference is misses (17-18 a step against ~11). Leaving prefill out of the eviction scores made decode after a
16K prompt worse (19.8 vs 15.4 misses a step, 18.6 vs 20.7 tok/s), as it did on GLM.

Miss prediction (`ds_run --route-study N`, story prompt, 300 steps, 9.4 misses a step; a predicted expert neither
resident nor selected counts as a wasted read):

| Predictor | Misses caught | Wasted reads a step |
| --- | ---: | ---: |
| previous token's experts / same input token earlier | 0-1% | ~0-1 |
| router 1 layer ahead (layer l's router on layer l-1's FFN input), top 2 / 4 / 6 / 8 | 15 / 34 / 53 / 63% | 0.9 / 3.4 / 9.9 / 21 |
| router 2 layers ahead, top 2 / 4 / 6 | 14 / 30 / 44% | 1.1 / 5.0 / 13 |
| 2-ahead top 8 that is also in 1-ahead top 8 | 47% | 7.4 |

After a 16K code prompt (15.4 misses a step) 1-ahead top 6 catches 39% (21 wasted). Recently used experts are
resident, so misses are experts not used for a while; only the router predicts them. A prediction is ready about one
layer (~0.7 ms) before it is needed, competing with that layer's own misses for the SSD: estimated 2-3 ms a step on
the story. Not built yet.

Kernel fusion in decode (2026-10-25). Fusing DeepSeek's hyper-connection step (stream norm, the 20480x24 F32
projection, mix, lagged collapse, the sublayer's RMS norm and the previous sublayer's expand: six dispatches into
GLM's two-kernel hc step with a lagged mode and a fused norm) gave 23.6 -> 24.0 tok/s (~0.6 ms a step), but KL against
the reference rose with every summation split tried (separate kernels 0.00557 / 0.00572; fused with 16 / 32 / 64 parts
0.00615 / 0.00632 / 0.00678, top-1 98.1-97.7%). Likely the reference computes in the separate order, so this measures
agreement with it rather than accuracy; not taken for 1.7%. Repeat-in-place costs of the remaining small kernels
(`MAKO_REPEAT=<kernel>:2`, busy GPU time, base 27.6-27.8 ms a step):

| Kernel (dispatches a step) | Added when doubled |
| --- | ---: |
| rms_norm (246) | +0.6 ms |
| miss-expert GEMVs (120, mostly with no misses) | +0.4 ms |
| ds_rope (125) | +0.25 ms |
| moe_route (40) | +0.15-0.2 ms |
| ds_round (85), swiglu (120) | +0.1 ms each |
| hc_mix, hc_collapse, hc_expand (80 each), moe_combine_split (40) | ~0 |

Repeats back to back undercount pipeline switches (the fused hc step saved ~0.6 ms where its kernels show ~0.4), so
fusing everything is worth ~2-3 ms a step (7-10% of busy time), each step likely with a similar shift against the
reference. Not pursued.

Short GEMV benchmarks at low GPU clocks read ~2x slow (a first measurement blamed the Q4_K kernel; warmed up it
runs at 464-542 GB/s).

## Apple10 (M5) versus Apple9 (M3/M4)

No code path checks the GPU family; the Apple10 hardware is reached through the same Metal 4 calls that run on Apple9.
What Apple10 adds that matters here is the per-core matrix units behind `matmul2d` (MetalPerformancePrimitives). Mako
uses `matmul2d` for every prefill GEMM (dense and expert) and DeepSeek-V4.1's attention, so those already run on
them on M5; decode is GEMV-bound by memory bandwidth, which no Apple10 feature changes.

`bench/mm_types_bench` (M5 Max, 4096^3, every operand pair `matmul2d` accepts):

| Operands | Throughput |
| --- | ---: |
| half x half -> float (mako's), relaxed precision, half -> half, bfloat | 63-64 TFLOP/s |
| half x int8 / int4 / int2 / fp8 / fp4 -> float | 63-65 (the compact operand is widened to half) |
| fp8 x fp8 -> float | 65 |
| fp4 x fp4 -> float | 57 |
| int8 x int8 -> int32 | 121 TOP/s |
| int8 x int4 / int2 -> int32 | 93 / 83 |

Only int8 x int8 is faster than half (2x). Using it for expert GEMMs means int8 activations (quantized per token) and
the weights' per-32 block scales applied outside the multiply (matmul2d has no scale operand), so a K step of 32 and
a scaled accumulation per step: not built. With 128/256-token tiles the expert GEMMs run at 50-60 TFLOP/s of the 63
(the rest is tile padding at ~114 tokens an expert), so int8 could at best double the multiply's rate while adding
activation quantization and per-block rescaling. fp8 / fp4 types need Metal 4.1 and bring no speed.
