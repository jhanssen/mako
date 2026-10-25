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
- DSpark not built yet. Measured groundwork (positions 600-900 of a prose text, 94.5 GB cache): a step of 1 / 2 / 4 /
  6 tokens costs 47-62 / 63 / 101 / 155-185 ms with ~12 misses per token at every width (tokens in a step share
  almost no experts; wider steps still pay off because the dense weights are read once and one step's misses are
  read in parallel). The reference engine's DSpark: on code 97% of drafts kept, 4.6 tokens a round (-> ~1.5-1.65x
  here); on a story 1.34 tokens a round, drafts offered in 113 of 299 rounds (-> ~0-8% here, before its 8 GB).

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
a scaled accumulation per step: not built; the expert GEMMs run at 38-44 TFLOP/s of the 63, so dequantization, not
the multiply, is part of the cost. fp8 / fp4 types need Metal 4.1 and bring no speed.
