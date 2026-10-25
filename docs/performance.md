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

- Decode misses rise after a long prompt (~34 vs ~16 a step for ~75 steps). Part of it is prefill's misses
  evicting decode's resident set (keeping them out cut the rise ~20%); a fix without staging's cost is untried.
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
| Routing profile through prefill (per-token counts) | the profile now reflects routing frequency; first attempt counted each expert once per group and made decode worse (41 misses/step) |

Prefill on prompts of a few thousand tokens is SSD-bound: ~6 s of expert reads once per group (overlapped with the
~1.8 ms/token of GPU work since staging). Decode runs 13-21 tok/s; misses per step vary 15-40 with content and history.

### Tried and rejected

| Idea | Result |
| --- | --- |
| Attention on fp32 simdgroup matrices | slower than the scalar kernel: ablation showed the multiply-accumulates themselves bound it (~0.9 TFLOP/s; they do not reach the matrix units) |
| Staged prefill reads for GLM-5.3 (87% of experts resident) | prefill -11% (429 -> 382 tok/s at 4K): little to read, and the ring's borrowed memory evicts resident experts. Staging is on only below 70% residency. It did cut decode misses in the 40 steps after the prompt from 32.6 to 26.2 a step (prefill misses no longer evict decode's resident set) |
| One 1024-thread threadgroup per token for the attention output (no repeated reads of probabilities and keys) | no change: redundant reads were not the cost |
| Per-row key descriptors with vector loads | no change for the output kernel (helped the score kernel slightly) |
| More query tokens per attention dispatch (32 -> 256) | no change: occupancy was not the cost |

### Open

- Prefill reads (~6 s a group, now overlapped with the GPU work): more resident experts or fewer bytes per expert.
- Decode misses (15-40 a step).
- DSpark (5 drafts verified in one 6-token step) not built yet.
