# Performance work: what landed, what didn't

Measured on an M5 Max (128 GB, 40-core GPU) unless noted. Numbers are from interleaved A/B runs where it mattered:
identical runs vary up to ~12 ms/step over long benchmark stretches, and the machine drifted ~15% slower over one
evening in both arms, so single runs or runs hours apart are not comparable.

## GLM-5.3-Flash (Maya-S-v2 IQ2_XXS, 96.5 GB; experts ~86 GB)

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
| MoE layers whose experts are all resident (every GLM layer on 128 GB) skip the CPU round trip and the miss pass: `moe_route` counts the selections on the GPU, passed to the arena after the step (usage, eviction scores, `--learn` unchanged) | 25.7 -> 25.1 ms/token (two A/B pairs), speculative step 32.9 -> 32.2 ms; tokens identical |
| Batched GEMVs for same-shape matrices on one input, weights adjacent: shared-expert and dense gate/up, KDA q/k/v (one token), KDA f_a/g_a | 25.1 -> 24.9 ms/token; tokens identical |
| The step's command buffer committed after layer 0 and the rest encoded into a second one while the GPU runs (`commit_ahead`; the CPU's ~0.3 ms of encoding came before the GPU started) | 24.7 -> 24.4 ms/token; tokens identical |
| hc_step_a, hc_step_b and the rms norm after them in one dispatch (`hc_step_norm`: the token's last threadgroup, found by a device-scope fence and a counter, mixes the streams and norms x) | 180 fewer dispatches a token: -0.25 ms; outputs bit-identical (e2e tests unchanged) |
| The router GEMV routing in its last threadgroup (`mako_mul_mv_f32_f32_4_route`); conv, gate and the q/k l2 norm in one kernel per head (`kda_prep`) | 110 fewer dispatches: no measurable change (+-0.1 ms). A dispatch boundary costs ~1 us; what a small kernel costs is its chain of dependent memory round trips (~0.8 us each), which a fused kernel keeps |
| Prefill: `kda_prep`'s work with a threadgroup per (head, token) past 8 rows (`kda_prep_rows`, the conv state moved on by `kda_conv_shift`), same arithmetic. `kda_prep` had made prefill's conv section 1.48 s of a 20K-token prompt with 64 threadgroups walking each chunk's tokens, against 0.66 s for the separate kernels before it; unseen because no prefill was timed | 20K tokens: 717-719 -> 742-743 tok/s in alternating runs (737-744 before `kda_prep`); conv 0.37 s. GLM e2e tests pass. `bench/perf_check` now flags such a change |
| `checkpoint()` reserves its 153 MB before copying the recurrent state in (as DeepSeek's does) | 5.9 -> 2.4 ms a checkpoint, the first in a process 22 -> 8 ms (alternating runs; one to three a request). Estimated "tens of ms" beforehand |
| Prefill GEMM: IQ weights in tiles of up to 32 tokens dequantized 64 columns a stage instead of 256 (12-23% faster per GEMM at 16 and 32 rows for every IQ format; from 64 rows even or slower, and Q6_K / Q8_0 not faster, so those keep 256) | 256 / 512 / 1024 new tokens after 4000, alternating, two runs each: ~755 -> ~702 ms, ~1111 -> ~1065 ms, ~1810 -> ~1780 ms; experts 585 -> 512 ms GPU at 512. GLM and DeepSeek e2e tests pass |
| Independent GEMVs of one input in one dispatch (`mako_mul_mv_jobs`: a job table of GPU addresses, each job its format's kernel over its own threadgroups, 4 simdgroups a threadgroup, every row summed as before): KDA q/k/v + f_a/g_a + beta; routed gate + up + the shared gate/up, both swiglus (`swiglu2`), routed down + shared down; MLA q_a + kv_a + the indexer's key, gate and weights, then q_b + the indexer's query | 24.8-25.3 -> 23.1 ms/token (three A/B pairs, 43.3 tok/s); tokens identical, e2e tests unchanged. ~8 us a merged-away dispatch: one ramp-up and tail for the group, the IQ2_XXS fixed cost once, the small Q6_K and Q8_0 GEMVs hidden under the large ones |
| Decode GEMVs row-invariant (a token's bits the same whether it runs alone or with up to 7 others, for batching several sequences): Q6_K always through mako's 1- and 2-token kernel with multiply-add fusion off (1 token used the per-format kernel, which the 2-token kernel did not match); the short-row kernel to 8 tokens. `gemv_rows` checks every format through the plain, multi-GEMV and indexed paths | no speculative decoding 23.1 -> 23.2 ms/token, speculative ~19.8 -> ~20.05 (three and four clean runs; within their spread); e2e tests unchanged |
| The step's wait for the GPU spins on the command buffer's status, at most 250 ms, before blocking (`--spin-wait`, default on) | with drafts, every expert resident: 31.5 / 31.4 / 31.4 -> 31.3 / 31.4 / 31.4 ms/step (three A/B pairs, 128 tokens after a 557-token prompt): nothing measurable. perf_check decode -0.6%, prefill -1.4% for the whole 2026-11-08 set of changes against 73edbb1 |

Where a short-prompt decode step goes (2026-10-26, 789-token prompt, 128 tokens, every expert resident, 25.0
ms/token; ~9.6 GB read a token, ~17.4 ms at 553 GB/s). Each line is the step time saved by leaving those dispatches
out (`MAKO_REPEAT=label:0`; output meaningless, timing valid; noise ~0.2 ms):

| Dispatches | Per token | Saved | Rate |
| --- | --- | --- | --- |
| KDA q/k/v, Q6_K 4096x8192 | 102 | 5.3 ms | ~530 GB/s |
| Routed gate/up, IQ2_XXS 4096x2048 x8 | 84 | 3.8 ms | ~384 GB/s |
| Routed down, IQ2_S 2048x4096 x8 / IQ3_XXS x8 | 34 / 8 | 2.0 / 0.5 ms | ~364 GB/s |
| KDA output, Q6_K 8192x4096 | 34 | 2.1 ms | ~444 GB/s |
| Shared gate/up, Q6_K 4096x2048 | 84 | 1.5 ms | ~385 GB/s |
| MLA output, Q6_K 16384x4096 | 11 | 1.3 ms | ~466 GB/s |
| Output head, Q6_K 4096x154880 | 1 | 0.9 ms | ~600 GB/s |
| Shared down, Q6_K 2048x4096 | 42 | 0.8 ms | ~362 GB/s |
| hc_step_a + hc_step_b | 180 | 1.4 ms | |
| moe_route, KDA scan, Q8_0 4096x128, rms_norm | | 0.7, 0.6, 0.6, 0.3 ms | |
| swiglu, kda_gate, kda_conv, l2_norm, rms_norm_gate, MLA projections and attention | | <= 0.5 ms each | |

The weights alone would take ~17.4 ms (~16.3 at the 589 GB/s a free-running chain of f16 GEMVs reads at). Leaving
out every small helper kernel at once (`MAKO_REPEAT=hc_step,rms_norm,swiglu,kda_conv,kda_gate,l2_norm,rms_norm_gate,
moe_route,moe_combine,hc_init,hc_mean:0`) saves 3.3-3.5 ms, and the Q8_0 projections another 1.0-1.3 ms: ~4.5 ms in
~700 dispatches of 1-15 us, each a chain of dependent memory round trips at ~0.8 us (`bench/chain_bench`: a chain of
tiny dependent dispatches costs 0.9 us a link in a serial encoder; a small kernel with 4 dependent loads before a 32
MB GEMV adds ~4 us over free-running). Fusing kernels removes the ~1 us boundary, not the chain. Isolated
(`gemv_bench`, time against matrices a dispatch): Q6_K streams at ~580 GB/s with ~3 us fixed a dispatch, IQ2_XXS
527 GB/s + ~9 us, IQ2_S 463 GB/s + ~7 us, IQ3_XXS 580 GB/s + ~7 us, Q8_0 4096x128 575 GB/s + ~2 us, the F32 router
475 GB/s; so the big GEMVs hold ~1.5-2 ms (the IQ2_S kernel, the IQ2_XXS fixed cost, independent GEMVs in separate
dispatches), the small kernels ~4.5 ms, the dense weights the rest. With the multi-GEMV dispatches: 23.1 ms/token.

Decode at 15K context after these, before every expert was resident (the cache margin, DeepSeek's table): ~46-54 ms
per speculative step (~1.95 tokens): GPU kernels ~36 ms, expert reads 10-19 ms (not overlapped), drafting ~2.3 ms.
Since then a two-token step costs ~34 ms after 21.9K tokens (MTP drafts, Open). The big GEMVs already run at ~400
GB/s of the 553 GB/s measured (614 GB/s specified); the estimated bandwidth floor is ~20 ms/step.

Where the step goes (2026-11-08, 963-token prompt, 128 tokens, every expert resident, 23.1 ms/token, GPU 22.9; M5
Max, 40 cores). The tensors read a token: dense 6.8 GB (Q6_K 6.26, F32 0.34, Q8_0 0.20) and 8 of 288 experts in 42
layers 2.4 GB, 9.52 GB in all: 16.1 ms at the 590 GB/s the Q8_0, Q6_K and F32 GEMVs stream at in `gemv_bench`.
Leaving dispatch groups out (`MAKO_REPEAT=<label substrings>:0`, which also runs the step without the multi-GEMV
dispatches and the fused router: 24.3 ms against 22.9 merged; one thermal-gated run each, noise ~0.2 ms):

| Left out | Saved | Its bytes at 590 GB/s |
| --- | ---: | ---: |
| routed experts (`expert`) | 6.2 ms | 4.05 ms |
| KDA q/k/v (`gemv Q6_K 4096x8192`) | 5.2 | 4.75 |
| MLA projections and attention, 11 layers | 2.9 | 1.92 |
| KDA output (`gemv Q6_K 8192x4096`) | 2.2 | 1.58 |
| shared experts (`gemv Q6_K 4096x2048,gemv Q6_K 2048x4096`) | 2.2 | 1.47 |
| hyper-connection steps (`hc_step`, 90 dispatches) | 1.9 | 0.23 |
| routing (`moe_route`, 42) | 1.5 | |
| KDA scan, prep, conv, gate, norms | 1.4 | |
| small Q8_0 projections (`gemv Q8_0`) | 1.1 | 0.36 |
| output head | 1.1 | 0.88 |
| rms_norm, swiglu, moe_combine | 0.2 | |
| every small kernel above at once | 4.3 | |

The groups overlap (their sum is more than the step), so the all-at-once run gives the small kernels' cost: 4.3 ms;
the GEMVs' excess over the memory rate is the other ~3.9 (~2.5 with the merged dispatches). In isolation
(`bench/hc_bench`, `bench/route_bench`) the hyper-connection step costs 12.2 us, 14.8 between 34 MB GEMVs (32
threadgroups is the best of 8-64; its phase A 5.7 us, the one-threadgroup phase B 5.6), and the route dispatch adds
11-13 us over the expert dispatch that follows it, fused into the router GEMV or not; the in-step costs above are 20
and 36 us a dispatch, the rest the dependent chain. Expert dispatches at their real size (`gemv_bench`, 8 experts a
dispatch): IQ2_XXS 410 GB/s, IQ2_S 401, IQ3_XXS 499, against 533 / 464 / 580 streaming at 64 experts and 588 for
Q8_0; the fixed cost a dispatch from the two points 9.5 / 7.7 / 6.7 us against Q8_0's 4.7. Reading the grids from
constant memory instead of filling a threadgroup table leaves IQ2_XXS and IQ3_XXS at the same rates, so the fixed
cost is not the table fill.

Dense re-quantization at load (`mako_run --dense-quant q5_k|q4_k`, or `MAKO_DENSE_QUANT`; every Q6_K and Q8_0
dense matrix, from the stored Q6_K rows, unweighted), 512-token parity prompt against the reference (`regress_test
parity`; the stored weights' decode numbers are the passing test's):

| Dense | ms/token | KL prefill / decode | Top-1 prefill / decode |
| --- | ---: | ---: | ---: |
| Q6_K as stored (7.1 GB) | 24.5 | 0.0091 / ~0.012 | 95.5% / 94.5% |
| Q5_K (6.0 GB) | 22.5 | 0.0154 / 0.0230 | 93.6% / 90.6% |
| Q4_K (5.0 GB) | 20.6 | 0.0251 / 0.0368 | 92.2% / 85.9% |

Loading takes ~24 s longer (the CPU re-quantizes 7 GB). A build from the original weights, or re-quantizing only
the KDA q/k/v projections (2.8 GB of the 7.1), would cost less quality; neither is measured.

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
| Concurrent encoders for decode | without barriers (wrong output) 25.2 -> 18.4 ms/token, the most overlap could give; with a barrier before each dispatch 25.9 ms, no better than serial order. Only independent GEMVs can overlap, and batching those into one dispatch (landed) gained 0.2 ms: little per-dependency overhead is left |
| IQ2_XXS expert GEMV rewrites (`gemv_bench 4096 2048 64 iq2_xxs 8`, 411 GB/s): grid as float and sign masks xor'ed into x; int8 signed weights built from the grid by byte masks; 2-16 rows per simdgroup, 1-16 simdgroups | 251, 389 GB/s; 310-420 GB/s. The kernel is not the limit: the same kernel reaches 467 / 497 / 515 GB/s at 16 / 32 / 64 experts a dispatch (Q8_0 ~580 at any size), so a 17 MB dispatch loses ~9 us to ramp-up and tail |
| Routing, shared expert and routed experts of a resident MoE layer in one concurrent encoder (barriers only where data flows) | 24.9 -> 24.9 ms/token: overlapping independent GEMVs does not recover that ramp-up and tail |
| `MAKO_REPEAT=label:2` for in-place costs of weights under ~50 MB | the repeat reads weights the cache still holds: Q6_K 4096x8192 came out at ~880 GB/s. `label:0` (leave out) instead |
| Per-kernel telemetry to find slow kernels | timestamps around each dispatch inflate small kernels several-fold (one "slow" shared-expert GEMV was 93 us measured, 13 us real); repeat-in-place costs (`MAKO_REPEAT`) replaced it |
| Software dependency tracking instead of encoder barriers: every dispatch in one concurrent encoder, each threadgroup waiting on a device counter the previous dispatch's threadgroups bump after a device-scope fence, the weights read into registers before the wait (`bench/chain_bench`) | 64 chained f16 GEMVs of 32 MB: 65 -> 63 us a link (free-running 57); a chain alternating a latency-bound small kernel and a GEMV: no change (one counter a dispatch: 20% slower from the atomics' contention; 32 counters: equal to serial). Outputs stayed bit-identical, no wait gave up. Not worth the hazard |
| Sparse attention in one kernel for prefill (2026-10-28): per (token, 32 heads) a flash-style pass over the token's list, 16 gathered latent rows at a time into threadgroup memory serving both products, running softmax, output in registers; then with the list split over 4-16 threadgroups and a combine kernel | 4096 tokens at 16K context: 1.42-1.62 s (one range), 1.32 / 1.46 / 1.78 s (4 / 8 / 16 ranges) against 0.55-0.58 s for the three kernels (scores, softmax, output); KL against the reference unchanged (0.0069-0.0072). The 512-wide latent leaves room for 16 rows a step in 32 KB of threadgroup memory: ~129 small steps a token, each two small matmuls, a score round trip through threadgroup memory, a rescale of the 32x512 accumulator and three barriers; splitting the output width instead would recompute the scores per slice or hold 64x512 floats. Reverted |
| Dense weights below Q6_K | Q5_K drops decode top-1 to ~90% (the dense-quant table above); declined |

### Open

- Decode misses rising after a long prompt: GLM-5.3 holds every expert on 128 GB now, so only on smaller Macs (~27
  vs ~15 a step for the first 40 steps, measured before the larger cache). DeepSeek: its Open list.
- Expert GEMMs at few rows per expert in short prefills (2026-10-28). The IQ tiles of up to 32 tokens now stage 64
  weight columns at a time instead of 256 (landed above): 16-row gate + up 3.5 -> 2.8 ms, 32-row 4.8 -> 3.9 ms
  (`gemm_bench 128 T 4096 2048`). Still per pass over the weights and well under their read bound (~200 GB/s of
  IQ2_XXS at 16 rows); smaller tiles lose (two 16-row passes cost more than one 32-row), and the multi-token GEMVs
  grow linearly past 2 tokens (~16 ms per 16 tokens).
  Future: the weights dequantized in registers (as the decode GEMVs do, ~1 T weights/s at 1-2 tokens) and fed to
  simdgroup matrices without the threadgroup staging. A 512-token GLM turn spends ~512 ms of GPU on experts against
  ~190 ms to read them; at ~350-400 GB/s instead of ~200 the estimate is 100-200 ms off the turn (10-18%), less at
  256 tokens, nothing for long prompts (64-row and larger tiles are near peak), little for DeepSeek (its short turns
  wait on SSD reads). Low to medium confidence: an earlier IQ2_XXS rewrite did not beat the existing kernels, and
  the fused sparse attention, estimated 4-5% faster, came out 2.3-3.2x slower. It changes summation order, which GLM's
  near-bit-exact e2e checks react to. First a standalone kernel in `bench/gemm_bench` against the 16- and 32-row
  tiles (2.8 / 3.9 ms for 128 experts' gate and up), before any engine change.
- Prefill at 20K tokens (2026-10-27, `prefill_sweep --parts`, ~745 tok/s, 27.6 s GPU): experts 9.7 s, KDA 8.4 s
  (q/k/v 3.1, scan 1.3, output 1.0, gates 1.0, conv 0.4), MLA 6.3 s, FFN 2.8 s, hc and norm 1.4 s. By expected gain,
  none measured:
  - Sparse attention in one kernel: tried and rejected (2026-10-28, table above): 2.3-3.2x slower than the three
    kernels at 4096 tokens. Another engine runs it as one kernel through a hard-coded M5 layout of the cooperative
    tensors' elements; how it fits the 512-wide rows was not examined. DeepSeek-V4.1's attention (64 heads, 512
    wide) would meet the same limit.
  - Expert GEMMs at ~36 TFLOP/s against ~60 for the same shapes in `bench/gemm_bench`: the dequantization does not
    overlap the multiply (~21%, above); double-buffering it against `matmul2d`. ~1-2 s.
  - The stream step and norm fused for prefill as decode has them (`hc_step_norm`): hc and norm is ~2x what its
    memory traffic needs (estimated). ~0.5 s.
  - The shared expert and dense FFN at ~22 TFLOP/s (2.8 s).
- MTP drafts: a two-token step costs ~31 ms at short contexts and ~34 ms after 21.9K tokens, against ~23.5 for one
  token, so drafting pays above ~33% / ~45% acceptance. After a 21.9K-token code prompt, 97% were kept: 18.5 ms a
  token against 23.5 without (2026-10-28). The earlier "33 against 24 ms/token" was the telemetry summary's
  per-step time read as per token (fixed: it now reports steps and tokens run a step). A gate that stops drafting
  below break-even would matter only for requests whose acceptance falls under a third; the server logs each
  request's share of drafts kept, so look there before building one.
- Decode at ~23.2 ms/token, the weights ~16.3 ms of it. Not built, by expected gain:
  - More independent GEMVs in one dispatch (`mako_mul_mv_jobs`): f_b + g_b (rows of 128: the job kernel needs the
    short-row kernel as a job type), the MLA per-head k_b and v_b with their neighbours, hc_mean + the output norm.
    ~0.3 ms.
  - An IQ2_S decode kernel at the IQ3_XXS kernel's rate (463 -> ~580 GB/s on the 34 down projections): ~0.5 ms;
    the IQ3_XXS rewrite was not bit-identical (compiler fusion), so expect a re-baseline.
  - Sampling on the GPU with the next step encoded and committed before this one ends (the token read from a
    buffer, the embedding row gathered on the GPU): the remaining CPU gap between steps, ~0.2-0.3 ms.
  - Trimming dependent memory round trips inside the small kernels (loads issued before the waits they do not
    depend on, e.g. hc_step_norm's phase B reading R before the parts): ~0.8 us a round trip, ~1 ms across the
    ~400 small dispatches at most. `bench/hc_bench` (2026-11-08): hc_step_norm 12.2 us alone, 14.8 between 34 MB
    GEMVs, ~20 in the step; phase A 5.7 us and the one-threadgroup phase B 5.6, so a shorter chain in B is worth
    ~5 us of 90 dispatches, ~0.5 ms.
  - The IQ2_XXS kernel's ~9 us fixed cost a dispatch (the grid table into threadgroup memory before the first
    weight loads): issue the first weight loads before the table build. ~0.3 ms.
  - Several sequences a step (continuous batching in the server) for parallel agent requests: throughput, not
    latency; the kernels take T <= 8 rows already, the KDA/MLA state and the session API do not.
  - An exact top-k head from a low-bit sketch (bounded error per row from the quantization residual and ||x||)
    with exact rescoring of the candidates: ~0.3-0.6 ms of the head's 0.9, greedy/top-k sampling only.

## DeepSeek-V4.1-Flash-1017 (Maya-S IQ2_XXS, 178 GB + Engram tables; ~55% of experts resident on 128 GB)

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
| Prefill skips work the result does not depend on: from layer 20 (the last KV source) on, a long prompt's earlier rows only add that layer's compressed rows and index keys, and each later layer starts one window later | 8K tokens 14.3 -> 10.5 s, 16K 31.2 -> 21.5 s, 24K 52.9 -> 30.4 s; 4K unchanged (SSD-bound); the last 1024 positions' logits and later decode identical against the reference (ds41_long) |
| Decode MoE layers with misses: the resident experts' gate and up with the shared gate/up in one dispatch (`mako_mul_mv_jobs`), both swiglus in one, the resident down with the shared down in one, all before the miss pass (the CPU reads the misses meanwhile, as before) | 42.4 -> 41.9 ms/step with `--telemetry layers`, 41.7 -> 41.3 without (9.3 misses a step in both arms; three A/B pairs each, 128 tokens, short story prompt); tokens identical |
| Decode: the step's command buffer committed after layer 0's attention (`commit_ahead`; the next commits stay every 4 layers), so the GPU starts while the rest is encoded | 41.4 -> 41.2 ms/step median (41.8/41.3/41.4 vs 41.5/41.2/41.2, three A/B pairs without telemetry, 9.3 misses a step in both arms; the routing wait unchanged); tokens identical. Under `--telemetry layers` no commit is made early |
| Every decode MoE layer on the multi-GEMV path: an IQ3_S job type (`mako_iq3_s_impl`: the per-format row loop, the grid table loaded for any number of simdgroups) for the IQ3_S down experts of layers 0-3 and 36-39, and the shared gate and up as two jobs (no adjacency needed; layer 16's are not adjacent) | 40 of 40 layers merged (was 31); 41.2/41.2/41.1 -> 41.1/41.0/41.2 ms/step (three A/B pairs without telemetry, 9.2-9.3 misses a step in both arms): within the drift, not slower. Logits of 110 decode positions bit-identical, tokens identical, DS and GLM e2e tests unchanged |
| Decode hyper-connection pre-chain in two dispatches a block instead of five (`hc_lagged_norm`: 6 threadgroups each norm the flat stream themselves and compute 4 of the 24 mixing rows as the F32 GEMV sums them, 20 collapse the streams with the previous block's pre, the last of those to finish norms x; then `hc_mix`). hc_mix inside the fused kernel rounded differently from its own dispatch (same inputs, same IR; the GPU compiler's code differs once inlined), so it stays separate | parts: attn + ffn hc and norm 0.95 + 0.94 -> 0.77 + 0.75 ms a step; 41.3/41.1/41.3 -> 40.7/40.7/41.1 ms/step (three A/B pairs without telemetry, 9.2-9.3 misses a step in both arms). Logits of 110 decode positions bit-identical, tokens identical |
| ISTA builds on the merged decode MoE path: job types for Q5_K (their shared experts, which kept every layer off it), IQ2_XS (`mako_iq2_xs_impl`: the per-format row loop, its tables loaded for any number of simdgroups) and IQ1_S / IQ1_M (mako's own row kernels) | IQ2_XS 31.3/31.0 -> 30.6/30.5 ms/step, IQ2_XXS 32.5/32.3 -> 32.1/31.9 (alternating runs, 128 steps, 6.2-6.4 misses a step in both arms); logits of 160 decode positions bit-identical on both builds. Each job matches the per-format kernel bit for bit (`gemv_rows`, plain and indexed). perf_check IQ2_XS: decode -2.5%; its part shares moved from "shared expert" (now merged away) to "miss experts", whose dispatch now starts earlier and waits for the SSD reads |
| Staged prefill reads only for groups of 2048 tokens or more (was 512); smaller groups read just the experts they route to, which stay resident. A 512-token group was assumed to route to nearly every expert; on ISTA IQ2_XS it routes to ~2000 of the ~5400 missing (57 GB staged against ~21 GB needed). The miss penalty in prefill keeps its own threshold (512), so routing is unchanged | 512 / 1024 tokens after 4000, first run (old -> new binary, the 4000 staged in both): ISTA IQ2_XS prefill 5.22 -> 3.08 s / 4.91 -> 4.31 s, Maya-S 6.55 -> 3.34 / 6.66 -> 4.88 s; the 64 steps after it 41.8 -> 35.8 ms (12.2 -> 7.5 misses) / 45.6 -> 37.2, Maya-S 52.0 -> 40.5 / 47.7 -> 43.3: the experts the prompt read are the ones its steps use. With staging off entirely, 2048 tokens were even and 4096 4% slower. GLM-5.3 holds every expert on 128 GB (no staging); on less memory it takes the same threshold, unmeasured |
| The step's wait for the GPU spins on the command buffer's status, at most 250 ms, before blocking (`--spin-wait`, default on) | 46.7 / 46.5 / 46.4 -> 46.3 / 46.3 / 46.3 ms a step (three A/B pairs, 128 steps after a 300-token prompt, 12.9 misses a step in both arms): 0.1-0.4 ms a step, for a core held during the step. perf_check decode -1.3%, prefill -1.1% for the whole 2026-11-08 set of changes against 73edbb1 |
| Drafts from repeats of the context (`--lookup-draft`, default 4; `gen::LookupDrafter`) | a prompt that repeats its start, 96 greedy tokens: 4921 -> 3068 ms, 84 of 96 tokens as context drafts, all kept, tokens identical (ds41_lookup; an upper bound: the reply is a copy of the context) |
| Expert misses and reads ahead through the page cache (`--page-cache`, default on: a second descriptor without F_NOCACHE; the arena's fill and a prompt's staging keep bypassing it), so an expert evicted and missed again soon comes back from RAM | 93.4 GB arena, 1000 steps after a staged 2048-token tail, 7.2 misses a step (two thermal-gated A/B pairs, the arms' order alternated): 37.7 / 37.4 -> 37.1 / 37.2 ms a step, reading 8.2 / 8.1 -> 7.6 / 7.7 ms a step. 74.5 GB arena, 300 steps, 17.7 misses a step (one pair, not thermal-gated, the cached arm second): 50.7 -> 44.5 ms, reading 20.1 -> 14.6. Of the misses 47% are experts evicted fewer than 4096 evictions (~43 GB) earlier and 48% experts never resident in the process (the arena's re-miss histogram, `ds_run --tail-decode` prints it), but what the cache serves is bounded by the RAM the wired arena leaves it: during the 93.4 GB runs vm_stat showed 2.3 GB free, 100-107 GB wired and a file-backed pool of ~23 GB that did not grow while ~58 GB of expert reads passed through it (2-3% of the bytes came back at memory speed), so at this machine's default sizing the gain is at the noise floor; it pays where RAM is left over (a smaller `--arena-gb`, memory lent and returned, Macs below 96 GB with the 40% margin) |

Prefill on prompts of a few thousand tokens is SSD-bound: ~6 s of expert reads once per group (overlapped with the
~1.8 ms/token of GPU work since staging). Decode first ran 13-21 tok/s (misses per step 15-40 with content and history);
the sections below take it to ~25 by default.

### Tried and rejected

| Idea | Result |
| --- | --- |
| Attention on fp32 simdgroup matrices | slower than the scalar kernel: ablation showed the multiply-accumulates themselves bound it (~0.9 TFLOP/s; they do not reach the matrix units) |
| Staged prefill reads for GLM-5.3 with a 3 GB ring | prefill -11% (429 -> 382 tok/s at 4K): the ring's borrowed memory evicts resident experts. Sized to two layers' missing experts instead (~0.55 GB): prefill unchanged, decode after a 4K prompt ~4.5% faster (first 40 steps 32.1 -> 26.6 misses a step: prefill no longer evicts the experts steps use). Landed that way |
| Marking slots filled by prefill misses as preferred victims (no staging) | no change (38.6 vs 38.7 misses a step after the prompt): in a group nearly every resident is in use, so each miss still evicts one of decode's experts before any marked slot frees up |
| One 1024-thread threadgroup per token for the attention output (no repeated reads of probabilities and keys) | no change: redundant reads were not the cost |
| Per-row key descriptors with vector loads | no change for the output kernel (helped the score kernel slightly) |
| More query tokens per attention dispatch (32 -> 256) | no change: occupancy was not the cost |
| Decode attention GEMVs in multi-GEMV dispatches: q_a + kv (+ the compressor's kv and gate, + the indexer's weights) on the attention input, q_b (+ the indexer's query) on the normed q latent | tokens identical, no measurable gain: short prompt 42.2 vs 42.2 ms/step with `--telemetry layers` (six A/B pairs, 9.3 misses a step) and 41.3 vs 41.2 without (three pairs), 3000-token prompt 49.9 vs 50.1 (three pairs, 16.2 misses). Q8_0 GEMVs have ~2-3 us fixed a dispatch (IQ2_XXS ~9), so ~40 merged-away dispatches are worth ~0.1 ms (inferred, not measured). Reverted |
| After a staged prefill, read ahead the experts its last 8 rows selected and the arena lacks when the first step starts (through `ExpertArena::prefetch`, a slot evicted per expert) | the 64 steps after a staged 2048-token tail on a 6144-token prompt: 62.6 / 62.9 -> 68.8 / 68.7 ms a step with 25.3 -> 25.1 misses a step (two A/B pairs, 74.5 GB arena, 45 s pause before each run): the reads ahead compete with the steps' own miss reads and the steps rarely want those experts. GLM-5.3 under `--simulate-ram 64` (8.4 GB arena, 7.3K-token prompt): misses identical (217.9 a token), 216 vs 216 ms. Reverted |

### Open

- Prefill reads (~6 s a group, now overlapped with the GPU work): more resident experts or fewer bytes per expert.
  Staged misses go to the GPU in batches of up to 32 only once every one of them has been taken from the ring, so
  one late read holds back experts that are ready; submitting the ready prefix (a minimum batch size, the order and
  the final combine unchanged) could close GPU idle gaps. First measure the gaps.
- Decode misses (~9 a step on Maya-S and 6-7 on ISTA IQ2_XS in the A/B runs above; 15-40 before read-ahead and the
  larger cache). Untried: experts pinned resident by their decode request rate (an EMA per route,
  the top share of each layer) beside the decayed score; another engine pins half of each layer for GLM-5.3 and has
  it off for this model, unmeasured. The trace replay (2026-11-08, above) puts the decayed score near the best of
  its variants; a weaker profile seed (0.3x) is the one it favours.
- Engram rows are read before the step's command buffer is made (`read_raw`, synchronous), both tables at once,
  though their consumers are layers 1 and 14. Decode's input reads are ~0.3 ms a step (telemetry), the most overlap
  can return there; prefill's are unmeasured: a group of 8192 tokens reads 393,216 rows with one `pread` each,
  repeated n-grams reading the same row again. Measure prefill's Engram read time and unique-row share (prose and
  code) before overlapping (layer 0 while layer 1's rows arrive, layer 14's later) or deduplicating.
- Reads ahead are a FIFO done one expert at a time, and `ensure()` waits for a selected expert that is still queued
  behind unselected predictions before it starts its other misses. Waits for reads in flight measured ~50-60 ms over
  200 tokens (~0.3 ms a step), the bound for moving selected jobs to the front; split queued from in-flight waits first.
- The GPU waits once per layer for all its misses. Gating each miss's gate/up on its own bytes and its down on the
  rest would overlap compute with the remaining reads, but a miss's compute is tens of us against ~1 ms of reading:
  well under 1 ms a step (estimated).
- Resident-expert GEMVs (planned 2026-10-26; measured 2026-11-08 with `gemv_bench` and `route_bench`): the event
  signal and encoder boundary after a layer's routing cost 8 us a layer, ~0.3 ms a step. At 6 experts a dispatch
  IQ2_XXS 5120x2304 runs at 404 GB/s (539 streaming at 48, 11 us fixed a dispatch), IQ3_XXS 2304x5120 at 437 (495
  streaming: the 2304-wide rows' idle lanes, 7.3 us fixed), Q8_0 at 564-580 (588 streaming, 4.7 us fixed). The
  headroom at the real sizes is ~1.7 ms a step in all, so a kernel at Q8_0's rate and fixed cost would save that
  much at most. Lanes over the 2304-wide rows without the idle pass: ~0.3 ms; not built.
- DSpark (opt-in, `--dspark DIR`): greedy output identical to plain decoding (verify up to 8 tokens runs the decode
  path); measured before read-ahead (2026-10-25): 19.8 -> 22.5 tok/s on code (93% of drafts kept), 23.5 -> 22.8 on
  a story (70%), its 8 GB costing the cache ~9%. Draft rounds take ~11 ms (the reference engine's ~2.4). Measured groundwork (positions 600-900 of a prose text, 94.5 GB cache): a step of 1 / 2 / 4 /
  6 tokens costs 47-62 / 63 / 101 / 155-185 ms with ~12 misses per token at every width (tokens in a step share
  almost no experts; wider steps still pay off because the dense weights are read once and one step's misses are
  read in parallel). The reference engine's DSpark: on code 97% of drafts kept, 4.6 tokens a round (-> ~1.5-1.65x
  here); on a story 1.34 tokens a round, drafts offered in 113 of 299 rounds (-> ~0-8% here, before its 8 GB).

### Where a decode step goes (2026-10-25, short story prompt, Q8_0 dense, 94.4 GB expert cache)

`ds_run --gen 200 --no-spec --telemetry layers|parts|kernels`. A step is ~41 ms, GPU-bound (inputs and encoding <1 ms):

| | ms/step | Note |
| --- | ---: | --- |
| GPU waiting for missing experts | 12.2 | ~10.7 misses of ~10 MB; one 7-14 MB read peaks at 10.5-11.9 GB/s on this SSD (`io_bench batch`), so ~0.9-1.1 ms each |
| Resident experts (IQ2_XXS / IQ3_XXS indexed GEMV) | 7.8 | the IQ kernels run at 360-385 GB/s (Q8_0 at 566-580 on the same shapes): ~4.6 ms at bandwidth |
| Attention output (out_a, out_b) | 6.1 | 3.2 GB of Q8_0, near bandwidth |
| Attention q/kv projections | 4.8 | 2.2 GB of Q8_0 |
| Shared expert | 3.6 | 1.5 GB of Q8_0; since the multi-GEMV MoE dispatches (2026-10-26) `parts` counts it under the resident experts |
| Router + routing | 2.1 | the F32 router GEMV is ~0.6; the route kernel and the event signal the rest |
| hc and norm (attention and FFN) | 1.9 | small kernels |
| Attention core | 1.8 | |
| Miss experts, Engram, compressor, combine, hc post | 2.4 | |

Weights read per token: ~8.5 GB dense + ~2.5 GB routed experts = ~11 GB, ~19.6 ms at 560 GB/s; GPU compute is
~28-30 ms. A step issues ~1800 dispatches; a tiny dependent dispatch costs ~2 us, so ~3.6 ms is launch overhead.

2026-11-08 (300-token prompt, 128 greedy steps, `ds_run --gen 128 --no-spec --telemetry parts`, 93.4 GB cache, 41.4
ms a step, GPU 38.8, 8.2 misses): the tensors read a token are dense 8.1 GB (Q8_0 7.6; the Engram tables' rows
aside) and 6 of 384 experts in 40 layers 2.5 GB, 11.1 GB, 18.8 ms at 590 GB/s. Parts a step: resident experts 9.75
ms (4.14 GB with the shared experts: 7.0 at the rate), attention output 6.22 (3.06 GB: 5.2), attention q/kv 4.83
(2.08 GB: 3.5), miss experts 3.82, router and routing 2.44 (0.3 GB: 0.5), attention core 2.17, hc and norm 1.49,
Engram 0.64, compressor 0.28, combine and hc post 0.36. The parts sum to 32.0 against 38.8 of GPU time; the 6.8 ms
between them is the waits for the SSD. `bench/route_bench` at this model's shapes: the router GEMV 15 us, the route
15 more over the expert dispatch that follows it, the event signal after the routing 8 us a layer (the encoder
boundary itself nothing): ~0.3 ms a step for the signals, so most of the router part's 61 us a layer is the chain
and the part timestamps' inflation of small parts.

Knobs measured the same day (45 s pause before each run, the thermal state nominal before it, the arms' order
alternated). `--dense-quant q6_k`, a 6144-token prompt with a staged 2048-token tail then 300 steps: 41.8 / 42.0 ->
39.2 / 39.2 ms a step (11.0 / 11.1 -> 10.8 misses: the cache gained 0.9 GB), the 2048-token prefill unchanged, but
the 2048 tokens as steps of 8: 37.9 / 38.7 -> 40.4 / 40.7 s (+6%, the Q6_K multi-token path) and the load 10.4 ->
18-19 s; the quality cost is the table above; the default stays Q8_0. Read-ahead, 300-token prompt then 300 steps:
the defaults (`--prefetch 4 --prefetch-ahead 2`) 36.4 / 36.5 ms a step at 6.4 misses, against `--prefetch 0` 37.5 /
37.6 (8.9 misses), 2: 37.6 / 37.3 (7.6), 3: 36.9 / 36.7 (6.9), 6: 36.7 / 36.6 (7.4), `--prefetch-ahead 1` 36.9 /
37.0 (6.0: fewer misses, more waiting for reads in flight), 3: 36.9 / 36.8 (7.7). Eviction policy:
`MAKO_ARENA_TRACE=file` records every ensure, touch and read-ahead with the initial residency and scores, and
`tools/arena_replay` replays the arena's policy and variants over it. The replay's own baseline comes out 8.7 misses
a step on the short prompt's trace against 6.4 recorded (the read-ahead's asynchronous publishing is not modelled)
and 11.35 against 11.11 on the long one's; half-lives of 64-1024 steps, no decay, least recently used and reads ahead
ignored come out within -7% and +30% of it, the profile seed at 0.3x -15% / -5% on the two traces, at 3-10x +25-70%.
Nothing changed; a live A/B of the 0.3x seed is the one candidate.

Dense re-quantization at load (`ds_run --dense-quant`, 1300 positions against the reference):

| Dense | Top-1 | KL mean | Median step |
| --- | ---: | ---: | ---: |
| Q8_0 (stored) | 98.2% | 0.00557 | 46.0 ms |
| Q6_K | 97.5% | 0.00755 | 40.9 ms |
| Q5_K | 96.8% | 0.00964 | 43.5 ms |
| Q4_K | 96.9% | 0.01609 | 38.6 ms |

Better dense re-quantization: after the 6-bit (Q4_K) or int8 (Q6_K) scale codes are rounded, the neighbouring codes
are tried with the weighted error of the final rounding; and activation statistics (`ds_run --gen N --imatrix-out F`:
the mean square of each input channel of every dense matrix's input, then `--imatrix F`) weight the columns.
1300 positions against the reference (Q8_0 as stored: 98.2%, KL 0.00572):

| Dense | Top-1 | KL mean |
| --- | ---: | ---: |
| Q6_K, first quantizer | 97.5% | 0.00755 |
| Q6_K, neighbour search | 98.2% | 0.00677 |
| Q6_K, + statistics (500 generated tokens of README and code) | 97.6% | 0.00660 |
| Q4_K, first quantizer | 96.3% | 0.01636 |
| Q4_K, neighbour search | 96.5% | 0.01575 |
| Q4_K, + statistics | 96.6% | 0.01455 |

Q4_K stays at 2.5x the stored weights' KL; Q6_K costs about as much as cache-aware routing at 0.05.

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
resident, so misses are experts not used for a while; only the router predicts them.

Built as an option (`--prefetch K --prefetch-ahead D`): each MoE layer runs layer l+D's router on its FFN input,
the top K not resident are read in the background into the least scored slots (published in the slot table when
complete; a selection still in flight waits for it), and the prediction is skipped while under 35% of recent reads
ahead were selected (one layer in 16 still predicts, to keep measuring). Results are bit-identical (dsv41 test).

| Story, 200 tokens | tok/s | misses waited on |
| --- | ---: | ---: |
| off | 24.7-24.8 | 10.7 |
| 1 ahead, top 2 / 4 / 6 / 8 | 24.7 / 25.2 / 24.7 / 22.8 | 9.0 / 7.0 / 5.2 / 4.2 |
| 2 ahead, top 3 / 4 / 5 / 6 | 25.2 / 25.5 / 25.3 / 24.7 | 8.3 / 7.5 / 6.8 / 6.2 |
| 3 ahead, top 4 | 25.4 | 7.8 |
| 2 ahead, top 4, with the gate | 25.6 (+3.5%) | 7.5 |

One layer ahead the reads often had not finished (0.4-1.5 s of waits over 200 steps); two ahead, ~0.1 s. After a 16K
code prompt only 21-31% of reads ahead are used: without the gate -2.7%, with it -0.3% (21.18 -> 21.11). On by
default (top 4, two layers ahead; `--prefetch 0` turns it off).

With cache-aware routing (story, 200 tokens; the penalty changes the text, so misses are not strictly comparable):
penalty 0.05: 27.07 tok/s, with read-ahead 27.66; penalty 0.1: 27.63, with read-ahead 29.14 (3.0 misses a step,
against 24.61 and 10.7 with neither).

Decode steps committed every 4 layers, with layers whose routing is back served between encodes (`serve_moe` without
blocking), and no empty command buffer after the logits: the GPU starts while the CPU still encodes. 41.4 -> 41.0 ms a
step (+1%, two A/B pairs); wall time is now ~0.5 ms over GPU time.

A second copy of the experts on an external drive (Thunderbolt 3 enclosure, PCIe 3.0 x4 drive): one 7-14 MB read from
it alone runs at 2.3-2.4 GB/s (`io_bench batch`). Split between the internal SSD and it (`io_bench pair`, chunks of
1 MiB): 10 MB spans 11.1 GB/s from the internal SSD alone, 11.7 with 10% from the external one, 8.9 with 20%; 14 MB
11.6 -> 12.4 with 15%. The drives do not slow each other, but beyond one chunk in ten the external share is the
slowest: ~5% off miss time, not built. A drive at ~6-7 GB/s (PCIe 4.0 over Thunderbolt 5) would move the best split
to ~35-40%: to measure with `io_bench pair` when one is available.

Accuracy at long context against the reference (a 24K-token prompt of mako's sources; `ds_run --ref-from`,
`--steps-from`; perplexity of the prompt's own next tokens, mako / reference):

| Positions | Candidate stage | Top-1 | KL mean | Perplexity |
| --- | --- | ---: | ---: | ---: |
| 14,848-15,871 (prefill) | off | 98.1% | 0.0064 | 1.2919 / 1.2956 |
| 23,552-24,575 (prefill) | on | 96.9% | 0.0102 | 1.5320 / 1.5351 |
| 23,552-24,575 (decode, after a 23,552-token prefill) | on | 97.3% | 0.0093 | 1.5440 / 1.5351 |

With the candidate stage (layer 20 keeping 2048 blocks of 8 for the later index layers) KL rises ~1.6x, perplexity
does not: its block choice is a hard cut that small numeric differences move; the selection logic matches the
reference's (max-pooled block scores, newest block always kept). Perplexity on 1300 tokens (prose / code), the
defaults then miss penalty 0.05, 0.1, 0.2, Q6_K + statistics, penalty 0.1 + ablation: 1.871 / 2.903, then 1.863 /
2.893, 1.860 / 2.900, 1.863 / 2.906, 1.868 / 2.919, 1.851 / 2.887 (the reference: 1.877 on the prose). The speed
options' extra KL against the reference does not show up as worse prediction of the text.

Short prompt additions (a conversation's next turn): after an 8K-token context, 16 / 32 new tokens took 558 / 891 ms
through prefill, 463 / 798 of it misses (~30 a token, against ~10 in steady decoding; with the experts resident, 8
tokens take ~110 ms). Running them as decode steps of 8 was slower (698 / 1211 ms, the same misses). Prefill groups
too small to stage now route with the miss penalty: at 0.1, 428 / 626 ms (312 / 525 misses). `ds_run --tail N`.

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

## DeepSeek-V4.1-Flash-1017: other model files (2026-10-25)

ISTA's GSQ-RCO builds against Maya-S on the 128 GB M5 Max (each with its routing profile). Decode: 128 tokens after a
300-token prose prompt. Perplexity: 1300-token prose and code texts, token by token (the reference engine, on Maya-S:
1.877 on the prose).

| Build | Size (backbone) | Experts in memory | Decode | Prose | Code |
| --- | ---: | ---: | ---: | ---: | ---: |
| Maya-S | 178 GB | ~55% | 24.2 tok/s | 1.871 | 2.884 |
| Maya-S, `--miss-penalty 0.1` | | | 28.3 tok/s | 1.860 | 2.905 |
| ISTA IQ2_XXS | 138 GB | ~73% | 33.1 tok/s | 1.935 | 3.025 |
| ISTA IQ2_XS | 155 GB | ~66% | 31.4 tok/s | 1.873 | 2.903 |
| ISTA IQ2_XS, `--miss-penalty 0.1` | | | 34.0 tok/s | 1.875 | 2.901 |

IQ2_XS with the penalty decodes 20% faster than Maya-S with it at the same perplexity; the penalty halves its misses
(9.5 -> 4.8 a step) at no measurable cost. ISTA's Engram tables (IQ4_NL) and Maya-S's (Q6_K) give the same results
with either backbone. Against the reference's logits (Maya-S's weights, so a measure of difference, not of quality):
IQ2_XS top-1 92.5% / KL 0.068, IQ2_XXS 91% / 0.103. Two texts; the differences in perplexity are within ~1% except
IQ2_XXS's.

mako's own IQ1_S / IQ1_M decode kernels (IQ1_M is in a third of IQ2_XXS's expert layers) are 6-13% faster per expert
dispatch than the per-format ones, but decode is unchanged within noise: they were ~1.5% of a step, and those formats
are arithmetic-bound. Q5_K (ISTA's dense weights) already runs at 430-540 GB/s against Q8_0's 450-570, so a kernel of
mako's own would gain ~1 ms a step at most; not written. The per-kernel timing mode serializes the GPU and inflated
both estimates beforehand; isolated benchmarks (`gemv_bench`) are the better guide.

## Both models: open (2026-10-27)

- Conversation switches copy whole contexts: parking copies the recurrent state and every cached position out, and
  the next switch copies them back. Measured for GLM-5.3 at 32K positions (2026-10-28): 0.78 GB, `save_positions`
  14 ms (67 ms when its allocation is fresh), `load_positions` 11 ms, two 2.4 ms checkpoints: ~30 ms a switch, ~3%
  of the time when two conversations alternate at the default 1 s slices. An uninitialized buffer instead of a
  zeroed vector and the copy on 4 threads would make it ~20 ms (copies of the same size: 13.6 -> 10.7 -> 8.0 ms);
  not done for ~1%. Per-session GPU buffers or a shared prefix would remove the copies but cost memory.
- Several sequences a step (continuous batching) for several agents at once: dense weights read once for all of
  them, experts by routing overlap; the row-invariant GEMVs keep each sequence's result independent of the batch.
  The largest throughput gain and the largest change (KDA/MLA state per sequence, the session API, the scheduler).
- `bench/perf_check`'s default run covers a 4096-token prompt; its prompt file now holds ~48K tokens, so a long-context
  baseline (`--prompt 20000 --baseline FILE --save`) can be kept beside it, for regressions that show only there (the
  `kda_prep` one was ~5% of a 20K prefill). None is saved yet.

## Both models: the first GPU work after a few seconds idle (2026-10-26)

A conversation's next turn with a handful of new tokens took ~0.6 s in a server session (GLM-5.3: 7 new tokens 568 ms,
22 in 665 ms), against ~0.15-0.25 s back to back. The engine's own prefill of those tokens is not it (`prefill_sweep`
at a 4K context: 8 tokens 130 ms, 16 in 180 ms). Timing each part of a turn showed the first GPU work of the turn (for
GLM the next-token block's seeding, normally 1.5-2.5 ms) taking ~350-370 ms when the server had been idle for 3 s or
more (10 s and 30 s the same; 1 s not). An empty command buffer is instant then, and the GPU executes the seeding's
command buffer in ~346 ms: the cost is the first pass that reads the expert arena's buffer (88.7 GB, one allocation).
A pass that only touches that buffer took 354-361 ms after a 10 s pause (0.2 ms back to back), and the seeding after
it 14 ms. It is the driver cost startup already pays once (the warm-up pass after the arena is built), paid again
after idle; DeepSeek uses the same arena code.

Residency sets (`MTLResidencySet` with the arena's buffers) did not help:

| Variant | Seeding after a 10 s pause | Requests |
| --- | ---: | --- |
| none | ~350 ms | normal |
| `requestResidency` only (set not attached to the queue) | ~355 ms | normal |
| set attached to the command queue, with or without `requestResidency`, with or without the arena's `mlock` | - | the first request timed out (> 90 s) |

A keepalive removes most of it (2026-11-08, `bench/idle_bench`, GLM-5.3, the step after 10 s idle, two runs each).
With a 75 GB arena and a 27 ms step: nothing 373-381 ms; a kernel a second reading one word per 1 MiB of the resident
weights and the arena's buffers 47-49 ms (the step after it 24-29); per 16 KiB 53-58; per 64 KiB 50, once 244. With a
27 GB arena (another process held 34 GB): nothing 281 / 294, an empty one-threadgroup kernel a second 279 / 305 (it is
not the queue that goes cold), per 1 MiB 78 / 80 against the warm step's 79. The kernel ran 0.3 ms of GPU a second for
24 GB. `mako_serve` runs it while no request is generating (`--no-keepalive` turns it off;
`gpu::Runtime::set_keepalive`).

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
