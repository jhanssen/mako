# Model architectures, as mako implements them

What each layer computes, in the terms mako's code uses. The LLMs' shapes are from the Maya-S GGUFs' metadata and
tensors, the image models' from their configs; semantics were checked against the reference implementations' outputs
(see `tests/data/README.md` and the `image` tests).

## Shared by both

- **Hyper-connections (mHC)**: the residual is 4 streams of `n_embd`. Around each sublayer (attention, FFN):
  `hc_*_fn` on the RMS-normed flattened streams gives `pre` (4, sigmoid), `post` (4, sigmoid) and a 4×4 `comb`
  (Sinkhorn-normalized, 20 iterations); the sublayer's input is `Σ pre_i · h_i`, and the streams become
  `comb · h + post · out`.
- **MoE FFN**: routed experts plus one shared expert, SwiGLU with the gate clamped above at 10 and up clamped to
  ±10; expert selection on score + a per-expert bias (`exp_probs_b`), weights from the unbiased scores.
- **Experts on SSD**: routed experts are most of the bytes (~86 GB GLM, ~170 GB DeepSeek) and live in mako's
  expert cache; the rest (attention, shared experts, router, embeddings) stays resident.

## GLM-5.3-Flash (`glm5-next`)

46 blocks: 45 trunk layers and one next-token (MTP) block. `n_embd` 4096, vocabulary 154,880.

- **Layer types**: a 3:1 hybrid. Layers 3, 7, …, 43 (11) are full attention (MLA with a sparse indexer); the other
  34 are linear attention (KDA). Layers 0-2 have a dense FFN (12288), the rest MoE.
- **KDA** (64 heads of 128, inner 8192): q, k, v projections → causal depthwise conv (4 taps) + SiLU → L2-normed
  q, k. Per channel decay `g = -5 · sigmoid(-(A_h · (f + dt_bias)))` with `f` from a low-rank projection
  (`f_a`, `f_b`); per head `β = sigmoid(·)`. Delta rule per head with state `S` (128×128):
  `S *= exp(g)` (per row), `δ = (v − Sᵀk) · β`, `S += k δᵀ`, `o = Sᵀq / √128`. Output
  `rms_norm(o) · sigmoid(g_b(g_a x))` → `attn_output`. The state is fixed-size, so these layers cost the same at
  any context length.
- **MLA** (64 heads, q LoRA 1536, KV latent 512, qk/v head 256, no RoPE): the cache holds one 512-wide latent
  row per token (plus the indexer's key rows); `k_b` / `v_b` are absorbed per head so attention runs in latent
  space.
- **Indexer** (32 heads of 128): keys are pooled over blocks of 4 tokens (softmax pooling with a gate and a learned
  position term); each query scores the pools (`Σ_h relu(q_h·k)·w_h`) and attends to the top 2048 tokens' worth
  (512 pools) plus the incomplete tail. Below 2048 tokens attention is dense.
- **MoE**: 288 experts, top 8, sigmoid scores, weights normalized and × 2.5; experts 2048 wide (Maya-S: gate/up
  IQ2_XXS, down IQ3_XXS), shared expert 2048.
- **Output**: the mean of the 4 streams → `output_norm` → `output`.
- **MTP block** (used as the speculative drafter): `eh_proj([enorm(embedding of the next token); hnorm(last
  hidden state)])` → one MLA + MoE layer with plain residuals → the shared output head. It proposes one token,
  verified with a two-token step; KDA states after the first token are kept for rollback.

## DeepSeek-V4.1-Flash-1017 (`deepseek41`)

40 layers, `n_embd` 5120, vocabulary 129,280 (BPE, DeepSeek-V3's pre-tokenizer). Every layer is attention + MoE.

- **Hyper-connections**: as above, with two differences: a sublayer's input is collapsed with the *previous*
  sublayer's `pre` (the first uses `[1,0,0,0]`), and there is no separate head mix: the logits use the last FFN's
  `pre`.
- **Attention**: 64 heads of 512, K = V (one KV head), RoPE on the last 64 dims (adjacent pairs).
  - q: `rms(x·q_a)` (1280) → `q_b` → 64×512, roped. kv: `rms(x·kv)` (512), roped, stored FP8 E4M3.
  - Every layer: a 128-token raw window of its own KV rows.
  - Compressed rows: layers 2, 8, 14 (2 tokens per row, per-channel softmax pooling with a gate) and 20 (one row
    per token) compute them; layers 3-19 read their source's, 21-39 read layer 20's. Stored FP4 (E2M1, E4M3 scale).
  - Indexer: index keys `rms(latent·attn_k)` (128, roped, FP4) at the source layers; index queries at layers 2, 8,
    14, 20, 24, 28, 32, 36 score the visible keys and take the top 512 compressed rows; layers between reuse the
    last selection. Layer 20 also keeps the top 2048 blocks of 8 rows (block max, newest kept), which restricts
    layers 24-36's scoring (the "candidate stage"; active past ~16K tokens).
  - Softmax over window ∪ selected rows, scale 1/√512, a per-head sink in the denominator.
  - Output: inverse RoPE on the last 64 dims, FP8, 8 groups of 8 heads through `output_a` (4096 → 1024 each),
    then `output_b` (8192 → 5120).
  - RoPE: layers 0-1 base 10000; the others base 160000 with YaRN (factor 16, original 65536, β 32 / 1), no mscale.
  - Attention state is small: at 131K tokens all caches together are ~0.1 GB.
- **MoE**: 384 experts, top 6; router F32 with `sqrt(softplus)` scores; weights = scores / sum × 1.5; experts 2304
  wide; one shared expert.
- **Engram** (layers 1 and 14, before the attention mixes): per token, 24 hashed n-gram rows (2-, 3-, 4-grams ×
  8 heads) of 256 FP8 values (+ E8M0 scales, 264 bytes) per layer: ~12.7 KB of random reads per token from ~203 GB
  of tables, which stay on SSD and are read as soon as the token is known. The rows → `engram_kv` (6144 → 4
  per-stream keys + 1 value of 5120) → per stream `h += sigmoid(sign(d)·√|d|) · value`, `d` a normalized dot of
  key and stream. The hash parameters (primes, multipliers, token map, offsets) are `deepseek41.engram.*` keys.
- **Output**: `rms(Σ pre·h, output_norm) · output`.
- **DSpark** (drafter, separate files, ~8 GB): 3 small V4.1-style blocks (128 experts, top 3, 128-token window
  attention) fed the mean of the streams entering layers 37-39; proposes up to 5 tokens with a confidence; one
  6-row target pass verifies them, and rollback rewinds the window rings.
- **Chat template**: DeepSeek's encoder (`<｜User｜>` / `<｜Assistant｜>`, `<think>` for thinking mode, a reasoning
  effort line, tool calls in DSML tags); `tests/data/ds41_template/` holds its fixtures.

## Image models

Both run from their original files (BF16 safetensors and configs), not a GGUF: the published GGUFs hold only the
transformer, as F16 (BF16 values below ~6.1e-5 lose bits there) or quantized. They share `image::Engine`: the Qwen3-VL
text encoder, the VAE decoder, the flow-matching Euler schedule and the denoising loop. `Engine::open` picks the model
from `model_index.json`.

### Qwen-Image 2.1

- Transformer (13.25 GiB): 32 layers, hidden 4096 (32 heads of 128), MLP ratio 3, 64 latent channels in and out,
  patch size 1, rotary positions on three axes (16 / 56 / 56), `causal_condition`. A 1024x1024 image is 4096 latent
  positions; a step is ~57 TFLOP (2 x 7B x 4096), compute bound like a prefill.
- Text encoder (16.3 GiB): Qwen3-VL; text part 36 layers, hidden 4096, 8 KV heads, interleaved multi-axis rotary
  positions; a 27-layer vision tower for image prompts (not used yet).
- VAE (1.26 GiB): 4 channels (RGBA), 64-channel latent, 16x spatial compression (sides multiples of 32), with the
  temporal settings of a video VAE.
- Schedule: dynamic shift; 40 steps.

### Krea 2

- Transformer (12B): 28 single-stream blocks over [text, image], hidden 6144 (48 query heads, 12 key / value heads
  of 128), a sigmoid gate on the attention output, SwiGLU 16384, RMS norms with zero-centred scales, one shared
  timestep modulation plus a learned table per block, rotary axes 32 / 48 / 48 (theta 1000; the text at the origin).
- Text: Qwen3-VL-4B's hidden states 2, 5, ..., 35 per token, fused by two blocks across the twelve layers, a learned
  mix, and two blocks across the tokens (2560 wide, 20 heads). The reference pads the text to 512 rows and masks the
  padding; mako runs only the prompt's rows, which gives the same result.
- VAE: the first Qwen-Image's (16 channels, 8x), the transformer's tokens its 2x2 patches; causal 3D convolutions
  reduced to their last time tap.
- Turbo (`is_distilled`): 8 steps, a fixed shift (mu 1.15), no guidance.

## How the engines are built

Three layers, shared as far down as the models allow:

- **`gpu::Runtime`** (`src/gpu`): Metal plumbing, shared with the image models (`image::Engine`): scratch buffers,
  the pipeline cache, dispatch helpers, encoders and timing.
- **`core::EngineBase`** (`src/core`): what the routed-expert LLMs share: weights from the model's GGUF files, the
  expert arena (cache, lending, scoring, profiles), decode's GPU routing with the CPU serving misses between a
  layer's router and its experts, staged prefill with batched miss reads, hyper-connections, input rows.
- **`glm::Engine`, `dsv41::Engine`**: the model-specific parts: the layer graph, attention and its caches, Engram,
  the drafter (GLM's MTP layer, DeepSeek's DSpark), the per-token state.

`Session`, the session pool, the sampler and the server talk to an LLM only through `gen::Model` (`src/gen/model.hpp`:
prefill, step, checkpoints, drafts and their verification), so both models share them.

Accuracy: where the reference rounds activations as part of the model (DeepSeek's FP8 window KV and attention output,
FP4 compressed KV, index keys and queries), mako rounds the same way, which also keeps those caches small; everything
else is computed in F32/F16. Both models are checked against the reference implementations' logits.
