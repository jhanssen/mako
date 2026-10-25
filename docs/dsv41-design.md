# DeepSeek-V4.1-Flash in mako: design

Status: design, before implementation. Target: DeepSeek-V4.1-Flash-1017 (`deepseek41` GGUF, Maya-S IQ2_XXS quant,
~178 GB in four shards, plus the Engram tables, ~203 GB MXFP8, in a separate GGUF) on a 128 GB M5 Max, where only
about 40% of the routed experts fit in memory. Goal: as fast as possible at reasonable accuracy; the reference
implementations are used for semantics and for checking logits, not as a design template.

## The model, as mako needs it

40 layers, n_embd 5120, vocabulary 129,280. Every layer is attention + MoE, wrapped in 4-stream hyper-connections.

- **Hyper-connections**: the same mixes as GLM-5 (`hc_fn` on the RMS-normed flat streams; pre / post sigmoids;
  Sinkhorn comb, 20 iterations), with two differences: a sublayer's input is collapsed with the *previous* sublayer's
  `pre` (the first sublayer uses `[1,0,0,0]`), and there is no hc head: the logits use the last FFN's `pre`.
- **Attention**: 64 heads of 512, K = V (one KV head), RoPE on the last 64 dims (adjacent pairs).
  - q: `rms(x·q_a)` (1280) → `q_b` → 64×512, rope. kv: `rms(x·kv)` (512), rope, FP8 E4M3 (pow2 scale / 32).
  - Every layer: a 128-token raw window of its own kv rows.
  - Compressed rows: only layers 2, 8, 14 (2 tokens per row, per-channel softmax pooling with a gate) and 20 (one
    row per token) compute them; layers 3-19 read their source's (2/8/14), 21-39 read layer 20's. Stored FP4 (E2M1,
    E4M3 scale / 16) at the group's first position, roped there.
  - Indexer: index keys `rms(latent·attn_k)` (128, roped, FP4) at the source layers; index queries at layers 2, 8,
    14, 20, 24, 28, 32, 36 score all visible keys (`Σ_h relu(q_h·k)·w_h`) and take the top 512 compressed rows; the
    layers between reuse the last selection. Layer 20 also keeps the top 2048 blocks of 8 rows (block max), which
    restricts layers 24-36's scoring.
  - Softmax over window ∪ selected compressed rows, scale 1/√512, a per-head sink in the denominator.
  - Output: inverse rope on the last 64 dims, FP8, 8 groups of 8 heads through `output_a` (4096 → 1024 each), then
    `output_b` (8192 → 5120).
  - RoPE: layers 0-1 base 10000; the others base 160000 with YaRN (factor 16, original 65536, β 32 / 1), no mscale.
- **MoE**: router F32, `sqrt(softplus)` scores; top 6 by score + bias (`exp_probs_b`, or `_vl` in image spans);
  weights = scores / sum × 1.5; experts and the shared expert use SwiGLU with the gate clamped above at 10 and up
  clamped to ±10.
- **Engram** (layers 1 and 14, before the attention mixes, text positions only): 24 hashed n-gram rows (2-, 3-,
  4-grams × 8 heads) of 256 FP8 values per layer per token, ~12.7 KB read per token, → `engram_kv` (6144 → 4 keys +
  1 value of 5120) → per stream `h += sigmoid(sign(d)·√|d|)·value`, d a normalized dot with the stream.
- **Output**: `rms(Σ pre·h, output_norm) · output`.
- **DSpark** (draft, separate file, ~8 GB): 3 small V4.1-style blocks fed with layers 37-39's collapsed streams;
  proposes up to 5 tokens with a confidence; one 6-row target pass verifies.

## Engine design

### Shared core (refactored out of the GLM engine as the DeepSeek engine needs it)

GLM's engine holds a lot that is not GLM-specific. Each piece moves to shared code when DeepSeek first needs it, and
each move is checked against GLM's bit-exact regression tests:

- Metal plumbing: scratch buffers, pipeline cache, dispatch helpers, encoders and timing, `proj`/`gemv` wrappers.
- Common kernels: RMS norm, the hc mixes / Sinkhorn / post, SwiGLU with clamps, router top-k, combine.
- The routed-expert path: the expert arena (cache, lending, scoring, profiles), decode's GPU routing with the CPU
  serving misses through shared events, prefill's grouped expert GEMMs with batched miss reads.
- Generation: `Session`, the sampler and the server talk to a model through an interface (prefill, step, verify k
  drafts, rollback, checkpoints) instead of `glm::Engine`, so both models share them. The verify path generalizes
  GLM's two-token step to DSpark's up to six rows.

Model-specific: the layer graph, attention and its caches, Engram, the drafter, the per-token state.

### Memory

- Dense weights (Q8_0/BF16/F32): ~9 GB resident. DSpark: ~8 GB resident.
- Attention state is small: 40 rings of 128×512 FP8; compressed rows for 4 sources (layer 20 one per token, the
  others one per 2) of 512 FP4 (~288 B) and index keys of 128 FP4 (~68 B). At 131K tokens all of it is ~0.1 GB, so
  long contexts cost the expert cache almost nothing.
- Everything else goes to the expert arena (~70 GB → ~6,400 of 15,360 experts of ~11 MB). Engram tables stay on SSD.

### Speed

Decode is bound by reading missing experts from SSD (~1 ms per 11 MB expert at the SSD's ~10-12 GB/s for single
reads). The levers, in order:
1. DSpark: verifying up to 5 drafts in one pass shares each expert read across the tokens that use it.
2. Cache policy and profile (decayed counts are near the best a counting rule gets).
3. Overlap: Engram reads start when the token is known; expert reads overlap the resident experts' work.
4. Later: the parked low-bit + correction idea (more experts resident, fewer bytes per miss).

Prefill: grouped expert GEMMs, as GLM's, with misses read in batches; IQ3_S needs a GEMM dequantizer (decode has
the GEMV).

### Accuracy

Match the reference's activation roundings where they are part of the model (FP8 window KV, FP4 compressed KV and
index keys and queries, FP8 attention output): they also make the caches small. Compute otherwise in F32/F16 as
mako does for GLM. Checked against the reference's logits.

## Verification

- Tokenizer: the reference tokenizer's ids on the shared 2,051 cases (done). Chat template: DeepSeek's encoding
  fixtures byte for byte (done).
- Logits: the reference engine on the same weights (on the Linux box, `--dump-logits` for every position) for short
  and long prompts; KL and top-1 per position, as for GLM. Per-layer probes where they diverge.
- Speed: against the reference engine's measured 14.7 tok/s decode / 599 tok/s prefill on its machine, and on this
  Mac with the self-check benchmark.

## Milestones

1. Config, tokenizer, template (done); loader for the shards + Engram GGUF; per-token Engram rows and hashing.
2. Decode step, token by token: attention with its caches, MoE through the shared expert path, Engram → logits
   matching the reference.
3. Batched prefill; checkpoints and rollback.
4. SSD streaming tuned for ~40% residency; the shared `Session`/server on the model interface.
5. DSpark drafting and multi-row verify.
6. Server: DeepSeek template variables, DSML tool calls and reasoning in the output parser.
