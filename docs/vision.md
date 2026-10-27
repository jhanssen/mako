# Vision (design, in progress)

Images as prompt input for both models, from a projector file (an "mmproj" GGUF) next to the text model:

| Model | mmproj | Projector type |
| --- | --- | --- |
| DeepSeek-V4.1-Flash-1017 | `mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf` (972 MB) from `ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF` | `deepseek41v`; the published file says `deepseek4v` (mako takes it as V4.1 without V4's padding vector; the reference needs `tools/gguf_retag.py <in> <out> clip.projector_type deepseek41v`) |
| GLM-5.3-Flash | `mmproj-BF16.gguf` (1.16 GB) from `unsloth/GLM-5.3-Flash-GGUF` | `glm5v` |

The reference is the llama.cpp checkout `ref_vision` links (`tools/ref_dump`), on its CPU backend.

## What the two pipelines do

Both: decoded RGB image -> size plan and resize/pad -> normalise -> 14x14 patch embedding -> a ViT (1024 wide, 16
heads of 64, RMSNorm, fused QKV with bias, 2D RoPE, full bidirectional attention, SwiGLU) -> post-norm -> merge
neighbouring patches -> projector to the text model's width. The 2D RoPE is the same in both: in each head, dims k
and k+32 rotate together, by the patch's row times `10000^(-k/16)` for k < 16 and by its column times
`10000^(-(k-16)/16)` for k = 16..31.

| | DeepSeek-V4.1 | GLM-5.3 |
| --- | --- | --- |
| Size plan | round up to a multiple of 14 within 1024 output tokens (start, end and newlines included), at least 544^2 px; aspect-fit, then centred on grey (127) | round up to a multiple of 28; resized only below 112^2 px or above 8000 tokens; content top-left, black padding right and bottom |
| Resize | bicubic (a = -0.5), separable, 22-bit fixed-point weights, uint8 between the passes | the same, when it resizes |
| Normalise | (x/255 - 0.5)/0.5, then rounded to bf16 | CLIP mean and std |
| Patch embedding | one 14x14 conv + bias, patches in raster order | two 14x14 convs on the same image, summed, + bias; patches in 2x2-block raster order |
| ViT | 32 blocks, FFN 2816 without biases, eps 1e-6 | 24 blocks, FFN 4096 with biases, per-head q/k RMSNorm before RoPE, SwiGLU clamped (`silu(min(g, 10)) * clamp(u, -10, 10)`), eps 1e-5 |
| Merge | 3x3 cells, features zero-padded to a multiple of 3, 9216-vector channel-major (`c*9 + kh*3 + kw`) | 2x2 conv (stride 2) over each block of 4 consecutive patches, + bias |
| Projector | Linear + bias, GELU (erf), Linear + bias -> 5120 | Linear, LayerNorm (eps 1e-5), GELU (erf), clamped SwiGLU MLP (10240) -> 4096 |
| Block in the prompt | `[START] ([cell]*w [NEWLINE])*h [END]`: h*(w+1)+2 rows, the three markers learned vectors from the mmproj | the cells only (w*h rows), between the vocabulary tokens `<|begin_of_image|>` and `<|end_of_image|>` |
| Text model | sequential positions, causal; Engram gives image positions nothing, and an n-gram reaching back into an image uses the pad id there | sequential positions (the model has no RoPE), causal |

A 1024x1024 image: DeepSeek 1036x1036, 74x74 patches, 25x25 cells, 652 rows; GLM 1036x1036, 74x74 patches, 37x37 =
1369 rows.

## In mako

- `src/vision/`: the mmproj loader (GGUF metadata and tensors), the size plan and resize (CPU, integer), and the
  encoder on Metal: a graph per projector type over shared pieces (patch conv, ViT block, merges, projectors), its
  linears through the GEMM path (BF16 weights), attention as a tiled kernel (up to ~32K patches for GLM's 8000
  tokens).
- Images in the prompt: the chat templates already render an image part as a placeholder (`<｜deepseek_image｜>`;
  GLM `<|begin_of_image|><|image|><|end_of_image|>`). The server expands it to the image's rows: DeepSeek's whole
  block replaces the placeholder; for GLM the block replaces `<|image|>` between the two markers.
- Prefill takes embedding rows for given positions (the engines dequantise token rows on the CPU into a shared
  buffer; image rows are copied there instead). DeepSeek's Engram masks image positions as above. GLM's next-token
  block, which reads the next token's embedding, gets the image row there.
- `gen::Session` keys image positions by the image's content hash, so a prefix holding a different image is not
  reused.
- Server: OpenAI `image_url` parts and Anthropic `image` blocks, base64 / `data:` URLs (no fetching), PNG and JPEG
  decoded with ImageIO. `--mmproj FILE` loads the projector (its ~1 GB comes out of the expert cache budget).

## Checking it

- Preprocessing: bit-exact against the reference's `inp_raw` for PNG input: `vision_prep` checks the images in
  `tests/data/vision` (made by `tools/gen_vision_fixtures.py`: every sizing path, grey, grey + alpha, palette,
  16-bit, RGBA) against the reference's hashes, for both projector types. JPEG decoders differ: on a photo ~10% of
  the samples differ, by 0.13 levels on average (99th percentile 2 levels, at most 35, at edges).
- Encoder: not bit-exact (the reference accumulates the patch embedding in fp16, rounds activations to bf16 before
  each linear and runs attention with fp16 K and V). Its two backends disagree with each other, so each is the
  yardstick for the other. 1024x1024 image, per-row cosine (mean / worst) and mean relative error of the final rows:

  | | reference CPU vs GPU | mako vs reference GPU | mako vs reference CPU |
  | --- | --- | --- | --- |
  | DeepSeek-V4.1 (652 rows) | 0.99971 / 0.985, 1.8% | 0.99999 / 0.998, 0.4% | 0.99970 / 0.980, 1.7% |
  | GLM-5.3 (1369 rows) | 0.990 / 0.50, 10% | 0.997 / 0.81, 4.5% | 0.989 / 0.49, 11% |

  GLM's ViT amplifies rounding on a few tokens: per layer the worst row's relative error roughly doubles, on the
  same rows in the reference's CPU-vs-GPU comparison as in mako's (layer 23: 4.34 vs 4.37 on row 1934). No
  automated encoder test yet (the reference rows are large); `vision_run --probe` and `ref_vision` with
  `REF_VISION_DUMP` compare stage by stage.
- Text: answers about a known image (the Sonoma wallpaper) are right for both models, through `vision_chat` and the
  server (colours, no text in it, the shapes). Not done yet: GLM logits after an image prompt against the reference
  (which can run GLM on the CPU).

## Milestones

1. mmproj loader, size plan, resize and normalise; `ref_vision` dumps named intermediates; preprocessing exact.
   Done: `src/vision/{image,mmproj,preprocess}`, `tools/vision_run --prep-out`.
2. Encoder on Metal for both projector types, compared stage by stage. Done: `src/vision/encoder`,
   `src/kernels/vision.metal`. Attention, 71-95% of the encode when it ran as per-head score GEMMs + a softmax +
   value GEMMs, is one kernel (`vit_flash_attn`: 32 queries x 128 keys a step, the running softmax in threadgroup
   memory, the output in registers, both products through matmul2d). Encode, M5 Max: 1024x1024 GLM 0.55 -> ~0.3 s,
   DeepSeek 0.66 -> ~0.45 s; GLM's largest (3200x2400, 7931 rows) 29 -> ~7 s (`vision_run --timing 1` gives the GPU
   time by stage; run-to-run spread is ~20%, so compare interleaved). Agreement with the reference unchanged.
   `--image-max-tokens N` (server, `vision_chat`) scales larger images down: GLM's 3200x2400 at 2048 takes ~4 s to
   encode and prefill instead of ~18 s.
3. Prefill with image rows, Session keys, Engram masking. Done: `gen::Model::set_input_rows`, the engines' token
   rows (`EngineBase::embed`), `GenerateParams::input_rows` (keys; a checkpoint at the end of the prompt's last
   image, so another question about it reuses the prefix: 1380 of 1396 tokens with GLM, its prefill 2.4 -> 0.19 s),
   `vision::expand_images`, `tools/vision_chat`.
4. Server API. Done: `--mmproj`, OpenAI `image_url` (data: URLs) and Anthropic `image` blocks (user messages and
   tool results), the last 8 encoded images cached (docs/server.md).
