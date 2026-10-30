# Several models in one server

Plan (2026-10-29) for serving the LLMs and an image model, Qwen-Image 2.1, from one `mako_serve`: one model resident
at a time, swapped when a request needs the other, conversations kept across a swap. The refactoring steps change no
behaviour and are checked as any engine change is (tests, `perf_check`); the image work starts after them.

## Layers

| Layer | Contents | Knows of |
| --- | --- | --- |
| `gpu` | device, pipeline cache, command recording and submission, timing and telemetry, probes; tensors from a format-neutral source; generic ops by type (GEMM / GEMV incl. BF16 with F32 accumulation, norms, attention causal / bidirectional / block-causal, elementwise, convolutions) | nothing above it |
| formats | `gguf` (as now), `safetensors`; both give tensor descriptors (name, type, dims, file, offset) | `gpu`'s tensor source |
| `llm` | the routed-expert machinery now in `core` (arena, stager, routing profiles, MoE decode and prefill, hyper-connections, input rows); `glm`, `dsv41`, `gen`, `text`, `chat`, the vision projector | `gpu`, formats |
| `image` | Qwen3-VL text encoder, the diffusion transformer, the VAE, the flow-matching scheduler, PNG output | `gpu`, formats |
| runtime | the model manager: loads and unloads models, decides swaps, holds conversations outside the engine | `llm`, `image` |
| `server` | chat endpoints to the LLM, image endpoints to the image model, through the runtime | runtime |

Each is a static library; `gpu` links nothing above it, so nothing model-specific can reach into it.

## Steps

1. Done (25645f9): `gpu::Runtime` (weights and scratch buffers, pipelines, dispatch, encoders and command buffers,
   telemetry, probes, the GEMV / GEMM wrappers, norms, the dense SwiGLU FFN) split from `core::EngineBase`, which keeps
   the routed-expert machinery and derives from it.
2. Not needed for one model at a time: the expert arena sizes itself from the process's footprint and the memory
   available when it is built (`arena_budget`), so a model loaded after another was unloaded gets what is free, and
   conversations kept in memory are already subtracted. A memory owner would matter for two models resident at once.
   The swap step checks that unloading gives the memory back.
3. Moved to the image work: the format-neutral tensor source, with GGUF and safetensors behind it, and the weight
   types out of `gguf`.
4. Done: conversations outside the engine. `SessionPool::suspend()` parks the live conversation and lets go of the
   engine; `resume(engine, identity)` rebinds the conversations to a new engine of the same model, or gives them up
   for another identity. `e2e_reload` / `ds41_reload`: the next turn across an engine destroyed and built again
   gives the same tokens and logits, bit for bit, reading only its new tokens.
5. Done: the model manager (`server::Models`): leases of the LLM or the image model, the swap when a lease of the
   other comes (an image lease waits for the LLM leases out; LLM leases asked for meanwhile wait for it), the
   conversations parked in memory across it. Tried with GLM-5.3: image model in 4.7 s, a 512x512 image in 12.3 s, the
   LLM back in 10.1 s, the conversation's next turn reusing its prompt.
6. Done: the image model from the original files.
   - safetensors reader (`mako_safetensors`); the weight types still come from `gguf` (step 3's move left out);
   - BF16 GEMM with F32 accumulation: as fast as F16 on the M5 (55 TFLOP/s on a 4096-token MLP);
   - the Qwen3-VL text encoder, the transformer (with the text's keys and values cached after the first step), the
     VAE decoder (convolutions as im2col + GEMM), the flow-matching Euler scheduler, PNG output, `tools/mako_image`;
   - against the reference pipeline's fp32 tensors (`image_ref_test`): the text encoder 1.2e-2 relative (the
     pipeline's own BF16 run: 7.8-8.6e-2), predictions <= 1.6e-2, the VAE 7e-4, the prompt to the image at 512x512
     and 3 steps 5e-3.
   - Speed, 1024x1024 and 40 steps, each after a 5-minute cool-down: 63.0 s (step 2 1.28 s, step 40 1.77 s); the
     reference pipeline on MPS in BF16 65.1 s (1.36 / 1.75 s). Both slow down with heat during a run while macOS
     reports the nominal thermal state (pauses between steps restore the speed). From 79.9 s by: attention with
     each simdgroup on its own rows (12.0 -> 7.1 ms a layer), the GEMMs run on a multiple of 256 rows (the first
     step's text + image rows took 64-row tiles, twice as slow), dense BF16 / F16 weights read by matmul2d straight
     from device memory (55 -> 63 TFLOP/s), q / k / v and gate / up one GEMM each with the heads' norms fused, the
     VAE's convolution weights in F16 on that GEMM.
   - mako's starting noise comes from its own generator: a seed gives another image than the reference's.
7. Done: `/v1/images/generations` with OpenAI's fields (`docs/server.md`), progress in the log, the loaded model on
   the status page. Extensions: `negative_prompt` and `guidance_scale` (step 10); seed, steps, presets, `quality` as
   steps and streamed progress wait for the user's decisions.
8. Next (the user's priorities, 2026-10-29): images as input and LoRAs.
   - Condition images (one pipeline mode in 2.1: edits and references by prompt): the Qwen3-VL vision tower in the
     text encoder, the VAE encoder (latent tokens ahead of the target image), image blocks in the attention mask
     (each condition image its own bidirectional block), the image template; `/v1/images/edits`.
   - img2img: the VAE encoder and the schedule started part-way (sigma below 1).
   - Done, LoRAs merged at load (`Engine::add_lora`): W += scale * alpha / rank * B A on the GPU, rounded back to
     BF16; several add up. Diffusers (`transformer.*.lora.down/up`), PEFT (`lora_A/B.default`) and ComfyUI
     (`diffusion_model.*.lora_A/B`, the MLP's `gate_up` gate rows first) names; alpha from `.alpha` tensors, else a
     PEFT config in the metadata, else the rank (files that state none: assumed, unverified). Tried: lightx2v's
     4-step Lightning (128 matrices; 4 steps without terminal stretch, 1024x1024 in 6.9 s, clean, where 4 steps
     without it are smeared), Next-Scene (ComfyUI, 192) and Natural-Exposure (PEFT, 224) stacked with it. Per-request
     LoRAs (applied at run time, or merged and taken back) later.

9. Done: Krea 2 (`image::Krea2`; `Engine::open` picks the model from `model_index.json`). The image models share
   `image::Engine`: the Qwen3-VL text encoder (its width, heads and taps from the weights), the VAE decoder (Qwen-Image
   2.1's 16x one, or the first Qwen-Image's 8x one with a token's 2x2 patch unpacked and its causal 3D convolutions
   reduced to their last time tap), the schedule and the denoising loop. Krea 2 from `krea/Krea-2-Turbo` (text
   encoder, VAE, configs), its transformer from that repository or a single-file checkpoint in the original names
   (`--transformer`). Against the reference pipeline in fp32 (`tests/krea_ref_test`, 256x256, 2 steps): the twelve
   tapped text states 2.5e-3, predictions 1.2-2.1e-2, the VAE 5e-3, the image 0.133 (the reference in BF16: 0.158).
   1024x1024 in 8 steps: 23 s.
10. Done: negative prompts (classifier-free guidance, `Request::negative` / `guidance`): a second text in its own
    slot (`begin` / `step` slot 1), velocity neg + scale * (pos - neg) each step, twice the transformer's work.
    Without a negative prompt nothing changes. Scales tried (`docs/server.md`): Qwen-Image 2.1 clean at 2-6 (default
    4); Krea 2 Turbo, distilled without guidance, clean only to 2 (default 2) and weak there.

## Krea 2

- Transformer (12B): 28 single-stream blocks over [text, image], hidden 6144 (48 query heads, 12 key / value heads
  of 128), a sigmoid gate on the attention output, SwiGLU 16384, RMS norms with zero-centred scales, one shared
  timestep modulation plus a learned table per block, rotary axes 32 / 48 / 48 (theta 1000; the text at the origin).
- Text: Qwen3-VL-4B's hidden states 2, 5, ..., 35 per token, fused by two blocks across the twelve layers, a learned
  mix, and two blocks across the tokens (2560 wide, 20 heads). The reference pads the text to 512 rows and masks the
  padding; mako runs only the prompt's rows, which gives the same result.
- VAE: the first Qwen-Image's (16 channels, 8x), the transformer's tokens its 2x2 patches.
- Turbo (`is_distilled`): 8 steps, a fixed shift (mu 1.15), no guidance.

## Qwen-Image 2.1

From `Qwen/Qwen-Image-2.1` (BF16 safetensors; the configs):

- Transformer (13.25 GiB): 32 layers, hidden 4096 (32 heads of 128), MLP ratio 3, 64 latent channels in and out,
  patch size 1, rotary positions on three axes (16 / 56 / 56), `causal_condition`.
- Text encoder (16.3 GiB): Qwen3-VL; text part 36 layers, hidden 4096, 8 KV heads, interleaved multi-axis rotary
  positions; a 27-layer vision tower for image prompts.
- VAE (1.26 GiB): 4 channels (RGBA), 64-channel latent, 16x spatial compression (sides multiples of 32), with
  temporal settings of a video VAE.
- Scheduler: flow-matching Euler with dynamic shift.

The originals, not a GGUF: the published GGUFs hold only the transformer, as F16 (BF16 values below ~6.1e-5 lose
bits there) or quantized.

A 1024x1024 image is 4096 latent positions; a step through the transformer is ~57 TFLOP (2 x 7B x 4096), compute
bound like a prefill. Its time on the M5 Max is unmeasured.

## Swaps

- A request for the model not loaded waits for the requests in flight, then the current model is unloaded and the
  other loaded (DeepSeek loads in ~10 s; ~31 GB of image weights from the SSD, unmeasured).
- Requests for the loaded model that are already queued run before a swap, so alternating requests do not swap at
  every one.
- Conversations stay in memory across a swap (a few hundred MB to a few GB each); keeping them on disk, across
  restarts, can follow.
