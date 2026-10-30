# Several models in one server

Plan (2026-10-29) for serving the LLMs and an image model, Qwen-Image 2.1, from one `mako_serve`: one model resident
at a time, swapped when a request needs the other, conversations kept across a swap. Steps 1-5 change no behaviour
and are checked as any engine change is (tests, `perf_check`); the image work starts after them.

## Layers

| Layer | Contents | Knows of |
| --- | --- | --- |
| `gpu` | device, pipeline cache, command recording and submission, timing and telemetry, probes; the memory owner; tensors from a format-neutral source; generic ops by type (GEMM / GEMV incl. BF16 with F32 accumulation, norms, attention causal / bidirectional / block-causal, elementwise, convolutions) | nothing above it |
| formats | `gguf` (as now), `safetensors`; both give tensor descriptors (name, type, dims, file, offset) | `gpu`'s tensor source |
| `llm` | the routed-expert machinery now in `core` (arena, stager, routing profiles, MoE decode and prefill, hyper-connections, input rows); `glm`, `dsv41`, `gen`, `text`, `chat`, the vision projector | `gpu`, formats |
| `image` | Qwen3-VL text encoder, the diffusion transformer, the VAE, the flow-matching scheduler, PNG output | `gpu`, formats |
| runtime | the model manager: loads and unloads models through the memory owner, decides swaps, holds conversations outside the engine | `llm`, `image` |
| `server` | chat endpoints to the LLM, image endpoints to the image model, through the runtime | runtime |

Each is a static library; `gpu` links nothing above it, so nothing model-specific can reach into it.

## Steps

1. Split `core::EngineBase` into `gpu::Runtime` (weights and scratch buffers, pipelines, dispatch, encoders and
   command buffers, telemetry, probes, the GEMV / GEMM wrappers, norms) and the LLM base (everything routed-expert).
   The engines keep deriving from the LLM base, which derives from the runtime.
2. The memory owner: one budget for the machine; residents (dense weights, scratch, the expert arena, KV caches,
   parked conversations) take and return bytes by name. The arena sizes itself from what it is granted instead of
   from the machine's free memory.
3. The format-neutral tensor source, with GGUF behind it.
4. Conversations outside the engine: a session parks into state the engine does not own, the engine is destroyed and
   built again, the session restores into it (refused for another model or file). Tested as an LLM unloaded and
   loaded again.
5. The model manager and its swap path, with the LLM as the only model.
6. The image model:
   - safetensors reader;
   - BF16 GEMM with F32 accumulation (first a `gemm_bench` check of the M5's `bfloat` support and speed);
   - the text encoder, the transformer, the VAE (convolutions), the scheduler;
   - each checked against the Diffusers pipeline's intermediate tensors at a fixed seed.
7. The image endpoint (`/v1/images/generations`), progress per denoising step, the status page.

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
