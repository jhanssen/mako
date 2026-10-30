# Development

## Tools

```sh
./build/tools/mako_run <GLM-5.3 .gguf> <token,ids,...> [-n N] [--spec] [--temp T]   # GLM-5.3: top logits, greedy tokens
./build/tools/ds_run <DeepSeek .gguf> <Engram .gguf> --ids I,J,... [-n N]          # DeepSeek-V4.1 (the Engram argument:
                                                                                    # the model's first file again)
./build/tools/mako_image <Qwen-Image dir> "<prompt>" [-o out.png] [--width N] [--height N] [--steps N] [--seed N] [--rgb]
                         [--count N] [--lora FILE[:SCALE] ...] [--terminal F] [--text-encoder DIR]
```

`mako_run` samples with `--temp T` (`--top-k`, `--top-p`, `--min-p`, `--seed`). `ds_run` also compares logits with a
reference dump (`--ref`), measures decode widths and collects an imatrix (`--imatrix-out`). Others: `prefill_sweep`
(prefill timing), `gguf_dump`, `tokenize`, `render_template`, `engram_rows`, `logit_compare`, and `ref_dump` and
`ref_vision` (an image's embeddings through the reference's vision encoder and projector) when CMake is given a
reference checkout (`MAKO_REF_SRC`, `MAKO_REF_BUILD`; `ref_vision` needs its `mtmd` library built). The DeepSeek-V4.1
projector published as `mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf` declares the V4 projector type, `deepseek4v`;
the reference loads it as `deepseek41v`, the V4.1 type, once that one string in its metadata is changed
(`tools/gguf_retag.py <in> <out> clip.projector_type deepseek41v`).

Programs generate through `gen::Session` (`src/gen/session.hpp`), for either model: it takes the whole prompt each
time, keeps the longest prefix the engine already holds (continuing the current state, or restoring a checkpoint of
the state taken before an earlier prompt's last token), samples, drafts and stops on stop tokens or a length limit.

## Tests

```sh
cmake -S . -B build -DMAKO_MODEL=<GLM-5.3 .gguf> -DMAKO_DS41_MODEL=<DeepSeek .gguf>   # or the same environment variables
ctest --test-dir build --output-on-failure                                           # -L cpu | kernels | tokenizer | sampler | server | quick | e2e | slow
```

`MAKO_DS41_DSPARK=<dir>` adds the drafter's test, `MAKO_DS41_ENGRAM=<GGUF>` runs DeepSeek's with other Engram tables.
`MAKO_QWEN_IMAGE=<dir>` (the model's files as published) and `MAKO_QWEN_IMAGE_REF=<dir>` (reference tensors from the
original pipeline, fp32 on the CPU: a Python script written beside the model, not part of the repository) enable the
`image` tests.

- No model needed (`ctest -L cpu` for those without a GPU too, as CI runs): the kernel checks on random data
  (`quant`, `gemv_own`, `gemv_rows`, `gemm_synthetic`, `engram_decode`, `hc`), `sampler`, `presets`, `cli`,
  `template`, `vision_prep`. The others report as skipped without their model.
- GLM-5.3's `e2e` tests (about three and a half minutes): greedy decode tokens, prefill against token-by-token
  decode, logits against stored reference top-k, speculative steps, generation and prefix reuse through `Session`, a
  prompt past the dense-attention window.
- DeepSeek's `ds41_*` tests (about eight minutes): logits against the reference's on 1300 tokens (prefill; steps),
  on a 24K-token prompt's last 1024 positions, with the miss penalty and with dense Q6_K (each over the first 300
  steps, and Q6_K also through prefill); the drafter's greedy output; two conversations taking turns. `ds41_long_noskip`
  (the 24K prompt with every row through every layer, which is not the default) is labelled `slow` only.
- Qwen-Image 2.1's `image` tests (about 20 seconds), against the reference tensors at 512x512 and 3 steps: the
  prompt's token ids, the text encoder's output, the transformer's stages and predictions, the VAE decoder, and the
  prompt to the image; `safetensors` reads the model's files.

Which to run:

| Change | While working | Before committing |
| --- | --- | --- |
| Kernels, `src/core`, `src/glm`, `src/dsv41` | `-L quick` (~1.3 min: GLM decode, prefill, parity, drafts; DeepSeek prefill) | `-L e2e`, and `bench/perf_check` for each model the change reaches |
| The prefill skip, prompt lengths past a group | `-L quick` | `-L e2e` and `-L slow` |
| `src/server`, `src/gen` | `-L server` and `-L cpu` | the same, and `-L e2e` for `gen::Session` or the pool |
| Tokenizer, templates | `-L tokenizer` | the same |
| `src/image`, `src/kernels/image.metal`, `src/kernels/ops.metal` | `-L image` | the same, and `bench/perf_check` for the LLMs when a shared kernel or `src/gpu` changed |

`tests/data/README.md` says how each data file was made. After an intended numerical change,
`regress_test <model> --update` re-records `tests/data/regress_expected.txt`; review that diff.

## Self-check

```sh
python3 tools/self_check.py --model <GLM-5.3 .gguf> [--quick] [--out report.md]
```

For GLM-5.3: runs the tests and a fixed benchmark (prefill of 512 and 4096 new tokens on a 4K-token context, decode of
128 tokens with and without drafts, each a freshly loaded engine) and prints a Markdown report to paste into an issue.
`--quick` runs only the tests that need no GPU work.

## Speed regressions

The tests check results, not speed: a change that keeps every output bit-identical but makes a kernel slower passes
them all. `bench/perf_check` catches that:

```sh
./build/bench/perf_check <model .gguf> --save   # on a commit known to be good: the baseline for this machine
./build/bench/perf_check <model .gguf>          # after a change: compared with it (exit 1 when flagged)
```

It times a 4096-token prefill of `bench/data/perf_prompt.txt` (a fixed text, so the workload does not change with
the sources) and 64 greedy decode steps, for GLM-5.3 or DeepSeek-V4.1, each once for its time and once with part
timing for the GPU time by named part (and, for prefill, by section). Every run waits for macOS's nominal thermal state
and 30 seconds more (`--pause`): a throttled M5 Max runs everything ~1.3x slower, and a second heavy run within a minute
of the first was 8% slower even at nominal. It flags a total more than 10% slower (only between runs at nominal) and a
part of at least 1% of the GPU time whose share grew by more than 25%; shares barely move when the whole machine slows
down. Baselines live in `~/Library/Application Support/mako/perf/`, one per model file.

Measured on the M5 Max: run against its own baseline, every share agreed within 0.1 point; GLM's times within 0.1%,
DeepSeek's decode within 1.3% and its prefill within 6.7% (expert reads from the SSD). The `kda_prep` prefill regression
(fixed in `ce60b6e`) moved GLM's prefill only 3.6% at 4096 tokens, under the total's threshold, and was flagged by its
part: kda conv 1.3% -> 5.2%. Run it before committing a change to `src/kernels`, `src/core`, `src/glm` or `src/dsv41`.
It does not cover long contexts (the prompt has ~6.8K tokens); `tools/prefill_sweep --parts` does.

## Layout

| Path | Contents |
| --- | --- |
| `src/gguf` | GGUF reader, multi-file models |
| `src/quant` | CPU dequantization and the dense re-quantizers |
| `src/app` | command-line options: the environment, presets |
| `src/mtl` | Metal context, GEMV wrapper, embedded shader libraries |
| `src/kernels` | Metal kernels |
| `src/image` | Qwen-Image 2.1: text encoder, diffusion transformer, VAE decoder, scheduler, PNG |
| `src/safetensors` | safetensors reader |
| `src/gpu` | what any model's GPU code shares: weights and scratch buffers, pipelines, dispatch, command buffers, timing, probes, GEMV / GEMM wrappers |
| `src/core` | what the routed-expert LLMs share on top of it: expert arena, staging, routing profiles, telemetry reports |
| `src/glm` | GLM-5.3-Flash config and engine |
| `src/dsv41` | DeepSeek-V4.1 config, Engram tables and engine |
| `src/chat` | chat templates (the GGUF's Jinja template) |
| `src/server` | OpenAI- and Anthropic-compatible HTTP server |
| `src/gen` | the model interface, generation session, sampling |
| `src/text` | tokenizer, UTF-8 streaming detokenizer |
| `tools` | the programs above; Python generators of test data and ablation files |
| `tests` | ctest suite and its data |
| `bench` | SSD, matmul, GEMV and GEMM microbenchmarks; `perf_check` (speed regressions) |
| `docs` | these notes, architectures, performance work, the DeepSeek design, example presets |
