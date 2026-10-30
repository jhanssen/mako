# Development

## Tools

Besides `mako_serve` and `mako_calibrate`, `build/tools` has programs for working on the engines. `--help` gives each
one's options.

- `mako_run` (GLM-5.3) and `ds_run` (DeepSeek-V4.1): run token ids through the engine; top logits, greedy or sampled
  tokens. `ds_run` also compares logits with a reference dump, measures decode widths and collects an imatrix.
- `mako_image`: the image models (Qwen-Image 2.1, Krea 2) from the command line.
- `vision_run`, `vision_chat`: the vision encoder on an image, and a chat with images.
- `prefill_sweep`: prefill timing over prompt lengths (`--parts` for the GPU time by part).
- Inspection: `gguf_dump`, `tokenize`, `render_template`, `engram_rows`, `logit_compare`.
- `ref_dump` and `ref_vision`: logits and image embeddings from the reference implementation, when CMake is given its
  checkout (`MAKO_REF_SRC`, `MAKO_REF_BUILD`; `ref_vision` needs its `mtmd` library built).
- Python, in `tools/`: the generators of test data (`gen_*.py`), `make_ablation.py`, `gguf_retag.py`, `self_check.py`.

Programs generate through `gen::Session` (`src/gen/session.hpp`), for either model: it takes the whole prompt each
time, keeps the longest prefix the engine already holds (continuing the current state, or restoring a checkpoint of
the state taken before an earlier prompt's last token), samples, drafts and stops on stop tokens or a length limit.

## Tests

```sh
cmake -S . -B build -DMAKO_MODEL=<GLM-5.3 .gguf> -DMAKO_DS41_MODEL=<DeepSeek .gguf>   # or the same environment variables
ctest --test-dir build --output-on-failure -L <label>
```

Tests whose model is not configured report as skipped. What each model variable enables:

| Variable | Tests |
| --- | --- |
| (none) | `-L cpu` (no GPU either; what CI runs) and `-L kernels` (GPU kernels on random data) |
| `MAKO_MODEL` (GLM-5.3) | `e2e_*`, the tokenizer, templates, GEMV/GEMM on real weights |
| `MAKO_DS41_MODEL` (DeepSeek-V4.1) | `ds41_*`, `dsv41`, its tokenizer, template and Engram tests. Optional: `MAKO_DS41_ENGRAM` (other Engram tables), `MAKO_DS41_DSPARK` (the drafter), `MAKO_DS41_ABLATION` |
| `MAKO_QWEN_IMAGE`, `MAKO_QWEN_IMAGE_REF` | `image_*`, `safetensors` |
| `MAKO_KREA2`, `MAKO_KREA2_TRANSFORMER`, `MAKO_KREA2_REF` | `krea_*` |

The image tests compare with reference tensors from the original pipeline (fp32 on the CPU), written by a Python
script kept beside the model, not in this repository. `ctest -N -L <label>` lists a label's tests, `ctest
--print-labels` the labels; `tests/CMakeLists.txt` says what each test checks.

Which to run:

| Change | While working | Before committing |
| --- | --- | --- |
| Kernels, `src/core`, `src/glm`, `src/dsv41` | `-L quick` | `-L e2e`, and `bench/perf_check` for each model the change reaches |
| The prefill skip, prompt lengths past a group | `-L quick` | `-L e2e` and `-L slow` |
| `src/server`, `src/gen` | `-L server` and `-L cpu` | the same, and `-L e2e` for `gen::Session` or the pool |
| Tokenizer, templates | `-L tokenizer` | the same |
| `src/vision` | `-L vision` | the same |
| `src/image`, `src/kernels/image.metal`, `src/kernels/ops.metal` | `-L image` | the same, and `bench/perf_check` for the LLMs when a shared kernel or `src/gpu` changed |

`tests/data/README.md` says how each data file was made. After an intended numerical change,
`regress_test <model> --update` re-records `tests/data/regress_expected.txt`; review that diff.

## Self-check

```sh
python3 tools/self_check.py --model <GLM-5.3 .gguf> [--quick] [--out report.md]
```

For GLM-5.3: runs the tests and a fixed benchmark (prefill of 512 and 4096 new tokens on a 4K-token context, decode of
128 tokens with and without drafts, each a freshly loaded engine) and prints a Markdown report to paste into an issue.
`--quick` runs only the tests that need no GPU work, and no benchmark.

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

Run it before committing a change to `src/kernels`, `src/core`, `src/glm` or `src/dsv41`. On the M5 Max, against its
own baseline, every share agreed within 0.1 point, so a flagged part is worth a look even when the total is under the
threshold (the `kda_prep` regression fixed in `ce60b6e` moved GLM's prefill only 3.6%, and was flagged by its part).
It does not cover long contexts; `prefill_sweep --parts` does.

## Layout

| Path | Contents |
| --- | --- |
| `src/gguf`, `src/safetensors` | model file readers (GGUF, multi-file models; safetensors) |
| `src/quant` | CPU dequantization and the dense re-quantizers |
| `src/app` | command-line options: the environment, presets |
| `src/mtl` | Metal context, GEMV/GEMM wrappers, ablation vectors, embedded shader libraries |
| `src/kernels` | Metal kernels |
| `src/gpu` | what any model's GPU code shares: buffers, pipelines, dispatch, command buffers, timing |
| `src/core` | what the routed-expert LLMs share on top of it: expert arena, staging, routing, telemetry |
| `src/glm`, `src/dsv41` | the GLM-5.3 and DeepSeek-V4.1 engines (DeepSeek's Engram tables too) |
| `src/gen` | the model interface, generation session, session pool, sampling |
| `src/text`, `src/chat` | tokenizer and streaming detokenizer; chat templates |
| `src/vision` | images as input: decoding and preprocessing, the vision encoder and projector, prompts with images |
| `src/image` | image generation: the shared engine, Qwen-Image 2.1, Krea 2 |
| `src/server` | the HTTP server: OpenAI and Anthropic APIs, model swaps |
| `tools`, `bench`, `tests` | the programs above; microbenchmarks and `perf_check`; the ctest suite and its data |

`docs/architecture.md` says how the engines fit together.
