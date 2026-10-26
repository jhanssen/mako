# Development

## Tools

```sh
./build/tools/mako_run <GLM-5.3 .gguf> <token,ids,...> [-n N] [--spec] [--temp T]   # GLM-5.3: top logits, greedy tokens
./build/tools/ds_run <DeepSeek .gguf> <Engram .gguf> --ids I,J,... [-n N]          # DeepSeek-V4.1 (the Engram argument:
                                                                                    # the model's first file again)
```

`mako_run` samples with `--temp T` (`--top-k`, `--top-p`, `--min-p`, `--seed`). `ds_run` also compares logits with a
reference dump (`--ref`), measures decode widths and collects an imatrix (`--imatrix-out`). Others: `prefill_sweep`
(prefill timing), `gguf_dump`, `tokenize`, `render_template`, `engram_rows`, `logit_compare`, and `ref_dump` when
CMake is given a reference checkout (`MAKO_REF_SRC`, `MAKO_REF_BUILD`).

Programs generate through `gen::Session` (`src/gen/session.hpp`), for either model: it takes the whole prompt each
time, keeps the longest prefix the engine already holds (continuing the current state, or restoring a checkpoint of
the state taken before an earlier prompt's last token), samples, drafts and stops on stop tokens or a length limit.

## Tests

```sh
cmake -S . -B build -DMAKO_MODEL=<GLM-5.3 .gguf> -DMAKO_DS41_MODEL=<DeepSeek .gguf>   # or the same environment variables
ctest --test-dir build --output-on-failure                                           # -L kernels | tokenizer | sampler | server | e2e
```

`MAKO_DS41_DSPARK=<dir>` adds the drafter's test, `MAKO_DS41_ENGRAM=<GGUF>` runs DeepSeek's with other Engram tables.

- No model needed: the kernel checks on random data (`quant`, `gemv_own`, `gemm_synthetic`, `engram_decode`),
  `sampler`, `presets`. The others report as skipped without their model.
- GLM-5.3's `e2e` tests (about a minute): greedy decode tokens, prefill against token-by-token decode, logits against
  stored reference top-k, speculative steps, generation and prefix reuse through `Session`, a prompt past the
  dense-attention window.
- DeepSeek's `ds41_*` tests (about seven minutes): logits against the reference's on 1300 tokens (steps and prefill),
  on a 24K-token prompt's last 1024 positions, with the miss penalty and with dense Q6_K; the drafter's greedy output.

`tests/data/README.md` says how each data file was made. After an intended numerical change,
`regress_test <model> --update` re-records `tests/data/regress_expected.txt`; review that diff.

## Self-check

```sh
python3 tools/self_check.py --model <GLM-5.3 .gguf> [--quick] [--out report.md]
```

For GLM-5.3: runs the tests and a fixed benchmark (prefill of 512 and 4096 new tokens on a 4K-token context, decode of
128 tokens with and without drafts, each a freshly loaded engine) and prints a Markdown report to paste into an issue.
`--quick` runs only the tests that need no GPU work.

## Layout

| Path | Contents |
| --- | --- |
| `src/gguf` | GGUF reader, multi-file models |
| `src/quant` | CPU dequantization and the dense re-quantizers |
| `src/app` | command-line presets |
| `src/mtl` | Metal context, GEMV wrapper, embedded shader libraries |
| `src/kernels` | Metal kernels |
| `src/core` | what the engines share: GPU runtime base, expert arena, routing profiles, telemetry |
| `src/glm` | GLM-5.3-Flash config and engine |
| `src/dsv41` | DeepSeek-V4.1 config, Engram tables and engine |
| `src/chat` | chat templates (the GGUF's Jinja template) |
| `src/server` | OpenAI- and Anthropic-compatible HTTP server |
| `src/gen` | the model interface, generation session, sampling |
| `src/text` | tokenizer, UTF-8 streaming detokenizer |
| `tools` | the programs above; Python generators of test data and ablation files |
| `tests` | ctest suite and its data |
| `bench` | SSD, matmul, GEMV and GEMM microbenchmarks |
| `docs` | these notes, architectures, performance work, the DeepSeek design, example presets |
