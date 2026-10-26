# mako

An inference engine for very large mixture-of-experts language models on Apple Silicon Macs, written in C++ and Metal.

Targets: **GLM-5.3-Flash** and **DeepSeek-V4.1-Flash-1017**, on Macs with 64-256 GB of unified memory.
Weights that don't fit in memory stream from the internal SSD into a unified-memory cache that the GPU reads directly.
It is tuned for M5 and meant to run on M3 and later (GPU family Apple9); so far it has only been tested on M4, and M3 is untested.

## Status

Early. GLM-5.3-Flash runs end to end: tokenizer and chat template, batched prompt processing, an expert cache in
unified memory, sampling and speculative decoding with the model's next-token block, matching a reference
implementation to within that reference's own precision. It serves an OpenAI-compatible API (`mako_serve`).

DeepSeek-V4.1-Flash-1017 runs decode and batched prompt processing (no speculative drafter yet), with its Engram
tables read per token, and serves through the same API. On a 128 GB M5 Max with the Maya-S IQ2_XXS quant (about 55%
of the experts resident) it decodes at 16-22 tokens/s and prefills a few thousand tokens at ~350 tokens/s.

The expert cache takes what is available at startup within a margin (a quarter of RAM from 96 GB up, 40% below);
`--arena-gb N` fixes its size instead, `--stage-gb N` limits the ring prompts' missing experts are read ahead into.

### DeepSeek-V4.1

The Maya-S set carries its Engram tables (Q6_K) in its last two shards. `--engram FILE` (`mako_serve`,
`mako_calibrate`) uses the tables of another GGUF instead (Q6_K or MXFP8). `ds_run`'s second argument names the
Engram GGUF; the model's first shard again uses its own.

```sh
build/tools/mako_calibrate <DeepSeek shard 1>                    # routing profile (once): which experts start resident
build/tools/mako_serve <DeepSeek shard 1> --ctx 65536 --port 8080
build/tools/ds_run <DeepSeek shard 1> <DeepSeek shard 1> --ids 0,671,6102 -n 32   # token ids in, greedy tokens out
```

`--ablate FILE` takes DeepSeek's sites `attn_out` (a layer's attention output) and `shexp_out` (its shared expert's
output).

## Requirements

- Apple Silicon Mac, macOS 26 or later
- Xcode with the Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`)
- CMake 4.3 or later, Ninja

## Build

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

The first configure downloads pinned dependencies (metal-cpp, third-party GEMV kernels, a Jinja engine, glaze and cpp-httplib).

## Run

```sh
./build/tools/mako_run <model-00001-of-0000N.gguf> <token,ids,...> [-n tokens-to-generate]
```

`--spec` decodes speculatively with the model's next-token (MTP) block: each step verifies the pending token and one
draft (about 14% faster on an M5 Max at ~67% of drafts accepted; output equals greedy decoding except where a
near-tied routing choice flips). It loads the block's weights and gives its experts cache slots. `--temp T`
(with `--top-k`, `--top-p`, `--min-p`, `--seed`) samples instead of decoding greedily; with `--spec`, drafts are
verified by rejection sampling, so the output has the same distribution as without them.

Programs generate through `glm::Session` (`src/glm/session.hpp`): it takes the whole prompt each time, keeps the
longest prefix the engine already holds (continuing the current state, or restoring a checkpoint of the
recurrent state taken before an earlier prompt's last token), samples, drafts and stops on stop tokens or a length limit.

The model is a GLM-5.3-Flash GGUF using the standard `glm5-next` tensor names, e.g. the Maya-S quantization
(`peasantsmith/GLM-5.3-Flash-Maya-GGUF`, `Maya-S-v2-IQ2_XXS`).

## Serve

```sh
./build/tools/mako_serve <model-00001-of-0000N.gguf> [--port 8080] [--ctx 131072] [--no-spec]
```

An OpenAI-compatible server at `http://127.0.0.1:8080/v1`: `POST /v1/chat/completions` (streaming or not, tool
calls, `reasoning_content`), `GET /v1/models`, `GET /health`. It also serves the Anthropic Messages API:
`POST /v1/messages` (text, thinking, tool_use and tool_result blocks, streaming events, `tool_choice` auto/any/tool/none;
thinking blocks only when `thinking` is enabled, its budget picking the template's low, high or maximum effort) and
`POST /v1/messages/count_tokens`. Image and document blocks and server tools are rejected. Claude Code works against
it (`ANTHROPIC_BASE_URL=http://127.0.0.1:8080`, any `ANTHROPIC_API_KEY`); its first request is a ~11K-token prompt.
`--dump-requests DIR` writes each request body there, for debugging a client. Prompts are rendered with the GGUF's chat template;
sampling defaults come from its `general.sampling.*` metadata. Requests run one at a time on one session, so a
conversation's next turn reuses the context its previous turn left; up to `--max-waiting` (default 4) wait behind
the one generating, more get HTTP 503 with `Retry-After`, and a waiting request whose client went away is dropped. `developer` messages are treated as `system`;
`reasoning_effort` medium maps to the template's high (it knows low and high, and reasons at its maximum
otherwise). SIGINT or SIGTERM stops the server; with `--learn` it then adds the session's routing counts to the model's profile, so
the experts that start resident follow how it was used. `chat_template_kwargs` adds template variables (GLM: `clear_thinking` drops the reasoning of earlier turns). `tool_choice`
`required` or a named function forces the answer to start with a call (of that function). Not supported (HTTP 400):
`n` > 1, `logprobs`, structured `response_format`.

`--ablate FILE` (also on `mako_run`) removes per-layer directions (abliteration) from chosen rows of the computation.
The file names its sites (`MAKOABL1`: n_embd, n_layer, site names, then float32 vectors [layer][site][k][n_embd]; see
`src/mtl/ablation.hpp`); each row loses x -= scale (x . v) v per direction, the vectors' norms carrying the strength and
`--ablate-scale` multiplying it. GLM-5.3's sites: `ffn_out` (a layer's FFN output) and `stream` (the hyper-connection
streams), both before the FFN output is mixed in; the next-token block is left alone. `tools/make_ablation.py` turns
raw float32 directions [layer][k][n_embd] into a file (`--sites ffn_out,stream` for GLM-5.3).

### Presets

`--preset NAME` (on `mako_serve`, `mako_run`, `ds_run`, `prefill_sweep`) applies a named set of options from
`~/Library/Application Support/mako/presets.json` (`--presets FILE` for another). Each `"--flag"` entry becomes that
option (`true`: the flag alone, `false`: left out), inserted where `--preset` stands, so options after it override
it and presets stack; `"model"` (a path or a list) is used when the command line names none; `~/` is the home
directory. `docs/presets.example.json` has presets for both models and the DeepSeek trade-offs below.

```sh
./build/tools/mako_serve --preset ds-route-0.1 --port 8091
```

DeepSeek-V4.1 speed/accuracy options (measured in `docs/performance.md`): `--miss-penalty F` lowers the routing
score of experts not in memory, so fewer are read from the SSD (0.05-0.2: 9-20% faster steps, KL against the
reference +20-80%; the output then depends on what is cached), and `--dense-quant q4_k|q5_k|q6_k` re-quantizes the
dense Q8_0 weights at load (`--imatrix FILE`: activation statistics from `ds_run --gen N --imatrix-out FILE`
weighting its rounding). Experts that the router two layers ahead predicts are read in the background (`--prefetch K`,
default 4; 0 turns it off); results are unchanged.

For pi, a provider in `~/.pi/agent/models.json` with `"api": "openai-completions"`, `"baseUrl":
"http://127.0.0.1:8080/v1"`, `"reasoning": true` and `"compat": {"supportsMidConvoSystemMessages": true}` (system
prompt changes then append instead of rewriting the start of the prompt, which would re-read the whole context).

## Self-check

```sh
python3 tools/self_check.py --model <model-00001-of-0000N.gguf> [--quick] [--out report.md]
```

Runs the test suite against the model and a fixed benchmark (prefill of 512 and 4096 new tokens on a 4K-token context,
decode of 128 tokens with and without drafts; each step a freshly loaded engine, so the decode numbers are the first
answer after a prompt, not a warmed-up session), and prints a Markdown report to paste into an issue. `--quick` runs
only the tests that need no GPU work.

## Tests

```sh
cmake -S . -B build -DMAKO_MODEL=<model-00001-of-0000N.gguf>   # or MAKO_MODEL in the environment at configure
ctest --test-dir build --output-on-failure                     # -L kernels | tokenizer | sampler | server | e2e
# MAKO_DS41_MODEL=<DeepSeek-V4.1 shard 1> (at configure) adds the DeepSeek tokenizer, template, Engram, GEMM,
# server and reference tests; MAKO_DS41_ENGRAM=<GGUF> runs them with other Engram tables, MAKO_DS41_DSPARK=<dir>
# adds the drafter's
```

Every test but `sampler` needs the model and reports as skipped without it. The `e2e` tests (about a minute) check
greedy decode tokens, prefill against token-by-token decode, logits against stored reference top-k
(`tests/data/*_ref.bin`), speculative steps, generation and prefix reuse through `Session`, and a prompt past the
dense-attention window (against the reference, not exact tokens). After an intended
numerical change, `regress_test <model> --update` re-records `tests/data/regress_expected.txt`; review that diff.

## Layout

| Path | Contents |
| --- | --- |
| `src/gguf` | GGUF reader, multi-file models |
| `src/mtl` | Metal context, GEMV wrapper, embedded shader libraries |
| `src/kernels` | Metal kernels |
| `src/core` | what the engines share: GPU runtime base, expert arena, routing profiles, telemetry |
| `src/glm` | GLM-5.3-Flash config and engine |
| `src/dsv41` | DeepSeek-V4.1 config, Engram tables and engine |
| `src/chat` | chat templates (the GGUF's Jinja template) |
| `src/server` | OpenAI- and Anthropic-compatible HTTP server |
| `src/gen` | the model interface, generation session, sampling |
| `src/text` | tokenizer, UTF-8 streaming detokenizer |
| `tools` | `mako_run`, `mako_serve`, `prefill_sweep`, `render_template`, `mako_calibrate`, `ds_run`, `engram_rows`, `logit_compare`, `gguf_dump`, optional `ref_dump` for parity checks |
| `tests` | kernel, tokenizer, chat template, server, sampler and end-to-end regression tests (ctest) |
| `bench` | SSD, matmul, GEMV and GEMM microbenchmarks |
