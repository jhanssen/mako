# Options

## Memory

The expert cache takes what is available at startup within a margin (a quarter of RAM from 96 GB up, 40% below).
`--arena-gb N` fixes its size; `--stage-gb N` limits the ring a prompt's missing experts are read ahead into.
`--page-cache` (default on; `--no-page-cache` to bypass it): a step's expert misses and reads ahead go through the
OS page cache, so an expert evicted from the expert cache and wanted again soon comes back at memory speed when RAM is
left over beyond it (performance.md: -12% a step with 19 GB left over, nothing measurable when the expert cache takes
what is free); filling the expert cache and a prompt's staged reads bypass the page cache either way.

## Waiting for the GPU

`--spin-wait` (default on; `--no-spin-wait` to block instead): the thread that runs a step spins on the command
buffer's status until the GPU finishes it, at most 250 ms, then blocks. Measured 0.1-0.4 ms a step on DeepSeek-V4.1
and nothing on GLM-5.3 (performance.md); spinning holds a core for the step's GPU time.

`--keepalive` (`mako_serve`, default on; `--no-keepalive` turns it off): while no request is generating, a kernel a
second reads one word per MiB of the resident weights and the expert cache, which keeps the GPU's mappings of them.
Without it the first step after a few seconds idle took ~0.37 s with a 75 GB expert cache instead of the usual tens
of ms (performance.md); the kernel costs ~0.3 ms of GPU a second per 24 GB.

## Presets and the environment

The tools (`mako_serve`, `mako_calibrate`, `mako_run`, `ds_run`, `prefill_sweep`) take their options
from, lowest precedence first, environment variables, presets and the command line. `--help` lists each tool's options
with their variables.

- Each long option has a variable: `--arena-gb 40` is `MAKO_ARENA_GB=40`, `--miss-penalty 0.1`
  `MAKO_MISS_PENALTY=0.1`. A flag's variable takes `1` / `0` (or `true`, `yes`, `on` / `false`, `no`, `off`); every flag
  has a `--no-` form (`--spec`, `--no-spec`). Short options (`-n`) have none, and empty variables are ignored.
- An option the command line gives, or a preset it names, ignores its variable. For options that add up
  (`--api-key`, `--api-key-file`) the command line's replace the environment's.
- Options taken from the environment are listed on stderr at startup (secrets without their value).

`--preset NAME` applies a named set of options from `~/Library/Application Support/mako/presets.json` (`--presets
FILE` for another file; `MAKO_PRESET` and `MAKO_PRESETS` when the command line names none); `presets.example.json`
here has examples for both models. mako-image's programs read the same file, and their variables start with
`MAKO_IMAGE_` instead (`MAKO_IMAGE_PORT`, `MAKO_IMAGE_PRESET`), so the two servers' settings stay apart.

- Each `"--flag": value` entry becomes that option (`true`: the flag alone, `false`: left out), inserted where
  `--preset` stands, so later options override it and presets stack.
- `"model"` (a path or a list) is used when the command line names none; `~/` is the home directory.

```sh
./build/tools/mako_serve --preset ds-route-0.1 --port 8091
MAKO_ARENA_GB=40 ./build/tools/mako_serve ~/models/glm/model-00001-of-00003.gguf
```

## DeepSeek-V4.1 speed and accuracy

Measured in `performance.md`.

- `--miss-penalty F`: lowers the routing score of experts not in memory, so fewer are read from the SSD. 0.05-0.2:
  9-20% faster steps, KL against the reference +20-80%; the output then depends on what is cached. 0.1 with the
  defaults: ~29 tokens/s on a short prompt instead of ~25.
- `--dense-quant q4_k|q5_k|q6_k`: re-quantizes the dense Q8_0 weights at load; `--imatrix FILE` (from `ds_run --gen N
  --imatrix-out FILE`) weights the rounding.
- `--prefetch K` (default 4; 0: off): reads ahead the experts the router D layers later predicts (`--prefetch-ahead
  D`, default 2). Results unchanged.
- `--prefill-skip 0|1` (default 1): a long prompt's earlier rows skip the layers after the last KV source, which only
  the last few thousand rows need. Results unchanged.
- `--engram FILE` (`mako_serve`, `mako_calibrate`): the Engram tables of another GGUF (Q6_K or MXFP8) instead of the
  model's own.

## Speculative decoding

- GLM-5.3 drafts with its next-token (MTP) block, on by default in `mako_serve` (`--no-spec` turns it off; `--spec`
  on `mako_run`): one draft a step, about 14% faster on an M5 Max at ~67% of drafts accepted. Greedy output equals
  plain decoding except where a near-tied routing choice flips; sampled output keeps its distribution (rejection
  sampling). The block's weights load and its experts take cache slots.
- DeepSeek-V4.1 drafts with DSpark (`--dspark DIR`, `mako_serve` and `ds_run`): up to 4 drafts a step; its 8 GB come out
  of the expert cache, so it helps on code (+13%) more than on prose (-3%).
- DeepSeek-V4.1 also drafts from the context itself (`--lookup-draft N`, default 4; 0: off), with or without DSpark:
  when the reply repeats an earlier stretch of the conversation (a file written back with edits, quoted tool output),
  the tokens that followed it are the drafts, 3 to 7 a step by the length of the match (at least N tokens). Each
  match length stops offering while under 55% of its first drafts are kept (one probe in eight), so text that merely
  shares a phrase with the context stops paying for rejected drafts after a few. No weights, no memory; the output is
  the same as without (the engine verifies every draft). GLM-5.3 verifies one draft a step and does not use it.
- Hy4-flash drafts from the context the same way (on with `--spec`, the default): a reply repeating a 4.6 KB file from
  its prompt 25.4 -> 14.8 ms/token (94% of drafts kept), a story unchanged. A verify's rows and the steps after its
  rejected tokens are undone are bit-identical to decoding the same tokens one at a time.
- Hy4-flash drafts with Tencent's DFly2 (`--dfly2 FILE`, `mako_serve` and `hy4f_run`; `mako-download.sh hy4f --dfly2`
  fetches the file, `model-00148-of-00148.safetensors` of tencent/Hy4-flash, 1.7 GB, held as 0.9 GB of Q8_0 plus the
  token embeddings on the GPU): 5 layers reading the KV and indexer choices of target layers 25-41, up to 5 drafts a
  step while each one's confidence is at least 0.5 and their product at least 0.5. Drafting pauses while under 0.4
  drafts a call are kept (one step in 8 tries again). Measured (M5 Max, Maya-M, 500 tokens): through the server code
  39.5 -> 50.8 tok/s, a story about even (39.6 -> 39.2-40.5 sampled); `hy4f_run --spec` code 40.8 -> 56.8 (77% of
  drafts kept, 3.6 tokens a round), prose about even. Greedy output is the same as without. A verify of n tokens costs
  ~24 / 30 / 38 / 48 / 58 / 75 ms for n = 1-6 (each token's 16 experts read on their own), a draft round ~7-8 ms.

## Ablation

`--ablate FILE` (`mako_serve`, `mako_run`, `ds_run`) removes per-layer directions from chosen rows of the computation:
x -= scale (x . v) v per direction, the vectors' norms carrying the strength, `--ablate-scale` multiplying it
(`mako_serve` and `mako_run`).

- Format (`MAKOABL1`, `src/mtl/ablation.hpp`): n_embd, n_layer, site names, then float32 vectors
  [layer][site][k][n_embd]. `tools/make_ablation.py` builds one from raw [layer][k][n_embd] directions,
  `tools/hy4f_ablation.py` one for Hy4-flash from per-stream directions [layers + 1][4][k][n_embd].
- Refusal directions: [skafte/mako-ablations](https://huggingface.co/skafte/mako-ablations) has one file per model
  (`mako-download.sh <model> --ablate`); the DeepSeek file serves both Maya-S and ISTA's builds.
- GLM-5.3 sites: `ffn_out` (a layer's FFN output) and `stream` (the hyper-connection streams), both before the FFN
  output is mixed in; the next-token block is left alone.
- DeepSeek-V4.1 sites: `attn_out` (a layer's attention output) and `shexp_out` (its shared expert's output).
- Hy4-flash sites: `stream0` to `stream3`, each hyper-connection stream entering a layer (after its Engram, before
  the attention's mix), and `head0` to `head3`, the streams before the output head (stored at the last layer). Each
  stream has its own directions; the four of a kind come in order. `mako_serve` only.

## Other model files

- GLM-5.3: any GGUF with the standard `glm5-next` tensor names. The Maya repository's other quants (Maya-S24, Maya-M,
  Maya-L) use expert formats mako has kernels for but have not been run; the tests use Maya-S-v2.
- DeepSeek-V4.1: ISTA's GSQ-RCO builds (IQ2_XS, IQ2_XXS, IQ1_M) with their IQ4_NL Engram file; IQ2_XS and IQ2_XXS are
  measured in `performance.md`, IQ1_M has not been run. Maya-XS (IQ1_S experts) uses formats mako has kernels for but
  has not been run.
