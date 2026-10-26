# Options

## Memory

The expert cache takes what is available at startup within a margin (a quarter of RAM from 96 GB up, 40% below).
`--arena-gb N` fixes its size; `--stage-gb N` limits the ring a prompt's missing experts are read ahead into.

## Presets

`--preset NAME` (`mako_serve`, `mako_run`, `ds_run`, `prefill_sweep`) applies a named set of options from
`~/Library/Application Support/mako/presets.json` (`--presets FILE` for another file); `presets.example.json` here
has examples for both models.

- Each `"--flag": value` entry becomes that option (`true`: the flag alone, `false`: left out), inserted where
  `--preset` stands, so later options override it and presets stack.
- `"model"` (a path or a list) is used when the command line names none; `~/` is the home directory.

```sh
./build/tools/mako_serve --preset ds-route-0.1 --port 8091
```

## DeepSeek-V4.1 speed and accuracy

Measured in `performance.md`.

- `--miss-penalty F`: lowers the routing score of experts not in memory, so fewer are read from the SSD. 0.05-0.2:
  9-20% faster steps, KL against the reference +20-80%; the output then depends on what is cached. 0.1 with the
  defaults: ~29 tokens/s on a short prompt instead of ~25.
- `--dense-quant q4_k|q5_k|q6_k`: re-quantizes the dense Q8_0 weights at load; `--imatrix FILE` (from `ds_run --gen N
  --imatrix-out FILE`) weights the rounding.
- `--prefetch K` (default 4; 0: off): reads ahead the experts the router two layers later predicts. Results unchanged.
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

## Ablation

`--ablate FILE` (`mako_serve`, `mako_run`, `ds_run`) removes per-layer directions from chosen rows of the computation:
x -= scale (x . v) v per direction, the vectors' norms carrying the strength, `--ablate-scale` multiplying it.

- Format (`MAKOABL1`, `src/mtl/ablation.hpp`): n_embd, n_layer, site names, then float32 vectors
  [layer][site][k][n_embd]. `tools/make_ablation.py` builds one from raw [layer][k][n_embd] directions.
- GLM-5.3 sites: `ffn_out` (a layer's FFN output) and `stream` (the hyper-connection streams), both before the FFN
  output is mixed in; the next-token block is left alone.
- DeepSeek-V4.1 sites: `attn_out` (a layer's attention output) and `shexp_out` (its shared expert's output).

## Other model files

- GLM-5.3: any GGUF with the standard `glm5-next` tensor names. The Maya repository's other quants (Maya-S24, Maya-M,
  Maya-L) use expert formats mako has kernels for but have not been run; the tests use Maya-S-v2.
- DeepSeek-V4.1: ISTA's GSQ-RCO builds (IQ2_XS, IQ2_XXS, IQ1_M) with their IQ4_NL Engram file; IQ2_XS and IQ2_XXS are
  measured in `performance.md`, IQ1_M has not been run. Maya-XS (IQ1_S experts) uses formats mako has kernels for but
  has not been run.
