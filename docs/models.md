# Models

## Which to pick

| `mako-download.sh` set | Model | Size | Speed on a 128 GB M5 Max |
| --- | --- | --- | --- |
| `glm` | GLM-5.3-Flash, Maya-S-v2 IQ2_XXS | 96.5 GB | |
| `ds` | DeepSeek-V4.1-Flash-1017, Maya-S IQ2_XXS with its Engram tables | 339 GB | ~25 tokens/s decode (~19 at 16-32K tokens of context), ~500-800 tokens/s prefill |
| `ds-ista-xs` | DeepSeek-V4.1-Flash-1017, ISTA GSQ-RCO IQ2_XS | 266 GB | ~34 tokens/s decode with `--miss-penalty 0.1`, the same perplexity as Maya-S on our prose and code texts |
| `ds-ista-xxs` | DeepSeek-V4.1-Flash-1017, ISTA GSQ-RCO IQ2_XXS | 249 GB | ~33 tokens/s decode without the penalty (IQ2_XS: ~31), perplexity 3-5% higher than Maya-S; with the penalty unmeasured |

Measurements and how they were taken: [performance.md](performance.md). The fastest setup measured:

```sh
./build/tools/mako_calibrate ~/models/ds-ista-xs/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00006.gguf
./build/tools/mako_serve ~/models/ds-ista-xs/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00006.gguf \
    --miss-penalty 0.1 --port 8080 --ctx 65536
```

## Download

```sh
tools/mako-download.sh glm|ds|ds-ista-xs|ds-ista-xxs [--vision] [--dspark] [--ablate] [--dir DIR] [--dry-run]
```

`bin/mako-download` in the release archive. It downloads at the repository revisions mako was tested with, using the
`hf` CLI when it is installed (`--curl` to use curl anyway) and curl otherwise. It resumes interrupted files, checks
each file's SHA-256 and prints the commands (and a preset) to run the model. Files go to `~/models/<set>` unless `--dir`
says otherwise; `--dry-run` lists them and the space needed.

- `--vision`: the vision projector (1.0-1.2 GB; `mako_serve --mmproj FILE`, [vision.md](vision.md))
- `--dspark`: DeepSeek's drafter for speculative decoding (8 GB; `mako_serve --dspark DIR`)
- `--ablate`: refusal ablation directions ([options.md](options.md#ablation))

## Download by hand

- **GLM-5.3-Flash**: `Maya-S-v2-IQ2_XXS/` from
  [peasantsmith/GLM-5.3-Flash-Maya-GGUF](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF) (3 files).
- **DeepSeek-V4.1-Flash-1017**: `Maya-S/` from
  [peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF](https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF)
  (6 files, the model and its Engram tables), and optionally `dspark/` (the drafter).
- **DeepSeek-V4.1-Flash-1017, ISTA builds**: the five `IQ2_XS` (or `IQ2_XXS`) files and
  `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` from
  [ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF),
  in one folder, with a link to the Engram file as the build's sixth shard:
  `ln -s DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00006-of-00006.gguf`. The
  four-shard files published before October 27, 2026 load too. The `dspark/` drafter has not been tried with them.
- **Vision projectors**: GLM-5.3's `mmproj-BF16.gguf` from
  [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF); DeepSeek-V4.1's
  `mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf` from
  [ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)
  (it serves all the DeepSeek builds above).

## Tested hardware

Tuned for M5; should work on any Apple Silicon Mac (M1 and later).

| Mac | GLM-5.3 | DeepSeek-V4.1 |
| --- | --- | --- |
| M5 Max, 128 GB | all tests | all tests |
| M1 Ultra, 64 GB | e2e tests pass | not run (only the tests that need no model) |
| M4, 16 GB | the tests that fit pass; decodes the same tokens as the M5 (`--arena-gb 3`, below the memory it is meant for) | not run |
