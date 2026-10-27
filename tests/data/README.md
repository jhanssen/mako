# Test data: what each file is and how it was made

The model-backed tests compare mako against reference implementations running the same GGUF files. The files here
hold the inputs and the reference's outputs; the reference itself is not needed to run the tests, only to regenerate
them. All of it was made with the Maya-S quants (see the README): GLM-5.3-Flash `GLM-5.3-Flash-Maya-S-v2-IQ2_XXS`
and DeepSeek-V4.1-Flash-1017 `DeepSeek-V4.1-Flash-1017-Maya-S-IQ2_XXS`. The DeepSeek reference engine read the
Engram tables in MXFP8 (the `UD-Q8_K_XL` build's shard 8 of 9); the current Maya-S set carries them in Q6_K in its
last two shards (its model shards are byte for byte the ones the references were made with), and the tests run with
those. Other quants of the same models will not stay within the logit limits.

## Reference logits (`*_ref.bin`)

Format: `uint32 positions`, `uint32 k` (64), then per position `float lse` (the log-sum-exp of the full logits) and
`k` pairs `(int32 id, float logit)`, best first. Tests compute KL(reference ‖ mako) over those k tokens (the "top-64
mass" they print is how much of the reference's probability that covers, ~0.99) and top-1 agreement.

### GLM-5.3 (`parity_ref.bin`, `long_ref.bin`)

- Prompts: `parity_ids.txt` (512 tokens), `long_ids.txt` (2391 tokens, past the dense-attention window).
- Reference: llama.cpp at commit `9ff0d7e90f0faaa7c16017e6dcf9cf91c50a100e` (the same commit the GEMV kernels are
  pinned to), CPU backend, through `tools/ref_dump` (built when CMake gets `-DMAKO_REF_SRC=<llama.cpp checkout>
  -DMAKO_REF_BUILD=<its build directory>`):

  ```sh
  build/tools/ref_dump/ref_dump <GLM shard 1> $(cat tests/data/long_ids.txt) /tmp/ref_long
  build/tests/regress_test - --import-ref /tmp/ref_long long
  ```

  `ref_dump` runs the ids one at a time (or `REF_BATCH=N` at a time, logits only, much faster for long prompts) and
  writes `logits.<pos>.f32` per position; `--import-ref` keeps the top 64 of each. (Which of the two modes made the
  committed files was not recorded.) The reference checkout has since moved to `7aca0d2ea8d92b56911233b822db01f9fbb1ae50`
  (2026-10-26, for DeepSeek-V4.1 vision); one token at a time it gives the first 32 positions of `parity_ref.bin`
  bit for bit.

### DeepSeek-V4.1 (`ds41_short_ref.bin`, `ds41_long_ref.bin`)

- Prompts: `ds41_short_ids.txt` (1300 tokens of prose), `ds41_long_ids.txt` (24576 tokens of mako's own sources;
  the reference file holds its last 1024 positions, where layer 20's candidate blocks restrict the later indexers).
  Token 0 is BOS.
- Reference: DeepSeek-V4.1 support in mainline llama.cpp at that commit covers V4 only (no Engram), so the logits come
  from a CUDA engine that runs these exact files (project-maya's `maya-ds4`, on an RTX 5090):

  ```sh
  maya-ds4 --pack <Maya-S dir>/maya-ds4 --ids $(cat ds41_long_ids.txt) --n 0 --dump-logits ref.f32 \
           --dspark none --max-context 32768
  build/tests/ds41_ref_test import ref.f32 tests/data/ds41_long_ref.bin 1024    # the last 1024 positions
  ```

  `--dump-logits` writes every position's logits as float32 rows of 129,280 (12.7 GB for the long prompt).

## Recorded outputs

- `regress_expected.txt`: greedy tokens recorded from mako itself (`regress_test <model> --update`), to catch
  unintended changes; not reference data.

## Tokenizers

- `glm_tokenizer_cases.bin`: 2051 cases (separated by 0x1E) from `tools/gen_tokenizer_cases.py`: scripts, emoji,
  numbers, whitespace runs, special tokens.
- `glm_tokenizer_expected.txt`, `ds41_tokenizer_expected.txt`: the reference tokenizer's ids for those cases, one line
  per case: `REF_TOKENIZE=tests/data/glm_tokenizer_cases.bin build/tools/ref_dump/ref_dump <model> 0 /tmp >
  expected.txt`. For DeepSeek, whose architecture that llama.cpp does not load, `tools/vocab_only_gguf.py` first
  writes a GGUF carrying only the model's `tokenizer.ggml.*` keys (architecture "llama", dummy geometry).

## Chat templates

- `glm_chat_cases.bin`: conversations rendered with jinja2 the way Hugging Face's `apply_chat_template` configures it
  (`tools/gen_chat_template_cases.py`).
- `ds41_template/`: DeepSeek's own encoder fixtures, unchanged (see the README there).

## Engram (`ds41_engram_expected.txt`)

The hashed table rows of a token sequence ("t layer rows..." per position and Engram layer, then "V t layer
values..." for the first column's decoded row of the first positions), from project-maya's Python n-gram hasher
(`tools/ds4/engram_hash.py` there), given the hash parameters (primes, multipliers, token map, offsets) read from the
Engram GGUF's `deepseek41.engram.*` keys. The decoded values are from the Maya-S set's Q6_K tables, by
`tools/gen_engram_values.py` (an independent decode of the raw bytes). `tools/engram_rows` prints mako's side for
comparison.
