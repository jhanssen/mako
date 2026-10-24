# mako

An inference engine for very large mixture-of-experts language models on Apple Silicon Macs, written in C++ and Metal.

Targets: **GLM-5.3-Flash** (first) and **DeepSeek-V4.1-Flash-1017** (next), on Macs with 64-256 GB of unified memory.
Weights that don't fit in memory stream from the internal SSD into a unified-memory cache that the GPU reads directly.
It is tuned for M5 and meant to run on M3 and later (GPU family Apple9); so far it has only been tested on M4, and M3 is untested.

## Status

Early. GLM-5.3-Flash decodes one token at a time and matches a reference implementation to within that
reference's own precision. Routed experts are read from the SSD for every token (no cache yet), there is no tokenizer
(input is token ids), no batched prompt processing and no speculative decoding. Expect it to be slow.

## Requirements

- Apple Silicon Mac, macOS 26 or later
- Xcode with the Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`)
- CMake 4.3 or later, Ninja

## Build

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

The first configure downloads pinned dependencies (metal-cpp and third-party GEMV kernels).

## Run

```sh
./build/tools/mako_run <model-00001-of-0000N.gguf> <token,ids,...> [-n tokens-to-generate]
```

The model is a GLM-5.3-Flash GGUF using the standard `glm5-next` tensor names, e.g. the Maya-S quantization
(`peasantsmith/GLM-5.3-Flash-Maya-GGUF`, `Maya-S-v2-IQ2_XXS`).

## Tests

```sh
cmake -S . -B build -DMAKO_MODEL=<model-00001-of-0000N.gguf>   # or MAKO_MODEL in the environment at configure
ctest --test-dir build --output-on-failure                     # -L kernels | tokenizer | e2e
```

Every test needs the model and reports as skipped without it. The `e2e` tests (about a minute) check greedy decode
tokens, prefill against token-by-token decode, logits against stored reference top-k (`tests/data/parity_ref.bin`)
and a prompt past the dense-attention window. After an intended numerical change, `regress_test <model> --update`
re-records `tests/data/regress_expected.txt`; review that diff.

## Layout

| Path | Contents |
| --- | --- |
| `src/gguf` | GGUF reader, multi-file models |
| `src/mtl` | Metal context, GEMV wrapper, embedded shader libraries |
| `src/kernels` | Metal kernels |
| `src/glm` | GLM-5.3-Flash config and engine |
| `tools` | `mako_run`, `mako_calibrate`, `logit_compare`, `gguf_dump`, optional `ref_dump` for parity checks |
| `tests` | kernel, tokenizer and end-to-end regression tests (ctest) |
| `bench` | SSD, sync, matmul and quantization-format microbenchmarks |
