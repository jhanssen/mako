# mako

An inference engine for very large mixture-of-experts models on Apple Silicon Macs (C++ and Metal), serving
**GLM-5.3-Flash** and **DeepSeek-V4.1-Flash-1017** through an OpenAI- and Anthropic-compatible API. Weights that don't
fit in memory stream from the SSD into a unified-memory expert cache.

Early, but usable: on a 128 GB M5 Max, DeepSeek-V4.1 decodes at ~25 tokens/s (~19 at 16-32K tokens of context) and
prefills ~500-800 tokens/s. Tuned for M5, meant for M3 and later: on a 16 GB M4 the tests that fit pass and GLM-5.3
decodes the same tokens as on the M5 (with `--arena-gb 3`, below the memory it is meant for); DeepSeek-V4.1 has run
only on the M5 Max, and M3 is untested.

## Requirements

- Apple Silicon Mac with 64-256 GB of memory, macOS 26 or later
- Xcode with the Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`)
- CMake 4.3 or later, Ninja

## Models

- **GLM-5.3-Flash**: `Maya-S-v2-IQ2_XXS/` from
  [peasantsmith/GLM-5.3-Flash-Maya-GGUF](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF) (3 files).
- **DeepSeek-V4.1-Flash-1017**: `Maya-S/` from
  [peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF](https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF)
  (6 files, the model and its Engram tables), and optionally `dspark/` (the drafter for speculative decoding).

## Build

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

The first configure downloads pinned dependencies.

## Run the server

```sh
./build/tools/mako_calibrate <first .gguf of the model>    # once per model: which experts start in memory
./build/tools/mako_serve <first .gguf of the model> --port 8080 --ctx 65536
```

Then point a client at `http://127.0.0.1:8080/v1` (OpenAI) or `http://127.0.0.1:8080` (Anthropic; Claude Code with
`ANTHROPIC_BASE_URL`). For DeepSeek's drafter add `--dspark <dspark dir>`. It listens on 127.0.0.1; for access
from other machines, see [Server](docs/server.md#access) (API key, encrypted transport).

Named option sets go in `~/Library/Application Support/mako/presets.json` (`--preset NAME`; examples in
`docs/presets.example.json`).

## More

- [Server](docs/server.md): API coverage, queueing, clients
- [Options](docs/options.md): memory, presets, DeepSeek speed/accuracy trade-offs, drafters, ablation
- [Development](docs/development.md): other tools, tests, self-check, source layout
- [Architecture](docs/architecture.md), [performance work](docs/performance.md),
  [DeepSeek design](docs/dsv41-design.md), [test data](tests/data/README.md)

## License

MIT (`LICENSE`). The GEMV kernels downloaded at configure are MIT-licensed by their authors; their license comes with
them.
