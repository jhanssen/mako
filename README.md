# mako

```
                                                   :
                                               :*%@@:
                                             :#@@@@@-
                                           :*@@@@@@@=
                                          =@@@@@@@@@*                                   =#%%
                                        :%@@@@@@@@@@*                                :*@@@@@
                                      +@@@@@@@@@@@@%:                             -#@@@@@*
                            :::::-++++@@@@@@@@@@@@@@@@=                          =%@@@@@@=
                    :---*##%@@@@@@@@@@@@@@@@@@@@@@@@@@@#+::    :++:            =%@@@@@@@*
             --+*#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@####@@@#=         =@@@@@@@@@+
     -=+*#%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%***-:::+@@@@@@@@@@+
  :@@@@@@@@@@@@%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@=
   :=+=-=*#%@@%-@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@=
      -+-    :=#%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@###@@@@@@@@@@+
        :--%@##-- ::-=#####%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%@@@@@@#  :   :*@@@@@@@@@=
            :=--==--      ::-#@@@@@@@@@@@@%%%%%%%%%%###===: :+#@@@#         :*%@@@@@@@=
                    :=------=+%@@@@@@@@@%-                      --:            =%@@@@@@
                               +@@@@@@@@                                         =#@@@@@:
                                :#@@@@@@                                            +%@@#
                                  :*@@@@=                                              --
                                    :=@@*
```

Runs mixture-of-experts models larger than your Mac's memory.

- **Models:** GLM-5.3-Flash and DeepSeek-V4.1-Flash-1017
- **API:** OpenAI- and Anthropic-compatible (works with Claude Code)
- **Speed:** DeepSeek-V4.1 decodes at ~25-34 tokens/s on a 128 GB M5 Max
- **Optional:** image input, image generation (Qwen-Image 2.1, Krea 2)

Early, but usable.

## Requirements

- Apple Silicon Mac (M1 or later) with 64-256 GB of memory, macOS 26 or later
- Xcode with the Metal toolchain: `xcodebuild -downloadComponent MetalToolchain`
- CMake 4.3 or later, Ninja
- 100-340 GB of disk for a model

## Quick start

**1. Build** (or take `mako_serve` and `mako_calibrate` from the [nightly release](https://github.com/jhanssen/mako/releases/tag/nightly)):

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

**2. Download a model:**

```sh
tools/mako-download.sh glm
```

| Set | Model | Size |
| --- | --- | --- |
| `glm` | GLM-5.3-Flash (Maya-S-v2) | 97 GB |
| `ds` | DeepSeek-V4.1-Flash-1017 (Maya-S) | 339 GB |
| `ds-ista-xs` | DeepSeek-V4.1-Flash-1017 (ISTA IQ2_XS, the fastest: ~34 tokens/s) | 266 GB |
| `ds-ista-xxs` | DeepSeek-V4.1-Flash-1017 (ISTA IQ2_XXS, the smallest) | 249 GB |

Files go to `~/models/<set>`. The script downloads with the `hf` CLI when it's installed (and uses its saved login),
otherwise with curl. It resumes interrupted downloads, checks every file's SHA-256 and, when done, prints the commands
to run the model. Options:

- `--vision`: add the vision projector, for images as input
- `--dspark`: add DeepSeek's drafter, for speculative decoding
- `--dir DIR`: download somewhere else
- `--dry-run`: list the files and the space needed

More on the models, and downloading by hand: [docs/models.md](docs/models.md).

**3. Calibrate once, then serve** (the download script prints these commands for the model you chose):

```sh
./build/tools/mako_calibrate ~/models/glm/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf
./build/tools/mako_serve ~/models/glm/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf --port 8080 --ctx 65536
```

**4. Connect a client:**

- OpenAI: `http://127.0.0.1:8080/v1`
- Anthropic: `http://127.0.0.1:8080` (for Claude Code, set `ANTHROPIC_BASE_URL`)

`mako_serve --help` lists every option.

## Docs

| | |
| --- | --- |
| [Models](docs/models.md) | which to pick, downloads, tested hardware |
| [Server](docs/server.md) | API coverage, remote access, image generation, clients |
| [Options](docs/options.md) | memory, presets and environment variables, speed/accuracy trade-offs, drafters |
| [Vision](docs/vision.md) | images as prompt input |
| [Development](docs/development.md) | tools, tests, source layout |
| [Architecture](docs/architecture.md), [Performance](docs/performance.md), [DeepSeek design](docs/dsv41-design.md) | how it works |

## License

MIT (`LICENSE`). The GEMV kernels downloaded at configure time are MIT-licensed by their authors; their license comes
with them.
