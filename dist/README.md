# @NAME@

An inference engine for very large mixture-of-experts models on Apple Silicon Macs, serving GLM-5.3-Flash and
DeepSeek-V4.1-Flash-1017 through an OpenAI- and Anthropic-compatible API. Built from commit
[@COMMIT@](https://github.com/jhanssen/mako/tree/@COMMIT@).

## Requirements

- An Apple Silicon Mac (M1 or later) with 64 GB of memory or more, macOS 26 or later
- A model, downloaded separately: which files to get is in the
  [README](https://github.com/jhanssen/mako/blob/@COMMIT@/README.md#models)

## Install

Files downloaded with a browser are quarantined, and macOS stops these unsigned programs until that is cleared: a
dialog says "mako_serve" Not Opened (Apple could not verify it) with Move to Trash and Done; click Done, and the shell
then prints `killed`. Clear it once for the whole folder (Safari unpacks the archive itself; otherwise
`tar xzf @ARCHIVE@.tar.gz` first):

```sh
xattr -dr com.apple.quarantine @NAME@
```

## Run

```sh
@NAME@/bin/mako_calibrate <first .gguf of the model>    # once per model: which experts start in memory
@NAME@/bin/mako_serve <first .gguf of the model> --port 8080 --ctx 65536
```

Then point a client at `http://127.0.0.1:8080/v1` (OpenAI) or `http://127.0.0.1:8080` (Anthropic). `--help` lists
each program's options; each can also come from the environment (`--arena-gb 40` is `MAKO_ARENA_GB=40`).
`examples/presets.json` has named option sets for both models (`--presets examples/presets.json --preset NAME`).

More: [server](https://github.com/jhanssen/mako/blob/@COMMIT@/docs/server.md),
[options](https://github.com/jhanssen/mako/blob/@COMMIT@/docs/options.md).

## Licenses

mako is MIT licensed (`LICENSE`); the third-party code built into it is listed with its licenses in
`THIRD_PARTY_LICENSES.txt`.
