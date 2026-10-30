# Server

```sh
./build/tools/mako_serve <first .gguf of the model> [--host 127.0.0.1] [--port 8080] [--ctx 131072] [--no-spec]
                         [--image-model <Qwen-Image 2.1 dir>]
```

One request runs at a time on one session, so a conversation's next turn reuses the context the previous one left.

## Access

The server listens on 127.0.0.1 unless `--host` says otherwise.

- **API keys**: `--api-key KEY[,KEY...]`, `--api-key-file FILE` (a key per line; keeps keys out of the shell history
  and the process list), or `MAKO_API_KEY` / `MAKO_API_KEY_FILE` (when neither option is given). Every request but `/health` and the `/status` page then needs one of them:
  `Authorization: Bearer KEY` (OpenAI clients' API key setting) or `x-api-key: KEY` (Anthropic clients, Claude Code's
  `ANTHROPIC_API_KEY`); others get HTTP 401. Several keys let one client's key be withdrawn. Keys under 16 characters
  get a warning (`openssl rand -hex 24` makes a good one). Another `--host` requires a key.
- **Without a key**, requests a web page could make through your browser are refused (HTTP 403): an `Origin` header
  that is not a loopback page, or a `Host` header that is not `localhost`, `127.0.0.1` or `[::1]` (against DNS
  rebinding). Clients on this machine (curl, SDKs, Claude Code) send a loopback `Host` and no `Origin`, so they are
  unaffected.

The key travels in clear text: for access from other machines, use an encrypted path (an SSH tunnel, Tailscale, or a
reverse proxy with TLS) rather than exposing the port. Not meant for the open internet.

## Status

`http://127.0.0.1:8080/status` is a page that shows `GET /v1/status` (JSON), refreshed every 2 seconds:

- the model, the context size, uptime and macOS's thermal state (`nominal`, `fair`, `serious` or `critical`);
- requests in flight and waiting for a turn, and each running one's phase (`waiting`, `reading` the prompt,
  `generating`), prompt and generated tokens and decode rate;
- the expert cache: size, the share of experts it holds, hits and misses so far;
- the conversations: live (the engine's context) or parked, in use, tokens, parked bytes against `--park-gb`, last
  use;
- the last 16 requests: how each ended, its conversation, prompt and reused tokens, prefill and decode rates, and
  experts read from the SSD per generated token.

The page itself holds no data and needs no key; with `--api-key`, it asks for one and keeps it in the browser's local
storage for `/v1/status`.

## OpenAI

- `POST /v1/chat/completions` (streaming, tool calls, `reasoning_content`), `GET /v1/models`, `GET /health`.
- `developer` messages count as `system`; `reasoning_effort` as in [Thinking](#thinking).
- `chat_template_kwargs` adds template variables (GLM: `clear_thinking` drops earlier turns' reasoning); thinking keys
  there (`reasoning_effort`, `enable_thinking`, `thinking`, `thinking_mode`) replace the request's effort.
- `tool_choice` `required` or a named function forces a call.
- Not supported (HTTP 400): `n` > 1, `logprobs`, structured `response_format`.

## Anthropic

- `POST /v1/messages` (text, thinking, tool_use / tool_result, streaming, `tool_choice`) and
  `POST /v1/messages/count_tokens`; thinking as in [Thinking](#thinking).
- Image and document blocks and server tools are rejected.
- Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`, any `ANTHROPIC_API_KEY` (its first request is a ~11K-token
  prompt).

## Images

`--mmproj FILE` loads the model's vision projector (docs/vision.md): DeepSeek-V4.1's
`mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf`, GLM-5.3's `mmproj-BF16.gguf`; ~1 GB, loaded before the expert cache is
sized. Images come as OpenAI `image_url` parts with a `data:` URL (base64) or Anthropic `image` blocks with base64
data (in user messages and in `tool_result` content); URLs are not fetched. PNG, JPEG and the other formats ImageIO
reads. Each image becomes its rows in the prompt (652 for a 1024x1024 image with DeepSeek, 1369 with GLM; DeepSeek
at most 1024, GLM at most 8000); the last 8 encoded images are kept, so a conversation that sends the same image
again does not encode it again, and its prompt prefix is reused only up to an image that differs. Without
`--mmproj` a request with an image gets HTTP 400. `--image-max-tokens N` scales larger images down to at most N
rows (encoding and prefill grow with the rows, GLM's encoder faster than linearly: its 3200x2400 at the default
8000 takes ~18 s, at 2048 ~4 s).

Context an image takes (rows; GLM keeps the image's own resolution up to its limit, DeepSeek scales it into its
block):

| Image | GLM-5.3 | DeepSeek-V4.1 |
| --- | ---: | ---: |
| 3200x2400 | ~7900 (at most 8000) | ~1000 |
| 2000x2000 | ~5200 | ~1000 |
| 1920x1080 | ~2700 | ~1000 |
| 1024x1024 | 1369 | 652 |
| 512x512 | ~350 | ~200 |

The log gives each new image's rows (`image 1036x1036: 1369 rows in 572 ms`) and each request's prompt
(`prompt 1395 tokens (1380 reused)`). A new image costs its encoding (~0.3 s at 1024x1024, ~3 s for GLM at
2000x2000) plus the prefill of its rows (GLM ~570 rows/s; DeepSeek ~100/s on a short prompt, its missing experts
read from the SSD); later turns reuse it.

## Image generation

`--image-model DIR` (Qwen-Image 2.1's directory, its files as published) adds `POST /v1/images/generations`, OpenAI's
image API:

```sh
curl -s localhost:8080/v1/images/generations -H 'Content-Type: application/json' \
  -d '{"prompt": "a misty harbour town at dawn", "size": "1024x1024"}' | jq -r '.data[0].b64_json' | base64 -d > out.png
```

- Fields: `prompt` (required), `n` (1-4), `size` (`WIDTHxHEIGHT` in multiples of 32 from 256, at most 2048x2048
  pixels; `auto`: 1024x1024), `response_format` (`b64_json` only: images are not stored), `output_format` (`png` or
  `jpeg`). `model`, `quality`, `style`, `background`, `moderation` and `user` are accepted and left alone: 40 steps,
  a random seed for each image (in the log).
- One model is in memory at a time. An image request waits for the chat requests in progress (chat requests that
  come meanwhile wait for it), then the LLM is unloaded, its conversations staying in memory, and the image model
  loaded (~5 s); the next chat request loads the LLM again (~10 s) and continues its conversation where it was. The
  status page shows which is loaded.
- Time: ~13 s for 512x512, ~63 s for 1024x1024 (40 steps), slower as the Mac heats up; the log has each image's
  progress.
- `tools/pi/scene-image` is a pi skill that turns the conversation into an image prompt and renders it through this
  endpoint (`ln -s "$PWD/tools/pi/scene-image" ~/.pi/agent/skills/`; then ask for an image of the scene, or
  `/skill:scene-image`). Its script saves to `~/Pictures/mako` (`MAKO_IMAGE_DIR`) and opens the image; `MAKO_URL`
  and `MAKO_API_KEY` point it at the server.
- LoRAs: `--image-lora FILE[:SCALE]` (repeatable) merges each into the image model when it loads; Diffusers, PEFT and
  ComfyUI key names. `--image-steps N` and `--image-terminal F` set the schedule: a 4-step distilled LoRA such as
  lightx2v's Qwen-Image-2.1-Lightning takes `--image-steps 4 --image-terminal 0` (1024x1024 in ~7 s).
- `--image-text-encoder DIR` takes the text encoder from another directory of safetensors (the same architecture
  with changed weights, such as a refusal-ablated one), leaving the model's own directory as published.

## Thinking

At startup the server renders the chat template to see what it does with the thinking controls and logs it
(`chat template: reasoning can be skipped; reasoning_effort by name (medium included) or 1..100` for DeepSeek-V4.1;
GLM-5.3's template knows low and high and always reasons).

| Request | DeepSeek-V4.1 (effort shown) | GLM-5.3 |
| --- | --- | --- |
| no effort | `--reasoning-effort`, else the template's default (75) | `--reasoning-effort`, else Max |
| `reasoning_effort` `none`, `minimal`, `0` | no reasoning (`</think>` in the prompt) | low |
| `low`, `medium`, `high`, `max` | 50, 62, 75, 100 | Low, High (medium), High, Max |
| a number 1..100 | that number | HTTP 400 |
| Anthropic `thinking` disabled or absent | no reasoning | low (hidden) |
| Anthropic `thinking` enabled, `budget_tokens` <= 4096 / <= 16384 / more | low / high / as no effort | the same |

`--reasoning-effort none|low|medium|high|max|N` sets the effort for requests that name none; a value the template
cannot take stops the server at startup. The effort is what the model was trained to follow, not a token limit.

A budget is the limit: `--reasoning-budget N` for requests that name none, `reasoning_budget_tokens` (or
`thinking_budget_tokens`) in a Chat Completions request, `thinking.budget_tokens` in an Anthropic one (0: none). In
the last stretch before it (a quarter of the budget, at most 1024 tokens) the reasoning is closed as soon as
`</think>` is among the model's three likeliest tokens; at the budget the server writes "I have to stop thinking here
and give my answer." and `</think>`, and the model answers. A reply capped at 1024 tokens or more also keeps half of the
cap for the answer, budget or not (GLM-5.3 at a cap of 1024 still ran out answering after 512 of reasoning). The log says when either cut the reasoning.
A cut in the middle of a calculation can leave the answer wrong.

## Behaviour

- Prompts use the GGUF's chat template, sampling defaults its `general.sampling.*` metadata.
- Up to `--max-waiting` requests (default 4) wait besides the one generating; more get HTTP 503 with `Retry-After`; a
  waiting request whose client left is dropped.
- Each conversation keeps its context between requests, so several clients (two pi sessions, an agent and its
  subagents) can take turns without one re-reading the other's prompt. A request continues the conversation that holds
  the most of its prompt: all of it when the prompt extends it, else up to its last checkpoint (the end of its last
  prompt: another try at the last turn); with none, it starts a new one. The others are parked in memory: GLM-5.3 ~25
  KB a token plus ~0.14 GB of recurrent state per copy (the parked state, the latest checkpoint and the one at the
  fork point), DeepSeek-V4.1 far less. `--park-gb` bounds them (default: what the expert cache leaves when every
  expert fits, 1-8 GB; GLM-5.3 on 128 GB gets ~2.9 GB at the default 131072-token context and ~5.5 GB at 32768, every
  expert still resident), `--conversations` their number (default 16); the least recently used is given up first. A
  conversation keeps a checkpoint where its first user turn starts (after the system prompt and tools); a new
  conversation that starts the same way begins from a copy of it and reads only the rest (sharing that checkpoint),
  the other conversation left as it is (the same tokens and logits as reading it all, bit for bit).
- Requests take turns: one generates at a time, and while others wait it gives the engine up every `--slice-ms`
  (default 1000) at a decode step or between pieces of its prompt (cut at multiples of the engine's prefill group,
  4096 tokens for GLM-5.3 and 8192 for DeepSeek-V4.1, so a long read yields every few seconds), then queues again.
  A second request starts within about a slice instead of after the first one's whole reply; a switch takes ~9 ms
  (GLM-5.3, short contexts). Interleaving changes no result: each request gives the tokens and logits it gives alone,
  bit for bit (tests `e2e_pool`, `ds41_pool`: two threads switching at every step), except with `--miss-penalty`,
  whose routing depends on which experts the other conversations left in memory. The log line of each request names
  its conversation and its turns.
- A prompt read for longer than 5 seconds logs its progress every 5 seconds: `chatcmpl-148: prompt read 40960 of
  79411 tokens (52%), 532 tok/s, ~72 s left`, the rate over the tokens read so far, waits for other requests' turns
  included. Within a prefill group (4096 tokens for GLM-5.3, 8192 for DeepSeek-V4.1) the count is the group's share of
  the layers done.
- A stream with nothing to send for 5 seconds (waiting for its turn, reading a long prompt, reasoning the client does
  not see) gets an SSE comment line (`: keepalive`), which clients skip, so clients that drop idle streams (pi's HTTP
  client after 300 s) keep the connection. A long prompt is polled between prefill groups, every ~10-20 s.
- Repetition loops (`--loop-guard N`, default 256; 0: off): the reasoning or the answer's text (not tool calls, not
  fenced code blocks) ending in a run that repeats at least 4 times, exactly over N tokens or more, or with 90% of its
  tokens the same over 2N (a loop with a counter). A loop in the reasoning closes it (`</think>`) and the answer
  follows; a loop in the answer ends the reply with "stop" rather than running on to "length" (on which pi compacts).
  A whole reply is cut back to the loop's first time through; a stream ends where the loop was found (~N tokens of
  it already sent). Each is logged. Over 473 reasoning and answer blocks of GLM-5.3 and DeepSeek-V4.1 pi sessions
  (~200K tokens) it found none; over this repository's sources (~450K tokens, code not fenced) one, a block of
  near-identical template instantiations.
- `--learn` adds the session's routing counts to the model's profile at shutdown (SIGINT / SIGTERM), so the experts
  that start in memory follow use.
- `--dump-requests DIR` saves each request body, for debugging a client.

## Clients

pi: a provider in `~/.pi/agent/models.json` with `"api": "openai-completions"`, `"baseUrl":
"http://127.0.0.1:8080/v1"`, `"reasoning": true` and `"compat": {"supportsMidConvoSystemMessages": true}` (a changed
system prompt is then appended instead of rewriting the prompt's start, which would re-read the whole context).
With `--mmproj`, `"input": ["text", "image"]` on the model lets pi send images (attachments and `read` results),
resized to at most 2000x2000 JPEG; `"inputLimits": {"images": {"resize": {"maxWidth": 1344, "maxHeight": 1344}}}`
keeps a GLM image to ~2300 rows.
