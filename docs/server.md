# Server

```sh
./build/tools/mako_serve <first .gguf of the model> [--host 127.0.0.1] [--port 8080] [--ctx 131072] [--no-spec]
```

One request runs at a time on one session, so a conversation's next turn reuses the context the previous one left.

## Access

The server listens on 127.0.0.1 unless `--host` says otherwise.

- **API keys**: `--api-key KEY[,KEY...]`, `--api-key-file FILE` (a key per line; keeps keys out of the shell history
  and the process list) or `MAKO_API_KEY` (when neither is given). Every request but `/health` then needs one of them:
  `Authorization: Bearer KEY` (OpenAI clients' API key setting) or `x-api-key: KEY` (Anthropic clients, Claude Code's
  `ANTHROPIC_API_KEY`); others get HTTP 401. Several keys let one client's key be withdrawn. Keys under 16 characters
  get a warning (`openssl rand -hex 24` makes a good one). Another `--host` requires a key.
- **Without a key**, requests a web page could make through your browser are refused (HTTP 403): an `Origin` header
  that is not a loopback page, or a `Host` header that is not `localhost`, `127.0.0.1` or `[::1]` (against DNS
  rebinding). Clients on this machine (curl, SDKs, Claude Code) send a loopback `Host` and no `Origin`, so they are
  unaffected.

The key travels in clear text: for access from other machines, use an encrypted path (an SSH tunnel, Tailscale, or a
reverse proxy with TLS) rather than exposing the port. Not meant for the open internet.

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
- Up to `--max-waiting` requests (default 4) wait behind the one generating; more get HTTP 503 with `Retry-After`; a
  waiting request whose client left is dropped.
- `--learn` adds the session's routing counts to the model's profile at shutdown (SIGINT / SIGTERM), so the experts
  that start in memory follow use.
- `--dump-requests DIR` saves each request body, for debugging a client.

## Clients

pi: a provider in `~/.pi/agent/models.json` with `"api": "openai-completions"`, `"baseUrl":
"http://127.0.0.1:8080/v1"`, `"reasoning": true` and `"compat": {"supportsMidConvoSystemMessages": true}` (a changed
system prompt is then appended instead of rewriting the prompt's start, which would re-read the whole context).
