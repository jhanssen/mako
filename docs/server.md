# Server

```sh
./build/tools/mako_serve <first .gguf of the model> [--host 127.0.0.1] [--port 8080] [--ctx 131072] [--no-spec]
```

One request runs at a time on one session, so a conversation's next turn reuses the context the previous one left.
The server has no authentication; it listens on 127.0.0.1 unless `--host` says otherwise.

## OpenAI

- `POST /v1/chat/completions` (streaming, tool calls, `reasoning_content`), `GET /v1/models`, `GET /health`.
- `developer` messages count as `system`; `reasoning_effort` medium maps to the template's high.
- `chat_template_kwargs` adds template variables (GLM: `clear_thinking` drops earlier turns' reasoning).
- `tool_choice` `required` or a named function forces a call.
- Not supported (HTTP 400): `n` > 1, `logprobs`, structured `response_format`.

## Anthropic

- `POST /v1/messages` (text, thinking, tool_use / tool_result, streaming, `tool_choice`) and
  `POST /v1/messages/count_tokens`; the `thinking` budget picks low, high or maximum effort.
- Image and document blocks and server tools are rejected.
- Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`, any `ANTHROPIC_API_KEY` (its first request is a ~11K-token
  prompt).

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
