# Model providers

The same `agents/*.py`, `ctx.model`, Python sandbox and scoring commands work with
three provider settings. Edit `.env`; exported shell variables take precedence.
The lab never chooses a model automatically, falls back to another provider, or
retries failed API calls silently. Existing `.env` files default to OpenAI.

| PROVIDER | API | Required configuration |
| --- | --- | --- |
| `openai` | OpenAI Responses | `MODEL`, `OPENAI_API_KEY` |
| `openrouter` | Chat Completions | `MODEL`, `OPENROUTER_API_KEY` |
| `local` | OpenAI-compatible Chat Completions | `MODEL`, `MODEL_BASE_URL`; optional `LOCAL_API_KEY` |

## OpenAI

```dotenv
PROVIDER=openai
OPENAI_API_KEY=your-key
MODEL=gpt-5-nano-2025-08-07
```

This preserves the existing Responses integration. The endpoint is fixed to
OpenAI; `OPENAI_BASE_URL` does not redirect the lab's client.

## OpenRouter

```dotenv
PROVIDER=openrouter
OPENROUTER_API_KEY=your-openrouter-key
MODEL=provider/exact-model-id
```

Replace the placeholder with the exact ID of a model that supports function
calling from [OpenRouter's tool-capable model list](https://openrouter.ai/models?supported_parameters=tools).
The endpoint is `https://openrouter.ai/api/v1`. An OpenAI key is neither required
nor sent to OpenRouter. The adapter requests a tool action and disables parallel
tool calls; the harness validates the returned action regardless of provider.
Some models/endpoints lack this feature even though they accept ordinary chat.

OpenRouter can route a model to different infrastructure providers. For a formal
comparison, record the chosen routing policy/account settings and provider-returned
metadata; a model name alone does not prove an identical underlying deployment.
The lab retains the native response in each trace. It does not currently expose
OpenRouter routing preferences as additional CLI flags.

## Local or self-hosted models

Start a server that implements `/v1/chat/completions` with function calling. For
an already installed and running Ollama server:

```dotenv
PROVIDER=local
MODEL=your-installed-tool-capable-model
MODEL_BASE_URL=http://localhost:11434/v1
LOCAL_API_KEY=
```

For vLLM the endpoint is commonly `http://localhost:8000/v1`. For a shared lab
server, replace `localhost` with the address reachable from participant laptops;
use its issued `LOCAL_API_KEY` if authentication is enabled. Never put credentials
in the URL. A blank local key uses an inert SDK placeholder, never a hosted key.

Use the exact served model name. The server, model's chat template and tool-call
parser must agree; vLLM configurations depend on the model. Follow the server's
documentation rather than copying a generic parser flag. Download and start
models before the event. This repository provides the client integration; it
does not provision GPU infrastructure or download weights.

Local requests use automatic tool selection for compatibility. The agent prompt
asks for `python` or `submit_answer`; the harness still requires exactly one valid
function call. Plain text answers, multiple actions or malformed arguments fail
clearly as protocol errors. There is no silent text-to-tool conversion. A model
that cannot call functions is not compatible with these baseline agents.

Local inference and Python execution are separate: the model server runs on the
host or a remote machine, while generated Python runs in the existing Docker
sandbox. Docker is still required. Test local latency and context capacity with
several tool rounds; the lab cannot guarantee that a model fits a laptop or that
a crowded shared server fits the episode deadline.

## Verify a backend

```bash
uv run mathlab doctor --config config.hard.toml
uv run mathlab doctor --live-tools --config config.hard.toml
uv run mathlab run --agent agents/tool.py --config config.hard.toml \
  --questions data/smoke/questions.jsonl --out runs/provider-smoke
```

Use the score command printed at completion. The offline doctor checks settings
and Docker. `--live-tools` makes **two model requests and one Python invocation**:
it requires a Python call that prints 2, replays that observation, then requires a
correct submitted answer. This tests the complete round trip. A model that ignores
the tool instruction fails the diagnostic rather than giving a false success.
`doctor --live` remains available for a cheaper one-request submission check.
Inspect additional practice traces before approving a backend for the event.

`run.json` now records `provider_configuration` (provider, API, endpoint and model),
alongside the existing limits, agent hash and Docker image. Credentials are excluded.
Native Chat Completions responses are retained under `provider_response`; token
counts are normalized to the existing input/output/reasoning fields. Missing usage
remains unknown. Provider-returned reasoning details are preserved across tool turns;
the adapter does not invent hidden reasoning.

The output-token limit is passed as `max_output_tokens` to OpenAI Responses and
`max_tokens` to Chat Completions. Providers may account for reasoning differently.
Equal numeric limits across different backends are not automatically equal compute.
Use one pinned model/backend/configuration per competition track.

## Implementation verification

Automated tests use the real SDK with an offline HTTP transport to exercise
multi-turn tool calls for both new backends, usage, truncations, malformed actions,
missing usage, error redaction and disabled retries. This verifies the integration
contract; it is not evidence that a particular OpenRouter model or local server
has passed a live rehearsal. Configure and rehearse each chosen deployment before
the lab. No OpenRouter key or local model server was configured during implementation.

Official references:

- [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling)
- [OpenRouter tool calling](https://openrouter.ai/docs/guides/features/tool-calling)
- [OpenRouter reasoning details](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
- [Ollama OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)
- [vLLM tool calling](https://docs.vllm.ai/en/latest/features/tool_calling/)
