"""All provider-specific code lives here. Requests are stateless and never retried."""

from typing import Any

from openai import APIError, APITimeoutError, AsyncOpenAI

from mathlab.files import strict_json
from mathlab.types import EpisodeTimeout, ModelTurn, ProtocolError, ProviderError


def provider_error_message(error: APIError) -> str:
    """Explain known provider codes without printing response bodies or credentials."""
    status = getattr(error, "status_code", None)
    summary = f"{type(error).__name__} (HTTP {status})"
    if status == 429:
        reasons = {
            "credit_balance_exhausted": "No API credits remain; check API billing and credits.",
            "insufficient_quota": (
                "API quota is unavailable; check API billing, credits, and organization limits. "
                "Retrying without resolving the quota will not help."
            ),
            "organization_spend_limit_exceeded": "The organization API spend limit was reached.",
            "project_spend_limit_exceeded": "The project's API spend limit was reached.",
            "organization_usage_limit_exceeded": "The organization API usage limit was reached.",
            "rate_limit_exceeded": (
                "The API request or token rate limit was reached; wait before retrying "
                "and check your project's model limits."
            ),
            "slow_down": "The API request rate increased too quickly; slow down before retrying.",
        }
        # A specific code takes precedence over the broader insufficient_quota type.
        for code in (error.code, error.type):
            if isinstance(code, str) and code in reasons:
                return f"{summary}; code={code}. {reasons[code]} Usage is unknown."
        summary += "; API rate or quota limit; check API billing and model limits"
    return f"{summary}; usage is unknown"


def function_tools(allow_python: bool) -> list[dict]:
    definitions = [("submit_answer", "answer", "Return a candidate scalar rational answer.")]
    if allow_python:
        definitions.insert(0, ("python", "code", "Run a self-contained Python 3.12 program."))
    return [
        {
            "type": "function",
            "name": name,
            "description": description,
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {argument: {"type": "string"}},
                "required": [argument],
                "additionalProperties": False,
            },
        }
        for name, argument, description in definitions
    ]


def parse_turn(response: dict, allow_python: bool) -> ModelTurn:
    if response.get("status") != "completed":
        raise ProtocolError("Model response was not completed; inspect raw_response in the trace")
    output = response.get("output")
    if not isinstance(output, list) or any(not isinstance(item, dict) for item in output):
        raise ProtocolError("Model output must be a list of items")
    if any(item.get("type") not in {"reasoning", "message", "function_call"} for item in output):
        raise ProtocolError("Unexpected output item or unavailable tool")
    calls = [item for item in output if item.get("type") == "function_call"]
    if len(calls) != 1:
        raise ProtocolError(f"Expected exactly one function call; received {len(calls)}")
    call = calls[0]
    name = call.get("name")
    if name not in ({"python", "submit_answer"} if allow_python else {"submit_answer"}):
        raise ProtocolError("Unknown or unavailable function tool")
    if call.get("status") not in (None, "completed"):
        raise ProtocolError("Incomplete function call")
    call_id = call.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        raise ProtocolError("Missing function call_id")
    try:
        arguments = strict_json(call["arguments"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Function arguments must be valid JSON") from exc
    argument = "code" if name == "python" else "answer"
    if (
        not isinstance(arguments, dict)
        or set(arguments) != {argument}
        or not isinstance(arguments[argument], str)
    ):
        raise ProtocolError(f"Function arguments must contain only the string {argument!r}")
    return ModelTurn(
        kind="python" if name == "python" else "answer",
        value=arguments[argument],
        call_id=call_id,
        output_items=output,
    )


def extract_usage(response: dict) -> dict[str, int] | None:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    if any(
        type(usage.get(k)) is not int or usage[k] < 0 for k in ("input_tokens", "output_tokens")
    ):
        return None
    details = usage.get("output_tokens_details") or {}
    if not isinstance(details, dict):
        details = {}
    reasoning = details.get("reasoning_tokens", 0)
    return {
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "reasoning_tokens": reasoning if type(reasoning) is int and reasoning >= 0 else 0,
    }


class ResponsesModel:
    def __init__(self, api_key: str, model: str):
        self.model = model
        self.client = AsyncOpenAI(
            api_key=api_key, base_url="https://api.openai.com/v1", max_retries=0
        )

    async def request(
        self,
        history: list[dict[str, Any]],
        instructions: str,
        *,
        allow_python: bool,
        max_output_tokens: int,
        timeout: float,
    ) -> dict:
        try:
            response = await self.client.responses.create(
                model=self.model,
                instructions=instructions,
                input=history,
                tools=function_tools(allow_python),
                tool_choice="required",
                parallel_tool_calls=False,
                max_output_tokens=max_output_tokens,
                store=False,
                include=["reasoning.encrypted_content"],
                timeout=timeout,
            )
        except APITimeoutError as exc:
            raise EpisodeTimeout("Model request timed out; usage is unknown") from exc
        except APIError as exc:
            # Avoid logging SDK exception bodies, headers, or credentials.
            raise ProviderError(provider_error_message(exc)) from exc
        return response.model_dump(mode="json", exclude_none=True)

    async def close(self) -> None:
        await self.client.close()


def chat_history(history: list[dict], instructions: str) -> list[dict]:
    """Translate the agent protocol while preserving native assistant tool/reasoning data."""
    messages = [{"role": "system", "content": instructions}]
    for item in history:
        if item.get("type") == "function_call_output":
            messages.append(
                {"role": "tool", "tool_call_id": item["call_id"], "content": item["output"]}
            )
        elif item.get("type") == "function_call":
            if "chat_message" in item:
                messages.append(item["chat_message"])
            else:
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": item["call_id"],
                                "type": "function",
                                "function": {"name": item["name"], "arguments": item["arguments"]},
                            }
                        ],
                    }
                )
        elif item.get("role") in {"user", "assistant", "system", "developer"}:
            content = item.get("content", "")
            if isinstance(content, list):
                if any(
                    part.get("type") not in {"input_text", "output_text", "text"}
                    for part in content
                ):
                    raise ProtocolError("Chat adapter supports text-only messages")
                content = "\n".join(part["text"] for part in content)
            messages.append({"role": item["role"], "content": content})
        else:
            raise ProtocolError("History item cannot be replayed to a Chat Completions provider")
    return messages


def normalize_chat_response(raw: dict) -> dict:
    """Keep raw provider evidence and usage even when the action is invalid or truncated."""
    normalized = {"status": "completed", "output": [], "provider_response": raw}
    usage = raw.get("usage")
    if isinstance(usage, dict):
        normalized["usage"] = {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "output_tokens_details": usage.get("completion_tokens_details"),
        }
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return normalized
    choice = choices[0]
    if not isinstance(choice, dict):
        return normalized
    finish = choice.get("finish_reason")
    if finish not in {"stop", "tool_calls"}:
        normalized["status"] = "incomplete"
        normalized["incomplete_details"] = {
            "reason": "max_output_tokens" if finish == "length" else str(finish)
        }
    message = choice.get("message") or {}
    if not isinstance(message, dict):
        return normalized
    # Preserve provider-returned reasoning fields/signatures verbatim across tool turns.
    replay = {
        key: message[key]
        for key in (
            "role",
            "content",
            "tool_calls",
            "reasoning",
            "reasoning_content",
            "reasoning_details",
        )
        if key in message
    }
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list) or any(not isinstance(call, dict) for call in calls):
        return normalized
    for call in calls:
        function = call.get("function") or {}
        if not isinstance(function, dict) or call.get("type") != "function":
            normalized["output"] = []
            return normalized
        normalized["output"].append(
            {
                "type": "function_call",
                "name": function.get("name"),
                "call_id": call.get("id"),
                "arguments": function.get("arguments"),
                "chat_message": replay,
            }
        )
    return normalized


class ChatCompletionsModel:
    def __init__(self, api_key: str, model: str, base_url: str, *, provider: str):
        self.model = model
        self.provider = provider
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)

    async def request(self, history, instructions, *, allow_python, max_output_tokens, timeout):
        tools = []
        for definition in function_tools(allow_python):
            # Strict-schema support varies on compatible servers; the harness still
            # validates every returned action identically to the OpenAI adapter.
            function = {k: v for k, v in definition.items() if k not in {"type", "strict"}}
            tools.append({"type": "function", "function": function})
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=chat_history(history, instructions),
                tools=tools,
                tool_choice="required" if self.provider == "openrouter" else "auto",
                parallel_tool_calls=False,
                max_tokens=max_output_tokens,
                timeout=timeout,
            )
        except APITimeoutError as exc:
            raise EpisodeTimeout("Model request timed out; usage is unknown") from exc
        except APIError as exc:
            raise ProviderError(provider_error_message(exc)) from exc
        return normalize_chat_response(response.model_dump(mode="json", exclude_none=True))

    async def close(self):
        await self.client.close()
