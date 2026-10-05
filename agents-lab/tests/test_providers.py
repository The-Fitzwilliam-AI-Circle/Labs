"""Provider selection, SDK wire protocol, usage and round-trip tool history."""

import asyncio
import io
import json

import httpx
import pytest
from openai import AsyncOpenAI

from agents.tool import solve
from mathlab.cli import live_check, main
from mathlab.config import Limits, ProviderConfiguration, provider_configuration
from mathlab.model import ChatCompletionsModel, extract_usage, normalize_chat_response, parse_turn
from mathlab.runtime import AgentContext, TraceWriter
from mathlab.types import LabError, ProtocolError, ProviderError, Task, ToolResult
from tests.conftest import ROOT, ScriptedModel, response


@pytest.fixture
def clean_provider_env(monkeypatch):
    for name in (
        "PROVIDER",
        "MODEL",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "LOCAL_API_KEY",
        "MODEL_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MODEL", "explicit-model")


def test_explicit_provider_keys_and_defaults(clean_provider_env, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "hosted-secret")
    assert provider_configuration().provider == "openai"
    monkeypatch.setenv("PROVIDER", "openrouter")
    with pytest.raises(LabError, match="OPENROUTER_API_KEY"):
        provider_configuration()
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-secret")
    settings = provider_configuration()
    assert settings.api_key == "router-secret"
    assert settings.base_url == "https://openrouter.ai/api/v1"
    assert "router-secret" not in repr(settings)
    assert "router-secret" not in json.dumps(settings.public_metadata())
    monkeypatch.setenv("PROVIDER", "local")
    monkeypatch.setenv("MODEL_BASE_URL", "http://localhost:11434/v1/")
    settings = provider_configuration()
    assert settings.api_key == "local-no-key"
    assert settings.base_url == "http://localhost:11434/v1"


@pytest.mark.parametrize(
    "url",
    [
        "",
        "file:///tmp/model",
        "https://key@server/v1",
        "http://server/v1?key=secret",
        "http://server:bad/v1",
        "http://server:0/v1",
        "http://server/v1#secret",
    ],
)
def test_invalid_local_endpoint(clean_provider_env, monkeypatch, url):
    monkeypatch.setenv("PROVIDER", "local")
    monkeypatch.setenv("MODEL_BASE_URL", url)
    with pytest.raises(LabError, match="MODEL_BASE_URL"):
        provider_configuration()


def chat_response(name="submit_answer", value="2", *, finish="tool_calls"):
    return {
        "id": "chat-test",
        "object": "chat.completion",
        "created": 0,
        "model": "explicit-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_details": [
                        {"type": "reasoning.encrypted", "data": "opaque-signature"}
                    ],
                    "tool_calls": [
                        {
                            "id": "call-test",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(
                                    {"code" if name == "python" else "answer": value}
                                ),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "completion_tokens_details": {"reasoning_tokens": 3},
        },
    }


@pytest.mark.parametrize("provider", ["openrouter", "local"])
def test_real_sdk_tool_loop_and_usage(provider):
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-provider-key"
        assert body["max_tokens"] == 2048
        assert body["tool_choice"] == ("required" if provider == "openrouter" else "auto")
        assert body["parallel_tool_calls"] is False
        if len(requests) == 1:
            return httpx.Response(200, json=chat_response("python", "print(1+1)"))
        assert body["messages"][2]["reasoning_details"][0]["data"] == "opaque-signature"
        assert body["messages"][3]["role"] == "tool"
        assert body["messages"][3]["tool_call_id"] == "call-test"
        assert json.loads(body["messages"][3]["content"])["stdout"] == "2\n"
        return httpx.Response(200, json=chat_response())

    class Python:
        async def execute(self, code, **kwargs):
            assert code == "print(1+1)"
            return ToolResult(stdout="2\n", stderr="", exit_code=0)

    async def scenario():
        adapter = ChatCompletionsModel(
            "test-provider-key", "explicit-model", "http://test/v1", provider=provider
        )
        await adapter.close()
        adapter.client = AsyncOpenAI(
            api_key="test-provider-key",
            base_url="http://test/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        )
        stream = io.StringIO()
        ctx = AgentContext("test", Limits(), adapter, Python(), TraceWriter(stream))
        try:
            assert await solve(Task("test", "What is 1+1?"), ctx) == "2"
            assert ctx.model_requests == 2 and ctx.python_calls == 1
            assert ctx.usage.input_tokens == 20 and ctx.usage.output_tokens == 10
            assert ctx.usage.reasoning_tokens == 6
            assert "provider_response" in stream.getvalue()
        finally:
            await adapter.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r["choices"][0].update(finish_reason="length"),
        lambda r: r["choices"][0]["message"].update(tool_calls=[]),
        lambda r: r["choices"][0]["message"]["tool_calls"].append(
            r["choices"][0]["message"]["tool_calls"][0]
        ),
        lambda r: r["choices"][0]["message"]["tool_calls"][0]["function"].update(
            arguments='{"answer": 2}'
        ),
        lambda r: r.update(choices=[None]),
    ],
)
def test_invalid_chat_actions_keep_usage(mutation):
    raw = chat_response()
    mutation(raw)
    normalized = normalize_chat_response(raw)
    assert extract_usage(normalized)["output_tokens"] == 5
    with pytest.raises(ProtocolError):
        parse_turn(normalized, True)


def test_missing_usage_is_unknown():
    raw = chat_response()
    raw.pop("usage")
    assert extract_usage(normalize_chat_response(raw)) is None


def test_local_doctor_needs_no_hosted_key(clean_provider_env, monkeypatch, capsys):
    monkeypatch.setattr("mathlab.cli.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("mathlab.cli.inspect_image", lambda *_: "sha256:test")
    monkeypatch.setenv("PROVIDER", "local")
    monkeypatch.setenv("MODEL_BASE_URL", "http://localhost:11434/v1")
    assert main(["doctor", "--config", str(ROOT / "config.toml")]) == 0
    assert "OK provider: local" in capsys.readouterr().out


def test_sdk_error_does_not_retry_or_leak():
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(
            429, json={"error": {"message": "private-key-in-body", "code": "rate_limit_exceeded"}}
        )

    async def scenario():
        adapter = ChatCompletionsModel("test", "explicit-model", "http://test/v1", provider="local")
        await adapter.close()
        adapter.client = AsyncOpenAI(
            api_key="test",
            base_url="http://test/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        )
        try:
            with pytest.raises(ProviderError) as caught:
                await adapter.request(
                    [], "instructions", allow_python=True, max_output_tokens=20, timeout=2
                )
            assert "private-key-in-body" not in str(caught.value)
            assert len(calls) == 1
        finally:
            await adapter.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("stdout,passes", [("2\n", True), ("wrong\n", False)])
def test_live_doctor_requires_successful_python_round_trip(monkeypatch, stdout, passes):
    class Model(ScriptedModel):
        async def close(self):
            pass

    class Python:
        async def execute(self, code, **kwargs):
            assert code == "print(1 + 1)"
            return ToolResult(stdout=stdout, stderr="", exit_code=0)

    model = Model([response("python", "print(1 + 1)"), response(value="2")])
    monkeypatch.setattr("mathlab.cli.create_model", lambda *_: model)
    monkeypatch.setattr("mathlab.cli.DockerPython", lambda *_: Python())
    settings = ProviderConfiguration("local", "test", "http://localhost/v1", "unused")
    if passes:
        asyncio.run(live_check(Limits(), settings, image_id="test", check_tools=True))
        assert len(model.requests) == 2
    else:
        with pytest.raises(LabError, match="expected Python output"):
            asyncio.run(live_check(Limits(), settings, image_id="test", check_tools=True))
