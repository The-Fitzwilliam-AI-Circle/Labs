import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from openai import APIConnectionError, RateLimitError

from mathlab.model import ResponsesModel, extract_usage, function_tools, parse_turn
from mathlab.types import ProtocolError, ProviderError
from tests.conftest import response


def test_tool_schemas_and_preserved_provider_items():
    assert [tool["name"] for tool in function_tools(False)] == ["submit_answer"]
    for tool in function_tools(True):
        assert tool["strict"]
        assert tool["parameters"]["additionalProperties"] is False
        assert tool["parameters"]["required"] == list(tool["parameters"]["properties"])
    raw = response("python", "print(42)")
    turn = parse_turn(raw, True)
    assert (turn.kind, turn.value, turn.call_id) == ("python", "print(42)", "call-1")
    assert turn.output_items == raw["output"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(status="incomplete"),
        lambda r: r.update(output=[]),
        lambda r: r["output"].append(r["output"][-1].copy()),
        lambda r: r["output"][-1].update(name="unknown"),
        lambda r: r["output"][-1].update(arguments="not JSON"),
        lambda r: r["output"][-1].update(arguments='{"answer": "1", "extra": "2"}'),
        lambda r: r["output"][-1].update(arguments='{"answer": 1}'),
        lambda r: r["output"][-1].update(arguments='{"answer":"1","answer":"2"}'),
        lambda r: r["output"][-1].update(arguments="[]"),
        lambda r: r["output"][-1].update(call_id=None),
        lambda r: r["output"][-1].update(status="in_progress"),
    ],
)
def test_protocol_rejects_ambiguous_or_malformed_actions(mutation):
    raw = response()
    mutation(raw)
    with pytest.raises(ProtocolError):
        parse_turn(raw, True)


def test_direct_rejects_python():
    with pytest.raises(ProtocolError, match="unavailable"):
        parse_turn(response("python", "print(1)"), False)


def test_unknown_usage_not_zero():
    assert extract_usage({"usage": None}) is None
    assert extract_usage({"usage": {"input_tokens": 2}}) is None
    assert extract_usage(response())["reasoning_tokens"] == 4


def test_sdk_request_disables_retries_and_preserves_history(monkeypatch):
    raw = response()
    client = SimpleNamespace(
        responses=SimpleNamespace(
            create=AsyncMock(
                return_value=SimpleNamespace(
                    model_dump=lambda **_: raw,
                )
            )
        ),
        close=AsyncMock(),
    )
    constructor = Mock(return_value=client)
    monkeypatch.setattr("mathlab.model.AsyncOpenAI", constructor)

    async def scenario():
        model = ResponsesModel("test-secret", "explicit-model-id")
        history = [{"role": "user", "content": "Question"}]
        assert (
            await model.request(
                history,
                "Instructions",
                allow_python=True,
                max_output_tokens=123,
                timeout=2,
            )
            == raw
        )
        await model.close()
        constructor.assert_called_once_with(
            api_key="test-secret", base_url="https://api.openai.com/v1", max_retries=0
        )
        kwargs = client.responses.create.call_args.kwargs
        assert kwargs["input"] == history
        assert kwargs["tool_choice"] == "required"
        assert kwargs["parallel_tool_calls"] is False
        assert kwargs["max_output_tokens"] == 123
        assert kwargs["model"] == "explicit-model-id"
        assert kwargs["store"] is False
        assert "reasoning.encrypted_content" in kwargs["include"]

    asyncio.run(scenario())


def test_sdk_errors_do_not_expose_credentials(monkeypatch):
    client = SimpleNamespace(
        responses=SimpleNamespace(
            create=AsyncMock(
                side_effect=APIConnectionError(
                    message="test-secret", request=httpx.Request("POST", "https://example.com")
                )
            )
        ),
        close=AsyncMock(),
    )
    monkeypatch.setattr("mathlab.model.AsyncOpenAI", lambda **_: client)

    async def scenario():
        model = ResponsesModel("test-secret", "test")
        with pytest.raises(ProviderError) as error:
            await model.request([], "", allow_python=False, max_output_tokens=100, timeout=1)
        assert "test-secret" not in str(error.value)
        await model.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "code, error_type, expected",
    [
        ("insufficient_quota", "insufficient_quota", "API quota is unavailable"),
        ("credit_balance_exhausted", "insufficient_quota", "No API credits remain"),
        ("project_spend_limit_exceeded", "insufficient_quota", "project's API spend limit"),
        ("organization_spend_limit_exceeded", "insufficient_quota", "organization API spend"),
        ("organization_usage_limit_exceeded", "insufficient_quota", "organization API usage"),
        ("rate_limit_exceeded", "rate_limit_error", "request or token rate limit"),
        ("slow_down", "rate_limit_error", "request rate increased too quickly"),
        (None, "insufficient_quota", "API quota is unavailable"),
        ("test-secret", "unknown", "API rate or quota limit"),
    ],
)
def test_rate_limit_diagnostics_distinguish_quota_without_exposing_body(
    monkeypatch,
    code,
    error_type,
    expected,
):
    error = RateLimitError(
        "Sensitive provider message: test-secret",
        response=httpx.Response(429, request=httpx.Request("POST", "https://example.com")),
        body={"code": code, "type": error_type, "message": "test-secret"},
    )
    create = AsyncMock(side_effect=error)
    client = SimpleNamespace(responses=SimpleNamespace(create=create), close=AsyncMock())
    monkeypatch.setattr("mathlab.model.AsyncOpenAI", lambda **_: client)

    async def scenario():
        model = ResponsesModel("test-secret", "test")
        try:
            with pytest.raises(ProviderError) as caught:
                await model.request([], "", allow_python=False, max_output_tokens=100, timeout=1)
            assert expected in str(caught.value)
            assert "test-secret" not in str(caught.value)
            assert "HTTP 429" in str(caught.value)
            assert create.await_count == 1
        finally:
            await model.close()

    asyncio.run(scenario())
