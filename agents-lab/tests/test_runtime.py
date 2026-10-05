import asyncio
import io
import json

import pytest

from mathlab.config import Limits
from mathlab.runtime import AgentContext, TraceWriter
from mathlab.types import BudgetExceeded, EpisodeTimeout, ProviderError, ToolResult
from tests.conftest import ScriptedModel, ScriptedPython, response


def context(model, python=None, **limits):
    return AgentContext(
        "task", Limits(**limits), model, python or ScriptedPython(), TraceWriter(io.StringIO())
    )


def test_new_conversations_and_failed_attempts_share_budget():
    model = ScriptedModel([ProviderError("offline failure"), response()])
    ctx = context(model, max_model_requests=2)

    async def scenario():
        with pytest.raises(ProviderError):
            await ctx.model([], "first candidate")
        await ctx.model([], "fresh verifier")
        with pytest.raises(BudgetExceeded):
            await ctx.model([], "one more candidate")

    asyncio.run(scenario())
    assert len(model.requests) == ctx.model_requests == 2
    assert ctx.usage.unknown_usage_requests == 1
    assert ctx.usage.output_tokens == 10
    assert ctx.usage.reasoning_tokens == 4


def test_tool_budget_and_observations_allow_recovery():
    python = ScriptedPython(ToolResult(stderr="NameError", exit_code=1))
    ctx = context(ScriptedModel([]), python, max_python_calls=1)

    async def scenario():
        assert (await ctx.python("bad_code")).stderr == "NameError"
        with pytest.raises(BudgetExceeded):
            await ctx.python("print(1)")

    asyncio.run(scenario())
    assert len(python.calls) == ctx.python_calls == 1


def test_inflight_model_obeys_deadline_and_unknown_usage():
    class SlowModel:
        async def request(self, *args, **kwargs):
            await asyncio.sleep(1)

    ctx = context(SlowModel(), episode_timeout_seconds=0.02)
    with pytest.raises(EpisodeTimeout):
        asyncio.run(ctx.model([], ""))
    assert ctx.model_requests == ctx.usage.unknown_usage_requests == 1


def test_exhausted_deadline_prevents_an_operation():
    model = ScriptedModel([response()])
    ctx = context(model)
    ctx.deadline = 0
    with pytest.raises(EpisodeTimeout):
        asyncio.run(ctx.model([], ""))
    assert ctx.model_requests == 0
    assert not model.requests


def test_trace_is_ordered_redacted_and_keeps_raw_protocol_error():
    raw = response()
    raw["status"] = "incomplete"
    model = ScriptedModel([raw])
    file = io.StringIO()
    ctx = context(model)
    ctx.trace = TraceWriter(file, ("secret-key",))
    with pytest.raises(Exception, match="not completed"):
        asyncio.run(ctx.model([], "secret-key"))
    events = [json.loads(line) for line in file.getvalue().splitlines()]
    assert [e["event_order"] for e in events] == [1, 2, 3]
    assert events[1]["raw_response"] == raw
    assert "secret-key" not in file.getvalue()
    assert ctx.usage.unknown_usage_requests == 0
