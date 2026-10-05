"""One context is one episode: histories never reset budgets or the deadline."""

import asyncio
import copy
import time
from dataclasses import asdict

from mathlab.config import Limits
from mathlab.files import write_jsonl_row
from mathlab.model import extract_usage, parse_turn
from mathlab.types import BudgetExceeded, EpisodeTimeout, ModelTurn, ToolResult, Usage


class TraceWriter:
    def __init__(self, file, secrets: tuple[str, ...] = ()):
        self.file = file
        self.secrets = tuple(secret for secret in secrets if secret)

    def redact(self, value):
        if isinstance(value, str):
            for secret in self.secrets:
                value = value.replace(secret, "[REDACTED]")
            return value
        if isinstance(value, dict):
            return {key: self.redact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.redact(item) for item in value]
        return value

    def write(self, value):
        write_jsonl_row(self.file, self.redact(value))


class AgentContext:
    def __init__(self, task_id: str, limits: Limits, model, python, trace: TraceWriter):
        self.task_id = task_id
        self.limits = limits
        self._model = model
        self._python = python
        self.trace = trace
        self.started = time.monotonic()
        self.deadline = self.started + limits.episode_timeout_seconds
        self.model_requests = 0
        self.python_calls = 0
        self.usage = Usage()
        self._event_order = 0

    def remaining(self) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise EpisodeTimeout("Episode deadline exhausted")
        return remaining

    def record(self, event: str, **details) -> None:
        self._event_order += 1
        self.trace.write(
            {
                "task_id": self.task_id,
                "event_order": self._event_order,
                "event": event,
                "elapsed_seconds": round(time.monotonic() - self.started, 6),
                **details,
            }
        )

    async def model(
        self,
        history: list[dict],
        instructions: str,
        allow_python: bool = True,
    ) -> ModelTurn:
        remaining = self.remaining()
        if self.model_requests >= self.limits.max_model_requests:
            raise BudgetExceeded("Model request allowance exhausted")
        self.model_requests += 1
        request_number = self.model_requests
        started = time.monotonic()
        # Freeze the input before a participant could mutate another conversation.
        history = copy.deepcopy(history)
        self.record(
            "model_request",
            request_number=request_number,
            history=history,
            instructions=instructions,
            allow_python=allow_python,
            max_output_tokens=self.limits.max_output_tokens_per_request,
        )
        response = None
        usage_recorded = False
        try:
            async with asyncio.timeout(remaining):
                response = await self._model.request(
                    history,
                    instructions,
                    allow_python=allow_python,
                    max_output_tokens=self.limits.max_output_tokens_per_request,
                    timeout=self.remaining(),
                )
            usage = extract_usage(response)
            if usage is None:
                self.usage.unknown_usage_requests += 1
            else:
                for key, value in usage.items():
                    setattr(self.usage, key, getattr(self.usage, key) + value)
            usage_recorded = True
            self.record(
                "model_response",
                request_number=request_number,
                raw_response=response,
                usage=usage,
                usage_known=usage is not None,
                duration_seconds=round(time.monotonic() - started, 6),
            )
            return parse_turn(response, allow_python)
        except BaseException as exc:
            if not usage_recorded:
                self.usage.unknown_usage_requests += 1
            self.record(
                "model_error",
                request_number=request_number,
                error=self.trace.redact(f"{type(exc).__name__}: {exc}")[:1000],
                usage_known=usage_recorded and extract_usage(response) is not None,
                duration_seconds=round(time.monotonic() - started, 6),
            )
            if isinstance(exc, TimeoutError):
                raise EpisodeTimeout("Model request exceeded the episode deadline") from exc
            raise

    async def python(self, code: str) -> ToolResult:
        remaining = self.remaining()
        if self.python_calls >= self.limits.max_python_calls:
            raise BudgetExceeded("Python call allowance exhausted")
        self.python_calls += 1
        call_number = self.python_calls
        self.record("python_request", call_number=call_number, code=code)
        started = time.monotonic()
        try:
            result = await self._python.execute(
                code,
                timeout=min(remaining, self.limits.python_timeout_seconds),
                output_limit=self.limits.python_output_limit_bytes,
            )
        except BaseException as exc:
            self.record("python_error", call_number=call_number, error=type(exc).__name__)
            raise
        self.record(
            "python_result",
            call_number=call_number,
            observation=asdict(result),
            duration_seconds=round(time.monotonic() - started, 6),
        )
        self.remaining()
        return result
