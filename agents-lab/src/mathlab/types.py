"""The small public contract shared by agents, runtime, and runner."""

from dataclasses import asdict, dataclass
from typing import Any, Literal

Status = Literal[
    "completed", "budget_exceeded", "timeout", "provider_error", "protocol_error", "agent_error"
]
STATUSES = {
    "completed",
    "budget_exceeded",
    "timeout",
    "provider_error",
    "protocol_error",
    "agent_error",
}


@dataclass(frozen=True)
class Task:
    id: str
    problem: str


@dataclass(frozen=True)
class ModelTurn:
    kind: Literal["python", "answer"]
    value: str
    call_id: str
    output_items: list[dict[str, Any]]


@dataclass(frozen=True)
class ToolResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    timed_out: bool = False
    output_limited: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0  # Already included in output_tokens.
    unknown_usage_requests: int = 0


@dataclass
class Result:
    id: str
    answer: str | None
    status: Status
    model_requests: int
    python_calls: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    unknown_usage_requests: int
    elapsed_seconds: float
    error: str | None = None


class LabError(Exception):
    """An expected user-facing error."""


class DatasetError(LabError):
    pass


class EpisodeError(LabError):
    status: Status = "agent_error"


class BudgetExceeded(EpisodeError):
    status = "budget_exceeded"


class EpisodeTimeout(EpisodeError):
    status = "timeout"


class ProviderError(EpisodeError):
    status = "provider_error"


class ProtocolError(EpisodeError):
    status = "protocol_error"
