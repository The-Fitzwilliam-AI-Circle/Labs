"""Validated configuration; intentionally no model default."""

import math
import os
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from urllib.parse import urlsplit

from mathlab.types import LabError


@dataclass(frozen=True)
class Limits:
    max_model_requests: int = 6
    max_output_tokens_per_request: int = 2048
    max_python_calls: int = 4
    episode_timeout_seconds: float = 90
    python_timeout_seconds: float = 5
    python_output_limit_bytes: int = 8192
    python_image: str = "mathlab-python:v0"

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "python_image":
                if not isinstance(value, str) or not value.strip() or value.startswith("-"):
                    raise LabError("python_image must be a nonempty Docker image name")
                continue
            is_timeout = name.endswith("_seconds")
            valid_type = type(value) in (int, float) if is_timeout else type(value) is int
            minimum = 0 if name in {"max_model_requests", "max_python_calls"} else 1
            if not valid_type or not math.isfinite(value) or value < minimum:
                # Subsecond deadlines are useful in tests and small pilots.
                if is_timeout and valid_type and math.isfinite(value) and value > 0:
                    continue
                raise LabError(
                    f"Invalid limit {name}: expected a positive number (budgets may be 0)"
                )


def load_limits(path: Path) -> Limits:
    try:
        with path.open("rb") as file:
            values = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise LabError(f"Cannot read configuration {path}: {exc}") from exc
    unknown = set(values) - {field.name for field in fields(Limits)}
    if unknown:
        raise LabError(f"Unknown configuration settings: {', '.join(sorted(unknown))}")
    return Limits(**values)


def model_configuration() -> tuple[str, str]:
    settings = provider_configuration()
    return settings.api_key, settings.model


@dataclass(frozen=True)
class ProviderConfiguration:
    provider: str
    model: str
    base_url: str
    api_key: str = field(repr=False)

    def public_metadata(self):
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api": "responses" if self.provider == "openai" else "chat_completions",
        }


def provider_configuration() -> ProviderConfiguration:
    provider = os.getenv("PROVIDER", "openai").strip().lower()
    if provider not in {"openai", "openrouter", "local"}:
        raise LabError("PROVIDER must be openai, openrouter, or local")
    model = os.getenv("MODEL", "").strip()
    key_name = {
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "local": "LOCAL_API_KEY",
    }[provider]
    key = os.getenv(key_name, "").strip()
    if not model or (not key and provider != "local"):
        raise LabError(f"Set {key_name} and an explicit MODEL in .env or the environment")
    base_url = {
        "openai": "https://api.openai.com/v1",
        "openrouter": "https://openrouter.ai/api/v1",
    }.get(provider)
    if provider == "local":
        base_url = os.getenv("MODEL_BASE_URL", "").strip().rstrip("/")
        try:
            url = urlsplit(base_url)
            valid = url.scheme in {"http", "https"} and url.hostname and url.port != 0
            valid = valid and not (url.username or url.password or url.query or url.fragment)
        except ValueError:
            valid = False
        if not valid:
            raise LabError(
                "Set MODEL_BASE_URL to an http(s) API base URL, e.g. http://localhost:11434/v1; "
                "no embedded credentials, query or fragment"
            )
    return ProviderConfiguration(provider, model, base_url, key or "local-no-key")
