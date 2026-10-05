import copy
import json
from pathlib import Path

import pytest

from mathlab.types import ToolResult

ROOT = Path(__file__).resolve().parents[1]


def response(name="submit_answer", value="3/4", call_id="call-1"):
    argument = "code" if name == "python" else "answer"
    return {
        "id": "response-test",
        "status": "completed",
        "output": [
            {
                "type": "reasoning",
                "id": "reasoning-test",
                "summary": [],
                "encrypted_content": "opaque-provider-content",
            },
            {
                "type": "function_call",
                "name": name,
                "call_id": call_id,
                "arguments": json.dumps({argument: value}),
                "status": "completed",
            },
        ],
        "usage": {
            "input_tokens": 20,
            "output_tokens": 10,
            "output_tokens_details": {"reasoning_tokens": 4},
        },
    }


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    async def request(self, history, instructions, **kwargs):
        self.requests.append(
            copy.deepcopy({"history": history, "instructions": instructions, **kwargs})
        )
        item = next(self.responses)
        if isinstance(item, BaseException):
            raise item
        return copy.deepcopy(item)


class ScriptedPython:
    def __init__(self, result=None):
        self.result = result if result is not None else ToolResult("42\n", "", 0)
        self.calls = []

    async def execute(self, code, **kwargs):
        self.calls.append((code, kwargs))
        return self.result


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


@pytest.fixture
def questions(tmp_path):
    return jsonl(
        tmp_path / "questions.jsonl",
        [
            {"id": "one", "problem": "What is 3 divided by 4?"},
            {"id": "two", "problem": "What is 6 times 7?"},
        ],
    )
