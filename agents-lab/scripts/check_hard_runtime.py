"""Run hard-set references in the configured five-second Docker sandbox, with no API calls."""

import asyncio
import json
from time import perf_counter

from hard_problems import ROOT, cases

from mathlab.config import load_limits
from mathlab.python_tool import DockerPython, inspect_image


async def main():
    limits = load_limits(ROOT / "config.toml")
    image = inspect_image(limits.python_image)
    tool = DockerPython(image)
    source = (ROOT / "scripts/hard_reference.py").read_text()
    answers = {
        row["id"]: row["answer"]
        for row in map(json.loads, (ROOT / "data/hard/answers.jsonl").read_text().splitlines())
    }
    print(f"Image: {image}", flush=True)
    slowest = 0
    for row in cases():
        start = perf_counter()
        result = await tool.execute(
            source + f"\nprint(solve({row['id']!r}))\n",
            timeout=limits.python_timeout_seconds,
            output_limit=limits.python_output_limit_bytes,
        )
        duration = perf_counter() - start
        slowest = max(slowest, duration)
        if (
            result.exit_code != 0
            or result.timed_out
            or result.output_limited
            or result.stdout.strip() != answers[row["id"]]
        ):
            raise RuntimeError(f"{row['id']}: reference failed: {result}")
        print(f"OK {row['id']}: {duration:.2f}s", flush=True)
    print(f"All 12 references passed. Slowest observed call including cleanup: {slowest:.2f}s.")


if __name__ == "__main__":
    asyncio.run(main())
