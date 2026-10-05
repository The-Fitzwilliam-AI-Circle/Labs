"""Check the 30 new reference solutions in the lab's configured Docker sandbox.

Run explicitly: uv run python scripts/check_ladder_runtime.py
Requires the existing local image, but never contacts a model or the network.
This verifies feasibility, not whether a model can discover a solution.
"""

import asyncio
import json
from pathlib import Path
from time import perf_counter

from ladder import cases

from mathlab.config import load_limits
from mathlab.python_tool import DockerPython, inspect_image

ROOT = Path(__file__).resolve().parents[1]


async def main():
    limits = load_limits(ROOT / "config.toml")
    image = inspect_image(limits.python_image)
    tool = DockerPython(image)
    source = (ROOT / "scripts/ladder.py").read_text()
    answers = {
        row["id"]: row["answer"]
        for row in map(json.loads, (ROOT / "data/dev/answers.jsonl").read_text().splitlines())
    }
    print(f"Image: {image}", flush=True)
    slowest = 0
    for index, case in enumerate(cases()):
        start = perf_counter()
        result = await tool.execute(
            source + f"\nprint(solve(cases()[{index}]))\n",
            timeout=limits.python_timeout_seconds,
            output_limit=limits.python_output_limit_bytes,
        )
        duration = perf_counter() - start
        slowest = max(slowest, duration)
        if (
            result.exit_code != 0
            or result.timed_out
            or result.output_limited
            or result.stdout.strip() != answers[case["id"]]
        ):
            raise RuntimeError(f"{case['id']}: reference did not complete correctly: {result}")
        print(f"OK {case['id']} ({case['family']}): {duration:.2f}s", flush=True)
    print(f"All 30 references passed. Slowest observed call including cleanup: {slowest:.2f}s.")


if __name__ == "__main__":
    asyncio.run(main())
