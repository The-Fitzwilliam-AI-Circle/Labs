"""Offline subprocess lifecycle checks using fixed, trusted process fixtures.

These substitute the Docker executable, never evaluate model code on the host,
and supplement (rather than replace) the real Docker integration checks.
"""

import asyncio
import sys

import pytest

from mathlab.python_tool import DockerPython


@pytest.mark.parametrize(
    "program, expected",
    [
        ("import sys; sys.stdin.read(); print('42')", "success"),
        ("import sys; sys.stdin.read(); print('error', file=sys.stderr); sys.exit(2)", "error"),
        ("import time; time.sleep(30)", "timeout"),
        ("import os\nwhile True: os.write(1, b'x'*65536)", "output_limit"),
    ],
)
def test_bounded_readers_and_explicit_removal(monkeypatch, program, expected):
    real_exec = asyncio.create_subprocess_exec
    commands = []

    async def fake_docker(*args, **kwargs):
        commands.append(args)
        fixture = program if args[1] == "run" else "pass"
        return await real_exec(sys.executable, "-c", fixture, **kwargs)

    monkeypatch.setattr("mathlab.python_tool.asyncio.create_subprocess_exec", fake_docker)

    async def scenario():
        result = await asyncio.wait_for(
            DockerPython("test-image").execute(
                "THIS INPUT MUST NEVER BE EVALUATED ON THE HOST",
                timeout=0.3,
                output_limit=8192,
            ),
            timeout=5,
        )
        if expected == "success":
            assert result.stdout == "42\n" and result.exit_code == 0
        elif expected == "error":
            assert result.stderr == "error\n" and result.exit_code == 2
        elif expected == "timeout":
            assert result.timed_out
        else:
            assert result.output_limited
            assert len(result.stdout.encode()) + len(result.stderr.encode()) <= 8192

    asyncio.run(scenario())
    assert commands[-1][:3] == ("docker", "rm", "--force")
    assert commands[-1][-1] == commands[0][commands[0].index("--name") + 1]


def test_cancellation_still_removes_named_container(monkeypatch):
    real_exec = asyncio.create_subprocess_exec
    commands = []

    async def scenario():
        started = asyncio.Event()

        async def fake_docker(*args, **kwargs):
            commands.append(args)
            fixture = "import time; time.sleep(30)" if args[1] == "run" else "pass"
            process = await real_exec(sys.executable, "-c", fixture, **kwargs)
            started.set()
            return process

        monkeypatch.setattr("mathlab.python_tool.asyncio.create_subprocess_exec", fake_docker)
        task = asyncio.create_task(
            DockerPython("test-image").execute(
                "UNUSED",
                timeout=10,
                output_limit=8192,
            )
        )
        await started.wait()
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)

    asyncio.run(scenario())
    assert commands[-1][:3] == ("docker", "rm", "--force")
