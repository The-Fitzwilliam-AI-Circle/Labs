import asyncio
import json
import subprocess

import pytest

from mathlab.config import Limits
from mathlab.files import read_jsonl
from mathlab.python_tool import DockerPython, inspect_image
from mathlab.runner import run
from mathlab.scoring import score
from tests.conftest import ROOT, jsonl, response


def containers():
    result = subprocess.run(
        ["docker", "ps", "-aq", "--filter", "label=mathlab.tool=true"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return set(result.stdout.split())


def test_required_container_controls():
    command = DockerPython("sha256:test").command("unique-name")
    for flag in (
        "--network=none",
        "--read-only",
        "--user=65534:65534",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--cpus=1",
        "--memory=256m",
        "--memory-swap=256m",
        "--pids-limit=64",
        "--pull=never",
        "--log-driver=none",
    ):
        assert flag in command
    assert not any("mount" in arg or arg in ("-v", "--env", "-e") for arg in command)


@pytest.fixture
def tool():
    # Explicit integration selection should fail, not silently skip, if setup is missing.
    return DockerPython(inspect_image("mathlab-python:v0"))


@pytest.mark.integration
def test_sympy_stateless_execution_and_cleanup(tool):
    before = containers()

    async def scenario():
        first = await tool.execute(
            "import sympy; print(sympy.Rational(6, 8)); open('/tmp/marker','w').write('x')",
            timeout=5,
            output_limit=8192,
        )
        assert first.exit_code == 0 and first.stdout == "3/4\n"
        second = await tool.execute(
            "from pathlib import Path; print(Path('/tmp/marker').exists())",
            timeout=5,
            output_limit=8192,
        )
        assert second.stdout == "False\n"

    asyncio.run(scenario())
    assert containers() == before


@pytest.mark.integration
def test_tool_timeout_output_limit_and_error_cleanup(tool):
    before = containers()

    async def scenario():
        timeout = await tool.execute("while True: pass", timeout=1, output_limit=8192)
        assert timeout.timed_out
        flood = await tool.execute(
            "import os\nwhile True:\n os.write(1,b'x'*1024); os.write(2,b'y'*1024)",
            timeout=5,
            output_limit=8192,
        )
        assert flood.output_limited
        assert len(flood.stdout.encode()) + len(flood.stderr.encode()) <= 8192
        error = await tool.execute("raise ValueError('recoverable')", timeout=5, output_limit=8192)
        assert error.exit_code != 0 and "recoverable" in error.stderr

    asyncio.run(scenario())
    assert containers() == before


@pytest.mark.integration
def test_isolation_no_host_files_credentials_network_or_root_writes(tool, tmp_path, monkeypatch):
    before = containers()
    sentinel = tmp_path / "host-only-secret"
    sentinel.write_text("never expose")
    monkeypatch.setenv("OPENAI_API_KEY", "host-only-key")
    code = f"""
import json, os, pathlib, socket
result = {{'uid': os.getuid(), 'key': os.getenv('OPENAI_API_KEY'),
          'host_file': pathlib.Path({str(sentinel)!r}).exists(),
          'docker_socket': pathlib.Path('/var/run/docker.sock').exists()}}
try:
    pathlib.Path('/usr/root-write-test').write_text('x')
    result['root_write'] = True
except OSError:
    result['root_write'] = False
try:
    socket.create_connection(('1.1.1.1', 443), timeout=0.3).close()
    result['network'] = True
except OSError:
    result['network'] = False
print(json.dumps(result))
"""
    result = asyncio.run(tool.execute(code, timeout=5, output_limit=8192))
    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "uid": 65534,
        "key": None,
        "host_file": False,
        "docker_socket": False,
        "root_write": False,
        "network": False,
    }
    assert containers() == before


@pytest.mark.integration
def test_cancellation_removes_container(tool):
    before = containers()

    async def scenario():
        task = asyncio.create_task(tool.execute("while True: pass", timeout=20, output_limit=8192))
        # Wait until Docker has actually created it; then cancel an active execution.
        for _ in range(50):
            await asyncio.sleep(0.1)
            if containers() - before:
                break
        else:
            pytest.fail("Test container did not start")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert containers() == before


@pytest.mark.integration
def test_parallel_episodes_keep_docker_observations_separate(tool, tmp_path):
    before = containers()
    questions = jsonl(
        tmp_path / "questions.jsonl", [{"id": str(i), "problem": str(i)} for i in range(3)]
    )
    answers = jsonl(
        tmp_path / "answers.jsonl", [{"id": str(i), "answer": str(i)} for i in range(3)]
    )

    async def scenario():
        ready = asyncio.Event()

        class Model:
            calls = 0

            async def request(self, history, instructions, **kwargs):
                value = history[0]["content"]
                if len(history) == 1:
                    self.calls += 1
                    if self.calls == 2:
                        ready.set()
                    await ready.wait()
                    return response("python", f"print({value})", f"python-{value}")
                assert history[-1]["call_id"] == f"python-{value}"
                observation = json.loads(history[-1]["output"])
                assert observation["stdout"].strip() == value
                assert observation["exit_code"] == 0
                return response(value=value)

        await run(
            agent_path=ROOT / "agents/tool.py",
            questions_path=questions,
            out=tmp_path / "run",
            limits=Limits(),
            model=Model(),
            python=tool,
            model_id="offline-fixture",
            concurrency=2,
        )

    asyncio.run(asyncio.wait_for(scenario(), timeout=20))
    assert score(tmp_path / "run", answers)["correct"] == 3
    assert all(
        row["model_requests"] == 2 and row["python_calls"] == 1
        for row in read_jsonl(tmp_path / "run/results.jsonl")
    )
    assert containers() == before
