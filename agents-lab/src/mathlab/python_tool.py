"""Fresh, resource-limited Docker executions; generated code never runs on the host."""

import asyncio
import contextlib
import subprocess
import uuid

from mathlab.types import LabError, ToolResult


def inspect_image(image: str) -> str:
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LabError("Docker is unavailable; start Docker and retry") from exc
    if result.returncode or not result.stdout.strip().startswith("sha256:"):
        raise LabError(
            f"Docker image {image!r} is unavailable. Start Docker and build Dockerfile.python."
        )
    return result.stdout.strip()


class DockerPython:
    def __init__(self, image: str):
        self.image = image

    def command(self, name: str) -> list[str]:
        return [
            "docker",
            "run",
            "--rm",
            "--pull=never",
            "--name",
            name,
            "--label",
            "mathlab.tool=true",
            "--network=none",
            "--read-only",
            "--user=65534:65534",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--cpus=1",
            "--memory=256m",
            "--memory-swap=256m",
            "--pids-limit=64",
            "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=16m,mode=1777",
            "--shm-size=16m",
            "--log-driver=none",
            "-i",
            self.image,
        ]

    async def _cleanup(self, name: str, process) -> str | None:
        if process is not None and process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
        if process is not None:
            # Drain without retaining bytes after killing the client. Waiting with
            # full pipe buffers can deadlock asyncio's subprocess transport.
            async def discard(stream):
                while await stream.read(1024):
                    pass

            await asyncio.gather(discard(process.stdout), discard(process.stderr), process.wait())
        removal = None
        try:
            removal = await asyncio.create_subprocess_exec(
                "docker",
                "rm",
                "--force",
                name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            async with asyncio.timeout(5):
                _, stderr = await removal.communicate()
            if removal.returncode and b"No such container" not in stderr:
                return "Container cleanup failed; check Docker for " + name
        except (OSError, TimeoutError):
            if removal is not None and removal.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    removal.kill()
                await removal.wait()
            return "Container cleanup could not be confirmed; check Docker for " + name
        return None

    async def execute(self, code: str, *, timeout: float, output_limit: int) -> ToolResult:
        name = "mathlab-" + uuid.uuid4().hex
        process = None
        tasks = []
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        bytes_retained = 0
        exceeded = asyncio.Event()
        timed_out = False
        exit_code = None
        failure = None

        async def read(stream, channel):
            nonlocal bytes_retained
            while chunk := await stream.read(1024):
                remaining = output_limit - bytes_retained
                buffers[channel].extend(chunk[:remaining])
                bytes_retained += min(len(chunk), remaining)
                if len(chunk) > remaining:
                    exceeded.set()
                    return

        async def write():
            try:
                process.stdin.write(code.encode("utf-8"))
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                process.stdin.close()

        try:
            async with asyncio.timeout(timeout):
                process = await asyncio.create_subprocess_exec(
                    *self.command(name),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    limit=1024,
                )
                tasks = [
                    asyncio.create_task(read(process.stdout, "stdout")),
                    asyncio.create_task(read(process.stderr, "stderr")),
                    asyncio.create_task(write()),
                    asyncio.create_task(process.wait()),
                ]
                finished = asyncio.gather(*tasks)
                limit_hit = asyncio.create_task(exceeded.wait())
                tasks.append(limit_hit)
                try:
                    await asyncio.wait([finished, limit_hit], return_when=asyncio.FIRST_COMPLETED)
                    if finished.done():
                        await finished
                        exit_code = process.returncode
                finally:
                    finished.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await finished
        except TimeoutError:
            timed_out = True
        except OSError:
            failure = "Docker execution failed; check that Docker is installed and running"
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            # Cleanup is allowed a short grace period even when the episode is cancelled.
            cleanup = asyncio.create_task(self._cleanup(name, process))
            try:
                cleanup_failure = await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise
            failure = cleanup_failure or failure

        stdout = buffers["stdout"].decode("utf-8", errors="ignore")
        stderr = buffers["stderr"].decode("utf-8", errors="ignore")
        if failure:
            # Keep diagnostics inside the same combined output allowance.
            diagnostic = (failure + "\n").encode("utf-8")[:output_limit]
            stdout = stdout.encode("utf-8")[: output_limit - len(diagnostic)].decode(
                "utf-8", errors="ignore"
            )
            available = max(0, output_limit - len(stdout.encode("utf-8")))
            stderr = (diagnostic + stderr.encode("utf-8"))[:available].decode(
                "utf-8", errors="ignore"
            )
        return ToolResult(stdout, stderr, exit_code, timed_out, exceeded.is_set())
