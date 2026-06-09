"""
SANDBOX — Phase 2 execution environment.

Wraps Docker so the agent's tool calls run in an isolated container.
The repo under test is COPIED to a tmpdir (not mounted directly) so:
  - Edits made by edit_file persist across tool calls within a run
  - Re-running the agent on the same fixture always starts fresh
  - The host filesystem is never modified by the agent

How it works:
  1. start(): copy repo → tmpdir, build Docker image, launch container
     with tmpdir mounted at /repo
  2. execute_tool(): dispatch each tool to the right implementation
     - File reads/writes go directly to tmpdir on the HOST (fast, no shell escaping)
     - Command execution (run_tests, run_command etc.) goes through docker exec
     Both operate on the same files because tmpdir IS /repo inside the container.
  3. stop(): docker stop, clean up tmpdir

Why long-running container instead of one-per-call?
Because edit_file must persist changes for run_tests to see them. A per-call
container would start fresh every time — the agent could never verify its fix.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from .config import settings
from .tools import ToolResult


SANDBOX_IMAGE = "react-debugger-sandbox:latest"
CONTAINER_REPO = "/repo"


class DockerSandbox:
    """
    A long-running Docker container that holds a copy of the repo being debugged.

    Instantiate, call start(), use execute_tool() for each agent action,
    then call stop(). The ReActAgent manages this lifecycle in run().
    """

    def __init__(self, repo_path: str, test_command: str) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.test_command = test_command
        self._tmpdir: tempfile.TemporaryDirectory | None = None
        self._work_dir: Path | None = None       # host path; mounted at /repo
        self._container_id: str | None = None

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """Build image (cached), copy repo to tmpdir, launch container."""
        self._build_image()
        self._copy_repo()
        self._start_container()

    def stop(self) -> None:
        """Stop the container and remove the tmpdir."""
        if self._container_id:
            subprocess.run(
                ["docker", "stop", self._container_id],
                capture_output=True,
            )
            self._container_id = None
        if self._tmpdir:
            self._tmpdir.cleanup()
            self._tmpdir = None
            self._work_dir = None

    # -----------------------------------------------------------------------
    # Tool dispatch
    # -----------------------------------------------------------------------

    def execute_tool(self, tool_name: str, tool_input: dict) -> ToolResult:
        """Route a tool call to its implementation. Catches all errors."""
        try:
            handlers = {
                "run_tests":   self._run_tests,
                "read_file":   self._read_file,
                "list_files":  self._list_files,
                "search_code": self._search_code,
                "run_command": self._run_command,
                "edit_file":   self._edit_file,
            }
            handler = handlers.get(tool_name)
            if not handler:
                return ToolResult(output=f"Unknown tool: {tool_name}", is_error=True)
            return handler(tool_input)
        except Exception as exc:
            return ToolResult(output=f"Sandbox error in {tool_name}: {exc}", is_error=True)

    # -----------------------------------------------------------------------
    # Setup helpers
    # -----------------------------------------------------------------------

    def _build_image(self) -> None:
        """
        Build the sandbox Docker image.

        Docker caches layers aggressively — if the Dockerfile hasn't changed
        this returns in milliseconds. We run it every time to ensure the image
        is always current.
        """
        dockerfile = Path(__file__).parent.parent.parent / "Dockerfile.sandbox"
        if not dockerfile.exists():
            raise FileNotFoundError(
                f"Dockerfile.sandbox not found at {dockerfile}. "
                "Make sure you're running from the react-debugger directory."
            )
        result = subprocess.run(
            [
                "docker", "build",
                "-t", SANDBOX_IMAGE,
                "-f", str(dockerfile),
                str(dockerfile.parent),   # build context = react-debugger/
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Docker build failed:\n{result.stderr}")

    def _copy_repo(self) -> None:
        """Copy the repo to a fresh temporary directory."""
        self._tmpdir = tempfile.TemporaryDirectory(prefix="react-debugger-")
        self._work_dir = Path(self._tmpdir.name) / "repo"
        shutil.copytree(str(self.repo_path), str(self._work_dir))

    def _start_container(self) -> None:
        """
        Launch a Docker container with work_dir mounted at /repo.

        Flags used:
          --detach          run in background, return container ID
          --volume          mount our tmpdir copy into the container
          --workdir         all docker exec commands run from here
          --network none    no outbound internet (sandboxing)
          --memory 512m     prevent runaway memory usage
          --cpus 1.0        one CPU core — enough for tests
        """
        result = subprocess.run(
            [
                "docker", "run",
                "--detach",
                "--volume", f"{self._work_dir}:{CONTAINER_REPO}",
                "--workdir", CONTAINER_REPO,
                "--network", "none",
                "--memory", "512m",
                "--cpus", "1.0",
                SANDBOX_IMAGE,
                "sleep", "infinity",   # keep container alive between tool calls
            ],
            check=True, capture_output=True, text=True,
        )
        self._container_id = result.stdout.strip()

    # -----------------------------------------------------------------------
    # Command execution
    # -----------------------------------------------------------------------

    def _exec(self, command: str, timeout: int | None = None) -> tuple[str, bool]:
        """
        Run a shell command inside the container.

        Returns (output, is_error). Both stdout and stderr are captured and
        combined — this matches what a developer sees in their terminal.
        """
        try:
            result = subprocess.run(
                ["docker", "exec", self._container_id, "sh", "-c", command],
                capture_output=True, text=True,
                timeout=timeout or settings.command_timeout,
            )
            output = result.stdout
            if result.stderr:
                output = (output + "\n" + result.stderr) if output else result.stderr
            return output.strip(), result.returncode != 0
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout or settings.command_timeout}s", True

    # -----------------------------------------------------------------------
    # Tool implementations
    # -----------------------------------------------------------------------

    def _run_tests(self, tool_input: dict) -> ToolResult:
        """
        Run the test suite (or a specific test) inside the container.

        We use the test_command configured at run() time, optionally scoped
        to a single test path (e.g. 'test_code.py::test_find_max').
        """
        test_path = tool_input.get("test_path", "")
        command = f"{self.test_command} {test_path}".strip()
        output, _ = self._exec(command)
        # pytest exit code 1 = test failures — that's expected, not an error
        return ToolResult(output=output)

    def _read_file(self, tool_input: dict) -> ToolResult:
        """
        Read a file from the work directory, with optional line range.

        File reads go directly to the host tmpdir — no docker exec needed,
        which is faster and avoids shell escaping issues.
        Line numbers are added to every line so the agent can reference them
        when calling edit_file.
        """
        path = tool_input["path"]
        host_path = self._work_dir / path

        if not host_path.exists():
            return ToolResult(output=f"File not found: {path}", is_error=True)

        content = host_path.read_text()
        lines = content.splitlines()

        start = tool_input.get("start_line")
        end = tool_input.get("end_line")

        if start or end:
            # Slice to requested range (convert from 1-indexed to 0-indexed)
            s = (start - 1) if start else 0
            e = end if end else len(lines)
            lines = lines[s:e]
            offset = s + 1
        else:
            offset = 1

        numbered = "\n".join(f"{offset + i}: {line}" for i, line in enumerate(lines))
        return ToolResult(output=numbered)

    def _list_files(self, tool_input: dict) -> ToolResult:
        """List files in a directory, recursively, sorted."""
        directory = tool_input.get("directory", ".")
        output, is_error = self._exec(f"find {shlex.quote(directory)} -type f | sort")
        return ToolResult(output=output, is_error=is_error)

    def _search_code(self, tool_input: dict) -> ToolResult:
        """
        Search files for a pattern (grep -rn).

        Returns file:line:match format. Useful for finding where a function
        is defined or where a pattern appears before reading the full file.
        """
        pattern = tool_input["pattern"]
        output, _ = self._exec(f"grep -rn {shlex.quote(pattern)} .")
        return ToolResult(output=output or "(no matches found)")

    def _run_command(self, tool_input: dict) -> ToolResult:
        """Run an arbitrary shell command inside the sandbox."""
        command = tool_input["command"]
        output, is_error = self._exec(command)
        return ToolResult(output=output, is_error=is_error)

    def _edit_file(self, tool_input: dict) -> ToolResult:
        """
        Replace an exact string in a file.

        Reads and writes via the host tmpdir (fast, no escaping). The change
        is immediately visible to the next docker exec (run_tests) because
        the tmpdir IS the container's /repo via the volume mount.

        Requires old_string to appear exactly once — if it appears zero times,
        the agent's assumption about the file content is wrong. If it appears
        multiple times, we need more context to make a unique match.
        """
        path = tool_input["path"]
        old_string = tool_input["old_string"]
        new_string = tool_input["new_string"]

        host_path = self._work_dir / path
        if not host_path.exists():
            return ToolResult(output=f"File not found: {path}", is_error=True)

        content = host_path.read_text()
        count = content.count(old_string)

        if count == 0:
            return ToolResult(
                output=(
                    f"old_string not found in {path}.\n"
                    "The file may have changed, or there's a whitespace mismatch.\n"
                    "Use read_file to see the current contents before trying again."
                ),
                is_error=True,
            )
        if count > 1:
            return ToolResult(
                output=(
                    f"old_string appears {count} times in {path}. "
                    "Add more surrounding lines to make it unique."
                ),
                is_error=True,
            )

        new_content = content.replace(old_string, new_string, 1)
        host_path.write_text(new_content)

        return ToolResult(output=f"Successfully replaced 1 occurrence in {path}")
