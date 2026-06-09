"""
TOOLS — the action space available to the ReAct agent.

Each tool has two parts:
  1. A JSON schema (the "declaration") — sent to Claude so it knows what tools exist
     and what parameters each one takes. Claude reads the descriptions to decide
     which tool to use.
  2. An execute function — called by the agent loop when Claude picks that tool.
     In Phase 1 these are STUBS returning fake observations.
     In Phase 2 they'll run real commands inside the Docker sandbox.

The tool descriptions are critical prompt engineering. The agent reads them to
decide what action to take — vague descriptions lead to bad tool choices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Tool declarations — sent to the Anthropic API as the `tools=` parameter.
# The API uses these to constrain Claude's output: it can only call tools
# whose names appear here, and only with inputs that match the schema.
# ---------------------------------------------------------------------------

TOOL_DECLARATIONS: list[dict] = [
    {
        "name": "run_tests",
        "description": (
            "Run the test suite and see which tests pass or fail. "
            "Returns the full pytest output including tracebacks. "
            "Use this to check the current state and to verify whether your fix worked. "
            "Always start a debugging session by running tests first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "test_path": {
                    "type": "string",
                    "description": "Optional: specific test file or test function to run (e.g. 'test_code.py::test_addition'). Omit to run all tests.",
                }
            },
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read the contents of a file. Optionally specify a line range to read "
            "only the relevant section. Use this to inspect code you suspect contains "
            "the bug."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to the repo root"},
                "start_line": {"type": "integer", "description": "First line to read (1-indexed)"},
                "end_line": {"type": "integer", "description": "Last line to read (inclusive)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_files",
        "description": (
            "List files in a directory (recursive). "
            "Use this to understand the project structure when you first start."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory to list. Defaults to repo root if omitted.",
                }
            },
        },
    },
    {
        "name": "search_code",
        "description": (
            "Search the codebase for a pattern (like grep). "
            "Returns matching lines with file names and line numbers. "
            "Use this to find where a function is defined or called."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "String or regex pattern to search for",
                }
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "run_command",
        "description": (
            "Run an arbitrary shell command in the sandbox "
            "(e.g., to inspect a value, check the Python version, print a variable). "
            "Returns stdout and stderr. Use sparingly — prefer read_file and run_tests."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"}
            },
            "required": ["command"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Make a targeted edit to a file by replacing an exact string with a new string. "
            "Use this to apply a fix once you've identified the bug. "
            "The old_string must be an exact, unique match — include enough surrounding "
            "context to make it unique if the pattern repeats."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {
                    "type": "string",
                    "description": "Exact text to replace (must appear exactly once in the file)",
                },
                "new_string": {"type": "string", "description": "Replacement text"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "finish",
        "description": (
            "Call this when the tests pass and the bug is fixed, "
            "OR when you are confident you cannot solve it. "
            "Provide a clear summary of the root cause and what you did (or why you're stuck)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "success": {
                    "type": "boolean",
                    "description": "True if tests pass, False if giving up",
                },
                "summary": {
                    "type": "string",
                    "description": "Plain-English explanation of what you did",
                },
                "root_cause": {
                    "type": "string",
                    "description": "What was actually wrong in the code",
                },
            },
            "required": ["success", "summary"],
        },
    },
]


# ---------------------------------------------------------------------------
# Stub executor — Phase 1 only.
# In Phase 2 this gets replaced by real sandbox execution.
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """What comes back from executing a tool — always a string observation."""
    output: str
    is_error: bool = False


def execute_stub(tool_name: str, tool_input: dict[str, Any]) -> ToolResult:
    """
    STUB: returns fake but plausible observations so the loop can cycle
    without any real files or Docker. Used in Phase 1 to test the loop
    structure in isolation from the execution machinery.
    """
    stubs: dict[str, str] = {
        "run_tests": (
            "============================= test session starts ==============================\n"
            "collected 3 items\n\n"
            "test_code.py::test_basic PASSED\n"
            "test_code.py::test_edge_case FAILED\n"
            "test_code.py::test_empty PASSED\n\n"
            "=================================== FAILURES ===================================\n"
            "_________________________ test_edge_case __________________________\n\n"
            "    def test_edge_case():\n"
            ">       assert find_max([1, 2, 3, 4, 5]) == 5\n"
            "E       AssertionError: assert 4 == 5\n\n"
            "test_code.py:12: AssertionError\n"
            "========================= 1 failed, 2 passed in 0.12s ========================="
        ),
        "read_file": (
            "1: def find_max(items):\n"
            "2:     \"\"\"Return the maximum value in a list.\"\"\"\n"
            "3:     max_val = items[0]\n"
            "4:     for i in range(len(items) - 1):  # BUG: should be range(len(items))\n"
            "5:         if items[i] > max_val:\n"
            "6:             max_val = items[i]\n"
            "7:     return max_val\n"
        ),
        "list_files": "buggy_code.py\ntest_code.py\nBUG.md",
        "search_code": "buggy_code.py:4:    for i in range(len(items) - 1):",
        "run_command": "[stub] command output here",
        "edit_file": "Successfully replaced 1 occurrence in buggy_code.py",
    }
    return ToolResult(output=stubs.get(tool_name, f"[stub] {tool_name} called with {tool_input}"))
