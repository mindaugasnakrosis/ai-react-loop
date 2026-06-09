# Self-Debugging Agent (ReAct Loop) — PRD for Claude Code

Copy everything below the line into Claude Code.

---

## Overview

Build a **self-debugging agent** called `react-debugger` — an autonomous agent that takes a code repository with a failing test suite, then debugs it on its own using the **ReAct loop** (Reason → Act → Observe → repeat). The agent forms a hypothesis about the bug, takes a diagnostic action (read a file, run a test, add a log statement), observes the result, and decides its next action based on what it learned — looping until the tests pass or it gives up.

**This is primarily a learning project.** The goal is for me to deeply understand the core agentic AI pattern by building it from scratch. Therefore:

- **Build the ReAct loop by hand using the raw Anthropic API.** Do NOT use LangChain, LlamaIndex agents, AutoGPT, CrewAI, or any framework that hides the loop. The entire point is that I see and understand every part of the loop. The Anthropic Python SDK for the LLM calls is fine; the agent loop itself must be hand-written.
- **Make the loop radically observable.** Every iteration must clearly show the agent's thought, the action it chose, the parameters, and the observation it got back. I want to watch the agent think.
- **Comment the core loop heavily** with educational comments explaining what each part does and why, as if teaching someone the ReAct pattern.

## Learning goals (what this project must teach me)

1. How a ReAct loop is actually structured in code — the turn-by-turn cycle
2. How tool-use / function-calling works at the API level (not hidden behind a framework)
3. How the agent's context window accumulates observations across iterations
4. How to design a good set of tools (the "act" space) for an agent
5. How termination conditions work (success, max iterations, agent-declares-stuck)
6. How to handle the agent making mistakes (bad tool calls, malformed output, infinite loops)
7. Why reasoning-before-acting produces better results than acting blindly

## Tech stack

```
Language:    Python 3.12
LLM:         Anthropic API (claude-sonnet-4-20250514) via anthropic SDK
                — use tool use / function calling
Terminal UI: Rich (for the live loop visualization)
Sandbox:     Docker (the agent runs arbitrary code — it MUST be sandboxed)
Testing:     pytest (both for the agent's own tests and for the buggy fixtures)
Config:      pydantic-settings
```

## The ReAct loop — core architecture

This is the heart of the project. Build it as an explicit, readable loop.

```
┌──────────────────────────────────────────────────────────────┐
│                      THE REACT LOOP                            │
│                                                                │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  1. Send conversation history to Claude               │     │
│  │     (system prompt + goal + all prior obs)            │     │
│  └────────────────────────┬────────────────────────────┘     │
│                           │                                   │
│                           ▼                                   │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  2. Claude responds with:                            │     │
│  │     - THOUGHT (reasoning text)                       │     │
│  │     - ACTION (a tool_use block)                      │     │
│  └────────────────────────┬────────────────────────────┘     │
│                           │                                   │
│                           ▼                                   │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  3. Is the action `finish`?                          │     │
│  │     YES → exit loop, report result                  │     │
│  │     NO  → continue                                  │     │
│  └────────────────────────┬────────────────────────────┘     │
│                           │ NO                                │
│                           ▼                                   │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  4. Execute the tool (in Docker sandbox)            │     │
│  │     → produces an OBSERVATION                        │     │
│  └────────────────────────┬────────────────────────────┘     │
│                           │                                   │
│                           ▼                                   │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  5. Append THOUGHT + ACTION + OBSERVATION            │     │
│  │     to conversation history                          │     │
│  └────────────────────────┬────────────────────────────┘     │
│                           │                                   │
│                           ▼                                   │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  6. Check termination:                              │     │
│  │     - tests pass? → success                         │     │
│  │     - max iterations? → give up                     │     │
│  │     - else → LOOP BACK TO STEP 1                    │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

## Directory structure

```
react-debugger/
├── src/
│   └── react_debugger/
│       ├── __init__.py
│       ├── agent.py            # THE REACT LOOP — the core, heavily commented
│       ├── tools.py            # Tool definitions + execution
│       ├── sandbox.py          # Docker sandbox wrapper for safe execution
│       ├── trace.py            # Structured trace recording (every step)
│       ├── display.py          # Rich-based live terminal visualization
│       ├── prompts.py          # System prompt + tool descriptions
│       ├── config.py           # Settings (model, max iterations, etc.)
│       └── cli.py              # Entry point: react-debugger run <repo> <test-cmd>
├── fixtures/                   # Buggy projects for the agent to debug
│   ├── 01_off_by_one/
│   │   ├── buggy_code.py
│   │   ├── test_code.py
│   │   └── BUG.md             # Describes the planted bug (for MY reference, not the agent's)
│   ├── 02_wrong_operator/
│   ├── 03_mutable_default/
│   ├── 04_dict_key_error/
│   ├── 05_type_coercion/
│   ├── 06_missing_null_check/
│   ├── 07_incorrect_recursion/
│   └── 08_async_race/         # Hard one
├── traces/                     # Saved JSON traces of agent runs
├── tests/
│   ├── test_tools.py
│   ├── test_sandbox.py
│   ├── test_agent_loop.py     # Test the loop with a mocked LLM
│   └── test_termination.py
├── pyproject.toml
├── Dockerfile.sandbox          # The sandbox image
├── README.md
└── .env.example                # ANTHROPIC_API_KEY
```

## Detailed component specifications

### 1. The agent loop — `agent.py` (THE CORE)

This is the most important file. It must be the clearest, best-commented code in the project.

```python
class ReActAgent:
    """
    A self-debugging agent built on the ReAct pattern.

    ReAct = Reasoning + Acting. The agent alternates between:
      - REASONING: thinking about what it knows and what to try next
      - ACTING: calling a tool to gather info or make a change
      - OBSERVING: reading the tool's result
    ...and loops until the goal is met.
    """

    def __init__(self, repo_path, test_command, config, trace, display):
        self.messages = []          # the growing conversation history
        self.iteration = 0
        # ... etc

    def run(self) -> RunResult:
        """The main ReAct loop. This is the function to study."""
        # 1. Initialize: set up the goal in the first user message
        # 2. Loop:
        #    a. Call Claude with full history + available tools
        #    b. Extract the thought (text block) and action (tool_use block)
        #    c. Display the thought + action
        #    d. If action == finish: break
        #    e. Execute the tool → observation
        #    f. Append observation to history (as tool_result)
        #    g. Record the full step in the trace
        #    h. Check termination (tests pass / max iters)
        # 3. Return RunResult (success/failure, iterations, final diff)
```

Requirements for the loop:
- Each iteration is one Claude API call with `tools=` parameter (tool use)
- The agent's reasoning (text content blocks) is captured and displayed BEFORE the action
- Tool results are fed back as `tool_result` content blocks in the next user message
- The loop must handle the case where Claude returns text but no tool call (prompt it to take an action or finish)
- The loop must handle malformed tool inputs gracefully (return an error observation, let the agent recover)
- Max iterations is configurable (default 25). When hit, the agent gets one final turn to summarize what it learned and why it's stuck.
- After each `edit_file` or `write_file`, the loop should NOT auto-run tests — the agent must explicitly choose to run tests. This teaches the agent (and me) the observe-decide rhythm.

### 2. Tools — `tools.py` (the ACTION space)

Define these tools using Anthropic's tool-use schema. Each tool needs a name, a clear description (the agent reads these to decide what to use), an input schema, and an execution function.

```python
TOOLS = [
    {
        "name": "run_tests",
        "description": "Run the test suite and see which tests pass or fail. Returns the full pytest output including tracebacks. Use this to check the current state and to verify whether your fix worked.",
        "input_schema": {
            "type": "object",
            "properties": {
                "test_path": {"type": "string", "description": "Optional: specific test file or test to run. Omit to run all tests."}
            }
        }
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file. Optionally specify a line range. Use this to inspect code you suspect contains the bug.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "list_files",
        "description": "List files in a directory (recursive). Use this to understand the project structure.",
        "input_schema": {
            "type": "object",
            "properties": {"directory": {"type": "string"}}
        }
    },
    {
        "name": "search_code",
        "description": "Search the codebase for a pattern (like grep). Returns matching lines with file names and line numbers. Use this to find where a function is defined or called.",
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"]
        }
    },
    {
        "name": "run_command",
        "description": "Run an arbitrary shell command in the sandbox (e.g., to add a print statement via python, check a value, inspect the environment). Returns stdout and stderr.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"]
        }
    },
    {
        "name": "edit_file",
        "description": "Make a targeted edit to a file by replacing an exact string with a new string. Use this to apply a fix once you've identified the bug.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string", "description": "Exact text to replace (must be unique in the file)"},
                "new_string": {"type": "string"}
            },
            "required": ["path", "old_string", "new_string"]
        }
    },
    {
        "name": "finish",
        "description": "Call this when the tests pass and the bug is fixed, OR when you're confident you cannot solve it. Provide a summary of what the bug was and how you fixed it (or why you're stuck).",
        "input_schema": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "summary": {"type": "string"},
                "root_cause": {"type": "string", "description": "What was actually wrong"}
            },
            "required": ["success", "summary"]
        }
    }
]
```

Each tool's execution function runs inside the Docker sandbox and returns a string observation. Keep observations concise — truncate huge outputs (e.g., limit file reads to a reasonable size, truncate test output to the relevant failure section) so the context window doesn't blow up.

### 3. Sandbox — `sandbox.py` (SAFETY — non-negotiable)

The agent runs arbitrary code and shell commands. This MUST be sandboxed. Do not run agent-generated commands directly on the host.

- Use Docker. On agent start, copy the target repo into a fresh container built from `Dockerfile.sandbox` (a Python image with pytest installed).
- All tool executions (`run_tests`, `run_command`, `edit_file`, etc.) operate on the copy inside the container, never the original repo.
- The container has no network access (`--network none`) — the agent shouldn't be able to phone home or pip-install arbitrary things unless explicitly allowed.
- Set resource limits (memory, CPU) and a per-command timeout (e.g., 30s) so a runaway command or infinite loop in the buggy code doesn't hang the agent.
- At the end of a run, extract the final diff (what the agent changed) from the container so I can review it, then tear down the container.

If Docker isn't available on the machine, fall back to a restricted subprocess mode with a loud warning that it's not truly sandboxed — but Docker is the default and strongly recommended path.

### 4. Trace — `trace.py` (observability for learning)

Record every single step of the loop to a structured JSON file in `traces/`. Each step records:

```python
{
    "iteration": 3,
    "timestamp": "...",
    "thought": "The test expects index 5 but the loop stops at index 4. This looks like an off-by-one in the range() call.",
    "action": {
        "tool": "read_file",
        "input": {"path": "buggy_code.py", "start_line": 10, "end_line": 20}
    },
    "observation": "10: def find_max(items):\n11:     ...",
    "tokens_used": {"input": 4521, "output": 234},
    "tests_passing": false
}
```

These traces are gold for learning — after a run I can read the full trace and see exactly how the agent reasoned. Also record run-level metadata: total iterations, total tokens, total cost estimate, time elapsed, final outcome.

### 5. Display — `display.py` (watch the agent think live)

Use Rich to render the loop live in the terminal as it runs. For each iteration show:
- A header: "Iteration 3/25"
- The THOUGHT in one color (e.g., cyan), formatted as the agent's reasoning
- The ACTION in another color (e.g., yellow): the tool name + key params
- The OBSERVATION in a dim/muted color, truncated to a few lines with a "[+N more lines]" indicator
- A status line: tests passing/failing, tokens used so far

The visualization is what makes the ReAct pattern click viscerally — I should be able to watch the reason→act→observe rhythm unfold in real time.

### 6. System prompt — `prompts.py`

The system prompt sets up the agent as a methodical debugger. Key elements:

```
You are a methodical debugging agent. You have been given a code repository with a failing test suite. Your job is to find and fix the bug so all tests pass.

Work in a loop:
1. THINK about what you know and form a hypothesis about the bug
2. Take ONE action to test your hypothesis (read code, run a command, run tests)
3. OBSERVE the result and update your understanding
4. Repeat until tests pass

Principles:
- Always start by running the tests to see the actual failure.
- Read the failing test FIRST to understand what behavior is expected.
- Form a hypothesis BEFORE making changes. Don't guess-and-check randomly.
- Make the smallest change that fixes the root cause, not the symptom.
- After making a fix, ALWAYS run the tests to verify.
- If a fix doesn't work, revert your thinking — don't pile changes on top of changes.
- When tests pass, call finish() with a clear explanation of the root cause.

You have a limited number of iterations. Be efficient. One good hypothesis beats ten random edits.
```

### 7. CLI — `cli.py`

```
react-debugger run <repo_path> --test-command "pytest" [--max-iterations 25] [--model claude-sonnet-4-20250514]
react-debugger run fixtures/01_off_by_one --test-command "pytest test_code.py"

# After a run, replay a saved trace:
react-debugger replay traces/<trace_file>.json
```

### 8. Fixtures — buggy projects to debug

Create 8 small Python projects, each with a single planted bug and a test suite that fails because of it. Each fixture has the buggy code, the tests, and a `BUG.md` that describes the planted bug (for MY reference — the agent never sees BUG.md). Make them progressively harder:

1. **off_by_one** — a `range()` or slice that's off by one (easy)
2. **wrong_operator** — `>` where it should be `>=`, or `and` where it should be `or` (easy)
3. **mutable_default** — the classic `def f(x, acc=[])` mutable default argument bug (medium)
4. **dict_key_error** — accessing a dict key that doesn't always exist, missing `.get()` (easy-medium)
5. **type_coercion** — comparing a string to an int, or `"5" + 5` style bug (medium)
6. **missing_null_check** — a function that crashes on `None` input it should handle (medium)
7. **incorrect_recursion** — a recursive function with a wrong base case (medium-hard)
8. **async_race** — a race condition in async code where order isn't guaranteed (hard — the agent may fail this, which is a GOOD learning outcome about agent limitations)

Each fixture should be realistic — not contrived one-liners, but small functions with plausible bugs a real developer would write.

### 9. Tests — `tests/`

- `test_tools.py`: each tool executes correctly and returns expected observation format
- `test_sandbox.py`: the sandbox isolates execution, enforces timeouts, has no network
- `test_agent_loop.py`: with a MOCKED LLM (canned responses), the loop correctly cycles through reason→act→observe, appends to history, and terminates on finish
- `test_termination.py`: loop terminates on success, on max iterations, and on agent-declared-stuck

## Build order

### Phase 1: The loop skeleton (do this first — it's the learning core)
1. Project structure + config + pyproject.toml
2. `tools.py` — tool definitions (schemas only, execution stubs returning fake data)
3. `agent.py` — the ReAct loop, calling the real Anthropic API with the tools, but executing against stub tools. Get the loop CYCLING and DISPLAYING before worrying about real execution.
4. `display.py` — live terminal output so you can watch the loop
5. Test the loop with one trivial fixture and stub tools — confirm reason→act→observe→repeat works and terminates

### Phase 2: Real execution
6. `sandbox.py` — Docker sandbox + `Dockerfile.sandbox`
7. Wire the real tool execution into the sandbox
8. `fixtures/01_off_by_one` — first real buggy project
9. Run the agent against fixture 01 end-to-end. Iterate until it can solve the easy bug.

### Phase 3: Robustness + observability
10. `trace.py` — structured trace recording + replay
11. Handle edge cases: malformed tool calls, no-action responses, context-length management, output truncation
12. Add the remaining fixtures (02–08)
13. Run against all fixtures, record traces, see which it solves and which it doesn't

### Phase 4: Polish
14. Tests (mocked-LLM loop test is the important one)
15. README with: what ReAct is, architecture diagram, a sample trace walkthrough, how to run, what I learned
16. A "trace gallery" in the README — show 2-3 annotated traces (an easy solve, a hard solve, a failure) so readers see the pattern

## What to build first

Build Phase 1 completely before anything else. I want to see the ReAct loop cycle — reason, act, observe, repeat — with stub tools and live terminal display, before any Docker or real execution exists. The loop is the thing I'm here to learn. Once I can watch it think against fake observations, we make the observations real.

When you build `agent.py`, comment it like you're teaching the ReAct pattern to someone who's never seen it. That file is the whole point of this project.

Before starting, check the current Anthropic Python SDK tool-use API — verify the exact parameter names and the structure of tool_use / tool_result content blocks, since the SDK evolves. Search the Anthropic docs if unsure.

Go. Start with Phase 1 — the loop skeleton with stub tools and live display.
