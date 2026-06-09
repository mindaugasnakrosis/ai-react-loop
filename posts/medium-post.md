# I Built a Self-Debugging AI Agent from Scratch — Here's Exactly How the Reasoning Cycle Works

*No LangChain. No agent framework. Just the raw Anthropic API and a Python loop.*

---

I recently built a self-debugging agent that takes a repository with failing tests and autonomously finds and fixes the bug. Not using LangChain, AutoGPT, or any agent framework — just the raw Anthropic API and about 200 lines of Python.

The reason: I wanted to actually understand how AI agents work, not just call a library and hope for the best. If you've ever looked at an "AI agent" and thought *but what is it actually doing?* — this article is for you.

---

## What Is a ReAct Agent?

ReAct (Reasoning + Acting) is a pattern introduced in a 2022 paper by Yao et al. The core idea is elegantly simple: give a language model tools it can call, and let it alternate between three steps until it's done:

1. **THOUGHT** — the model reasons about what it knows and what it needs to find out
2. **ACTION** — it calls a tool to gather information or make a change
3. **OBSERVATION** — it reads the result and updates its understanding

Then it loops. Each THOUGHT is informed by every OBSERVATION that came before it. The context window becomes a working memory that accumulates evidence across iterations.

This sounds simple. The implementation is where it gets interesting.

---

## The Core Problem: The API Has No Memory

Here's the thing most people miss: **Claude has no memory between API calls.** Every single call is completely stateless. Send a message, get a response — the model has no idea what happened in the previous call.

So how does an "agent" that runs for 10 iterations remember what it tried in iteration 3?

The answer is almost anticlimactic: **you keep a list and send the entire thing on every call.**

```python
self.messages: list[dict] = []  # this IS the agent's memory
```

After three iterations, this list looks like:

```
[
  { role: "user",      content: "Debug fixtures/01_off_by_one, tests are failing..." },
  { role: "assistant", content: [TextBlock("I'll run the tests first"), ToolUseBlock("run_tests")] },
  { role: "user",      content: [tool_result("1 failed: assert find_max([1,2,3,4,5]) == 5, got 4")] },
  { role: "assistant", content: [TextBlock("The test shows find_max returns 4 instead of 5..."), ToolUseBlock("read_file")] },
  { role: "user",      content: [tool_result("4: for i in range(len(items) - 1):  # BUG")] },
  { role: "assistant", content: [TextBlock("Found it. range(len(items) - 1) skips the last element"), ToolUseBlock("edit_file")] },
  { role: "user",      content: [tool_result("Successfully replaced 1 occurrence in buggy_code.py")] },
]
```

Every iteration appends two things to this list: the assistant's thought + tool call, and the tool's result. The next API call sends the whole list. The model reads the entire transcript from scratch and responds as if it "remembers" — but it doesn't. It just re-reads the conversation you handed it.

**The context window is the agent's working memory. Nothing more.**

---

## How Tool Calls Actually Work at the API Level

When you pass `tools=TOOL_DECLARATIONS` to the Anthropic API, you're sending Claude a list of JSON schemas describing what actions are available. Claude reads the descriptions to decide which tool to use. Vague descriptions lead to bad tool choices — this is actually prompt engineering.

```python
{
    "name": "read_file",
    "description": "Read the contents of a file. Optionally specify a line range. Use this to inspect code you suspect contains the bug.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"},
        },
        "required": ["path"],
    },
}
```

The API response has a `content` field that is a **list of blocks**, not a single string. Two types matter:

- **TextBlock** — the model's reasoning in plain English (the THOUGHT)
- **ToolUseBlock** — which tool to call and what arguments (the ACTION)

```python
response.content = [
    TextBlock(text="The test fails on the last element. Let me look at the loop bounds..."),
    ToolUseBlock(id="toolu_abc123", name="read_file", input={"path": "buggy_code.py"}),
]
```

The TextBlock comes first — reasoning before acting. That's the "Reason" in ReAct. Then comes the action.

---

## The Contract You Must Not Break

Here's the part that took me a debugging session to understand properly.

When Claude returns a `ToolUseBlock`, the API establishes a **strict contract**: before your next API call, you must send back a `tool_result` block whose `tool_use_id` matches the `id` in the `ToolUseBlock`. If you don't — 400 error.

The conversation structure the API enforces:

```
Turn 1: user      → task description
Turn 2: assistant → [thought text] + [tool_use  id="abc"  name="run_tests"]
Turn 3: user      → [tool_result   tool_use_id="abc"  content="1 failed..."]
Turn 4: assistant → [thought text] + [tool_use  id="def"  name="read_file"]
Turn 5: user      → [tool_result   tool_use_id="def"  content="line 4: range(len..."]
```

This broke for me when Claude returned **two** `ToolUseBlock`s in a single response — `list_files` and `read_file` in the same turn. My code only sent back one `tool_result`. The API rejected it immediately.

The fix: collect every `ToolUseBlock` from the response, execute every one, return all results in a single user message.

```python
# WRONG: only handles the last tool call
for block in response.content:
    if block.type == "tool_use":
        tool_call = ToolCall(name=block.name, input=block.input, id=block.id)

# RIGHT: collect all of them
tool_calls: list[ToolCall] = []
for block in response.content:
    if block.type == "tool_use":
        tool_calls.append(ToolCall(name=block.name, input=block.input, id=block.id))
```

Claude may request parallel tool calls when it can gather multiple pieces of information at once. Every single one needs a matching result in the next message, or the entire request fails.

---

## The Loop Structure in Full

Here's the complete loop, simplified. This is the entire agentic pattern:

```python
# 1. Seed the conversation with the task
messages.append({"role": "user", "content": "Debug this repo, tests are failing..."})

while iteration < max_iterations:

    # 2. Ask Claude: given everything you've seen, what next?
    response = client.messages.create(
        model="claude-sonnet-4-6",
        tools=TOOL_DECLARATIONS,
        messages=messages,       # the FULL history every time
    )

    # 3. Parse the response into thought + tool calls
    thought = ""
    tool_calls = []
    for block in response.content:
        if block.type == "text":
            thought += block.text          # THOUGHT: the reasoning
        elif block.type == "tool_use":
            tool_calls.append(block)       # ACTION: what to do

    # 4. Stop if the agent declares success or failure
    if any(tc.name == "finish" for tc in tool_calls):
        break

    # 5. Execute every tool call
    results = [(tc.id, execute(tc)) for tc in tool_calls]

    # 6. Append everything to history
    messages.append({"role": "assistant", "content": response.content})
    messages.append({"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": id, "content": result.output}
        for id, result in results
    ]})

    # 7. Loop — Claude now sees the new observations and reasons again
```

That's it. Seven steps. Every agent framework you'll ever use — LangChain, the Claude agent SDK, AutoGen — is some variation of this loop with more scaffolding around it.

---

## What the Agent Actually Did

I gave the agent a Python file with an off-by-one bug: `range(len(items) - 1)` instead of `range(len(items))`. The tests were failing because the loop skipped the last element.

The agent never saw the bug description. It had to find it.

In six iterations it:
1. Ran tests → saw `assert find_max([1,2,3,4,5]) == 5` failing with value `4`
2. Listed files → identified `buggy_code.py` as the source
3. Read the file → saw `range(len(items) - 1)` with no surrounding context
4. Formed the hypothesis: "the loop iterates indices 0 through n-2, skipping the last element"
5. Called `edit_file` with the fix
6. Called `finish` with an accurate root cause summary

Nobody told it that `range(len(items) - 1)` skips the last element. It connected: test fails on the last element + loop stops one short of the last element = bug.

That connection happened inside a single THOUGHT block — plain English reasoning, visible in the terminal, formed by reading the test failure and the source code in sequence. The context window was doing what a developer's working memory does.

---

## Why Not Just Ask "What's Wrong With This Code?" in One Shot?

You could. For a small, self-contained file, a one-shot prompt would likely find the same bug.

But the agent pattern scales to things a single prompt can't handle:

- **The bug is in file B, but you only know it exists because test A failed** — you need to navigate there
- **The fix requires reading context** — imports, constants, how functions call each other
- **Verification matters** — apply the fix, then run the tests again to confirm
- **Errors teach** — if `edit_file` fails because the string wasn't found exactly, the agent reads the error and tries differently

A single prompt gets one shot. The agent gets as many shots as you give it iterations. Each shot is informed by every observation that came before it.

---

## The Execution Sandbox

One last piece: when the agent calls `run_tests` or `edit_file`, those calls need to execute somewhere safely. We used Docker.

The agent's edits need to persist across tool calls within a run — it can't apply a fix and then have `run_tests` see the original file. So the setup is:

1. Copy the repo to a temporary directory at the start of a run
2. Start a Docker container with that directory mounted at `/repo`
3. File reads and writes go directly to the tmpdir on the host
4. Command execution (`pytest`, arbitrary shell commands) goes through `docker exec`
5. Both see the same files because the tmpdir IS `/repo` via the volume mount

The container runs with `--network none` so the agent can't make outbound HTTP requests, and `--memory 512m` to prevent runaway resource usage. At the end of the run — success or failure — the container stops and the tmpdir is deleted.

The clean separation is the point: the ReAct loop itself doesn't change between Phase 1 (fake stubs) and Phase 2 (real Docker). Only what `execute_tool()` calls at the bottom changes. The reasoning cycle is completely independent of the execution environment.

---

## What I Learned

The ReAct pattern is not magic. It's a while loop with a list. The "intelligence" is entirely the language model's — the loop just gives it a structure to gather information before committing to an answer.

The hardest part isn't the loop. It's the details:
- Every `tool_use` needs a matching `tool_result` in the very next message
- Parallel tool calls are real and your code must handle all of them
- Tool descriptions are prompt engineering — write them carefully
- The context window fills up — you need to truncate long observations before they blow the token budget

Build it from scratch once and you'll understand every agent framework you ever use.

The full code is on GitHub: [link]
