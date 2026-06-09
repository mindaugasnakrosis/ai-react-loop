"""
AGENT — the ReAct loop.

This file is the whole point of the project. Read it carefully.

ReAct = Reasoning + Acting. Introduced in the 2022 paper "ReAct: Synergizing
Reasoning and Acting in Language Models" (Yao et al.). The core idea:
give the LLM tools it can call, and let it alternate between:
  - REASONING: forming hypotheses about the world in text
  - ACTING: calling a tool to get new information or make a change
  - OBSERVING: reading the tool's result and updating its understanding

Each THOUGHT + ACTION + OBSERVATION is one iteration. The loop continues
until the agent calls the special `finish` tool, hits max iterations,
or an unrecoverable error occurs.

Why is this better than just asking "what's wrong with this code?" in one shot?
Because the agent can GATHER INFORMATION across turns. It reads the test failure,
then reads the relevant code, then forms a hypothesis, then applies a fix,
then verifies — each step informed by the last. The context window becomes
a working memory that accumulates evidence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
import anthropic

from .config import settings
from .display import (
    show_action,
    show_error,
    show_finish,
    show_iteration_header,
    show_max_iterations_warning,
    show_observation,
    show_run_summary,
    show_sandbox_event,
    show_status,
    show_thought,
)
from .prompts import SYSTEM_PROMPT, make_initial_user_message
from .tools import TOOL_DECLARATIONS, ToolResult, execute_stub


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """One tool invocation requested by the model in a single turn."""
    name: str
    input: dict
    id: str


@dataclass
class RunResult:
    success: bool
    iterations: int
    total_tokens: int
    elapsed_seconds: float
    summary: str = ""
    root_cause: str = ""


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

class ReActAgent:
    """
    A self-debugging agent built on the ReAct pattern.

    Instantiate with a repo path and test command, then call run().
    The loop runs synchronously — it blocks until finished.

    Key state:
      self.messages   — the CONVERSATION HISTORY. This is the agent's memory.
                        Every thought, action, and observation gets appended here
                        so the next Claude call has full context. It grows each
                        iteration. This is how the agent "remembers" what it tried.

      self.iteration  — which loop cycle we're on (1-indexed for display)
      self.tokens     — cumulative token count across all API calls
    """

    def __init__(
        self,
        repo_path: str,
        test_command: str,
        use_stubs: bool = False,
    ) -> None:
        self.repo_path = repo_path
        self.test_command = test_command
        self.use_stubs = use_stubs  # Phase 1: True. Phase 2: False (real sandbox)

        # The conversation history — starts empty, grows each iteration.
        # Format: list of {"role": "user"|"assistant", "content": [...]}
        # This exact structure is what gets sent to the Anthropic API each turn.
        self.messages: list[dict] = []

        self.iteration = 0
        self.tokens = 0
        self.start_time: float = 0.0

        # Phase 2: Docker sandbox instance, alive for the duration of run()
        self._sandbox = None

        # The Anthropic client — authenticated from ANTHROPIC_API_KEY env var
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    # -----------------------------------------------------------------------
    # THE MAIN REACT LOOP
    # -----------------------------------------------------------------------

    def run(self) -> RunResult:
        """
        Entry point. Manages the sandbox lifecycle, then delegates to _run_loop().

        Phase 1 (use_stubs=True): no Docker, tools return hardcoded observations.
        Phase 2 (use_stubs=False): starts a Docker container, runs the loop, stops
        the container on exit — even if the loop throws. The finally block guarantees
        cleanup so no orphan containers are left behind.
        """
        self.start_time = time.monotonic()

        if not self.use_stubs:
            # Import here to avoid a hard dependency on Docker when running stubs
            from .sandbox import DockerSandbox
            self._sandbox = DockerSandbox(self.repo_path, self.test_command)
            show_sandbox_event("Building sandbox image (cached after first run)...")
            self._sandbox.start()
            show_sandbox_event("Sandbox ready.")

        try:
            return self._run_loop()
        finally:
            if self._sandbox:
                show_sandbox_event("Stopping sandbox...")
                self._sandbox.stop()
                self._sandbox = None

    def _run_loop(self) -> RunResult:
        """
        The ReAct loop itself. This is the function to study.

        Structure:
          1. Seed the conversation with the task (one user message)
          2. Loop:
             a. Call Claude → get THOUGHT + list of ACTIONS (ToolCalls)
             b. Display thought and each action
             c. If any action == finish → return
             d. Execute every tool → collect OBSERVATIONS
             e. Append assistant turn + all observations to history
             f. Check termination (max iterations)
          3. Return RunResult

        Note on parallel tool calls: Claude may return multiple tool_use blocks
        in a single response. The API requires a tool_result for EVERY tool_use
        block in the preceding assistant turn — missing even one is a 400 error.
        That's why we execute all tool calls and return all results together in
        one user message before the next Claude call.
        """

        # ── STEP 1: Seed the conversation ────────────────────────────────
        # The first user message describes the task. After this, all subsequent
        # user messages are tool_result blocks (observations). Claude's messages
        # are the assistant turns containing thoughts + tool calls.
        initial_message = make_initial_user_message(self.repo_path, self.test_command)
        self.messages.append({"role": "user", "content": initial_message})

        # ── STEP 2: The loop ──────────────────────────────────────────────
        while self.iteration < settings.max_iterations:
            self.iteration += 1
            show_iteration_header(self.iteration, settings.max_iterations)

            # ── 2a. Call Claude ───────────────────────────────────────────
            # We send the FULL conversation history every time. This is not
            # inefficient — the Anthropic API is stateless, so we must resend
            # the whole context. The model sees everything it's done so far.
            #
            # The `tools=` parameter tells Claude what actions are available.
            # Claude reads the tool descriptions to decide which one to use.
            # Without tools=, Claude can only respond with text — it cannot act.
            thought, tool_calls, assistant_content = self._call_claude()

            # ── 2b. Display the thought ───────────────────────────────────
            # The thought is the TEXT portion of Claude's response — its
            # reasoning before it chose an action. This is the "Reason" in ReAct.
            show_thought(thought)

            # ── 2c. Handle the `finish` action ───────────────────────────
            # `finish` is a special tool — it doesn't execute anything, it just
            # signals that the agent is done. When the agent calls finish(),
            # we exit the loop immediately.
            finish_call = next((tc for tc in tool_calls if tc.name == "finish"), None)
            if finish_call:
                success = finish_call.input.get("success", False)
                summary = finish_call.input.get("summary", "")
                root_cause = finish_call.input.get("root_cause", "")
                show_action(finish_call.name, finish_call.input)
                show_finish(success, summary, root_cause)
                elapsed = time.monotonic() - self.start_time
                show_run_summary(self.iteration, self.tokens, elapsed, success)
                return RunResult(
                    success=success,
                    iterations=self.iteration,
                    total_tokens=self.tokens,
                    elapsed_seconds=elapsed,
                    summary=summary,
                    root_cause=root_cause,
                )

            # ── Handle no tool call ───────────────────────────────────────
            # Sometimes Claude responds with text but doesn't call a tool.
            # This shouldn't happen if the system prompt is good, but we handle
            # it gracefully: show the text and nudge Claude to take an action.
            if not tool_calls:
                show_thought("(No tool call — nudging agent to take an action)")
                self._append_assistant_message(assistant_content)
                self._append_nudge()
                continue

            # ── 2d. Execute all tool calls → collect observations ─────────
            # This is the "Act" in ReAct. Each tool runs in the sandbox (Phase 2)
            # or against stubs (Phase 1) and returns an OBSERVATION string.
            #
            # IMPORTANT: we must return a tool_result for EVERY tool_use block
            # the model returned — the API rejects mismatches as a 400 error.
            # So even if the model returned two calls in one turn, we execute
            # both and return both results in the same user message.
            self._append_assistant_message(assistant_content)
            tool_results: list[tuple[str, ToolResult]] = []
            for tc in tool_calls:
                show_action(tc.name, tc.input)
                result = self._execute_tool(tc.name, tc.input)
                show_observation(result.output)
                tool_results.append((tc.id, result))

            show_status(self.iteration, self.tokens, tests_passing=None)

            # ── 2e. Append all observations to conversation history ────────
            # A single user message carries ALL tool_result blocks for this turn.
            # After n iterations, messages looks like:
            #   [user: task,
            #    assistant: thought1 + tool_call(s)1,
            #    user: tool_result(s)1,
            #    assistant: thought2 + tool_call(s)2,
            #    user: tool_result(s)2, ...]
            #
            # Every API call sends this entire list. The model has full memory
            # of everything it's thought, done, and observed.
            self._append_tool_results(tool_results)

        # ── Max iterations reached ────────────────────────────────────────
        # We've exhausted the budget. Give the agent one final call to
        # summarise what it learned, then return failure.
        show_max_iterations_warning(settings.max_iterations)
        summary = self._call_for_stuck_summary()
        elapsed = time.monotonic() - self.start_time
        show_finish(False, summary, None)
        show_run_summary(self.iteration, self.tokens, elapsed, False)
        return RunResult(
            success=False,
            iterations=self.iteration,
            total_tokens=self.tokens,
            elapsed_seconds=elapsed,
            summary=summary,
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _call_claude(self) -> tuple[str, list[ToolCall], list]:
        """
        Make one API call to Claude and parse the response.

        Returns:
          thought           — the text content (the agent's reasoning)
          tool_calls        — list of ToolCall objects (often one, sometimes more)
          assistant_content — the raw SDK content blocks (stored as-is in history)

        How Claude's response is structured:
          response.content is a list of content blocks. Two types matter here:
            - TextBlock:    type="text"    → the agent's THOUGHT / reasoning
            - ToolUseBlock: type="tool_use" → the ACTION: which tool + what args

          Text blocks come first (the reasoning), then one or more tool_use blocks.
          Claude may return multiple tool_use blocks in one response (parallel calls).
          We must return a tool_result for EVERY one — hence the list return.
        """
        response = self.client.messages.create(
            model=settings.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOL_DECLARATIONS,
            messages=self.messages,
        )

        # Accumulate token usage across the run
        self.tokens += response.usage.input_tokens + response.usage.output_tokens

        thought = ""
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                # THOUGHT — the agent's reasoning in plain text
                thought += block.text
            elif block.type == "tool_use":
                # ACTION — collect all tool calls; the API requires a result for each
                tool_calls.append(ToolCall(name=block.name, input=block.input, id=block.id))

        # Store the raw SDK objects — the SDK serialises them correctly on replay.
        # Do NOT convert to plain dicts here: the SDK's internal representation
        # preserves the id/name/input structure the API needs for tool_result matching.
        return thought, tool_calls, list(response.content)

    def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> ToolResult:
        """
        Dispatch to the right executor.

        Phase 1 (use_stubs=True):  execute_stub()        — hardcoded fake output
        Phase 2 (use_stubs=False): self._sandbox.execute_tool() — real Docker

        Errors are caught and returned as error observations. This means a bad
        tool call (wrong path, malformed regex, etc.) doesn't crash the loop —
        the agent sees the error text and can try a different approach.
        """
        try:
            if self.use_stubs:
                return execute_stub(tool_name, tool_input)
            return self._sandbox.execute_tool(tool_name, tool_input)
        except Exception as exc:
            show_error(f"Tool execution failed: {exc}")
            return ToolResult(
                output=f"ERROR executing {tool_name}: {exc}",
                is_error=True,
            )

    def _append_assistant_message(self, content_blocks: list) -> None:
        """
        Append Claude's response to the conversation history.

        Pass the raw SDK content block objects back as-is — the Anthropic SDK
        serialises its own types correctly when building the next request.
        Converting them to plain dicts with a helper breaks the correlation
        the API uses to match tool_use blocks to their tool_result blocks.
        """
        self.messages.append({
            "role": "assistant",
            "content": content_blocks,
        })

    def _append_tool_results(self, results: list[tuple[str, ToolResult]]) -> None:
        """
        Append all tool observations from this turn to the conversation history.

        One USER message carries ALL tool_result blocks for the turn — the API
        requires every tool_use block in the preceding assistant turn to have a
        matching tool_result block in the very next user message.

        After this append, the next _call_claude() will have all observations
        in context so the model can reason about what it found.
        """
        content = []
        for tool_use_id, result in results:
            # Truncate very long observations to protect the context window.
            observation = result.output
            if len(observation) > settings.max_observation_chars:
                half = settings.max_observation_chars // 2
                observation = (
                    observation[:half]
                    + f"\n\n[... truncated {len(result.output) - settings.max_observation_chars} chars ...]\n\n"
                    + observation[-half:]
                )
            entry: dict = {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": observation,
            }
            if result.is_error:
                entry["is_error"] = True
            content.append(entry)

        self.messages.append({"role": "user", "content": content})

    def _append_nudge(self) -> None:
        """
        When Claude responds with text but no tool call, nudge it.

        This handles an edge case: sometimes the model produces a concluding
        paragraph without calling finish(). We inject a user message reminding
        it to take an explicit action so the loop can continue normally.
        """
        self.messages.append({
            "role": "user",
            "content": (
                "Please take an action using one of the available tools. "
                "If you believe the task is complete, call finish()."
            ),
        })

    def _call_for_stuck_summary(self) -> str:
        """
        One final API call when max iterations is reached.

        We ask the agent to summarise what it tried and why it got stuck.
        This is useful for learning — it surfaces the limits of the approach.
        """
        self.messages.append({
            "role": "user",
            "content": (
                f"You have reached the maximum of {settings.max_iterations} iterations "
                "without solving the problem. Please call finish(success=False) with "
                "a summary of what you tried, what you learned, and why you couldn't fix it."
            ),
        })
        _, tool_calls, assistant_content = self._call_claude()
        finish_call = next((tc for tc in tool_calls if tc.name == "finish"), None)
        if finish_call:
            return finish_call.input.get("summary", "Max iterations reached.")
        for block in assistant_content:
            if hasattr(block, "text"):
                return block.text
        return "Max iterations reached with no summary."

